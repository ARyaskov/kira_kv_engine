//! End-to-end engine benchmark (no external harness): build time, bytes per key
//! and lookup latency for every engine, printed as a Markdown table.
//!
//! ```bash
//! cargo bench --bench engines                 # 10M keys
//! KIRA_BENCH_N=1000000 cargo bench --bench engines
//! ```
//!
//! Lookup numbers are per-key averages over 2M random hits (and misses where the
//! engine can reject them), single thread, index warm in cache where it fits.

use kira_kv_engine::{
    DynamicConfig, DynamicIndex, HybridBuilder, IndexBuilder, IndexConfig, PgmBuilder, PgmIndexU128,
};
use std::hint::black_box;
use std::time::Instant;

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn ns_per<T>(iters: usize, f: impl FnOnce() -> T) -> (f64, T) {
    let t = Instant::now();
    let r = black_box(f());
    (t.elapsed().as_nanos() as f64 / iters as f64, r)
}

fn row(
    engine: &str,
    build_s: f64,
    bytes_per_key: f64,
    lookup: f64,
    batch: Option<f64>,
    miss: Option<(f64, f64)>,
) {
    let fmt = |v: Option<f64>| v.map(|x| format!("{x:.1}")).unwrap_or_else(|| "—".into());
    let (miss_ns, false_positive_percent) = miss.unzip();
    println!(
        "| {engine} | {build_s:.2} | {bytes_per_key:.2} | {lookup:.1} | {} | {} | {} |",
        fmt(batch),
        fmt(miss_ns),
        false_positive_percent.map_or_else(|| "—".into(), |percent| format!("{percent:.4}%"))
    );
}

fn main() {
    let n: usize =
        std::env::var("KIRA_BENCH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(10_000_000);
    assert!(n > 0, "KIRA_BENCH_N must be positive");
    let q = 2_000_000usize.min(n);
    let miss_end = n.checked_add(q).expect("query range overflow");
    let keys_u64: Vec<u64> = (0..n as u64).map(splitmix).collect();
    let keys_bytes: Vec<[u8; 8]> = keys_u64.iter().map(|k| k.to_le_bytes()).collect();
    let hits: Vec<u64> =
        (0..q as u64).map(|i| keys_u64[(splitmix(i ^ 0x77) % n as u64) as usize]).collect();
    // splitmix is bijective, so these queries are disjoint from the build input.
    let misses: Vec<u64> = (n as u64..miss_end as u64).map(splitmix).collect();
    let mut sink = 0usize;

    println!("n = {n}, queries = {q}");
    println!("| Engine | Build s | B/key | Lookup ns | Batch ns | Miss ns | False positives |");
    println!("|---|---:|---:|---:|---:|---:|---:|");

    for lean in [false, true] {
        let mut cfg = IndexConfig::default();
        cfg.lean_mph = lean;
        let t = Instant::now();
        let idx = IndexBuilder::new().with_config(cfg).build_index_ref(&keys_bytes).unwrap();
        let build = t.elapsed().as_secs_f64();
        let bpk = idx.stats().total_memory as f64 / n as f64;
        // Independent of query timing: every input must map bijectively into
        // the minimal output range. Never turn a missing known key into slot0.
        let mut seen = vec![false; n];
        for key in &keys_bytes {
            let slot = idx.lookup(key).expect("known static key was rejected");
            assert!(slot < n && !seen[slot], "static output is not a bijection");
            seen[slot] = true;
        }
        let (lookup, s) = ns_per(q, || {
            hits.iter()
                .fold(0usize, |a, &k| a.wrapping_add(idx.lookup_u64(k).expect("known static hit")))
        });
        sink = sink.wrapping_add(s);
        let mut out = vec![None; q];
        let mut canon = vec![0u64; q];
        let (batch, _) = ns_per(q, || {
            idx.lookup_batch_u64_simd_into(black_box(&hits), &mut canon, &mut out);
            black_box(&out);
        });
        for (&key, result) in hits.iter().zip(&out) {
            assert_eq!(*result, idx.lookup_u64(key).ok());
        }
        let miss = if lean {
            None
        } else {
            let (time, accepted) =
                ns_per(q, || misses.iter().filter(|&&k| idx.lookup_u64(k).is_ok()).count());
            Some((time, 100.0 * accepted as f64 / q as f64))
        };
        row(
            if lean { "Index (lean)" } else { "Index (default)" },
            build,
            bpk,
            lookup,
            Some(batch),
            miss,
        );
    }

    for lean in [false, true] {
        let t = Instant::now();
        let idx = HybridBuilder::new()
            .with_pgm_epsilon(2048)
            .with_lean(lean)
            .build_from_u64(&keys_u64)
            .unwrap();
        let build = t.elapsed().as_secs_f64();
        let bpk = idx.memory_usage() as f64 / n as f64;
        for (position, &key) in keys_u64.iter().enumerate() {
            assert_eq!(idx.lookup_u64(key), Some(position as u32));
        }
        let (lookup, s) = ns_per(q, || {
            hits.iter().fold(0usize, |a, &k| {
                a.wrapping_add(idx.lookup_u64(k).expect("known Hybrid hit") as usize)
            })
        });
        sink = sink.wrapping_add(s);
        let (batch, results) = ns_per(q, || idx.lookup_batch_u64_simd(&hits));
        for (&key, result) in hits.iter().zip(&results) {
            assert_eq!(*result, idx.lookup_u64(key));
        }
        let miss = if lean {
            None
        } else {
            let (time, accepted) =
                ns_per(q, || misses.iter().filter(|&&k| idx.lookup_u64(k).is_some()).count());
            Some((time, 100.0 * accepted as f64 / q as f64))
        };
        row(
            if lean { "HybridIndex (lean)" } else { "HybridIndex" },
            build,
            bpk,
            lookup,
            Some(batch),
            miss,
        );
    }

    {
        let mut sorted = keys_u64.clone();
        sorted.sort_unstable();
        sorted.dedup();
        let t = Instant::now();
        let pgm = PgmBuilder::new().with_epsilon(64).build(sorted.clone()).unwrap();
        let build = t.elapsed().as_secs_f64();
        let bpk = pgm.stats().memory_usage as f64 / sorted.len() as f64;
        for (rank, &key) in sorted.iter().enumerate() {
            assert_eq!(pgm.index(key).ok(), Some(rank));
        }
        let (lookup, s) = ns_per(q, || {
            hits.iter().fold(0usize, |a, &k| a.wrapping_add(pgm.index(k).expect("known PGM hit")))
        });
        sink = sink.wrapping_add(s);
        let (miss, accepted) =
            ns_per(q, || misses.iter().filter(|&&k| pgm.index(k).is_ok()).count());
        assert_eq!(accepted, 0);
        row("PgmIndex ε=64", build, bpk, lookup, None, Some((miss, 0.0)));
    }

    {
        let m = n.min(2_000_000);
        let mut keys: Vec<u128> = (0..m as u64)
            .map(|i| ((splitmix(i) as u128) << 64) | splitmix(i ^ 0x55) as u128)
            .collect();
        keys.sort_unstable();
        keys.dedup();
        let t = Instant::now();
        let idx = PgmIndexU128::build(keys.clone(), 64).unwrap();
        let build = t.elapsed().as_secs_f64();
        let bpk = idx.memory_usage() as f64 / keys.len() as f64;
        for (rank, &key) in keys.iter().enumerate() {
            assert_eq!(idx.index(key).ok(), Some(rank));
        }
        let qs: Vec<u128> =
            (0..q as u64).map(|i| keys[(splitmix(i) % keys.len() as u64) as usize]).collect();
        let (lookup, s) = ns_per(q, || {
            qs.iter().fold(0usize, |a, &k| a.wrapping_add(idx.index(k).expect("known u128 hit")))
        });
        sink = sink.wrapping_add(s);
        row(&format!("PgmIndexU128 ε=64 ({m} keys)"), build, bpk, lookup, None, None);
    }

    {
        let m = n.min(1_000_000);
        let mut idx = DynamicIndex::with_config(DynamicConfig {
            flush_threshold: 64 * 1024,
            ..DynamicConfig::default()
        });
        let t = Instant::now();
        for k in &keys_bytes[..m] {
            idx.insert(k.to_vec());
        }
        let insert = t.elapsed().as_nanos() as f64 / m as f64;
        idx.compact().unwrap();
        let build = t.elapsed().as_secs_f64();
        let bpk = idx.memory_usage() as f64 / m as f64;
        for (position, key) in keys_bytes[..m].iter().enumerate() {
            assert_eq!(idx.lookup(key), Some(position as u32));
        }
        let hb: Vec<[u8; 8]> = (0..q.min(m))
            .map(|i| keys_u64[(splitmix(i as u64 ^ 0x77) % m as u64) as usize].to_le_bytes())
            .collect();
        let (lookup, s) = ns_per(hb.len(), || {
            hb.iter().fold(0usize, |a, k| {
                a.wrapping_add(idx.lookup(k).expect("known dynamic hit") as usize)
            })
        });
        sink = sink.wrapping_add(s);
        row(
            &format!("DynamicIndex ({m} keys, insert {insert:.0} ns)"),
            build,
            bpk,
            lookup,
            None,
            None,
        );
    }
    eprintln!("(sink {sink})");
}
