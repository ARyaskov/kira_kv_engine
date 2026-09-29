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
    DynamicConfig, DynamicIndex, HybridBuilder, IndexBuilder, IndexConfig, PgmBuilder,
    PgmIndexU128,
};
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
    let r = f();
    (t.elapsed().as_nanos() as f64 / iters as f64, r)
}

fn row(engine: &str, build_s: f64, bytes_per_key: f64, lookup: f64, batch: Option<f64>, miss: Option<f64>) {
    let fmt = |v: Option<f64>| v.map(|x| format!("{x:.1}")).unwrap_or_else(|| "—".into());
    println!(
        "| {engine} | {build_s:.2} | {bytes_per_key:.2} | {lookup:.1} | {} | {} |",
        fmt(batch),
        fmt(miss)
    );
}

fn main() {
    let n: usize = std::env::var("KIRA_BENCH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(10_000_000);
    let q = 2_000_000usize.min(n);
    let keys_u64: Vec<u64> = (0..n as u64).map(splitmix).collect();
    let keys_bytes: Vec<[u8; 8]> = keys_u64.iter().map(|k| k.to_le_bytes()).collect();
    let hits: Vec<u64> = (0..q as u64).map(|i| keys_u64[(splitmix(i ^ 0x77) % n as u64) as usize]).collect();
    let misses: Vec<u64> = (0..q as u64).map(|i| splitmix(i ^ 0xDEAD_BEEF) | 1 << 63).collect();
    let mut sink = 0usize;

    println!("n = {n}, queries = {q}");
    println!("| Engine | Build s | B/key | Lookup ns | Batch ns | Miss ns |");
    println!("|---|---:|---:|---:|---:|---:|");

    for lean in [false, true] {
        let mut cfg = IndexConfig::default();
        cfg.lean_mph = lean;
        let t = Instant::now();
        let idx = IndexBuilder::new().with_config(cfg).build_index_ref(&keys_bytes).unwrap();
        let build = t.elapsed().as_secs_f64();
        let bpk = idx.stats().total_memory as f64 / n as f64;
        let (lookup, s) = ns_per(q, || hits.iter().fold(0usize, |a, &k| a.wrapping_add(idx.lookup_u64(k).unwrap_or(0))));
        sink = sink.wrapping_add(s);
        let mut out = vec![None; q];
        let mut canon = vec![0u64; q];
        let (batch, _) = ns_per(q, || idx.lookup_batch_u64_simd_into(&hits, &mut canon, &mut out));
        let miss = if lean {
            None
        } else {
            Some(ns_per(q, || misses.iter().filter(|&&k| idx.lookup_u64(k).is_ok()).count()).0)
        };
        row(if lean { "Index (lean)" } else { "Index (default)" }, build, bpk, lookup, Some(batch), miss);
    }

    for lean in [false, true] {
        let t = Instant::now();
        let idx = HybridBuilder::new().with_pgm_epsilon(2048).with_lean(lean).build_from_u64(&keys_u64).unwrap();
        let build = t.elapsed().as_secs_f64();
        let bpk = idx.memory_usage() as f64 / n as f64;
        let (lookup, s) = ns_per(q, || hits.iter().fold(0usize, |a, &k| a.wrapping_add(idx.lookup_u64(k).unwrap_or(0) as usize)));
        sink = sink.wrapping_add(s);
        let (batch, _) = ns_per(q, || idx.lookup_batch_u64_simd(&hits));
        let miss = if lean { None } else { Some(ns_per(q, || misses.iter().filter(|&&k| idx.lookup_u64(k).is_some()).count()).0) };
        row(if lean { "HybridIndex (lean)" } else { "HybridIndex" }, build, bpk, lookup, Some(batch), miss);
    }

    {
        let mut sorted = keys_u64.clone();
        sorted.sort_unstable();
        sorted.dedup();
        let t = Instant::now();
        let pgm = PgmBuilder::new().with_epsilon(64).build(sorted.clone()).unwrap();
        let build = t.elapsed().as_secs_f64();
        let bpk = pgm.stats().memory_usage as f64 / sorted.len() as f64;
        let (lookup, s) = ns_per(q, || hits.iter().fold(0usize, |a, &k| a.wrapping_add(pgm.index(k).unwrap_or(0))));
        sink = sink.wrapping_add(s);
        let (miss, _) = ns_per(q, || misses.iter().filter(|&&k| pgm.index(k).is_ok()).count());
        row("PgmIndex ε=64", build, bpk, lookup, None, Some(miss));
    }

    {
        let m = n.min(2_000_000);
        let mut keys: Vec<u128> = (0..m as u64).map(|i| ((splitmix(i) as u128) << 64) | splitmix(i ^ 0x55) as u128).collect();
        keys.sort_unstable();
        keys.dedup();
        let t = Instant::now();
        let idx = PgmIndexU128::build(keys.clone(), 64).unwrap();
        let build = t.elapsed().as_secs_f64();
        let bpk = idx.memory_usage() as f64 / keys.len() as f64;
        let qs: Vec<u128> = (0..q as u64).map(|i| keys[(splitmix(i) % keys.len() as u64) as usize]).collect();
        let (lookup, s) = ns_per(q, || qs.iter().fold(0usize, |a, &k| a.wrapping_add(idx.index(k).unwrap_or(0))));
        sink = sink.wrapping_add(s);
        row(&format!("PgmIndexU128 ε=64 ({m} keys)"), build, bpk, lookup, None, None);
    }

    {
        let m = n.min(1_000_000);
        let mut idx = DynamicIndex::with_config(DynamicConfig { flush_threshold: 64 * 1024, ..DynamicConfig::default() });
        let t = Instant::now();
        for k in &keys_bytes[..m] {
            idx.insert(k.to_vec());
        }
        let insert = t.elapsed().as_nanos() as f64 / m as f64;
        idx.compact().unwrap();
        let build = t.elapsed().as_secs_f64();
        let bpk = idx.memory_usage() as f64 / m as f64;
        let hb: Vec<[u8; 8]> = hits.iter().take(m).map(|k| k.to_le_bytes()).collect();
        let (lookup, s) = ns_per(hb.len(), || hb.iter().fold(0usize, |a, k| a.wrapping_add(idx.lookup(k).unwrap_or(0) as usize)));
        sink = sink.wrapping_add(s);
        row(&format!("DynamicIndex ({m} keys, insert {insert:.0} ns)"), build, bpk, lookup, None, None);
    }
    eprintln!("(sink {sink})");
}
