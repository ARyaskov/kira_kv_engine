//! Reproducible 10M-key correctness/performance gate for the staged optimizations.
//! cargo run --profile bench --example staged_bench -- baseline
use kira_kv_engine::{DynamicConfig, DynamicIndex, Index, IndexBuilder};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

struct CountAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn allocated(n: usize) {
    let v = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(v, Relaxed);
}
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            allocated(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            allocated(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe {
            System.dealloc(p, l);
        }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            LIVE.fetch_sub(l.size(), Relaxed);
            allocated(n);
        }
        q
    }
}
#[global_allocator]
static ALLOC: CountAlloc = CountAlloc;
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn metric(stage: &str, name: &str, value: f64, unit: &str) {
    println!("{stage},{name},{value:.6},{unit}");
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
fn timed(runs: usize, n: usize, mut f: impl FnMut()) -> f64 {
    f();
    let values = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64 / n as f64
        })
        .collect();
    median(values)
}
fn main() {
    #[cfg(feature = "parallel")]
    {
        let core = std::env::var("KIRA_BENCH_CORE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        assert!(core_affinity::set_for_current(core_affinity::CoreId { id: core }));
    }
    let stage = std::env::args().nth(1).unwrap_or("baseline".into());
    let n =
        std::env::var("KIRA_STAGE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000_000usize);
    let runs = std::env::var("KIRA_STAGE_RUNS").ok().and_then(|s| s.parse().ok()).unwrap_or(5usize);
    let q = n.min(1_000_000);
    assert!(n > 0 && n < u32::MAX as usize && runs > 0);
    println!("stage,metric,value,unit");
    metric(&stage, "keys", n as f64, "keys");
    let keys: Vec<[u8; 8]> = (0..n as u64).map(|i| mix(i).to_le_bytes()).collect();
    let ordinals: Vec<usize> = (0..q as u64).map(|i| (mix(i ^ 0x77) % n as u64) as usize).collect();
    let hits: Vec<u64> = ordinals.iter().map(|&i| u64::from_le_bytes(keys[i])).collect();
    let misses: Vec<u64> = (n as u64..n as u64 + q as u64).map(mix).collect();
    let refs: Vec<&[u8]> = ordinals.iter().map(|&i| keys[i].as_slice()).collect();
    for lean in [false, true] {
        let tag = if lean { "lean" } else { "default" };
        let t = Instant::now();
        let idx = IndexBuilder::new().with_lean_mph(lean).build_index_ref(&keys).unwrap();
        metric(&stage, &format!("{tag}.build"), t.elapsed().as_secs_f64(), "s");
        metric(
            &stage,
            &format!("{tag}.memory"),
            idx.stats().total_memory as f64 / n as f64,
            "B/key",
        );
        // Verify every built key and the MPH bijection, not only a sampled sum.
        let mut seen = vec![false; n];
        for k in &keys {
            let slot = idx.lookup(k).unwrap();
            assert!(slot < n && !seen[slot]);
            seen[slot] = true;
        }
        drop(seen);
        metric(
            &stage,
            &format!("{tag}.hit"),
            timed(runs, q, || {
                black_box(
                    hits.iter().fold(0usize, |s, &k| {
                        s.wrapping_add(idx.lookup_u64(black_box(k)).unwrap())
                    }),
                );
            }),
            "ns/key",
        );
        if !lean {
            metric(
                &stage,
                "default.miss",
                timed(runs, q, || {
                    black_box(
                        misses.iter().filter(|&&k| idx.lookup_u64(black_box(k)).is_ok()).count(),
                    );
                }),
                "ns/key",
            );
        }
        let mut out = vec![None; q];
        let mut canon = vec![0; q];
        metric(
            &stage,
            &format!("{tag}.batch_u64"),
            timed(runs, q, || {
                idx.lookup_batch_u64_simd_into(black_box(&hits), &mut canon, &mut out);
                black_box(&out);
            }),
            "ns/key",
        );
        for (i, &k) in hits.iter().enumerate() {
            assert_eq!(out[i], idx.lookup_u64(k).ok());
        }
        metric(
            &stage,
            &format!("{tag}.batch_bytes"),
            timed(runs, q, || {
                black_box(idx.lookup_batch_pipelined(black_box(&refs)));
            }),
            "ns/key",
        );
        if !lean {
            std::fs::create_dir_all("performance").unwrap();
            let path = "performance/staged.idx";
            idx.save_mmap(path).unwrap();
            let mut times = Vec::new();
            let mut heaps = Vec::new();
            for _ in 0..runs {
                let before = LIVE.load(Relaxed);
                PEAK.store(before, Relaxed);
                let t = Instant::now();
                let loaded = Index::open_mmap(path).unwrap();
                times.push(t.elapsed().as_secs_f64() * 1000.);
                heaps.push(PEAK.load(Relaxed).saturating_sub(before) as f64);
                for &k in hits.iter().take(10000) {
                    assert_eq!(loaded.lookup_u64(k).ok(), idx.lookup_u64(k).ok());
                }
                black_box(&loaded);
            }
            metric(&stage, "mmap.load", median(times), "ms");
            metric(&stage, "mmap.peak_heap", median(heaps), "bytes");
            std::fs::remove_file(path).unwrap();
        }
    }
    for bits in [
        kira_kv_engine::FingerprintBits::Bits16,
        kira_kv_engine::FingerprintBits::Bits24,
        kira_kv_engine::FingerprintBits::Bits32,
    ] {
        let tag = format!("fp{}_no_bloom", bits.bits());
        let t = Instant::now();
        let idx = IndexBuilder::new()
            .with_bloom_filter(false)
            .with_fingerprint_bits(bits)
            .build_index_ref(&keys)
            .unwrap();
        metric(&stage, &format!("{tag}.build"), t.elapsed().as_secs_f64(), "s");
        metric(
            &stage,
            &format!("{tag}.memory"),
            idx.stats().total_memory as f64 / n as f64,
            "B/key",
        );
        for key in &keys {
            assert!(idx.lookup(key).is_ok());
        }
        metric(
            &stage,
            &format!("{tag}.hit"),
            timed(runs, q, || {
                black_box(
                    hits.iter().fold(0usize, |s, &k| {
                        s.wrapping_add(idx.lookup_u64(black_box(k)).unwrap())
                    }),
                );
            }),
            "ns/key",
        );
        let mut out = vec![None; q];
        let mut canon = vec![0; q];
        metric(
            &stage,
            &format!("{tag}.batch"),
            timed(runs, q, || {
                idx.lookup_batch_u64_simd_into(black_box(&hits), &mut canon, &mut out);
                black_box(&out);
            }),
            "ns/key",
        );
        let false_positives = misses.iter().filter(|&&k| idx.lookup_u64(k).is_ok()).count();
        metric(&stage, &format!("{tag}.false_positives"), false_positives as f64, "per_1M");
        metric(
            &stage,
            &format!("{tag}.miss"),
            timed(runs, q, || {
                black_box(misses.iter().filter(|&&k| idx.lookup_u64(black_box(k)).is_ok()).count());
            }),
            "ns/key",
        );
        assert!(
            false_positives
                <= if bits == kira_kv_engine::FingerprintBits::Bits16 { 100 } else { 10 }
        );
    }
    let mut dynamic = DynamicIndex::with_config(DynamicConfig {
        flush_threshold: 65536,
        ..DynamicConfig::default()
    });
    // BENCH_NEW_BACKGROUND_BEGIN
    if stage.contains("background") {
        dynamic.set_background_compaction(true).unwrap();
    }
    // BENCH_NEW_BACKGROUND_END
    let t = Instant::now();
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(dynamic.insert(k.to_vec()), i as u32);
    }
    dynamic.flush().unwrap();
    dynamic.finish_background_compaction().unwrap();
    metric(&stage, "dynamic.insert", t.elapsed().as_nanos() as f64 / n as f64, "ns/key");
    let (merges, rebuilt) = dynamic.compaction_stats();
    metric(&stage, "dynamic.auto_merges", merges as f64, "merges");
    metric(&stage, "dynamic.rebuilt_keys", rebuilt as f64, "keys");
    metric(&stage, "dynamic.tiers", dynamic.tier_count() as f64, "tiers");
    metric(&stage, "dynamic.memory_tiered", dynamic.memory_usage() as f64 / n as f64, "B/key");
    metric(
        &stage,
        "dynamic.hit_tiered",
        timed(runs, q, || {
            black_box(
                refs.iter().fold(0u64, |s, k| s + dynamic.lookup(black_box(k)).unwrap() as u64),
            );
        }),
        "ns/key",
    );
    let t = Instant::now();
    dynamic.compact().unwrap();
    metric(&stage, "dynamic.compact", t.elapsed().as_secs_f64(), "s");
    metric(&stage, "dynamic.memory", dynamic.memory_usage() as f64 / n as f64, "B/key");
    metric(
        &stage,
        "dynamic.hit",
        timed(runs, q, || {
            black_box(
                refs.iter().fold(0u64, |s, k| s + dynamic.lookup(black_box(k)).unwrap() as u64),
            );
        }),
        "ns/key",
    );
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(dynamic.lookup(k), Some(i as u32));
    }
    for &k in &misses {
        assert_eq!(dynamic.lookup(&k.to_le_bytes()), None);
    }
    // Exercise deletion, revival, shadowing and round trips on the same 10M-key index.
    for i in (0..n).step_by(1000) {
        assert_eq!(dynamic.delete(&keys[i]), Some(i as u32));
    }
    dynamic.flush().unwrap();
    for i in (0..n).step_by(1000) {
        assert_eq!(dynamic.lookup(&keys[i]), None);
        let id = dynamic.insert(keys[i].to_vec());
        assert!(id >= n as u32);
    }
    dynamic.flush().unwrap();
    assert_eq!(dynamic.len(), n);
    dynamic.finish_background_compaction().unwrap();
    let bytes = dynamic.to_bytes();
    let restored = DynamicIndex::from_bytes(&bytes).unwrap();
    assert_eq!(restored.len(), n);
    for key in &keys {
        assert_eq!(restored.lookup(key), dynamic.lookup(key));
    }
    metric(&stage, "correctness", 1., "pass");
}
