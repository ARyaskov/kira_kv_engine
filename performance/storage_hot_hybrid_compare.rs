//! Single-pass storage, tracker and Hybrid comparison with independent oracles.
use engine::{
    DynamicHotTier, FingerprintBits, HybridBuilder, HybridIndex, Index, IndexBuilder, SpaceSaving,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

struct CountAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn allocated(size: usize) {
    ALLOCATIONS.fetch_add(1, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: unchanged requests are forwarded to System; counters never allocate
// and do not change ownership, initialization, alignment or pointer provenance.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            allocated(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            allocated(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(p, layout) };
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(p, layout, size) };
        if !result.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        result
    }
}

#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn inverse_mix(mut value: u64) -> u64 {
    value ^= value >> 31 ^ value >> 62;
    value = value.wrapping_mul(0x319642b2d24d8ec3);
    value ^= value >> 27 ^ value >> 54;
    value = value.wrapping_mul(0x96de1b173f119089);
    value ^= value >> 30 ^ value >> 60;
    value.wrapping_sub(0x9e3779b97f4a7c15)
}

fn mmap_save(label: &str, n: usize, lean: bool) {
    let keys: Vec<_> = (0..n as u64).map(u64::to_le_bytes).collect();
    let index = IndexBuilder::new()
        .with_parallel_build(false)
        .with_lean_mph(lean)
        .with_fingerprint_bits(FingerprintBits::Bits32)
        .build_index_ref(&keys)
        .unwrap();
    let dataset = if lean { "lean" } else { "fp32_bloom" };
    let path = format!("target/storage-hot-hybrid-comparison/{label}-{dataset}.mmap");
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let allocations = ALLOCATIONS.load(Relaxed);
    let started = Instant::now();
    index.save_mmap(&path).unwrap();
    let ns = started.elapsed().as_nanos() as f64;
    let peak = PEAK.load(Relaxed).saturating_sub(base);
    let retained = LIVE.load(Relaxed).saturating_sub(base);
    let allocations = ALLOCATIONS.load(Relaxed) - allocations;
    let wire = std::fs::metadata(&path).unwrap().len();
    let restored = Index::open_mmap(&path).unwrap();
    let mut seen = vec![false; n];
    for key in &keys {
        let slot = index.lookup(key).unwrap();
        assert!(slot < n && !seen[slot]);
        seen[slot] = true;
        assert_eq!(restored.lookup(key).unwrap(), slot);
    }
    println!(
        "{label},mmap,{dataset},{n},1,0,{ns:.3},0,{retained},{peak},{allocations},0,{wire},pass"
    );
}

fn tracker(label: &str, clustered: bool) {
    let capacity: usize = if clustered { 2048 } else { 4096 };
    let n = if clustered { 256 } else { 500_000 };
    let prefix: Vec<u64> = if clustered {
        (0..capacity as u64).map(|i| inverse_mix(i * (capacity * 4) as u64)).collect()
    } else {
        Vec::new()
    };
    let stream: Vec<u64> = if clustered {
        (capacity as u64..capacity as u64 + n as u64)
            .map(|i| inverse_mix(i * (capacity * 4) as u64))
            .collect()
    } else {
        (0..n as u64).map(|i| if i % 4 != 0 { i % 32 } else { mix(i) % 8192 }).collect()
    };
    if clustered {
        for (i, &key) in prefix.iter().chain(stream.iter()).enumerate() {
            assert_eq!(mix(key), i as u64 * (capacity * 4) as u64);
        }
    }
    let base = LIVE.load(Relaxed);
    let mut ss = SpaceSaving::new(capacity);
    for &key in &prefix {
        ss.observe(key);
    }
    let retained = LIVE.load(Relaxed).saturating_sub(base);
    let steady = LIVE.load(Relaxed);
    PEAK.store(steady, Relaxed);
    let allocations = ALLOCATIONS.load(Relaxed);
    let started = Instant::now();
    for &key in &stream {
        ss.observe(black_box(key));
    }
    let ns = started.elapsed().as_nanos() as f64 / n as f64;
    let peak = PEAK.load(Relaxed).saturating_sub(steady);
    let allocations = ALLOCATIONS.load(Relaxed) - allocations;
    let top = ss.top_k(capacity);
    assert_eq!(ss.total_observed(), (prefix.len() + stream.len()) as u64);
    assert_eq!(top.iter().map(|p| p.1).sum::<u64>(), ss.total_observed());
    if clustered {
        for &(key, count) in &top {
            assert_eq!(mix(key) & (capacity * 4 - 1) as u64, 0);
            assert!((1..=2).contains(&count));
        }
    } else {
        let mut exact = vec![0u64; 8192];
        for &key in &stream {
            exact[key as usize] += 1;
        }
        for &(key, count) in &top {
            assert!(count >= exact[key as usize]);
            assert!(count - exact[key as usize] <= n as u64 / capacity as u64);
        }
        let selected = ss.top_k(16);
        let mut expected: Vec<_> =
            exact.iter().copied().enumerate().map(|(k, c)| (k as u64, c)).collect();
        expected.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let truth: std::collections::HashSet<_> = expected[..16].iter().map(|p| p.0).collect();
        assert!(
            selected.iter().all(|p| truth.contains(&p.0)),
            "top16 recall must be 1 on this separated workload"
        );
    }
    let dataset = if clustered { "colliding_eviction" } else { "mixed_observe" };
    println!(
        "{label},tracker,{dataset},{capacity},{n},0,{ns:.3},0,{retained},{peak},{allocations},0,0,pass"
    );
}

fn extraction(label: &str, dynamic: bool) {
    let capacity = 65_536usize;
    let base = LIVE.load(Relaxed);
    let mut ss = if dynamic { None } else { Some(SpaceSaving::new(capacity)) };
    let hot = if dynamic { Some(DynamicHotTier::new(None, capacity, 1)) } else { None };
    for key in 0..capacity as u64 {
        let count = if key < 64 { 10 + key % 11 } else { 1 };
        for _ in 0..count {
            if let Some(t) = &hot {
                t.observe(key);
            } else {
                ss.as_mut().unwrap().observe(key);
            }
        }
    }
    let retained = LIVE.load(Relaxed).saturating_sub(base);
    let steady = LIVE.load(Relaxed);
    PEAK.store(steady, Relaxed);
    let allocations = ALLOCATIONS.load(Relaxed);
    let started = Instant::now();
    let out = if let Some(t) = &hot { t.take_top_k(16) } else { ss.as_ref().unwrap().top_k(16) };
    let ns = started.elapsed().as_nanos() as f64;
    let peak = PEAK.load(Relaxed).saturating_sub(steady);
    let allocations = ALLOCATIONS.load(Relaxed) - allocations;
    let mut expected: Vec<_> = (0..64u64).map(|key| (key, 10 + key % 11)).collect();
    expected.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    assert_eq!(out, expected[..16]);
    if let Some(t) = &hot {
        assert!(!t.should_rebuild());
        for _ in 0..37 {
            t.observe(999);
        }
        assert!(t.should_rebuild());
        assert_eq!(t.take_top_k(16), vec![(999, 37)]);
        assert_eq!(t.dropped_observations(), 0);
    }
    let dataset = if dynamic { "dynamic_take16" } else { "top16" };
    println!(
        "{label},tracker,{dataset},{capacity},1,0,0,{ns:.3},{retained},{peak},{allocations},0,0,pass"
    );
}

fn hybrid(label: &str, n: usize, epsilon: u32, dataset: &str) {
    let keys: Vec<u64> = (0..n as u64).collect();
    let q = 100_000usize;
    let query: Vec<u64> = (0..q).map(|i| mix(i as u64) % n as u64).collect();
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let started = Instant::now();
    let index = HybridBuilder::new()
        .with_parallel(false)
        .with_seed(17)
        .with_pgm_epsilon(epsilon)
        .build_from_u64(&keys)
        .unwrap();
    let build = started.elapsed().as_secs_f64() * 1000.;
    let peak = PEAK.load(Relaxed).saturating_sub(base);
    let retained = LIVE.load(Relaxed).saturating_sub(base);
    let allocations = ALLOCATIONS.load(Relaxed);
    let started = Instant::now();
    for &key in &query {
        assert_eq!(index.lookup_u64(black_box(key)), Some(key as u32));
    }
    let ns = started.elapsed().as_nanos() as f64 / q as f64;
    let allocations = ALLOCATIONS.load(Relaxed) - allocations;
    let bytes = index.to_bytes();
    let restored = HybridIndex::from_bytes(&bytes).unwrap();
    for &key in &keys {
        assert_eq!(index.lookup_u64(key), Some(key as u32));
        assert_eq!(restored.lookup_u64(key), Some(key as u32));
    }
    assert_eq!(restored.to_bytes(), bytes);
    for chunk in query.chunks(997) {
        assert_eq!(
            index.lookup_batch_u64_simd(chunk),
            chunk.iter().map(|&k| Some(k as u32)).collect::<Vec<_>>()
        );
    }
    println!(
        "{label},hybrid,{dataset},{n},{q},{build:.3},{ns:.3},0,{retained},{peak},{allocations},{},{},pass",
        index.num_segments(),
        bytes.len()
    );
}

fn fixture() {
    let keys: Vec<_> = (0..257u64).map(u64::to_le_bytes).collect();
    let index = IndexBuilder::new()
        .with_parallel_build(false)
        .with_fingerprint_bits(FingerprintBits::Bits24)
        .build_index_ref(&keys)
        .unwrap();
    index.save_mmap("tests/fixtures/mmap_stream_fp24_v081.bin").unwrap();
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args[1] == "fixtures" {
        fixture();
        return;
    }
    let label = &args[1];
    let n: usize = args[2].parse().unwrap();
    assert!(n >= 262_144 && n < u32::MAX as usize);
    if let Ok(core) = std::env::var("KIRA_BENCH_CORE") {
        assert!(core_affinity::set_for_current(core_affinity::CoreId {
            id: core.parse().unwrap()
        }));
    }
    assert!(Path::new("target/storage-hot-hybrid-comparison").is_dir());
    mmap_save(label, n, false);
    mmap_save(label, n, true);
    tracker(label, false);
    tracker(label, true);
    extraction(label, false);
    extraction(label, true);
    for (epsilon, name) in [(0, "epsilon0"), (64, "epsilon64"), (2048, "default")] {
        hybrid(label, n.min(262_144), epsilon, name);
    }
    hybrid(label, 32, 2048, "tiny");
}
