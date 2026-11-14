//! Single-pass learned-index comparison, driven by compare_pgm_ef.py.
use engine::{EliasFano, PgmBuilder, PgmIndex};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
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

// SAFETY: requests are forwarded unchanged to System; counters do not allocate
// or modify storage, pointer ownership, alignment or initialization.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            allocated(layout.size());
        }
        result
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            allocated(layout.size());
        }
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        // SAFETY: pointer and layout identify the original System allocation.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the allocation and replacement size satisfy GlobalAlloc.
        let result = unsafe { System.realloc(pointer, layout, size) };
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

fn pgm(label: &str, dataset: &str, keys: &[u64], epsilon: u32, compressed: bool) {
    let n = keys.len();
    let q = n.min(100_000);
    let queries: Vec<_> = (0..q)
        .map(|i| {
            let rank = (mix(i as u64) % n as u64) as usize;
            let lower_query = keys[rank].saturating_add(1);
            (rank, lower_query, keys.partition_point(|&key| key < lower_query))
        })
        .collect();
    let base = LIVE.load(Relaxed);
    let owned = keys.to_vec();
    PEAK.store(LIVE.load(Relaxed), Relaxed);
    let started = Instant::now();
    let index = PgmBuilder::new()
        .with_epsilon(epsilon)
        .with_parallel(false)
        .with_elias_fano(compressed)
        .build(owned)
        .unwrap();
    let build_ms = started.elapsed().as_secs_f64() * 1000.;
    let heap = LIVE.load(Relaxed).saturating_sub(base);
    let peak = PEAK.load(Relaxed).saturating_sub(base);
    let stats = index.stats();
    let allocations = ALLOCATIONS.load(Relaxed);
    let started = Instant::now();
    for &(rank, _, _) in &queries {
        assert_eq!(index.index(black_box(keys[rank])).unwrap(), rank);
    }
    let lookup_ns = started.elapsed().as_nanos() as f64 / q as f64;
    let started = Instant::now();
    for &(_, query, expected) in &queries {
        assert_eq!(index.lower_bound(black_box(query)), expected);
    }
    let lower_ns = started.elapsed().as_nanos() as f64 / q as f64;
    let query_allocations = ALLOCATIONS.load(Relaxed) - allocations;
    // Full key oracle is independent of predictions and includes round trips.
    let bytes = index.to_bytes().unwrap();
    let restored = PgmIndex::from_bytes(&bytes).unwrap();
    for (rank, &key) in keys.iter().enumerate() {
        assert_eq!(index.index(key).unwrap(), rank);
        assert_eq!(restored.index(key).unwrap(), rank);
    }
    println!(
        "{label},pgm,{dataset},{n},{q},{build_ms:.3},{lookup_ns:.3},{lower_ns:.3},0,{heap},{peak},{},{query_allocations},{},{},{},pass",
        stats.memory_usage,
        stats.total_segments,
        stats.max_error,
        bytes.len()
    );
}

fn ef(label: &str, dataset: &str, keys: &[u64], adversarial: bool) {
    let n = keys.len();
    let q = if adversarial { 16_384 } else { 100_000.min(n) };
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let started = Instant::now();
    let index = EliasFano::from_sorted(keys).unwrap();
    let build_ms = started.elapsed().as_secs_f64() * 1000.;
    let heap = LIVE.load(Relaxed).saturating_sub(base);
    let peak = PEAK.load(Relaxed).saturating_sub(base);
    let mut materialized = Vec::with_capacity(64);
    let allocations = ALLOCATIONS.load(Relaxed);
    let started = Instant::now();
    for i in 0..q {
        let rank = if adversarial { 1 } else { (mix(i as u64) % n as u64) as usize };
        assert_eq!(black_box(&index).get(black_box(rank)), keys[rank]);
    }
    let lookup_ns = started.elapsed().as_nanos() as f64 / q as f64;
    let started = Instant::now();
    let count = keys.len().min(64);
    for _ in 0..1024 {
        index.materialize_range(black_box(0), count, &mut materialized);
        assert_eq!(&materialized, &keys[..count]);
    }
    let materialize_ns = started.elapsed().as_nanos() as f64 / (1024 * count) as f64;
    let query_allocations = ALLOCATIONS.load(Relaxed) - allocations;
    let mut bytes = Vec::new();
    index.write_to(&mut bytes);
    let mut cursor = 0;
    let restored = EliasFano::read_from(&bytes, &mut cursor).unwrap();
    assert_eq!(cursor, bytes.len());
    for (rank, &key) in keys.iter().enumerate() {
        assert_eq!(index.get(rank), key);
        assert_eq!(restored.get(rank), key);
    }
    println!(
        "{label},ef,{dataset},{n},{q},{build_ms:.3},{lookup_ns:.3},0,{materialize_ns:.3},{heap},{peak},{},{query_allocations},0,0,{},pass",
        index.memory_usage(),
        bytes.len()
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let label = &args[1];
    let n: usize = args[2].parse().unwrap();
    assert!(n >= 256 && n < u32::MAX as usize);
    if let Ok(core) = std::env::var("KIRA_BENCH_CORE") {
        assert!(core_affinity::set_for_current(core_affinity::CoreId {
            id: core.parse().unwrap()
        }));
    }
    let dense: Vec<u64> = (0..n as u64).collect();
    pgm(label, "dense", &dense, 0, false);
    pgm(label, "dense_ef", &dense, 0, true);
    let offset = (1u64 << 52) + (1u64 << 27);
    let shifted: Vec<u64> = dense.iter().map(|&key| offset + key).collect();
    pgm(label, "offset_dense", &shifted, 0, false);
    let maximum: Vec<u64> = dense.iter().map(|&key| u64::MAX - (n as u64 - 1) + key).collect();
    pgm(label, "max_dense", &maximum, 0, false);
    let mut random: Vec<u64> = (0..n as u64).map(mix).collect();
    random.sort_unstable();
    pgm(label, "random", &random, 64, false);
    pgm(label, "random_ef", &random, 64, true);
    let regular: Vec<u64> = (0..n as u64).map(|i| i * 13 + 7).collect();
    ef(label, "regular", &regular, false);
    let power = n.next_power_of_two() as u64;
    let upper = power * power;
    let clustered: Vec<u64> = std::iter::once(0).chain(upper - n as u64 + 2..=upper).collect();
    assert_eq!(clustered.len(), n);
    ef(label, "long_gap", &clustered, true);
}
