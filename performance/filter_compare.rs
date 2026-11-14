//! Isolated comparison: engine dependency plus xorf 0.13.0 with binary-fuse only.
use engine::__internal::BlockBloom;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
use xorf::{BinaryFuse8, Filter};

struct CountAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn allocated(size: usize) {
    ALLOCATIONS.fetch_add(1, Relaxed);
    let current = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(current, Relaxed);
}

// SAFETY: requests are forwarded unchanged to System, and the counters neither
// allocate nor alter pointer ownership, initialization, alignment or provenance.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, size) };
        if !replacement.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        replacement
    }
}

#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

struct QueryMetrics {
    hit_ns: f64,
    miss_ns: f64,
    false_positives: usize,
    allocations: usize,
}

fn queries(hits: &[u64], misses: &[u64], contains: impl Fn(u64) -> bool) -> QueryMetrics {
    let allocations = ALLOCATIONS.load(Relaxed);
    let started = Instant::now();
    let present = hits.iter().filter(|&&key| contains(black_box(key))).count();
    let hit_ns = started.elapsed().as_nanos() as f64 / hits.len() as f64;
    assert_eq!(present, hits.len(), "no inserted key may be rejected");
    let started = Instant::now();
    let false_positives = misses.iter().filter(|&&key| contains(black_box(key))).count();
    let miss_ns = started.elapsed().as_nanos() as f64 / misses.len() as f64;
    QueryMetrics {
        hit_ns,
        miss_ns,
        false_positives,
        allocations: ALLOCATIONS.load(Relaxed) - allocations,
    }
}

fn bloom(label: &str, keys: &[u64], hits: &[u64], misses: &[u64]) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let started = Instant::now();
    let filter = BlockBloom::build_from_prehashed(keys);
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;
    let heap = LIVE.load(Relaxed).saturating_sub(base);
    let peak = PEAK.load(Relaxed).saturating_sub(base);
    let result = queries(hits, misses, |key| black_box(&filter).contains_hash(key));
    // Check every inserted value, not only the random timing sample.
    for &key in keys {
        assert!(filter.contains_hash(key));
    }
    let mut bytes = Vec::new();
    filter.write_to(&mut bytes);
    let mut cursor = 0;
    let restored = BlockBloom::read_from(&bytes, &mut cursor).unwrap();
    assert_eq!(cursor, bytes.len());
    for &key in keys {
        assert!(restored.contains_hash(key));
    }
    assert_eq!(result.allocations, 0);
    println!(
        "{label},block_bloom,{},{},{build_ms:.6},{:.3},{:.3},{heap},{peak},{},{},{:.8},{},pass",
        keys.len(),
        hits.len(),
        result.hit_ns,
        result.miss_ns,
        filter.memory_usage(),
        result.false_positives,
        result.false_positives as f64 / misses.len() as f64,
        result.allocations
    );
}

fn binary_fuse(label: &str, keys: &[u64], hits: &[u64], misses: &[u64]) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let started = Instant::now();
    let filter = BinaryFuse8::try_from(keys).expect("distinct synthetic keys must construct");
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;
    let heap = LIVE.load(Relaxed).saturating_sub(base);
    let peak = PEAK.load(Relaxed).saturating_sub(base);
    let result = queries(hits, misses, |key| black_box(&filter).contains(&key));
    for &key in keys {
        assert!(filter.contains(&key));
    }
    assert_eq!(result.allocations, 0);
    let reported = std::mem::size_of_val(&filter) + filter.len();
    println!(
        "{label},binary_fuse8,{},{},{build_ms:.6},{:.3},{:.3},{heap},{peak},{reported},{},{:.8},{},pass",
        keys.len(),
        hits.len(),
        result.hit_ns,
        result.miss_ns,
        result.false_positives,
        result.false_positives as f64 / misses.len() as f64,
        result.allocations
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let label = args.get(1).map_or("current", String::as_str);
    let largest = args.get(2).map_or(500_000, |n| n.parse::<usize>().unwrap());
    // These sizes keep BlockBloom below its native hugepage threshold, so heap
    // counter peaks include the complete resident filter in every row.
    assert!((4097..=500_000).contains(&largest));
    if let Ok(core) = std::env::var("KIRA_BENCH_CORE") {
        assert!(core_affinity::set_for_current(core_affinity::CoreId {
            id: core.parse().unwrap()
        }));
    }
    const QUERIES: usize = 1_000_000;
    println!(
        "label,filter,keys,queries,build_ms,hit_ns,miss_ns,heap_bytes,peak_heap_bytes,reported_bytes,false_positives,false_positive_rate,query_allocations,correctness"
    );
    for n in [1, 257, 4096, largest] {
        let keys: Vec<u64> = (0..n as u64).map(mix).collect();
        let hits: Vec<_> =
            (0..QUERIES).map(|rank| keys[(mix(rank as u64) % n as u64) as usize]).collect();
        // SplitMix64 is a permutation; disjoint input ranges imply disjoint
        // hashed sets. None of the timing misses can be a true positive.
        let misses: Vec<_> = (0..QUERIES as u64).map(|rank| mix((1 << 40) + rank)).collect();
        bloom(label, &keys, &hits, &misses);
        binary_fuse(label, &keys, &hits, &misses);
    }
}
