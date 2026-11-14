//! Used by compare_compaction.py against frozen and current source trees.
use engine::{DynamicConfig, DynamicIndex, IndexBuilder};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

struct CountAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn allocated(size: usize) {
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: every allocation is forwarded unchanged to System. Counters only
// observe sizes and never allocate or alter the returned storage.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc's caller provides a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller provides a valid allocation layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        // SAFETY: pointer and layout are the original System allocation.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the original allocation and new size satisfy GlobalAlloc.
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
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let label = &args[1];
    let policy = &args[2];
    let n: usize = args[3].parse().unwrap();
    assert!(n > 0 && n < u32::MAX as usize);
    if let Ok(core) = std::env::var("KIRA_BENCH_CORE") {
        assert!(core_affinity::set_for_current(core_affinity::CoreId {
            id: core.parse().unwrap()
        }));
    }
    // Warm persistent worker initialization outside the measured interval.
    drop(IndexBuilder::new().with_bloom_filter(false).build_index_ref(&[[0u8; 8]]).unwrap());
    let keys: Vec<[u8; 8]> = (0..n as u64).map(|i| mix(i).to_le_bytes()).collect();
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let mut index = DynamicIndex::with_config(DynamicConfig {
        flush_threshold: 4096,
        max_tiers: 8,
        lean_tiers: false,
        parallel_build: true,
    });
    match policy.as_str() {
        "full" => {}
        "size" => index.set_compaction_ratio(4).unwrap(),
        "geometric" => {
            #[cfg(feature = "geometric")]
            index.set_compaction_policy(engine::CompactionPolicy::Geometric).unwrap();
            #[cfg(not(feature = "geometric"))]
            panic!("geometric policy is unavailable in the baseline");
        }
        _ => panic!("unknown compaction policy"),
    }
    let started = Instant::now();
    for (position, key) in keys.iter().enumerate() {
        assert_eq!(index.insert(key.to_vec()), position as u32);
    }
    index.flush().unwrap();
    index.finish_background_compaction().unwrap();
    let append_ns = started.elapsed().as_nanos() as f64 / n as f64;
    let append_peak = PEAK.load(Relaxed).saturating_sub(base);
    let append_heap = LIVE.load(Relaxed).saturating_sub(base);
    let append_estimate = index.memory_usage();
    let tiers = index.tier_count();
    let rebuilt = index.compaction_stats().1;
    let queries = n.min(100_000);
    let started = Instant::now();
    for query in 0..queries {
        let position = (mix(query as u64 ^ 0x77) % n as u64) as usize;
        assert_eq!(index.lookup(black_box(&keys[position])), Some(position as u32));
    }
    let lookup_ns = started.elapsed().as_nanos() as f64 / queries as f64;
    PEAK.store(LIVE.load(Relaxed), Relaxed);
    for (position, key) in keys.iter().enumerate() {
        if position % 5 != 0 {
            assert_eq!(index.delete(key), Some(position as u32));
        }
    }
    let before_gc_heap = LIVE.load(Relaxed).saturating_sub(base);
    let started = Instant::now();
    index.flush().unwrap();
    index.finish_background_compaction().unwrap();
    let gc_ms = started.elapsed().as_secs_f64() * 1000.;
    let after_gc_heap = LIVE.load(Relaxed).saturating_sub(base);
    let gc_peak = PEAK.load(Relaxed).saturating_sub(base);
    let dead = index.tombstone_count();
    let after_gc_estimate = index.memory_usage();
    for (position, key) in keys.iter().enumerate() {
        assert_eq!(index.lookup(key), (position % 5 == 0).then_some(position as u32));
    }
    let restored = DynamicIndex::from_bytes(&index.to_bytes()).unwrap();
    assert_eq!(restored.len(), n.div_ceil(5));
    for (position, key) in keys.iter().enumerate() {
        assert_eq!(restored.lookup(key), (position % 5 == 0).then_some(position as u32));
    }
    println!(
        "{label},{policy},{n},{append_ns:.3},{lookup_ns:.3},{append_heap},{append_peak},{append_estimate},{tiers},{rebuilt},{before_gc_heap},{gc_ms:.3},{after_gc_heap},{gc_peak},{after_gc_estimate},{dead},pass"
    );
}
