//! Isolated construction comparison. Enable the driver's `candidate` feature
//! only when its engine dependency includes MphExternalBuilder.
use engine::__internal::{
    BuildConfig, Builder, PtrHash25Error, PtrHash25Mphf, read_ptrhash25, write_ptrhash25,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

struct CountAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn allocated(size: usize) {
    let current = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(current, Relaxed);
}

// SAFETY: requests are forwarded unchanged to System. Atomic counters never
// allocate or change ownership, initialization, alignment or pointer provenance.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            allocated(layout.size());
        }
        result
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            allocated(layout.size());
        }
        result
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        result
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe {
            System.dealloc(pointer, layout);
        }
    }
}

#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

struct Metrics {
    build_ms: f64,
    retained: usize,
    peak: usize,
}

fn measure<T>(build: impl FnOnce() -> T) -> (T, Metrics) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let started = Instant::now();
    let result = build();
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;
    let retained = LIVE.load(Relaxed).saturating_sub(base);
    let peak = PEAK.load(Relaxed).saturating_sub(base);
    (result, Metrics { build_ms, retained, peak })
}

fn mix(mut key: u64) -> u64 {
    key = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
    key = (key ^ (key >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    key = (key ^ (key >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    key ^ (key >> 31)
}

fn validate_raw(index: &PtrHash25Mphf, keys: impl IntoIterator<Item = u64>, count: usize) -> usize {
    assert_eq!(index.slot_capacity(), count);
    let mut bytes = Vec::new();
    write_ptrhash25(index, &mut bytes);
    let mut cursor = 0;
    let restored = read_ptrhash25(&bytes, &mut cursor).unwrap();
    assert_eq!(cursor, bytes.len());
    let mut seen = vec![false; count];
    let mut actual_count = 0;
    for key in keys {
        let slot = index.index_u64(key) as usize;
        assert!(slot < count && !seen[slot], "known-key mapping must be bijective");
        seen[slot] = true;
        actual_count += 1;
        assert_eq!(index.lookup_u64(key), Some(slot as u32));
        assert_eq!(restored.index_u64(key), slot as u32);
    }
    assert_eq!(actual_count, count);
    assert!(seen.into_iter().all(|present| present));
    bytes.len()
}

fn row(label: &str, case: &str, count: usize, metrics: Metrics, reported: usize, wire: usize) {
    println!(
        "{label},{case},{count},{:.6},{},{},{reported},{wire},pass",
        metrics.build_ms, metrics.retained, metrics.peak,
    );
}

fn ordinary(label: &str, count: usize) {
    let keys: Vec<u64> = (0..count as u64).map(mix).collect();
    let config = BuildConfig { with_fingerprints: false, ..BuildConfig::default() };
    let pool = engine::__internal::pool();
    let (result, metrics) =
        measure(|| pool.install(|| Builder::new().with_config(config).build(&keys)));
    let index = result.unwrap();
    let reported = index.memory_usage();
    let wire = validate_raw(&index, keys.iter().copied(), count);
    row(label, "ordinary", count, metrics, reported, wire);
}

fn oversized_duplicate(label: &str) {
    let mut keys: Vec<u64> = (0..16_384u64).map(mix).collect();
    keys.push(keys[0]);
    let config = BuildConfig {
        lambda: 1_000_000.0,
        max_rehash: 0,
        with_fingerprints: false,
        ..BuildConfig::default()
    };
    let pool = engine::__internal::pool();
    let (result, metrics) =
        measure(|| pool.install(|| Builder::new().with_config(config).build(&keys)));
    assert!(matches!(result, Err(PtrHash25Error::DuplicateKey)));
    row(label, "oversized_duplicate", keys.len(), metrics, 0, 0);
}

fn large_buckets(label: &str) {
    let keys: Vec<u64> = (0..8192u64).map(mix).collect();
    let config = BuildConfig {
        lambda: 32.0,
        alpha: 0.02,
        with_fingerprints: false,
        ..BuildConfig::default()
    };
    let pool = engine::__internal::pool();
    let (result, metrics) =
        measure(|| pool.install(|| Builder::new().with_config(config).build(&keys)));
    let index = result.unwrap();
    let reported = index.memory_usage();
    let wire = validate_raw(&index, keys.iter().copied(), keys.len());
    row(label, "large_buckets", keys.len(), metrics, reported, wire);
}

#[cfg(feature = "candidate")]
fn external(label: &str, count: usize) {
    let directory = std::path::Path::new("target/mph-external-scratch");
    std::fs::create_dir_all(directory).unwrap();
    assert_eq!(std::fs::read_dir(directory).unwrap().count(), 0, "scratch must start empty");
    let config = BuildConfig { with_fingerprints: false, ..BuildConfig::default() };
    let budget = 1024 * 1024;
    // The key iterator generates each value once; it owns no N-element array.
    let (result, metrics) = measure(|| {
        engine::MphExternalBuilder::new()
            .with_config(config)
            .with_memory_budget_bytes(budget)
            .build((0..count as u64).map(mix), count, directory)
    });
    let index = result.unwrap();
    assert!(metrics.peak <= budget + index.memory_usage(), "auxiliary heap exceeded budget");
    assert_eq!(
        std::fs::read_dir(directory).unwrap().count(),
        0,
        "owned scratch files must be removed"
    );
    let bytes = index.to_bytes();
    let raw = read_ptrhash25(&bytes, &mut 0).unwrap();
    let restored = engine::MphIndex::from_bytes(&bytes).unwrap();
    assert_eq!(restored.to_bytes(), bytes);
    let wire = validate_raw(&raw, (0..count as u64).map(mix), count);
    for key in (0..count as u64).map(mix) {
        assert_eq!(index.index_u64(key), restored.index_u64(key));
        assert_eq!(index.index_u64(key), raw.index_u64(key));
    }
    row(label, "external_1mib", count, metrics, index.memory_usage(), wire);
}

fn main() {
    let arguments: Vec<_> = std::env::args().collect();
    let label = arguments.get(1).map_or("current", String::as_str);
    let count = arguments.get(2).map_or(1_000_000, |text| text.parse::<usize>().unwrap());
    assert!((8192..=1_000_000).contains(&count));
    if let Ok(core) = std::env::var("KIRA_BENCH_CORE") {
        assert!(core_affinity::set_for_current(core_affinity::CoreId {
            id: core.parse().unwrap()
        }));
    }
    // Pool startup is outside every operation timer and allocation checkpoint.
    let _ = engine::__internal::pool();
    eprintln!(
        "native_hugepages={}; allocator counters exclude native mappings",
        engine::hugepages_available()
    );
    println!(
        "label,case,keys,build_ms,heap_retained_bytes,heap_peak_bytes,reported_index_bytes,wire_bytes,correctness"
    );
    ordinary(label, count);
    oversized_duplicate(label);
    large_buckets(label);
    #[cfg(feature = "candidate")]
    external(label, count);
}
