use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use kira_kv_engine::__internal::{Builder, PtrHash25Error, read_ptrhash25, write_ptrhash25};
use kira_kv_engine::{MphConfig, MphExternalBuildError, MphExternalBuilder, MphIndex};

#[derive(Clone, Copy)]
struct Probe {
    live: isize,
    peak: usize,
}

thread_local! { static ALLOCATIONS: Cell<Option<Probe>> = const { Cell::new(None) }; }

fn allocate(size: usize) {
    let _ = ALLOCATIONS.try_with(|state| {
        if let Some(mut probe) = state.get() {
            probe.live += size as isize;
            probe.peak = probe.peak.max(probe.live.max(0) as usize);
            state.set(Some(probe));
        }
    });
}

fn deallocate(size: usize) {
    let _ = ALLOCATIONS.try_with(|state| {
        if let Some(mut probe) = state.get() {
            probe.live -= size as isize;
            state.set(Some(probe));
        }
    });
}

struct Allocator;

// SAFETY: every allocation request is forwarded unchanged to System; the
// thread-local accounting does not allocate or alter pointer ownership.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            allocate(layout.size());
        }
        result
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            allocate(layout.size());
        }
        result
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            deallocate(layout.size());
            allocate(size);
        }
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        deallocate(layout.size());
        unsafe {
            System.dealloc(pointer, layout);
        }
    }
}

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[test]
fn one_pass_stream_has_a_budgeted_peak_and_every_slot_is_bijective() {
    let directory = tempfile::tempdir().unwrap();
    let count = 250_000usize;
    let budget = 1024 * 1024;
    let builder = MphExternalBuilder::new().with_memory_budget_bytes(budget);
    ALLOCATIONS.with(|state| state.set(Some(Probe { live: 0, peak: 0 })));
    let result = builder.build((0..count as u64).map(mix), count, directory.path());
    let peak = ALLOCATIONS.with(|state| state.replace(None).unwrap().peak);
    let index = result.unwrap();
    assert!(
        peak <= budget + index.memory_usage(),
        "peak {peak} exceeds working budget plus final index {}",
        index.memory_usage()
    );
    let mut seen = vec![false; count];
    assert_eq!(index.len(), count);
    assert!(!index.is_empty());
    for key in (0..count as u64).map(mix) {
        let slot = index.index_u64(key) as usize;
        assert!(slot < count && !seen[slot]);
        seen[slot] = true;
        assert_eq!(index.lookup_u64(key), Some(slot as u32));
    }
    assert!(seen.into_iter().all(|value| value));
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}

#[test]
fn external_and_existing_raw_payloads_remain_interoperable() {
    let directory = tempfile::tempdir().unwrap();
    let keys: Vec<_> = (0..2048u64).map(mix).collect();
    for aes in [false, true] {
        for fingerprints in [false, true] {
            let config = MphConfig {
                use_aes_hash: aes,
                with_fingerprints: fingerprints,
                ..MphConfig::default()
            };
            let index = MphExternalBuilder::new()
                .with_config(config.clone())
                .build(keys.iter().copied(), keys.len(), directory.path())
                .unwrap();
            let bytes = index.to_bytes();
            let restored = MphIndex::from_bytes(&bytes).unwrap();
            assert_eq!(restored.to_bytes(), bytes);
            let mut position = 0;
            let raw = read_ptrhash25(&bytes, &mut position).unwrap();
            assert_eq!(position, bytes.len());
            for &key in &keys {
                assert_eq!(index.index_u64(key), restored.index_u64(key));
                assert_eq!(index.index_u64(key), raw.index_u64(key));
                assert_eq!(index.lookup_u64(key), Some(index.index_u64(key)));
            }
            let ordinary = Builder::new().with_config(config).build(&keys).unwrap();
            let mut bytes = Vec::new();
            write_ptrhash25(&ordinary, &mut bytes);
            let wrapped = MphIndex::from_bytes(&bytes).unwrap();
            for &key in &keys {
                assert_eq!(wrapped.index_u64(key), ordinary.index_u64(key));
            }
        }
    }
}

#[test]
fn count_and_duplicate_failures_remove_only_owned_temporary_files() {
    let directory = tempfile::tempdir().unwrap();
    let sentinel = directory.path().join("keep.txt");
    std::fs::write(&sentinel, b"caller-owned").unwrap();
    for (actual, expected) in [(9, 10), (11, 10)] {
        let error = MphExternalBuilder::new()
            .build(0..actual as u64, expected, directory.path())
            .unwrap_err();
        assert!(
            matches!(error, MphExternalBuildError::KeyCountMismatch { expected: 10, observed } if observed == actual)
        );
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }
    let error = MphExternalBuilder::new().build([1, 7, 9, 7], 4, directory.path()).unwrap_err();
    assert!(matches!(error, MphExternalBuildError::Build(PtrHash25Error::DuplicateKey)));
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    assert_eq!(std::fs::read(sentinel).unwrap(), b"caller-owned");
}

#[test]
fn budget_and_configuration_reject_before_consuming_the_iterator() {
    let directory = tempfile::tempdir().unwrap();
    let reads = Cell::new(0);
    let keys = || (0..10u64).inspect(|_| reads.set(reads.get() + 1));
    let error = MphExternalBuilder::new()
        .with_memory_budget_bytes(1024)
        .build(keys(), 10, directory.path())
        .unwrap_err();
    assert!(
        matches!(error, MphExternalBuildError::BudgetExceeded { required, budget: 1024 } if required > 1024)
    );
    let config = MphConfig { lambda: f64::NAN, ..MphConfig::default() };
    let error = MphExternalBuilder::new()
        .with_config(config)
        .build(keys(), 10, directory.path())
        .unwrap_err();
    assert!(matches!(error, MphExternalBuildError::InvalidConfig(_)));
    assert_eq!(reads.get(), 0);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}

#[test]
fn input_unwind_and_io_failures_preserve_the_callers_directory() {
    let directory = tempfile::tempdir().unwrap();
    let result = std::panic::catch_unwind(|| {
        let keys = (0..10u64).inspect(|&key| {
            assert!(key != 5, "iterator unwind");
        });
        let _ = MphExternalBuilder::new().build(keys, 10, directory.path());
    });
    assert!(result.is_err());
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    let sentinel = directory.path().join("file");
    std::fs::write(&sentinel, b"caller-owned").unwrap();
    assert!(matches!(
        MphExternalBuilder::new().build([7], 1, &sentinel),
        Err(MphExternalBuildError::Io(_))
    ));
    assert_eq!(std::fs::read(sentinel).unwrap(), b"caller-owned");
}

#[test]
fn empty_and_trailing_or_truncated_raw_inputs_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    assert!(matches!(
        MphExternalBuilder::new().build([], 0, directory.path()),
        Err(MphExternalBuildError::EmptyInput)
    ));
    let index = MphExternalBuilder::new().build([0, 7, u64::MAX], 3, directory.path()).unwrap();
    let mut bytes = index.to_bytes();
    for length in [0, 1, 7, bytes.len() - 1] {
        assert!(MphIndex::from_bytes(&bytes[..length]).is_err());
    }
    bytes.push(0);
    assert!(MphIndex::from_bytes(&bytes).is_err());
}
