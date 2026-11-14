use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use kira_kv_engine::__internal::{MmapIndex, MmapIndexWriter, SectionKind};
use kira_kv_engine::{HybridBuilder, HybridIndex, PgmBuilder, PgmIndex};

struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

fn record_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn assert_queries(index: &PgmIndex, keys: &[u64], queries: &[u64]) {
    for &query in queries {
        assert_eq!(index.index(query).ok(), keys.binary_search(&query).ok(), "index({query})");
        assert_eq!(
            index.lower_bound(query),
            keys.partition_point(|&key| key < query),
            "lower_bound({query})"
        );
        assert_eq!(
            index.upper_bound(query),
            keys.partition_point(|&key| key <= query),
            "upper_bound({query})"
        );
    }
    for pair in queries.windows(2) {
        let start = keys.partition_point(|&key| key < pair[0]);
        let end = keys.partition_point(|&key| key <= pair[1]).max(start);
        assert_eq!(index.range(pair[0], pair[1]), start..end);
    }
}

fn adjacent_queries(keys: &[u64]) -> Vec<u64> {
    keys.iter()
        .flat_map(|&key| [key.saturating_sub(1), key, key.saturating_add(1)])
        .chain([0, u64::MAX])
        .collect()
}

fn raw_payload(bytes: &[u8]) -> &[u8] {
    assert_eq!(bytes[0], 5);
    assert_eq!(&bytes[1..5], b"KIRA");
    &bytes[9..bytes.len() - 8]
}

#[test]
fn epsilon_zero_is_precise_at_the_absolute_intercept_rounding_boundary() {
    let base = (1u64 << 52) + (1u64 << 27);
    let keys = vec![base, base + 1, base + 2];
    let index = PgmBuilder::new().with_epsilon(0).with_parallel(false).build(keys.clone()).unwrap();
    assert_eq!(index.num_segments(), 1);
    assert_eq!(index.stats().max_error, 0);
    assert_queries(&index, &keys, &adjacent_queries(&keys));

    let sealed = index.to_bytes().unwrap();
    assert_eq!(raw_payload(&sealed)[0], 6);
    for bytes in [sealed.as_slice(), raw_payload(&sealed)] {
        let restored = PgmIndex::from_bytes(bytes).unwrap();
        assert_eq!(restored.stats().max_error, 0);
        assert_queries(&restored, &keys, &adjacent_queries(&keys));
    }
}

#[test]
fn relative_segments_are_invariant_under_integer_key_translation() {
    let local: Vec<u64> = (0..4096).map(|i| i * 7 + i % 3).collect();
    for epsilon in [0, 1, 8, 64] {
        let reference = PgmBuilder::new()
            .with_epsilon(epsilon)
            .with_parallel(false)
            .build(local.clone())
            .unwrap();
        for base in [0, (1u64 << 52) + (1u64 << 27), u64::MAX - *local.last().unwrap()] {
            let keys: Vec<u64> = local.iter().map(|&key| base + key).collect();
            let index = PgmBuilder::new()
                .with_epsilon(epsilon)
                .with_parallel(false)
                .build(keys.clone())
                .unwrap();
            assert_eq!(index.num_segments(), reference.num_segments(), "epsilon={epsilon}");
            assert_eq!(index.stats().max_error, reference.stats().max_error);
            assert!(index.stats().max_error <= epsilon + 1);
            assert_queries(&index, &keys, &adjacent_queries(&keys));
        }
    }
}

#[test]
fn dense_maximum_keys_keep_one_segment_and_exact_ranks() {
    let keys: Vec<u64> = (0..8192).map(|i| u64::MAX - 8191 + i).collect();
    for parallel in [false, true] {
        let index =
            PgmBuilder::new().with_epsilon(0).with_parallel(parallel).build(keys.clone()).unwrap();
        assert_eq!(index.num_segments(), 1);
        assert_eq!(index.stats().max_error, 0);
        assert_queries(&index, &keys, &adjacent_queries(&keys));
    }
}

#[test]
fn parallel_chunk_positions_use_the_same_relative_origin() {
    // Cross the actual 2 * 128K parallel threshold, including a partial chunk.
    let n = 2 * 128 * 1024 + 17;
    for parallel in [false, true] {
        let low: Vec<u64> = (0..n as u64).map(|i| i * 3).collect();
        let high: Vec<u64> = low.iter().map(|&key| u64::MAX - low[n - 1] + key).collect();
        let build =
            |keys| PgmBuilder::new().with_epsilon(0).with_parallel(parallel).build(keys).unwrap();
        let low_index = build(low.clone());
        let high_index = build(high.clone());
        assert_eq!(low_index.num_segments(), high_index.num_segments());
        assert_eq!(low_index.stats().max_error, high_index.stats().max_error);
        assert!(high_index.stats().max_error <= 1);
        let queries = adjacent_queries(&high.iter().copied().step_by(997).collect::<Vec<_>>());
        assert_queries(&high_index, &high, &queries);
        for boundary in [128 * 1024 - 1, 128 * 1024, 2 * 128 * 1024 - 1, n - 1] {
            assert_eq!(high_index.index(high[boundary]).ok(), Some(boundary));
        }
    }
}

#[test]
fn full_u64_span_and_large_epsilon_match_sorted_array() {
    let keys = vec![0, 1, 3, 4, 1 << 32, 1 << 53, u64::MAX - 2, u64::MAX - 1, u64::MAX];
    for epsilon in [0, 1, 8, 255, u32::MAX] {
        let index = PgmBuilder::new()
            .with_epsilon(epsilon)
            .with_parallel(false)
            .build(keys.clone())
            .unwrap();
        assert!(index.stats().max_error <= epsilon.saturating_add(1));
        assert_queries(&index, &keys, &adjacent_queries(&keys));
        let restored = PgmIndex::from_bytes(&index.to_bytes().unwrap()).unwrap();
        assert_queries(&restored, &keys, &adjacent_queries(&keys));
    }
}

#[test]
fn compressed_small_and_large_windows_are_exact_without_query_allocations() {
    let keys: Vec<u64> = (0..16_000).map(|i| i * i * 3 + i).collect();
    let queries = adjacent_queries(&keys.iter().copied().step_by(47).collect::<Vec<_>>());
    for epsilon in [2, 512] {
        let mut index = PgmBuilder::new()
            .with_epsilon(epsilon)
            .with_parallel(false)
            .build(keys.clone())
            .unwrap();
        if epsilon == 512 {
            assert!(index.stats().max_error > 24, "exercise compressed binary search");
        } else {
            assert!(index.stats().max_error <= 3, "exercise stack-buffer SIMD search");
        }
        assert!(index.compact_keys() > 0, "exercise actual compressed storage");
        assert_queries(&index, &keys, &queries);
        // Warm up runtime SIMD dispatch before observing allocations.
        let _ = index.index(keys[100]);
        ALLOCATIONS.with(|count| count.set(Some(0)));
        let mut correct = true;
        for &query in &queries {
            correct &= index.index(query).ok() == keys.binary_search(&query).ok();
            correct &= index.lower_bound(query) == keys.partition_point(|&key| key < query);
            correct &= index.upper_bound(query) == keys.partition_point(|&key| key <= query);
            correct &= index.range(query, query)
                == (keys.partition_point(|&key| key < query)
                    ..keys.partition_point(|&key| key <= query));
        }
        let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        assert!(correct);
        assert_eq!(allocations, 0, "compressed queries must use constant stack space");
        let restored = PgmIndex::from_bytes(&index.to_bytes().unwrap()).unwrap();
        assert_queries(&restored, &keys, &queries);
    }
}

#[test]
fn frozen_absolute_models_keep_their_representation_and_error_budget() {
    let ordinary_keys: Vec<u64> = (0..256).map(|i| i * 13 + 7).collect();
    let base = (1u64 << 52) + (1u64 << 27);
    let offset_keys = vec![base, base + 1, base + 2];
    for (bytes, keys) in [
        (include_bytes!("fixtures/pgm_absolute_v081.bin").as_slice(), ordinary_keys),
        (include_bytes!("fixtures/pgm_absolute_offset_v081.bin").as_slice(), offset_keys),
    ] {
        let index = PgmIndex::from_bytes(bytes).unwrap();
        let original_error = index.stats().max_error;
        if keys.len() == 3 {
            assert_eq!(original_error, 1 << 27);
        }
        assert_queries(&index, &keys, &adjacent_queries(&keys));
        let saved = index.to_bytes().unwrap();
        assert_eq!(raw_payload(&saved)[0], 4);
        assert_eq!(raw_payload(&saved), raw_payload(bytes));
        for representation in [saved.as_slice(), raw_payload(&saved)] {
            let restored = PgmIndex::from_bytes(representation).unwrap();
            assert_eq!(restored.stats().max_error, original_error);
            assert_queries(&restored, &keys, &adjacent_queries(&keys));
        }
    }
}

#[test]
fn frozen_and_current_hybrid_embedded_models_roundtrip() {
    let old =
        HybridIndex::from_bytes(include_bytes!("fixtures/hybrid_absolute_pgm_v081.bin")).unwrap();
    let old_again = HybridIndex::from_bytes(&old.to_bytes()).unwrap();
    for i in 0..256u64 {
        assert_eq!(old.lookup_u64(i * 13 + 7), Some(i as u32));
        assert_eq!(old_again.lookup_u64(i * 13 + 7), Some(i as u32));
    }
    let keys: Vec<u64> = (0..2048).rev().map(|i| u64::MAX - i * 7).collect();
    let index = HybridBuilder::new().with_pgm_epsilon(0).build_from_u64(&keys).unwrap();
    let restored = HybridIndex::from_bytes(&index.to_bytes()).unwrap();
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(index.lookup_u64(key), Some(i as u32));
        assert_eq!(restored.lookup_u64(key), Some(i as u32));
    }
}

static FILE_ID: AtomicUsize = AtomicUsize::new(0);

fn section_file() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kira_pgm_precision_{}_{}.bin",
        std::process::id(),
        FILE_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn section_roundtrip(index: &PgmIndex) -> PgmIndex {
    let path = section_file();
    let mut writer = MmapIndexWriter::create(&path, index.stats().total_keys as u64).unwrap();
    index.write_to_sections(&mut writer);
    writer.finalize().unwrap();
    let mmap = MmapIndex::open(&path).unwrap();
    let header = mmap.parse_header().unwrap();
    let restored = PgmIndex::read_from_sections(&mmap, &header).unwrap();
    drop(mmap);
    std::fs::remove_file(path).unwrap();
    restored
}

#[test]
fn section_roundtrips_preserve_legacy_relative_and_compressed_models() {
    let keys: Vec<u64> = (0..8192).map(|i| (1 << 52) + i * 13 + 7).collect();
    let mut index = PgmBuilder::new().with_epsilon(0).build(keys.clone()).unwrap();
    let restored = section_roundtrip(&index);
    assert_queries(&restored, &keys, &adjacent_queries(&keys));
    assert!(index.compact_keys() > 0);
    let restored = section_roundtrip(&index);
    assert_queries(&restored, &keys, &adjacent_queries(&keys));

    let legacy = PgmIndex::from_bytes(include_bytes!("fixtures/pgm_absolute_v081.bin")).unwrap();
    let restored = section_roundtrip(&legacy);
    let legacy_keys: Vec<u64> = (0..256).map(|i| i * 13 + 7).collect();
    assert_eq!(raw_payload(&restored.to_bytes().unwrap())[0], 4);
    assert_queries(&restored, &legacy_keys, &adjacent_queries(&legacy_keys));
}

#[test]
fn section_reader_rejects_unknown_encoding_and_partial_arrays() {
    let index = PgmBuilder::new().with_epsilon(0).build(vec![10, 20, 30]).unwrap();
    let path = section_file();
    let mut writer = MmapIndexWriter::create(&path, 3).unwrap();
    index.write_to_sections(&mut writer);
    writer.finalize().unwrap();
    let mmap = MmapIndex::open(&path).unwrap();
    let header = mmap.parse_header().unwrap();
    let sections = [
        SectionKind::PgmKeys,
        SectionKind::PgmSlopes,
        SectionKind::PgmIntercepts,
        SectionKind::PgmMinKeys,
        SectionKind::PgmMaxKeys,
        SectionKind::PgmMaxErrors,
        SectionKind::PgmOverflowErrors,
        SectionKind::PgmStarts,
        SectionKind::PgmEnds,
        SectionKind::PgmMeta,
    ]
    .map(|kind| (kind, mmap.section(&header, kind).unwrap().to_vec()));
    drop(mmap);
    std::fs::remove_file(path).unwrap();
    for case in 0..4 {
        let path = section_file();
        let mut writer = MmapIndexWriter::create(&path, 3).unwrap();
        for (kind, bytes) in &sections {
            let mut bytes = bytes.clone();
            match (case, kind) {
                (0, SectionKind::PgmMeta) => bytes[5] = 2,
                (1, SectionKind::PgmSlopes) => bytes.push(0),
                (2, SectionKind::PgmIntercepts) => bytes.extend_from_slice(&0f32.to_le_bytes()),
                (3, SectionKind::PgmMeta) => bytes[5] = 0,
                _ => {}
            }
            writer.add_section(*kind, bytes);
        }
        writer.finalize().unwrap();
        let mmap = MmapIndex::open(&path).unwrap();
        let header = mmap.parse_header().unwrap();
        assert!(PgmIndex::read_from_sections(&mmap, &header).is_none(), "case {case}");
        drop(mmap);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn raw_relative_reader_rejects_invalid_lines_and_unbounded_error() {
    let index = PgmBuilder::new().with_epsilon(0).build(vec![10, 20, 30]).unwrap();
    let sealed = index.to_bytes().unwrap();
    let raw = raw_payload(&sealed);
    let slope_offset = 21 + 3 * 8 + 8;
    for slope in [f64::NAN, f64::INFINITY, -1.0, 5.0] {
        let mut changed = raw.to_vec();
        changed[slope_offset..slope_offset + 8].copy_from_slice(&slope.to_le_bytes());
        assert!(PgmIndex::from_bytes(&changed).is_err(), "slope={slope}");
    }
    let mut changed = raw.to_vec();
    changed[slope_offset + 8 + 8 + 8] = 2;
    assert!(PgmIndex::from_bytes(&changed).is_err());
    for version in [1, 5, 7, 255] {
        let mut changed = raw.to_vec();
        changed[0] = version;
        assert!(PgmIndex::from_bytes(&changed).is_err(), "version={version}");
    }
}
