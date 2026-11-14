use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use kira_kv_engine::{FingerprintBits, Index, IndexBuilder, MphConfig};

struct AllocationProbe;

thread_local! {
    static LARGEST_ALLOCATION: Cell<Option<usize>> = const { Cell::new(None) };
}

fn record_allocation(size: usize) {
    let _ = LARGEST_ALLOCATION.try_with(|probe| {
        if let Some(previous) = probe.get() {
            probe.set(Some(previous.max(size)));
        }
    });
}

// SAFETY: requests are forwarded unchanged to System; the thread-local probe
// neither allocates nor changes pointer ownership or initialization.
unsafe impl GlobalAlloc for AllocationProbe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: AllocationProbe = AllocationProbe;

#[test]
fn saving_large_mmap_tables_has_bounded_auxiliary_allocations() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bounded.mmap");
    let keys: Vec<_> = (0..250_000u64).map(u64::to_le_bytes).collect();
    let index = IndexBuilder::new()
        .with_parallel_build(false)
        .with_fingerprint_bits(FingerprintBits::Bits32)
        .build_index_ref(&keys)
        .unwrap();
    LARGEST_ALLOCATION.with(|probe| probe.set(Some(0)));
    let result = index.save_mmap(&path);
    let largest = LARGEST_ALLOCATION.with(|probe| probe.replace(None).unwrap());
    result.unwrap();
    assert!(largest <= 128 * 1024, "save allocated a table-sized buffer: {largest}");
    let restored = Index::open_mmap(&path).unwrap();
    for key in &keys {
        assert_eq!(restored.lookup(key).unwrap(), index.lookup(key).unwrap());
    }
}

#[test]
fn streamed_owned_and_mapped_tables_preserve_bytes_for_all_fingerprint_widths() {
    let directory = tempfile::tempdir().unwrap();
    let original_path = directory.path().join("owned.mmap");
    let rewritten_path = directory.path().join("mapped.mmap");
    let keys: Vec<_> = (0..20_000u64).map(u64::to_le_bytes).collect();
    for width in [FingerprintBits::Bits16, FingerprintBits::Bits24, FingerprintBits::Bits32] {
        for bloom in [false, true] {
            let index = IndexBuilder::new()
                .with_parallel_build(false)
                .with_bloom_filter(bloom)
                .with_fingerprint_bits(width)
                .build_index_ref(&keys)
                .unwrap();
            index.save_mmap(&original_path).unwrap();
            let original = std::fs::read(&original_path).unwrap();
            let mapped = Index::open_mmap(&original_path).unwrap();
            mapped.save_mmap(&rewritten_path).unwrap();
            assert_eq!(std::fs::read(&rewritten_path).unwrap(), original);
            let restored = Index::open_mmap(&rewritten_path).unwrap();
            for key in &keys {
                assert_eq!(restored.lookup(key).unwrap(), index.lookup(key).unwrap());
            }
        }
    }
}

#[test]
fn empty_lean_and_forced_map_fallback_keep_their_public_mmap_behavior() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fallback.mmap");
    Index::empty().save_mmap(&path).unwrap();
    let empty = Index::open_mmap(&path).unwrap();
    assert!(empty.is_empty());
    assert!(empty.lookup(b"missing").is_err());
    drop(empty);
    let keys: Vec<_> = (0..257u64).map(u64::to_le_bytes).collect();
    for fallback in [false, true] {
        let config = MphConfig {
            lambda: if fallback { 1_000_000.0 } else { 3.0 },
            max_rehash: 0,
            ..MphConfig::default()
        };
        let index = IndexBuilder::new()
            .with_parallel_build(false)
            .with_lean_mph(true)
            .with_mph_config(config)
            .build_index_ref(&keys)
            .unwrap();
        index.save_mmap(&path).unwrap();
        let mapped = Index::open_mmap(&path).unwrap();
        for key in &keys {
            assert_eq!(mapped.lookup(key).unwrap(), index.lookup(key).unwrap());
        }
        if fallback {
            let reader = kira_kv_engine::__internal::MmapIndex::open(&path).unwrap();
            let header = reader.parse_header().unwrap();
            let payload = reader
                .section(&header, kira_kv_engine::__internal::SectionKind::LegacyPayload)
                .expect("forced fallback uses the existing payload section");
            assert_eq!(payload, index.to_bytes().unwrap());
        }
    }
}

#[test]
fn frozen_legacy_hash_models_keep_positions_after_streamed_mmap_resave() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("legacy.mmap");
    let keys = [b"a".as_slice(), b"bc", b"hello", b"12345678", b"long-serialized-key"];
    for fixture in [
        include_bytes!("fixtures/index_v081.bin").as_slice(),
        include_bytes!("fixtures/index_tag0.bin").as_slice(),
        include_bytes!("fixtures/index_tag2.bin").as_slice(),
    ] {
        let index = Index::from_bytes(fixture).unwrap();
        index.save_mmap(&path).unwrap();
        let mapped = Index::open_mmap(&path).unwrap();
        for key in keys {
            assert_eq!(mapped.lookup(key).unwrap(), index.lookup(key).unwrap());
        }
    }
}

#[test]
fn streaming_keeps_frozen_mmap_bytes_and_compact_build_positions() {
    let directory = tempfile::tempdir().unwrap();
    let original_path = directory.path().join("baseline.mmap");
    let rewritten_path = directory.path().join("streamed.mmap");
    let fixture = include_bytes!("fixtures/mmap_stream_fp24_v081.bin");
    let keys: Vec<_> = (0..257u64).map(u64::to_le_bytes).collect();
    let index = IndexBuilder::new()
        .with_parallel_build(false)
        .with_fingerprint_bits(FingerprintBits::Bits24)
        .build_index_ref(&keys)
        .unwrap();
    index.save_mmap(&rewritten_path).unwrap();
    let compact_bytes = std::fs::read(&rewritten_path).unwrap();
    assert!(compact_bytes.len() < fixture.len(), "small builds use fewer Bloom blocks");
    let compact = Index::open_mmap(&rewritten_path).unwrap();
    for key in &keys {
        assert_eq!(compact.lookup(key).unwrap(), index.lookup(key).unwrap());
    }
    drop(compact);
    std::fs::write(&original_path, fixture).unwrap();
    let old = Index::open_mmap(&original_path).unwrap();
    old.save_mmap(&rewritten_path).unwrap();
    assert_eq!(std::fs::read(&rewritten_path).unwrap(), fixture);
    for key in &keys {
        assert_eq!(old.lookup(key).unwrap(), index.lookup(key).unwrap());
    }
}

#[test]
fn lean_unit_load_keeps_an_empty_final_remap_section_readable() {
    use kira_kv_engine::__internal::{MmapIndex, SectionKind};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty-remap.mmap");
    let config = MphConfig { alpha: 1.0, ..MphConfig::default() };
    let index = IndexBuilder::new()
        .with_parallel_build(false)
        .with_lean_mph(true)
        .with_mph_config(config)
        .build_index_ref(&[7u64.to_le_bytes()])
        .unwrap();
    index.save_mmap(&path).unwrap();
    let reader = MmapIndex::open(&path).unwrap();
    let header = reader.parse_header().unwrap();
    assert_eq!(header.sections.last().unwrap().kind, SectionKind::PtrHash25Remap as u32);
    assert!(reader.section(&header, SectionKind::PtrHash25Remap).unwrap().is_empty());
    let restored = Index::open_mmap(&path).unwrap();
    assert_eq!(restored.lookup_u64(7).unwrap(), 0);
}

#[test]
fn forced_map_optional_membership_payloads_match_the_ordinary_serializer() {
    use kira_kv_engine::__internal::{MmapIndex, SectionKind};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("map-membership.mmap");
    let keys: Vec<_> = (0..257u64).map(u64::to_le_bytes).collect();
    for width in [FingerprintBits::Bits16, FingerprintBits::Bits24, FingerprintBits::Bits32] {
        for bloom in [false, true] {
            let config = MphConfig { lambda: 1_000_000.0, max_rehash: 0, ..MphConfig::default() };
            let index = IndexBuilder::new()
                .with_parallel_build(false)
                .with_mph_config(config)
                .with_bloom_filter(bloom)
                .with_fingerprint_bits(width)
                .build_index_ref(&keys)
                .unwrap();
            index.save_mmap(&path).unwrap();
            let reader = MmapIndex::open(&path).unwrap();
            let header = reader.parse_header().unwrap();
            let payload = reader.section(&header, SectionKind::LegacyPayload).unwrap();
            assert_eq!(payload, index.to_bytes().unwrap());
            let restored = Index::open_mmap(&path).unwrap();
            for key in &keys {
                assert_eq!(restored.lookup(key).unwrap(), index.lookup(key).unwrap());
            }
        }
    }
}
