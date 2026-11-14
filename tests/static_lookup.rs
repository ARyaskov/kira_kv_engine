use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use kira_kv_engine::{FingerprintBits, Index, IndexBuilder, MphConfig};

struct AllocationProbe;

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

fn record_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(previous) = count.get() {
            count.set(Some(previous + 1));
        }
    });
}

// SAFETY: allocation requests are forwarded unchanged to System. The optional
// thread-local counter neither allocates nor changes ownership or storage.
unsafe impl GlobalAlloc for AllocationProbe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: AllocationProbe = AllocationProbe;

fn mixed_keys() -> Vec<Vec<u8>> {
    let mut keys: Vec<_> = (0..70_000u64).map(|value| value.to_le_bytes().to_vec()).collect();
    keys.extend((0..=129).filter(|&length| length != 8).map(|length| {
        (0..length).map(|offset| (length * 17 + offset * 31) as u8).collect::<Vec<_>>()
    }));
    keys
}

fn queries(keys: &[Vec<u8>]) -> Vec<Vec<u8>> {
    keys.iter()
        .step_by(97)
        .chain(keys.iter().skip(70_000))
        .cloned()
        .chain([b"absent byte-key payload".to_vec(), vec![0; 256], u64::MAX.to_le_bytes().to_vec()])
        .collect()
}

fn assert_batch_contract(index: &Index, query: &[Vec<u8>]) {
    for length in [0, 1, 7, 15, 16, 31, 63, 64, 65, 127, 129, query.len()] {
        let input: Vec<_> = query[..length].iter().map(Vec::as_slice).collect();
        let expected: Vec<_> = input.iter().map(|key| index.lookup(key).ok()).collect();
        assert_eq!(index.lookup_batch_pipelined(&input), expected);
        let mut out = vec![Some(usize::MAX); length + 3];
        index.lookup_batch_pipelined_into(&input, &mut out);
        assert_eq!(&out[..length], expected);
        assert_eq!(&out[length..], &[Some(usize::MAX); 3]);
        // Reuse the same output with misses after hits to reject stale outputs.
        let absent = [b"missing reused output".as_slice()];
        index.lookup_batch_pipelined_into(&absent, &mut out);
        assert_eq!(out[0], index.lookup(absent[0]).ok());
    }
}

#[test]
fn byte_pipeline_preserves_membership_widths_tails_and_mapped_formats() {
    let keys = mixed_keys();
    let query = queries(&keys);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("lookup.mmap");
    for bloom in [false, true] {
        for width in [FingerprintBits::Bits16, FingerprintBits::Bits24, FingerprintBits::Bits32] {
            let index = IndexBuilder::new()
                .with_parallel_build(false)
                .with_bloom_filter(bloom)
                .with_fingerprint_bits(width)
                .build_index_ref(&keys)
                .unwrap();
            let restored = Index::from_bytes(&index.to_bytes().unwrap()).unwrap();
            index.save_mmap(&path).unwrap();
            let mapped = Index::open_mmap(&path).unwrap();
            for instance in [&index, &restored, &mapped] {
                assert_batch_contract(instance, &query);
            }
            drop(mapped);
        }
    }
}

#[test]
fn caller_owned_byte_results_allocate_nothing_and_wrapper_allocates_once() {
    let keys = mixed_keys();
    let query = queries(&keys);
    let input: Vec<_> = query.iter().map(Vec::as_slice).collect();
    for lean in [false, true] {
        let index = IndexBuilder::new()
            .with_parallel_build(false)
            .with_lean_mph(lean)
            .build_index_ref(&keys)
            .unwrap();
        let expected: Vec<_> = input.iter().map(|key| index.lookup(key).ok()).collect();
        let mut out = vec![None; input.len()];
        index.lookup_batch_pipelined_into(&input, &mut out);
        ALLOCATIONS.with(|count| count.set(Some(0)));
        index.lookup_batch_pipelined_into(&input, &mut out);
        let count = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        assert_eq!(count, 0);
        assert_eq!(out, expected);
        ALLOCATIONS.with(|count| count.set(Some(0)));
        let result = index.lookup_batch_pipelined(&input);
        let count = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        assert_eq!(count, 1, "only the result vector may allocate");
        assert_eq!(result, expected);
    }
}

#[test]
fn byte_pipeline_preserves_legacy_hashes_map_fallback_and_empty_outputs() {
    let query = vec![
        b"a".to_vec(),
        b"bc".to_vec(),
        b"hello".to_vec(),
        b"12345678".to_vec(),
        b"long-serialized-key".to_vec(),
        b"absent".to_vec(),
    ];
    for fixture in [
        include_bytes!("fixtures/index_v081.bin").as_slice(),
        include_bytes!("fixtures/index_tag0.bin").as_slice(),
        include_bytes!("fixtures/index_tag2.bin").as_slice(),
    ] {
        let index = Index::from_bytes(fixture).unwrap();
        let repeated: Vec<_> = query.iter().cycle().take(193).cloned().collect();
        assert_batch_contract(&index, &repeated);
    }
    let keys: Vec<_> = (0..257u64).map(|value| value.to_le_bytes().to_vec()).collect();
    let index = IndexBuilder::new()
        .with_parallel_build(false)
        .with_mph_config(MphConfig { lambda: 1_000_000.0, max_rehash: 0, ..MphConfig::default() })
        .build_index_ref(&keys)
        .unwrap();
    let query: Vec<_> = keys.iter().cloned().chain(query).collect();
    assert_batch_contract(&index, &query);
    assert_batch_contract(&Index::empty(), &query);
    let mut out = [Some(17); 2];
    Index::empty().lookup_batch_pipelined_into(&[], &mut out);
    assert_eq!(out, [Some(17); 2]);
}

#[test]
#[should_panic(expected = "out slice too small")]
fn caller_owned_byte_results_reject_short_output() {
    Index::empty().lookup_batch_pipelined_into(&[b"key"], &mut []);
}

#[test]
fn legacy_map_batches_honor_the_same_bloom_rejection_as_scalar_lookup() {
    use kira_kv_engine::__internal::{BlockBloom, simd_hash::hash_u64_scalar};

    let key = 7u64.to_le_bytes();
    let mut canonical = [0u64];
    hash_u64_scalar(&[7], 0, &mut canonical);
    // Legacy tag2 permits a canonical-hash map and an independent Bloom table.
    // A structurally valid zero table rejects the key before map membership.
    let mut bytes = vec![2];
    bytes.extend_from_slice(&1u64.to_le_bytes()); // Key count.
    bytes.extend_from_slice(&0u64.to_le_bytes()); // Canonical-hash seed.
    bytes.extend_from_slice(&[0, 1]); // PtrHash25 dispatch, map storage.
    bytes.extend_from_slice(&1u64.to_le_bytes()); // One map entry.
    bytes.extend_from_slice(&canonical[0].to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.push(1); // Bloom present.
    BlockBloom::build_from_prehashed(&[]).write_to(&mut bytes);
    bytes.push(0); // No outer fingerprints.
    let index = Index::from_bytes(&bytes).unwrap();
    assert!(index.lookup(&key).is_err());
    for length in [0, 1, 15, 16, 64, 65, 129] {
        let input = vec![key.as_slice(); length];
        assert_eq!(index.lookup_batch_pipelined(&input), vec![None; length]);
        let mut output = vec![Some(17); length + 1];
        index.lookup_batch_pipelined_into(&input, &mut output);
        assert_eq!(&output[..length], vec![None; length]);
        assert_eq!(output[length], Some(17));
    }
}
