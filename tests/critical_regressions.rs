use kira_kv_engine::{
    DynamicConfig, DynamicIndex, HybridBuilder, HybridIndex, Index, IndexBuilder, IndexError,
    PgmIndex, PgmIndexU128,
};

const LEGACY_KEYS: &[&[u8]] = &[b"a", b"bc", b"hello", b"12345678", b"long-serialized-key"];
const LEGACY_SLOTS: &[usize] = &[3, 0, 2, 4, 1];

#[test]
fn distinct_short_binary_keys_build_and_roundtrip() {
    let keys: &[&[u8]] = &[&[3], &[0, 0], &[2], &[1, 0], b"", b"a", b"a\0", b"12345678"];
    for lean in [false, true] {
        let index = IndexBuilder::new().with_lean_mph(lean).build_index_ref(keys).unwrap();
        let restored = Index::from_bytes(&index.to_bytes().unwrap()).unwrap();
        let mut slots: Vec<usize> = keys.iter().map(|key| index.lookup(key).unwrap()).collect();
        slots.sort_unstable();
        assert_eq!(slots, (0..keys.len()).collect::<Vec<_>>());
        let refs: Vec<&[u8]> = keys.to_vec();
        for instance in [&index, &restored] {
            for ((key, batch), pipeline) in keys
                .iter()
                .zip(instance.lookup_batch(&refs))
                .zip(instance.lookup_batch_pipelined(&refs))
            {
                let expected = index.lookup(key).unwrap();
                assert_eq!(instance.lookup(key).unwrap(), expected);
                assert_eq!(batch, Some(expected));
                assert_eq!(pipeline, Some(expected));
                assert!(instance.contains(key));
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("index.mmap");
        index.save_mmap(&path).unwrap();
        let mapped = Index::open_mmap(&path).unwrap();
        for key in keys {
            assert_eq!(mapped.lookup(key).unwrap(), index.lookup(key).unwrap());
        }
    }
    assert!(matches!(
        IndexBuilder::new().build_index_ref(&[b"a".as_slice(), b"a".as_slice()]),
        Err(IndexError::DuplicateKey)
    ));
}

#[test]
fn hybrid_and_dynamic_accept_distinct_short_keys() {
    let keys: &[&[u8]] = &[&[3], &[0, 0], b"a", b"a\0"];
    let hybrid = HybridBuilder::new().build(keys).unwrap();
    let restored = HybridIndex::from_bytes(&hybrid.to_bytes()).unwrap();
    let mut dynamic =
        DynamicIndex::with_config(DynamicConfig { flush_threshold: 2, ..DynamicConfig::default() });
    let ids: Vec<_> = keys.iter().map(|key| dynamic.insert(key.to_vec())).collect();
    dynamic.flush().unwrap();
    dynamic.compact().unwrap();
    let back = DynamicIndex::from_bytes(&dynamic.to_bytes()).unwrap();
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(hybrid.lookup(key), Some(i as u32));
        assert_eq!(restored.lookup(key), Some(i as u32));
        assert_eq!(dynamic.lookup(key), Some(ids[i]));
        assert_eq!(back.lookup(key), Some(ids[i]));
    }
}

fn assert_legacy_index(index: &Index) {
    let batched = index.lookup_batch_pipelined(LEGACY_KEYS);
    for ((key, &slot), batch) in LEGACY_KEYS.iter().zip(LEGACY_SLOTS).zip(batched) {
        assert_eq!(index.lookup(key).unwrap(), slot);
        assert_eq!(batch, Some(slot));
        assert!(index.contains(key));
    }
    assert!(index.contains_batch(LEGACY_KEYS).iter().all(|&hit| hit));
    let queries: Vec<&[u8]> = LEGACY_KEYS.iter().copied().cycle().take(131).collect();
    for (query, result) in queries.iter().zip(index.lookup_batch_pipelined(&queries)) {
        assert_eq!(result, Some(index.lookup(query).unwrap()));
    }
}

#[test]
fn frozen_legacy_indexes_keep_hashes_and_slots_after_resaving() {
    for bytes in [
        include_bytes!("fixtures/index_v081.bin").as_slice(),
        include_bytes!("fixtures/index_tag0.bin").as_slice(),
        include_bytes!("fixtures/index_tag2.bin").as_slice(),
    ] {
        let index = Index::from_bytes(bytes).unwrap();
        assert_legacy_index(&index);
        assert_legacy_index(&Index::from_bytes(&index.to_bytes().unwrap()).unwrap());
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("index.mmap");
        index.save_mmap(&path).unwrap();
        assert_legacy_index(&Index::open_mmap(&path).unwrap());
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("legacy.mmap");
    std::fs::write(&path, include_bytes!("fixtures/index_mmap_v081.bin")).unwrap();
    let mapped = Index::open_mmap(&path).unwrap();
    assert_legacy_index(&mapped);
    assert_legacy_index(&Index::from_bytes(&mapped.to_bytes().unwrap()).unwrap());
}

#[test]
fn frozen_legacy_hybrid_keeps_hashes_after_resaving() {
    let old = HybridIndex::from_bytes(include_bytes!("fixtures/hybrid_v081.bin")).unwrap();
    let back = HybridIndex::from_bytes(&old.to_bytes()).unwrap();
    for (i, key) in LEGACY_KEYS.iter().enumerate() {
        assert_eq!(old.lookup(key), Some(i as u32));
        assert_eq!(back.lookup(key), Some(i as u32));
    }
}

#[test]
fn frozen_legacy_pgm_keeps_sorted_ranks() {
    let old = PgmIndex::from_bytes(include_bytes!("fixtures/pgm_v4.bin")).unwrap();
    let back = PgmIndex::from_bytes(&old.to_bytes().unwrap()).unwrap();
    for (i, key) in [0, 2, 4, 1000, 1001].into_iter().enumerate() {
        assert_eq!(old.index(key).unwrap(), i);
        assert_eq!(back.index(key).unwrap(), i);
    }
}

#[test]
fn u128_bounds_include_maximum_and_absent_keys() {
    let keys = vec![0, 1, 2, 1000, 1001, u128::MAX - 1, u128::MAX];
    for epsilon in [0, 1, 64] {
        let index = PgmIndexU128::build(keys.clone(), epsilon).unwrap();
        for query in [0, 1, 3, 500, 999, 1000, 1002, u128::MAX - 2, u128::MAX - 1, u128::MAX] {
            assert_eq!(index.lower_bound(query), keys.partition_point(|&key| key < query));
            assert_eq!(index.upper_bound(query), keys.partition_point(|&key| key <= query));
        }
        assert_eq!(index.range(u128::MAX, u128::MAX), 6..7);
    }
}

fn refresh_checksum(bytes: &mut [u8]) {
    let end = bytes.len() - 8;
    let mut state = 0x4B49_5241_5F43_4B53;
    for chunk in bytes[..end].chunks(64 * 1024) {
        state = wyhash::wyhash(chunk, state);
    }
    let sum = wyhash::wyhash(&state.to_le_bytes(), state);
    bytes[end..].copy_from_slice(&sum.to_le_bytes());
}

#[test]
fn u128_loader_rejects_a_finite_model_that_overflows_its_search_window() {
    let index = PgmIndexU128::build(vec![10, 20, 30], 0).unwrap();
    let mut bytes = index.to_bytes();
    let slope_offset = 9 + 4 + 8 + 3 * 16 + 8;
    bytes[slope_offset..slope_offset + 8].copy_from_slice(&f64::MAX.to_le_bytes());
    refresh_checksum(&mut bytes);
    assert!(PgmIndexU128::from_bytes(&bytes).is_err());
}

#[test]
fn legacy_empty_index_rejects_an_unrepresentable_key_count() {
    let mut bytes = vec![3];
    bytes.extend_from_slice(&(1u64 << 32).to_le_bytes());
    assert!(matches!(Index::from_bytes(&bytes), Err(IndexError::CorruptData)));
}

#[test]
fn empty_u128_loader_rejects_unused_overflow_entries() {
    let mut bytes = PgmIndexU128::build(Vec::new(), 0).unwrap().to_bytes();
    let count_offset = 9 + 4 + 8 + 8;
    bytes[count_offset..count_offset + 8].copy_from_slice(&1u64.to_le_bytes());
    bytes.splice(
        count_offset + 8..count_offset + 8,
        0u32.to_le_bytes().into_iter().chain(255u32.to_le_bytes()),
    );
    refresh_checksum(&mut bytes);
    assert!(PgmIndexU128::from_bytes(&bytes).is_err());
}
