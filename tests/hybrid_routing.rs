use kira_kv_engine::__internal::simd_hash::hash_u64_scalar;
use kira_kv_engine::{HybridBuilder, HybridIndex};

fn check_numeric_queries(index: &HybridIndex, keys: &[u64]) {
    let mut hashes = vec![0; keys.len()];
    hash_u64_scalar(keys, index.seed(), &mut hashes);
    let bytes: Vec<_> = keys.iter().map(|key| key.to_le_bytes()).collect();
    for (position, (&key, &hash)) in keys.iter().zip(&hashes).enumerate() {
        assert_eq!(index.lookup_u64(key), Some(position as u32));
        assert_eq!(index.lookup_hash(hash), Some(position as u32));
        assert_eq!(index.lookup(&bytes[position]), Some(position as u32));
    }
    let expected: Vec<_> = (0..keys.len()).map(|position| Some(position as u32)).collect();
    assert_eq!(index.lookup_batch(&bytes), expected);
    assert_eq!(index.lookup_batch_hashes(&hashes), expected);
    assert_eq!(index.lookup_batch_u64_simd(keys), expected);
}

#[test]
fn radix_routes_all_public_numeric_paths_and_preserves_original_positions() {
    let keys: Vec<u64> = (0..24_000).rev().map(|key| key * 997 + 11).collect();
    for epsilon in [0, 8, 64] {
        for lean in [false, true] {
            let index = HybridBuilder::new()
                .with_parallel(false)
                .with_pgm_epsilon(epsilon)
                .with_lean(lean)
                .build_from_u64(&keys)
                .unwrap();
            if epsilon == 0 {
                assert!(index.num_segments() > 32, "exercise radix routing");
            }
            check_numeric_queries(&index, &keys);
            let bytes = index.to_bytes();
            let restored = HybridIndex::from_bytes(&bytes).unwrap();
            assert_eq!(restored.to_bytes(), bytes);
            check_numeric_queries(&restored, &keys);
            for query in [0, 1, u64::MAX, u64::MAX - 1, 24_000 * 997 + 11] {
                assert_eq!(index.lookup_u64(query), restored.lookup_u64(query));
            }
        }
    }
}

#[test]
fn variable_length_keys_and_negative_queries_keep_their_public_behavior() {
    let keys: Vec<_> = (0..8192).rev().map(|i| format!("route-key-{i:06}").into_bytes()).collect();
    let index = HybridBuilder::new().with_parallel(false).with_pgm_epsilon(0).build(&keys).unwrap();
    assert!(index.num_segments() > 32);
    let restored = HybridIndex::from_bytes(&index.to_bytes()).unwrap();
    for instance in [&index, &restored] {
        for (position, key) in keys.iter().enumerate() {
            assert_eq!(instance.lookup(key), Some(position as u32));
        }
        assert_eq!(
            instance.lookup_batch(&keys),
            (0..keys.len()).map(|position| Some(position as u32)).collect::<Vec<_>>()
        );
        for absent in [b"".as_slice(), b"route-key-999999", b"route-key-000001\0"] {
            assert_eq!(instance.lookup(absent), None);
        }
    }
}

#[test]
fn frozen_legacy_hybrid_bytes_and_hash_interpretation_are_unchanged() {
    for (fixture, keys) in [
        (
            include_bytes!("fixtures/hybrid_v081.bin").as_slice(),
            vec![
                b"a".to_vec(),
                b"bc".to_vec(),
                b"hello".to_vec(),
                b"12345678".to_vec(),
                b"long-serialized-key".to_vec(),
            ],
        ),
        (
            include_bytes!("fixtures/hybrid_absolute_pgm_v081.bin").as_slice(),
            (0..256u64).map(|i| (i * 13 + 7).to_le_bytes().to_vec()).collect(),
        ),
    ] {
        let index = HybridIndex::from_bytes(fixture).unwrap();
        assert_eq!(index.to_bytes(), fixture);
        for (position, key) in keys.iter().enumerate() {
            assert_eq!(index.lookup(key), Some(position as u32));
        }
        assert_eq!(
            index.lookup_batch(&keys),
            (0..keys.len()).map(|position| Some(position as u32)).collect::<Vec<_>>()
        );
    }
}

#[test]
fn empty_and_tiny_hybrid_queries_roundtrip_without_routing_state() {
    for keys in [vec![], vec![u64::MAX], vec![7, 0, u64::MAX]] {
        let index = HybridBuilder::new().with_parallel(false).build_from_u64(&keys).unwrap();
        let restored = HybridIndex::from_bytes(&index.to_bytes()).unwrap();
        check_numeric_queries(&index, &keys);
        check_numeric_queries(&restored, &keys);
        if keys.is_empty() {
            assert_eq!(restored.lookup_hash(0), None);
            assert_eq!(restored.lookup_hash(u64::MAX), None);
        }
    }
}
