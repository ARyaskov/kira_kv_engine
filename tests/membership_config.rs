use kira_kv_engine::{FingerprintBits, Index, IndexBuilder};

#[test]
fn all_membership_modes_agree_on_hits_batches_and_roundtrips() {
    let keys: Vec<[u8; 8]> = (0..80_000u64).map(u64::to_le_bytes).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mapped");
    for bloom in [false, true] {
        for bits in [FingerprintBits::Bits16, FingerprintBits::Bits24, FingerprintBits::Bits32] {
            let idx = IndexBuilder::new()
                .with_bloom_filter(bloom)
                .with_fingerprint_bits(bits)
                .build_index_ref(&keys)
                .unwrap();
            let restored = Index::from_bytes(&idx.to_bytes().unwrap()).unwrap();
            idx.save_mmap(&path).unwrap();
            let mapped = Index::open_mmap(&path).unwrap();
            let queries: Vec<u64> = (0..90_000).collect();
            let batch = idx.lookup_batch_u64_simd(&queries);
            for (i, &key) in queries.iter().enumerate() {
                let expected = idx.lookup_u64(key).ok();
                assert_eq!(batch[i], expected);
                assert_eq!(restored.lookup_u64(key).ok(), expected);
                assert_eq!(mapped.lookup_u64(key).ok(), expected);
                if key < 80_000 {
                    assert!(expected.is_some());
                }
            }
            assert!(idx.supports_negative_lookups());
            let export = idx.gpu_export().unwrap();
            assert_eq!(export.fingerprint_bits, bits.bits());
            assert_eq!(
                export.fingerprint_bytes.unwrap().len(),
                keys.len() * bits.bits() as usize / 8
            );
            drop(mapped);
        }
    }
}

#[test]
fn packed_fingerprints_match_canonical_hashes_in_each_final_slot() {
    use kira_kv_engine::__internal::simd_hash::hash_u64_scalar;
    use kira_kv_engine::MphConfig;
    let keys: Vec<_> = (0..513u64).map(u64::to_le_bytes).collect();
    let input: Vec<_> = (0..513u64).collect();
    for lambda in [3.0, 10000.0] {
        let config = MphConfig { lambda, max_rehash: 0, ..MphConfig::default() };
        let mut canonical = vec![0; input.len()];
        hash_u64_scalar(&input, config.seed, &mut canonical);
        for bits in [FingerprintBits::Bits24, FingerprintBits::Bits32] {
            let stride = bits.bits() as usize / 8;
            let index = IndexBuilder::new()
                .with_mph_config(config.clone())
                .with_fingerprint_bits(bits)
                .build_index_ref(&keys)
                .unwrap();
            let mut expected = vec![0u8; keys.len() * stride];
            for (key, hash) in keys.iter().zip(&canonical) {
                let slot = index.lookup(key).unwrap();
                expected[slot * stride..(slot + 1) * stride]
                    .copy_from_slice(&hash.to_le_bytes()[..stride]);
            }
            // The ordinary format ends with the packed table and its outer checksum.
            let payload = index.to_bytes().unwrap();
            assert_eq!(&payload[payload.len() - 8 - expected.len()..payload.len() - 8], &expected);
            assert_eq!(index.gpu_export().is_some(), lambda == 3.0);
        }
    }
}
