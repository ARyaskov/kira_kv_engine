use kira_kv_engine::__internal::BlockBloom;
use kira_kv_engine::{PgmBuilder, PgmIndex};

fn assert_bounds(index: &PgmIndex, keys: &[u64], queries: &[u64]) {
    for &query in queries {
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
    for &start in queries {
        for &end in queries {
            let expected_start = keys.partition_point(|&key| key < start);
            let expected_end = keys.partition_point(|&key| key <= end).max(expected_start);
            assert_eq!(index.range(start, end), expected_start..expected_end);
        }
    }
}

#[test]
fn bounds_in_inter_segment_gaps_match_sorted_array() {
    let keys = vec![0, 1, 2, 1000, 1001];
    let index = PgmBuilder::new().with_epsilon(0).with_parallel(false).build(keys.clone()).unwrap();
    assert_eq!(index.lower_bound(500), 3);
    assert_bounds(&index, &keys, &[0, 1, 2, 3, 499, 500, 999, 1000, 1001, 1002, u64::MAX]);
}

#[test]
fn elias_fano_bounds_for_absent_keys_match_sorted_array() {
    let keys: Vec<u64> = (0..1024).map(|i| i * 2).collect();
    let mut index =
        PgmBuilder::new().with_epsilon(0).with_parallel(false).build(keys.clone()).unwrap();
    assert!(index.compact_keys() > 0);
    assert_eq!(index.lower_bound(1), 1);
    assert_bounds(&index, &keys, &[0, 1, 2, 3, 31, 255, 1023, 2045, 2046, 2047, u64::MAX]);
}

#[test]
fn upper_bound_includes_maximum_key() {
    let keys = vec![0, 1, u64::MAX - 1, u64::MAX];
    let index = PgmBuilder::new().with_epsilon(0).with_parallel(false).build(keys.clone()).unwrap();
    assert_eq!(index.upper_bound(u64::MAX), keys.len());
    assert_eq!(index.range(u64::MAX, u64::MAX), 3..4);
    assert_bounds(&index, &keys, &[0, 1, 2, u64::MAX - 2, u64::MAX - 1, u64::MAX]);
}

#[test]
fn bounds_with_nonuniform_spacing_match_sorted_array() {
    let keys = vec![7, 10, 100, 101, 102, 1000, 10_000, 10_001, 1_000_000];
    let queries: Vec<u64> = keys
        .iter()
        .flat_map(|&key| [key - 1, key, key + 1])
        .chain([0, 50, 500, 5000, u64::MAX])
        .collect();
    for epsilon in [0, 1, 4, 64] {
        let index = PgmBuilder::new()
            .with_epsilon(epsilon)
            .with_parallel(false)
            .build(keys.clone())
            .unwrap();
        assert_bounds(&index, &keys, &queries);
    }
}

fn legacy_pgm(version: u8) -> Vec<u8> {
    // A single line y = 0.1x - 1 over the independently specified keys 10, 20, 30.
    let mut bytes = vec![version];
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&3u64.to_le_bytes());
    if version >= 3 {
        bytes.extend_from_slice(&3u64.to_le_bytes());
    }
    for key in [10u64, 20, 30] {
        bytes.extend_from_slice(&key.to_le_bytes());
    }
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0.1f32.to_le_bytes());
    bytes.extend_from_slice(&(-1.0f32).to_le_bytes());
    bytes.extend_from_slice(&10u64.to_le_bytes());
    bytes.extend_from_slice(&30u64.to_le_bytes());
    bytes.push(1);
    bytes.extend_from_slice(&0u64.to_le_bytes());
    if version < 4 {
        bytes.extend_from_slice(&0u64.to_le_bytes());
    }
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.push(0);
    bytes
}

#[test]
fn standalone_pgm_reads_all_legacy_layouts() {
    for version in [2, 3, 4] {
        let bytes = legacy_pgm(version);
        let index = PgmIndex::from_bytes(&bytes).unwrap();
        for (position, key) in [10u64, 20, 30].into_iter().enumerate() {
            assert_eq!(index.index(key).ok(), Some(position));
        }
        assert_bounds(&index, &[10, 20, 30], &[0, 9, 10, 11, 19, 20, 21, 30, 31, u64::MAX]);
    }
    // Raw v4 starts with the same tag as the other engines' common container.
    // Even its epsilon may spell the magic; standalone PGM uses a distinct tag.
    let mut bytes = legacy_pgm(4);
    bytes[1..5].copy_from_slice(b"KIRA");
    assert_eq!(PgmIndex::from_bytes(&bytes).unwrap().index(20).ok(), Some(1));
}

#[test]
fn standalone_pgm_serialization_is_checksummed() {
    let index = PgmBuilder::new().build(vec![10, 20, 30]).unwrap();
    let bytes = index.to_bytes().unwrap();
    assert_eq!(bytes[0], 5);
    assert_eq!(&bytes[1..5], b"KIRA");
    for offset in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[offset] ^= 0x80;
        assert!(PgmIndex::from_bytes(&changed).is_err(), "accepted mutation at {offset}");
    }
    for length in 0..bytes.len() {
        assert!(PgmIndex::from_bytes(&bytes[..length]).is_err(), "accepted truncation to {length}");
    }
}

#[test]
fn standalone_pgm_rejects_trailing_bytes_in_new_and_legacy_formats() {
    let index = PgmBuilder::new().build(vec![10, 20, 30]).unwrap();
    let mut variants = vec![index.to_bytes().unwrap()];
    variants.extend([2, 3, 4].map(legacy_pgm));
    for mut bytes in variants {
        bytes.push(0);
        assert!(PgmIndex::from_bytes(&bytes).is_err());
    }
}

#[test]
fn legacy_pgm_rejects_inconsistent_key_bounds_and_model_error() {
    let bytes = legacy_pgm(4);
    // v4 fixed header (21 bytes), three keys (24 bytes), segment count (8 bytes).
    let slope_offset = 21 + 24 + 8;
    let min_key_offset = slope_offset + 8;
    let mut changed = bytes.clone();
    changed[min_key_offset..min_key_offset + 8].copy_from_slice(&9u64.to_le_bytes());
    assert!(PgmIndex::from_bytes(&changed).is_err());
    let mut changed = bytes;
    changed[slope_offset..slope_offset + 4].copy_from_slice(&2.0f32.to_le_bytes());
    assert!(PgmIndex::from_bytes(&changed).is_err());
}

#[test]
fn legacy_pgm_rejects_incomplete_and_unused_overflow_entries() {
    let bytes = legacy_pgm(4);
    let error_offset = 21 + 24 + 8 + 4 + 4 + 8 + 8;
    let overflow_count_offset = error_offset + 1;
    let mut missing = bytes.clone();
    missing[error_offset] = 0xFF;
    assert!(PgmIndex::from_bytes(&missing).is_err());

    let mut unused = bytes;
    unused[overflow_count_offset..overflow_count_offset + 8].copy_from_slice(&1u64.to_le_bytes());
    let entry: Vec<u8> = 0u32.to_le_bytes().into_iter().chain(255u32.to_le_bytes()).collect();
    unused.splice(overflow_count_offset + 8..overflow_count_offset + 8, entry);
    assert!(PgmIndex::from_bytes(&unused).is_err());
}

#[test]
fn legacy_pgm_rejects_a_structurally_valid_bloom_with_false_negatives() {
    let mut bytes = legacy_pgm(4);
    *bytes.last_mut().unwrap() = 1;
    let bloom_offset = bytes.len();
    BlockBloom::build_from_u64(&[10, 20, 30], 42).write_to(&mut bytes);
    assert!(PgmIndex::from_bytes(&bytes).is_ok());

    // Leave the Bloom seed, length and format flag intact; clear only its words.
    bytes[bloom_offset + 16..].fill(0);
    let mut pos = bloom_offset;
    let bloom = BlockBloom::read_from(&bytes, &mut pos).expect("valid Bloom structure");
    assert_eq!(pos, bytes.len());
    assert!(!bloom.contains_u64(10));
    assert!(PgmIndex::from_bytes(&bytes).is_err());
}

#[test]
fn raw_v6_and_checksummed_pgm_load_multiple_segments_without_bloom() {
    for keys in [
        vec![0, 1, 2, 1000, 1001],
        vec![0, 1, 2, 1000, 1001, 1002, 1_000_000, 1_000_001, 1_000_002],
    ] {
        let index = PgmBuilder::new()
            .with_epsilon(0)
            .with_parallel(false)
            .with_bloom_filter(false)
            .build(keys.clone())
            .unwrap();
        assert!(!index.has_bloom());
        assert!(index.num_segments() >= if keys.len() == 5 { 2 } else { 3 });
        let sealed = index.to_bytes().unwrap();
        // Standalone header is nine bytes and its checksum is eight bytes.
        let raw_v6 = sealed[9..sealed.len() - 8].to_vec();
        assert_eq!(raw_v6[0], 6);
        for bytes in [sealed, raw_v6] {
            let restored = PgmIndex::from_bytes(&bytes).unwrap();
            for (position, &key) in keys.iter().enumerate() {
                assert_eq!(restored.index(key).ok(), Some(position));
            }
            assert_bounds(&restored, &keys, &[0, 2, 3, 500, 999, 1000, 1001, 1002, u64::MAX]);
        }
    }
}
