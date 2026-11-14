use kira_kv_engine::__internal::BlockBloom;

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[test]
fn small_filters_use_only_the_required_cache_lines() {
    for count in [0usize, 1, 46, 47, 128, 257, 1000, 4096, 10_000] {
        let hashes: Vec<_> = (0..count as u64).map(mix).collect();
        let filter = BlockBloom::build_from_prehashed(&hashes);
        let expected_blocks = (count.max(1) * 11).div_ceil(512);
        assert_eq!(filter.export_words().len(), expected_blocks * 8, "count={count}");
        assert!(filter.export_words().len() < 256 * 8, "exercise compact geometry");
        for &hash in &hashes {
            assert!(filter.contains_hash(hash), "count={count} hash={hash}");
        }
    }
}

#[test]
fn large_filter_geometry_keeps_the_existing_partition_rounding() {
    for count in [11_915usize, 11_916, 12_000, 50_000, 262_144] {
        let hashes: Vec<_> = (0..count as u64).map(mix).collect();
        let filter = BlockBloom::build_from_prehashed(&hashes);
        let expected_blocks = ((count * 11) / 512).max(1).next_multiple_of(256);
        assert_eq!(filter.export_words().len(), expected_blocks * 8, "count={count}");
        assert!(filter.export_words().len().is_multiple_of(256 * 8));
        for &hash in &hashes {
            assert!(filter.contains_hash(hash));
        }
    }
}

#[test]
fn compact_filters_keep_wire_layout_seed_and_cursor_continuation() {
    let keys: Vec<_> = (0..257u64).map(|key| key * 17 + 11).collect();
    let filter = BlockBloom::build_from_u64(&keys, 12345);
    let mut bytes = vec![7, 11];
    filter.write_to(&mut bytes);
    let end = bytes.len();
    bytes.extend_from_slice(b"next item");
    assert_eq!(end, 2 + 16 + filter.export_words().len() * 8);
    let raw_len = u64::from_le_bytes(bytes[10..18].try_into().unwrap());
    assert_ne!(raw_len & (1 << 63), 0);
    assert_eq!(raw_len & !(1 << 63), filter.export_words().len() as u64);
    let mut cursor = 2;
    let restored = BlockBloom::read_from(&bytes, &mut cursor).unwrap();
    assert_eq!(cursor, end);
    assert_eq!(restored.seed(), 12345);
    assert_eq!(restored.bit_shift(), 26);
    for &key in &keys {
        assert!(restored.contains_u64(key));
    }
    let mut rewritten = vec![7, 11];
    restored.write_to(&mut rewritten);
    assert_eq!(rewritten, bytes[..end]);
    let mut streamed = Vec::new();
    restored.write_into(&mut streamed).unwrap();
    assert_eq!(streamed, bytes[2..end]);
}

#[test]
fn compact_prehashed_scratch_remains_a_permutation_without_false_negatives() {
    for count in [0usize, 1, 257, 4096] {
        let hashes: Vec<_> = (0..count as u64).map(mix).collect();
        let mut scratch = vec![0; count];
        let filter = BlockBloom::build_from_prehashed_into(&hashes, &mut scratch);
        let mut expected = hashes.clone();
        expected.sort_unstable();
        scratch.sort_unstable();
        assert_eq!(scratch, expected);
        for &hash in &hashes {
            assert!(filter.contains_hash(hash));
        }
    }
}

#[test]
fn compact_filters_keep_a_bounded_false_positive_rate() {
    const QUERIES: u64 = 100_000;
    for count in [1usize, 46, 47, 128, 257, 1000, 4096, 10_000] {
        let hashes: Vec<_> = (0..count as u64).map(mix).collect();
        let filter = BlockBloom::build_from_prehashed(&hashes);
        let false_positives = (0..QUERIES)
            .map(|rank| mix(rank + 1_000_000_000))
            .filter(|&hash| filter.contains_hash(hash))
            .count();
        assert!(
            false_positives < QUERIES as usize / 50,
            "count={count}, false positives={false_positives}/{QUERIES}"
        );
    }
}

#[test]
fn bloom_reader_rejects_overflowing_initial_cursors_without_panicking() {
    let filter = BlockBloom::build_from_prehashed(&[7]);
    let mut bytes = Vec::new();
    filter.write_to(&mut bytes);
    for initial in [usize::MAX, usize::MAX - 15, bytes.len() + 1] {
        let mut cursor = initial;
        assert!(BlockBloom::read_from(&bytes, &mut cursor).is_none());
        assert_eq!(cursor, initial);
    }
}
