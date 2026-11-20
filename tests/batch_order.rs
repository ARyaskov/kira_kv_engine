use kira_kv_engine::{FingerprintBits, IndexBuilder};

#[test]
fn alternating_negative_and_positive_blocks_preserve_positions() {
    let keys: Vec<_> = (0..512u64).map(u64::to_le_bytes).collect();
    let queries: Vec<_> = (0..256u64).map(|k| if k / 64 % 2 == 0 { k + 1000 } else { k }).collect();
    let bytes: Vec<_> = queries.iter().map(|k| k.to_le_bytes()).collect();
    let refs: Vec<_> = bytes.iter().map(|k| k.as_slice()).collect();
    for bloom in [false, true] {
        let index = IndexBuilder::new()
            .with_bloom_filter(bloom)
            .with_fingerprint_bits(FingerprintBits::Bits24)
            .build_index_ref(&keys)
            .unwrap();
        let expected: Vec<_> = queries.iter().map(|&k| index.lookup_u64(k).ok()).collect();
        assert_eq!(index.lookup_batch_u64_simd(&queries), expected);
        assert_eq!(index.lookup_batch_pipelined(&refs), expected);
    }
}
