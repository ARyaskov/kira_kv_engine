use kira_kv_engine::IndexBuilder;

#[test]
fn pipeline_matches_scalar_for_mixed_queries_tails_and_reused_outputs() {
    let keys: Vec<[u8; 8]> = (0..70_000u64).map(u64::to_le_bytes).collect();
    for lean in [false, true] {
        let idx = IndexBuilder::new().with_lean_mph(lean).build_index_ref(&keys).unwrap();
        for n in [0, 1, 7, 31, 63, 64, 65, 127, 128, 129, 999] {
            let queries: Vec<u64> =
                (0..n).map(|i| if i % 3 == 0 { 100_000 + i as u64 } else { i as u64 }).collect();
            let expected: Vec<_> = queries.iter().map(|&k| idx.lookup_u64(k).ok()).collect();
            let mut output = vec![Some(usize::MAX); n + 3];
            let mut scratch = vec![0; n + 3];
            idx.lookup_batch_u64_simd_into(&queries, &mut scratch, &mut output);
            assert_eq!(&output[..n], expected.as_slice());
            assert_eq!(&output[n..], &[Some(usize::MAX); 3]);
            let bytes: Vec<_> = queries.iter().map(|k| k.to_le_bytes()).collect();
            let refs: Vec<&[u8]> = bytes.iter().map(|b| b.as_slice()).collect();
            assert_eq!(idx.lookup_batch_pipelined(&refs), expected);
        }
    }
}
