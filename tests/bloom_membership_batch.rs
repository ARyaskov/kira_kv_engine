use kira_kv_engine::IndexBuilder;

#[test]
fn batch_matches_scalar_across_blocks_and_tails() {
    let keys: Vec<_> = (0..300).map(|i| format!("membership-{i}")).collect();
    let queries: Vec<_> = keys.iter().map(|s| s.as_bytes()).collect();
    for bloom in [false, true] {
        let index =
            IndexBuilder::new().with_bloom_filter(bloom).build_index_ref(&keys[..130]).unwrap();
        for n in [0, 7, 8, 63, 64, 65, 300] {
            assert_eq!(
                index.contains_batch(&queries[..n]),
                queries[..n].iter().map(|k| index.contains(k)).collect::<Vec<_>>()
            );
        }
    }
}
