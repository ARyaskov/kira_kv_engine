#[path = "../examples/support/validation.rs"]
mod validation;

use kira_kv_engine::IndexBuilder;
use validation::{slot_oracle, verify_index, verify_results};

#[test]
fn oracle_rejects_missing_duplicate_and_out_of_range_static_slots() {
    let keys = [b"first".as_slice(), b"second"];
    assert!(slot_oracle(&keys, 2, |_| None).is_err());
    assert!(slot_oracle(&keys, 2, |_| Some(0)).is_err());
    assert!(slot_oracle(&keys, 2, |_| Some(2)).is_err());
    assert!(slot_oracle(&keys, 1, |_| Some(0)).is_err());
    assert!(slot_oracle(&keys, 2, |key| Some(usize::from(key == b"first"))).is_ok());
}

#[test]
fn oracle_rejects_batch_false_negatives_wrong_keys_and_wrong_lengths() {
    let keys = [b"first".as_slice(), b"second"];
    let oracle = slot_oracle(&keys, 2, |key| Some(usize::from(key == b"second"))).unwrap();
    let queries = [(keys[0], true), (keys[1], true)];
    assert!(verify_results(&oracle, &keys, queries, &[None, Some(1)]).is_err());
    assert!(verify_results(&oracle, &keys, queries, &[Some(1), Some(0)]).is_err());
    assert!(verify_results(&oracle, &keys, queries, &[Some(0), Some(2)]).is_err());
    assert!(verify_results(&oracle, &keys, queries, &[Some(0)]).is_err());
    assert!(verify_results(&oracle, &keys, queries, &[Some(0), Some(1), None]).is_err());
}

#[test]
fn benchmark_oracle_counts_foreign_candidates_without_claiming_exact_membership() {
    let keys = [b"first".as_slice(), b"second"];
    for lean in [false, true] {
        let index = IndexBuilder::new()
            .with_parallel_build(false)
            .with_lean_mph(lean)
            .build_index_ref(&keys)
            .unwrap();
        let oracle = verify_index(&index, &keys).unwrap();
        let queries = [keys[0], b"absent", keys[1]];
        let results = index.lookup_batch_pipelined(&queries);
        let stats =
            verify_results(&oracle, &keys, queries.into_iter().zip([true, false, true]), &results)
                .unwrap();
        assert_eq!(stats.queries, 3);
        assert_eq!(stats.known_hits, 2);
        assert_eq!(stats.foreign_queries, 1);
        if lean {
            assert_eq!(stats.foreign_candidates, 1);
        }
    }
}

#[test]
fn parallel_byte_batches_are_verified_against_the_original_keys() {
    let keys: Vec<_> = (0..10_000u64).map(u64::to_le_bytes).collect();
    for lean in [false, true] {
        let index = IndexBuilder::new()
            .with_parallel_build(false)
            .with_lean_mph(lean)
            .build_index_ref(&keys)
            .unwrap();
        let oracle = verify_index(&index, &keys).unwrap();
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let index = &index;
                let keys = &keys;
                let oracle = &oracle;
                scope.spawn(move || {
                    let queries: Vec<_> = (0..5089u64)
                        .map(|i| if i % 3 == 0 { 100_000 + i } else { (i + worker * 17) % 10_000 })
                        .map(u64::to_le_bytes)
                        .collect();
                    let refs: Vec<_> = queries.iter().map(|key| key.as_slice()).collect();
                    let results = index.lookup_batch_pipelined(&refs);
                    let expected = refs.iter().enumerate().map(|(i, &key)| (key, i % 3 != 0));
                    let stats = verify_results(oracle, keys, expected, &results).unwrap();
                    assert_eq!(stats.queries, queries.len());
                    assert!(stats.known_hits > 0 && stats.foreign_queries > 0);
                });
            }
        });
    }
}
