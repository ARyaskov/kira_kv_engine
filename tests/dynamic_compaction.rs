use kira_kv_engine::{CompactionPolicy, DynamicConfig, DynamicIndex};
use std::collections::HashMap;

#[test]
fn default_policy_preserves_full_merge_boundaries() {
    let mut index = DynamicIndex::with_config(DynamicConfig {
        flush_threshold: 8,
        max_tiers: 8,
        lean_tiers: false,
        parallel_build: false,
    });
    for i in 0..32u64 {
        index.insert(i.to_le_bytes().to_vec());
    }
    index.flush().unwrap();
    index.finish_background_compaction().unwrap();
    assert_eq!(index.tier_count(), 4);
    assert_eq!(index.compaction_stats(), (0, 0));
    for i in 32..72u64 {
        index.insert(i.to_le_bytes().to_vec());
    }
    index.flush().unwrap();
    index.finish_background_compaction().unwrap();
    assert_eq!(index.tier_count(), 1);
    assert_eq!(index.compaction_stats(), (1, 72));
    for i in 0..72u64 {
        assert_eq!(index.lookup(&i.to_le_bytes()), Some(i as u32));
    }
}

#[test]
fn tiered_merges_preserve_shadowing_deletion_and_revival() {
    for background in [false, true] {
        for lean in [false, true] {
            for policy in [CompactionPolicy::SizeTiered, CompactionPolicy::Geometric] {
                check_tiered_merges(background, lean, policy);
            }
        }
    }
}

fn check_tiered_merges(background: bool, lean: bool, policy: CompactionPolicy) {
    let mut idx = DynamicIndex::with_config(DynamicConfig {
        flush_threshold: 32,
        max_tiers: 6,
        lean_tiers: lean,
        parallel_build: false,
    });
    idx.set_background_compaction(background).unwrap();
    idx.set_compaction_ratio(3).unwrap();
    idx.set_compaction_policy(policy).unwrap();
    assert_eq!(idx.compaction_policy(), policy);
    let mut model = HashMap::new();
    let mut deleted = HashMap::new();
    let mut deletes = 0;
    let mut revivals = 0;
    let mut next_id = 0u32;
    let mut random = 42u64;
    for operation in 0..12000 {
        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
        let key = format!("key-{}", random % 1500).into_bytes();
        // Select the operation from different bits than the key.
        // Otherwise every fifth key would only ever be deleted.
        if (random >> 32).is_multiple_of(5) {
            let expected = model.remove(&key);
            assert_eq!(idx.delete(&key), expected);
            if let Some(id) = expected {
                deletes += 1;
                deleted.insert(key.clone(), id);
            }
        } else {
            let id = *model.entry(key.clone()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            assert_eq!(idx.insert(key.clone()), id);
            if let Some(old_id) = deleted.remove(&key) {
                revivals += 1;
                assert!(id > old_id);
            }
        }
        assert_eq!(idx.lookup(&key), model.get(&key).copied());
        assert_eq!(idx.len(), model.len());
        if operation % 257 == 0 {
            idx.flush().unwrap();
            assert!(idx.tier_count() <= idx.tier_limit());
            if policy == CompactionPolicy::SizeTiered {
                assert_eq!(idx.tier_limit(), 6);
            }
            for (key, &id) in &model {
                assert_eq!(idx.lookup(key), Some(id));
            }
            // Serialization must remain consistent even with a worker in flight.
            let restored = DynamicIndex::from_bytes(&idx.to_bytes()).unwrap();
            assert_eq!(restored.len(), model.len());
            for (key, &id) in &model {
                assert_eq!(restored.lookup(key), Some(id));
            }
        }
    }
    assert!(deletes > 0 && revivals > 0);
    idx.finish_background_compaction().unwrap();
    assert!(idx.compaction_stats().0 > 0);
    idx.compact().unwrap();
    assert_eq!(idx.tier_count(), 1);
    assert_eq!(idx.tombstone_count(), 0);
    for i in 0..1500 {
        let key = format!("key-{i}").into_bytes();
        assert_eq!(idx.lookup(&key), model.get(&key).copied());
    }
}

#[test]
fn arena_handles_empty_and_variable_length_keys_and_index_stays_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DynamicIndex>();
    let mut idx = DynamicIndex::with_config(DynamicConfig {
        flush_threshold: 8,
        max_tiers: 4,
        ..DynamicConfig::default()
    });
    let keys: Vec<Vec<u8>> = (0..400u32)
        .map(|i| {
            let mut key = vec![0; i as usize * 11];
            key.extend_from_slice(&i.to_le_bytes());
            key
        })
        .chain(std::iter::once(Vec::new()))
        .collect();
    let ids: Vec<_> = keys.iter().map(|key| idx.insert(key.clone())).collect();
    idx.flush().unwrap();
    idx.finish_background_compaction().unwrap();
    idx.compact().unwrap();
    for (key, &id) in keys.iter().zip(&ids) {
        assert_eq!(idx.lookup(key), Some(id));
    }
    assert!(idx.set_compaction_ratio(1).is_err());
}
