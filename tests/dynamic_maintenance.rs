use kira_kv_engine::{CompactionPolicy, DynamicConfig, DynamicIndex, DynamicInsertError};

fn config(batch: usize, tiers: usize) -> DynamicConfig {
    DynamicConfig {
        flush_threshold: batch,
        max_tiers: tiers,
        lean_tiers: true,
        parallel_build: false,
    }
}

fn exhausted_index() -> DynamicIndex {
    // Existing wire layout: config, next ID, live count, then key/ID pairs.
    let mut bytes = vec![4];
    bytes.extend_from_slice(b"KIRA");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&[4, 0]);
    bytes.extend_from_slice(&8u64.to_le_bytes());
    bytes.extend_from_slice(&8u64.to_le_bytes());
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&5u32.to_le_bytes());
    bytes.extend_from_slice(b"alive");
    bytes.extend_from_slice(&(u32::MAX - 1).to_le_bytes());
    let state = wyhash::wyhash(&bytes, 0x4B49_5241_5F43_4B53);
    let sum = wyhash::wyhash(&state.to_le_bytes(), state);
    bytes.extend_from_slice(&sum.to_le_bytes());
    DynamicIndex::from_bytes(&bytes).unwrap()
}

#[test]
fn exhausted_ids_reject_new_and_revived_keys_before_mutation() {
    let mut index = exhausted_index();
    assert_eq!(index.try_insert(b"alive".to_vec()).unwrap(), u32::MAX - 1);
    assert_eq!(index.buffer_len(), 0);
    assert!(matches!(index.try_insert(b"other".to_vec()), Err(DynamicInsertError::IdExhausted)));
    assert_eq!(index.lookup(b"alive"), Some(u32::MAX - 1));
    assert_eq!(index.delete(b"alive"), Some(u32::MAX - 1));
    let before = index.to_bytes();
    assert!(matches!(index.try_insert(b"alive".to_vec()), Err(DynamicInsertError::IdExhausted)));
    assert_eq!(index.to_bytes(), before);
    assert_eq!(index.lookup(b"alive"), None);
    assert_eq!(index.len(), 0);
    assert_eq!(index.tombstone_count(), 1);
    assert_eq!(index.buffer_len(), 0);
}

#[test]
fn inserting_an_existing_tier_key_does_not_promote_or_rebuild_it() {
    let mut index = DynamicIndex::with_config(config(2, 8));
    let first = index.insert(b"first".to_vec());
    index.insert(b"second".to_vec());
    assert_eq!(index.buffer_len(), 0);
    let stats = index.compaction_stats();
    for _ in 0..32 {
        assert_eq!(index.try_insert(b"first".to_vec()).unwrap(), first);
        assert_eq!(index.insert(b"first".to_vec()), first);
    }
    assert_eq!(index.buffer_len(), 0);
    assert_eq!(index.len(), 2);
    assert_eq!(index.compaction_stats(), stats);
}

fn assert_model(index: &DynamicIndex, count: usize) {
    assert_eq!(index.len(), count);
    for key in 0..count as u64 {
        assert_eq!(index.lookup(&key.to_le_bytes()), Some(key as u32));
    }
}

#[test]
fn geometric_rebuilding_has_a_logarithmic_work_bound_and_preserves_ids() {
    let mut index = DynamicIndex::with_config(config(8, 1));
    index.set_compaction_policy(CompactionPolicy::Geometric).unwrap();
    let count = 4096;
    for key in 0..count as u64 {
        assert_eq!(index.insert(key.to_le_bytes().to_vec()), key as u32);
    }
    index.flush().unwrap();
    index.finish_background_compaction().unwrap();
    assert_model(&index, count);
    assert!(index.tier_count() <= index.tier_limit());
    assert!(index.compaction_stats().0 > 0);
    assert!(index.compaction_stats().1 <= count as u64 * 5);
    let restored = DynamicIndex::from_bytes(&index.to_bytes()).unwrap();
    assert_model(&restored, count);
    assert_eq!(restored.compaction_policy(), CompactionPolicy::Full);
}

#[test]
fn geometric_partial_flushes_promote_runs_even_below_batch_size() {
    let mut index = DynamicIndex::with_config(config(16_384, 1));
    index.set_compaction_policy(CompactionPolicy::Geometric).unwrap();
    let count = 256;
    for key in 0..count as u64 {
        index.insert(key.to_le_bytes().to_vec());
        index.flush().unwrap();
    }
    index.finish_background_compaction().unwrap();
    assert_model(&index, count);
    assert!(index.tier_count() <= index.tier_limit());
    assert!(index.compaction_stats().1 <= count as u64 * 4);
}

#[test]
fn changing_from_geometric_enforces_the_selected_legacy_tier_cap() {
    for policy in [CompactionPolicy::Full, CompactionPolicy::SizeTiered] {
        let mut index = DynamicIndex::with_config(config(1, 1));
        index.set_compaction_policy(CompactionPolicy::Geometric).unwrap();
        for key in 0..15u64 {
            index.insert(key.to_le_bytes().to_vec());
        }
        index.finish_background_compaction().unwrap();
        assert!(index.tier_count() > 1);
        index.set_compaction_policy(policy).unwrap();
        assert_eq!(index.compaction_policy(), policy);
        assert!(index.tier_count() <= 1);
        assert_model(&index, 15);
    }
}

#[test]
fn deletion_density_reclaims_dead_bytes_and_tombstone_capacity() {
    for background in [false, true] {
        for policy in [CompactionPolicy::SizeTiered, CompactionPolicy::Geometric] {
            let mut index = DynamicIndex::with_config(config(16, 4));
            index.set_compaction_policy(policy).unwrap();
            index.set_background_compaction(background).unwrap();
            let keys: Vec<_> = (0..256u64)
                .map(|number| {
                    let mut key = vec![b'x'; 512];
                    key[..8].copy_from_slice(&number.to_le_bytes());
                    key
                })
                .collect();
            for (id, key) in keys.iter().enumerate() {
                assert_eq!(index.insert(key.clone()), id as u32);
            }
            index.flush().unwrap();
            index.finish_background_compaction().unwrap();
            let before = index.memory_usage();
            let before_stats = index.compaction_stats();
            for key in &keys[..224] {
                assert!(index.delete(key).is_some());
            }
            assert_eq!(index.tombstone_count(), 224);
            index.flush().unwrap();
            index.finish_background_compaction().unwrap();
            assert_eq!(index.len(), 32);
            assert_eq!(index.tombstone_count(), 0);
            assert!(index.memory_usage() < before / 2);
            assert!(index.compaction_stats().0 > before_stats.0);
            assert!(index.compaction_stats().1 > before_stats.1);
            for (id, key) in keys.iter().enumerate() {
                assert_eq!(index.lookup(key), if id < 224 { None } else { Some(id as u32) });
            }
            let restored = DynamicIndex::from_bytes(&index.to_bytes()).unwrap();
            assert_eq!(restored.len(), 32);
            for (id, key) in keys.iter().enumerate() {
                assert_eq!(restored.lookup(key), if id < 224 { None } else { Some(id as u32) });
            }
        }
    }
}

#[test]
fn background_collection_preserves_deletion_and_revival_before_publication() {
    let mut index = DynamicIndex::with_config(config(1, 4));
    index.set_compaction_policy(CompactionPolicy::Geometric).unwrap();
    index.set_background_compaction(true).unwrap();
    for key in 0..4u64 {
        index.insert(key.to_le_bytes().to_vec());
    }
    let deleted = 0u64.to_le_bytes();
    let revived = 1u64.to_le_bytes();
    assert_eq!(index.delete(&deleted), Some(0));
    assert_eq!(index.delete(&revived), Some(1));
    let new_id = index.try_insert(revived.to_vec()).unwrap();
    assert_ne!(new_id, 1);
    index.flush().unwrap();
    index.finish_background_compaction().unwrap();
    assert_eq!(index.lookup(&deleted), None);
    assert_eq!(index.lookup(&revived), Some(new_id));
    index.compact().unwrap();
    assert_eq!(index.lookup(&deleted), None);
    assert_eq!(index.lookup(&revived), Some(new_id));
    assert_eq!(index.tombstone_count(), 0);
}

#[test]
fn frozen_dynamic_format_preserves_ids_deletions_and_next_id() {
    let mut index = DynamicIndex::from_bytes(include_bytes!("fixtures/dynamic_v081.bin")).unwrap();
    assert_eq!(index.compaction_policy(), CompactionPolicy::Full);
    assert_eq!(index.len(), 2);
    assert_eq!(index.lookup(b"alice"), Some(0));
    assert_eq!(index.lookup(b"bob"), None);
    assert_eq!(index.lookup(b"carol"), Some(2));
    assert_eq!(index.try_insert(b"dave".to_vec()).unwrap(), 3);
    index.flush().unwrap();
    index.compact().unwrap();
    let restored = DynamicIndex::from_bytes(&index.to_bytes()).unwrap();
    assert_eq!(restored.len(), 3);
    assert_eq!(restored.lookup(b"alice"), Some(0));
    assert_eq!(restored.lookup(b"bob"), None);
    assert_eq!(restored.lookup(b"carol"), Some(2));
    assert_eq!(restored.lookup(b"dave"), Some(3));
}
