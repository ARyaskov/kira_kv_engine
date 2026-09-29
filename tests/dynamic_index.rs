use kira_kv_engine::{DynamicConfig, DynamicIndex};

fn k(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

#[test]
fn insert_lookup_basic() {
    let mut idx = DynamicIndex::new();
    let id_a = idx.insert(k("alpha"));
    let id_b = idx.insert(k("beta"));
    assert_ne!(id_a, id_b);
    assert_eq!(idx.lookup(b"alpha"), Some(id_a));
    assert_eq!(idx.lookup(b"beta"), Some(id_b));
    assert_eq!(idx.lookup(b"gamma"), None);
}

#[test]
fn insert_reuses_id_for_duplicate() {
    let mut idx = DynamicIndex::new();
    let id = idx.insert(k("alpha"));
    assert_eq!(idx.insert(k("alpha")), id);
    assert_eq!(idx.lookup(b"alpha"), Some(id));
}

#[test]
fn delete_basic() {
    let mut idx = DynamicIndex::new();
    let id = idx.insert(k("alpha"));
    assert_eq!(idx.delete(b"alpha"), Some(id));
    assert_eq!(idx.lookup(b"alpha"), None);
    assert_eq!(idx.delete(b"alpha"), None);
}

#[test]
fn reinsert_after_delete_gets_new_id() {
    let mut idx = DynamicIndex::new();
    let first = idx.insert(k("alpha"));
    idx.delete(b"alpha");
    let second = idx.insert(k("alpha"));
    assert_eq!(second, first + 1);
    assert_eq!(idx.lookup(b"alpha"), Some(second));
}

fn small_config() -> DynamicConfig {
    DynamicConfig { flush_threshold: 16, max_tiers: 8, lean_tiers: false, parallel_build: false }
}

#[test]
fn flush_and_lookup_across_tier() {
    let mut idx = DynamicIndex::with_config(small_config());
    let mut expected = Vec::new();
    for i in 0..100u32 {
        let key = format!("key-{i}").into_bytes();
        let id = idx.insert(key.clone());
        expected.push((key, id));
    }
    idx.flush().unwrap();
    for (key, id) in &expected {
        assert_eq!(idx.lookup(key), Some(*id), "miss for {key:?}");
    }
    assert!(idx.tier_count() >= 1);
}

#[test]
fn delete_persists_across_flush() {
    let mut cfg = small_config();
    cfg.flush_threshold = 4;
    let mut idx = DynamicIndex::with_config(cfg);
    for i in 0..20u32 {
        idx.insert(format!("k-{i}").into_bytes());
    }
    idx.flush().unwrap();
    idx.delete(b"k-7");
    assert_eq!(idx.lookup(b"k-7"), None);
    let new_id = idx.insert(b"k-7".to_vec());
    idx.flush().unwrap();
    assert_eq!(idx.lookup(b"k-7"), Some(new_id));
}

#[test]
fn compact_collapses_tiers_into_one() {
    let mut cfg = small_config();
    cfg.flush_threshold = 8;
    cfg.max_tiers = 16;
    let mut idx = DynamicIndex::with_config(cfg);
    for i in 0..200u32 {
        idx.insert(format!("k-{i}").into_bytes());
    }
    idx.flush().unwrap();
    assert!(idx.tier_count() > 1);
    idx.compact().unwrap();
    assert_eq!(idx.tier_count(), 1);
    for i in 0..200u32 {
        assert!(idx.lookup(format!("k-{i}").as_bytes()).is_some());
    }
}

#[test]
fn stable_ids_survive_flush_and_compact() {
    let mut cfg = small_config();
    cfg.flush_threshold = 8;
    cfg.max_tiers = 4;
    let mut idx = DynamicIndex::with_config(cfg);
    let mut ids = std::collections::HashMap::new();
    for i in 0..50u32 {
        let key = format!("k-{i}").into_bytes();
        ids.insert(key.clone(), idx.insert(key));
    }
    idx.flush().unwrap();
    idx.compact().unwrap();
    for (key, &expected_id) in &ids {
        assert_eq!(idx.lookup(key), Some(expected_id), "id changed for {key:?}");
    }
}

#[test]
fn len_is_exact_across_promote_delete_and_revive() {
    let mut cfg = small_config();
    cfg.flush_threshold = 8;
    let mut idx = DynamicIndex::with_config(cfg);
    for i in 0..40u32 {
        idx.insert(format!("k-{i}").into_bytes());
    }
    idx.flush().unwrap();
    assert_eq!(idx.len(), 40);
    // Re-inserting a tiered key promotes it but must not double count.
    idx.insert(k("k-3"));
    assert_eq!(idx.len(), 40);
    assert_eq!(idx.delete(b"k-3"), Some(3));
    assert_eq!(idx.len(), 39);
    assert_eq!(idx.delete(b"k-3"), None);
    assert_eq!(idx.len(), 39);
    let revived = idx.insert(k("k-3"));
    assert_ne!(revived, 3);
    assert_eq!(idx.len(), 40);
    assert_eq!(idx.lookup(b"k-3"), Some(revived));
    idx.compact().unwrap();
    assert_eq!(idx.len(), 40);
    assert_eq!(idx.lookup(b"k-3"), Some(revived));
    assert_eq!(idx.tombstone_count(), 0);
}

/// Lean tiers have no filter, so the static index maps every foreign key to
/// *some* slot. The tier must verify the stored key instead of trusting it.
#[test]
fn lean_tiers_never_return_foreign_ids() {
    let mut cfg = small_config();
    cfg.flush_threshold = 64;
    cfg.max_tiers = 64;
    cfg.lean_tiers = true;
    let mut idx = DynamicIndex::with_config(cfg);
    for i in 0..1_000u32 {
        idx.insert(format!("present-{i}").into_bytes());
    }
    idx.flush().unwrap();
    assert!(idx.tier_count() > 1);
    for i in 0..5_000u32 {
        assert_eq!(idx.lookup(format!("absent-{i}").as_bytes()), None, "foreign key #{i}");
    }
    for i in 0..1_000u32 {
        assert_eq!(idx.lookup(format!("present-{i}").as_bytes()), Some(i));
    }
    // A key that lives in a deep tier must not be shadowed by a younger tier's
    // arbitrary slot.
    idx.delete(b"present-10");
    assert_eq!(idx.lookup(b"present-10"), None);
}

#[test]
fn flush_and_compact_report_success_and_keep_state() {
    let mut idx = DynamicIndex::with_config(small_config());
    assert!(idx.flush().is_ok());
    assert!(idx.compact().is_ok());
    assert_eq!(idx.tier_count(), 0);
    idx.insert(k("x"));
    idx.flush().unwrap();
    idx.flush().unwrap();
    assert_eq!(idx.tier_count(), 1);
    assert_eq!(idx.buffer_len(), 0);
}
