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
    DynamicConfig {
        flush_threshold: 16,
        max_tiers: 8,
        lean_tiers: false,
        parallel_build: false,
    }
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
    idx.flush();
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
    idx.flush();
    idx.delete(b"k-7");
    assert_eq!(idx.lookup(b"k-7"), None);
    let new_id = idx.insert(b"k-7".to_vec());
    idx.flush();
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
    idx.flush();
    assert!(idx.tier_count() > 1);
    idx.compact();
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
    idx.flush();
    idx.compact();
    for (key, &expected_id) in &ids {
        assert_eq!(idx.lookup(key), Some(expected_id), "id changed for {key:?}");
    }
}
