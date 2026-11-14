use kira_kv_engine::__internal::{
    BuildConfig, Builder, PtrHash25Error, read_ptrhash25, write_ptrhash25,
};

#[test]
fn duplicate_in_an_oversized_bucket_returns_the_exact_error() {
    let mut keys: Vec<u64> = (0..4096).collect();
    keys.push(keys[17]);
    let config = BuildConfig { lambda: 1_000_000.0, max_rehash: 0, ..BuildConfig::default() };
    assert!(matches!(
        Builder::new().with_config(config).build(&keys),
        Err(PtrHash25Error::DuplicateKey)
    ));
}

#[test]
fn large_buckets_at_low_load_remain_bijective_and_roundtrip() {
    let keys: Vec<u64> =
        (0..4096).map(|key: u64| key.wrapping_mul(0xD6E8_FEB8_6659_FD93)).collect();
    let config = BuildConfig { lambda: 16.0, alpha: 0.05, ..BuildConfig::default() };
    let index = Builder::new().with_config(config).build(&keys).unwrap();
    let mut seen = vec![false; keys.len()];
    for &key in &keys {
        let slot = index.index_u64(key) as usize;
        assert!(slot < keys.len());
        assert!(!seen[slot]);
        seen[slot] = true;
        assert_eq!(index.lookup_u64(key), Some(slot as u32));
    }
    assert!(seen.into_iter().all(|slot| slot));
    let mut bytes = Vec::new();
    write_ptrhash25(&index, &mut bytes);
    let mut position = 0;
    let restored = read_ptrhash25(&bytes, &mut position).unwrap();
    assert_eq!(position, bytes.len());
    for &key in &keys {
        assert_eq!(index.index_u64(key), restored.index_u64(key));
    }
}
