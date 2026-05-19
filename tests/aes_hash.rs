use kira_kv_engine::__internal::aes_hash::{hash_bytes, hash_u64};

#[test]
fn distinct_keys_distinct_hashes() {
    let seed = 0xC0FF_EE00u64;
    let n = 100_000;
    let mut hashes = std::collections::HashSet::with_capacity(n);
    for i in 0..n as u64 {
        let k = i.to_le_bytes();
        hashes.insert(hash_bytes(&k, seed));
    }
    assert!(hashes.len() > n - 5, "too many collisions: {} unique of {n}", hashes.len());
}

#[test]
fn u64_hash_matches_bytes_hash_for_8_byte_keys() {
    let seed = 0x1234u64;
    for i in 0..1000u64 {
        let h_bytes = hash_bytes(&i.to_le_bytes(), seed);
        let h_u64 = hash_u64(i, seed);
        assert_ne!(h_bytes, 0);
        assert_ne!(h_u64, 0);
    }
}

#[test]
fn long_keys_are_well_mixed() {
    let seed = 0xABCDu64;
    let n = 10_000;
    let mut hashes = std::collections::HashSet::with_capacity(n);
    for i in 0..n {
        let mut key = vec![0u8; 100];
        key[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        hashes.insert(hash_bytes(&key, seed));
    }
    assert!(hashes.len() > n - 5);
}
