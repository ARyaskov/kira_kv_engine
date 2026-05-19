use kira_kv_engine::__internal::{MiniChd, MiniChdError};

#[test]
fn build_and_lookup_unique_small() {
    let keys: Vec<u64> = (0u64..100).map(|i| i * 0x9E37).collect();
    let chd = MiniChd::build(&keys, 0xDEAD_BEEF).unwrap();
    let mut seen = vec![false; chd.n as usize];
    for &k in &keys {
        let s = chd.index(k) as usize;
        assert!(s < chd.n as usize);
        assert!(!seen[s], "collision at key {k} → slot {s}");
        seen[s] = true;
    }
}

#[test]
fn build_and_lookup_2k_keys() {
    let keys: Vec<u64> = (0u64..2000).map(|i| (i * 0xABCD_1234) ^ (i << 17)).collect();
    let chd = MiniChd::build(&keys, 0xCAFE_BABE).unwrap();
    let mut seen = vec![false; chd.n as usize];
    for &k in &keys {
        let s = chd.index(k) as usize;
        assert!(!seen[s]);
        seen[s] = true;
    }
}

#[test]
fn rejects_empty() {
    assert!(matches!(MiniChd::build(&[], 0), Err(MiniChdError::Empty)));
}
