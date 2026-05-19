use kira_kv_engine::{PgmIndexU128, PgmU128Error};

#[test]
fn build_and_lookup_dense() {
    let keys: Vec<u128> = (1..=10_000).map(|i| (i as u128) * 1_000_003).collect();
    let idx = PgmIndexU128::build(keys.clone(), 64).unwrap();
    for (expected, k) in keys.iter().enumerate() {
        assert_eq!(idx.index(*k).unwrap(), expected, "for key {k}");
    }
    assert!(matches!(idx.index(7).err(), Some(PgmU128Error::KeyNotFound)));
}

#[test]
fn range_query() {
    let keys: Vec<u128> = (10..1010).map(|i| i as u128).collect();
    let idx = PgmIndexU128::build(keys, 16).unwrap();
    let r = idx.range(100, 200);
    assert_eq!(r.first(), Some(&90));
    assert_eq!(r.last(), Some(&190));
}

#[test]
fn bytes16_roundtrip() {
    let mut bytes: Vec<[u8; 16]> = (0u128..1000).map(|i| (i * 1_000_003).to_be_bytes()).collect();
    bytes.sort();
    let idx = PgmIndexU128::build_from_bytes16(&bytes, 64).unwrap();
    for (i, b) in bytes.iter().enumerate() {
        assert_eq!(idx.index_bytes16(b).unwrap(), i);
    }
}
