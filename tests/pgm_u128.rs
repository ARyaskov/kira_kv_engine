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

fn splitmix(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Random 128-bit keys (UUID-like): every key must be found and the segments
/// must be long — the absolute-key fit used to give ~4 keys per segment.
#[test]
fn random_uuid_like_keys_find_all_and_segment_well() {
    let mut s = 0xABCDu64;
    let mut keys: Vec<u128> = (0..200_000)
        .map(|_| ((splitmix(&mut s) as u128) << 64) | splitmix(&mut s) as u128)
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let eps = 64;
    let idx = PgmIndexU128::build(keys.clone(), eps).unwrap();
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(idx.index(k).ok(), Some(i), "miss at #{i}");
        assert_eq!(idx.lower_bound(k), i);
    }
    let avg = keys.len() as f64 / idx.segments_count() as f64;
    assert!(avg > 200.0, "only {avg:.1} keys per segment");
    let per_key = idx.memory_usage() as f64 / keys.len() as f64;
    assert!(per_key < 17.0, "{per_key:.2} B/key");
    assert!(idx.index(keys[10] + 1).is_err());
}
