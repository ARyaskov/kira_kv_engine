use kira_kv_engine::__internal::{Builder, read_ptrhash25, write_ptrhash25};

#[test]
fn round_trip_small() {
    let keys: Vec<u64> = (0..1000u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let mph = Builder::new().build(&keys).expect("build failed");
    let slot_space = mph.n as usize;
    let mut seen = vec![false; slot_space];
    for &k in &keys {
        let slot = mph.index_u64(k) as usize;
        assert!(slot < slot_space, "slot out of range");
        assert!(!seen[slot], "collision at slot {slot}");
        seen[slot] = true;
    }
    for &k in &keys {
        assert!(mph.lookup_u64(k).is_some(), "key {k:#x} rejected");
    }
}

#[test]
fn round_trip_100k() {
    let n = 100_000usize;
    let keys: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0xBF58_476D_1CE4_E5B9)).collect();
    let mph = Builder::new().build(&keys).expect("build failed");
    let slot_space = mph.n as usize;
    let mut seen = vec![false; slot_space];
    for &k in &keys {
        let slot = mph.index_u64(k) as usize;
        assert!(slot < slot_space);
        assert!(!seen[slot]);
        seen[slot] = true;
    }
}

#[test]
fn fingerprint_rejects_foreign_keys() {
    let keys: Vec<u64> = (0..10_000u64).map(|i| i.wrapping_mul(0xD6E8_FEB8_6659_FD93)).collect();
    let mph = Builder::new().build(&keys).expect("build failed");
    let mut false_hits = 0;
    for k in 1_000_000u64..1_010_000 {
        if mph.lookup_u64(k).is_some() {
            false_hits += 1;
        }
    }
    let rate = false_hits as f64 / 10_000.0;
    assert!(rate < 0.02, "false hit rate too high: {rate}");
}

#[test]
fn serialize_round_trip() {
    let keys: Vec<u64> = (0..5000u64).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();
    let mph = Builder::new().build(&keys).expect("build failed");
    let mut bytes = Vec::new();
    write_ptrhash25(&mph, &mut bytes);
    let mut pos = 0;
    let restored = read_ptrhash25(&bytes, &mut pos).expect("read failed");
    assert_eq!(pos, bytes.len());
    for &k in &keys {
        assert_eq!(mph.index_u64(k), restored.index_u64(k));
    }
}
