use kira_kv_engine::__internal::BlockBloom;

#[test]
fn no_false_negatives_and_low_fp() {
    let n = 100_000usize;
    let keys: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let bb = BlockBloom::build_from_u64(&keys, 0xABCD_1234);

    for &k in &keys {
        assert!(bb.contains_u64(k), "false negative for {k:#x}");
    }

    let probes: Vec<u64> =
        (0..n as u64).map(|i| (i + 1_000_000_000).wrapping_mul(0xBF58_476D_1CE4_E5B9)).collect();
    let mut fp = 0usize;
    for &k in &probes {
        if bb.contains_u64(k) {
            fp += 1;
        }
    }
    let rate = fp as f64 / n as f64;
    assert!(rate < 0.02, "false positive rate too high: {rate}");
}

#[test]
fn serialize_round_trip_keeps_format_version() {
    let keys: Vec<u64> = (0..50_000u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let bb = BlockBloom::build_from_u64(&keys, 0x1234);
    assert_eq!(bb.bit_shift(), 26);
    let mut bytes = Vec::new();
    bb.write_to(&mut bytes);
    let mut pos = 0;
    let back = BlockBloom::read_from(&bytes, &mut pos).expect("read");
    assert_eq!(pos, bytes.len());
    assert_eq!(back.bit_shift(), 26);
    for &k in &keys {
        assert!(back.contains_u64(k));
    }
    // A blob without the v2 flag (bit 63 of the length word) is a 0.6 filter and must be
    // read with the legacy 5-bit lane index.
    let mut legacy = bytes.clone();
    let len = u64::from_le_bytes(legacy[8..16].try_into().unwrap()) & !(1u64 << 63);
    legacy[8..16].copy_from_slice(&len.to_le_bytes());
    let mut pos = 0;
    let old = BlockBloom::read_from(&legacy, &mut pos).expect("read legacy");
    assert_eq!(old.bit_shift(), 27);
}
