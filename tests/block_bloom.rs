use kira_kv_engine::__internal::BlockBloom;

#[test]
fn no_false_negatives_and_low_fp() {
    let n = 100_000usize;
    let keys: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let bb = BlockBloom::build_from_u64(&keys, 0xABCD_1234);

    for &k in &keys {
        assert!(bb.contains_u64(k), "false negative for {k:#x}");
    }

    let probes: Vec<u64> = (0..n as u64)
        .map(|i| (i + 1_000_000_000).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .collect();
    let mut fp = 0usize;
    for &k in &probes {
        if bb.contains_u64(k) {
            fp += 1;
        }
    }
    let rate = fp as f64 / n as f64;
    assert!(rate < 0.02, "false positive rate too high: {rate}");
}
