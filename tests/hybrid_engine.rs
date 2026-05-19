use kira_kv_engine::HybridBuilder;

#[test]
fn build_and_lookup_byte_keys() {
    let keys: Vec<Vec<u8>> = (0..1000).map(|i| format!("key-{i}").into_bytes()).collect();
    let idx = HybridBuilder::new()
        .with_pgm_epsilon(32)
        .with_linear_threshold(32)
        .build(&keys)
        .unwrap();
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(
            idx.lookup(k),
            Some(i as u32),
            "miss at #{i}: {:?}",
            std::str::from_utf8(k).unwrap()
        );
    }
    assert_eq!(idx.lookup(b"missing-key"), None);
}

#[test]
fn build_and_lookup_u64_keys() {
    let keys: Vec<[u8; 8]> = (0u64..5_000).map(|i| (i * 997).to_le_bytes()).collect();
    let idx = HybridBuilder::new().with_pgm_epsilon(64).build(&keys).unwrap();
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(idx.lookup(k), Some(i as u32));
    }
}

#[test]
fn lookup_batch_works() {
    let keys: Vec<Vec<u8>> = (0u32..500).map(|i| i.to_le_bytes().repeat(4)).collect();
    let idx = HybridBuilder::new().build(&keys).unwrap();
    let res = idx.lookup_batch(&keys);
    for (i, opt) in res.into_iter().enumerate() {
        assert_eq!(opt, Some(i as u32));
    }
}

#[test]
fn build_from_u64_simd_path() {
    let keys: Vec<u64> = (0u64..5_000).map(|i| i * 1_000_003).collect();
    let idx = HybridBuilder::new().with_pgm_epsilon(64).build_from_u64(&keys).unwrap();
    let key_bytes: Vec<[u8; 8]> = keys.iter().map(|k| k.to_le_bytes()).collect();
    let idx_scalar = HybridBuilder::new()
        .with_pgm_epsilon(64)
        .build(&key_bytes)
        .unwrap();
    for (k_u64, k_bytes) in keys.iter().zip(key_bytes.iter()) {
        let a = idx.lookup_u64(*k_u64);
        let b = idx_scalar.lookup(k_bytes);
        assert!(a.is_some());
        assert_eq!(a, b, "key {k_u64}");
    }
}

#[test]
fn lookup_batch_u64_simd_correctness() {
    let keys: Vec<u64> = (0u64..2_000).map(|i| i * 17 + 3).collect();
    let idx = HybridBuilder::new().with_pgm_epsilon(64).build_from_u64(&keys).unwrap();
    let res = idx.lookup_batch_u64_simd(&keys);
    for (i, r) in res.iter().enumerate() {
        assert_eq!(*r, Some(i as u32), "batch miss at #{i}");
    }
    let bad: Vec<u64> = (0u64..100).map(|i| 999_999_999 + i).collect();
    let res_bad = idx.lookup_batch_u64_simd(&bad);
    let hits = res_bad.iter().filter(|r| r.is_some()).count();
    assert!(hits < 5, "too many false positives: {hits}/100");
}

#[test]
fn lean_mode_lookup_works_on_valid_keys() {
    let keys: Vec<Vec<u8>> = (0u32..2000).map(|i| format!("hot-key-{i}").into_bytes()).collect();
    let lean = HybridBuilder::new()
        .with_pgm_epsilon(256)
        .with_lean(true)
        .build(&keys)
        .unwrap();
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(lean.lookup(k), Some(i as u32), "miss at #{i}");
    }
    let full = HybridBuilder::new()
        .with_pgm_epsilon(256)
        .with_lean(false)
        .build(&keys)
        .unwrap();
    assert!(lean.memory_usage() < full.memory_usage());
}

#[test]
fn lookup_after_negative_workload() {
    let keys: Vec<Vec<u8>> = (0..1000).map(|i| format!("real-{i}").into_bytes()).collect();
    let idx = HybridBuilder::new().build(&keys).unwrap();
    for i in 0..100 {
        assert_eq!(idx.lookup(format!("fake-{i}").as_bytes()), None);
    }
    for (i, k) in keys.iter().enumerate().take(100) {
        assert_eq!(idx.lookup(k), Some(i as u32));
    }
}

#[test]
fn storage_stats_mixed() {
    let keys: Vec<[u8; 8]> = (0u64..20_000).map(|i| (i * 13).to_le_bytes()).collect();
    let idx = HybridBuilder::new()
        .with_pgm_epsilon(256)
        .with_linear_threshold(64)
        .build(&keys)
        .unwrap();
    let stats = idx.storage_stats();
    assert_eq!(
        stats.total_segments,
        stats.linear_segments + stats.chd_segments + stats.mph_segments
    );
    assert_eq!(stats.linear_keys + stats.chd_keys + stats.mph_keys, 20_000);
}
