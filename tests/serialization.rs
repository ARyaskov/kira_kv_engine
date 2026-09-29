//! Every engine round-trips through its checksummed byte form, empty inputs
//! build always-miss indexes, and corrupt bytes are rejected.

use kira_kv_engine::{
    DynamicConfig, DynamicIndex, HotTierIndex, HybridBuilder, HybridIndex, PgmIndexU128,
    PgmU128Error,
};

fn splitmix(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn mutations(bytes: &[u8], rounds: usize, seed: u64, mut check: impl FnMut(&[u8])) {
    let mut s = seed;
    for _ in 0..rounds {
        let mut b = bytes.to_vec();
        let pos = (splitmix(&mut s) % b.len() as u64) as usize;
        b[pos] ^= 1 << (splitmix(&mut s) % 8);
        check(&b);
    }
    check(&bytes[..bytes.len() / 2]);
    check(&bytes[..bytes.len() - 1]);
    check(&[]);
}

#[test]
fn hybrid_roundtrip_all_segment_kinds() {
    let mut s = 0x11u64;
    // Small epsilon → linear segments, large → CHD and PtrHash segments.
    for (n, eps, lean) in [(3_000usize, 16u32, false), (60_000, 512, true), (120_000, 4096, false)]
    {
        let keys: Vec<u64> = (0..n).map(|_| splitmix(&mut s)).collect();
        let idx = HybridBuilder::new()
            .with_pgm_epsilon(eps)
            .with_lean(lean)
            .build_from_u64(&keys)
            .unwrap();
        let st = idx.storage_stats();
        let bytes = idx.to_bytes();
        let back = HybridIndex::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(back.len(), idx.len());
        assert_eq!(back.num_segments(), idx.num_segments());
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(back.lookup_u64(k), Some(i as u32), "n={n} eps={eps} #{i} ({st:?})");
        }
        if !lean {
            for _ in 0..2_000 {
                let k = splitmix(&mut s);
                assert_eq!(back.lookup_u64(k), idx.lookup_u64(k));
            }
        }
        mutations(&bytes, 300, 0x22, |b| assert!(HybridIndex::from_bytes(b).is_err()));
    }
    // Byte keys through the scalar hash path.
    let keys: Vec<Vec<u8>> = (0..5_000).map(|i| format!("key-{i}").into_bytes()).collect();
    let idx = HybridBuilder::new().with_pgm_epsilon(64).build(&keys).unwrap();
    let back = HybridIndex::from_bytes(&idx.to_bytes()).unwrap();
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(back.lookup(k), Some(i as u32));
    }
    assert_eq!(back.lookup(b"absent"), None);
}

#[test]
fn hybrid_accepts_empty_input() {
    let idx = HybridBuilder::new().build(&Vec::<Vec<u8>>::new()).unwrap();
    assert!(idx.is_empty());
    assert_eq!(idx.lookup(b"x"), None);
    assert_eq!(idx.lookup_u64(7), None);
    assert_eq!(idx.lookup_batch_u64_simd(&[1, 2, 3]), vec![None, None, None]);
    let back = HybridIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert!(back.is_empty());
    let u = HybridBuilder::new().build_from_u64(&[]).unwrap();
    assert!(u.is_empty());
}

#[test]
fn pgm_u128_roundtrip_empty_and_duplicates() {
    let mut s = 0x33u64;
    let mut keys: Vec<u128> = (0..50_000)
        .map(|_| ((splitmix(&mut s) as u128) << 64) | splitmix(&mut s) as u128)
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let idx = PgmIndexU128::build(keys.clone(), 32).unwrap();
    let bytes = idx.to_bytes();
    let back = PgmIndexU128::from_bytes(&bytes).unwrap();
    assert_eq!(back.len(), keys.len());
    for (i, &k) in keys.iter().enumerate().step_by(3) {
        assert_eq!(back.index(k).ok(), Some(i));
        assert_eq!(back.lower_bound(k), i);
    }
    assert!(back.index(keys[5] + 1).is_err());
    mutations(&bytes, 400, 0x44, |b| assert!(PgmIndexU128::from_bytes(b).is_err()));

    let empty = PgmIndexU128::build(Vec::new(), 16).unwrap();
    assert!(empty.is_empty());
    assert!(empty.index(1).is_err());
    assert_eq!(empty.lower_bound(0), 0);
    assert!(empty.range(0, u128::MAX).is_empty());
    let back = PgmIndexU128::from_bytes(&empty.to_bytes()).unwrap();
    assert!(back.is_empty());

    let dup = PgmIndexU128::build(vec![5, 3, 5], 16).err().unwrap();
    assert!(matches!(dup, PgmU128Error::DuplicateKeys));
}

#[test]
fn dynamic_index_roundtrip_preserves_ids_and_deletes() {
    let mut idx = DynamicIndex::with_config(DynamicConfig {
        flush_threshold: 32,
        max_tiers: 64,
        lean_tiers: false,
        parallel_build: false,
    });
    let mut expect = std::collections::HashMap::new();
    for i in 0..500u32 {
        let k = format!("k-{i}").into_bytes();
        let id = idx.insert(k.clone());
        expect.insert(k, id);
    }
    for i in (0..500u32).step_by(7) {
        let k = format!("k-{i}").into_bytes();
        idx.delete(&k);
        expect.remove(&k);
    }
    // A revived key gets a fresh id; keep it in the buffer (unflushed).
    let revived = idx.insert(b"k-14".to_vec());
    expect.insert(b"k-14".to_vec(), revived);
    assert!(idx.tier_count() > 1 && idx.buffer_len() > 0 && idx.tombstone_count() > 0);
    let bytes = idx.to_bytes();
    let back = DynamicIndex::from_bytes(&bytes).unwrap();
    assert_eq!(back.len(), expect.len());
    assert_eq!(back.tier_count(), 1);
    for (k, &id) in &expect {
        assert_eq!(back.lookup(k), Some(id), "{}", String::from_utf8_lossy(k));
    }
    for i in (0..500u32).step_by(7).skip(3) {
        assert_eq!(back.lookup(format!("k-{i}").as_bytes()), None);
    }
    // Fresh ids continue after the highest stored one.
    let mut back = back;
    let new_id = back.insert(b"brand-new".to_vec());
    assert!(expect.values().all(|&id| id < new_id));
    mutations(&bytes, 300, 0x55, |b| assert!(DynamicIndex::from_bytes(b).is_err()));
    let empty = DynamicIndex::from_bytes(&DynamicIndex::new().to_bytes()).unwrap();
    assert!(empty.is_empty());
}

#[test]
fn hot_tier_index_is_constructible_from_the_public_api() {
    let keys: Vec<u64> = (0..1_000u64).map(|i| i * 7919).collect();
    let indices: Vec<u32> = (0..1_000u32).collect();
    let tier = HotTierIndex::build_from_u64(&keys, &indices, 42).expect("build");
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(tier.lookup_u64(k), Some(i as u32));
    }
    let misses =
        (1..2_000u64).map(|i| i * 7919 + 1).filter(|&k| tier.lookup_u64(k).is_some()).count();
    assert!(misses < 40, "false positives: {misses}");
    let mut bytes = Vec::new();
    tier.write_to(&mut bytes);
    let mut pos = 0;
    let back = HotTierIndex::read_from(&bytes, &mut pos).unwrap();
    assert_eq!(back.lookup_u64(keys[3]), Some(3));
    let dynamic = kira_kv_engine::DynamicHotTier::new(Some(tier), 64, 0);
    assert_eq!(dynamic.lookup_u64(keys[10]), Some(10));
}
