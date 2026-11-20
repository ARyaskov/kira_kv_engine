use kira_kv_engine::{
    DynamicHotTier, HotTierError, HotTierIndex, Index, IndexBuilder, IndexError, MembershipMode,
};

#[test]
fn static_membership_capabilities_and_explicit_candidate_checks() {
    assert_eq!(Index::empty().membership_mode(), MembershipMode::Exact);
    let keys: &[&[u8]] = &[b"alpha", b"beta", b"gamma"];
    for lean in [false, true] {
        let index = IndexBuilder::new().with_lean_mph(lean).build_index_ref(keys).unwrap();
        assert_eq!(
            index.membership_mode(),
            if lean { MembershipMode::ClosedWorld } else { MembershipMode::Probabilistic }
        );
        let pattern: &[&[u8]] = &[b"alpha", b"absent", b"beta", b"also absent"];
        let queries: Vec<_> = pattern.iter().copied().cycle().take(145).collect();
        let expected: Vec<bool> = queries.iter().map(|key| index.lookup(key).is_ok()).collect();
        assert_eq!(index.may_contain_batch(&queries), expected);
        for (&key, answer) in queries.iter().zip(expected) {
            assert_eq!(index.may_contain(key), answer);
        }
    }
}

fn legacy_tag2_without_outer_membership(backend: &[u8]) -> Index {
    let mut bytes = vec![2];
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(backend);
    bytes.extend_from_slice(&[0, 0]);
    Index::from_bytes(&bytes).unwrap()
}

#[test]
fn legacy_inner_mph_fingerprints_do_not_claim_negative_lookup_support() {
    use kira_kv_engine::__internal::{BuildConfig, Builder, write_ptrhash25};
    let mph = Builder::new()
        .with_config(BuildConfig { with_fingerprints: true, ..BuildConfig::default() })
        .build(&[0])
        .unwrap();
    assert_eq!(mph.fingerprints().len(), 1);
    let mut backend = vec![0, 0];
    write_ptrhash25(&mph, &mut backend);
    let index = legacy_tag2_without_outer_membership(&backend);
    assert_eq!(index.membership_mode(), MembershipMode::ClosedWorld);
    assert!(!index.supports_negative_lookups());
    assert_eq!(index.lookup_u64(0).unwrap(), 0);
    assert_eq!(index.lookup_u64(1).unwrap(), 0);
}

#[test]
fn legacy_canonical_hash_map_reports_its_negative_lookup_support() {
    let mut backend = vec![0, 1];
    backend.extend_from_slice(&1u64.to_le_bytes());
    backend.extend_from_slice(&0u64.to_le_bytes());
    backend.extend_from_slice(&0u32.to_le_bytes());
    let index = legacy_tag2_without_outer_membership(&backend);
    assert_eq!(index.membership_mode(), MembershipMode::Probabilistic);
    assert!(index.supports_negative_lookups());
    assert_eq!(index.lookup_u64(0).unwrap(), 0);
    assert!(matches!(index.lookup_u64(1), Err(IndexError::KeyNotFound)));
}

#[test]
fn verified_static_lookup_compares_the_authoritative_full_key() {
    let keys: &[&[u8]] = &[b"alpha", b"beta", b"gamma"];
    let index = IndexBuilder::new().with_lean_mph(true).build_index_ref(keys).unwrap();
    let mut slots = vec![&[][..]; index.slot_capacity()];
    for key in keys {
        slots[index.lookup(key).unwrap()] = key;
    }
    for key in keys {
        assert_eq!(
            index.lookup_verified(key, |slot, requested| slots[slot] == requested).unwrap(),
            index.lookup(key).unwrap()
        );
    }
    let foreign = b"not in the build set";
    assert!(index.lookup(foreign).is_ok());
    assert!(matches!(
        index.lookup_verified(foreign, |slot, requested| slots[slot] == requested),
        Err(IndexError::KeyNotFound)
    ));
    assert!(matches!(
        Index::empty().lookup_verified(foreign, |_, _| panic!("no candidate to verify")),
        Err(IndexError::KeyNotFound)
    ));
}

#[test]
fn explicit_membership_checks_fingerprints_when_legacy_contains_accepts_bloom() {
    use kira_kv_engine::__internal::{
        BlockBloom, BuildConfig, Builder, simd_hash, write_ptrhash25,
    };
    let mut hashes = [0; 2];
    simd_hash::hash_u64_scalar(&[7, 8], 0, &mut hashes);
    assert_ne!(hashes[0] as u16, hashes[1] as u16);
    let mph = Builder::new()
        .with_config(BuildConfig { with_fingerprints: false, ..BuildConfig::default() })
        .build(&hashes[..1])
        .unwrap();
    let mut bytes = vec![0];
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&[0, 0]);
    write_ptrhash25(&mph, &mut bytes);
    let bloom_start = bytes.len();
    BlockBloom::build_from_prehashed(&hashes[..1]).write_to(&mut bytes);
    bytes[bloom_start + 16..].fill(255);
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&(hashes[0] as u16).to_le_bytes());
    let index = Index::from_bytes(&bytes).unwrap();
    let present = 7u64.to_le_bytes();
    let foreign = 8u64.to_le_bytes();
    assert!(index.contains(&foreign));
    assert!(matches!(index.lookup(&foreign), Err(IndexError::KeyNotFound)));
    assert!(!index.may_contain(&foreign));
    assert_eq!(index.contains_batch(&[&present, &foreign]), [true, true]);
    assert_eq!(index.may_contain_batch(&[&present, &foreign]), [true, false]);
}

#[test]
fn checked_hot_builder_reports_invalid_inputs_without_panicking() {
    assert!(matches!(
        HotTierIndex::try_build_from_u64(&[1, 2], &[9], 42),
        Err(HotTierError::LengthMismatch)
    ));
    assert!(HotTierIndex::build_from_u64(&[1, 2], &[9], 42).is_none());
    assert!(matches!(HotTierIndex::try_build_from_u64(&[], &[], 42), Err(HotTierError::EmptyKeys)));
    assert!(matches!(
        HotTierIndex::try_build_from_u64(&[1, 1], &[9, 10], 42),
        Err(HotTierError::DuplicateKey)
    ));
    assert!(matches!(
        HotTierIndex::try_build_from_u64(&[1], &[u32::MAX], 42),
        Err(HotTierError::ReservedIndex)
    ));
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn unmix(mut x: u64) -> u64 {
    x ^= x >> 31;
    x ^= x >> 62;
    x = x.wrapping_mul(0x3196_42B2_D24D_8EC3);
    x ^= x >> 27;
    x ^= x >> 54;
    x = x.wrapping_mul(0x96DE_1B17_3F11_9089);
    x ^= x >> 30;
    x ^= x >> 60;
    x.wrapping_sub(0x9E37_79B9_7F4A_7C15)
}

fn same_fingerprint_key(key: u64) -> u64 {
    let salt = 0xA24B_1F6F_1234_5678;
    let hash = mix(key ^ salt);
    let foreign = unmix(hash ^ (1 << 16)) ^ salt;
    assert_ne!(foreign, key);
    assert_eq!(mix(foreign ^ salt) as u16, hash as u16);
    foreign
}

fn legacy_hot_payload(key: u64, index: u32) -> Vec<u8> {
    use kira_kv_engine::__internal::{BlockBloom, BuildConfig, Builder, write_ptrhash25};
    let mut bytes = Vec::new();
    BlockBloom::build_from_u64(&[key], 42).write_to(&mut bytes);
    // A saturated Bloom still has valid structure and no false negatives.
    bytes[16..].fill(255);
    let mph = Builder::new()
        .with_config(BuildConfig { with_fingerprints: false, ..BuildConfig::default() })
        .build(&[key])
        .unwrap();
    write_ptrhash25(&mph, &mut bytes);
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&(mix(key ^ 0xA24B_1F6F_1234_5678) as u16).to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&index.to_le_bytes());
    bytes
}

#[test]
fn legacy_hot_tiers_remain_readable_but_cannot_claim_an_exact_cache_hit() {
    let bytes = legacy_hot_payload(7, 77);
    let foreign = same_fingerprint_key(7);
    let mut cursor = 0;
    let legacy = HotTierIndex::read_from(&bytes, &mut cursor).unwrap();
    assert_eq!(cursor, bytes.len());
    assert_eq!(legacy.membership_mode(), MembershipMode::Probabilistic);
    assert_eq!(legacy.lookup_u64(7), Some(77));
    assert_eq!(legacy.lookup_u64(foreign), Some(77));
    assert_eq!(legacy.lookup_exact_u64(7), None);
    assert_eq!(legacy.lookup_exact_u64(foreign), None);
    let mut rewritten = Vec::new();
    legacy.write_to(&mut rewritten);
    assert_eq!(rewritten, bytes);
    let cache = DynamicHotTier::new(Some(legacy), 16, 0);
    assert_eq!(cache.lookup_u64(foreign), None);
    assert_eq!(cache.lookup_u64(7), None);
}

fn refresh_checksum(bytes: &mut [u8]) {
    let end = bytes.len() - 8;
    let mut state = 0x4B49_5241_5F43_4B53;
    for chunk in bytes[..end].chunks(64 * 1024) {
        state = wyhash::wyhash(chunk, state);
    }
    let sum = wyhash::wyhash(&state.to_le_bytes(), state);
    bytes[end..].copy_from_slice(&sum.to_le_bytes());
}

#[test]
fn exact_hot_tier_rejects_a_foreign_key_even_when_bloom_and_fingerprint_match() {
    let tier = HotTierIndex::try_build_from_u64(&[7], &[77], 42).unwrap();
    assert_eq!(tier.membership_mode(), MembershipMode::Exact);
    let mut bytes = Vec::new();
    tier.write_to(&mut bytes);
    let bloom_offset = 24;
    let words = u64::from_le_bytes(bytes[bloom_offset + 8..bloom_offset + 16].try_into().unwrap())
        & !(1 << 63);
    bytes[bloom_offset + 16..bloom_offset + 16 + words as usize * 8].fill(255);
    refresh_checksum(&mut bytes);
    let tier = HotTierIndex::read_from(&bytes, &mut 0).unwrap();
    let foreign = same_fingerprint_key(7);
    assert_eq!(tier.lookup_u64(7), Some(77));
    assert_eq!(tier.lookup_exact_u64(7), Some(77));
    assert_eq!(tier.lookup_u64(foreign), None);
    let cache = DynamicHotTier::new(Some(tier), 16, 0);
    assert_eq!(cache.lookup_u64(foreign), None);
    assert_eq!(cache.lookup_u64(7), Some(77));
}

#[test]
fn exact_hot_frames_preserve_cursor_boundaries_and_reject_corruption() {
    let tier = HotTierIndex::try_build_from_u64(&[0, 7, u64::MAX], &[11, 77, 99], 42).unwrap();
    let mut bytes = vec![13, 17];
    tier.write_to(&mut bytes);
    let end = bytes.len();
    bytes.extend_from_slice(b"following payload");
    let mut cursor = 2;
    let restored = HotTierIndex::read_from(&bytes, &mut cursor).unwrap();
    assert_eq!(cursor, end);
    assert_eq!(restored.membership_mode(), MembershipMode::Exact);
    assert_eq!(restored.lookup_u64(u64::MAX), Some(99));
    let mut saved = Vec::new();
    restored.write_to(&mut saved);
    assert_eq!(saved, bytes[2..end]);
    for offset in [2, 10, 18, end - 9, end - 1] {
        let mut changed = bytes.clone();
        changed[offset] ^= 1;
        let mut cursor = 2;
        assert!(HotTierIndex::read_from(&changed, &mut cursor).is_none());
        assert_eq!(cursor, 2);
    }
    for length in [0, 1, 15, 23, end - 3] {
        assert!(HotTierIndex::read_from(&bytes[..length], &mut 2).is_none());
    }
    let mut cursor = usize::MAX;
    assert!(HotTierIndex::read_from(&[], &mut cursor).is_none());
    assert_eq!(cursor, usize::MAX);
}

#[test]
fn frozen_legacy_hot_fixture_preserves_payload_positions_and_following_bytes() {
    let fixture = include_bytes!("fixtures/hot_tier_v081.bin");
    let mut bytes = vec![5, 9, 17];
    bytes.extend_from_slice(fixture);
    let end = bytes.len();
    bytes.extend_from_slice(b"next value");
    let mut cursor = 3;
    let legacy = HotTierIndex::read_from(&bytes, &mut cursor).unwrap();
    assert_eq!(cursor, end);
    assert_eq!(legacy.membership_mode(), MembershipMode::Probabilistic);
    for (key, index) in [(7, 100), (11, 200), (42, 300), (99, 400)] {
        assert_eq!(legacy.lookup_u64(key), Some(index));
    }
    let mut saved = Vec::new();
    legacy.write_to(&mut saved);
    assert_eq!(saved, fixture);
}
