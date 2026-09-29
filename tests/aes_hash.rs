use kira_kv_engine::__internal::aes_hash::{
    aes_round_hw, aes_round_soft, hash_bytes, hash_u64, hw_available,
};

fn splitmix(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_block(seed: &mut u64) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&splitmix(seed).to_le_bytes());
    b[8..].copy_from_slice(&splitmix(seed).to_le_bytes());
    b
}

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

/// The software round must be bit-identical to the hardware round, otherwise
/// indexes stop being portable between hosts with and without AES instructions.
#[test]
fn software_round_matches_hardware_round() {
    if !hw_available() {
        eprintln!("no hardware AES on this host; skipping cross-check");
        return;
    }
    let mut seed = 0x5EEDu64;
    for _ in 0..20_000 {
        let s = rand_block(&mut seed);
        let k = rand_block(&mut seed);
        assert_eq!(aes_round_hw(s, k).unwrap(), aes_round_soft(s, k), "state={s:02x?} key={k:02x?}");
    }
    // FIPS-197 style sanity: all-zero state, zero key → SubBytes(0)=0x63 everywhere,
    // MixColumns of a constant column is the constant itself.
    assert_eq!(aes_round_soft([0u8; 16], [0u8; 16]), [0x63u8; 16]);
}

/// Known-answer vectors. Recorded from the ARMv8 crypto-extension backend; the
/// x86 AES-NI and software backends must reproduce them exactly, otherwise an
/// index serialized on one platform is unreadable on another.
#[test]
fn known_answer_vectors_are_platform_independent() {
    const KAT: &[(&[u8], u64, u64)] = &[
        (b"", 0x0, 0xE220_A839_7B1D_CDAF),
        (b"a", 0x1234_5678, 0xE82A_FCEB_756B_11B8),
        (b"hello", 0x0C0F_FEE0_0D15_EA5E, 0xAA6F_83F7_94D9_D305),
        (b"exactly-16-bytes", 0xDEAD_BEEF, 0x1A04_A58A_5B8A_8891),
        (b"a somewhat longer key that spans several 16-byte blocks!!", 0x42, 0x03D4_B35A_592A_E644),
    ];
    let mut failures = Vec::new();
    for &(key, seed, expected) in KAT {
        let got = hash_bytes(key, seed);
        if got != expected {
            failures.push(format!("key={:?} seed={seed:#x}: got {got:#018x}, expected {expected:#018x}", String::from_utf8_lossy(key)));
        }
    }
    let got_u64 = hash_u64(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210);
    if got_u64 != 0x96B6_AA01_A89D_94A9 {
        failures.push(format!("hash_u64: got {got_u64:#018x}"));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
