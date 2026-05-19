use kira_kv_engine::__internal::simd_hash;

fn gen_keys(n: usize) -> Vec<u64> {
    let mut keys = vec![0u64; n];
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for k in &mut keys {
        s ^= s << 7;
        s ^= s >> 9;
        s = s.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        *k = s;
    }
    keys
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_matches_scalar_mix64() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    let n = 4099usize;
    let keys = gen_keys(n);
    let seed = 0xA24B_1F6F_DA39_2B31u64;
    let mut scalar = vec![0u64; n];
    let mut avx2 = vec![0u64; n];
    simd_hash::hash_u64_scalar(&keys, seed, &mut scalar);
    unsafe { simd_hash::hash_u64_avx2(&keys, seed, &mut avx2) };
    assert_eq!(scalar, avx2);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_scalar_mix64() {
    let n = 4099usize;
    let keys = gen_keys(n);
    let seed = 0xA24B_1F6F_DA39_2B31u64;
    let mut scalar = vec![0u64; n];
    let mut neon = vec![0u64; n];
    simd_hash::hash_u64_scalar(&keys, seed, &mut scalar);
    unsafe { simd_hash::hash_u64_neon(&keys, seed, &mut neon) };
    assert_eq!(scalar, neon);
}
