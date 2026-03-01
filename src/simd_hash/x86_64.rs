use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi64x,
    _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_xor_si256,
};

const MIX64_MUL: u64 = 0xd6e8_feb8_6659_fd93;

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn mul_const_u64(x: __m256i, c: u64) -> __m256i {
    // x = a + (b << 32), c = c0 + (c1 << 32)
    // x*c mod 2^64 = a*c0 + ((a*c1 + b*c0) << 32)
    let c0 = _mm256_set1_epi64x((c as u32) as i64);
    let c1 = _mm256_set1_epi64x((c >> 32) as i64);

    let a_c0 = _mm256_mul_epu32(x, c0);
    let a_c1 = _mm256_mul_epu32(x, c1);
    let b = _mm256_srli_epi64(x, 32);
    let b_c0 = _mm256_mul_epu32(b, c0);
    let hi = _mm256_slli_epi64(_mm256_add_epi64(a_c1, b_c0), 32);
    _mm256_add_epi64(a_c0, hi)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn mix64_vec(mut x: __m256i) -> __m256i {
    x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 32));
    x = mul_const_u64(x, MIX64_MUL);
    x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 32));
    x = mul_const_u64(x, MIX64_MUL);
    _mm256_xor_si256(x, _mm256_srli_epi64(x, 32))
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn hash_u64_avx2(keys: &[u64], seed: u64, out: &mut [u64]) {
    debug_assert_eq!(keys.len(), out.len());
    let n = keys.len();
    let mut i = 0usize;
    let seed_v = _mm256_set1_epi64x(seed as i64);

    while i + 4 <= n {
        let x = _mm256_loadu_si256(keys.as_ptr().add(i) as *const __m256i);
        let y = _mm256_xor_si256(x, seed_v);
        let h = mix64_vec(y);
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, h);
        i += 4;
    }

    for j in i..n {
        out[j] = crate::simd_hash::scalar::mix64(keys[j] ^ seed);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn avx2_matches_scalar_mix64() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let n = 4099usize;
        let mut keys = vec![0u64; n];
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for k in &mut keys {
            s ^= s << 7;
            s ^= s >> 9;
            s = s.wrapping_mul(0xD6E8_FEB8_6659_FD93);
            *k = s;
        }
        let seed = 0xA24B_1F6F_DA39_2B31u64;
        let mut scalar = vec![0u64; n];
        let mut avx2 = vec![0u64; n];
        crate::simd_hash::scalar::hash_u64_scalar(&keys, seed, &mut scalar);
        unsafe { super::hash_u64_avx2(&keys, seed, &mut avx2) };
        assert_eq!(scalar, avx2);
    }
}
