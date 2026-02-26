use core::arch::x86_64::{
    __m256i, _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_or_si256, _mm256_set_epi32,
    _mm256_set1_epi32, _mm256_slli_epi64, _mm256_srli_epi32, _mm256_srli_epi64,
    _mm256_storeu_si256, _mm256_xor_si256,
};

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn mix32_vec(x: __m256i) -> __m256i {
    let mut v = x;
    v = _mm256_xor_si256(v, _mm256_srli_epi32(v, 16));
    v = _mm256_mullo_epi32(v, _mm256_set1_epi32(0x7FEB_352D_u32 as i32));
    v = _mm256_xor_si256(v, _mm256_srli_epi32(v, 15));
    v = _mm256_mullo_epi32(v, _mm256_set1_epi32(0x846C_A68B_u32 as i32));
    _mm256_xor_si256(v, _mm256_srli_epi32(v, 16))
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
pub unsafe fn hash_u64_avx2(keys: &[u64], seed: u64, out: &mut [u64]) {
    crate::simd_hash::scalar::hash_u64_scalar(keys, seed, out);
}
