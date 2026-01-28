use core::arch::x86_64::{
    __m256i, _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_set_epi32, _mm256_set1_epi32,
    _mm256_srli_epi32, _mm256_storeu_si256, _mm256_xor_si256,
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
#[target_feature(enable = "avx2")]
pub unsafe fn hash_u64_avx2(keys: &[u64], seed: u64, out: &mut [u64]) {
    let mut i = 0usize;
    let seed_lo = seed as u32 as i32;
    let seed_hi = (seed >> 32) as u32 as i32;
    let seed_vec = _mm256_set_epi32(
        seed_hi, seed_lo, seed_hi, seed_lo, seed_hi, seed_lo, seed_hi, seed_lo,
    );

    while i + 4 <= keys.len() {
        // Load 4 u64 = 8 u32 lanes (lo/hi interleaved).
        let ptr = keys.as_ptr().add(i) as *const __m256i;
        let v = _mm256_loadu_si256(ptr);
        let vec = _mm256_xor_si256(v, seed_vec);
        let mixed = mix32_vec(vec);
        let mut out_u32: [u32; 8] = [0; 8];
        _mm256_storeu_si256(out_u32.as_mut_ptr() as *mut __m256i, mixed);
        for j in 0..4 {
            let lo = out_u32[j * 2] as u64;
            let hi = out_u32[j * 2 + 1] as u64;
            out[i + j] = (lo << 32) | hi;
        }
        i += 4;
    }

    if i < keys.len() {
        crate::simd_hash::scalar::hash_u64_scalar(&keys[i..], seed, &mut out[i..]);
    }
}
