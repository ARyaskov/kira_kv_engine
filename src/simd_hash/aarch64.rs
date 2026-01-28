use core::arch::aarch64::{
    uint32x4_t, vdupq_n_u32, veorq_u32, vld1q_u32, vmulq_u32, vshrq_n_u32, vst1q_u32,
};

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn mix32_vec(mut v: uint32x4_t) -> uint32x4_t {
    v = veorq_u32(v, vshrq_n_u32(v, 16));
    v = vmulq_u32(v, vdupq_n_u32(0x7FEB_352D));
    v = veorq_u32(v, vshrq_n_u32(v, 15));
    v = vmulq_u32(v, vdupq_n_u32(0x846C_A68B));
    veorq_u32(v, vshrq_n_u32(v, 16))
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
pub unsafe fn hash_u64_neon(keys: &[u64], seed: u64, out: &mut [u64]) {
    let mut i = 0usize;
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;

    while i + 2 <= keys.len() {
        let ptr = keys.as_ptr().add(i) as *const u32;
        let mut v = vld1q_u32(ptr);
        let mut tmp = [seed_lo, seed_hi, seed_lo, seed_hi];
        let seedv = vld1q_u32(tmp.as_ptr());
        v = veorq_u32(v, seedv);
        let mixed = mix32_vec(v);
        let mut out_u32 = [0u32; 4];
        vst1q_u32(out_u32.as_mut_ptr(), mixed);
        let lo0 = out_u32[0] as u64;
        let hi0 = out_u32[1] as u64;
        let lo1 = out_u32[2] as u64;
        let hi1 = out_u32[3] as u64;
        out[i] = (lo0 << 32) | hi0;
        out[i + 1] = (lo1 << 32) | hi1;
        i += 2;
    }

    if i < keys.len() {
        crate::simd_hash::scalar::hash_u64_scalar(&keys[i..], seed, &mut out[i..]);
    }
}
