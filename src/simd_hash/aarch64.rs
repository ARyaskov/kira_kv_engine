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
#[inline]
pub unsafe fn hash_u64_neon(keys: &[u64], seed: u64, out: &mut [u64]) {
    crate::simd_hash::scalar::hash_u64_scalar(keys, seed, out);
}
