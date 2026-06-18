use core::arch::aarch64::{uint64x2_t, vdupq_n_u64, veorq_u64, vld1q_u64, vshrq_n_u64, vst1q_u64};

const MIX64_MUL: u64 = 0xd6e8_feb8_6659_fd93;

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn mul_const_u64x2(x: uint64x2_t, c: u64) -> uint64x2_t {
    // Stable Rust does not expose a portable AArch64 u64x2 lane multiply
    // intrinsic. Keep the vectorized load/store path and multiply lanes
    // explicitly until upstream provides a stable intrinsic.
    let mut lanes = [0u64; 2];
    vst1q_u64(lanes.as_mut_ptr(), x);
    lanes[0] = lanes[0].wrapping_mul(c);
    lanes[1] = lanes[1].wrapping_mul(c);
    vld1q_u64(lanes.as_ptr())
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn mix64_vec(mut x: uint64x2_t) -> uint64x2_t {
    x = veorq_u64(x, vshrq_n_u64(x, 32));
    x = mul_const_u64x2(x, MIX64_MUL);
    x = veorq_u64(x, vshrq_n_u64(x, 32));
    x = mul_const_u64x2(x, MIX64_MUL);
    veorq_u64(x, vshrq_n_u64(x, 32))
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
pub unsafe fn hash_u64_neon(keys: &[u64], seed: u64, out: &mut [u64]) {
    debug_assert_eq!(keys.len(), out.len());
    let n = keys.len();
    let seed_v = vdupq_n_u64(seed);
    let mut i = 0usize;
    while i + 2 <= n {
        let x = vld1q_u64(keys.as_ptr().add(i));
        let y = veorq_u64(x, seed_v);
        let h = mix64_vec(y);
        vst1q_u64(out.as_mut_ptr().add(i), h);
        i += 2;
    }
    for j in i..n {
        out[j] = crate::simd_hash::scalar::mix64(keys[j] ^ seed);
    }
}
