//! AES-round based hash for byte keys — the canonical hash of every non-8-byte key.
//!
//! Each AES round (`AESENC`) costs ~4 cycles on modern x86 and Apple/ARMv8 cores and
//! mixes all 128 bits, so short keys hash in ~10 cycles versus ~25 for wyhash.
//!
//! **The function is defined once, in terms of the AES round, and produces the same
//! value on every platform**:
//!
//! - x86_64 with AES-NI: `_mm_aesenc_si128`.
//! - aarch64 with the crypto extension: `vaesmcq_u8(vaeseq_u8(x, 0)) ^ k`, which is
//!   bit-identical to `AESENC` (ARM's `AESE` = AddRoundKey∘SubBytes∘ShiftRows,
//!   `AESMC` = MixColumns).
//! - everything else: a table-driven software round (slow, ~100 ns/key, but exact).
//!
//! An index serialized on one machine therefore looks up the same keys on any other.
//! Earlier versions fell back to wyhash on non-AES hosts, which silently broke every
//! index with byte keys moved between x86 and ARM.
//!
//! The hash is **not** cryptographically secure (no key schedule, few rounds); it is
//! only meant for in-process MPHF derivation.

/// 16-byte AES state in the layout the intrinsics use (byte `i` of the register).
type Block = [u8; 16];

#[inline(always)]
fn block_from_u64(lo: u64, hi: u64) -> Block {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&lo.to_le_bytes());
    b[8..].copy_from_slice(&hi.to_le_bytes());
    b
}

#[inline(always)]
fn low_u64(b: &Block) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}

#[inline(always)]
fn xor_block(a: &Block, b: &Block) -> Block {
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = a[i] ^ b[i];
    }
    out
}

/// One AES encryption round, `AESENC(state, round_key)` semantics.
trait AesEnc {
    /// # Safety
    /// Hardware implementations require the corresponding CPU feature.
    unsafe fn enc(state: Block, key: Block) -> Block;
}

// -----------------------------------------------------------------------------
// The hash itself, written once and monomorphized per backend.
// -----------------------------------------------------------------------------

#[inline(always)]
unsafe fn hash_u64_impl<E: AesEnc>(key: u64, seed: u64) -> u64 {
    unsafe {
        // Pack (key, seed) into one block; two rounds against fixed round keys.
        let block = block_from_u64(key, seed);
        let rk1 = block_from_u64(0xB492_5BB1_8B82_FBD7, 0xC3A5_C85C_97CB_3127);
        let rk2 = block_from_u64(0x9E37_79B9_7F4A_7C15, 0xCBF2_9CE4_8422_2325);
        let m1 = E::enc(block, rk1);
        let m2 = E::enc(m1, rk2);
        // Fold both u64 lanes into one to keep entropy from both halves.
        let lo = low_u64(&m2);
        let hi_block = E::enc(m2, block_from_u64(0, 0x1234_5678_9ABC_DEF0));
        let hi = low_u64(&hi_block);
        lo ^ hi.rotate_left(17)
    }
}

#[inline(always)]
unsafe fn hash_bytes_impl<E: AesEnc>(key: &[u8], seed: u64) -> u64 {
    unsafe {
        let len = key.len();
        if len == 0 {
            return splitmix64(seed);
        }
        if len <= 8 {
            let mut tail = [0u8; 8];
            tail[..len].copy_from_slice(key);
            let k = u64::from_le_bytes(tail);
            return hash_u64_impl::<E>(k ^ (len as u64), seed);
        }
        if len <= 16 {
            // Head + overlapping tail fill exactly 16 lanes.
            let mut block = [0u8; 16];
            block[..8].copy_from_slice(&key[..8]);
            block[8..].copy_from_slice(&key[len - 8..]);
            let rk1 = block_from_u64(seed ^ (len as u64), seed.rotate_left(17));
            let rk2 = block_from_u64(0x9E37_79B9_7F4A_7C15, 0xC3A5_C85C_97CB_3127);
            let m1 = E::enc(block, rk1);
            let m2 = E::enc(m1, rk2);
            let lo = low_u64(&m2);
            let hi = low_u64(&E::enc(m2, block_from_u64(0x1234_5678_9ABC_DEF0, 0)));
            return lo ^ hi.rotate_left(23);
        }
        // Long key: 16-byte chunks XOR-folded into the accumulator with one round each,
        // the 1..15-byte tail read as the overlapping last 16-byte window.
        let seed_v = block_from_u64(seed ^ (len as u64), seed);
        let mut acc = seed_v;
        let mut i = 0usize;
        while i + 16 <= len {
            let chunk: Block = key[i..i + 16].try_into().unwrap();
            acc = E::enc(xor_block(&acc, &chunk), seed_v);
            i += 16;
        }
        if i < len {
            let tail: Block = key[len - 16..].try_into().unwrap();
            acc = E::enc(xor_block(&acc, &tail), seed_v);
        }
        let finalize_key = block_from_u64(0xC3A5_C85C_97CB_3127, 0xCBF2_9CE4_8422_2325);
        let final_block = E::enc(acc, finalize_key);
        let lo = low_u64(&final_block);
        let hi = low_u64(&E::enc(final_block, seed_v));
        lo ^ hi.rotate_left(29)
    }
}

// -----------------------------------------------------------------------------
// Public entry points with one-time runtime dispatch.
// -----------------------------------------------------------------------------

/// Hash an arbitrary-length byte slice into a u64. Same value on every platform.
#[inline]
pub fn hash_bytes(key: &[u8], seed: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if hw_available() {
            // SAFETY: AES-NI presence checked (compile-time or cached runtime probe).
            return unsafe { x86::hash_bytes(key, seed) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if hw_available() {
            // SAFETY: the crypto extension presence was checked.
            return unsafe { arm::hash_bytes(key, seed) };
        }
    }
    // SAFETY: the software round has no hardware requirement.
    unsafe { hash_bytes_impl::<SoftAes>(key, seed) }
}

/// Hash a single u64 with two AES rounds. Same value on every platform.
#[inline]
pub fn hash_u64(key: u64, seed: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if hw_available() {
            return unsafe { x86::hash_u64(key, seed) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if hw_available() {
            return unsafe { arm::hash_u64(key, seed) };
        }
    }
    unsafe { hash_u64_impl::<SoftAes>(key, seed) }
}

/// Whether the hardware AES round is used on this host (diagnostic; the result of
/// the hash does not depend on it).
#[inline]
pub fn hw_available() -> bool {
    #[cfg(all(target_arch = "x86_64", target_feature = "aes"))]
    {
        true
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "aes")))]
    {
        std::arch::is_x86_feature_detected!("aes")
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        true
    }
    #[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
    {
        std::arch::is_aarch64_feature_detected!("aes")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Software AES round of `state` with `key`; exposed for cross-checking the
/// hardware backends in tests.
pub fn aes_round_soft(state: [u8; 16], key: [u8; 16]) -> [u8; 16] {
    unsafe { SoftAes::enc(state, key) }
}

/// Hardware AES round, `None` when the host has no AES instructions.
pub fn aes_round_hw(state: [u8; 16], key: [u8; 16]) -> Option<[u8; 16]> {
    if !hw_available() {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        Some(unsafe { x86::HwAes::enc(state, key) })
    }
    #[cfg(target_arch = "aarch64")]
    {
        Some(unsafe { arm::HwAes::enc(state, key) })
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        None
    }
}

// -----------------------------------------------------------------------------
// x86_64 AES-NI backend.
// -----------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{AesEnc, Block};
    use core::arch::x86_64::{__m128i, _mm_aesenc_si128, _mm_loadu_si128, _mm_storeu_si128};

    pub struct HwAes;

    impl AesEnc for HwAes {
        #[inline(always)]
        unsafe fn enc(state: Block, key: Block) -> Block {
            unsafe {
                let s = _mm_loadu_si128(state.as_ptr() as *const __m128i);
                let k = _mm_loadu_si128(key.as_ptr() as *const __m128i);
                let r = _mm_aesenc_si128(s, k);
                let mut out = [0u8; 16];
                _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, r);
                out
            }
        }
    }

    #[target_feature(enable = "aes,sse2")]
    pub unsafe fn hash_bytes(key: &[u8], seed: u64) -> u64 {
        unsafe { super::hash_bytes_impl::<HwAes>(key, seed) }
    }

    #[target_feature(enable = "aes,sse2")]
    pub unsafe fn hash_u64(key: u64, seed: u64) -> u64 {
        unsafe { super::hash_u64_impl::<HwAes>(key, seed) }
    }
}

// -----------------------------------------------------------------------------
// aarch64 crypto-extension backend.
// -----------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{AesEnc, Block};
    use core::arch::aarch64::{vaeseq_u8, vaesmcq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vst1q_u8};

    pub struct HwAes;

    impl AesEnc for HwAes {
        #[inline(always)]
        unsafe fn enc(state: Block, key: Block) -> Block {
            unsafe {
                let s = vld1q_u8(state.as_ptr());
                let k = vld1q_u8(key.as_ptr());
                // AESE with a zero key = ShiftRows(SubBytes(s)); AESMC = MixColumns;
                // the final XOR is AddRoundKey — exactly x86's AESENC.
                let r = veorq_u8(vaesmcq_u8(vaeseq_u8(s, vdupq_n_u8(0))), k);
                let mut out = [0u8; 16];
                vst1q_u8(out.as_mut_ptr(), r);
                out
            }
        }
    }

    #[target_feature(enable = "aes")]
    pub unsafe fn hash_bytes(key: &[u8], seed: u64) -> u64 {
        unsafe { super::hash_bytes_impl::<HwAes>(key, seed) }
    }

    #[target_feature(enable = "aes")]
    pub unsafe fn hash_u64(key: u64, seed: u64) -> u64 {
        unsafe { super::hash_u64_impl::<HwAes>(key, seed) }
    }
}

// -----------------------------------------------------------------------------
// Portable software round (FIPS-197), used where no AES instructions exist.
// -----------------------------------------------------------------------------

struct SoftAes;

const fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    let mut i = 0;
    while i < 8 {
        if b & 1 != 0 {
            p ^= a;
        }
        let carry = a & 0x80;
        a <<= 1;
        if carry != 0 {
            a ^= 0x1B;
        }
        b >>= 1;
        i += 1;
    }
    p
}

const fn gf_inv(x: u8) -> u8 {
    // x^254 in GF(2^8) is the multiplicative inverse (0 maps to 0).
    let mut result = 1u8;
    let mut base = x;
    let mut e = 254u32;
    while e > 0 {
        if e & 1 != 0 {
            result = gf_mul(result, base);
        }
        base = gf_mul(base, base);
        e >>= 1;
    }
    result
}

const fn build_sbox() -> [u8; 256] {
    let mut sbox = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let inv = gf_inv(i as u8);
        let s = inv
            ^ inv.rotate_left(1)
            ^ inv.rotate_left(2)
            ^ inv.rotate_left(3)
            ^ inv.rotate_left(4)
            ^ 0x63;
        sbox[i] = s;
        i += 1;
    }
    sbox
}

static SBOX: [u8; 256] = build_sbox();

impl AesEnc for SoftAes {
    #[inline]
    unsafe fn enc(state: Block, key: Block) -> Block {
        // State is column-major: byte r + 4c is row r of column c.
        let mut t = [0u8; 16];
        // SubBytes + ShiftRows (row r rotates left by r).
        for c in 0..4 {
            for r in 0..4 {
                t[r + 4 * c] = SBOX[state[r + 4 * ((c + r) & 3)] as usize];
            }
        }
        // MixColumns + AddRoundKey.
        let mut out = [0u8; 16];
        for c in 0..4 {
            let a0 = t[4 * c];
            let a1 = t[4 * c + 1];
            let a2 = t[4 * c + 2];
            let a3 = t[4 * c + 3];
            let x = |v: u8| gf_mul(v, 2);
            out[4 * c] = x(a0) ^ x(a1) ^ a1 ^ a2 ^ a3;
            out[4 * c + 1] = a0 ^ x(a1) ^ x(a2) ^ a2 ^ a3;
            out[4 * c + 2] = a0 ^ a1 ^ x(a2) ^ x(a3) ^ a3;
            out[4 * c + 3] = x(a0) ^ a0 ^ a1 ^ a2 ^ x(a3);
        }
        for i in 0..16 {
            out[i] ^= key[i];
        }
        out
    }
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
