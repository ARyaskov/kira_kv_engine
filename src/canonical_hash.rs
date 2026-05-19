//! Canonical key→u64 hash used at every Index build/lookup boundary.
//!
//! Dispatches to:
//! - `simd_hash::hash_u64_one` for the 8-byte fast path.
//! - `aes_hash::hash_bytes` for variable-length byte keys (AES-NI on x86,
//!   wyhash fallback otherwise).

#[inline(always)]
pub fn canonical_hash_bytes(key: &[u8], seed: u64) -> u64 {
    if key.len() == 8 {
        return crate::simd_hash::hash_u64_one(load_u64_le(key, 0), seed);
    }
    crate::aes_hash::hash_bytes(key, seed)
}

#[inline(always)]
fn load_u64_le(key: &[u8], offset: usize) -> u64 {
    // SAFETY: callers must ensure `offset + 8 <= key.len()`.
    let v = unsafe { std::ptr::read_unaligned(key.as_ptr().add(offset) as *const u64) };
    u64::from_le(v)
}
