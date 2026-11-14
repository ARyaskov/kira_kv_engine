//! Canonical key→u64 hash used at every Index build/lookup boundary.
//!
//! Dispatches to:
//! - `simd_hash::hash_u64_one` for the 8-byte fast path (`mix64`, a bijection).
//! - length-delimited AES hashing for variable-length byte keys.
//!
//! Both are platform-independent: the same key and seed hash to the same value on
//! x86_64, aarch64 and everything else, so serialized indexes are portable.

#[inline(always)]
pub fn canonical_hash_bytes(key: &[u8], seed: u64) -> u64 {
    CanonicalHashVersion::Current.hash(key, seed)
}

/// The on-disk identifier follows the hash used to build the immutable tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CanonicalHashVersion {
    Legacy = crate::wire::KIND_INDEX,
    Current = crate::wire::KIND_INDEX_V2,
}

impl CanonicalHashVersion {
    pub(crate) fn from_id(id: u8) -> Option<Self> {
        match id {
            crate::wire::KIND_INDEX => Some(Self::Legacy),
            crate::wire::KIND_INDEX_V2 => Some(Self::Current),
            _ => None,
        }
    }

    #[inline(always)]
    pub(crate) fn hash(self, key: &[u8], seed: u64) -> u64 {
        if key.len() == 8 {
            return crate::simd_hash::hash_u64_one(load_u64_le(key, 0), seed);
        }
        match self {
            Self::Legacy => crate::aes_hash::hash_bytes(key, seed),
            Self::Current => crate::aes_hash::hash_bytes_v2(key, seed),
        }
    }

    /// Preserve scalar hash values while hoisting version and CPU dispatch out
    /// of the individual key loop.
    pub(crate) fn hash_batch(self, keys: &[&[u8]], seed: u64, out: &mut [u64]) {
        match self {
            Self::Legacy => crate::aes_hash::hash_canonical_batch::<false>(keys, seed, out),
            Self::Current => crate::aes_hash::hash_canonical_batch::<true>(keys, seed, out),
        }
    }
}

#[inline(always)]
fn load_u64_le(key: &[u8], offset: usize) -> u64 {
    // SAFETY: callers must ensure `offset + 8 <= key.len()`.
    let v = unsafe { std::ptr::read_unaligned(key.as_ptr().add(offset) as *const u64) };
    u64::from_le(v)
}
