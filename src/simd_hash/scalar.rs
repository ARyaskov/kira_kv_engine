#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 32;
    x = x.wrapping_mul(0xd6e8feb86659fd93);
    x ^= x >> 32;
    x = x.wrapping_mul(0xd6e8feb86659fd93);
    x ^= x >> 32;
    x
}

pub fn hash_u64_scalar(keys: &[u64], seed: u64, out: &mut [u64]) {
    for (i, &k) in keys.iter().enumerate() {
        out[i] = mix64(k ^ seed);
    }
}

#[inline]
pub fn hash_u64_one(key: u64, seed: u64) -> u64 {
    mix64(key ^ seed)
}

/// Undo the canonical hash for an eight-byte key.
#[inline]
pub(crate) fn invert_hash_u64(mut hash: u64, seed: u64) -> u64 {
    // Multiplicative inverse of the mix64 constant modulo 2^64.
    const INVERSE: u64 = 0xcfee_444d_8b59_a89b;
    hash ^= hash >> 32;
    hash = hash.wrapping_mul(INVERSE);
    hash ^= hash >> 32;
    hash = hash.wrapping_mul(INVERSE);
    (hash ^ (hash >> 32)) ^ seed
}
