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
