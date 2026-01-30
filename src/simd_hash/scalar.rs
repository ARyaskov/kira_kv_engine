#[inline]
fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x
}

pub fn hash_u64_scalar(keys: &[u64], seed: u64, out: &mut [u64]) {
    for (i, &k) in keys.iter().enumerate() {
        let v = k ^ seed;
        let lo = v as u32;
        let hi = (v >> 32) as u32;
        let h1 = mix32(lo);
        let h2 = mix32(hi);
        out[i] = ((h1 as u64) << 32) | (h2 as u64);
    }
}

#[inline]
pub fn hash_u64_one(key: u64, seed: u64) -> u64 {
    let v = key ^ seed;
    let lo = v as u32;
    let hi = (v >> 32) as u32;
    let h1 = mix32(lo);
    let h2 = mix32(hi);
    ((h1 as u64) << 32) | (h2 as u64)
}
