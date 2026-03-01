#[inline(always)]
pub fn canonical_hash_bytes(key: &[u8], seed: u64) -> u64 {
    if key.len() == 8 {
        return crate::simd_hash::hash_u64_one(load_u64_le(key, 0), seed);
    }
    if key.len() <= 32 {
        return canonical_hash_small(key, seed);
    }
    wyhash::wyhash(key, seed)
}

#[inline(always)]
fn canonical_hash_small(key: &[u8], seed: u64) -> u64 {
    let len = key.len();
    let mut acc = splitmix64(seed ^ (len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

    if len >= 16 {
        let a = load_u64_le(key, 0);
        let b = load_u64_le(key, len - 8);
        acc ^= splitmix64(a ^ seed.rotate_left(17));
        acc = splitmix64(acc ^ b.rotate_left(23));
        if len > 16 {
            let mid = (len >> 1) - 4;
            let c = load_u64_le(key, mid);
            acc ^= splitmix64(c ^ seed.rotate_left(41));
        }
        return splitmix64(acc);
    }

    if len >= 8 {
        let a = load_u64_le(key, 0);
        acc ^= splitmix64(a ^ seed.rotate_left(7));
        return splitmix64(acc ^ (len as u64).wrapping_mul(0xA24B_1F6F_DA39_2B31));
    }

    let mut tail = 0u64;
    for (i, &b) in key.iter().enumerate() {
        tail |= (b as u64) << (i * 8);
    }
    splitmix64(acc ^ tail ^ 0xD6E8_FD9D_50E9_4A4D)
}

#[inline(always)]
fn load_u64_le(key: &[u8], offset: usize) -> u64 {
    // Bounds are guaranteed by callers based on key length checks.
    let v = unsafe { std::ptr::read_unaligned(key.as_ptr().add(offset) as *const u64) };
    u64::from_le(v)
}

#[inline(always)]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
