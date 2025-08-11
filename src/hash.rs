use xxhash_rust::xxh3::xxh3_64_with_seed;

#[derive(Clone, Copy, Debug)]
pub struct KeyHash {
    pub h1: u64, // bucket selector
    pub h2: u64, // base shift
    pub h3: u64, // multiplier for displacement (spreads values)
}

impl KeyHash {
    #[inline]
    pub fn from_key(bytes: &[u8], salt: u64) -> Self {
        let s1 = salt ^ 0x9E37_79B9_7F4A_7C15;
        let s2 = salt.wrapping_mul(0xA24B_1F6F);
        let s3 = salt ^ 0x853C_49E6_0A6C_9D39;
        Self {
            h1: xxh3_64_with_seed(bytes, s1),
            h2: xxh3_64_with_seed(bytes, s2),
            h3: xxh3_64_with_seed(bytes, s3),
        }
    }

    #[inline]
    pub fn bucket(&self, buckets: u64) -> usize {
        (self.h1 % buckets.max(1)) as usize
    }

    /// Position for given displacement `d` and size `n`:
    /// pos = (h2 + d * h3) % n
    #[inline]
    pub fn place(&self, n: u64, d: u64) -> usize {
        let mixed = self.h2.wrapping_add(d.wrapping_mul(self.h3));
        (mixed % n.max(1)) as usize
    }
}