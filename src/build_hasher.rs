use std::hash::Hasher;

#[derive(Default)]
pub struct FastBuildHasher {
    state: u64,
}

impl Hasher for FastBuildHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state = fast_hash_bytes(bytes);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }
}

#[inline]
pub fn fast_hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ (bytes.len() as u64).wrapping_mul(0xA24B_1F6F);
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let mut chunk = [0u8; 8];
        chunk.copy_from_slice(&bytes[i..i + 8]);
        let v = u64::from_le_bytes(chunk);
        h ^= splitmix64(v);
        i += 8;
    }
    while i < bytes.len() {
        h ^= (bytes[i] as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h = splitmix64(h);
        i += 1;
    }
    splitmix64(h)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
