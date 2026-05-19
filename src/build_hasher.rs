// FastBuildHasher is referenced via BuildHasherDefault in ptrhash.rs::Builder::build,
// which is itself part of the public API but not on the hot bench path.
#![allow(dead_code)]
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
    // wyhash is well-mixed, branch-light, and on x86_64-v3 vectorizes better than
    // the splitmix loop it replaced. Same seed across the crate so hashes stay stable
    // between callers that rely on equal-bytes-equal-hash.
    wyhash::wyhash(bytes, 0xA076_1D64_78BD_642F)
}
