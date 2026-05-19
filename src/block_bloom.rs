//! Split-block Bloom filter (Putze/Sanders/Singler 2007, popularised by Impala/Parquet).
//!
//! Each query touches **exactly one 64-byte cache line** — eight 64-bit words, 1 bit set per
//! word for a total of 8 bits per element. Roughly equivalent FPR to a classic Bloom with 8
//! hash functions, but one cache miss instead of eight independent random loads.
//!
//! On AVX2-capable x86_64 the lookup compiles to ~12 instructions: one `vmovdqu` for the
//! block, one `vpcmpeqq` against the broadcast bitmask, and a final mask test.

const BLOCK_WORDS: usize = 8; // 8 * 8 B = 64 B = one cache line.
const BITS_PER_KEY: f64 = 11.0; // tunes false positive rate; ~0.4% @ 11 bits, ~0.04% @ 16.

// Serde/rkyv derive omitted: words is HugeVec which doesn't trivially serialize.
// Manual serialization via write_to / read_from below.
#[derive(Debug, Clone)]
pub struct BlockBloom {
    seed: u64,
    /// 64-bit words; layout is [block0_word0..block0_word7, block1_word0..block1_word7, ...].
    /// Length is always a multiple of BLOCK_WORDS. Hugepage-backed for ≥ 1 MB filters
    /// (100M-key index → 16 MB filter → ~7 TLB pages instead of ~4000).
    words: crate::hugepage::HugeVec<u64>,
}

impl BlockBloom {
    pub fn build_from_prehashed(hashes: &[u64]) -> Self {
        Self::build_with_seed(hashes, 0xC1B5_4A32_D192_ED03)
    }

    pub fn build_from_u64(keys: &[u64], seed: u64) -> Self {
        let mut hashes = vec![0u64; keys.len()];
        crate::simd_hash::hash_u64(keys, seed, &mut hashes);
        Self::build_with_seed(&hashes, seed)
    }

    pub fn build_from_bytes(keys: &[Vec<u8>], seed: u64) -> Self {
        let mut hashes = Vec::with_capacity(keys.len());
        for k in keys {
            hashes.push(wyhash::wyhash(k.as_slice(), seed));
        }
        Self::build_with_seed(&hashes, seed)
    }

    fn build_with_seed(hashes: &[u64], seed: u64) -> Self {
        let n = hashes.len().max(1);
        let total_bits = ((n as f64) * BITS_PER_KEY).ceil() as usize;
        let mut blocks = (total_bits / (BLOCK_WORDS * 64)).max(1);
        // Round up to a power of two so the block index can be derived with a multiply-high
        // reduction without bias.
        blocks = blocks.next_power_of_two();
        // Sequential build. A parallel per-chunk-shadow variant was tried but lost to alloc
        // + memset of 8 × 16 MB shadow buffers + the OR-reduce on Windows. The single-pass
        // loop is bandwidth-bound at ~200 ms for N=10M which is already small relative to
        // the pilot search.
        let mut words = crate::hugepage::HugeVec::<u64>::zeroed(blocks * BLOCK_WORDS);
        {
            let slice = words.as_mut_slice();
            for &h in hashes {
                let (block, mask) = block_and_mask(h, blocks);
                let base = block * BLOCK_WORDS;
                for w in 0..BLOCK_WORDS {
                    unsafe { *slice.get_unchecked_mut(base + w) |= mask[w] };
                }
            }
        }

        Self { seed, words }
    }

    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[inline]
    pub fn hash_u64(&self, key: u64) -> u64 {
        crate::simd_hash::hash_u64_one(key, self.seed)
    }

    #[inline]
    pub fn hash_bytes(&self, key: &[u8]) -> u64 {
        wyhash::wyhash(key, self.seed)
    }

    /// Pointer to the block (64 bytes) that `hash` would land in. Used by the pipelined
    /// lookup to issue an `_mm_prefetch` ahead of the actual contains check.
    #[inline]
    pub fn block_ptr(&self, hash: u64) -> *const u64 {
        let words = self.words.as_slice();
        let blocks = words.len() / BLOCK_WORDS;
        let (block, _) = block_and_mask(hash, blocks);
        unsafe { words.as_ptr().add(block * BLOCK_WORDS) }
    }

    #[inline]
    pub fn contains_hash(&self, hash: u64) -> bool {
        let words = self.words.as_slice();
        let blocks = words.len() / BLOCK_WORDS;
        let (block, mask) = block_and_mask(hash, blocks);
        let base = block * BLOCK_WORDS;
        // 64 bytes — one cache line — touched per query.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            return contains_hash_avx2(words.as_ptr().add(base), &mask);
        }
        #[allow(unreachable_code)]
        {
            for w in 0..BLOCK_WORDS {
                let word = unsafe { *words.get_unchecked(base + w) };
                if word & mask[w] != mask[w] {
                    return false;
                }
            }
            true
        }
    }

    #[inline]
    pub fn contains_u64(&self, key: u64) -> bool {
        let h = self.hash_u64(key);
        self.contains_hash(h)
    }

    #[inline]
    pub fn contains_bytes(&self, key: &[u8]) -> bool {
        let h = self.hash_bytes(key);
        self.contains_hash(h)
    }

    pub fn memory_usage(&self) -> usize {
        std::mem::size_of_val(self) + self.words.memory_usage()
    }

    /// Export the raw 64-bit word array for upload to an external
    /// accelerator. Length is always a multiple of `BLOCK_WORDS` (8) and
    /// `len() / 8` is always a power of two. The on-device kernel uses
    /// the same `block_and_mask` formula as the CPU path:
    ///
    /// ```text
    /// block_idx = (((hash >> 32) * blocks) >> 32)
    /// for w in 0..8: bit = ((hash & 0xFFFF_FFFF) * SALT[w]) >> 27) & 0x3F
    ///                mask[w] = 1u64 << bit
    /// contains = all of (words[block_idx*8 + w] & mask[w]) == mask[w]
    /// ```
    ///
    /// SALT constants live in `block_bloom.rs::block_and_mask` and must
    /// be replicated bit-exact in the kernel.
    pub fn export_words(&self) -> Vec<u64> {
        self.words.as_slice().to_vec()
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&(self.words.len() as u64).to_le_bytes());
        for &w in self.words.as_slice() {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }

    pub fn read_from(buf: &[u8], pos: &mut usize) -> Option<Self> {
        if *pos + 16 > buf.len() {
            return None;
        }
        let mut a = [0u8; 8];
        a.copy_from_slice(&buf[*pos..*pos + 8]);
        let seed = u64::from_le_bytes(a);
        a.copy_from_slice(&buf[*pos + 8..*pos + 16]);
        let len = u64::from_le_bytes(a) as usize;
        *pos += 16;
        if *pos + len * 8 > buf.len() {
            return None;
        }
        let mut words = crate::hugepage::HugeVec::<u64>::zeroed(len);
        {
            let slice = words.as_mut_slice();
            for w in slice.iter_mut() {
                a.copy_from_slice(&buf[*pos..*pos + 8]);
                *w = u64::from_le_bytes(a);
                *pos += 8;
            }
        }
        Some(Self { seed, words })
    }
}

#[inline]
fn block_and_mask(hash: u64, blocks: usize) -> (usize, [u64; BLOCK_WORDS]) {
    // High 32 bits choose the block (multiply-high reduction); low 32 bits seed the 8 lane bits.
    let block = (((hash >> 32) as u128 * blocks as u128) >> 32) as usize;
    let seed = hash & 0xFFFF_FFFF;
    // Salt constants from Impala's split-block-Bloom implementation; chosen so that the
    // 8 derived 5-bit indices avoid pairwise correlation.
    const SALT: [u32; 8] = [
        0x47b6_137b, 0x4476_8924, 0x1820_5237, 0x2384_8965, 0x8e6e_2354, 0x0f7c_c9b6, 0xe43d_5fa5,
        0xa4d5_2dc1,
    ];
    let mut mask = [0u64; BLOCK_WORDS];
    for (i, &s) in SALT.iter().enumerate() {
        let bit = ((seed as u32).wrapping_mul(s) >> 27) & 0x3F;
        mask[i] = 1u64 << bit;
    }
    (block, mask)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn contains_hash_avx2(words_ptr: *const u64, mask: &[u64; BLOCK_WORDS]) -> bool {
    use core::arch::x86_64::{
        _mm256_and_si256, _mm256_cmpeq_epi64, _mm256_loadu_si256, _mm256_movemask_epi8,
    };
    // Load two 32-byte halves.
    let block_lo = _mm256_loadu_si256(words_ptr as *const _);
    let block_hi = _mm256_loadu_si256(words_ptr.add(4) as *const _);
    let mask_lo = _mm256_loadu_si256(mask.as_ptr() as *const _);
    let mask_hi = _mm256_loadu_si256(mask.as_ptr().add(4) as *const _);
    let and_lo = _mm256_and_si256(block_lo, mask_lo);
    let and_hi = _mm256_and_si256(block_hi, mask_hi);
    let eq_lo = _mm256_cmpeq_epi64(and_lo, mask_lo);
    let eq_hi = _mm256_cmpeq_epi64(and_hi, mask_hi);
    // 8 lanes; we need ALL lanes equal. movemask_epi8 == 0xFFFF_FFFF iff every byte set.
    let mm_lo = _mm256_movemask_epi8(eq_lo) as u32;
    let mm_hi = _mm256_movemask_epi8(eq_hi) as u32;
    mm_lo == 0xFFFF_FFFF && mm_hi == 0xFFFF_FFFF
}

