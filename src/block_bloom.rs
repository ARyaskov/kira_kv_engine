//! Split-block Bloom filter (Putze/Sanders/Singler 2007, popularised by Impala/Parquet).
//!
//! Each query touches **exactly one 64-byte cache line** — eight 64-bit words, 1 bit set per
//! word for a total of 8 bits per element. Roughly equivalent FPR to a classic Bloom with 8
//! hash functions, but one cache miss instead of eight independent random loads.
//!
//! On AVX2-capable x86_64 the lookup compiles to ~12 instructions: one `vmovdqu` for the
//! block, one `vpcmpeqq` against the broadcast bitmask, and a final mask test.
//!
//! ## Bit-index width (format v2)
//!
//! Filters built before the partitioned build derived each lane's bit index with
//! `(x * SALT) >> 27` — 5 bits, i.e. only the low half of every 64-bit word was ever
//! used, so the nominal 11 bits/key were effectively 5.5 (measured ~10% FPR at 100K keys
//! without the power-of-two size rounding that used to hide it). Current filters use
//! `>> 26` (6 bits, all 64 bits of the word) and mark that in the serialized length word
//! (bit 63); old blobs load with the old shift, so every stored filter keeps answering
//! exactly as it did.

const BLOCK_WORDS: usize = 8; // 8 * 8 B = 64 B = one cache line.
const BITS_PER_KEY: f64 = 11.0; // ~0.5% FPR with 6-bit lane indices.

/// Bit-index shift of filters built by this version: 6-bit lane indices.
const BIT_SHIFT_V2: u32 = 26;
/// Bit-index shift of legacy filters: 5-bit lane indices (half of every word unused).
const BIT_SHIFT_LEGACY: u32 = 27;
/// Set on the serialized word-count to mark a v2 (6-bit) filter.
const LEN_FLAG_V2: u64 = 1 << 63;

// Serde/rkyv derive omitted: words is HugeVec which doesn't trivially serialize.
// Manual serialization via write_to / read_from below.
#[derive(Debug, Clone)]
pub struct BlockBloom {
    seed: u64,
    /// 64-bit words; layout is [block0_word0..block0_word7, block1_word0..block1_word7, ...].
    /// Length is always a multiple of BLOCK_WORDS. Hugepage-backed for ≥ 1 MB filters
    /// (100M-key index → 137 MB filter → ~70 TLB pages instead of ~34000).
    words: crate::hugepage::HugeVec<u64>,
    /// `BIT_SHIFT_V2` for filters built here, `BIT_SHIFT_LEGACY` for deserialized old ones.
    bit_shift: u32,
}

impl BlockBloom {
    pub fn build_from_prehashed(hashes: &[u64]) -> Self {
        Self::build_with_seed(hashes, 0xC1B5_4A32_D192_ED03, None)
    }

    /// Like [`BlockBloom::build_from_prehashed`], but the caller supplies the scratch
    /// buffer (`grouped.len() == hashes.len()`). On return `grouped` holds a permutation
    /// of `hashes` (grouped by filter window when the parallel builder ran, a plain copy
    /// otherwise), so the caller can keep using it as an alternative copy of the hashes
    /// instead of allocating — at 100M keys that is 800 MB of fresh pages not faulted in.
    pub fn build_from_prehashed_into(hashes: &[u64], grouped: &mut [u64]) -> Self {
        assert_eq!(grouped.len(), hashes.len(), "scratch must match the hash count");
        Self::build_with_seed(hashes, 0xC1B5_4A32_D192_ED03, Some(grouped))
    }

    pub fn build_from_u64(keys: &[u64], seed: u64) -> Self {
        let mut hashes = vec![0u64; keys.len()];
        crate::simd_hash::hash_u64(keys, seed, &mut hashes);
        Self::build_with_seed(&hashes, seed, None)
    }

    pub fn build_from_bytes(keys: &[Vec<u8>], seed: u64) -> Self {
        let mut hashes = Vec::with_capacity(keys.len());
        for k in keys {
            hashes.push(wyhash::wyhash(k.as_slice(), seed));
        }
        Self::build_with_seed(&hashes, seed, None)
    }

    fn build_with_seed(hashes: &[u64], seed: u64, grouped: Option<&mut [u64]>) -> Self {
        let n = hashes.len().max(1);
        let total_bits = ((n as f64) * BITS_PER_KEY).ceil() as usize;
        // The block index is a multiply-high reduction of the top 32 hash bits, which is
        // unbiased for *any* block count — no power-of-two rounding (that used to cost up
        // to 2× filter memory: 100M keys → 268 MB instead of 137 MB). Blocks are rounded
        // up to a multiple of `BLOOM_RANGES` so the parallel builder can split the filter
        // into equal windows whose index is simply the top 8 bits of the hash.
        let blocks = (total_bits / (BLOCK_WORDS * 64)).max(1).next_multiple_of(BLOOM_RANGES);
        let mut words = crate::hugepage::HugeVec::<u64>::zeroed(blocks * BLOCK_WORDS);
        let bit_shift = BIT_SHIFT_V2;

        // Parallel build. The filter is split into 256 windows; hashes are radix-grouped
        // by window (two parallel passes: histogram + scatter, both sequential sweeps),
        // then every window is filled by one task whose random writes never leave its own
        // window (256 KB at 10M keys, 2.7 MB at 100M). No atomics, no shadow filters,
        // no reduce.
        #[cfg(feature = "parallel")]
        {
            if rayon::current_num_threads() > 1 && hashes.len() >= PARALLEL_MIN_KEYS {
                match grouped {
                    Some(g) => build_parallel_ranges(hashes, words.as_mut_slice(), blocks, bit_shift, g),
                    None => {
                        let mut g = crate::hugepage::HugeVec::<u64>::zeroed(hashes.len());
                        build_parallel_ranges(
                            hashes,
                            words.as_mut_slice(),
                            blocks,
                            bit_shift,
                            g.as_mut_slice(),
                        );
                    }
                }
                return Self { seed, words, bit_shift };
            }
        }

        {
            let slice = words.as_mut_slice();
            for &h in hashes {
                let (block, mask) = block_and_mask(h, blocks, bit_shift);
                let base = block * BLOCK_WORDS;
                for w in 0..BLOCK_WORDS {
                    unsafe { *slice.get_unchecked_mut(base + w) |= mask[w] };
                }
            }
            // Keep the contract of `build_from_prehashed_into`: `grouped` is a permutation
            // of `hashes` (here the identity).
            if let Some(g) = grouped {
                g.copy_from_slice(hashes);
            }
        }

        Self { seed, words, bit_shift }
    }

    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Shift used to derive the 8 lane bit indices (`26` for current filters, `27` for
    /// legacy ones). Needed by off-host kernels; see [`BlockBloom::export_words`].
    #[inline]
    pub fn bit_shift(&self) -> u32 {
        self.bit_shift
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
        let block = block_of(hash, blocks);
        unsafe { words.as_ptr().add(block * BLOCK_WORDS) }
    }

    #[inline]
    pub fn contains_hash(&self, hash: u64) -> bool {
        let words = self.words.as_slice();
        let blocks = words.len() / BLOCK_WORDS;
        let (block, mask) = block_and_mask(hash, blocks, self.bit_shift);
        let base = block * BLOCK_WORDS;
        // 64 bytes — one cache line — touched per query.
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx2") || std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 verified at compile time or by the cached runtime probe;
                // `base + 8 ≤ words.len()` by construction of `block_of`.
                return unsafe { contains_hash_avx2(words.as_ptr().add(base), &mask) };
            }
        }
        for w in 0..BLOCK_WORDS {
            let word = unsafe { *words.get_unchecked(base + w) };
            if word & mask[w] != mask[w] {
                return false;
            }
        }
        true
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
    /// `len() / 8` is always a multiple of 256. The on-device kernel uses
    /// the same `block_and_mask` formula as the CPU path:
    ///
    /// ```text
    /// block_idx = (((hash >> 32) * blocks) >> 32)
    /// for w in 0..8: bit = ((hash & 0xFFFF_FFFF) * SALT[w]) >> bit_shift) & 0x3F
    ///                mask[w] = 1u64 << bit
    /// contains = all of (words[block_idx*8 + w] & mask[w]) == mask[w]
    /// ```
    ///
    /// `bit_shift` is [`BlockBloom::bit_shift`] (26 for current filters, 27 for legacy).
    /// SALT constants live in `block_bloom.rs::mask_for` and must be replicated
    /// bit-exact in the kernel.
    pub fn export_words(&self) -> Vec<u64> {
        self.words.as_slice().to_vec()
    }

    /// Layout: `[seed u64][len u64][words…]`. Bit 63 of `len` marks a v2 (6-bit lane
    /// index) filter; legacy blobs have it clear and are read with the 5-bit formula.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.seed.to_le_bytes());
        let flag = if self.bit_shift == BIT_SHIFT_V2 { LEN_FLAG_V2 } else { 0 };
        out.extend_from_slice(&((self.words.len() as u64) | flag).to_le_bytes());
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
        let raw_len = u64::from_le_bytes(a);
        let bit_shift = if raw_len & LEN_FLAG_V2 != 0 { BIT_SHIFT_V2 } else { BIT_SHIFT_LEGACY };
        let len = (raw_len & !LEN_FLAG_V2) as usize;
        *pos += 16;
        // At least one block (the lookup indexes block 0 unconditionally), whole
        // blocks only, and the payload must fit — with overflow-safe arithmetic so a
        // huge length word cannot slip past the check and trigger a giant allocation.
        if len < BLOCK_WORDS || len % BLOCK_WORDS != 0 {
            return None;
        }
        let payload = len.checked_mul(8)?;
        if payload > buf.len() - *pos {
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
        Some(Self { seed, words, bit_shift })
    }
}

/// Below this many keys the parallel build isn't worth the extra passes.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_KEYS: usize = 1 << 18;
/// Windows of the parallel builder; `blocks` is always a multiple of this so the window
/// of a hash is its top 8 bits.
const BLOOM_RANGES: usize = 256;

/// Raw pointer shared across rayon workers; every worker writes disjoint positions.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct SyncPtr(*mut u64);
#[cfg(feature = "parallel")]
unsafe impl Send for SyncPtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for SyncPtr {}

#[cfg(feature = "parallel")]
impl SyncPtr {
    /// Accessor on purpose: closures must capture the whole `SyncPtr` (`Sync`), not
    /// the raw pointer field edition-2021 precise capture would pick otherwise.
    #[inline(always)]
    fn get(self) -> *mut u64 {
        self.0
    }
}

/// Three parallel passes: histogram of window ids per chunk, scatter of the hashes
/// grouped by window, then one task per window OR-ing its own slice of the filter.
#[cfg(feature = "parallel")]
fn build_parallel_ranges(
    hashes: &[u64],
    words: &mut [u64],
    blocks: usize,
    bit_shift: u32,
    grouped: &mut [u64],
) {
    use rayon::prelude::*;
    debug_assert_eq!(blocks % BLOOM_RANGES, 0);
    debug_assert_eq!(grouped.len(), hashes.len());
    let n = hashes.len();
    // Sequential first touch of the scratch buffer — see `hugepage::prefault`.
    crate::hugepage::prefault(grouped);
    let ranges = BLOOM_RANGES;
    let range_blocks = blocks / ranges;
    // With `blocks = 256·k`, `block / k == (hash >> 32) >> 24` exactly (nested-floor
    // identity), so the window of a hash is its top 8 bits.
    let range_of = |h: u64| -> usize { (h >> 56) as usize };

    // Pass 1: per-chunk histograms of window ids.
    let chunks = (rayon::current_num_threads() * 2).clamp(1, 64);
    let chunk_len = n.div_ceil(chunks);
    let mut hist = vec![0usize; chunks * ranges];
    hist.par_chunks_mut(ranges).enumerate().for_each(|(c, row)| {
        let lo = (c * chunk_len).min(n);
        let hi = (lo + chunk_len).min(n);
        for &h in &hashes[lo..hi] {
            row[range_of(h)] += 1;
        }
    });

    // Prefix sums: window offsets, then per-chunk write cursors (in place of `hist`).
    let mut range_off = vec![0usize; ranges + 1];
    for r in 0..ranges {
        let mut s = 0usize;
        for c in 0..chunks {
            s += hist[c * ranges + r];
        }
        range_off[r + 1] = range_off[r] + s;
    }
    for r in 0..ranges {
        let mut running = range_off[r];
        for c in 0..chunks {
            let v = hist[c * ranges + r];
            hist[c * ranges + r] = running;
            running += v;
        }
    }

    // Pass 2: scatter hashes grouped by window.
    let gp = SyncPtr(grouped.as_mut_ptr());
    hist.par_chunks_mut(ranges).enumerate().for_each(|(c, cursors)| {
        let lo = (c * chunk_len).min(n);
        let hi = (lo + chunk_len).min(n);
        for &h in &hashes[lo..hi] {
            let r = range_of(h);
            let pos = cursors[r];
            cursors[r] = pos + 1;
            // SAFETY: `pos` is unique across all chunks (disjoint cursor ranges).
            unsafe { *gp.get().add(pos) = h };
        }
    });

    // Pass 3: fill every window from its group. Random writes stay inside the window.
    words
        .par_chunks_mut(range_blocks * BLOCK_WORDS)
        .enumerate()
        .for_each(|(r, win)| {
            let blk_lo = r * range_blocks;
            for &h in &grouped[range_off[r]..range_off[r + 1]] {
                let (block, mask) = block_and_mask(h, blocks, bit_shift);
                let base = (block - blk_lo) * BLOCK_WORDS;
                for w in 0..BLOCK_WORDS {
                    // SAFETY: `block ∈ [blk_lo, blk_lo + range_blocks)` by grouping.
                    unsafe { *win.get_unchecked_mut(base + w) |= mask[w] };
                }
            }
        });
}

/// High 32 bits choose the block (multiply-high reduction, unbiased for any block count).
#[inline(always)]
fn block_of(hash: u64, blocks: usize) -> usize {
    (((hash >> 32) as u128 * blocks as u128) >> 32) as usize
}

#[inline]
fn block_and_mask(hash: u64, blocks: usize, bit_shift: u32) -> (usize, [u64; BLOCK_WORDS]) {
    (block_of(hash, blocks), mask_for(hash, bit_shift))
}

/// The low 32 hash bits seed the 8 lane bits. Salt constants from Impala's split-block
/// Bloom implementation, chosen so the derived indices avoid pairwise correlation;
/// `bit_shift` selects 6-bit (26, current) or 5-bit (27, legacy) indices.
#[inline(always)]
fn mask_for(hash: u64, bit_shift: u32) -> [u64; BLOCK_WORDS] {
    let seed = hash as u32;
    const SALT: [u32; 8] = [
        0x47b6_137b, 0x4476_8924, 0x1820_5237, 0x2384_8965, 0x8e6e_2354, 0x0f7c_c9b6, 0xe43d_5fa5,
        0xa4d5_2dc1,
    ];
    let mut mask = [0u64; BLOCK_WORDS];
    for (m, &s) in mask.iter_mut().zip(SALT.iter()) {
        let bit = (seed.wrapping_mul(s) >> bit_shift) & 0x3F;
        *m = 1u64 << bit;
    }
    mask
}

#[cfg(target_arch = "x86_64")]
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
