//! Split-block Bloom filter (Putze/Sanders/Singler 2007, popularised by Impala/Parquet).
//!
//! Each query touches **exactly one 64-byte cache line** — eight 64-bit words, 1 bit set per
//! word for a total of 8 bits per element. Roughly equivalent FPR to a classic Bloom with 8
//! hash functions, but one cache miss instead of eight independent random loads.
//!
//! On AVX2-capable x86_64 lookup generates the eight lane masks in vector
//! registers and checks both block halves with `vptest`. The scalar fallback
//! uses the same lane-bit formula.
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
/// Lane multipliers from Impala's split-block Bloom filter. The scalar builder,
/// scalar lookup and AVX2 lookup must use these exact wire-compatible constants.
const MASK_SALTS: [u32; BLOCK_WORDS] = [
    0x47b6_137b,
    0x4476_8924,
    0x1820_5237,
    0x2384_8965,
    0x8e6e_2354,
    0x0f7c_c9b6,
    0xe43d_5fa5,
    0xa4d5_2dc1,
];

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
    pub(crate) fn mapped(
        seed: u64,
        bit_shift: u32,
        words: crate::hugepage::HugeVec<u64>,
    ) -> Option<Self> {
        if !matches!(bit_shift, BIT_SHIFT_V2 | BIT_SHIFT_LEGACY)
            || words.len() < BLOCK_WORDS
            || !words.len().is_multiple_of(BLOCK_WORDS)
        {
            return None;
        }
        Some(Self { seed, words, bit_shift })
    }
    pub(crate) fn mmap_data(&self) -> (u64, u32, &[u64]) {
        (self.seed, self.bit_shift, self.words.as_slice())
    }
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
        // unbiased for *any* block count. Small filters need only enough complete
        // cache lines for the target bit budget. Large filters retain the existing
        // 256-block rounding so the parallel builder's equal windows and existing
        // large-build geometry stay unchanged.
        let minimum_blocks = total_bits.div_ceil(BLOCK_WORDS * 64).max(1);
        let blocks = if minimum_blocks < BLOOM_RANGES {
            minimum_blocks
        } else {
            (total_bits / (BLOCK_WORDS * 64)).max(1).next_multiple_of(BLOOM_RANGES)
        };
        let mut words = crate::hugepage::HugeVec::<u64>::zeroed(blocks * BLOCK_WORDS);
        let bit_shift = BIT_SHIFT_V2;

        // Parallel build. The filter is split into 256 windows; hashes are radix-grouped
        // by window (two parallel passes: histogram + scatter, both sequential sweeps),
        // then every window is filled by one task whose random writes never leave its own
        // window (256 KB at 10M keys, 2.7 MB at 100M). No atomics, no shadow filters,
        // no reduce.
        #[cfg(feature = "parallel")]
        {
            if crate::build_pool::parallel_enabled()
                && hashes.len() >= PARALLEL_MIN_KEYS
                && rayon::current_num_threads() > 1
            {
                match grouped {
                    Some(g) => {
                        build_parallel_ranges(hashes, words.as_mut_slice(), blocks, bit_shift, g)
                    }
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
        let block = block_of(hash, blocks);
        let base = block * BLOCK_WORDS;
        // 64 bytes — one cache line — touched per query.
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx2") || std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 verified at compile time or by the cached runtime probe;
                // `base + 8 ≤ words.len()` by construction of `block_of`.
                return unsafe {
                    contains_hash_avx2(words.as_ptr().add(base), hash as u32, self.bit_shift)
                };
            }
        }
        let mask = mask_for(hash, self.bit_shift);
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
    /// accelerator. Length is always a multiple of `BLOCK_WORDS` (8). Small
    /// filters may use any positive block count; large builds retain a multiple
    /// of 256 blocks. The on-device kernel uses
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
    /// SALT constants live in `block_bloom.rs::MASK_SALTS` and must be replicated
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
        crate::wire::extend_le(out, self.words.as_slice());
    }

    /// Streaming form of [`BlockBloom::write_to`].
    pub fn write_into<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&self.seed.to_le_bytes())?;
        let flag = if self.bit_shift == BIT_SHIFT_V2 { LEN_FLAG_V2 } else { 0 };
        w.write_all(&((self.words.len() as u64) | flag).to_le_bytes())?;
        crate::wire::write_le(w, self.words.as_slice())
    }

    pub fn read_from(buf: &[u8], pos: &mut usize) -> Option<Self> {
        let header_end = pos.checked_add(16)?;
        if header_end > buf.len() {
            return None;
        }
        let mut a = [0u8; 8];
        a.copy_from_slice(&buf[*pos..*pos + 8]);
        let seed = u64::from_le_bytes(a);
        a.copy_from_slice(&buf[*pos + 8..*pos + 16]);
        let raw_len = u64::from_le_bytes(a);
        let bit_shift = if raw_len & LEN_FLAG_V2 != 0 { BIT_SHIFT_V2 } else { BIT_SHIFT_LEGACY };
        let len = usize::try_from(raw_len & !LEN_FLAG_V2).ok()?;
        *pos = header_end;
        // At least one block (the lookup indexes block 0 unconditionally), whole
        // blocks only, and the payload must fit — with overflow-safe arithmetic so a
        // huge length word cannot slip past the check and trigger a giant allocation.
        if len < BLOCK_WORDS || !len.is_multiple_of(BLOCK_WORDS) {
            return None;
        }
        let payload = len.checked_mul(8)?;
        if payload > buf.len() - *pos {
            return None;
        }
        let mut words = crate::hugepage::HugeVec::<u64>::zeroed(len);
        {
            let slice = words.as_mut_slice();
            #[cfg(target_endian = "little")]
            {
                // SAFETY: `payload == len * 8` bytes are in bounds (checked above) and
                // the destination holds `len` u64s; every bit pattern is a valid u64.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.as_ptr().add(*pos),
                        slice.as_mut_ptr() as *mut u8,
                        payload,
                    );
                }
            }
            #[cfg(not(target_endian = "little"))]
            for (w, chunk) in slice.iter_mut().zip(buf[*pos..*pos + payload].chunks_exact(8)) {
                *w = u64::from_le_bytes(chunk.try_into().unwrap());
            }
            *pos += payload;
        }
        Some(Self { seed, words, bit_shift })
    }
}

/// Below this many keys the parallel build isn't worth the extra passes.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_KEYS: usize = 1 << 18;
/// Windows of the parallel builder; at parallel-build sizes `blocks` is a multiple
/// of this count, so the window of a hash is its top 8 bits.
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
    words.par_chunks_mut(range_blocks * BLOCK_WORDS).enumerate().for_each(|(r, win)| {
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
    let mut mask = [0u64; BLOCK_WORDS];
    for (m, &s) in mask.iter_mut().zip(MASK_SALTS.iter()) {
        let bit = (seed.wrapping_mul(s) >> bit_shift) & 0x3F;
        *m = 1u64 << bit;
    }
    mask
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn contains_hash_avx2(words_ptr: *const u64, seed: u32, bit_shift: u32) -> bool {
    use core::arch::x86_64::{
        _mm_cvtsi32_si128, _mm256_castsi256_si128, _mm256_cvtepu32_epi64, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_set1_epi32, _mm256_set1_epi64x,
        _mm256_sllv_epi64, _mm256_srl_epi32, _mm256_testc_si256,
    };
    // Compute all eight wrapped u32 products and lane indices without building
    // a stack mask. Logical shifts preserve both current (26) and legacy (27)
    // formulas; widening precedes the variable 64-bit one-bit shifts.
    // SAFETY: the static table contains exactly eight initialized u32 values.
    let salts = unsafe { _mm256_loadu_si256(MASK_SALTS.as_ptr().cast()) };
    let products = _mm256_mullo_epi32(_mm256_set1_epi32(seed as i32), salts);
    let bits = _mm256_srl_epi32(products, _mm_cvtsi32_si128(bit_shift as i32));
    let ones = _mm256_set1_epi64x(1);
    let mask_lo = _mm256_sllv_epi64(ones, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(bits)));
    let mask_hi =
        _mm256_sllv_epi64(ones, _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(bits)));
    // SAFETY: the caller supplies eight readable words. Both loads are unaligned
    // so mapped/owned tables need not start at a 32-byte boundary.
    let block_lo = unsafe { _mm256_loadu_si256(words_ptr.cast()) };
    let block_hi = unsafe { _mm256_loadu_si256(words_ptr.add(4).cast()) };
    // VPTEST checks (~block & mask) == 0 directly. Bitwise AND checks both halves
    // without a data-dependent branch between the two four-lane tests.
    (_mm256_testc_si256(block_lo, mask_lo) & _mm256_testc_si256(block_hi, mask_hi)) != 0
}

#[cfg(all(test, target_arch = "x86_64"))]
mod avx2_tests {
    use super::{BLOCK_WORDS, BlockBloom, block_of, contains_hash_avx2, mask_for};

    fn mix(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn scalar(words: &[u64], hash: u64, shift: u32) -> bool {
        let base = block_of(hash, words.len() / BLOCK_WORDS) * BLOCK_WORDS;
        let mask = mask_for(hash, shift);
        (0..BLOCK_WORDS).all(|lane| words[base + lane] & mask[lane] == mask[lane])
    }

    #[test]
    fn direct_avx2_matches_scalar_masks_for_legacy_current_and_unaligned_blocks() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let edges = [0, 1, u64::MAX, u32::MAX as u64, (u32::MAX as u64) << 32, 1 << 31, 1 << 63];
        for blocks in [1, 2, 6, 255, 256, 512] {
            for shift in [26, 27] {
                for fill in [0u64, u64::MAX, 0x1234_5678] {
                    let mut backing = vec![fill; blocks * BLOCK_WORDS + 1];
                    let offset = usize::from((backing.as_ptr() as usize).is_multiple_of(32));
                    let words = &mut backing[offset..offset + blocks * BLOCK_WORDS];
                    assert!(!(words.as_ptr() as usize).is_multiple_of(32));
                    if fill == 0x1234_5678 {
                        for (index, word) in words.iter_mut().enumerate() {
                            *word = mix(index as u64);
                        }
                        // Guarantee positive matches as well as random negatives.
                        for hash in edges.into_iter().chain((0..128).map(mix)) {
                            let base = block_of(hash, blocks) * BLOCK_WORDS;
                            for (word, mask) in words[base..base + BLOCK_WORDS]
                                .iter_mut()
                                .zip(mask_for(hash, shift))
                            {
                                *word |= mask;
                            }
                        }
                    }
                    let filter = BlockBloom {
                        seed: 0,
                        words: crate::hugepage::HugeVec::from_slice(words),
                        bit_shift: shift,
                    };
                    for hash in edges.into_iter().chain((0..4096).map(mix)) {
                        let expected = scalar(words, hash, shift);
                        let base = block_of(hash, blocks) * BLOCK_WORDS;
                        // SAFETY: AVX2 is checked above; the selected block has
                        // eight initialized readable words, deliberately unaligned.
                        let actual = unsafe {
                            contains_hash_avx2(words.as_ptr().add(base), hash as u32, shift)
                        };
                        assert_eq!(actual, expected, "blocks={blocks} shift={shift} hash={hash}");
                        assert_eq!(filter.contains_hash(hash), expected);
                    }
                }
            }
        }
    }
}
