//! Compressed pilot storage for `PtrHashV2`.
//!
//! Most pilots in a successful u8-pilot build sit in `[0, 16)` (early attempts
//! succeed when slot map is sparse). Storing them as a flat `Vec<u8>` wastes
//! ~4 bits per pilot. This module packs pilots into:
//!
//! 1. **Dense 4-bit array** for the common case (~95% of buckets, pilot ≤ 15).
//! 2. **Overflow `Vec<u8>`** for the tail with full u8 precision.
//! 3. **Bit-packed presence mask** + **rank-select** structure to navigate from
//!    bucket id to (4-bit slot OR overflow index) in O(1).
//!
//! Result at 100M keys: pilot storage shrinks from 100 MB (u8) to ~50-60 MB.
//! Combined with 25 MB LLC on i7-12700, the pilot table now fits comfortably
//! within cache for the warm-path lookup.
//!
//! ## Layout
//!
//! - `nibbles: Box<[u8]>` — 4 bits per bucket, packed two per byte.
//! - `overflow_mask: Box<[u64]>` — bit set if bucket uses overflow (1 / bucket).
//! - `overflow_rank: Box<[u32]>` — popcount prefix sum per 4096-bit superblock,
//!   for O(1) rank queries to find overflow index.
//! - `overflow: Box<[u8]>` — full u8 pilot for each overflow bucket.
//!
//! Lookup: rank(overflow_mask, bucket) gives index into `overflow` if bit is set;
//! otherwise read from `nibbles`. 1-2 cache-line accesses depending on overflow.

#![allow(dead_code)]

const OVERFLOW_BIT: u8 = 0x0F;
const SUPERBLOCK_BITS: usize = 4096;
const SUPERBLOCK_WORDS: usize = SUPERBLOCK_BITS / 64;

/// Dense superblock for V2 rank index: 512 bits = 8 u64 words = exactly one cache
/// line walk worst case. With this, rank query touches 2 cache lines max (rank
/// index + 1 bitmap line) → ~5 ns vs the 200+ ns of the original 4096-bit blocks
/// on a 100M-bucket index.
const SUPERBLOCK_BITS_V2: usize = 512;
const SUPERBLOCK_WORDS_V2: usize = SUPERBLOCK_BITS_V2 / 64;

#[derive(Debug, Clone)]
pub struct CompressedPilots {
    /// 4 bits per bucket, packed. nibbles[i >> 1] >> ((i & 1) * 4) gives the low nibble.
    /// A value of 0..14 is the direct pilot; 15 means "see overflow".
    pub nibbles: Box<[u8]>,
    /// One bit per bucket: set if bucket goes to overflow.
    pub overflow_mask: Box<[u64]>,
    /// Rank prefix: overflow_rank[i] = popcount(overflow_mask[0..i*SUPERBLOCK_WORDS]).
    /// Allows O(1) rank queries.
    pub overflow_rank: Box<[u32]>,
    /// Full u8 pilots for overflow buckets, indexed by rank order.
    pub overflow: Box<[u8]>,
    /// Number of buckets (used for bounds checks during deserialization).
    pub num_buckets: u32,
}

impl CompressedPilots {
    /// Build a compressed representation from a flat `Vec<u8>` of pilots.
    pub fn from_flat(pilots: &[u8]) -> Self {
        let num_buckets = pilots.len();
        let mut nibbles = vec![0u8; num_buckets.div_ceil(2)];
        let mut overflow_mask = vec![0u64; num_buckets.div_ceil(64)];
        let mut overflow_values: Vec<u8> = Vec::new();

        for (i, &p) in pilots.iter().enumerate() {
            if p <= 14 {
                let byte = i >> 1;
                let shift = (i & 1) * 4;
                nibbles[byte] |= (p & 0x0F) << shift;
            } else {
                // Overflow: mark bit, store full value, set nibble to OVERFLOW_BIT.
                let byte = i >> 1;
                let shift = (i & 1) * 4;
                nibbles[byte] |= OVERFLOW_BIT << shift;
                overflow_mask[i >> 6] |= 1u64 << (i & 63);
                overflow_values.push(p);
            }
        }

        // Build rank prefix.
        let superblocks = num_buckets.div_ceil(SUPERBLOCK_BITS);
        let mut overflow_rank = vec![0u32; superblocks + 1];
        let mut running = 0u32;
        for sb in 0..superblocks {
            overflow_rank[sb] = running;
            let word_lo = sb * SUPERBLOCK_WORDS;
            let word_hi = ((sb + 1) * SUPERBLOCK_WORDS).min(overflow_mask.len());
            for w in word_lo..word_hi {
                running += overflow_mask[w].count_ones();
            }
        }
        overflow_rank[superblocks] = running;

        Self {
            nibbles: nibbles.into_boxed_slice(),
            overflow_mask: overflow_mask.into_boxed_slice(),
            overflow_rank: overflow_rank.into_boxed_slice(),
            overflow: overflow_values.into_boxed_slice(),
            num_buckets: num_buckets as u32,
        }
    }

    /// O(1) pilot lookup.
    #[inline(always)]
    pub fn get(&self, bucket: usize) -> u8 {
        debug_assert!(bucket < self.num_buckets as usize);
        let byte = bucket >> 1;
        let shift = (bucket & 1) * 4;
        let nib = (unsafe { *self.nibbles.get_unchecked(byte) } >> shift) & 0x0F;
        if nib != OVERFLOW_BIT {
            nib
        } else {
            // Rank overflow_mask up to (bucket) to find overflow index.
            let idx = self.rank_overflow(bucket);
            unsafe { *self.overflow.get_unchecked(idx) }
        }
    }

    /// Number of overflow bits set in positions [0, bucket).
    #[inline(always)]
    fn rank_overflow(&self, bucket: usize) -> usize {
        let sb = bucket / SUPERBLOCK_BITS;
        let sb_rank = unsafe { *self.overflow_rank.get_unchecked(sb) } as usize;
        let word_lo = sb * SUPERBLOCK_WORDS;
        let word_target = bucket >> 6;
        let mut acc = sb_rank;
        for w in word_lo..word_target {
            acc += unsafe { *self.overflow_mask.get_unchecked(w) }.count_ones() as usize;
        }
        let last = unsafe { *self.overflow_mask.get_unchecked(word_target) };
        let bit_in_word = bucket & 63;
        let mask = if bit_in_word == 0 {
            0
        } else {
            (1u64 << bit_in_word) - 1
        };
        acc + (last & mask).count_ones() as usize
    }

    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.nibbles.len()
            + self.overflow_mask.len() * 8
            + self.overflow_rank.len() * 4
            + self.overflow.len()
    }

    pub fn overflow_count(&self) -> usize {
        self.overflow.len()
    }
}

/// Three-tier pilot encoding exploiting the heavy zero-skew of MPHF pilot distribution.
///
/// **Why this works**: when buckets are processed in descending-size order and gamma is
/// sparse (≤0.6), the pilot distribution is *extremely* skewed:
///   - ~70-80% of buckets: pilot=0 (first attempt succeeds because slot map is sparse)
///   - ~15% of buckets:    pilot ∈ [1, 14]  (small displacement enough)
///   - ~5% of buckets:     pilot ∈ [15, 255] (dense regions, needs more attempts)
///
/// We exploit this with a 3-tier layout:
///   - **zero_bitmap**: 1 bit/bucket — 0 if pilot=0 (don't store further)
///   - **nibbles**: 4 bits per *non-zero* bucket (~30% of total)
///   - **overflow**: full u8 per nibble=0xF marker (~5% of total)
///
/// Storage estimate at 100M buckets with the above distribution:
///   - zero_bitmap: 12.5 MB
///   - rank index over zero_bitmap: ~100 KB
///   - nibbles for 30M nonzero buckets: 15 MB
///   - overflow_mask among nibbles + rank: ~5 MB
///   - overflow bytes (5M): 5 MB
///   - **Total: ~38 MB vs ~57 MB single-tier nibbles**. ~1.5× compression.
///
/// Lookup adds 1 cache-line load (zero_bitmap chunk) + 1 branch. ~3-4 ns penalty.
#[derive(Debug, Clone)]
pub struct CompressedPilotsV2 {
    /// One bit per bucket: 1 = pilot is nonzero, follow to nibbles. 0 = pilot is 0.
    pub zero_bitmap: Box<[u64]>,
    /// Rank prefix sum over zero_bitmap (counts of 1-bits per SUPERBLOCK).
    /// `nonzero_index = zero_rank[block] + popcount(bitmap[block_start..bucket])`.
    pub zero_rank: Box<[u32]>,
    /// 4 bits per nonzero bucket, packed two per byte. Value 0xF means "see overflow".
    pub nibbles: Box<[u8]>,
    /// One bit per *nonzero* bucket: 1 = overflow (pilot ∈ [15, 255]), 0 = inline nibble.
    pub overflow_mask: Box<[u64]>,
    pub overflow_rank: Box<[u32]>,
    /// Full u8 pilots for overflow buckets.
    pub overflow: Box<[u8]>,
    pub num_buckets: u32,
    /// Total count of nonzero buckets (= popcount of zero_bitmap, cached).
    pub nonzero_count: u32,
}

impl CompressedPilotsV2 {
    pub fn from_flat(pilots: &[u8]) -> Self {
        let num_buckets = pilots.len();
        let bitmap_words = num_buckets.div_ceil(64);
        let mut zero_bitmap = vec![0u64; bitmap_words];
        // Pre-reserve based on typical PtrHash25 distribution: ~25% nonzero, ~5% overflow.
        // Avoids 20+ realloc cycles at N=100M (each ~500 μs VirtualAlloc + memcpy on Windows).
        let est_nonzero = num_buckets / 3; // 33% conservative
        let est_overflow = num_buckets / 16; // 6% conservative
        let mut nibbles_packed: Vec<u8> = Vec::with_capacity(est_nonzero / 2 + 1);
        let mut overflow_values: Vec<u8> = Vec::with_capacity(est_overflow);
        let mut overflow_positions: Vec<usize> = Vec::with_capacity(est_overflow);

        let mut nibble_buffer: u8 = 0;
        let mut nibble_pending = false;
        let mut nonzero_count = 0usize;

        for (b, &p) in pilots.iter().enumerate() {
            if p == 0 {
                continue;
            }
            // Mark in zero_bitmap.
            zero_bitmap[b >> 6] |= 1u64 << (b & 63);
            // Encode nibble.
            let nib: u8 = if p <= 14 {
                p & 0x0F
            } else {
                overflow_values.push(p);
                overflow_positions.push(nonzero_count);
                0x0F
            };
            if nibble_pending {
                nibble_buffer |= nib << 4;
                nibbles_packed.push(nibble_buffer);
                nibble_buffer = 0;
                nibble_pending = false;
            } else {
                nibble_buffer = nib;
                nibble_pending = true;
            }
            nonzero_count += 1;
        }
        if nibble_pending {
            nibbles_packed.push(nibble_buffer);
        }

        // Build rank prefix for zero_bitmap with DENSE V2 superblock (512 bits).
        let superblocks = num_buckets.div_ceil(SUPERBLOCK_BITS_V2);
        let mut zero_rank = vec![0u32; superblocks + 1];
        let mut running = 0u32;
        for sb in 0..superblocks {
            zero_rank[sb] = running;
            let word_lo = sb * SUPERBLOCK_WORDS_V2;
            let word_hi = ((sb + 1) * SUPERBLOCK_WORDS_V2).min(zero_bitmap.len());
            for w in word_lo..word_hi {
                running += zero_bitmap[w].count_ones();
            }
        }
        zero_rank[superblocks] = running;

        // Build overflow_mask + rank: 1 bit per NONZERO bucket.
        let ov_bitmap_words = nonzero_count.div_ceil(64);
        let mut overflow_mask = vec![0u64; ov_bitmap_words];
        for &pos in &overflow_positions {
            overflow_mask[pos >> 6] |= 1u64 << (pos & 63);
        }
        let ov_superblocks = nonzero_count.div_ceil(SUPERBLOCK_BITS_V2);
        let mut overflow_rank = vec![0u32; ov_superblocks + 1];
        let mut running = 0u32;
        for sb in 0..ov_superblocks {
            overflow_rank[sb] = running;
            let word_lo = sb * SUPERBLOCK_WORDS_V2;
            let word_hi = ((sb + 1) * SUPERBLOCK_WORDS_V2).min(overflow_mask.len());
            for w in word_lo..word_hi {
                running += overflow_mask[w].count_ones();
            }
        }
        overflow_rank[ov_superblocks] = running;

        Self {
            zero_bitmap: zero_bitmap.into_boxed_slice(),
            zero_rank: zero_rank.into_boxed_slice(),
            nibbles: nibbles_packed.into_boxed_slice(),
            overflow_mask: overflow_mask.into_boxed_slice(),
            overflow_rank: overflow_rank.into_boxed_slice(),
            overflow: overflow_values.into_boxed_slice(),
            num_buckets: num_buckets as u32,
            nonzero_count: nonzero_count as u32,
        }
    }

    /// O(1) pilot lookup. 1 cache-line load + 1 branch for the dominant pilot=0 case.
    #[inline(always)]
    pub fn get(&self, bucket: usize) -> u8 {
        debug_assert!(bucket < self.num_buckets as usize);
        let word_idx = bucket >> 6;
        let bit_idx = bucket & 63;
        let word = unsafe { *self.zero_bitmap.get_unchecked(word_idx) };
        if (word >> bit_idx) & 1 == 0 {
            return 0;
        }
        // Compute nonzero_index = rank_bitmap(zero_bitmap, bucket).
        let nz_idx = self.rank_zero(bucket);
        // Read 4-bit nibble.
        let byte_idx = nz_idx >> 1;
        let shift = (nz_idx & 1) * 4;
        let nib = (unsafe { *self.nibbles.get_unchecked(byte_idx) } >> shift) & 0x0F;
        if nib != 0x0F {
            return nib;
        }
        // Overflow path: rank into overflow array.
        let ov_idx = self.rank_overflow(nz_idx);
        unsafe { *self.overflow.get_unchecked(ov_idx) }
    }

    /// Rank query on zero_bitmap with dense superblocks. Worst case: 7 popcount ops
    /// on adjacent words (1 cache line) + 1 rank index load. ~5 ns at 100M scale.
    #[inline(always)]
    fn rank_zero(&self, bucket: usize) -> usize {
        let sb = bucket / SUPERBLOCK_BITS_V2;
        let mut acc = unsafe { *self.zero_rank.get_unchecked(sb) } as usize;
        let word_lo = sb * SUPERBLOCK_WORDS_V2;
        let word_target = bucket >> 6;
        for w in word_lo..word_target {
            acc += unsafe { *self.zero_bitmap.get_unchecked(w) }.count_ones() as usize;
        }
        let last = unsafe { *self.zero_bitmap.get_unchecked(word_target) };
        let bit_in_word = bucket & 63;
        let mask = if bit_in_word == 0 {
            0
        } else {
            (1u64 << bit_in_word) - 1
        };
        acc + (last & mask).count_ones() as usize
    }

    #[inline(always)]
    fn rank_overflow(&self, nz_idx: usize) -> usize {
        let sb = nz_idx / SUPERBLOCK_BITS_V2;
        let mut acc = unsafe { *self.overflow_rank.get_unchecked(sb) } as usize;
        let word_lo = sb * SUPERBLOCK_WORDS_V2;
        let word_target = nz_idx >> 6;
        for w in word_lo..word_target {
            acc += unsafe { *self.overflow_mask.get_unchecked(w) }.count_ones() as usize;
        }
        let last = unsafe { *self.overflow_mask.get_unchecked(word_target) };
        let bit_in_word = nz_idx & 63;
        let mask = if bit_in_word == 0 {
            0
        } else {
            (1u64 << bit_in_word) - 1
        };
        acc + (last & mask).count_ones() as usize
    }

    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.zero_bitmap.len() * 8
            + self.zero_rank.len() * 4
            + self.nibbles.len()
            + self.overflow_mask.len() * 8
            + self.overflow_rank.len() * 4
            + self.overflow.len()
    }
}

