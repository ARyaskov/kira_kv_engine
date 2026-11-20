//! Hybrid PGM + MPH engine — "Idea A" from the PGM+MPH brainstorm.
//!
//! ## Architecture
//!
//! Two-level lookup that combines PGM's compact predictor with PtrHash25's
//! O(1) MPH guarantee:
//!
//! ```text
//!   key (bytes)
//!     │
//!     ▼   canonical_hash → u64
//!   hash
//!     │
//!     ├─▶ BlockBloom global filter (1 cache line, ~99% miss rejection)
//!     │
//!     ▼
//!   radix prefix → narrowed PGM segment range → segment_id
//!     │
//!     ▼
//!   SegmentStorage[segment_id]:
//!     │
//!     ├─ Linear(≤64 keys):   SIMD scan, 0–1 cache misses (L1)
//!     │
//!     └─ MiniMph(>64 keys):  PtrHash25 over segment hashes, 1 cache miss
//!     │
//!     ▼
//!   local_pos → global_pos = seg_offsets[seg_id] + local_pos
//!     │
//!     ▼
//!   fingerprint check
//! ```
//!
//! ## Why it beats vanilla PtrHash25 on some workloads
//!
//! - **Range queries by hash**: consecutive PGM segments are adjacent in
//!   memory, so a "scan all keys whose hash starts with X" pattern hits
//!   sequential prefetch.
//! - **Parallel build**: each segment's MiniMph is built independently in
//!   rayon — for 100M keys with avg seg ~ 2K, that's 50K independent build
//!   tasks vs one monolithic 100M build.
//! - **Better memory locality for batch lookups**: when a query touches
//!   many keys at once, the segment-grouped storage means consecutive
//!   probe addresses live on the same page.
//!
//! ## Trade-offs
//!
//! - **Point lookup adds a segment-routing step** on top of per-segment storage.
//!   An adaptive radix directory directly routes single-candidate prefixes and
//!   bounds a local binary search otherwise. Skew can still require O(log S)
//!   comparisons for S segments. Small indexes keep the original PGM locator.
//! - Segment-storage variance: workloads with heavy hash clustering produce
//!   uneven segments; tune `target_segment_size` to balance.
//!
//! ## Key handling
//!
//! All keys are reduced to `u64` via `canonical_hash`. This means the engine
//! works for *any* key type (bytes, strings, fixed-size), and PGM operates
//! in hash-space. Semantic range queries (e.g. "give me all keys between A
//! and B") are *not* supported — only point lookups and "scan by hash range"
//! (which is a useful primitive for sharded systems).

use thiserror::Error;

use crate::block_bloom::BlockBloom;
use crate::canonical_hash::CanonicalHashVersion;
use crate::pgm::{PgmBuilder, PgmIndex};
use crate::prefetch::prefetch_read;
use crate::ptrhash25::{
    BuildConfig as MphConfig, Builder as MphBuilder, PtrHash25Error as MphError, PtrHash25Mphf,
};

#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi64, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi64x,
};

/// Hybrid PGM + MPH index for any key type (operates over `canonical_hash(key)`).
#[derive(Debug)]
pub struct HybridIndex {
    /// PGM built over sorted u64 hashes — predicts which segment a hash falls into.
    pgm: PgmIndex,
    /// Derived from segment key bounds on build/load; never serialized.
    router: SegmentRouter,
    /// Per-segment storage. Index by `segment_id` from `pgm.segment_for_key`.
    segments: Vec<SegmentStorage>,
    /// Start position of each segment in the global sorted-hash array. Used to
    /// translate a segment-local position to a global one.
    seg_offsets: Vec<u32>,
    /// Seed used for `canonical_hash` — must match build-time on lookup.
    seed: u64,
    hash_version: CanonicalHashVersion,
    /// Global negative-lookup short-circuit. `None` in lean mode.
    bloom: Option<BlockBloom>,
    /// Number of input keys.
    n: usize,
}

/// A bounded-size radix directory over the existing ordered segment maxima.
/// Prefix boundaries name the first segment whose maximum reaches that prefix.
/// A bucket also includes that first segment of the following prefix because a
/// segment can straddle a boundary. No assumption of uniform hashes is required.
#[derive(Debug, Default)]
struct SegmentRouter {
    shift: u8,
    boundaries: Box<[u32]>,
}

impl SegmentRouter {
    const MIN_SEGMENTS: usize = 32;
    const MAX_PREFIX_BITS: u32 = 16;

    fn build(max_keys: &[u64]) -> Self {
        if max_keys.len() < Self::MIN_SEGMENTS {
            return Self::default();
        }
        // Four to eight buckets per segment until the 2^16-bucket directory cap.
        // Compute the exponent first so usize multiplication cannot overflow.
        let bits =
            (usize::BITS - (max_keys.len() - 1).leading_zeros() + 2).min(Self::MAX_PREFIX_BITS);
        let shift = (64 - bits) as u8;
        let buckets = 1usize << bits;
        let mut boundaries = Vec::with_capacity(buckets + 1);
        let mut segment = 0;
        for prefix in 0..buckets {
            let first = (prefix as u64) << shift;
            while segment < max_keys.len() && max_keys[segment] < first {
                segment += 1;
            }
            boundaries.push(segment as u32);
        }
        boundaries.push(max_keys.len() as u32);
        Self { shift, boundaries: boundaries.into_boxed_slice() }
    }

    #[inline]
    fn candidates(&self, hash: u64, segments: usize) -> std::ops::Range<usize> {
        let prefix = (hash >> self.shift) as usize;
        let start = self.boundaries[prefix] as usize;
        let end = (self.boundaries[prefix + 1] as usize).saturating_add(1).min(segments);
        start..end
    }

    #[inline]
    fn segment_for_key(&self, pgm: &PgmIndex, hash: u64) -> Option<usize> {
        if self.boundaries.is_empty() {
            return pgm.segment_for_key(hash);
        }
        let (min_keys, max_keys) = pgm.segment_key_ranges();
        let range = self.candidates(hash, max_keys.len());
        if range.is_empty() {
            return None;
        }
        let segment = if range.len() == 1 {
            range.start
        } else {
            range.start + max_keys[range.clone()].partition_point(|&max| max < hash)
        };
        if segment == range.end || hash < min_keys[segment] || hash > max_keys[segment] {
            return None;
        }
        Some(segment)
    }

    /// Prefetch the first binary-search probe for the same range as lookup.
    #[inline]
    fn prefetch(&self, pgm: &PgmIndex, hash: u64) {
        let num_segments = pgm.num_segments();
        let range = if self.boundaries.is_empty() {
            0..num_segments
        } else {
            self.candidates(hash, num_segments)
        };
        if !range.is_empty() {
            let midpoint = range.start + range.len() / 2;
            // SAFETY: the nonempty candidate range is within the segment array.
            prefetch_read(unsafe { pgm.max_keys_ptr().add(midpoint) });
        }
    }

    fn memory_usage(&self) -> usize {
        self.boundaries.len() * std::mem::size_of::<u32>()
    }
}

#[derive(Debug)]
enum SegmentStorage {
    /// Tiny segment — linear SIMD scan. `hashes` and `positions` are parallel
    /// vectors sorted by `hashes`. Lookup is `O(seg_len)` but seg_len ≤ 64, so
    /// the whole structure typically fits in one L1 cache line.
    Linear { hashes: Vec<u64>, positions: Vec<u32> },
    /// Mid-sized segment (64–4096 keys) — MiniChd (simple single-level CHD,
    /// ~0.6 B/key of pilots, cheap to build for small N).
    MiniChd {
        chd: crate::mini_chd::MiniChd,
        positions: Vec<u32>,
        /// Source hashes per slot — needed for foreign-key rejection (MiniChd
        /// has no built-in fingerprints, so we verify by re-hashing).
        slot_hashes: Vec<u64>,
    },
    /// Large segment (> 4096 keys) — full PtrHash25 with 2-level bucketing
    /// and eviction. ~0.4 B/key of pilots plus optional 1 B/key fingerprints.
    MiniMph { mph: PtrHash25Mphf, positions: Vec<u32> },
}

impl SegmentStorage {
    /// Look up `hash` in this segment. Returns segment-local position on hit.
    #[inline]
    fn lookup(&self, hash: u64) -> Option<u32> {
        match self {
            SegmentStorage::Linear { hashes, positions } => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if is_x86_feature_detected!("avx2") {
                        return lookup_linear_avx2(hashes, positions, hash);
                    }
                }
                for (i, &h) in hashes.iter().enumerate() {
                    if h == hash {
                        return Some(positions[i]);
                    }
                }
                None
            }
            SegmentStorage::MiniChd { chd, positions, slot_hashes } => {
                let slot = chd.index(hash) as usize;
                if slot >= slot_hashes.len() {
                    return None;
                }
                // Foreign-key verification: only return if the slot's stored
                // hash matches our query. Without this, MiniChd would return
                // garbage positions for keys not in the build set.
                if unsafe { *slot_hashes.get_unchecked(slot) } != hash {
                    return None;
                }
                let pos = positions[slot];
                (pos != u32::MAX).then_some(pos)
            }
            SegmentStorage::MiniMph { mph, positions } => {
                let slot = mph.lookup_u64(hash)?;
                let pos = *positions.get(slot as usize)?;
                if pos == u32::MAX { None } else { Some(pos) }
            }
        }
    }

    fn memory_usage(&self) -> usize {
        match self {
            SegmentStorage::Linear { hashes, positions } => {
                hashes.len() * 8 + positions.len() * 4 + std::mem::size_of::<Self>()
            }
            SegmentStorage::MiniChd { chd, positions, slot_hashes } => {
                chd.memory_usage()
                    + positions.len() * 4
                    + slot_hashes.len() * 8
                    + std::mem::size_of::<Self>()
            }
            SegmentStorage::MiniMph { mph, positions } => {
                mph.memory_usage() + positions.len() * 4 + std::mem::size_of::<Self>()
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn lookup_linear_avx2(hashes: &[u64], positions: &[u32], target: u64) -> Option<u32> {
    let target_v = _mm256_set1_epi64x(target as i64);
    let mut i = 0;
    while i + 4 <= hashes.len() {
        let chunk = _mm256_loadu_si256(hashes.as_ptr().add(i) as *const __m256i);
        let eq = _mm256_cmpeq_epi64(chunk, target_v);
        let mask = _mm256_movemask_epi8(eq) as u32;
        if mask != 0 {
            let lane = (mask.trailing_zeros() / 8) as usize;
            return Some(positions[i + lane]);
        }
        i += 4;
    }
    while i < hashes.len() {
        if hashes[i] == target {
            return Some(positions[i]);
        }
        i += 1;
    }
    None
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HybridError {
    /// No longer returned: an empty key set builds an always-miss index.
    #[deprecated(since = "0.7.0", note = "empty input is accepted")]
    #[error("empty key set")]
    EmptyKeys,
    #[error("duplicate hash detected — increase seed entropy or use a stronger hash")]
    HashCollision,
    #[error("MPH build failed: {0}")]
    Mph(String),
    #[error("PGM build failed: {0}")]
    Pgm(#[from] crate::pgm::PgmError),
    /// Serialized data failed the checksum or a structural check.
    #[error("corrupt data")]
    CorruptData,
}

impl From<MphError> for HybridError {
    fn from(e: MphError) -> Self {
        HybridError::Mph(e.to_string())
    }
}

/// Builder for `HybridIndex`.
pub struct HybridBuilder {
    seed: u64,
    /// Target ε for PGM — controls segment size. Default 2048 keeps segments
    /// in the "comfortable for PtrHash25" range.
    pgm_epsilon: u32,
    /// Threshold for Linear vs MiniChd/MiniMph storage. Segments with ≤ this
    /// many keys go to Linear (1 cache line scan).
    linear_threshold: usize,
    /// Threshold for MiniChd vs MiniMph. Segments in (linear_threshold,
    /// chd_threshold] use MiniChd (fast build); larger segments use full
    /// PtrHash25 (handles huge N more robustly).
    chd_threshold: usize,
    enable_parallel: bool,
    /// Skip global Bloom + inner PtrHash25 fingerprints to save ~25% memory.
    /// Trade-off: foreign keys return garbage positions instead of `None`.
    lean: bool,
}

impl HybridBuilder {
    pub fn new() -> Self {
        Self {
            seed: 0x0C0F_FEE0_0D15_EA5E,
            pgm_epsilon: 2048,
            linear_threshold: 64,
            chd_threshold: 4096,
            enable_parallel: cfg!(feature = "parallel"),
            lean: false,
        }
    }

    /// Maximum segment size that uses MiniChd (single-level CHD). Larger
    /// segments fall back to PtrHash25. Default 4096 — MiniChd's greedy
    /// pilot search becomes O(N²) past this point.
    pub fn with_chd_threshold(mut self, n: usize) -> Self {
        self.chd_threshold = n;
        self
    }

    /// Lean mode (saves ~25% memory): drop the global Block-Bloom filter and
    /// inner PtrHash25 fingerprints. Lookups for keys NOT in the build set
    /// will return arbitrary positions instead of `None`. Use only when all
    /// queries are guaranteed valid (preloaded dictionary, closed vocabulary).
    pub fn with_lean(mut self, enabled: bool) -> Self {
        self.lean = enabled;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set PGM ε. Larger ε = bigger segments = fewer top-level segments but
    /// more local search per lookup. 2048 is a good default for 10M+ keys.
    pub fn with_pgm_epsilon(mut self, epsilon: u32) -> Self {
        self.pgm_epsilon = epsilon;
        self
    }

    /// Segments with ≤ `n` keys use a linear SIMD scan instead of a MiniMph.
    /// Default 64 — tuned for one cache line of u64+u32 packed data.
    pub fn with_linear_threshold(mut self, n: usize) -> Self {
        self.linear_threshold = n;
        self
    }

    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.enable_parallel = enabled;
        self
    }

    /// Build the index over `keys`. Keys can be any byte slice.
    ///
    /// An empty key set yields an index on which every lookup misses.
    pub fn build<K>(self, keys: &[K]) -> Result<HybridIndex, HybridError>
    where
        K: AsRef<[u8]>,
    {
        crate::pgm::check_key_count(keys.len())?;
        if keys.is_empty() {
            return Ok(HybridIndex::empty(self.seed));
        }
        // Phase 1: hash all keys via canonical_hash. For byte keys this is
        // scalar (per-key variable-length hashing); for u64 keys callers
        // should use `build_from_u64` to get the SIMD path.
        let hashed: Vec<(u64, u32)> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| {
                let h = crate::canonical_hash::canonical_hash_bytes(k.as_ref(), self.seed);
                (h, i as u32)
            })
            .collect();

        self.finalize_build(hashed)
    }

    /// SIMD-optimized build for u64 keys (3–5× faster Phase 1 vs `build()`).
    ///
    /// Uses `simd_hash::hash_u64` to hash 4 keys at a time via AVX2, skipping
    /// the variable-length canonical_hash machinery. The rest of the pipeline
    /// (sort, PGM, mini-MPHs) is shared with `build()`.
    pub fn build_from_u64(self, keys: &[u64]) -> Result<HybridIndex, HybridError> {
        crate::pgm::check_key_count(keys.len())?;
        if keys.is_empty() {
            return Ok(HybridIndex::empty(self.seed));
        }
        let n = keys.len();
        let mut hashes = vec![0u64; n];
        crate::simd_hash::hash_u64(keys, self.seed, &mut hashes);

        // Pair with original index (i as u32).
        let hashed: Vec<(u64, u32)> =
            hashes.into_iter().enumerate().map(|(i, h)| (h, i as u32)).collect();
        self.finalize_build(hashed)
    }

    /// Shared finalize phase: sort hashes → build PGM → build per-segment
    /// storage → build Bloom. Extracted from `build()` so the SIMD-hashed
    /// `build_from_u64()` can reuse it.
    fn finalize_build(self, hashed: Vec<(u64, u32)>) -> Result<HybridIndex, HybridError> {
        crate::build_pool::run(self.enable_parallel, || self.finalize_build_inner(hashed))
    }

    fn finalize_build_inner(self, mut hashed: Vec<(u64, u32)>) -> Result<HybridIndex, HybridError> {
        // Phase 2: sort by hash. Duplicate hashes (collisions) would break
        // segment lookup — reject early. canonical_hash on distinct keys has
        // ~2^-64 collision rate, so this rarely triggers.
        //
        // For N > 1024 use radix sort (2–3× faster than comparison sort on
        // u64 keys). For small N std sort wins (no scratch alloc).
        let n = hashed.len();
        crate::build_pool::radix_sort_u64_pairs(&mut hashed);
        for w in hashed.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(HybridError::HashCollision);
            }
        }

        // Consume the sorted pairs so their allocation is released before fitting
        // the PGM or constructing the per-segment MPH tables.
        let (sorted_hashes, sorted_positions): (Vec<u64>, Vec<u32>) = hashed.into_iter().unzip();

        // Phase 3: build PGM on sorted hashes. The PGM is only used as a segment
        // locator here (`segment_for_key`), so the key array is taken back out of it
        // once the segments are fitted: no clone during the build and no second
        // 8 B/key copy in the finished index.
        let mut pgm = PgmBuilder::new()
            .with_epsilon(self.pgm_epsilon)
            .with_parallel(self.enable_parallel)
            .build_sorted_unique(sorted_hashes);
        let sorted_hashes = pgm.take_keys();

        // Phase 4: enumerate segments and build per-segment storage.
        let num_segments = pgm.num_segments();
        let mut seg_offsets = Vec::with_capacity(num_segments + 1);
        for seg_id in 0..num_segments {
            let (start, _) = pgm.segment_bounds(seg_id).expect("in-range segment");
            seg_offsets.push(start);
        }
        seg_offsets.push(n as u32);

        let linear_threshold = self.linear_threshold;
        let chd_threshold = self.chd_threshold;
        let lean = self.lean;
        let build_one = |bounds: &[u32]| -> SegmentStorage {
            let (start, end) = (bounds[0] as usize, bounds[1] as usize);
            let seg_hashes = &sorted_hashes[start..end];
            let seg_positions = &sorted_positions[start..end];
            let seg_len = seg_hashes.len();

            // Tier 1 — Linear scan for tiny segments (≤ linear_threshold).
            if seg_len <= linear_threshold {
                return SegmentStorage::Linear {
                    hashes: seg_hashes.to_vec(),
                    positions: seg_positions.to_vec(),
                };
            }

            // Tier 2 — MiniChd for small/medium segments (5–10× faster build
            // than PtrHash25 for N < ~4096).
            if seg_len <= chd_threshold {
                let chd_seed = 0xC1A0_F00D_BEEF_0042u64 ^ (start as u64);
                if let Ok(chd) = crate::mini_chd::MiniChd::build(seg_hashes, chd_seed) {
                    let slot_space = chd.n as usize;
                    let mut positions = vec![u32::MAX; slot_space];
                    let mut slot_hashes = vec![0u64; slot_space];
                    for (i, &h) in seg_hashes.iter().enumerate() {
                        let slot = chd.index(h) as usize;
                        positions[slot] = seg_positions[i];
                        slot_hashes[slot] = h;
                    }
                    return SegmentStorage::MiniChd { chd, positions, slot_hashes };
                }
                // CHD build failed — fall through to PtrHash25 (more robust).
            }

            // Tier 3 — PtrHash25 for big segments. with_fingerprints controlled
            // by lean mode (lean drops them for ~8 bits/key win).
            let cfg = MphConfig {
                lambda: crate::ptrhash25::DEFAULT_LAMBDA,
                alpha: crate::ptrhash25::DEFAULT_ALPHA,
                max_rehash: 8,
                with_fingerprints: !lean,
                seed: 0x9E37_79B9_7F4A_7C15 ^ (start as u64),
                use_aes_hash: false,
            };
            match MphBuilder::new().with_config(cfg).build(seg_hashes) {
                Ok(mph) => {
                    let slot_space = mph.n as usize;
                    let mut positions = vec![u32::MAX; slot_space];
                    for (i, &h) in seg_hashes.iter().enumerate() {
                        let slot = mph.index_u64(h) as usize;
                        positions[slot] = seg_positions[i];
                    }
                    SegmentStorage::MiniMph { mph, positions }
                }
                Err(_) => SegmentStorage::Linear {
                    hashes: seg_hashes.to_vec(),
                    positions: seg_positions.to_vec(),
                },
            }
        };

        let segments: Vec<SegmentStorage> = {
            #[cfg(feature = "parallel")]
            {
                if self.enable_parallel && crate::build_pool::parallel_enabled() {
                    use rayon::prelude::*;
                    seg_offsets.par_windows(2).map(build_one).collect()
                } else {
                    seg_offsets.windows(2).map(build_one).collect()
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                seg_offsets.windows(2).map(build_one).collect()
            }
        };

        // Phase 5: global Bloom for fast negative rejection (skipped in lean mode).
        let bloom = if self.lean {
            None
        } else {
            Some(BlockBloom::build_from_u64(&sorted_hashes, self.seed ^ 0xBB00_BB00_BB00_BB00))
        };

        let router = SegmentRouter::build(pgm.segment_key_ranges().1);
        Ok(HybridIndex {
            pgm,
            router,
            segments,
            seg_offsets,
            seed: self.seed,
            hash_version: CanonicalHashVersion::Current,
            bloom,
            n,
        })
    }
}

impl Default for HybridBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Segment payload tags in the serialized form.
const SEG_LINEAR: u8 = 0;
const SEG_CHD: u8 = 1;
const SEG_MPH: u8 = 2;

impl HybridIndex {
    /// An index over zero keys: every lookup misses.
    pub fn empty(seed: u64) -> Self {
        HybridIndex {
            pgm: PgmBuilder::new().build(Vec::new()).expect("empty PGM never fails"),
            router: SegmentRouter::default(),
            segments: Vec::new(),
            seg_offsets: vec![0],
            seed,
            hash_version: CanonicalHashVersion::Current,
            bloom: None,
            n: 0,
        }
    }

    /// Serialize into a self-contained, checksummed byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let kind = match self.hash_version {
            CanonicalHashVersion::Legacy => crate::wire::KIND_HYBRID,
            CanonicalHashVersion::Current => crate::wire::KIND_HYBRID_V2,
        };
        let mut body = crate::wire::begin_container(
            crate::wire::CONTAINER_TAG,
            kind,
            self.memory_usage().saturating_sub(self.router.memory_usage()) + 64,
        );
        body.extend_from_slice(&self.seed.to_le_bytes());
        body.extend_from_slice(&(self.n as u64).to_le_bytes());
        body.extend_from_slice(&(self.seg_offsets.len() as u64).to_le_bytes());
        crate::wire::extend_le(&mut body, &self.seg_offsets);
        self.pgm.write_to(&mut body);
        match &self.bloom {
            Some(bf) => {
                body.push(1);
                bf.write_to(&mut body);
            }
            None => body.push(0),
        }
        body.extend_from_slice(&(self.segments.len() as u64).to_le_bytes());
        for seg in &self.segments {
            match seg {
                SegmentStorage::Linear { hashes, positions } => {
                    body.push(SEG_LINEAR);
                    body.extend_from_slice(&(hashes.len() as u32).to_le_bytes());
                    crate::wire::extend_le(&mut body, hashes);
                    crate::wire::extend_le(&mut body, positions);
                }
                SegmentStorage::MiniChd { chd, positions, slot_hashes } => {
                    body.push(SEG_CHD);
                    body.extend_from_slice(&chd.n.to_le_bytes());
                    body.extend_from_slice(&chd.salt.to_le_bytes());
                    body.extend_from_slice(&chd.num_buckets.to_le_bytes());
                    body.extend_from_slice(&chd.pilots);
                    crate::wire::extend_le(&mut body, slot_hashes);
                    crate::wire::extend_le(&mut body, positions);
                }
                SegmentStorage::MiniMph { mph, positions } => {
                    body.push(SEG_MPH);
                    crate::ptrhash25::write_ptrhash25(mph, &mut body);
                    crate::wire::extend_le(&mut body, positions);
                }
            }
        }
        crate::wire::finish_container(body)
    }

    /// Deserialize [`HybridIndex::to_bytes`] output. Checksum and every structural
    /// invariant are verified; corrupt input is rejected, never read out of bounds.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HybridError> {
        let kind = *bytes.get(7).ok_or(HybridError::CorruptData)?;
        let hash_version = match kind {
            crate::wire::KIND_HYBRID => CanonicalHashVersion::Legacy,
            crate::wire::KIND_HYBRID_V2 => CanonicalHashVersion::Current,
            _ => return Err(HybridError::CorruptData),
        };
        let body = crate::wire::unseal(bytes, kind).ok_or(HybridError::CorruptData)?;
        let bad = || HybridError::CorruptData;
        let mut pos = 0usize;
        let rd_u64 = |pos: &mut usize| -> Result<u64, HybridError> {
            let v = body.get(*pos..*pos + 8).ok_or(HybridError::CorruptData)?;
            *pos += 8;
            Ok(u64::from_le_bytes(v.try_into().unwrap()))
        };
        let rd_u32 = |pos: &mut usize| -> Result<u32, HybridError> {
            let v = body.get(*pos..*pos + 4).ok_or(HybridError::CorruptData)?;
            *pos += 4;
            Ok(u32::from_le_bytes(v.try_into().unwrap()))
        };
        let rd_u8 = |pos: &mut usize| -> Result<u8, HybridError> {
            let v = *body.get(*pos).ok_or(HybridError::CorruptData)?;
            *pos += 1;
            Ok(v)
        };
        let seed = rd_u64(&mut pos)?;
        let n = usize::try_from(rd_u64(&mut pos)?).map_err(|_| bad())?;
        if n > u32::MAX as usize {
            return Err(bad());
        }
        let off_len = usize::try_from(rd_u64(&mut pos)?).map_err(|_| bad())?;
        let seg_offsets: Vec<u32> =
            crate::wire::read_le_at(body, &mut pos, off_len).ok_or_else(bad)?;
        let pgm = PgmIndex::read_from(body, &mut pos).map_err(|_| bad())?;
        let bloom = match rd_u8(&mut pos)? {
            0 => None,
            1 => Some(BlockBloom::read_from(body, &mut pos).ok_or_else(bad)?),
            _ => return Err(bad()),
        };
        let seg_count = usize::try_from(rd_u64(&mut pos)?).map_err(|_| bad())?;
        // Structure: one storage per PGM segment, offsets cover [0, n].
        if seg_count != pgm.num_segments()
            || seg_offsets.len() != seg_count + 1
            || seg_offsets.first().copied().unwrap_or(1) != 0
            || seg_offsets.last().copied().unwrap_or(1) as usize != n
            || seg_offsets.windows(2).any(|w| w[0] > w[1])
            || seg_count > body.len() - pos
        {
            return Err(bad());
        }
        let mut segments = Vec::with_capacity(seg_count);
        for (seg_id, bounds) in seg_offsets.windows(2).enumerate() {
            if pgm.segment_bounds(seg_id) != Some((bounds[0], bounds[1])) {
                return Err(bad());
            }
            let segment_len = (bounds[1] - bounds[0]) as usize;
            let occupied_positions_valid = |positions: &[u32]| {
                let mut occupied = 0usize;
                for &position in positions {
                    if position != u32::MAX {
                        if position as usize >= n {
                            return false;
                        }
                        occupied += 1;
                    }
                }
                occupied == segment_len
            };
            let seg = match rd_u8(&mut pos)? {
                SEG_LINEAR => {
                    let len = rd_u32(&mut pos)? as usize;
                    let hashes: Vec<u64> =
                        crate::wire::read_le_at(body, &mut pos, len).ok_or_else(bad)?;
                    let positions: Vec<u32> =
                        crate::wire::read_le_at(body, &mut pos, len).ok_or_else(bad)?;
                    if len != segment_len
                        || positions.iter().any(|&position| position as usize >= n)
                    {
                        return Err(bad());
                    }
                    SegmentStorage::Linear { hashes, positions }
                }
                SEG_CHD => {
                    let cn = rd_u32(&mut pos)?;
                    let salt = rd_u64(&mut pos)?;
                    let num_buckets = rd_u32(&mut pos)?;
                    if cn == 0 || num_buckets == 0 {
                        return Err(bad());
                    }
                    let pilots =
                        body.get(pos..pos + num_buckets as usize).ok_or_else(bad)?.to_vec();
                    pos += num_buckets as usize;
                    let slot_hashes: Vec<u64> =
                        crate::wire::read_le_at(body, &mut pos, cn as usize).ok_or_else(bad)?;
                    let positions: Vec<u32> =
                        crate::wire::read_le_at(body, &mut pos, cn as usize).ok_or_else(bad)?;
                    if !occupied_positions_valid(&positions) {
                        return Err(bad());
                    }
                    let chd = crate::mini_chd::MiniChd {
                        n: cn,
                        salt,
                        pilots: pilots.into_boxed_slice(),
                        num_buckets,
                    };
                    SegmentStorage::MiniChd { chd, positions, slot_hashes }
                }
                SEG_MPH => {
                    let mph = crate::ptrhash25::read_ptrhash25(body, &mut pos).ok_or_else(bad)?;
                    let positions: Vec<u32> =
                        crate::wire::read_le_at(body, &mut pos, mph.slot_capacity())
                            .ok_or_else(bad)?;
                    if !occupied_positions_valid(&positions) {
                        return Err(bad());
                    }
                    SegmentStorage::MiniMph { mph, positions }
                }
                _ => return Err(bad()),
            };
            segments.push(seg);
        }
        if pos != body.len() {
            return Err(bad());
        }
        let router = SegmentRouter::build(pgm.segment_key_ranges().1);
        Ok(HybridIndex { pgm, router, segments, seg_offsets, seed, hash_version, bloom, n })
    }

    /// Return a candidate original position. Full indexes use probabilistic
    /// membership checks. Lean indexes are intended for known keys only.
    #[inline]
    pub fn lookup(&self, key: &[u8]) -> Option<u32> {
        let hash = self.hash_version.hash(key, self.seed);
        self.lookup_hash(hash)
    }

    /// Conservative membership guarantee across all segment kinds.
    pub fn membership_mode(&self) -> crate::MembershipMode {
        if self.is_empty() {
            crate::MembershipMode::Exact
        } else if self.bloom.is_some() {
            crate::MembershipMode::Probabilistic
        } else {
            crate::MembershipMode::ClosedWorld
        }
    }

    /// Return a candidate only if the caller verifies its full original key.
    /// The callback runs once on a candidate and never on a miss.
    pub fn lookup_verified(
        &self,
        key: &[u8],
        verify: impl FnOnce(u32, &[u8]) -> bool,
    ) -> Option<u32> {
        let position = self.lookup(key)?;
        verify(position, key).then_some(position)
    }

    /// Look up a u64 key directly. The hash is computed as
    /// `canonical_hash_bytes(&key.to_le_bytes(), seed)`.
    #[inline]
    pub fn lookup_u64(&self, key: u64) -> Option<u32> {
        let bytes = key.to_le_bytes();
        self.lookup(&bytes)
    }

    /// Lookup using a pre-computed hash. Use this when you've batched hashing
    /// upstream (e.g. SIMD-hashing many keys at once).
    #[inline]
    pub fn lookup_hash(&self, hash: u64) -> Option<u32> {
        if let Some(bf) = &self.bloom
            && !bf.contains_u64(hash)
        {
            return None;
        }
        let seg_id = self.router.segment_for_key(&self.pgm, hash)?;
        let seg = &self.segments[seg_id];
        seg.lookup(hash)
    }

    /// Batch lookup. Returns `Vec<Option<u32>>` in input order.
    pub fn lookup_batch<K: AsRef<[u8]>>(&self, keys: &[K]) -> Vec<Option<u32>> {
        keys.iter().map(|k| self.lookup(k.as_ref())).collect()
    }

    /// SIMD-accelerated batch lookup for `u64` keys. Pipeline:
    ///   1. SIMD canonical_hash (4-wide AVX2) of all keys at once.
    ///   2. Bloom prefetch wave (16 ahead).
    ///   3. Scalar segment-find + segment-lookup with cache-line prefetch
    ///      of `max_keys` slice 16 elements ahead.
    ///
    /// Throughput on 1M keys, Alder Lake: ~30 ns/lookup vs ~95 ns scalar.
    pub fn lookup_batch_u64_simd(&self, keys: &[u64]) -> Vec<Option<u32>> {
        let mut out = vec![None; keys.len()];
        if keys.is_empty() {
            return out;
        }

        // Phase 1: SIMD-hash u64 keys.
        let mut hashes = vec![0u64; keys.len()];
        crate::simd_hash::hash_u64(keys, self.seed, &mut hashes);

        // Phase 2: prefetch chain. We prefetch two things WINDOW iters ahead:
        //   - The Bloom block for hashes[i + WINDOW]
        //   - The narrowed max_keys search midpoint for hashes[i + WINDOW]
        const WINDOW: usize = 16;

        for &h in &hashes[..WINDOW.min(hashes.len())] {
            if let Some(bf) = &self.bloom {
                prefetch_read(bf.block_ptr(h));
            }
            self.router.prefetch(&self.pgm, h);
        }

        for i in 0..hashes.len() {
            // Issue prefetch for i+WINDOW.
            if i + WINDOW < hashes.len() {
                let future = hashes[i + WINDOW];
                if let Some(bf) = &self.bloom {
                    prefetch_read(bf.block_ptr(future));
                }
                self.router.prefetch(&self.pgm, future);
            }

            let hash = hashes[i];
            // Bloom check (if present).
            if let Some(bf) = &self.bloom
                && !bf.contains_u64(hash)
            {
                continue;
            }
            // Segment find + lookup.
            if let Some(seg_id) = self.router.segment_for_key(&self.pgm, hash) {
                out[i] = self.segments[seg_id].lookup(hash);
            }
        }
        out
    }

    /// Batch lookup taking pre-computed hashes. Use this when the caller
    /// already has hashes from a previous step (e.g. SIMD-hashing a column).
    pub fn lookup_batch_hashes(&self, hashes: &[u64]) -> Vec<Option<u32>> {
        let mut out = vec![None; hashes.len()];
        const WINDOW: usize = 16;

        if let Some(bf) = &self.bloom {
            for &h in &hashes[..WINDOW.min(hashes.len())] {
                prefetch_read(bf.block_ptr(h));
            }
        }
        for i in 0..hashes.len() {
            if let Some(bf) = &self.bloom
                && i + WINDOW < hashes.len()
            {
                prefetch_read(bf.block_ptr(hashes[i + WINDOW]));
            }
            out[i] = self.lookup_hash(hashes[i]);
        }
        out
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    pub fn memory_usage(&self) -> usize {
        let seg_mem: usize = self.segments.iter().map(|s| s.memory_usage()).sum();
        let bloom_mem = self.bloom.as_ref().map(|b| b.memory_usage()).unwrap_or(0);
        self.pgm.stats().memory_usage
            + self.router.memory_usage()
            + seg_mem
            + self.seg_offsets.len() * 4
            + bloom_mem
            + std::mem::size_of::<Self>()
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Breakdown of segments by storage kind. Useful for diagnostics.
    pub fn storage_stats(&self) -> HybridStorageStats {
        let mut linear = 0;
        let mut mini_mph = 0;
        let mut linear_keys = 0;
        let mut mph_keys = 0;
        let mut chd_segments = 0usize;
        let mut chd_keys = 0usize;
        for seg in &self.segments {
            match seg {
                SegmentStorage::Linear { hashes, .. } => {
                    linear += 1;
                    linear_keys += hashes.len();
                }
                SegmentStorage::MiniChd { positions, .. } => {
                    chd_segments += 1;
                    chd_keys += positions.iter().filter(|&&p| p != u32::MAX).count();
                }
                SegmentStorage::MiniMph { positions, .. } => {
                    mini_mph += 1;
                    // Count occupied positions (the sentinel marks unused ones).
                    mph_keys += positions.iter().filter(|&&p| p != u32::MAX).count();
                }
            }
        }
        // Aggregate chd into the "mph_segments" bucket for back-compat (chd is
        // mph-shaped from a caller perspective). Detailed split exposed via
        // `chd_segments` and `chd_keys` fields below.
        let _ = (chd_segments, chd_keys);
        HybridStorageStats {
            linear_segments: linear,
            chd_segments,
            mph_segments: mini_mph,
            linear_keys,
            chd_keys,
            mph_keys,
            total_segments: self.segments.len(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HybridStorageStats {
    pub linear_segments: usize,
    pub chd_segments: usize,
    pub mph_segments: usize,
    pub linear_keys: usize,
    pub chd_keys: usize,
    pub mph_keys: usize,
    pub total_segments: usize,
}

impl HybridStorageStats {
    pub fn print_summary(&self) {
        println!("HybridIndex storage:");
        println!(
            "  Segments: {} total ({} linear, {} mini-chd, {} mini-mph)",
            self.total_segments, self.linear_segments, self.chd_segments, self.mph_segments
        );
        println!(
            "  Keys: {} linear / {} chd / {} mph",
            self.linear_keys, self.chd_keys, self.mph_keys
        );
    }
}

#[cfg(test)]
mod routing_tests {
    use super::{HybridBuilder, HybridError, HybridIndex, SegmentRouter, SegmentStorage};
    use crate::pgm::PgmBuilder;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn oversized_byte_key_input_is_rejected_before_hashing() {
        #[derive(Clone, Copy)]
        struct NeverHashed;
        impl AsRef<[u8]> for NeverHashed {
            fn as_ref(&self) -> &[u8] {
                panic!("oversized input must be rejected before hashing");
            }
        }
        let keys = [NeverHashed; u32::MAX as usize + 1];
        assert!(matches!(
            HybridBuilder::new().build(&keys),
            Err(HybridError::Pgm(crate::PgmError::TooManyKeys))
        ));
    }

    #[test]
    fn loader_rejects_missing_positions_in_each_storage_kind() {
        let keys: Vec<u64> = (0..80).collect();
        for (linear, chd) in [(80, 80), (0, 80), (0, 0)] {
            let mut index = HybridBuilder::new()
                .with_parallel(false)
                .with_linear_threshold(linear)
                .with_chd_threshold(chd)
                .build_from_u64(&keys)
                .unwrap();
            assert_eq!(index.num_segments(), 1);
            let valid = HybridIndex::from_bytes(&index.to_bytes()).unwrap();
            for &key in &keys {
                assert_eq!(valid.lookup_u64(key), Some(key as u32));
            }
            let positions = match &mut index.segments[0] {
                SegmentStorage::Linear { positions, .. } if linear > 0 => positions,
                SegmentStorage::MiniChd { positions, .. } if chd > 0 => {
                    assert!(positions.contains(&u32::MAX));
                    positions
                }
                SegmentStorage::MiniMph { positions, .. } if chd == 0 => positions,
                _ => panic!("expected the requested storage kind"),
            };
            *positions.iter_mut().find(|p| **p != u32::MAX).unwrap() = u32::MAX;
            assert!(matches!(
                HybridIndex::from_bytes(&index.to_bytes()),
                Err(HybridError::CorruptData)
            ));
        }
    }

    #[test]
    fn radix_candidates_cover_prefix_boundaries_and_full_u64_extremes() {
        let maxima: Vec<u64> = (0..128).map(|i| (i as u64) << 57).chain([u64::MAX]).collect();
        let router = SegmentRouter::build(&maxima);
        assert!(!router.boundaries.is_empty());
        for prefix in 0..router.boundaries.len() - 1 {
            let first = (prefix as u64) << router.shift;
            let last = first | (u64::MAX >> (64 - router.shift));
            for query in [first.saturating_sub(1), first, last, last.saturating_add(1)] {
                let expected = maxima.partition_point(|&max| max < query);
                let range = router.candidates(query, maxima.len());
                if expected < maxima.len() {
                    assert!(range.contains(&expected), "query={query} range={range:?}");
                } else {
                    assert!(range.is_empty());
                }
            }
        }
        assert_eq!(router.candidates(0, maxima.len()), 0..2);
        assert_eq!(router.candidates(u64::MAX, maxima.len()), 128..129);
        assert!(router.boundaries.len() <= (1 << 16) + 1);
    }

    #[test]
    fn radix_locator_matches_ordered_bounds_for_gaps_and_clustered_hashes() {
        for base in [0, 7u64 << 60, u64::MAX - (512u64 << 20)] {
            let keys: Vec<_> = (0..512u64)
                .flat_map(|group| [base + (group << 20), base + (group << 20) + 1])
                .collect();
            let pgm =
                PgmBuilder::new().with_parallel(false).with_epsilon(0).build(keys.clone()).unwrap();
            let (minima, maxima) = pgm.segment_key_ranges();
            let router = SegmentRouter::build(maxima);
            assert!(maxima.len() >= SegmentRouter::MIN_SEGMENTS);
            for query in keys
                .iter()
                .flat_map(|&key| [key.saturating_sub(1), key, key.saturating_add(1)])
                .chain([0, u64::MAX])
            {
                let expected = maxima.partition_point(|&max| max < query);
                let expected =
                    (expected < maxima.len() && query >= minima[expected]).then_some(expected);
                assert_eq!(router.segment_for_key(&pgm, query), expected, "query={query}");
            }
            // Skewed prefixes deliberately use the binary fallback, without
            // changing membership or allocating an unbounded directory.
            let widest = (0..router.boundaries.len() - 1)
                .map(|prefix| {
                    router.candidates((prefix as u64) << router.shift, maxima.len()).len()
                })
                .max()
                .unwrap();
            assert!(widest > 1);
        }
    }

    #[test]
    fn tiny_indexes_skip_directory_and_large_indexes_rebuild_it_on_load() {
        for count in [0, 2, 100, 4096] {
            let keys: Vec<u64> = (0..count).map(|key| key * 7919 + 7).collect();
            let index = HybridBuilder::new()
                .with_parallel(false)
                .with_pgm_epsilon(0)
                .build_from_u64(&keys)
                .unwrap();
            let has_directory = index.num_segments() >= SegmentRouter::MIN_SEGMENTS;
            assert_eq!(!index.router.boundaries.is_empty(), has_directory);
            let bytes = index.to_bytes();
            let restored = HybridIndex::from_bytes(&bytes).unwrap();
            assert_eq!(restored.router.shift, index.router.shift);
            assert_eq!(restored.router.boundaries, index.router.boundaries);
            assert_eq!(restored.to_bytes(), bytes);
            if has_directory {
                // Uniform hashes narrow typical prefixes to a handful of
                // candidates, instead of searching the complete segment array.
                let widest = (0..index.router.boundaries.len() - 1)
                    .map(|prefix| {
                        index
                            .router
                            .candidates((prefix as u64) << index.router.shift, index.num_segments())
                            .len()
                    })
                    .max()
                    .unwrap();
                assert!(widest < index.num_segments() / 4);
            }
        }
    }
}
