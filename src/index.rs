use crate::block_bloom::BlockBloom;
#[allow(deprecated)]
use crate::mph_backend::{
    BackendDispatch, BackendKind, BuildConfig as BackendConfig, BuildProfile, PtrHash25Backend,
};
use crate::pgm::PgmError;
use crate::ptrhash25::{BuildConfig as MphConfig, PtrHash25Error as MphError};
use thiserror::Error;

use crate::prefetch::prefetch_read;

#[derive(Debug)]
struct MphEngine {
    backend: BackendDispatch,
    prehash_seed: u64,
    /// Membership filter on the lookup hot path. Block-Bloom touches exactly one
    /// 64-byte cache line per query, vs three random byte loads for Xor8.
    ///
    /// `None` when `lean_mph = true` — saves ~10 bits/key. The Lean mode
    /// assumes the caller only queries with keys from the build set; foreign
    /// keys return garbage positions.
    filter: Option<BlockBloom>,
    /// u16 fingerprint per slot — distinguishes real hits from MPH collisions
    /// of foreign keys. `None` in Lean mode (~16 bits/key saved).
    ///
    /// Invariant: `len() == slot_capacity() + 1`. The trailing element is padding
    /// so the AVX2 gather (a 32-bit load at every `u16`) never reads past the
    /// allocation for the last slot; it is neither serialized nor exported.
    fingerprints: Option<Box<[u16]>>,
}

/// Static MPH-backed key→id index. For semantic range queries on u64 keys
/// use `PgmBuilder` directly; for byte-key range queries use `HybridBuilder`.
///
/// Empty indexes are valid: all lookups return `KeyNotFound`, batch APIs
/// return all-`None`. Wire tag `3` is reserved for the empty marker.
pub struct Index {
    engine: Option<MphEngine>,
    key_count: usize,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IndexError {
    /// The input contains the same key twice. Detected exactly, never by chance.
    #[error("duplicate key in input")]
    DuplicateKey,
    /// The pilot search gave up after `max_rehash` salts. Practically unreachable
    /// for distinct keys; a different `mph_config.seed` is the remedy.
    #[error("could not place all keys after max rehash rounds")]
    Unresolvable,
    #[error("key not found")]
    KeyNotFound,
    #[error("invalid key format")]
    InvalidKey,
    /// Serialized data failed the checksum or a structural check.
    #[error("corrupt data")]
    CorruptData,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The operation is not available for this index configuration (for example
    /// a negative-lookup guarantee on a `lean_mph` index).
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    /// Any other MPH build failure.
    #[error("MPH error: {0}")]
    Mph(String),
    /// Any other PGM failure.
    #[error("PGM error: {0}")]
    Pgm(String),
}

impl From<MphError> for IndexError {
    fn from(err: MphError) -> Self {
        match err {
            MphError::DuplicateKey => IndexError::DuplicateKey,
            MphError::Unresolvable => IndexError::Unresolvable,
        }
    }
}

impl From<PgmError> for IndexError {
    fn from(err: PgmError) -> Self {
        match err {
            PgmError::CorruptData => IndexError::CorruptData,
            PgmError::KeyNotFound => IndexError::KeyNotFound,
            other => IndexError::Pgm(other.to_string()),
        }
    }
}

/// Configuration for the index.
///
/// The live options are `mph_config`, `enable_parallel_build` and `lean_mph`.
/// The PGM-related fields and `backend`/`hot_fraction`/`build_fast_profile` are
/// left over from the removed in-`Index` PGM engine (0.6) and the multi-backend
/// era (0.5): they have no effect, are deprecated, and will be removed in the next
/// breaking release. Use [`crate::PgmBuilder`] / [`crate::HybridBuilder`] for
/// numeric range indexes.
#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub mph_config: MphConfig,
    #[deprecated(since = "0.7.0", note = "no effect; use PgmBuilder::with_epsilon")]
    pub pgm_epsilon: u32,
    #[deprecated(since = "0.7.0", note = "no effect; Index is always MPH-backed")]
    pub auto_detect_numeric: bool,
    #[deprecated(since = "0.7.0", note = "no effect; PtrHash25 is the only backend")]
    pub backend: BackendKind,
    #[deprecated(since = "0.7.0", note = "no effect")]
    pub hot_fraction: f32,
    pub enable_parallel_build: bool,
    /// No effect. The build always detects duplicate keys exactly.
    #[deprecated(since = "0.7.0", note = "no effect; duplicates are always detected")]
    pub build_fast_profile: bool,
    #[deprecated(since = "0.7.0", note = "no effect; use PgmBuilder::with_bloom_filter")]
    pub pgm_enable_bloom: bool,
    #[deprecated(since = "0.7.0", note = "no effect; use PgmBuilder::with_elias_fano")]
    pub pgm_enable_elias_fano: bool,
    #[deprecated(since = "0.7.0", note = "no effect; use PgmBuilder::with_target_lookup_ns")]
    pub pgm_target_lookup_ns: Option<u32>,
    /// **Lean MPH mode** (saves ~50% memory for positive-only workloads).
    ///
    /// When `true`, the MPH engine skips:
    ///   - The outer Block-Bloom filter (saves ~10 bits/key)
    ///   - The u16 fingerprint table (saves ~16 bits/key)
    ///   - The inner PtrHash25 fingerprints (saves ~8 bits/key)
    ///
    /// Total saving: ~34 bits/key = ~4.5 bytes/key down to ~2.5 bytes/key
    /// at 100M scale (450 MB → 250 MB).
    ///
    /// **Trade-off**: a lookup with a key that wasn't in the build set returns
    /// an arbitrary in-range position instead of `KeyNotFound`. Use only when
    /// you can guarantee all queries are valid keys (e.g. preloaded dictionary,
    /// closed-world vocabulary, deduped log lines).
    pub lean_mph: bool,
}

#[allow(deprecated)]
impl Default for IndexConfig {
    fn default() -> Self {
        let mut cfg = crate::cpu::detect_features().optimal_index_config();
        cfg.auto_detect_numeric = false;
        cfg.backend = BackendKind::PtrHash25;
        cfg.hot_fraction = 0.15;
        cfg.enable_parallel_build = true;
        cfg.build_fast_profile = true;
        cfg.pgm_enable_bloom = false;
        cfg.pgm_enable_elias_fano = false;
        cfg.pgm_target_lookup_ns = None;
        cfg.lean_mph = false;
        cfg
    }
}

#[derive(Debug, Clone)]
pub struct IndexStats {
    pub engine: &'static str,
    pub total_keys: usize,
    pub mph_memory: usize,
    pub pgm_memory: usize,
    pub total_memory: usize,
}

/// Self-contained snapshot of the MPH state, sufficient to reproduce
/// `Index::lookup_u64(key)` on an external accelerator. See
/// [`Index::gpu_export`].
///
/// Field semantics — all little-endian, all directly usable as device
/// memory:
///
/// * `prehash_seed`: seed for the outer canonical hash.
///   Lookup formula: `canonical = mix64(key ^ prehash_seed)`.
/// * `mph_salt`, `num_buckets`, `num_slots`, `prerotate`, `pilots`, `remap`:
///   PtrHash25 constants. See `ptrhash25::PtrHash25Mphf::index_u64`
///   and [`GpuPart`] for the lookup formula.
/// * `bloom`: optional Bloom filter (present for non-lean indexes).
///   Reject early if `!bloom.contains(canonical)`.
/// * `fingerprints`: optional 16-bit fingerprint table for negative-query
///   rejection at the slot level. Check
///   `fingerprints[slot] == (canonical & 0xFFFF) as u16`.
#[derive(Debug, Clone)]
pub struct GpuExport {
    pub prehash_seed: u64,
    /// Multi-part indexes: salt of the global base hash. Single-part: the part's own
    /// hash salt (`parts[0].salt`).
    pub mph_salt: u64,
    /// Total buckets across all parts (`pilots.len()`).
    pub num_buckets: u32,
    /// Size of the output range: every lookup result is `< num_slots`. Equals the
    /// key count for indexes built by this version.
    pub num_slots: u64,
    pub prerotate: u8,
    pub pilots: Vec<u8>,
    /// Tail redirection table shared by all parts (see [`GpuPart::remap_off`]).
    pub remap: Vec<u32>,
    pub bloom: Option<BloomExport>,
    pub fingerprints: Option<Vec<u16>>,
    /// Salt of the part selector:
    /// `part = mulhi((rotated ^ part_salt) * 0x9E3779B97F4A7C15, parts.len())`.
    /// Ignored when `parts.len() == 1`.
    pub part_salt: u64,
    /// Per-part geometry in part order. Always at least one entry; a single entry
    /// means the single-partition formula (as in 0.6) applies unchanged.
    pub parts: Vec<GpuPart>,
}

/// One PtrHash25 part for GPU export. Lookup inside part `p`
/// (`rotated = key.rotate_left(prerotate)`):
///
/// ```text
/// base   = mix64(rotated ^ mph_salt)
/// h1     = parts.len() > 1 ? (base ^ p.salt) * 0xBF58476D1CE4E5B9 : base
/// h2     = rotl(h1, 23) ^ 0xA24B1F6FDA392B31
/// bucket = p.bucket_off + bucket_for(h1, p.num_buckets, p.large_buckets)
/// local  = slot_for(h2, pilots[bucket], p.num_slots)
/// if local >= p.num_keys { local = remap[p.remap_off + local - p.num_keys] }
/// slot   = p.slot_off + local
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GpuPart {
    pub slot_off: u64,
    pub salt: u64,
    pub bucket_off: u32,
    /// Output positions owned by this part. Equal to `num_slots` for 0.6 files.
    pub num_keys: u32,
    /// Slots the pilot search hashes into.
    pub num_slots: u32,
    pub num_buckets: u32,
    /// `floor(num_buckets × 0.30)` — size of the dense "large" zone.
    pub large_buckets: u32,
    /// Offset of this part's `num_slots - num_keys` entries in `GpuExport::remap`.
    pub remap_off: u32,
}

/// Bloom filter snapshot for GPU export. The Bloom uses
/// split-block layout with `BLOCK_WORDS = 8` (`64 B` per block).
/// Block index is `(((canonical >> 32) * blocks) >> 32)`.
#[derive(Debug, Clone)]
pub struct BloomExport {
    /// Lane bit-index shift: 26 for filters built by this version (6-bit index), 27
    /// for filters loaded from 0.6 files (5-bit index). See `BlockBloom::export_words`.
    pub bit_shift: u32,
    /// Number of 64-byte blocks. Always a multiple of 256 (no longer a power of two
    /// since the partitioned build; the multiply-high reduction is unbiased regardless).
    pub blocks: usize,
    /// Concatenated 64-bit words. Length is `blocks * 8`.
    pub words: Vec<u64>,
}

impl Index {
    /// Build from owned keys. Equivalent to [`Index::build_index_ref`] followed by
    /// dropping `keys` (the drop of millions of small allocations runs in parallel).
    ///
    /// Prefer [`Index::build_index_ref`] when the keys stay alive anyway: it avoids the
    /// clone on the caller side and the deallocation cost inside the build.
    pub fn build_index<K>(keys: Vec<K>, config: IndexConfig) -> Result<Self, IndexError>
    where
        K: AsRef<[u8]> + Send + Sync,
    {
        let index = Self::build_index_ref(&keys, config)?;
        drop_keys_parallel(keys);
        Ok(index)
    }

    /// Build from borrowed keys (must be unique). Keys are hashed in place in parallel;
    /// nothing is copied into an intermediate arena.
    ///
    /// Duplicate keys are detected exactly (`IndexError::DuplicateKey`).
    pub fn build_index_ref<K>(keys: &[K], config: IndexConfig) -> Result<Self, IndexError>
    where
        K: AsRef<[u8]> + Sync,
    {
        if keys.is_empty() {
            return Ok(Self::empty());
        }
        let mut config = config;
        if config.lean_mph {
            config.mph_config.with_fingerprints = false;
        }
        let engine =
            run_in_build_pool(config.enable_parallel_build, || build_engine(keys, &config))?;
        Ok(Index {
            engine: Some(engine),
            key_count: keys.len(),
        })
    }

    #[inline]
    pub fn empty() -> Self {
        Self { engine: None, key_count: 0 }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.key_count == 0
    }

    /// Upper bound on `lookup`'s return value. Equal to `len()` for indexes built by
    /// this version (the MPH is minimal); `~1.1 * len()` for indexes loaded from 0.6
    /// files; zero for empty. Side arrays indexed by `lookup` results must be sized
    /// to this.
    #[inline]
    pub fn slot_capacity(&self) -> usize {
        match &self.engine {
            Some(engine) => engine.backend.slot_capacity(),
            None => 0,
        }
    }

    pub fn lookup(&self, key: &[u8]) -> Result<usize, IndexError> {
        let Some(engine) = self.engine.as_ref() else {
            return Err(IndexError::KeyNotFound);
        };
        self.lookup_mph(engine, key)
    }

    #[deprecated(since = "0.7.0", note = "alias; use lookup")]
    pub fn get(&self, key: &[u8]) -> Result<usize, IndexError> {
        self.lookup(key)
    }

    pub fn lookup_str(&self, key: &str) -> Result<usize, IndexError> {
        self.lookup(key.as_bytes())
    }

    #[deprecated(since = "0.7.0", note = "alias; use lookup_str")]
    pub fn get_str(&self, key: &str) -> Result<usize, IndexError> {
        self.lookup_str(key)
    }

    pub fn lookup_u64(&self, key: u64) -> Result<usize, IndexError> {
        let Some(engine) = self.engine.as_ref() else {
            return Err(IndexError::KeyNotFound);
        };
        self.lookup_mph(engine, &key.to_le_bytes())
    }

    /// AVX2-gather based fingerprint check for 8 indices at a time.
    /// Returns a bitmask: bit `i` is set if fingerprints[indices[i]] == expected_fps[i].
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    #[inline]
    unsafe fn gather_fp_check_x8(
        fp_base: *const u16,
        indices: [u32; 8],
        expected: [u16; 8],
    ) -> u8 {
        use core::arch::x86_64::{
            _mm256_and_si256, _mm256_cmpeq_epi32, _mm256_i32gather_epi32, _mm256_movemask_epi8,
            _mm256_set1_epi32, _mm256_set_epi32,
        };
        let vindex = _mm256_set_epi32(
            indices[7] as i32, indices[6] as i32, indices[5] as i32, indices[4] as i32,
            indices[3] as i32, indices[2] as i32, indices[1] as i32, indices[0] as i32,
        );
        // scale = 2 bytes per u16 element; gather 8 u32, lower 16 bits = fp value.
        let loaded = _mm256_i32gather_epi32::<2>(fp_base as *const i32, vindex);
        let mask_lo = _mm256_set1_epi32(0x0000_FFFF);
        let loaded_lo = _mm256_and_si256(loaded, mask_lo);
        let expected_v = _mm256_set_epi32(
            expected[7] as i32, expected[6] as i32, expected[5] as i32, expected[4] as i32,
            expected[3] as i32, expected[2] as i32, expected[1] as i32, expected[0] as i32,
        );
        let cmp = _mm256_cmpeq_epi32(loaded_lo, expected_v);
        let mm = _mm256_movemask_epi8(cmp) as u32;
        let mut out = 0u8;
        for i in 0..8 {
            if (mm >> (i * 4)) & 0xF == 0xF {
                out |= 1 << i;
            }
        }
        out
    }

    /// SIMD-batched u64 lookup: hashes 4 keys at a time via AVX2 `hash_u64_avx2`,
    /// then issues paired prefetches for each lane's Bloom block + pilot byte. On
    /// 100M-scale indexes this beats `lookup_batch_pipelined` by 20-35% because the
    /// 4-wide hash phase happens 4× faster and the prefetch wave is wider.
    ///
    /// Available only for u64 keys (cannot vectorize variable-length byte hashing).
    /// Returns Some(idx) on hit, None on miss (filter or fingerprint reject).
    ///
    /// **Allocation profile**: allocates `out: Vec<Option<usize>>` + a `canon:
    /// Vec<u64>` scratch buffer (each `keys.len()` long) on every call. For
    /// hot paths that call this millions of times per second (e.g.
    /// per-read seeding in an aligner), prefer
    /// [`lookup_batch_u64_simd_into`] which takes both buffers from the
    /// caller and does zero allocation.
    pub fn lookup_batch_u64_simd(&self, keys: &[u64]) -> Vec<Option<usize>> {
        let mut out = vec![None; keys.len()];
        let mut canon = vec![0u64; keys.len()];
        self.lookup_batch_u64_simd_into(keys, &mut canon, &mut out);
        out
    }

    /// Zero-allocation variant of [`lookup_batch_u64_simd`]. The caller
    /// supplies both `canon` (canonical-hash scratch) and `out` (the
    /// `Option<usize>` result slice). Both must already be sized to at
    /// least `keys.len()`; any extra slots are left untouched.
    ///
    /// **Why this exists**: the alloc'ing variant burns ~2 mallocs per
    /// call. At the scale of ~4 M per-read calls in an aligner that's
    /// 8 M allocations, which on Windows allocator turned out to be the
    /// dominant cost — using `lookup_batch_u64_simd` made a real-world
    /// hg38 alignment **3.7× slower** in the seeding stage vs scalar
    /// `lookup_u64`. With caller-owned buffers (one per thread, sized to
    /// max minimizers-per-read) the SIMD path actually wins.
    ///
    /// Buffers should be sized once at thread start; subsequent calls
    /// just overwrite their first `keys.len()` slots.
    ///
    /// # Panics
    ///
    /// Panics if `canon.len() < keys.len()` or `out.len() < keys.len()`.
    /// Use `&mut canon[..keys.len()]` and `&mut out[..keys.len()]` from a
    /// larger scratch buffer if reusing across variable-sized calls.
    pub fn lookup_batch_u64_simd_into(
        &self,
        keys: &[u64],
        canon: &mut [u64],
        out: &mut [Option<usize>],
    ) {
        assert!(canon.len() >= keys.len(), "canon scratch too small");
        assert!(out.len() >= keys.len(), "out slice too small");
        let n = keys.len();
        if n == 0 {
            return;
        }
        // Reset the result range — callers may reuse `out` across calls
        // of varying length, so we can't trust the previous values.
        for slot in &mut out[..n] {
            *slot = None;
        }
        let Some(engine) = self.engine.as_ref() else { return };

        // Hash all keys up-front via AVX2 (4-wide).
        crate::simd_hash::hash_u64(keys, engine.prehash_seed, &mut canon[..n]);
        let canon = &canon[..n];

        // Lean-mode fast path: no Bloom, no fingerprints — straight from
        // hash → backend.lookup → output. Saves ~30 % per-key work.
        if engine.filter.is_none() && engine.fingerprints.is_none() {
            for (i, &hash) in canon.iter().enumerate() {
                if let Some(idx) = engine.backend.lookup(hash) {
                    out[i] = Some(idx as usize);
                }
            }
            return;
        }

        // Pre-prefetch the first WINDOW Bloom blocks (only when filter present).
        const WINDOW: usize = 16;
        if let Some(bf) = &engine.filter {
            for &h in &canon[..WINDOW.min(n)] {
                prefetch_read(bf.block_ptr(h));
            }
        }

        // Main loop: 8 keys per iteration with an AVX2 gather for the fingerprint
        // stage (one instruction instead of 8 scalar loads). Dispatched at runtime so
        // dependents built without `-C target-cpu` get it too; the gather takes i32
        // lane indices, so it is skipped for slot counts beyond 2^31.
        let mut i = 0usize;
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = cfg!(target_feature = "avx2") || std::arch::is_x86_feature_detected!("avx2");
            if avx2 && engine.backend.slot_capacity() <= i32::MAX as usize {
                // SAFETY: AVX2 presence checked above.
                i = unsafe { Self::batch_u64_avx2(engine, canon, out, WINDOW) };
            }
        }

        // Tail (and the whole batch on non-AVX2 hosts): scalar with the same prefetch wave.
        while i < n {
            let hash = canon[i];
            if let Some(bf) = &engine.filter
                && i + WINDOW < n
            {
                prefetch_read(bf.block_ptr(canon[i + WINDOW]));
            }
            let bloom_ok = match &engine.filter {
                Some(bf) => bf.contains_hash(hash),
                None => true,
            };
            if bloom_ok {
                if let Some(idx) = engine.backend.lookup(hash) {
                    let idx = idx as usize;
                    let ok = match &engine.fingerprints {
                        Some(fps) => {
                            let fp = fingerprint16_mph(hash);
                            unsafe { *fps.get_unchecked(idx) == fp }
                        }
                        None => true,
                    };
                    if ok {
                        out[i] = Some(idx);
                    }
                }
            }
            i += 1;
        }
    }

    /// 8-wide body of [`Index::lookup_batch_u64_simd_into`]. Processes whole groups
    /// of 8 and returns the number of keys handled.
    ///
    /// # Safety
    /// AVX2 must be available; `engine.backend.slot_capacity() <= i32::MAX`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn batch_u64_avx2(
        engine: &MphEngine,
        canon: &[u64],
        out: &mut [Option<usize>],
        window: usize,
    ) -> usize {
        let n = canon.len();
        let fp_base = engine
            .fingerprints
            .as_ref()
            .map(|fp| fp.as_ptr())
            .unwrap_or(std::ptr::null());
        let mut i = 0usize;
        while i + 8 <= n {
            // Lookahead Bloom prefetch.
            if let Some(bf) = &engine.filter {
                for k in 0..8 {
                    if i + window + k < n {
                        prefetch_read(bf.block_ptr(canon[i + window + k]));
                    }
                }
            }

            // Bloom + backend.lookup for 8 keys, collecting (idx, expected_fp) pairs.
            let mut indices = [0u32; 8];
            let mut expected = [0u16; 8];
            let mut alive = [false; 8];
            for k in 0..8 {
                let hash = canon[i + k];
                if let Some(bf) = &engine.filter
                    && !bf.contains_hash(hash)
                {
                    continue;
                }
                if let Some(idx) = engine.backend.lookup(hash) {
                    indices[k] = idx;
                    expected[k] = fingerprint16_mph(hash);
                    alive[k] = true;
                }
            }
            if !fp_base.is_null() {
                // Single gather for fingerprint validation. Dead lanes carry index 0;
                // the last live slot's 32-bit load ends in the padding element.
                let bitmask = unsafe { Self::gather_fp_check_x8(fp_base, indices, expected) };
                for k in 0..8 {
                    if alive[k] && (bitmask & (1 << k)) != 0 {
                        out[i + k] = Some(indices[k] as usize);
                    }
                }
            } else {
                for k in 0..8 {
                    if alive[k] {
                        out[i + k] = Some(indices[k] as usize);
                    }
                }
            }
            i += 8;
        }
        i
    }

    /// Same result as [`Index::lookup_u64`]; the extra `Option` layer dates from
    /// the multi-backend era and is always `Some` for a non-empty index.
    #[deprecated(since = "0.7.0", note = "use lookup_u64; it is the same code path")]
    #[inline]
    pub fn lookup_u64_fast(&self, key: u64) -> Option<Result<usize, IndexError>> {
        let engine = self.engine.as_ref()?;
        // Explicit destructure: adding another backend variant must break here.
        use crate::mph_backend::BackendDispatch;
        let BackendDispatch::PtrHash25(_) = &engine.backend;
        // Hash key with same canonical path the index was built against.
        let canonical = canonical_hash_key(&key.to_le_bytes(), engine.prehash_seed);
        if let Some(bf) = &engine.filter
            && !bf.contains_hash(canonical)
        {
            return Some(Err(IndexError::KeyNotFound));
        }
        let idx = match engine.backend.lookup(canonical) {
            Some(i) => i as usize,
            None => return Some(Err(IndexError::KeyNotFound)),
        };
        if let Some(fps) = &engine.fingerprints {
            let fp = fingerprint16_mph(canonical);
            let ok = unsafe { *fps.get_unchecked(idx) == fp };
            Some(if ok { Ok(idx) } else { Err(IndexError::KeyNotFound) })
        } else {
            Some(Ok(idx))
        }
    }

    #[deprecated(since = "0.7.0", note = "alias; use lookup_u64")]
    pub fn get_u64(&self, key: u64) -> Result<usize, IndexError> {
        self.lookup_u64(key)
    }

    /// Range queries are not supported by the hash-based `Index`; this always
    /// returns an empty vector. Use [`crate::PgmIndex`] for semantic u64 ranges.
    #[deprecated(since = "0.7.0", note = "Index has no order; use PgmIndex::range")]
    pub fn range(&self, _min_key: u64, _max_key: u64) -> Vec<usize> {
        Vec::new()
    }

    #[deprecated(since = "0.7.0", note = "Index has no order; use PgmIndex::range")]
    #[allow(deprecated)]
    pub fn get_all(&self, min_key: u64, max_key: u64) -> Vec<usize> {
        self.range(min_key, max_key)
    }

    /// Whether this index can reject keys that were not in the build set.
    ///
    /// `false` for `lean_mph` indexes: they carry neither a Bloom filter nor
    /// fingerprints, so `lookup` returns an arbitrary in-range position for a
    /// foreign key and [`Index::contains`] can only answer `true`.
    #[inline]
    pub fn supports_negative_lookups(&self) -> bool {
        match &self.engine {
            Some(engine) => engine.filter.is_some() || engine.fingerprints.is_some(),
            // An empty index rejects every key exactly.
            None => true,
        }
    }

    /// Probabilistic membership test: `false` means the key is definitely absent,
    /// `true` means it is present with the Bloom filter's false-positive rate
    /// (~0.5% at 11 bits/key). For an exact answer use `lookup(..).is_ok()`.
    ///
    /// **Lean mode** (`lean_mph = true`): there is no membership information at
    /// all, so this always returns `true` for a non-empty index. Check
    /// [`Index::supports_negative_lookups`] before relying on this method.
    pub fn contains(&self, key: &[u8]) -> bool {
        let Some(engine) = self.engine.as_ref() else {
            return false;
        };
        let canonical = canonical_hash_key(key, engine.prehash_seed);
        match &engine.filter {
            Some(bf) => bf.contains_hash(canonical),
            // Fingerprints without a filter never happen for indexes built by
            // this crate; go through the full lookup so the answer stays exact
            // if such a combination is ever loaded. In lean mode the lookup
            // cannot fail, so this is the documented always-`true`.
            None => self.lookup_mph(engine, key).is_ok(),
        }
    }

    #[deprecated(since = "0.7.0", note = "alias; use contains")]
    pub fn has(&self, key: &[u8]) -> bool {
        self.contains(key)
    }

    #[deprecated(since = "0.7.0", note = "alias; use contains")]
    pub fn exists(&self, key: &[u8]) -> bool {
        self.contains(key)
    }

    /// Batched [`Index::contains`]; same probabilistic semantics and the same
    /// always-`true` answer in lean mode.
    pub fn contains_batch(&self, keys: &[&[u8]]) -> Vec<bool> {
        let Some(engine) = self.engine.as_ref() else {
            return vec![false; keys.len()];
        };
        keys.iter()
            .map(|&key| {
                let canonical = canonical_hash_key(key, engine.prehash_seed);
                match &engine.filter {
                    Some(bf) => bf.contains_hash(canonical),
                    None => self.lookup_mph(engine, key).is_ok(),
                }
            })
            .collect()
    }

    #[deprecated(since = "0.7.0", note = "alias; use contains_batch")]
    pub fn has_batch(&self, keys: &[&[u8]]) -> Vec<bool> {
        self.contains_batch(keys)
    }

    #[deprecated(since = "0.7.0", note = "alias; use contains_batch")]
    pub fn exists_batch(&self, keys: &[&[u8]]) -> Vec<bool> {
        self.contains_batch(keys)
    }

    pub fn lookup_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>> {
        let Some(engine) = self.engine.as_ref() else {
            return vec![None; keys.len()];
        };
        let mut out = Vec::with_capacity(keys.len());
        let mut i = 0usize;
        while i + 16 <= keys.len() {
            prefetch_key_batch(keys, i, 16);
            for j in 0..16 {
                out.push(self.lookup_mph(engine, keys[i + j]).ok());
            }
            i += 16;
        }
        while i + 8 <= keys.len() {
            prefetch_key_batch(keys, i, 8);
            for j in 0..8 {
                out.push(self.lookup_mph(engine, keys[i + j]).ok());
            }
            i += 8;
        }
        while i < keys.len() {
            out.push(self.lookup_mph(engine, keys[i]).ok());
            i += 1;
        }
        out
    }

    #[deprecated(since = "0.7.0", note = "alias; use lookup_batch")]
    pub fn get_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>> {
        self.lookup_batch(keys)
    }

    /// Software-pipelined batched lookup with WINDOW=32 prefetch depth. At 100M index
    /// size the per-key working set spans 3 random cache lines (BlockBloom block,
    /// fingerprint byte, and pilot byte from the backend) — together ~190 ns of stalls
    /// per key with sequential code. Issuing 32 prefetches ahead overlaps all three
    /// loads with ongoing compute, getting throughput close to the DRAM bandwidth limit.
    ///
    /// Empirically window=32 beats window=8 by 2-3× on 100M-key indexes on Alder Lake
    /// (where each core has ~10 outstanding L1/L2 misses, scaled by 3 prefetch streams).
    pub fn lookup_batch_pipelined(&self, keys: &[&[u8]]) -> Vec<Option<usize>> {
        let Some(engine) = self.engine.as_ref() else {
            return vec![None; keys.len()];
        };
        let mut out = Vec::with_capacity(keys.len());
        {
            {
                // Lean-mode fast path: no Bloom, no fingerprints. Strip down to
                // hash → backend.lookup. Still does WINDOW-ahead hashing for
                // pilot-table prefetch hit, but skips Bloom/fp waves entirely.
                if engine.filter.is_none() && engine.fingerprints.is_none() {
                    for &k in keys {
                        let h = canonical_hash_key(k, engine.prehash_seed);
                        out.push(engine.backend.lookup(h).map(|i| i as usize));
                    }
                    return out;
                }
                let filter = engine.filter.as_ref();
                let fingerprints = engine.fingerprints.as_ref();
                const WINDOW: usize = 32;
                if keys.len() < WINDOW * 2 {
                    for &k in keys {
                        out.push(self.lookup_mph(engine, k).ok());
                    }
                    return out;
                }
                let mut canon = vec![0u64; WINDOW * 2];

                // Bootstrap: hash first WINDOW keys + issue first wave of Bloom prefetches.
                for slot in 0..WINDOW {
                    let h = canonical_hash_key(keys[slot], engine.prehash_seed);
                    canon[slot] = h;
                    if let Some(bf) = filter {
                        prefetch_read(bf.block_ptr(h));
                    }
                }

                let mut hash_head = WINDOW;
                for i in 0..keys.len() {
                    let ring_pos = i % (WINDOW * 2);
                    let hash = canon[ring_pos];

                    // Wave A: hash + Bloom prefetch for key[i + WINDOW].
                    if hash_head < keys.len() {
                        let next_pos = hash_head % (WINDOW * 2);
                        let h = canonical_hash_key(keys[hash_head], engine.prehash_seed);
                        canon[next_pos] = h;
                        if let Some(bf) = filter {
                            prefetch_read(bf.block_ptr(h));
                        }
                        hash_head += 1;
                    }

                    // Wave B: optional Bloom check.
                    if let Some(bf) = filter {
                        if !bf.contains_hash(hash) {
                            out.push(None);
                            continue;
                        }
                    }
                    let idx_opt = engine.backend.lookup(hash);

                    // Wave C: prefetch fingerprint (only if fingerprints present).
                    if let (Some(fps), Some(idx)) = (fingerprints, idx_opt) {
                        // SAFETY: `idx < slot_capacity() < fps.len()`; a prefetch is a hint.
                        prefetch_read(unsafe { fps.as_ptr().add(idx as usize) });
                    }

                    let res = match idx_opt {
                        Some(idx) => {
                            let idx = idx as usize;
                            match fingerprints {
                                Some(fps) => {
                                    let fp = fingerprint16_mph(hash);
                                    let ok = unsafe { *fps.get_unchecked(idx) == fp };
                                    if ok { Some(idx) } else { None }
                                }
                                None => Some(idx),
                            }
                        }
                        None => None,
                    };
                    out.push(res);
                }
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.key_count
    }

    /// **DEBUG-ONLY**: directly probe the Bloom filter on a precomputed
    /// canonical hash. Returns `None` if the index has no filter
    /// (`lean_mph` mode). Used by GPU correctness tests to verify that the
    /// exported `bloom_words` match the engine's actual behavior.
    pub fn debug_bloom_contains_canonical(&self, canonical: u64) -> Option<bool> {
        let engine = self.engine.as_ref()?;
        engine.filter.as_ref().map(|bf| bf.contains_hash(canonical))
    }

    /// Export every constant needed to reproduce `lookup_u64` on an
    /// external accelerator (GPU, FPGA). All buffers are owned `Vec`s
    /// already in the exact layout the on-device kernel will use — no
    /// further packing required.
    ///
    /// **Use case**: the Kira aligner ships a CUDA path that needs to
    /// run MPH lookups for ~100 M minimizer hashes per pipeline batch
    /// on a GTX 1060. Doing the lookups CPU-side hits a memory-bandwidth
    /// wall (~15 s/batch). Uploading the MPH state to GPU and running
    /// `mph_bucket_lookup_batch.cu` (in `kira-ls-aligner/src/cuda/`)
    /// gets the GPU's 192 GB/s vs CPU's ~30 GB/s for random reads.
    ///
    /// Constraints: works only with `BackendKind::PtrHash25` and
    /// `use_aes_hash == false`. Returns `None` otherwise — caller falls
    /// back to CPU lookups. The AES-NI path can be added later but
    /// requires a separate CUDA implementation.
    pub fn gpu_export(&self) -> Option<GpuExport> {
        use crate::mph_backend::{BackendDispatch, PtrHash25Storage};
        let engine = self.engine.as_ref()?;

        let BackendDispatch::PtrHash25(ph_backend) = &engine.backend;
        // Hot-tier `Map` variant isn't a true MPH; skip GPU export.
        let PtrHash25Storage::Mph(mph) = &ph_backend.storage else {
            return None;
        };
        if mph.use_aes_hash {
            return None;
        }

        let pilots_flat: Vec<u8> = mph.pilots.as_slice().to_vec();

        let bloom_export = engine.filter.as_ref().map(|bf| {
            let words = bf.export_words();
            BloomExport {
                bit_shift: bf.bit_shift(),
                blocks: words.len() / 8, // BLOCK_WORDS=8
                words,
            }
        });

        let fingerprints = engine
            .fingerprints
            .as_ref()
            .map(|fps| fps[..fps.len() - 1].to_vec());

        let parts = mph
            .parts
            .iter()
            .map(|p| GpuPart {
                slot_off: p.slot_off,
                salt: p.salt,
                bucket_off: p.bucket_off,
                num_keys: p.num_keys,
                num_slots: p.num_slots,
                num_buckets: p.num_buckets,
                large_buckets: p.large_buckets,
                remap_off: p.remap_off,
            })
            .collect();

        Some(GpuExport {
            prehash_seed: engine.prehash_seed,
            mph_salt: mph.salt,
            num_buckets: mph.num_buckets,
            num_slots: mph.n,
            prerotate: mph.prerotate,
            pilots: pilots_flat,
            remap: mph.remap.to_vec(),
            bloom: bloom_export,
            fingerprints,
            part_salt: mph.part_salt,
            parts,
        })
    }

    pub fn stats(&self) -> IndexStats {
        let Some(engine) = self.engine.as_ref() else {
            return IndexStats {
                engine: "mph-empty",
                total_keys: 0,
                mph_memory: 0,
                pgm_memory: 0,
                total_memory: 0,
            };
        };
        let mph_memory = engine.backend.memory_usage_bytes();
        let filter_memory = engine.filter.as_ref().map(|b| b.memory_usage()).unwrap_or(0);
        let fp_memory = engine
            .fingerprints
            .as_ref()
            .map(|fp| (fp.len() - 1) * std::mem::size_of::<u16>())
            .unwrap_or(0);
        IndexStats {
            engine: "mph",
            total_keys: self.key_count,
            mph_memory,
            pgm_memory: 0,
            total_memory: mph_memory + filter_memory + fp_memory,
        }
    }

    pub fn print_detailed_stats(&self) {
        let stats = self.stats();
        println!("Index Statistics:");
        println!("  Engine: {}", stats.engine);
        println!("  Total keys: {}", stats.total_keys);
        if stats.mph_memory > 0 {
            println!(
                "  MPH index: {:.2} MB",
                stats.mph_memory as f64 / 1_048_576.0
            );
        }
        if stats.pgm_memory > 0 {
            println!(
                "  PGM index: {:.2} MB",
                stats.pgm_memory as f64 / 1_048_576.0
            );
        }
    }

    /// Save the index using a section-based on-disk layout. For indexes whose primary
    /// backend is `PtrHashV2`, each component (pilots, fingerprints, bloom-words, meta)
    /// lives in its own 64-byte-aligned section so a future `open_mmap_zero_copy` can
    /// reference the data in place via mmap. For other backends we fall back to a
    /// single `LegacyPayload` section that wraps `to_bytes()`.
    pub fn save_mmap<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), IndexError> {
        use crate::mmap_index::{MmapIndexWriter, SectionKind};
        let mut w = MmapIndexWriter::create(path, self.key_count as u64)?;
        // We always write the legacy payload (so open_mmap continues to work).
        // For PtrHashV2-backed Mph engines we ALSO add per-field sections, enabling
        // zero-copy reads via Index::open_mmap_zero_copy in future versions.
        let bytes = self.to_bytes()?;
        w.add_section(SectionKind::LegacyPayload, bytes);
        Ok(w.finalize()?)
    }

    /// Open a previously `save_mmap`'d index. Currently does a one-time read from the
    /// LegacyPayload section into a `Vec<u8>`. A future zero-copy variant will keep
    /// the mmap alive and return views into it without copying.
    pub fn open_mmap<P: AsRef<std::path::Path>>(path: P) -> Result<Self, IndexError> {
        use crate::mmap_index::{MmapIndex, SectionKind};
        let mmap = MmapIndex::open(path)?;
        let header = mmap.parse_header().map_err(|e| match e.kind() {
            std::io::ErrorKind::InvalidData => IndexError::CorruptData,
            _ => IndexError::Io(e),
        })?;
        let bytes = mmap
            .section(&header, SectionKind::LegacyPayload)
            .or_else(|| mmap.section(&header, SectionKind::PtrHash25Pilots))
            .ok_or(IndexError::CorruptData)?;
        Self::from_bytes(bytes)
    }

    /// Serialize to a self-contained byte vector.
    ///
    /// Layout (tag 4, written by this version):
    /// `[tag=4][magic "KIRA"][format u16][hash_id u8][reserved u8][key_count u64]`
    /// then either nothing (empty index) or
    /// `[prehash_seed u64][backend][has_filter u8][filter?][has_fp u8][fp?]`,
    /// and always a trailing `[checksum u64]` over everything before it.
    /// Tags 0, 2 and 3 written by earlier versions are still readable (without
    /// checksum verification).
    pub fn to_bytes(&self) -> Result<Vec<u8>, IndexError> {
        let mut out = Vec::with_capacity(self.stats().total_memory + 64);
        self.write_to(&mut out)?;
        Ok(out)
    }

    /// Stream the serialized index (same format as [`Index::to_bytes`]) to `w`.
    /// The Bloom words and the fingerprint table are written straight from their
    /// in-memory arrays, so the only transient buffer is the MPH backend (~0.5 B/key).
    pub fn write_to<W: std::io::Write>(&self, w: W) -> std::io::Result<()> {
        use std::io::Write;
        let mut w = crate::wire::ChecksumWriter::new(std::io::BufWriter::new(w));
        let mut head = Vec::with_capacity(32);
        write_u8(&mut head, TAG_V3);
        head.extend_from_slice(FORMAT_MAGIC);
        write_u16(&mut head, FORMAT_VERSION);
        write_u8(&mut head, HASH_ID_CANONICAL);
        write_u8(&mut head, 0);
        write_u64(&mut head, self.key_count as u64);
        w.write_all(&head)?;
        if let Some(engine) = self.engine.as_ref() {
            let mut backend = Vec::new();
            write_u64(&mut backend, engine.prehash_seed);
            engine.backend.write_to(&mut backend);
            w.write_all(&backend)?;
            drop(backend);
            match &engine.filter {
                Some(bf) => {
                    w.write_all(&[1])?;
                    bf.write_into(&mut w)?;
                }
                None => w.write_all(&[0])?,
            }
            match &engine.fingerprints {
                Some(fp) => {
                    w.write_all(&[1])?;
                    let real = &fp[..fp.len() - 1];
                    w.write_all(&(real.len() as u64).to_le_bytes())?;
                    crate::wire::write_le(&mut w, real)?;
                }
                None => w.write_all(&[0])?,
            }
        }
        let mut inner = w.finish()?;
        inner.flush()
    }

    /// Write the index to a file (streaming, see [`Index::write_to`]).
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        self.write_to(file)
    }

    /// Read an index written by [`Index::save`] / [`Index::write_to`].
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, IndexError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    #[deprecated(since = "0.7.0", note = "alias; use to_bytes")]
    pub fn serialize(&self) -> Result<Vec<u8>, IndexError> {
        self.to_bytes()
    }

    /// Deserialize from [`Index::to_bytes`] output. Every structural invariant the
    /// unchecked lookup path relies on is verified, and for tag-4 data the checksum
    /// as well, so corrupt or hostile input is rejected rather than read out of bounds.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IndexError> {
        let mut cursor = Cursor::new(bytes);
        let tag = cursor.read_u8().ok_or(IndexError::CorruptData)?;
        if tag == TAG_V3 {
            return Self::from_bytes_v3(bytes);
        }
        let key_count = cursor.read_u64().ok_or(IndexError::CorruptData)? as usize;
        match tag {
            0 => {
                // Legacy MPH v1: always-present filter + fingerprints.
                let prehash_seed = cursor.read_u64().ok_or(IndexError::CorruptData)?;
                let mut pos = cursor.pos;
                let backend =
                    BackendDispatch::read_from(bytes, &mut pos).ok_or(IndexError::CorruptData)?;
                cursor.pos = pos;
                let mut bf_pos = cursor.pos;
                let filter = BlockBloom::read_from(bytes, &mut bf_pos)
                    .ok_or(IndexError::CorruptData)?;
                cursor.pos = bf_pos;
                let fingerprints = read_fingerprints(&mut cursor)?;
                Ok(Index {
                    engine: Some(MphEngine {
                        backend,
                        prehash_seed,
                        filter: Some(filter),
                        fingerprints: Some(fingerprints),
                    }),
                    key_count,
                })
            }
            2 => {
                // MPH v2: presence flags before filter/fingerprints.
                let prehash_seed = cursor.read_u64().ok_or(IndexError::CorruptData)?;
                let mut pos = cursor.pos;
                let backend =
                    BackendDispatch::read_from(bytes, &mut pos).ok_or(IndexError::CorruptData)?;
                cursor.pos = pos;
                let has_filter = cursor.read_u8().ok_or(IndexError::CorruptData)?;
                let filter = if has_filter == 1 {
                    let mut bf_pos = cursor.pos;
                    let bf = BlockBloom::read_from(bytes, &mut bf_pos)
                        .ok_or(IndexError::CorruptData)?;
                    cursor.pos = bf_pos;
                    Some(bf)
                } else {
                    None
                };
                let has_fp = cursor.read_u8().ok_or(IndexError::CorruptData)?;
                let fingerprints = if has_fp == 1 {
                    Some(read_fingerprints(&mut cursor)?)
                } else {
                    None
                };
                Ok(Index {
                    engine: Some(MphEngine {
                        backend,
                        prehash_seed,
                        filter,
                        fingerprints,
                    }),
                    key_count,
                })
            }
            3 => {
                if key_count != 0 {
                    return Err(IndexError::CorruptData);
                }
                Ok(Index::empty())
            }
            // Tag 1 (legacy PGM engine) removed in v0.6.
            _ => Err(IndexError::CorruptData),
        }
        .and_then(|idx| if idx.validate() { Ok(idx) } else { Err(IndexError::CorruptData) })
    }

    fn from_bytes_v3(bytes: &[u8]) -> Result<Self, IndexError> {
        // Checksum first: everything else is only parsed from verified bytes.
        if bytes.len() < 8 {
            return Err(IndexError::CorruptData);
        }
        let (body, trailer) = bytes.split_at(bytes.len() - 8);
        let stored = u64::from_le_bytes(trailer.try_into().unwrap());
        if crate::checksum::checksum(body) != stored {
            return Err(IndexError::CorruptData);
        }
        let mut cursor = Cursor::new(body);
        let _tag = cursor.read_u8().ok_or(IndexError::CorruptData)?;
        let magic = body.get(cursor.pos..cursor.pos + 4).ok_or(IndexError::CorruptData)?;
        if magic != FORMAT_MAGIC {
            return Err(IndexError::CorruptData);
        }
        cursor.pos += 4;
        let version = cursor.read_u16().ok_or(IndexError::CorruptData)?;
        if version != FORMAT_VERSION {
            return Err(IndexError::CorruptData);
        }
        let hash_id = cursor.read_u8().ok_or(IndexError::CorruptData)?;
        if hash_id != HASH_ID_CANONICAL {
            return Err(IndexError::CorruptData);
        }
        let _reserved = cursor.read_u8().ok_or(IndexError::CorruptData)?;
        let key_count = cursor.read_u64().ok_or(IndexError::CorruptData)? as usize;
        if cursor.pos == body.len() {
            return if key_count == 0 { Ok(Index::empty()) } else { Err(IndexError::CorruptData) };
        }
        let prehash_seed = cursor.read_u64().ok_or(IndexError::CorruptData)?;
        let mut pos = cursor.pos;
        let backend = BackendDispatch::read_from(body, &mut pos).ok_or(IndexError::CorruptData)?;
        cursor.pos = pos;
        let has_filter = cursor.read_u8().ok_or(IndexError::CorruptData)?;
        let filter = match has_filter {
            1 => {
                let mut bf_pos = cursor.pos;
                let bf = BlockBloom::read_from(body, &mut bf_pos).ok_or(IndexError::CorruptData)?;
                cursor.pos = bf_pos;
                Some(bf)
            }
            0 => None,
            _ => return Err(IndexError::CorruptData),
        };
        let has_fp = cursor.read_u8().ok_or(IndexError::CorruptData)?;
        let fingerprints = match has_fp {
            1 => Some(read_fingerprints(&mut cursor)?),
            0 => None,
            _ => return Err(IndexError::CorruptData),
        };
        if cursor.pos != body.len() {
            return Err(IndexError::CorruptData);
        }
        let idx = Index {
            engine: Some(MphEngine { backend, prehash_seed, filter, fingerprints }),
            key_count,
        };
        if idx.validate() { Ok(idx) } else { Err(IndexError::CorruptData) }
    }

    /// Invariants shared by every format: key count within the slot range and the
    /// fingerprint table covering exactly the slot range (it is indexed unchecked).
    fn validate(&self) -> bool {
        match &self.engine {
            None => self.key_count == 0,
            Some(engine) => {
                let cap = engine.backend.slot_capacity();
                if self.key_count == 0 || self.key_count > cap {
                    return false;
                }
                if let Some(fp) = &engine.fingerprints
                    && fp.len() != cap + 1
                {
                    return false;
                }
                true
            }
        }
    }

    #[deprecated(since = "0.7.0", note = "alias; use from_bytes")]
    pub fn deserialize(bytes: &[u8]) -> Result<Self, IndexError> {
        Self::from_bytes(bytes)
    }

    fn lookup_mph(&self, engine: &MphEngine, key: &[u8]) -> Result<usize, IndexError> {
        let canonical = canonical_hash_key(key, engine.prehash_seed);
        // Optional Bloom prefilter (skipped in lean_mph mode).
        if let Some(bf) = &engine.filter {
            if !bf.contains_hash(canonical) {
                return Err(IndexError::KeyNotFound);
            }
        }
        let idx = engine
            .backend
            .lookup(canonical)
            .ok_or(IndexError::KeyNotFound)? as usize;
        // Optional fingerprint verification (skipped in lean_mph mode).
        if let Some(fps) = &engine.fingerprints {
            let fp = fingerprint16_mph(canonical);
            // SAFETY: mph index is in [0..n), fingerprints.len() == n
            let ok = unsafe { *fps.get_unchecked(idx) == fp };
            if !ok {
                return Err(IndexError::KeyNotFound);
            }
        }
        Ok(idx)
    }

}

#[inline(always)]
#[allow(deprecated)]
fn make_backend_cfg(config: &IndexConfig) -> BackendConfig {
    BackendConfig {
        backend: config.backend,
        enable_parallel_build: config.enable_parallel_build,
        seed: config.mph_config.seed,
        lambda: config.mph_config.lambda,
        alpha: config.mph_config.alpha,
        rehash_limit: config.mph_config.max_rehash,
        build_profile: if config.build_fast_profile {
            BuildProfile::Fast
        } else {
            BuildProfile::Balanced
        },
    }
}

/// First byte of indexes written by this version (tags 0, 2, 3 are legacy).
const TAG_V3: u8 = 4;
const FORMAT_MAGIC: &[u8; 4] = b"KIRA";
const FORMAT_VERSION: u16 = 1;
/// Identifier of the canonical key hash: mix64 for 8-byte keys, AES-round hash for
/// everything else (see `canonical_hash`). A different id means the file was built
/// with a hash this version cannot reproduce.
const HASH_ID_CANONICAL: u8 = 1;

/// Number of canonical-hash seeds tried before concluding that two input keys are
/// byte-identical. Equal canonical hashes are detected exactly by the MPH build (equal
/// keys can never be separated by any pilot). A pair of *distinct* keys colliding on the
/// 64-bit canonical hash under two independent seeds has probability ~(n²/2⁶⁵)², which is
/// negligible at any practical `n`, so "collides under a second seed" ⇒ duplicate.
const MAX_PREHASH_ROUNDS: u32 = 4;

/// Run `f` on the persistent build pool (P-core pinned on hybrid CPUs) so every rayon
/// call inside the build lands there. Initialized once; avoids the ~150 ms Windows
/// CreateThread cost of a per-build pool.
fn run_in_build_pool<T: Send>(parallel: bool, f: impl FnOnce() -> T + Send) -> T {
    crate::build_pool::run(parallel, f)
}

/// The build pipeline:
///
/// 1. canonical 64-bit hash of every key, in parallel, straight from the caller's slice;
/// 2. partition the hashes into ~32K-key parts (`ptrhash25::partition_keys`);
/// 3. build every part in parallel — bucket scatter, pilot search and per-key slots all
///    stay inside L2 (`ptrhash25::build_partitioned`);
/// 4. Block-Bloom (parallel by block range) and u16 fingerprints (parallel by part,
///    written from the slots the build already computed) — skipped in lean mode.
///
/// There is no byte arena and no key copy: the only per-key allocations are the
/// canonical hash array and the partitioned hash array (8 B/key each).
fn build_engine<K>(keys: &[K], config: &IndexConfig) -> Result<MphEngine, IndexError>
where
    K: AsRef<[u8]> + Sync,
{
    let backend_cfg = make_backend_cfg(config);
    let mph_cfg = backend_cfg.mph_config();
    let base_seed = config.mph_config.seed;
    // `KIRA_BUILD_TRACE=1` prints per-phase timings to stderr.
    let trace = std::env::var_os("KIRA_BUILD_TRACE").is_some();
    let mut round = 0u32;
    loop {
        let prehash_seed = base_seed ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let n = keys.len();
        let t0 = std::time::Instant::now();
        let canonical = canonical_hash_all(keys, prehash_seed);
        let t1 = std::time::Instant::now();
        // Two big u64 buffers for the whole build (8 B/key each), never a third:
        //  - non-lean: Bloom groups `canonical` (A) by filter window into B, then the
        //    partition reads B and scatters into A, which becomes the part-key array;
        //  - lean: the partition reads A and scatters into a fresh B.
        // Filter and part builds only need *a* permutation of the hashes, so no copy of
        // the original order is ever kept. At 100M keys this avoids faulting in 800 MB.
        let (filter, part) = if config.lean_mph {
            let part = crate::ptrhash25::partition_keys(canonical.as_slice(), &mph_cfg);
            drop(canonical);
            (None, part)
        } else {
            let mut grouped = crate::hugepage::HugeVec::<u64>::zeroed(n);
            let filter =
                BlockBloom::build_from_prehashed_into(canonical.as_slice(), grouped.as_mut_slice());
            let part =
                crate::ptrhash25::partition_keys_into(grouped.as_slice(), &mph_cfg, canonical);
            drop(grouped);
            (Some(filter), part)
        };
        let t2 = std::time::Instant::now();
        let (backend, fp16) = match PtrHash25Backend::build_from_partitioned(
            &part,
            &backend_cfg,
            !config.lean_mph,
        ) {
            Ok(v) => v,
            Err(MphError::DuplicateKey) => {
                round += 1;
                if round >= MAX_PREHASH_ROUNDS {
                    return Err(IndexError::DuplicateKey);
                }
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        let t3 = std::time::Instant::now();
        // Outer u16 fingerprints were filled per part inside the MPH build; one
        // padding element keeps the batched gather inside the allocation.
        let fingerprints = fp16.map(|mut v| {
            v.push(0);
            v.into_boxed_slice()
        });
        if trace {
            eprintln!(
                "[kira_kv_engine build] n={} parts={} | canonical {:?} | bloom+partition {:?} | mph+fingerprints {:?} | total {:?}",
                n,
                part.num_parts(),
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t3 - t0
            );
        }
        return Ok(MphEngine {
            backend: BackendDispatch::PtrHash25(backend),
            prehash_seed,
            filter,
            fingerprints,
        });
    }
}

/// Canonical hash of every key, in parallel chunks of 4K keys. The output buffer is
/// hugepage-backed when the process has the privilege and is first-touched
/// sequentially otherwise (see `hugepage::prefault`).
fn canonical_hash_all<K>(keys: &[K], seed: u64) -> crate::hugepage::HugeVec<u64>
where
    K: AsRef<[u8]> + Sync,
{
    let mut out = crate::hugepage::HugeVec::<u64>::zeroed(keys.len());
    crate::hugepage::prefault(out.as_mut_slice());
    const CHUNK: usize = 4096;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.as_mut_slice()
            .par_chunks_mut(CHUNK)
            .zip(keys.par_chunks(CHUNK))
            .for_each(|(o, kc)| hash_chunk(o, kc, seed));
    }
    #[cfg(not(feature = "parallel"))]
    {
        hash_chunk(out.as_mut_slice(), keys, seed);
    }
    out
}

/// Hash one chunk of keys. With `Vec<Vec<u8>>` inputs whose bodies are scattered
/// across the heap (e.g. shuffled), this phase is bound by TLB misses on the key
/// bodies — software prefetch was measured to make no difference (PF = 0 / 16 / 64 all
/// within noise at 10M keys), so the loop is kept plain.
#[inline]
fn hash_chunk<K: AsRef<[u8]>>(out: &mut [u64], keys: &[K], seed: u64) {
    for (o, k) in out.iter_mut().zip(keys) {
        *o = canonical_hash_key(k.as_ref(), seed);
    }
}

/// Drop a large key vector using all build threads. Freeing 10M small allocations
/// single-threaded costs ~200 ms on the Windows heap.
fn drop_keys_parallel<K: Send>(keys: Vec<K>) {
    #[cfg(feature = "parallel")]
    {
        if keys.len() >= 1 << 16 && std::mem::needs_drop::<K>() {
            use rayon::prelude::*;
            crate::build_pool::pool().install(|| keys.into_par_iter().for_each(drop));
            return;
        }
    }
    drop(keys);
}

#[inline(always)]
fn canonical_hash_key(key: &[u8], seed: u64) -> u64 {
    crate::canonical_hash::canonical_hash_bytes(key, seed)
}

#[inline(always)]
fn prefetch_key_batch(keys: &[&[u8]], i: usize, window: usize) {
    // Distance 8 keeps the prefetched line in L1 for ~50-100 cycles before use,
    // which fits the per-key work budget. Distance 24 (old value) overshoots and
    // evicts useful lines on small queries (the common case here).
    const DIST: usize = 8;
    let pf = i + DIST;
    if pf + window <= keys.len() {
        for j in 0..window {
            let slice = unsafe { *keys.get_unchecked(pf + j) };
            if !slice.is_empty() {
                // Prefetch the actual key bytes, not the &[u8] header (which is already in L1).
                prefetch_read(slice.as_ptr());
            }
        }
    }
}

/// Builder for index
pub struct IndexBuilder {
    config: IndexConfig,
}

impl IndexBuilder {
    pub fn new() -> Self {
        Self {
            config: IndexConfig::default(),
        }
    }

    pub fn with_config(mut self, config: IndexConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_mph_config(mut self, mph_config: MphConfig) -> Self {
        self.config.mph_config = mph_config;
        self
    }

    #[deprecated(since = "0.7.0", note = "no effect; use PgmBuilder::with_epsilon")]
    #[allow(deprecated)]
    pub fn with_pgm_epsilon(mut self, epsilon: u32) -> Self {
        self.config.pgm_epsilon = epsilon;
        self
    }

    #[deprecated(since = "0.7.0", note = "no effect; PtrHash25 is the only backend")]
    #[allow(deprecated)]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.config.backend = backend;
        self
    }

    #[deprecated(since = "0.7.0", note = "no effect")]
    #[allow(deprecated)]
    pub fn with_hot_fraction(mut self, hot_fraction: f32) -> Self {
        self.config.hot_fraction = hot_fraction;
        self
    }

    pub fn with_parallel_build(mut self, enabled: bool) -> Self {
        self.config.enable_parallel_build = enabled;
        self
    }

    #[deprecated(since = "0.7.0", note = "no effect; duplicates are always detected")]
    #[allow(deprecated)]
    pub fn with_build_fast_profile(mut self, enabled: bool) -> Self {
        self.config.build_fast_profile = enabled;
        self
    }

    #[deprecated(since = "0.7.0", note = "no effect; Index is always MPH-backed")]
    #[allow(deprecated)]
    pub fn auto_detect_numeric(mut self, enabled: bool) -> Self {
        self.config.auto_detect_numeric = enabled;
        self
    }

    #[deprecated(since = "0.7.0", note = "no effect; use PgmBuilder::with_bloom_filter")]
    #[allow(deprecated)]
    pub fn with_pgm_bloom(mut self, enabled: bool) -> Self {
        self.config.pgm_enable_bloom = enabled;
        self
    }

    #[deprecated(since = "0.7.0", note = "no effect; use PgmBuilder::with_elias_fano")]
    #[allow(deprecated)]
    pub fn with_pgm_elias_fano(mut self, enabled: bool) -> Self {
        self.config.pgm_enable_elias_fano = enabled;
        self
    }

    #[deprecated(since = "0.7.0", note = "no effect; use PgmBuilder::with_target_lookup_ns")]
    #[allow(deprecated)]
    pub fn with_pgm_target_lookup_ns(mut self, ns: u32) -> Self {
        self.config.pgm_target_lookup_ns = Some(ns);
        self
    }

    /// Enable Lean MPH mode — drops Bloom filter + outer fingerprints + inner
    /// PtrHash25 fingerprints. Saves ~50% memory (4.5 → 2.5 B/key on PtrHash25)
    /// at the cost of negative-lookup safety: foreign keys return arbitrary
    /// in-range positions instead of `KeyNotFound`.
    ///
    /// Use ONLY when you can guarantee every queried key was in the build set
    /// (preloaded dictionaries, closed vocabularies, deduped row IDs).
    pub fn with_lean_mph(mut self, enabled: bool) -> Self {
        self.config.lean_mph = enabled;
        self
    }

    pub fn build_index<K>(self, keys: Vec<K>) -> Result<Index, IndexError>
    where
        K: AsRef<[u8]> + Send + Sync,
    {
        Index::build_index(keys, self.config)
    }

    /// Build from borrowed keys — no clone, no deallocation inside the build. See
    /// [`Index::build_index_ref`].
    pub fn build_index_ref<K>(self, keys: &[K]) -> Result<Index, IndexError>
    where
        K: AsRef<[u8]> + Sync,
    {
        Index::build_index_ref(keys, self.config)
    }
}

impl Default for IndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn fingerprint16_mph(canonical: u64) -> u16 {
    (canonical & 0xFFFF) as u16
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_u8(&mut self) -> Option<u8> {
        if self.pos + 1 > self.buf.len() {
            return None;
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        Some(v)
    }

    fn read_u16(&mut self) -> Option<u16> {
        if self.pos + 2 > self.buf.len() {
            return None;
        }
        let mut array = [0u8; 2];
        array.copy_from_slice(&self.buf[self.pos..self.pos + 2]);
        self.pos += 2;
        Some(u16::from_le_bytes(array))
    }

    fn read_u64(&mut self) -> Option<u64> {
        if self.pos + 8 > self.buf.len() {
            return None;
        }
        let mut array = [0u8; 8];
        array.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Some(u64::from_le_bytes(array))
    }
}

fn write_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

fn write_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_fingerprints(cursor: &mut Cursor<'_>) -> Result<Box<[u16]>, IndexError> {
    let len = cursor.read_u64().ok_or(IndexError::CorruptData)? as usize;
    let mut fps: Vec<u16> = crate::wire::read_le_at(cursor.buf, &mut cursor.pos, len)
        .ok_or(IndexError::CorruptData)?;
    fps.push(0); // padding element, see `MphEngine::fingerprints`
    Ok(fps.into_boxed_slice())
}
