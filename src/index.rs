use crate::block_bloom::BlockBloom;
use crate::fingerprints::{FingerprintBits, Fingerprints};
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
    /// Optional 16/24/32-bit membership fingerprints. Mapped tables retain their
    /// owner; packed 24-bit entries carry one padding byte for unaligned loads.
    fingerprints: Option<Fingerprints>,
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
/// Live options are `mph_config`, `enable_parallel_build`, `lean_mph`,
/// `bloom_filter` and `fingerprint_bits`.
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
    /// Lean MPH mode drops Bloom and outer fingerprints. With default geometry
    /// this reduces the index from about 3.79 to 0.42 B/key. The inner MPH never
    /// stores an additional fingerprint table in an Index.
    ///
    /// **Trade-off**: a lookup with a key that wasn't in the build set returns
    /// an arbitrary in-range position instead of `KeyNotFound`. Use only when
    /// you can guarantee all queries are valid keys (e.g. preloaded dictionary,
    /// closed-world vocabulary, deduped log lines).
    pub lean_mph: bool,
    /// Independent prefilter; ignored in lean mode.
    pub bloom_filter: bool,
    /// Outer membership fingerprint width; ignored in lean mode.
    pub fingerprint_bits: FingerprintBits,
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
    /// Legacy fp16 export; None for wider fingerprints.
    pub fingerprints: Option<Vec<u16>>,
    /// Actual width and little-endian packed bytes for accelerator membership checks.
    pub fingerprint_bits: u8,
    pub fingerprint_bytes: Option<Vec<u8>>,
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
        Ok(Index { engine: Some(engine), key_count: keys.len() })
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

    #[inline]
    pub fn lookup_u64(&self, key: u64) -> Result<usize, IndexError> {
        let Some(engine) = self.engine.as_ref() else {
            return Err(IndexError::KeyNotFound);
        };
        self.lookup_mph(engine, &key.to_le_bytes())
    }

    /// SIMD-hashed u64 lookup with three stages over 64 independent queries:
    /// pilot/Bloom prefetch, slot calculation/fingerprint prefetch, verification.
    /// Tiny batches use the scalar path to avoid pipeline setup overhead.
    ///
    /// Available only for u64 keys (cannot vectorize variable-length byte hashing).
    /// Returns Some(idx) on hit, None on miss (filter or fingerprint reject).
    ///
    /// **Allocation profile**: allocates `out: Vec<Option<usize>>` + a `canon:
    /// Vec<u64>` scratch buffer (each `keys.len()` long) on every call. For
    /// hot paths that call this millions of times per second (e.g.
    /// per-read seeding in an aligner), prefer
    /// [`Index::lookup_batch_u64_simd_into`] which takes both buffers from the
    /// caller and does zero allocation.
    pub fn lookup_batch_u64_simd(&self, keys: &[u64]) -> Vec<Option<usize>> {
        let mut out = vec![None; keys.len()];
        let mut canon = vec![0u64; keys.len()];
        self.lookup_batch_u64_simd_into(keys, &mut canon, &mut out);
        out
    }

    /// Zero-allocation variant of [`Index::lookup_batch_u64_simd`]. The caller
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
        let Some(engine) = self.engine.as_ref() else {
            out[..n].fill(None);
            return;
        };
        crate::simd_hash::hash_u64(keys, engine.prehash_seed, &mut canon[..n]);
        Self::lookup_canonical_batch(engine, &canon[..n], &mut out[..n]);
    }

    /// Three stages across 64 independent queries: pilot/Bloom prefetch,
    /// slot calculation/fingerprint prefetch, then fingerprint validation.
    /// Every output is overwritten, including misses when scratch is reused.
    fn lookup_canonical_batch(engine: &MphEngine, hashes: &[u64], out: &mut [Option<usize>]) {
        use crate::mph_backend::PtrHash25Storage;
        let BackendDispatch::PtrHash25(backend) = &engine.backend;
        if hashes.len() < 16 {
            for (&hash, result) in hashes.iter().zip(out) {
                *result = if engine.filter.as_ref().is_some_and(|bf| !bf.contains_hash(hash)) {
                    None
                } else {
                    engine.backend.lookup(hash).map(|s| s as usize).filter(|&s| {
                        engine.fingerprints.as_ref().is_none_or(|fp| fp.matches(s, hash))
                    })
                };
            }
            return;
        }
        let PtrHash25Storage::Mph(mph) = &backend.storage else {
            for (&hash, result) in hashes.iter().zip(out) {
                *result =
                    engine.backend.lookup(hash).map(|s| s as usize).filter(|&s| {
                        engine.fingerprints.as_ref().is_none_or(|fp| fp.matches(s, hash))
                    });
            }
            return;
        };
        const LANES: usize = 64;
        for (hashes, out) in hashes.chunks(LANES).zip(out.chunks_mut(LANES)) {
            let mut prepared = [crate::ptrhash25::PreparedLookup::default(); LANES];
            let mut slots = [usize::MAX; LANES];
            for (i, &hash) in hashes.iter().enumerate() {
                prepared[i] = mph.prepare_lookup(hash);
                if let Some(bf) = &engine.filter {
                    prefetch_read(bf.block_ptr(hash));
                }
            }
            for (i, &hash) in hashes.iter().enumerate() {
                if engine.filter.as_ref().is_some_and(|bf| !bf.contains_hash(hash)) {
                    continue;
                }
                let slot = mph.finish_lookup(prepared[i]);
                slots[i] = slot;
                if let Some(fp) = &engine.fingerprints {
                    fp.prefetch(slot);
                }
            }
            for (i, &hash) in hashes.iter().enumerate() {
                let slot = slots[i];
                out[i] = if slot != usize::MAX
                    && engine.fingerprints.as_ref().is_none_or(|fp| fp.matches(slot, hash))
                {
                    Some(slot)
                } else {
                    None
                };
            }
        }
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
            let ok = fps.matches(idx, canonical);
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
    /// (~0.5% at 11 bits/key). Without Bloom, checks the configured fingerprints.
    /// `lookup(..).is_ok()` uses all configured filters but remains probabilistic.
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
            // Without Bloom, use the fingerprints. Lean lookups cannot fail.
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

    /// Batched byte-key lookup through the same three-stage pipeline as the u64
    /// API. Uses bounded stack scratch and preserves query order, including misses.
    pub fn lookup_batch_pipelined(&self, keys: &[&[u8]]) -> Vec<Option<usize>> {
        let Some(engine) = self.engine.as_ref() else { return vec![None; keys.len()] };
        let mut out = vec![None; keys.len()];
        // Bounded scratch; byte-key batches no longer allocate a full hash array.
        let mut hashes = [0u64; 64];
        for (input, result) in keys.chunks(64).zip(out.chunks_mut(64)) {
            for (h, key) in hashes.iter_mut().zip(input) {
                *h = canonical_hash_key(key, engine.prehash_seed);
            }
            Self::lookup_canonical_batch(engine, &hashes[..input.len()], result);
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

        let fingerprints = engine.fingerprints.as_ref().and_then(Fingerprints::export_low16);
        let fingerprint_bits = engine.fingerprints.as_ref().map_or(0, |fp| fp.width().bits());
        let fingerprint_bytes = engine.fingerprints.as_ref().map(|fp| {
            let mut bytes = fp.mmap_bytes();
            bytes.truncate(fp.memory_usage());
            bytes
        });

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
            fingerprint_bits,
            fingerprint_bytes,
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
        let fp_memory = engine.fingerprints.as_ref().map(Fingerprints::memory_usage).unwrap_or(0);
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
            println!("  MPH index: {:.2} MB", stats.mph_memory as f64 / 1_048_576.0);
        }
        if stats.pgm_memory > 0 {
            println!("  PGM index: {:.2} MB", stats.pgm_memory as f64 / 1_048_576.0);
        }
    }

    /// Save aligned, checksummed tables for zero-copy loading. The rare map
    /// fallback is stored as a legacy payload and remains readable.
    pub fn save_mmap<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), IndexError> {
        use crate::mmap_index::{MmapIndexWriter, SectionKind};
        let mut w = MmapIndexWriter::create(path, self.key_count as u64)?;
        if let Some(e) = &self.engine {
            use crate::mph_backend::PtrHash25Storage;
            let BackendDispatch::PtrHash25(b) = &e.backend;
            if let PtrHash25Storage::Mph(m) = &b.storage {
                let (bseed, bshift) = e
                    .filter
                    .as_ref()
                    .map(|f| {
                        let (s, b, _) = f.mmap_data();
                        (s, b as u64)
                    })
                    .unwrap_or((0, 0));
                let fields = [
                    e.fingerprints.as_ref().map_or(16, |fp| fp.width().bits() as u64),
                    e.prehash_seed,
                    m.n,
                    m.num_buckets as u64,
                    m.salt,
                    m.use_aes_hash as u64,
                    m.prerotate as u64,
                    m.part_salt,
                    bseed,
                    bshift,
                ];
                let mut meta = Vec::new();
                crate::wire::extend_le(&mut meta, &fields);
                w.add_section(SectionKind::PtrHash25Meta, meta);
                w.add_section(SectionKind::PtrHash25Pilots, m.pilots.as_slice().to_vec());
                let mut parts = Vec::new();
                for p in &m.parts {
                    crate::wire::extend_le(&mut parts, &[p.slot_off, p.salt]);
                    crate::wire::extend_le(
                        &mut parts,
                        &[
                            p.bucket_off,
                            p.num_keys,
                            p.num_slots,
                            p.num_buckets,
                            p.large_buckets,
                            p.remap_off,
                        ],
                    );
                }
                w.add_section(SectionKind::PtrHash25Parts, parts);
                let mut remap = Vec::new();
                crate::wire::extend_le(&mut remap, &m.remap);
                w.add_section(SectionKind::PtrHash25Remap, remap);
                if !m.fingerprints.is_empty() {
                    w.add_section(SectionKind::PtrHash25Fingerprints, m.fingerprints.to_vec());
                }
                if let Some(f) = &e.filter {
                    let mut words = Vec::new();
                    crate::wire::extend_le(&mut words, f.mmap_data().2);
                    w.add_section(SectionKind::BlockBloomWords, words);
                }
                if let Some(f) = &e.fingerprints {
                    w.add_section(SectionKind::OuterFingerprints, f.mmap_bytes());
                }
            } else {
                w.add_section(SectionKind::LegacyPayload, self.to_bytes()?);
            }
        } else {
            w.add_section(SectionKind::LegacyPayload, self.to_bytes()?);
        }
        Ok(w.finalize()?)
    }

    /// Open aligned tables without copying on little-endian Windows. A retained
    /// read handle excludes write/delete opens until the index is dropped.
    /// Other platforms use owned file bytes for safety; use `open_mmap_unchecked`
    /// there when the application guarantees an immutable backing file.
    /// Legacy payloads still load through the validated copying decoder.
    pub fn open_mmap<P: AsRef<std::path::Path>>(path: P) -> Result<Self, IndexError> {
        Self::from_mmap_reader(crate::mmap_index::MmapIndex::open(path)?)
    }

    /// Whether the lookup tables reference an OS file mapping. False for owned
    /// builds, legacy payloads, empty indexes and the safe Unix byte-buffer fallback.
    pub fn is_memory_mapped(&self) -> bool {
        let Some(engine) = &self.engine else {
            return false;
        };
        let BackendDispatch::PtrHash25(backend) = &engine.backend;
        match &backend.storage {
            crate::mph_backend::PtrHash25Storage::Mph(m) => m.pilots.is_memory_mapped(),
            _ => false,
        }
    }

    /// Zero-copy opener on Unix as well as Windows.
    /// # Safety
    /// The backing file must not be modified or truncated while the index lives,
    /// including by other processes. Prefer `open_mmap` on Windows.
    pub unsafe fn open_mmap_unchecked<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, IndexError> {
        Self::from_mmap_reader(unsafe { crate::mmap_index::MmapIndex::open_unchecked(path)? })
    }

    fn from_mmap_reader(mmap: crate::mmap_index::MmapIndex) -> Result<Self, IndexError> {
        use crate::mmap_index::SectionKind;
        let header = mmap.parse_header().map_err(|e| match e.kind() {
            std::io::ErrorKind::InvalidData => IndexError::CorruptData,
            _ => IndexError::Io(e),
        })?;
        if let Some(meta) = mmap.section(&header, SectionKind::PtrHash25Meta) {
            let bad = || IndexError::CorruptData;
            // Aligned tables have no inner payload checksum. Never accept a
            // version downgrade that would bypass the container checksum.
            if header.version != 2 {
                return Err(bad());
            }
            if meta.len() != 80 {
                return Err(bad());
            }
            let f: Vec<u64> =
                meta.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)).collect();
            if FingerprintBits::from_bits(f[0]).is_none() || f[5] > 1 || f[6] >= 64 {
                return Err(bad());
            }
            let raw = mmap.section(&header, SectionKind::PtrHash25Parts).ok_or_else(bad)?;
            if !raw.len().is_multiple_of(40) {
                return Err(bad());
            }
            let parts: Vec<crate::ptrhash25::PartInfo> = raw
                .as_chunks::<40>()
                .0
                .iter()
                .map(|b| {
                    let u64_at = |i| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
                    let u32_at = |i| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
                    crate::ptrhash25::PartInfo {
                        slot_off: u64_at(0),
                        salt: u64_at(8),
                        bucket_off: u32_at(16),
                        num_keys: u32_at(20),
                        num_slots: u32_at(24),
                        num_buckets: u32_at(28),
                        large_buckets: u32_at(32),
                        remap_off: u32_at(36),
                    }
                })
                .collect();
            let m = crate::ptrhash25::PtrHash25Mphf {
                n: f[2],
                num_buckets: u32::try_from(f[3]).map_err(|_| bad())?,
                salt: f[4],
                pilots: mmap.table(&header, SectionKind::PtrHash25Pilots).ok_or_else(bad)?,
                remap: mmap.table(&header, SectionKind::PtrHash25Remap).ok_or_else(bad)?,
                fingerprints: mmap
                    .section(&header, SectionKind::PtrHash25Fingerprints)
                    .unwrap_or(&[])
                    .to_vec(),
                use_aes_hash: f[5] != 0,
                prerotate: f[6] as u8,
                part_salt: f[7],
                parts: parts.into_boxed_slice(),
            };
            if !m.validate() {
                return Err(bad());
            }
            let filter = if mmap.section(&header, SectionKind::BlockBloomWords).is_some() {
                Some(
                    BlockBloom::mapped(
                        f[8],
                        u32::try_from(f[9]).map_err(|_| bad())?,
                        mmap.table(&header, SectionKind::BlockBloomWords).ok_or_else(bad)?,
                    )
                    .ok_or_else(bad)?,
                )
            } else {
                None
            };
            let fingerprints = if mmap.section(&header, SectionKind::OuterFingerprints).is_some() {
                Some(match FingerprintBits::from_bits(f[0]).ok_or_else(bad)? {
                    FingerprintBits::Bits16 => Fingerprints::u16_table(
                        mmap.table(&header, SectionKind::OuterFingerprints).ok_or_else(bad)?,
                    ),
                    width => Fingerprints::packed(
                        mmap.table(&header, SectionKind::OuterFingerprints).ok_or_else(bad)?,
                        width,
                        usize::try_from(header.key_count).map_err(|_| bad())?,
                    ),
                })
            } else {
                None
            };
            let idx = Self {
                key_count: usize::try_from(header.key_count).map_err(|_| bad())?,
                engine: Some(MphEngine {
                    backend: BackendDispatch::PtrHash25(PtrHash25Backend {
                        storage: crate::mph_backend::PtrHash25Storage::Mph(m),
                    }),
                    prehash_seed: f[1],
                    filter,
                    fingerprints,
                }),
            };
            return if idx.validate() { Ok(idx) } else { Err(bad()) };
        }
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
                    w.write_all(&[fp.wire_flag()])?;
                    fp.write_into(&mut w)?;
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
                let filter =
                    BlockBloom::read_from(bytes, &mut bf_pos).ok_or(IndexError::CorruptData)?;
                cursor.pos = bf_pos;
                let fingerprints = read_fingerprints(&mut cursor, 1)?;
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
                    let bf =
                        BlockBloom::read_from(bytes, &mut bf_pos).ok_or(IndexError::CorruptData)?;
                    cursor.pos = bf_pos;
                    Some(bf)
                } else {
                    None
                };
                let has_fp = cursor.read_u8().ok_or(IndexError::CorruptData)?;
                let fingerprints =
                    if has_fp == 1 { Some(read_fingerprints(&mut cursor, 1)?) } else { None };
                Ok(Index {
                    engine: Some(MphEngine { backend, prehash_seed, filter, fingerprints }),
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
            1..=3 => Some(read_fingerprints(&mut cursor, has_fp)?),
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
                    && !fp.validate(cap)
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

    #[inline(always)]
    fn lookup_mph(&self, engine: &MphEngine, key: &[u8]) -> Result<usize, IndexError> {
        let canonical = canonical_hash_key(key, engine.prehash_seed);
        // Optional Bloom prefilter (skipped in lean_mph mode).
        if let Some(bf) = &engine.filter
            && !bf.contains_hash(canonical)
        {
            return Err(IndexError::KeyNotFound);
        }
        let idx = engine.backend.lookup(canonical).ok_or(IndexError::KeyNotFound)? as usize;
        // Optional fingerprint verification (skipped in lean_mph mode).
        if let Some(fps) = &engine.fingerprints {
            let ok = fps.matches(idx, canonical);
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
const HASH_ID_CANONICAL: u8 = crate::wire::KIND_INDEX;

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
        let (filter, part) = if config.lean_mph || !config.bloom_filter {
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
            !config.lean_mph && config.fingerprint_bits == FingerprintBits::Bits16,
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
        // Default fp16 is filled inside the MPH build. Wider packed tables
        // are filled from the original canonical hashes and the final slots.
        let fingerprints = if config.lean_mph {
            None
        } else if config.fingerprint_bits == FingerprintBits::Bits16 {
            fp16.map(Fingerprints::low16)
        } else {
            let stride = config.fingerprint_bits.bits() as usize / 8;
            let mut data = vec![0u8; n * stride + 1];
            for i in 0..part.len() {
                let hash = part.original_key(i);
                let slot = match &backend.storage {
                    crate::mph_backend::PtrHash25Storage::Mph(m) => m.index_u64(hash),
                    crate::mph_backend::PtrHash25Storage::Map(m) => m[&hash],
                } as usize;
                data[slot * stride..(slot + 1) * stride]
                    .copy_from_slice(&hash.to_le_bytes()[..stride]);
            }
            Some(Fingerprints::packed(data.into(), config.fingerprint_bits, n))
        };
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
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        const CHUNK: usize = 4096;
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
        Self { config: IndexConfig::default() }
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

    pub fn with_bloom_filter(mut self, enabled: bool) -> Self {
        self.config.bloom_filter = enabled;
        self
    }

    pub fn with_fingerprint_bits(mut self, width: FingerprintBits) -> Self {
        self.config.fingerprint_bits = width;
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

    /// Enable Lean MPH mode: drops Bloom and outer fingerprints, reducing the
    /// default index from about 3.79 to 0.42 B/key
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

fn read_fingerprints(cursor: &mut Cursor<'_>, flag: u8) -> Result<Fingerprints, IndexError> {
    let len = usize::try_from(cursor.read_u64().ok_or(IndexError::CorruptData)?)
        .map_err(|_| IndexError::CorruptData)?;
    if flag == 1 {
        let fps: Vec<u16> = crate::wire::read_le_at(cursor.buf, &mut cursor.pos, len)
            .ok_or(IndexError::CorruptData)?;
        return Ok(Fingerprints::low16(fps));
    }
    let width = FingerprintBits::from_bits((flag as u64 + 1) * 8).ok_or(IndexError::CorruptData)?;
    let bytes = len.checked_mul(width.bits() as usize / 8).ok_or(IndexError::CorruptData)?;
    let end = cursor.pos.checked_add(bytes).ok_or(IndexError::CorruptData)?;
    let mut data = cursor.buf.get(cursor.pos..end).ok_or(IndexError::CorruptData)?.to_vec();
    cursor.pos = end;
    data.push(0);
    Ok(Fingerprints::packed(data.into(), width, len))
}
