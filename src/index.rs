use crate::block_bloom::BlockBloom;
use crate::canonical_hash::CanonicalHashVersion;
use crate::fingerprints::{FingerprintBits, Fingerprints};
use crate::mph_backend::{BackendDispatch, BackendKind, PtrHash25Backend};
use crate::pgm::PgmError;
use crate::ptrhash25::{BuildConfig as MphConfig, PtrHash25Error as MphError};
use thiserror::Error;

use crate::prefetch::prefetch_read;

#[derive(Debug)]
struct MphEngine {
    backend: BackendDispatch,
    prehash_seed: u64,
    hash_version: CanonicalHashVersion,
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
    /// A build option is outside its supported domain.
    #[error("invalid build configuration: {0}")]
    InvalidConfig(&'static str),
    /// The requested table cannot be represented by the index or target platform.
    #[error("index capacity exceeded: {0}")]
    CapacityExceeded(&'static str),
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

impl From<crate::ptrhash25::BuildValidationError> for IndexError {
    fn from(err: crate::ptrhash25::BuildValidationError) -> Self {
        match err {
            crate::ptrhash25::BuildValidationError::InvalidConfig(reason) => {
                Self::InvalidConfig(reason)
            }
            crate::ptrhash25::BuildValidationError::CapacityExceeded(reason) => {
                Self::CapacityExceeded(reason)
            }
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
    /// MPH geometry, seed, hash choice and retry limit. The low-level
    /// `with_fingerprints` option applies to raw u64 MPH builders only;
    /// this index controls membership using `fingerprint_bits` and `lean_mph`.
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
    /// Number of 64-byte blocks. Small filters use any positive count; large
    /// builds round up to a multiple of 256 (no longer a power of two
    /// since the partitioned build; the multiply-high reduction is unbiased regardless).
    pub blocks: usize,
    /// Concatenated 64-bit words. Length is `blocks * 8`.
    pub words: Vec<u64>,
}

impl Index {
    /// Build from owned keys. Equivalent to [`Index::build_index_ref`] followed by
    /// dropping `keys` (large drops run in parallel when enabled in the config).
    ///
    /// Prefer [`Index::build_index_ref`] when the keys stay alive anyway: it avoids the
    /// clone on the caller side and the deallocation cost inside the build.
    pub fn build_index<K>(keys: Vec<K>, config: IndexConfig) -> Result<Self, IndexError>
    where
        K: AsRef<[u8]> + Send + Sync,
    {
        let parallel = config.enable_parallel_build;
        let index = Self::build_index_ref(&keys, config)?;
        drop_keys_parallel(keys, parallel);
        Ok(index)
    }

    /// Build from borrowed keys (must be unique). Keys are hashed in place, using
    /// parallel workers only when enabled in the config.
    ///
    /// Duplicate keys are detected exactly (`IndexError::DuplicateKey`).
    pub fn build_index_ref<K>(keys: &[K], config: IndexConfig) -> Result<Self, IndexError>
    where
        K: AsRef<[u8]> + Sync,
    {
        config.mph_config.validate_for_len(keys.len())?;
        if keys.is_empty() {
            return Ok(Self::empty());
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
        Self::lookup_canonical_batch(engine, &canon[..n], &mut out[..n], &mut false);
    }

    /// Three stages across 64 independent queries: pilot/Bloom prefetch,
    /// slot calculation/fingerprint prefetch, then fingerprint validation.
    /// Every output is overwritten, including misses when scratch is reused.
    fn lookup_canonical_batch(
        engine: &MphEngine,
        hashes: &[u64],
        out: &mut [Option<usize>],
        bloom_first: &mut bool,
    ) {
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
                *result = if engine.filter.as_ref().is_some_and(|bf| !bf.contains_hash(hash)) {
                    None
                } else {
                    engine.backend.lookup(hash).map(|s| s as usize).filter(|&s| {
                        engine.fingerprints.as_ref().is_none_or(|fp| fp.matches(s, hash))
                    })
                };
            }
            return;
        };
        const LANES: usize = 64;
        if engine.filter.is_none() && engine.fingerprints.is_none() {
            // Lean indexes have no membership checks. Produce their results in
            // the pilot stage instead of creating a third pass and slot scratch.
            for (hashes, out) in hashes.chunks(LANES).zip(out.chunks_mut(LANES)) {
                let mut prepared = [crate::ptrhash25::PreparedLookup::default(); LANES];
                for (query, &hash) in prepared.iter_mut().zip(hashes) {
                    *query = mph.prepare_lookup(hash);
                }
                for (query, result) in prepared.iter().zip(out) {
                    *result = Some(mph.finish_lookup(*query));
                }
            }
            return;
        }
        for (hashes, out) in hashes.chunks(LANES).zip(out.chunks_mut(LANES)) {
            let mut prepared = [crate::ptrhash25::PreparedLookup::default(); LANES];
            let mut slots = [usize::MAX; LANES];
            let mut prefetch_slot = |i: usize, query| {
                let slot = mph.finish_lookup(query);
                slots[i] = slot;
                if let Some(fp) = &engine.fingerprints {
                    fp.prefetch(slot);
                }
            };
            let mut count = 0;
            if let Some(filter) = engine.filter.as_ref().filter(|_| *bloom_first) {
                for &hash in hashes {
                    prefetch_read(filter.block_ptr(hash));
                }
                let mut survivors = [0u8; LANES];
                for (i, &hash) in hashes.iter().enumerate() {
                    if filter.contains_hash(hash) {
                        survivors[count] = i as u8;
                        prepared[count] = mph.prepare_lookup(hash);
                        count += 1;
                    }
                }
                for (&i, &query) in survivors[..count].iter().zip(&prepared) {
                    prefetch_slot(i as usize, query);
                }
            } else {
                for (i, &hash) in hashes.iter().enumerate() {
                    prepared[i] = mph.prepare_lookup(hash);
                    if let Some(filter) = &engine.filter {
                        prefetch_read(filter.block_ptr(hash));
                    }
                }
                for (i, &hash) in hashes.iter().enumerate() {
                    if engine.filter.as_ref().is_none_or(|f| f.contains_hash(hash)) {
                        prefetch_slot(i, prepared[i]);
                        count += 1;
                    }
                }
            }
            // Use the observed Bloom rejection rate to schedule the next block.
            *bloom_first = count <= hashes.len() / 4;
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
        let canonical = engine.hash_version.hash(&key.to_le_bytes(), engine.prehash_seed);
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
        self.membership_mode() != crate::MembershipMode::ClosedWorld
    }

    /// Membership guarantee of [`Index::lookup`]. Fingerprints, including fp32,
    /// only provide probabilistic membership. Empty indexes reject every key.
    pub fn membership_mode(&self) -> crate::MembershipMode {
        let Some(engine) = &self.engine else {
            return crate::MembershipMode::Exact;
        };
        let BackendDispatch::PtrHash25(backend) = &engine.backend;
        // The MPH backend uses index_u64, so its optional inner fingerprints
        // are not consulted by Index lookups. Only the canonical-hash map can
        // reject a missing hash independently of the outer membership tables.
        let backend_membership =
            matches!(&backend.storage, crate::mph_backend::PtrHash25Storage::Map(_));
        if engine.filter.is_some() || engine.fingerprints.is_some() || backend_membership {
            crate::MembershipMode::Probabilistic
        } else {
            crate::MembershipMode::ClosedWorld
        }
    }

    /// Use every configured membership check, exactly as [`Index::lookup`] does.
    /// A positive answer remains probabilistic unless the index is empty.
    /// Unlike [`Index::contains`], this also checks fingerprints when Bloom exists.
    pub fn may_contain(&self, key: &[u8]) -> bool {
        self.lookup(key).is_ok()
    }

    /// Batched [`Index::may_contain`], preserving input order.
    pub fn may_contain_batch(&self, keys: &[&[u8]]) -> Vec<bool> {
        keys.iter().map(|key| self.may_contain(key)).collect()
    }

    /// Verify a candidate against the caller's authoritative key storage.
    /// `verify(slot, requested_key)` must compare the full stored key at `slot`
    /// with `requested_key`; accepting a fingerprint comparison is not exact.
    /// The callback runs at most once, only after the configured checks pass.
    /// This also supports lean indexes, whose ordinary lookups assume known keys.
    pub fn lookup_verified(
        &self,
        key: &[u8],
        verify: impl FnOnce(usize, &[u8]) -> bool,
    ) -> Result<usize, IndexError> {
        let slot = self.lookup(key)?;
        if verify(slot, key) { Ok(slot) } else { Err(IndexError::KeyNotFound) }
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
        match &engine.filter {
            Some(filter) => {
                filter.contains_hash(engine.hash_version.hash(key, engine.prehash_seed))
            }
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
        let Some(filter) = &engine.filter else {
            return keys.iter().map(|key| self.lookup_mph(engine, key).is_ok()).collect();
        };
        let mut result = Vec::with_capacity(keys.len());
        let mut scratch = [0u64; 64];
        for input in keys.chunks(scratch.len()) {
            if input.len() < 8 {
                result.extend(input.iter().map(|key| self.contains(key)));
                continue;
            }
            let hashes = &mut scratch[..input.len()];
            engine.hash_version.hash_batch(input, engine.prehash_seed, hashes);
            for &hash in hashes.iter() {
                prefetch_read(filter.block_ptr(hash));
            }
            result.extend(hashes.iter().map(|&hash| filter.contains_hash(hash)));
        }
        result
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
        // Each output is produced once; avoid first initializing a potentially
        // large result allocation only to overwrite it in the pipeline.
        let mut out = Vec::with_capacity(keys.len());
        let mut results = [None; 64];
        let mut bloom_first = false;
        for input in keys.chunks(64) {
            Self::lookup_byte_chunk(engine, input, &mut results[..input.len()], &mut bloom_first);
            out.extend_from_slice(&results[..input.len()]);
        }
        out
    }

    /// Allocation-free byte-key pipeline with caller-owned output. Every result
    /// in the requested prefix is overwritten, including misses; extra output
    /// elements are left untouched.
    ///
    /// # Panics
    /// Panics if `out.len() < keys.len()`.
    pub fn lookup_batch_pipelined_into(&self, keys: &[&[u8]], out: &mut [Option<usize>]) {
        assert!(out.len() >= keys.len(), "out slice too small");
        let Some(engine) = self.engine.as_ref() else {
            out[..keys.len()].fill(None);
            return;
        };
        let mut bloom_first = false;
        for (input, result) in keys.chunks(64).zip(out[..keys.len()].chunks_mut(64)) {
            Self::lookup_byte_chunk(engine, input, result, &mut bloom_first);
        }
    }

    #[inline]
    fn lookup_byte_chunk(
        engine: &MphEngine,
        input: &[&[u8]],
        result: &mut [Option<usize>],
        bloom_first: &mut bool,
    ) {
        // A whole block of independent payload addresses is known before any
        // hash is needed. Warm those bytes while prior keys are processed.
        if input.len() == 64 {
            for key in input {
                if !key.is_empty() {
                    prefetch_read(key.as_ptr());
                }
            }
        }
        let mut hashes = [0u64; 64];
        engine.hash_version.hash_batch(input, engine.prehash_seed, &mut hashes[..input.len()]);
        Self::lookup_canonical_batch(engine, &hashes[..input.len()], result, bloom_first);
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
    /// fallback is stored as a legacy payload and remains readable. Tables are
    /// streamed from borrowed storage with bounded buffers and no file reread.
    pub fn save_mmap<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), IndexError> {
        save_atomically(path.as_ref(), |file| self.write_mmap(file))
    }

    fn write_mmap(&self, file: &mut std::fs::File) -> Result<(), IndexError> {
        use crate::mmap_index::{SectionKind, write_sections};
        use crate::mph_backend::PtrHash25Storage;
        if let Some(e) = &self.engine {
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
                    e.hash_version as u64,
                ];
                let mut sections = vec![
                    (SectionKind::PtrHash25Meta, std::mem::size_of_val(&fields) as u64),
                    (SectionKind::PtrHash25Pilots, m.pilots.len() as u64),
                    (SectionKind::PtrHash25Parts, mmap_byte_len(m.parts.len(), 40)?),
                    (SectionKind::PtrHash25Remap, mmap_byte_len(m.remap.len(), 4)?),
                ];
                if !m.fingerprints.is_empty() {
                    sections
                        .push((SectionKind::PtrHash25Fingerprints, m.fingerprints.len() as u64));
                }
                if let Some(f) = &e.filter {
                    sections.push((
                        SectionKind::BlockBloomWords,
                        mmap_byte_len(f.mmap_data().2.len(), 8)?,
                    ));
                }
                if let Some(f) = &e.fingerprints {
                    sections.push((SectionKind::OuterFingerprints, f.mmap_len() as u64));
                }
                write_sections(
                    file,
                    self.key_count as u64,
                    &sections,
                    |index, out| match sections[index].0 {
                        SectionKind::PtrHash25Meta => crate::wire::write_le(out, &fields),
                        SectionKind::PtrHash25Pilots => out.write_all(m.pilots.as_slice()),
                        SectionKind::PtrHash25Parts => {
                            for p in &m.parts {
                                let mut bytes = [0; 40];
                                bytes[..8].copy_from_slice(&p.slot_off.to_le_bytes());
                                bytes[8..16].copy_from_slice(&p.salt.to_le_bytes());
                                for (chunk, value) in
                                    bytes[16..].as_chunks_mut::<4>().0.iter_mut().zip([
                                        p.bucket_off,
                                        p.num_keys,
                                        p.num_slots,
                                        p.num_buckets,
                                        p.large_buckets,
                                        p.remap_off,
                                    ])
                                {
                                    chunk.copy_from_slice(&value.to_le_bytes());
                                }
                                out.write_all(&bytes)?;
                            }
                            Ok(())
                        }
                        SectionKind::PtrHash25Remap => crate::wire::write_le(out, &m.remap),
                        SectionKind::PtrHash25Fingerprints => out.write_all(&m.fingerprints),
                        SectionKind::BlockBloomWords => {
                            crate::wire::write_le(out, e.filter.as_ref().unwrap().mmap_data().2)
                        }
                        SectionKind::OuterFingerprints => {
                            e.fingerprints.as_ref().unwrap().write_mmap_into(out)
                        }
                        _ => unreachable!("only declared MPH sections are written"),
                    },
                )?;
                return Ok(());
            }
        }
        let sections = [(SectionKind::LegacyPayload, self.mmap_legacy_payload_len()?)];
        write_sections(file, self.key_count as u64, &sections, |_, out| {
            self.write_mmap_legacy_payload(out)
        })?;
        Ok(())
    }

    /// The fallback keeps the ordinary sealed Index payload inside one section.
    /// Its size is known from table lengths, without materializing the payload.
    fn mmap_legacy_payload_len(&self) -> std::io::Result<u64> {
        let mut length = 25u64; // Sealed header, key count and checksum.
        if let Some(engine) = &self.engine {
            let BackendDispatch::PtrHash25(backend) = &engine.backend;
            let crate::mph_backend::PtrHash25Storage::Map(map) = &backend.storage else {
                return Err(std::io::Error::other("MPH uses aligned sections"));
            };
            length = length
                .checked_add(20)
                .and_then(|v| v.checked_add(mmap_byte_len(map.len(), 12).ok()?))
                .ok_or_else(mmap_length_overflow)?;
            if let Some(filter) = &engine.filter {
                length = length
                    .checked_add(16)
                    .and_then(|v| v.checked_add(mmap_byte_len(filter.mmap_data().2.len(), 8).ok()?))
                    .ok_or_else(mmap_length_overflow)?;
            }
            if let Some(fp) = &engine.fingerprints {
                length = length
                    .checked_add(8)
                    .and_then(|v| v.checked_add(fp.memory_usage() as u64))
                    .ok_or_else(mmap_length_overflow)?;
            }
        }
        Ok(length)
    }

    fn write_mmap_legacy_payload<W: std::io::Write>(&self, writer: W) -> std::io::Result<()> {
        use std::io::Write;
        let mut out = crate::wire::ChecksumWriter::new(writer);
        out.write_all(&[TAG_V3])?;
        out.write_all(FORMAT_MAGIC)?;
        out.write_all(&FORMAT_VERSION.to_le_bytes())?;
        let hash = self.engine.as_ref().map_or(CanonicalHashVersion::Current, |e| e.hash_version);
        out.write_all(&[hash as u8, 0])?;
        out.write_all(&(self.key_count as u64).to_le_bytes())?;
        if let Some(engine) = &self.engine {
            let BackendDispatch::PtrHash25(backend) = &engine.backend;
            let crate::mph_backend::PtrHash25Storage::Map(map) = &backend.storage else {
                return Err(std::io::Error::other("MPH uses aligned sections"));
            };
            out.write_all(&engine.prehash_seed.to_le_bytes())?;
            out.write_all(&[0, 1])?; // PtrHash25 dispatch, map fallback storage.
            out.write_all(&(map.len() as u64).to_le_bytes())?;
            for (key, slot) in map {
                out.write_all(&key.to_le_bytes())?;
                out.write_all(&slot.to_le_bytes())?;
            }
            if let Some(filter) = &engine.filter {
                out.write_all(&[1])?;
                filter.write_into(&mut out)?;
            } else {
                out.write_all(&[0])?;
            }
            if let Some(fp) = &engine.fingerprints {
                out.write_all(&[fp.wire_flag()])?;
                fp.write_into(&mut out)?;
            } else {
                out.write_all(&[0])?;
            }
        }
        out.finish().map(|_| ())
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
            if meta.len() != 80 && meta.len() != 88 {
                return Err(bad());
            }
            let f: Vec<u64> =
                meta.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)).collect();
            let hash_version = if f.len() == 10 {
                CanonicalHashVersion::Legacy
            } else {
                CanonicalHashVersion::from_id(u8::try_from(f[10]).map_err(|_| bad())?)
                    .ok_or_else(bad)?
            };
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
                    hash_version,
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
        let hash_version =
            self.engine.as_ref().map_or(CanonicalHashVersion::Current, |e| e.hash_version);
        write_u8(&mut head, hash_version as u8);
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
        save_atomically(path.as_ref(), |file| self.write_to(file))
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
        let key_count = usize::try_from(cursor.read_u64().ok_or(IndexError::CorruptData)?)
            .map_err(|_| IndexError::CorruptData)?;
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
                        hash_version: CanonicalHashVersion::Legacy,
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
                    engine: Some(MphEngine {
                        backend,
                        prehash_seed,
                        hash_version: CanonicalHashVersion::Legacy,
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
        let hash_version = CanonicalHashVersion::from_id(hash_id).ok_or(IndexError::CorruptData)?;
        let _reserved = cursor.read_u8().ok_or(IndexError::CorruptData)?;
        let key_count = usize::try_from(cursor.read_u64().ok_or(IndexError::CorruptData)?)
            .map_err(|_| IndexError::CorruptData)?;
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
            engine: Some(MphEngine { backend, prehash_seed, hash_version, filter, fingerprints }),
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
        let canonical = engine.hash_version.hash(key, engine.prehash_seed);
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

/// First byte of indexes written by this version (tags 0, 2, 3 are legacy).
const TAG_V3: u8 = 4;
const FORMAT_MAGIC: &[u8; 4] = b"KIRA";
const FORMAT_VERSION: u16 = 1;
/// Number of canonical-hash seeds tried for distinct keys with colliding hashes.
/// Only an exact byte comparison may report a duplicate key.
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
/// 4. Block-Bloom (parallel by block range) and fingerprints (parallel by part,
///    written from the slots the build already computed) — skipped in lean mode.
///
/// There is no byte arena and no key copy: the only per-key allocations are the
/// canonical hash array and the partitioned hash array (8 B/key each).
fn build_engine<K>(keys: &[K], config: &IndexConfig) -> Result<MphEngine, IndexError>
where
    K: AsRef<[u8]> + Sync,
{
    // Index verifies its own fingerprint table; only the raw u64 builder needs fp8.
    let mph_cfg = MphConfig { with_fingerprints: false, ..config.mph_config.clone() };
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
        crate::ptrhash25::validate_partitioned(&part, &mph_cfg)?;
        let (backend, fingerprints) = match PtrHash25Backend::build_from_partitioned(
            &part,
            &mph_cfg,
            (!config.lean_mph).then_some(config.fingerprint_bits),
        ) {
            Ok(v) => v,
            Err(MphError::DuplicateKey) => {
                if round == 0 {
                    let mut unique = std::collections::HashSet::with_capacity(n);
                    if keys.iter().any(|key| !unique.insert(key.as_ref())) {
                        return Err(IndexError::DuplicateKey);
                    }
                }
                round += 1;
                if round >= MAX_PREHASH_ROUNDS {
                    return Err(IndexError::Unresolvable);
                }
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        let t3 = std::time::Instant::now();
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
            hash_version: CanonicalHashVersion::Current,
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
        if crate::build_pool::parallel_enabled() {
            use rayon::prelude::*;
            const CHUNK: usize = 4096;
            out.as_mut_slice()
                .par_chunks_mut(CHUNK)
                .zip(keys.par_chunks(CHUNK))
                .for_each(|(o, kc)| hash_chunk(o, kc, seed));
            return out;
        }
    }
    hash_chunk(out.as_mut_slice(), keys, seed);
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

fn mmap_byte_len(count: usize, width: u64) -> std::io::Result<u64> {
    (count as u64).checked_mul(width).ok_or_else(mmap_length_overflow)
}

fn mmap_length_overflow() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, "mmap section size overflow")
}

/// Drop a large key vector using all build threads. Freeing 10M small allocations
/// single-threaded costs ~200 ms on the Windows heap.
fn drop_keys_parallel<K: Send>(keys: Vec<K>, parallel: bool) {
    #[cfg(feature = "parallel")]
    {
        if parallel
            && crate::build_pool::parallel_enabled()
            && keys.len() >= 1 << 16
            && std::mem::needs_drop::<K>()
        {
            use rayon::prelude::*;
            crate::build_pool::pool().install(|| keys.into_par_iter().for_each(drop));
            return;
        }
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;
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

/// Publish only a complete snapshot. A failed write leaves the prior file intact.
fn save_atomically<E: From<std::io::Error>>(
    path: &std::path::Path,
    write: impl FnOnce(&mut std::fs::File) -> Result<(), E>,
) -> Result<(), E> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(".".as_ref());
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    if let Ok(metadata) = std::fs::metadata(path) {
        temporary.as_file().set_permissions(metadata.permissions())?;
    }
    write(temporary.as_file_mut())?;
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(())
}

#[cfg(test)]
mod atomic_save_tests {
    #[test]
    fn failed_replacement_preserves_the_previous_snapshot() {
        use std::io::{self, Write};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index");
        std::fs::write(&path, b"previous").unwrap();
        let result = super::save_atomically(&path, |file| {
            file.write_all(b"incomplete")?;
            Err::<(), _>(io::Error::other("write failed"))
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"previous");
    }
}
