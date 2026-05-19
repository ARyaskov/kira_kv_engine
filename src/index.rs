use crate::block_bloom::BlockBloom;
use crate::mph_backend::{
    BackendDispatch, BackendKind, BuildConfig as BackendConfig, BuildProfile, build_dispatch,
    prehash_u64_arena,
};
use crate::pgm::PgmError;
use crate::ptrhash25::{BuildConfig as MphConfig, PtrHash25Error as MphError};
use thiserror::Error;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaeseq_u8, vdupq_n_u64, vgetq_lane_u64, vld1q_u8, vreinterpretq_u8_u64, vreinterpretq_u64_u8,
};
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

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
    fingerprints: Option<Box<[u16]>>,
}

/// Static MPH-backed key→id index. For semantic range queries on u64 keys
/// use `PgmBuilder` directly; for byte-key range queries use `HybridBuilder`.
pub struct Index {
    engine: MphEngine,
    key_count: usize,
}

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("MPH error: {0}")]
    Mph(String),
    #[error("PGM error: {0}")]
    Pgm(String),
    #[error("key not found")]
    KeyNotFound,
    #[error("invalid key format")]
    InvalidKey,
    #[error("corrupt data")]
    CorruptData,
}

impl From<MphError> for IndexError {
    fn from(err: MphError) -> Self {
        IndexError::Mph(err.to_string())
    }
}

impl From<PgmError> for IndexError {
    fn from(err: PgmError) -> Self {
        IndexError::Pgm(err.to_string())
    }
}

/// Configuration for the index.
///
/// As of v0.5 there's only one MPH backend (PtrHash25); the `backend` field is kept
/// as a 1-variant enum for future extensibility. `hot_fraction` controls the PGM
/// hot-tier cache for numeric workloads.
///
/// PGM-specific options (`pgm_*`) apply only when `auto_detect_numeric = true` and
/// all input keys are 8-byte little-endian u64.
#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub mph_config: MphConfig,
    pub pgm_epsilon: u32,
    pub auto_detect_numeric: bool,
    pub backend: BackendKind,
    pub hot_fraction: f32,
    pub enable_parallel_build: bool,
    pub build_fast_profile: bool,
    /// Enable Block-Bloom for fast PGM negative-lookup path. Costs
    /// ~10–12 bits/key, rejects ~99% of misses in O(1) without touching segments.
    pub pgm_enable_bloom: bool,
    /// Compact PGM keys via Elias-Fano. Saves ~30–50% on key memory at
    /// large N, costs ~50 ns/lookup for materialization. Default off.
    pub pgm_enable_elias_fano: bool,
    /// Auto-tune ε for a target per-lookup latency in nanoseconds. When
    /// set, overrides `pgm_epsilon`. Default: None (use `pgm_epsilon` as-is).
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
/// * `mph_salt`, `num_buckets`, `num_slots`, `prerotate`, `pilots`:
///   PtrHash25 constants. See `ptrhash25::PtrHash25Mphf::index_u64`
///   for the lookup formula.
/// * `bloom`: optional Bloom filter (present for non-lean indexes).
///   Reject early if `!bloom.contains(canonical)`.
/// * `fingerprints`: optional 16-bit fingerprint table for negative-query
///   rejection at the slot level. Check
///   `fingerprints[slot] == (canonical & 0xFFFF) as u16`.
#[derive(Debug, Clone)]
pub struct GpuExport {
    pub prehash_seed: u64,
    pub mph_salt: u64,
    pub num_buckets: u32,
    pub num_slots: u64,
    pub prerotate: u8,
    pub pilots: Vec<u8>,
    pub bloom: Option<BloomExport>,
    pub fingerprints: Option<Vec<u16>>,
}

/// Bloom filter snapshot for GPU export. The Bloom uses
/// split-block layout with `BLOCK_WORDS = 8` (`64 B` per block).
/// Block index is `(((canonical >> 32) * blocks) >> 32)`.
#[derive(Debug, Clone)]
pub struct BloomExport {
    /// Number of 64-byte blocks. Always a power of two.
    pub blocks: usize,
    /// Concatenated 64-bit words. Length is `blocks * 8`.
    pub words: Vec<u64>,
}

impl Index {
    pub fn build_index<K>(keys: Vec<K>, config: IndexConfig) -> Result<Self, IndexError>
    where
        K: AsRef<[u8]>,
    {
        if keys.is_empty() {
            return Err(IndexError::InvalidKey);
        }

        // In lean MPH mode we also disable the inner PtrHash25 fingerprints —
        // they would otherwise add another 8 bits/key that the engine isn't
        // checking against anyway.
        let mut config = config;
        if config.lean_mph {
            config.mph_config.with_fingerprints = false;
        }

        let arena = build_key_arena(keys, config.mph_config.seed)?;
        let key_count = arena.len();

        let (prehash_seed, _canonical, backend, fingerprints, filter) =
            run_build_pipeline_with_pool(&arena, &config)?;

        Ok(Index {
            engine: MphEngine {
                backend,
                prehash_seed,
                filter,
                fingerprints,
            },
            key_count,
        })
    }

    #[inline(always)]
    fn simd_touch(key: &[u8]) {
        // On aarch64 we used to issue a NEON load as a soft TLB/cache-line touch.
        // On x86_64 the equivalent is _mm_prefetch already called in prefetch_key_batch.
        // For short keys (≤16 B) both arches handle it via a regular load anyway.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if key.len() >= 16 {
                let _ = vld1q_u8(key.as_ptr());
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = key;
    }

    pub fn lookup(&self, key: &[u8]) -> Result<usize, IndexError> {
        self.lookup_mph(&self.engine, key)
    }

    pub fn get(&self, key: &[u8]) -> Result<usize, IndexError> {
        self.lookup(key)
    }

    pub fn lookup_str(&self, key: &str) -> Result<usize, IndexError> {
        self.lookup(key.as_bytes())
    }

    pub fn get_str(&self, key: &str) -> Result<usize, IndexError> {
        self.lookup_str(key)
    }

    pub fn lookup_u64(&self, key: u64) -> Result<usize, IndexError> {
        self.lookup_mph(&self.engine, &key.to_le_bytes())
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
        let engine = &self.engine;

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
        #[cfg(target_arch = "x86_64")]
        if let Some(bf) = &engine.filter {
            for i in 0..WINDOW.min(n) {
                unsafe {
                    let p = bf.block_ptr(canon[i]);
                    _mm_prefetch(p as *const i8, _MM_HINT_T0);
                }
            }
        }

        // Main loop: process 8 keys per iteration when possible. For each chunk:
        //  - Bloom check (scalar — already cheap with prefetch in flight)
        //  - backend.lookup for each (gathers pilot)
        //  - AVX2 gather_fp_check_x8 for the fingerprint stage (1 instruction vs 8 scalar loads)
        let mut i = 0usize;
        let fp_base = engine
            .fingerprints
            .as_ref()
            .map(|fp| fp.as_ptr())
            .unwrap_or(std::ptr::null());

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        while i + 8 <= n {
            // Lookahead Bloom prefetch.
            #[cfg(target_arch = "x86_64")]
            if let Some(bf) = &engine.filter {
                for k in 0..8 {
                    if i + WINDOW + k < n {
                        unsafe {
                            let p = bf.block_ptr(canon[i + WINDOW + k]);
                            _mm_prefetch(p as *const i8, _MM_HINT_T0);
                        }
                    }
                }
            }

            // Bloom + backend.lookup for 8 keys, collecting (idx, expected_fp) pairs.
            let mut indices = [0u32; 8];
            let mut expected = [0u16; 8];
            let mut alive = [false; 8];
            for k in 0..8 {
                let hash = canon[i + k];
                if let Some(bf) = &engine.filter {
                    if !bf.contains_hash(hash) {
                        continue;
                    }
                }
                if let Some(idx) = engine.backend.lookup(hash) {
                    indices[k] = idx;
                    expected[k] = fingerprint16_mph(hash);
                    alive[k] = true;
                }
            }
            if !fp_base.is_null() {
                // Single gather for fingerprint validation.
                let bitmask = unsafe { Self::gather_fp_check_x8(fp_base, indices, expected) };
                for k in 0..8 {
                    if alive[k] && (bitmask & (1 << k)) != 0 {
                        out[i + k] = Some(indices[k] as usize);
                    }
                }
            } else {
                // Lean mode: trust the MPH result.
                for k in 0..8 {
                    if alive[k] {
                        out[i + k] = Some(indices[k] as usize);
                    }
                }
            }
            i += 8;
        }

        // Tail: scalar.
        while i < n {
            let hash = canon[i];
            #[cfg(target_arch = "x86_64")]
            if let Some(bf) = &engine.filter {
                if i + WINDOW < n {
                    unsafe {
                        let p = bf.block_ptr(canon[i + WINDOW]);
                        _mm_prefetch(p as *const i8, _MM_HINT_T0);
                    }
                }
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

    /// Fast-path lookup for u64 keys on PtrHash25-backed Mph engines. Skips the
    /// canonical bytes path (no `to_le_bytes` allocation, no wyhash dispatch) and
    /// the BackendDispatch enum match — straight to PtrHash25::index_u64. Saves
    /// ~5-10 ns per lookup vs `lookup_u64` for hot paths.
    ///
    /// Returns `None` for non-PtrHash25 engines or non-Mph engines; callers
    /// detecting `None` should fall back to `lookup_u64`.
    #[inline]
    pub fn lookup_u64_fast(&self, key: u64) -> Option<Result<usize, IndexError>> {
        let engine = &self.engine;
        // Probe BackendDispatch for the PtrHash25 variant; if not, return None
        // so caller falls back. The match itself is cheap and predictable.
        use crate::mph_backend::BackendDispatch;
        if let BackendDispatch::PtrHash25(_) = &engine.backend {
            // Hash key with same canonical path the index was built against.
            let canonical = canonical_hash_key(&key.to_le_bytes(), engine.prehash_seed);
            if let Some(bf) = &engine.filter {
                if !bf.contains_hash(canonical) {
                    return Some(Err(IndexError::KeyNotFound));
                }
            }
            let idx = match engine.backend.lookup(canonical) {
                Some(i) => i as usize,
                None => return Some(Err(IndexError::KeyNotFound)),
            };
            if let Some(fps) = &engine.fingerprints {
                let fp = fingerprint16_mph(canonical);
                let ok = unsafe { *fps.get_unchecked(idx) == fp };
                Some(if ok {
                    Ok(idx)
                } else {
                    Err(IndexError::KeyNotFound)
                })
            } else {
                Some(Ok(idx))
            }
        } else {
            None
        }
    }

    pub fn get_u64(&self, key: u64) -> Result<usize, IndexError> {
        self.lookup_u64(key)
    }

    /// Range queries are not supported by the PtrHash25-only `Index`. Use
    /// `PgmIndex` directly for semantic u64 range queries.
    pub fn range(&self, _min_key: u64, _max_key: u64) -> Vec<usize> {
        Vec::new()
    }

    pub fn get_all(&self, min_key: u64, max_key: u64) -> Vec<usize> {
        self.range(min_key, max_key)
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        let engine = &self.engine;
        {
            let canonical = canonical_hash_key(key, engine.prehash_seed);
            match &engine.filter {
                Some(bf) => bf.contains_hash(canonical),
                // Lean mode has no Bloom — fall back to full lookup (slower but
                // correctness preserved). For high-throughput contains() calls,
                // disable lean_mph.
                None => self.lookup_mph(engine, key).is_ok(),
            }
        }
    }

    pub fn has(&self, key: &[u8]) -> bool {
        self.contains(key)
    }

    pub fn exists(&self, key: &[u8]) -> bool {
        self.contains(key)
    }

    pub fn contains_batch(&self, keys: &[&[u8]]) -> Vec<bool> {
        let engine = &self.engine;
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

    pub fn has_batch(&self, keys: &[&[u8]]) -> Vec<bool> {
        self.contains_batch(keys)
    }

    pub fn exists_batch(&self, keys: &[&[u8]]) -> Vec<bool> {
        self.contains_batch(keys)
    }

    pub fn lookup_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>> {
        let mut out = Vec::with_capacity(keys.len());
        let engine = &self.engine;
        let mut i = 0usize;
        #[cfg(target_arch = "x86_64")]
        while i + 16 <= keys.len() {
            prefetch_key_batch(keys, i, 16);
            for j in 0..16 {
                out.push(self.lookup_mph(engine, keys[i + j]).ok());
            }
            i += 16;
        }
        #[cfg(target_arch = "x86_64")]
        while i + 8 <= keys.len() {
            prefetch_key_batch(keys, i, 8);
            for j in 0..8 {
                out.push(self.lookup_mph(engine, keys[i + j]).ok());
            }
            i += 8;
        }
        #[cfg(target_arch = "aarch64")]
        while i + 8 <= keys.len() {
            for j in 0..8 {
                let key = keys[i + j];
                Self::simd_touch(key);
            }
            for j in 0..8 {
                let key = keys[i + j];
                out.push(self.lookup_mph(engine, key).ok());
            }
            i += 8;
        }
        while i + 4 <= keys.len() {
            for j in 0..4 {
                let key = keys[i + j];
                Self::simd_touch(key);
                out.push(self.lookup_mph(engine, key).ok());
            }
            i += 4;
        }
        while i < keys.len() {
            let key = keys[i];
            Self::simd_touch(key);
            out.push(self.lookup_mph(engine, key).ok());
            i += 1;
        }
        out
    }

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
        let mut out = Vec::with_capacity(keys.len());
        let engine = &self.engine;
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
                    #[cfg(target_arch = "x86_64")]
                    if let Some(bf) = filter {
                        unsafe {
                            let bloom_ptr = bf.block_ptr(h);
                            _mm_prefetch(bloom_ptr as *const i8, _MM_HINT_T0);
                        }
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
                        #[cfg(target_arch = "x86_64")]
                        if let Some(bf) = filter {
                            unsafe {
                                let bloom_ptr = bf.block_ptr(h);
                                _mm_prefetch(bloom_ptr as *const i8, _MM_HINT_T0);
                            }
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
                    #[cfg(target_arch = "x86_64")]
                    if let (Some(fps), Some(idx)) = (fingerprints, idx_opt) {
                        unsafe {
                            let fp_ptr = fps.as_ptr().add(idx as usize);
                            _mm_prefetch(fp_ptr as *const i8, _MM_HINT_T0);
                        }
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
        self.engine.filter.as_ref().map(|bf| bf.contains_hash(canonical))
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
        let engine = &self.engine;

        let BackendDispatch::PtrHash25(ph_backend) = &engine.backend;
        // Hot-tier `Map` variant isn't a true MPH; skip GPU export.
        let PtrHash25Storage::Mph(mph) = &ph_backend.storage else {
            return None;
        };
        if mph.use_aes_hash {
            return None;
        }

        // Materialise pilots to a flat byte array regardless of in-memory
        // representation. CompressedV2 (the typical on-disk form for huge
        // indexes) gets unpacked into a contiguous Vec<u8> — the kernel
        // wants a direct array lookup.
        let pilots_flat: Vec<u8> = (0..mph.pilots.len()).map(|b| mph.pilots.get(b)).collect();

        let bloom_export = engine.filter.as_ref().map(|bf| {
            let words = bf.export_words();
            BloomExport {
                blocks: words.len() / 8, // BLOCK_WORDS=8
                words,
            }
        });

        let fingerprints = engine
            .fingerprints
            .as_ref()
            .map(|fps| fps.iter().copied().collect::<Vec<u16>>());

        Some(GpuExport {
            prehash_seed: engine.prehash_seed,
            mph_salt: mph.salt,
            num_buckets: mph.num_buckets,
            num_slots: mph.n,
            prerotate: mph.prerotate,
            pilots: pilots_flat,
            bloom: bloom_export,
            fingerprints,
        })
    }

    pub fn stats(&self) -> IndexStats {
        let engine = &self.engine;
        let mph_memory = engine.backend.memory_usage_bytes();
        let filter_memory = engine.filter.as_ref().map(|b| b.memory_usage()).unwrap_or(0);
        let fp_memory = engine
            .fingerprints
            .as_ref()
            .map(|fp| fp.len() * std::mem::size_of::<u16>())
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
        let mut w = MmapIndexWriter::create(path, self.key_count as u64)
            .map_err(|_| IndexError::CorruptData)?;
        // We always write the legacy payload (so open_mmap continues to work).
        // For PtrHashV2-backed Mph engines we ALSO add per-field sections, enabling
        // zero-copy reads via Index::open_mmap_zero_copy in future versions.
        let bytes = self.to_bytes()?;
        w.add_section(SectionKind::LegacyPayload, bytes);
        w.finalize().map_err(|_| IndexError::CorruptData)
    }

    /// Open a previously `save_mmap`'d index. Currently does a one-time read from the
    /// LegacyPayload section into a `Vec<u8>`. A future zero-copy variant will keep
    /// the mmap alive and return views into it without copying.
    pub fn open_mmap<P: AsRef<std::path::Path>>(path: P) -> Result<Self, IndexError> {
        use crate::mmap_index::{MmapIndex, SectionKind};
        let mmap = MmapIndex::open(path).map_err(|_| IndexError::CorruptData)?;
        let header = mmap.parse_header().map_err(|_| IndexError::CorruptData)?;
        let bytes = mmap
            .section(&header, SectionKind::LegacyPayload)
            .or_else(|| mmap.section(&header, SectionKind::PtrHash25Pilots))
            .ok_or(IndexError::CorruptData)?;
        Self::from_bytes(bytes)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, IndexError> {
        let mut out = Vec::new();
        let engine = &self.engine;
        // Tag 2 = MPH v2 with optional filter/fingerprints (presence flags).
        // Tag 0 still readable for legacy v1 indexes (always-present filter/fps).
        write_u8(&mut out, 2);
        write_u64(&mut out, self.key_count as u64);
        write_u64(&mut out, engine.prehash_seed);
        engine.backend.write_to(&mut out);
        match &engine.filter {
            Some(bf) => {
                write_u8(&mut out, 1);
                bf.write_to(&mut out);
            }
            None => write_u8(&mut out, 0),
        }
        match &engine.fingerprints {
            Some(fp) => {
                write_u8(&mut out, 1);
                write_fingerprints(&mut out, fp.as_ref());
            }
            None => write_u8(&mut out, 0),
        }
        Ok(out)
    }

    pub fn serialize(&self) -> Result<Vec<u8>, IndexError> {
        self.to_bytes()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IndexError> {
        let mut cursor = Cursor::new(bytes);
        let tag = cursor.read_u8().ok_or(IndexError::CorruptData)?;
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
                    engine: MphEngine {
                        backend,
                        prehash_seed,
                        filter: Some(filter),
                        fingerprints: Some(fingerprints),
                    },
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
                    engine: MphEngine {
                        backend,
                        prehash_seed,
                        filter,
                        fingerprints,
                    },
                    key_count,
                })
            }
            // Tag 1 (legacy PGM engine) was removed in v0.6 — use PgmIndex
            // directly for range queries on u64 keys.
            _ => Err(IndexError::CorruptData),
        }
    }

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
fn make_backend_cfg(config: &IndexConfig) -> BackendConfig {
    BackendConfig {
        backend: config.backend,
        enable_parallel_build: config.enable_parallel_build,
        seed: config.mph_config.seed,
        gamma: config.mph_config.gamma,
        rehash_limit: config.mph_config.max_rehash,
        build_profile: if config.build_fast_profile {
            BuildProfile::Fast
        } else {
            BuildProfile::Balanced
        },
    }
}

fn run_build_pipeline(
    arena: &KeyArena,
    config: &IndexConfig,
) -> Result<
    (
        u64,
        Vec<u64>,
        BackendDispatch,
        Option<Box<[u16]>>,
        Option<BlockBloom>,
    ),
    IndexError,
> {
    let (prehash_seed, canonical) = prehash_u64_arena(
        arena.bytes.as_slice(),
        arena.offsets.as_slice(),
        config.mph_config.seed,
        !config.build_fast_profile,
    )
    .ok_or(IndexError::CorruptData)?;

    let filter = if config.lean_mph {
        None
    } else {
        Some(BlockBloom::build_from_prehashed(&canonical))
    };
    let backend_cfg = make_backend_cfg(config);
    let backend = build_dispatch(&canonical, &backend_cfg);
    let fingerprints = if config.lean_mph {
        None
    } else {
        Some(build_fingerprints_hashed(&backend, &canonical).into_boxed_slice())
    };
    Ok((prehash_seed, canonical, backend, fingerprints, filter))
}

#[cfg(feature = "parallel")]
fn run_build_pipeline_with_pool(
    arena: &KeyArena,
    config: &IndexConfig,
) -> Result<
    (
        u64,
        Vec<u64>,
        BackendDispatch,
        Option<Box<[u16]>>,
        Option<BlockBloom>,
    ),
    IndexError,
> {
    if !config.enable_parallel_build {
        return run_build_pipeline(arena, config);
    }
    // Per-call pinned pool. Tried OnceLock'd global pool to skip ~150 ms / build of
    // thread spawn, but a single global pool pinned to core IDs interfered with the
    // Use the global persistent pool (initialized once, reused across builds).
    // Avoids the ~150 ms Windows CreateThread cost per build that the previous
    // per-call pool incurred.
    crate::build_pool::pool().install(|| run_build_pipeline(arena, config))
}

#[cfg(not(feature = "parallel"))]
fn run_build_pipeline_with_pool(
    arena: &KeyArena,
    config: &IndexConfig,
) -> Result<
    (
        u64,
        Vec<u64>,
        BackendDispatch,
        Option<Box<[u16]>>,
        Option<BlockBloom>,
    ),
    IndexError,
> {
    run_build_pipeline(arena, config)
}

// Thread-pool & core-pinning logic now lives in `build_pool::pool()`.

#[inline(always)]
fn canonical_hash_key(key: &[u8], seed: u64) -> u64 {
    crate::canonical_hash::canonical_hash_bytes(key, seed)
}

#[cfg(target_arch = "x86_64")]
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
                // SAFETY: prefetch is a hint; pointer is derived from valid slice with len > 0.
                unsafe { _mm_prefetch(slice.as_ptr() as *const i8, _MM_HINT_T0) };
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

    pub fn with_pgm_epsilon(mut self, epsilon: u32) -> Self {
        self.config.pgm_epsilon = epsilon;
        self
    }

    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.config.backend = backend;
        self
    }

    pub fn with_hot_fraction(mut self, hot_fraction: f32) -> Self {
        self.config.hot_fraction = hot_fraction;
        self
    }

    pub fn with_parallel_build(mut self, enabled: bool) -> Self {
        self.config.enable_parallel_build = enabled;
        self
    }

    pub fn with_build_fast_profile(mut self, enabled: bool) -> Self {
        self.config.build_fast_profile = enabled;
        self
    }

    pub fn auto_detect_numeric(mut self, enabled: bool) -> Self {
        self.config.auto_detect_numeric = enabled;
        self
    }

    /// Enable the PGM Block-Bloom fast-negative-lookup filter. Only
    /// effective when `auto_detect_numeric=true` and all keys are 8-byte u64.
    pub fn with_pgm_bloom(mut self, enabled: bool) -> Self {
        self.config.pgm_enable_bloom = enabled;
        self
    }

    /// Compact PGM key storage via Elias-Fano. 30–50% memory win for
    /// large indexes; ~50 ns/lookup overhead.
    pub fn with_pgm_elias_fano(mut self, enabled: bool) -> Self {
        self.config.pgm_enable_elias_fano = enabled;
        self
    }

    /// Auto-tune PGM ε for a target per-lookup latency. Overrides
    /// `with_pgm_epsilon` if set.
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
        K: AsRef<[u8]>,
    {
        Index::build_index(keys, self.config)
    }
}

impl Default for IndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

struct KeyArena {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
}

impl KeyArena {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

fn build_key_arena<K>(keys: Vec<K>, seed: u64) -> Result<KeyArena, IndexError>
where
    K: AsRef<[u8]>,
{
    let total_bytes = keys.iter().map(|k| k.as_ref().len()).sum();
    let mut bytes = Vec::with_capacity(total_bytes);
    let mut offsets = Vec::with_capacity(keys.len() + 1);
    offsets.push(0u32);
    // Fast-path detector: if every key is exactly 8 bytes, we can dedupe by sorting
    // the u64 values directly — no hashing required.
    let mut all_u64 = true;
    let mut u64_values: Vec<u64> = Vec::with_capacity(keys.len());
    let mut hashes_with_idx = Vec::with_capacity(keys.len());

    for (i, key) in keys.into_iter().enumerate() {
        let k = key.as_ref();
        if all_u64 && k.len() == 8 {
            let v = unsafe { std::ptr::read_unaligned(k.as_ptr() as *const u64) };
            u64_values.push(u64::from_le(v));
        } else if all_u64 {
            // First non-u64 key: switch to hash-based dedup, backfill what we already have.
            all_u64 = false;
            hashes_with_idx.reserve(u64_values.len() + 1);
            for (j, &v) in u64_values.iter().enumerate() {
                hashes_with_idx.push((crate::build_hasher::fast_hash_bytes(&v.to_le_bytes()), j as u32));
            }
            u64_values.clear();
        }
        if !all_u64 {
            let h = crate::build_hasher::fast_hash_bytes(k);
            hashes_with_idx.push((h, i as u32));
        }
        bytes.extend_from_slice(k);
        offsets.push(bytes.len() as u32);
    }

    if all_u64 && !u64_values.is_empty() {
        // O(N log N) sort of plain u64 — no hashing collisions to worry about, no byte compare.
        let mut sorted = u64_values.clone();
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            sorted.par_sort_unstable();
        }
        #[cfg(not(feature = "parallel"))]
        {
            sorted.sort_unstable();
        }
        for w in sorted.windows(2) {
            if w[0] == w[1] {
                return Err(IndexError::Mph("DuplicateKey".to_string()));
            }
        }
    } else {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            hashes_with_idx.par_sort_unstable_by_key(|&(h, _)| h);
        }
        #[cfg(not(feature = "parallel"))]
        {
            hashes_with_idx.sort_unstable_by_key(|&(h, _)| h);
        }

        for window in hashes_with_idx.windows(2) {
            if window[0].0 == window[1].0 {
                let idx1 = window[0].1 as usize;
                let idx2 = window[1].1 as usize;
                let k1 = &bytes[(offsets[idx1] as usize)..(offsets[idx1 + 1] as usize)];
                let k2 = &bytes[(offsets[idx2] as usize)..(offsets[idx2 + 1] as usize)];
                if k1 == k2 {
                    return Err(IndexError::Mph("DuplicateKey".to_string()));
                }
            }
        }
    }

    if offsets.len() <= 2 {
        return Ok(KeyArena { bytes, offsets });
    }

    let mut order: Vec<usize> = (0..offsets.len() - 1).collect();
    permute_order_for_builder(&mut order, &bytes, &offsets, seed);
    compact_arena_by_order(&bytes, &offsets, &order)
}

fn permute_order_for_builder(order: &mut [usize], bytes: &[u8], offsets: &[u32], seed: u64) {
    if order.len() <= 1 {
        return;
    }
    let mut s = seed ^ (order.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let sample = order.len().min(8);
    for i in 0..sample {
        let idx = order[i];
        let start = offsets[idx] as usize;
        let end = offsets[idx + 1] as usize;
        s ^= crate::build_hasher::fast_hash_bytes(&bytes[start..end]);
        s = xorshift64(s);
    }
    for i in (1..order.len()).rev() {
        s = xorshift64(s);
        let j = (s % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

fn compact_arena_by_order(
    bytes: &[u8],
    offsets: &[u32],
    order: &[usize],
) -> Result<KeyArena, IndexError> {
    let mut out_offsets = Vec::with_capacity(order.len() + 1);
    out_offsets.push(0u32);
    let mut out_bytes = Vec::with_capacity(bytes.len());
    for &idx in order {
        let start = offsets[idx] as usize;
        let end = offsets[idx + 1] as usize;
        out_bytes.extend_from_slice(&bytes[start..end]);
        if out_bytes.len() > u32::MAX as usize {
            return Err(IndexError::CorruptData);
        }
        out_offsets.push(out_bytes.len() as u32);
    }
    Ok(KeyArena {
        bytes: out_bytes,
        offsets: out_offsets,
    })
}

#[inline]
fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn build_fingerprints_hashed(backend: &BackendDispatch, keys: &[u64]) -> Vec<u16> {
    // First pass: find the maximum slot the backend returns. Most backends map to
    // exactly [0, N) but PtrHashV2's near-minimal variant maps to [0, ~1.1*N).
    let mut max_idx = keys.len();
    for &k in keys {
        let idx = backend.lookup(k).expect("backend must map training keys") as usize;
        if idx >= max_idx {
            max_idx = idx + 1;
        }
    }
    let mut fps = vec![0u16; max_idx];
    for &k in keys {
        let idx = backend.lookup(k).expect("backend must map training keys") as usize;
        let fp = fingerprint16_mph(k);
        fps[idx] = fp;
    }
    fps
}

#[inline]
fn fingerprint16_mph(canonical: u64) -> u16 {
    (canonical & 0xFFFF) as u16
}


#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn hash_u64_crc(key: u64, seed: u64) -> u64 {
    use std::arch::aarch64::__crc32d;
    let mut crc = seed as u32;
    crc = __crc32d(crc, key);
    let mixed = ((crc as u64) << 32) ^ (seed.rotate_left(17) ^ key);
    splitmix64(mixed)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn hash_bytes_crc(key: &[u8], seed: u64) -> u64 {
    use std::arch::aarch64::{__crc32b, __crc32d};
    let mut crc = seed as u32;
    let mut i = 0usize;
    while i + 8 <= key.len() {
        let chunk = u64::from_le_bytes(key[i..i + 8].try_into().unwrap());
        crc = __crc32d(crc, chunk);
        i += 8;
    }
    while i < key.len() {
        crc = __crc32b(crc, key[i]);
        i += 1;
    }
    let mixed = ((crc as u64) << 32) ^ seed.wrapping_add(key.len() as u64);
    splitmix64(mixed)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn aes_mix_u64(hash: u64, seed: u64) -> u64 {
    let block = vdupq_n_u64(hash ^ seed);
    let key = vdupq_n_u64(seed.rotate_left(23) ^ 0xA5A5_A5A5_A5A5_A5A5);
    let mixed = vaeseq_u8(vreinterpretq_u8_u64(block), vreinterpretq_u8_u64(key));
    let out = vreinterpretq_u64_u8(mixed);
    vgetq_lane_u64(out, 0) ^ vgetq_lane_u64(out, 1)
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

fn write_fingerprints(out: &mut Vec<u8>, fps: &[u16]) {
    write_u64(out, fps.len() as u64);
    for &fp in fps {
        write_u16(out, fp);
    }
}

fn read_fingerprints(cursor: &mut Cursor<'_>) -> Result<Box<[u16]>, IndexError> {
    let len = cursor.read_u64().ok_or(IndexError::CorruptData)? as usize;
    let mut fps = Vec::with_capacity(len);
    for _ in 0..len {
        fps.push(cursor.read_u16().ok_or(IndexError::CorruptData)?);
    }
    Ok(fps.into_boxed_slice())
}
