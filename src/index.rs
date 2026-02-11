use crate::hot_tier::HotTierIndex;
use crate::mph_backend::{
    BackendDispatch, BackendKind, BuildConfig as BackendConfig, BuildProfile, build_dispatch,
    prehash_u64_arena,
};
use crate::pgm::{PgmBuilder, PgmError, PgmIndex};
use crate::ptrhash::{BuildConfig as MphConfig, Builder as MphBuilder, MphError, Mphf};
use crate::remap::{remap_id_from_index, remap_ids_for_pgm};
use crate::xor_filter::{Cursor as XorCursor, Xor8};
use hashbrown::HashMap;
#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;
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
    xor: Xor8,
    fingerprints: Box<[u16]>,
}

#[derive(Debug)]
struct PgmEngine {
    pgm: PgmIndex,
    xor: Xor8,
    mph: Mphf,
    fingerprints: Box<[u16]>,
    hot: Option<HotTierIndex>,
}

#[derive(Debug)]
enum Engine {
    Pgm(PgmEngine),
    Mph(MphEngine),
}

/// Index: auto-selects PGM for purely numeric keys, MPH otherwise
pub struct Index {
    engine: Engine,
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

/// Configuration for index
#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub mph_config: MphConfig,
    pub pgm_epsilon: u32,
    pub auto_detect_numeric: bool,
    pub backend: BackendKind,
    pub hot_fraction: f32,
    pub hot_backend: BackendKind,
    pub cold_backend: BackendKind,
    pub enable_parallel_build: bool,
    pub build_fast_profile: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        let mut cfg = crate::cpu::detect_features().optimal_index_config();
        cfg.auto_detect_numeric = false;
        cfg.backend = BackendKind::PtrHash2025;
        cfg.hot_fraction = 0.15;
        cfg.hot_backend = BackendKind::CHD;
        cfg.cold_backend = BackendKind::RecSplit;
        cfg.enable_parallel_build = true;
        cfg.build_fast_profile = true;
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

impl Index {
    pub fn build_index<K>(keys: Vec<K>, config: IndexConfig) -> Result<Self, IndexError>
    where
        K: AsRef<[u8]>,
    {
        if keys.is_empty() {
            return Err(IndexError::InvalidKey);
        }

        let arena = build_key_arena(keys, config.mph_config.salt)?;
        let key_count = arena.len();

        let mut numeric_keys = Vec::with_capacity(key_count);
        let mut all_numeric = false;
        if config.auto_detect_numeric {
            all_numeric = true;
            for key in arena.keys() {
                if let Some(num) = try_parse_u64(key) {
                    numeric_keys.push(num);
                } else {
                    all_numeric = false;
                    numeric_keys.clear();
                    break;
                }
            }
        }

        if config.auto_detect_numeric && all_numeric {
            let pgm = PgmBuilder::new()
                .with_epsilon(config.pgm_epsilon)
                .build(numeric_keys.clone())?;
            let xor = build_xor_u64(&numeric_keys)?;
            let remap_ids = remap_ids_for_pgm(&pgm);
            let remap_bytes: Vec<Vec<u8>> =
                remap_ids.iter().map(|k| k.to_le_bytes().to_vec()).collect();
            let mph_config = config.mph_config.clone();
            let mph = MphBuilder::new()
                .with_config(mph_config.clone())
                .build_unique_ref(&remap_bytes)?;
            let fingerprints = build_fingerprints_u64(&mph, &remap_ids);
            let hot = build_hot_tier(&pgm, &mph_config);

            Ok(Index {
                engine: Engine::Pgm(PgmEngine {
                    pgm,
                    xor,
                    mph,
                    fingerprints: fingerprints.into_boxed_slice(),
                    hot,
                }),
                key_count,
            })
        } else {
            let (prehash_seed, _canonical, backend, fingerprints, xor) =
                run_build_pipeline_with_pool(&arena, &config)?;

            Ok(Index {
                engine: Engine::Mph(MphEngine {
                    backend,
                    prehash_seed,
                    xor,
                    fingerprints,
                }),
                key_count,
            })
        }
    }

    #[inline(always)]
    fn simd_touch(key: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        let _ = key;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if key.len() >= 16 {
                let _ = vld1q_u8(key.as_ptr());
            }
        }
    }

    pub fn lookup(&self, key: &[u8]) -> Result<usize, IndexError> {
        match &self.engine {
            Engine::Pgm(engine) => {
                let num = try_parse_u64(key).ok_or(IndexError::InvalidKey)?;
                self.lookup_pgm(engine, num)
            }
            Engine::Mph(engine) => self.lookup_mph(engine, key),
        }
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
        match &self.engine {
            Engine::Pgm(engine) => self.lookup_pgm(engine, key),
            Engine::Mph(engine) => self.lookup_mph(engine, &key.to_le_bytes()),
        }
    }

    pub fn get_u64(&self, key: u64) -> Result<usize, IndexError> {
        self.lookup_u64(key)
    }

    pub fn range(&self, min_key: u64, max_key: u64) -> Vec<usize> {
        match &self.engine {
            Engine::Pgm(engine) => engine.pgm.range(min_key, max_key),
            Engine::Mph(_) => Vec::new(),
        }
    }

    pub fn get_all(&self, min_key: u64, max_key: u64) -> Vec<usize> {
        self.range(min_key, max_key)
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        match &self.engine {
            Engine::Pgm(engine) => {
                if let Some(num) = try_parse_u64(key) {
                    engine.pgm.range_guard(num) && engine.xor.contains_u64(num)
                } else {
                    false
                }
            }
            Engine::Mph(engine) => {
                let canonical = canonical_hash_key(key, engine.prehash_seed);
                engine.xor.contains_u64(canonical)
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
        match &self.engine {
            Engine::Pgm(engine) => keys
                .iter()
                .map(|&key| {
                    if let Some(num) = try_parse_u64(key) {
                        engine.pgm.range_guard(num) && engine.xor.contains_u64(num)
                    } else {
                        false
                    }
                })
                .collect(),
            Engine::Mph(engine) => keys
                .iter()
                .map(|&key| {
                    let canonical = canonical_hash_key(key, engine.prehash_seed);
                    engine.xor.contains_u64(canonical)
                })
                .collect(),
        }
    }

    pub fn has_batch(&self, keys: &[&[u8]]) -> Vec<bool> {
        self.contains_batch(keys)
    }

    pub fn exists_batch(&self, keys: &[&[u8]]) -> Vec<bool> {
        self.contains_batch(keys)
    }

    pub fn lookup_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>> {
        let mut out = Vec::with_capacity(keys.len());
        match &self.engine {
            Engine::Pgm(engine) => {
                let mut i = 0usize;
                #[cfg(target_arch = "x86_64")]
                while i + 16 <= keys.len() {
                    prefetch_key_batch(keys, i, 16);
                    for j in 0..16 {
                        let key = keys[i + j];
                        let res = match try_parse_u64(key) {
                            Some(num) => self.lookup_pgm(engine, num).ok(),
                            None => None,
                        };
                        out.push(res);
                    }
                    i += 16;
                }
                #[cfg(target_arch = "x86_64")]
                while i + 8 <= keys.len() {
                    prefetch_key_batch(keys, i, 8);
                    for j in 0..8 {
                        let key = keys[i + j];
                        let res = match try_parse_u64(key) {
                            Some(num) => self.lookup_pgm(engine, num).ok(),
                            None => None,
                        };
                        out.push(res);
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
                        let res = match try_parse_u64(key) {
                            Some(num) => self.lookup_pgm(engine, num).ok(),
                            None => None,
                        };
                        out.push(res);
                    }
                    i += 8;
                }
                while i + 4 <= keys.len() {
                    for j in 0..4 {
                        let key = keys[i + j];
                        Self::simd_touch(key);
                        let res = match try_parse_u64(key) {
                            Some(num) => self.lookup_pgm(engine, num).ok(),
                            None => None,
                        };
                        out.push(res);
                    }
                    i += 4;
                }
                while i < keys.len() {
                    let key = keys[i];
                    Self::simd_touch(key);
                    let res = match try_parse_u64(key) {
                        Some(num) => self.lookup_pgm(engine, num).ok(),
                        None => None,
                    };
                    out.push(res);
                    i += 1;
                }
            }
            Engine::Mph(engine) => {
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
            }
        }
        out
    }

    pub fn get_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>> {
        self.lookup_batch(keys)
    }

    pub fn len(&self) -> usize {
        self.key_count
    }

    pub fn stats(&self) -> IndexStats {
        match &self.engine {
            Engine::Pgm(engine) => {
                let pgm_memory = engine.pgm.stats().memory_usage;
                let mph_memory = std::mem::size_of_val(&engine.mph)
                    + engine.mph.g.len() * std::mem::size_of::<u32>();
                let xor_memory = engine.xor.memory_usage();
                let fp_memory = engine.fingerprints.len() * std::mem::size_of::<u16>();
                let hot_memory = engine.hot.as_ref().map(|h| h.memory_usage()).unwrap_or(0);
                IndexStats {
                    engine: "pgm",
                    total_keys: self.key_count,
                    mph_memory,
                    pgm_memory,
                    total_memory: mph_memory + pgm_memory + xor_memory + fp_memory + hot_memory,
                }
            }
            Engine::Mph(engine) => {
                let mph_memory = engine.backend.memory_usage_bytes();
                let xor_memory = engine.xor.memory_usage();
                let fp_memory = engine.fingerprints.len() * std::mem::size_of::<u16>();
                IndexStats {
                    engine: "mph",
                    total_keys: self.key_count,
                    mph_memory,
                    pgm_memory: 0,
                    total_memory: mph_memory + xor_memory + fp_memory,
                }
            }
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

    pub fn to_bytes(&self) -> Result<Vec<u8>, IndexError> {
        let mut out = Vec::new();
        match &self.engine {
            Engine::Mph(engine) => {
                write_u8(&mut out, 0);
                write_u64(&mut out, self.key_count as u64);
                write_u64(&mut out, engine.prehash_seed);
                engine.backend.write_to(&mut out);
                engine.xor.write_to(&mut out);
                write_fingerprints(&mut out, engine.fingerprints.as_ref());
            }
            Engine::Pgm(engine) => {
                write_u8(&mut out, 1);
                write_u64(&mut out, self.key_count as u64);
                engine.pgm.write_to(&mut out);
                engine.xor.write_to(&mut out);
                write_mph(&mut out, &engine.mph);
                write_fingerprints(&mut out, engine.fingerprints.as_ref());
                match &engine.hot {
                    Some(hot) => {
                        write_u8(&mut out, 1);
                        hot.write_to(&mut out);
                    }
                    None => {
                        write_u8(&mut out, 0);
                    }
                }
            }
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
                let prehash_seed = cursor.read_u64().ok_or(IndexError::CorruptData)?;
                let mut pos = cursor.pos;
                let backend =
                    BackendDispatch::read_from(bytes, &mut pos).ok_or(IndexError::CorruptData)?;
                cursor.pos = pos;
                let mut xor_cursor = XorCursor::new(bytes);
                xor_cursor.pos = cursor.pos;
                let xor = Xor8::read_from(&mut xor_cursor).ok_or(IndexError::CorruptData)?;
                cursor.pos = xor_cursor.pos;
                let fingerprints = read_fingerprints(&mut cursor)?;
                Ok(Index {
                    engine: Engine::Mph(MphEngine {
                        backend,
                        prehash_seed,
                        xor,
                        fingerprints,
                    }),
                    key_count,
                })
            }
            1 => {
                let mut pos = cursor.pos;
                let pgm = PgmIndex::read_from(bytes, &mut pos)?;
                cursor.pos = pos;
                let mut xor_cursor = XorCursor::new(bytes);
                xor_cursor.pos = cursor.pos;
                let xor = Xor8::read_from(&mut xor_cursor).ok_or(IndexError::CorruptData)?;
                cursor.pos = xor_cursor.pos;
                let mph = read_mph(&mut cursor)?;
                let fingerprints = read_fingerprints(&mut cursor)?;
                let hot_flag = cursor.read_u8().ok_or(IndexError::CorruptData)?;
                let hot = if hot_flag == 1 {
                    let mut pos = cursor.pos;
                    let hot =
                        HotTierIndex::read_from(bytes, &mut pos).ok_or(IndexError::CorruptData)?;
                    Some(hot)
                } else {
                    None
                };
                Ok(Index {
                    engine: Engine::Pgm(PgmEngine {
                        pgm,
                        xor,
                        mph,
                        fingerprints,
                        hot,
                    }),
                    key_count,
                })
            }
            _ => Err(IndexError::CorruptData),
        }
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, IndexError> {
        Self::from_bytes(bytes)
    }

    fn lookup_mph(&self, engine: &MphEngine, key: &[u8]) -> Result<usize, IndexError> {
        let canonical = canonical_hash_key(key, engine.prehash_seed);
        if !engine.xor.contains_u64(canonical) {
            return Err(IndexError::KeyNotFound);
        }
        let idx = engine
            .backend
            .lookup(canonical)
            .ok_or(IndexError::KeyNotFound)? as usize;
        let fp = fingerprint16(hash_u64_det(canonical));
        // SAFETY: mph index is in [0..n), fingerprints.len() == n
        let ok = unsafe { *engine.fingerprints.get_unchecked(idx) == fp };
        if ok {
            Ok(idx)
        } else {
            Err(IndexError::KeyNotFound)
        }
    }

    fn lookup_pgm(&self, engine: &PgmEngine, key: u64) -> Result<usize, IndexError> {
        if let Some(hot) = engine.hot.as_ref() {
            if let Some(idx) = hot.lookup_u64(key) {
                return Ok(idx as usize);
            }
        }
        if !engine.pgm.filter_allows(key) {
            return Err(IndexError::KeyNotFound);
        }
        let hash = engine.xor.hash_u64(key);
        if !engine.xor.contains_hash(hash) {
            return Err(IndexError::KeyNotFound);
        }
        let global_idx = engine.pgm.index(key)?;
        let seg_id = engine
            .pgm
            .segment_for_key(key)
            .ok_or(IndexError::KeyNotFound)?;
        let remap_id = remap_id_from_index(&engine.pgm, seg_id, global_idx);
        let idx = engine.mph.index(&remap_id.to_le_bytes()) as usize;
        let fp = fingerprint16(hash_u64_det(remap_id));
        // SAFETY: mph index is in [0..n), fingerprints.len() == n
        let ok = unsafe { *engine.fingerprints.get_unchecked(idx) == fp };
        if ok {
            Ok(global_idx)
        } else {
            Err(IndexError::KeyNotFound)
        }
    }
}

#[inline(always)]
fn make_backend_cfg(config: &IndexConfig) -> BackendConfig {
    BackendConfig {
        backend: config.backend,
        hot_fraction: config.hot_fraction,
        hot_backend: config.hot_backend,
        cold_backend: config.cold_backend,
        enable_parallel_build: config.enable_parallel_build,
        seed: config.mph_config.salt,
        gamma: config.mph_config.gamma,
        rehash_limit: config.mph_config.rehash_limit,
        max_pilot_attempts: 8_192,
        build_profile: if config.build_fast_profile {
            BuildProfile::Fast
        } else {
            BuildProfile::Balanced
        },
        fast_fail_rounds: if config.build_fast_profile { 3 } else { 2 },
        frequencies: None,
    }
}

fn run_build_pipeline(
    arena: &KeyArena,
    config: &IndexConfig,
) -> Result<(u64, Vec<u64>, BackendDispatch, Box<[u16]>, Xor8), IndexError> {
    let (prehash_seed, canonical) = prehash_u64_arena(
        arena.bytes.as_slice(),
        arena.offsets.as_slice(),
        config.mph_config.salt,
        !config.build_fast_profile,
    )
    .ok_or(IndexError::CorruptData)?;

    let xor = build_xor_u64(&canonical)?;
    let backend_cfg = make_backend_cfg(config);
    let backend = build_dispatch(&canonical, &backend_cfg);
    let fingerprints = build_fingerprints_hashed(&backend, &canonical).into_boxed_slice();
    Ok((prehash_seed, canonical, backend, fingerprints, xor))
}

#[cfg(feature = "parallel")]
fn run_build_pipeline_with_pool(
    arena: &KeyArena,
    config: &IndexConfig,
) -> Result<(u64, Vec<u64>, BackendDispatch, Box<[u16]>, Xor8), IndexError> {
    if !config.enable_parallel_build {
        return run_build_pipeline(arena, config);
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(detect_build_threads())
        .start_handler(|idx| {
            #[allow(unused_variables)]
            {
                #[cfg(feature = "parallel")]
                if let Some(cores) = select_affinity_cores() {
                    if !cores.is_empty() {
                        let core_id = cores[idx % cores.len()];
                        let _ =
                            core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                    }
                }
            }
        })
        .build();

    match pool {
        Ok(pool) => pool.install(|| run_build_pipeline(arena, config)),
        Err(_) => run_build_pipeline(arena, config),
    }
}

#[cfg(not(feature = "parallel"))]
fn run_build_pipeline_with_pool(
    arena: &KeyArena,
    config: &IndexConfig,
) -> Result<(u64, Vec<u64>, BackendDispatch, Box<[u16]>, Xor8), IndexError> {
    run_build_pipeline(arena, config)
}

#[cfg(feature = "parallel")]
fn detect_build_threads() -> usize {
    if let Some(v) = std::env::var_os("KIRA_BUILD_THREADS") {
        if let Ok(parsed) = v.to_string_lossy().parse::<usize>() {
            return parsed.max(1);
        }
    }
    if let Ok(threads) = std::thread::available_parallelism() {
        let t = threads.get();
        #[cfg(target_arch = "x86_64")]
        {
            return (t / 2).clamp(4, 8);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            return t.clamp(2, 8);
        }
    }
    4
}

#[cfg(feature = "parallel")]
fn select_affinity_cores() -> Option<Vec<usize>> {
    if let Some(v) = std::env::var_os("KIRA_BUILD_CORE_IDS") {
        let ids = v
            .to_string_lossy()
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect::<Vec<_>>();
        if !ids.is_empty() {
            return Some(ids);
        }
    }

    let core_ids = core_affinity::get_core_ids()?;
    if core_ids.is_empty() {
        return None;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let half = (core_ids.len() / 2).clamp(1, 8);
        return Some(core_ids.iter().take(half).map(|c| c.id).collect());
    }

    Some(core_ids.iter().map(|c| c.id).collect())
}

#[inline(always)]
fn canonical_hash_key(key: &[u8], seed: u64) -> u64 {
    if key.len() == 8 {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(key);
        let v = u64::from_le_bytes(arr);
        crate::simd_hash::hash_u64_one(v, seed)
    } else {
        wyhash::wyhash(key, seed)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_key_batch(keys: &[&[u8]], i: usize, window: usize) {
    const DIST: usize = 24;
    let pf = i + DIST;
    if pf + window <= keys.len() {
        for j in 0..window {
            let ptr = keys[pf + j].as_ptr() as *const i8;
            // SAFETY: prefetch is a hint; pointer is derived from valid slice.
            unsafe { _mm_prefetch(ptr, _MM_HINT_T0) };
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

    pub fn with_hot_backend(mut self, backend: BackendKind) -> Self {
        self.config.hot_backend = backend;
        self
    }

    pub fn with_cold_backend(mut self, backend: BackendKind) -> Self {
        self.config.cold_backend = backend;
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

fn try_parse_u64(bytes: &[u8]) -> Option<u64> {
    if bytes.len() != 8 {
        return None;
    }
    let mut array = [0u8; 8];
    array.copy_from_slice(bytes);
    Some(u64::from_le_bytes(array))
}

struct KeyArena {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
}

impl KeyArena {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn key_at(&self, idx: usize) -> &[u8] {
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        &self.bytes[start..end]
    }

    fn keys(&self) -> KeyArenaIter<'_> {
        KeyArenaIter {
            arena: self,
            idx: 0,
        }
    }
}

struct KeyArenaIter<'a> {
    arena: &'a KeyArena,
    idx: usize,
}

impl<'a> Iterator for KeyArenaIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.arena.len() {
            return None;
        }
        let out = self.arena.key_at(self.idx);
        self.idx += 1;
        Some(out)
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
    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::with_capacity(keys.len() * 2);

    for key in keys {
        let k = key.as_ref();
        let h = crate::build_hasher::fast_hash_bytes(k);
        if let Some(indices) = buckets.get(&h) {
            for &idx in indices {
                let start = offsets[idx] as usize;
                let end = offsets[idx + 1] as usize;
                if &bytes[start..end] == k {
                    return Err(MphError::DuplicateKey.into());
                }
            }
        }

        let idx = offsets.len() - 1;
        bytes.extend_from_slice(k);
        offsets.push(bytes.len() as u32);
        buckets.entry(h).or_default().push(idx);
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

fn build_xor_u64(keys: &[u64]) -> Result<Xor8, IndexError> {
    let mut seed = 0xD1B5_4A32_D192_ED03u64;
    for _ in 0..16 {
        if let Ok(xor) = Xor8::build_from_u64(keys, seed) {
            return Ok(xor);
        }
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    }
    Err(IndexError::CorruptData)
}

fn build_fingerprints_hashed(backend: &BackendDispatch, keys: &[u64]) -> Vec<u16> {
    let mut fps = vec![0u16; keys.len()];
    for &k in keys {
        let idx = backend.lookup(k).expect("backend must map training keys") as usize;
        let fp = fingerprint16(hash_u64_det(k));
        fps[idx] = fp;
    }
    fps
}

fn build_fingerprints_u64(mph: &Mphf, keys: &[u64]) -> Vec<u16> {
    let mut fps = vec![0u16; keys.len()];
    for &key in keys {
        let idx = mph.index(&key.to_le_bytes()) as usize;
        let fp = fingerprint16(hash_u64_det(key));
        fps[idx] = fp;
    }
    fps
}

fn build_hot_tier(pgm: &PgmIndex, mph_config: &MphConfig) -> Option<HotTierIndex> {
    const HOT_FRACTION: f64 = 0.10;
    let segs = pgm.segment_density_order(HOT_FRACTION);
    if segs.is_empty() {
        return None;
    }
    let mut hot_keys = Vec::new();
    let mut hot_indices = Vec::new();
    let all_keys = pgm.keys();
    for seg_id in segs {
        if let Some((start, end)) = pgm.segment_bounds(seg_id) {
            let mut i = start as usize;
            let end = end as usize;
            while i < end {
                hot_keys.push(all_keys[i]);
                hot_indices.push(i as u32);
                i += 1;
            }
        }
    }
    HotTierIndex::build_from_u64(&hot_keys, &hot_indices, mph_config)
}

fn hash_bytes(key: &[u8]) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("crc") {
            let mut h = unsafe { hash_bytes_crc(key, 0xA24B_1F6F_1234_5678) };
            if is_aarch64_feature_detected!("aes") {
                h = unsafe { aes_mix_u64(h, 0xA24B_1F6F_1234_5678) };
            }
            return h;
        }
    }
    wyhash::wyhash(key, 0xA24B_1F6F_1234_5678)
}

fn hash_u64(key: u64) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("crc") {
            let mut h = unsafe { hash_u64_crc(key, 0xA24B_1F6F_1234_5678) };
            if is_aarch64_feature_detected!("aes") {
                h = unsafe { aes_mix_u64(h, 0xA24B_1F6F_1234_5678) };
            }
            return h;
        }
    }
    splitmix64(key ^ 0xA24B_1F6F_1234_5678)
}

#[inline]
fn hash_u64_det(key: u64) -> u64 {
    splitmix64(key ^ 0xA24B_1F6F_1234_5678)
}

fn fingerprint16(hash: u64) -> u16 {
    (hash & 0xFFFF) as u16
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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

    fn read_u32(&mut self) -> Option<u32> {
        if self.pos + 4 > self.buf.len() {
            return None;
        }
        let mut array = [0u8; 4];
        array.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Some(u32::from_le_bytes(array))
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

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_mph(out: &mut Vec<u8>, mph: &Mphf) {
    write_u64(out, mph.n);
    write_u32(out, mph.m);
    write_u64(out, mph.salt);
    write_u64(out, mph.g.len() as u64);
    for v in &mph.g {
        write_u32(out, *v);
    }
}

fn read_mph(cursor: &mut Cursor<'_>) -> Result<Mphf, IndexError> {
    let n = cursor.read_u64().ok_or(IndexError::CorruptData)?;
    let m = cursor.read_u32().ok_or(IndexError::CorruptData)?;
    let salt = cursor.read_u64().ok_or(IndexError::CorruptData)?;
    let g_len = cursor.read_u64().ok_or(IndexError::CorruptData)? as usize;
    let mut g = Vec::with_capacity(g_len);
    for _ in 0..g_len {
        g.push(cursor.read_u32().ok_or(IndexError::CorruptData)?);
    }
    Ok(Mphf { n, m, salt, g })
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
