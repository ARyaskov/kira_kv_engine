use crate::bdz::{BuildConfig as MphConfig, Builder as MphBuilder, MphError, Mphf};
use crate::hot_tier::HotTierIndex;
use crate::pgm::{PgmBuilder, PgmError, PgmIndex};
use crate::remap::{remap_id_from_index, remap_ids_for_pgm};
use crate::xor_filter::{Cursor as XorCursor, Xor8};
use hashbrown::HashMap;
use thiserror::Error;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaeseq_u8, vdupq_n_u64, vgetq_lane_u64, vld1q_u8, vreinterpretq_u8_u64, vreinterpretq_u64_u8,
};
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m128i, _mm_loadu_si128};

#[derive(Debug)]
struct MphEngine {
    mph: Mphf,
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

/// Hybrid index: auto-selects PGM for purely numeric keys, MPH otherwise
pub struct HybridIndex {
    engine: Engine,
    key_count: usize,
}

#[derive(Debug, Error)]
pub enum HybridError {
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

impl From<MphError> for HybridError {
    fn from(err: MphError) -> Self {
        HybridError::Mph(err.to_string())
    }
}

impl From<PgmError> for HybridError {
    fn from(err: PgmError) -> Self {
        HybridError::Pgm(err.to_string())
    }
}

/// Configuration for hybrid index
#[derive(Debug, Clone)]
pub struct HybridConfig {
    pub mph_config: MphConfig,
    pub pgm_epsilon: u32,
    pub auto_detect_numeric: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        crate::cpu::detect_features().optimal_hybrid_config()
    }
}

#[derive(Debug, Clone)]
pub struct HybridStats {
    pub engine: &'static str,
    pub total_keys: usize,
    pub mph_memory: usize,
    pub pgm_memory: usize,
    pub total_memory: usize,
}

impl HybridIndex {
    pub fn build_index<K>(keys: Vec<K>, config: HybridConfig) -> Result<Self, HybridError>
    where
        K: AsRef<[u8]>,
    {
        if keys.is_empty() {
            return Err(HybridError::InvalidKey);
        }

        let byte_keys: Vec<Vec<u8>> = keys.into_iter().map(|k| k.as_ref().to_vec()).collect();
        let mut byte_keys = dedup_unique_keys(byte_keys)?;
        permute_keys_for_builder(&mut byte_keys, config.mph_config.salt);
        let key_count = byte_keys.len();

        let mut numeric_keys = Vec::with_capacity(byte_keys.len());
        let mut all_numeric = false;
        if config.auto_detect_numeric {
            all_numeric = true;
            for key in &byte_keys {
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

            Ok(HybridIndex {
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
            let xor = build_xor_bytes(&byte_keys)?;
            let mph = MphBuilder::new()
                .with_config(config.mph_config)
                .build_unique_ref(&byte_keys)?;
            let fingerprints = build_fingerprints_bytes(&mph, &byte_keys);

            Ok(HybridIndex {
                engine: Engine::Mph(MphEngine {
                    mph,
                    xor,
                    fingerprints: fingerprints.into_boxed_slice(),
                }),
                key_count,
            })
        }
    }

    #[inline]
    fn simd_touch(key: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if key.len() >= 16 {
                let _ = _mm_loadu_si128(key.as_ptr() as *const __m128i);
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if key.len() >= 16 {
                let _ = vld1q_u8(key.as_ptr());
            }
        }
    }

    pub fn lookup(&self, key: &[u8]) -> Result<usize, HybridError> {
        match &self.engine {
            Engine::Pgm(engine) => {
                let num = try_parse_u64(key).ok_or(HybridError::InvalidKey)?;
                self.lookup_pgm(engine, num)
            }
            Engine::Mph(engine) => self.lookup_mph(engine, key),
        }
    }

    pub fn get(&self, key: &[u8]) -> Result<usize, HybridError> {
        self.lookup(key)
    }

    pub fn lookup_str(&self, key: &str) -> Result<usize, HybridError> {
        self.lookup(key.as_bytes())
    }

    pub fn get_str(&self, key: &str) -> Result<usize, HybridError> {
        self.lookup_str(key)
    }

    pub fn lookup_u64(&self, key: u64) -> Result<usize, HybridError> {
        match &self.engine {
            Engine::Pgm(engine) => self.lookup_pgm(engine, key),
            Engine::Mph(engine) => self.lookup_mph(engine, &key.to_le_bytes()),
        }
    }

    pub fn get_u64(&self, key: u64) -> Result<usize, HybridError> {
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
            Engine::Mph(engine) => engine.xor.contains_bytes(key),
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
                .map(|&key| engine.xor.contains_bytes(key))
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

    pub fn stats(&self) -> HybridStats {
        match &self.engine {
            Engine::Pgm(engine) => {
                let pgm_memory = engine.pgm.stats().memory_usage;
                let mph_memory = std::mem::size_of_val(&engine.mph)
                    + engine.mph.g.len() * std::mem::size_of::<u32>();
                let xor_memory = engine.xor.memory_usage();
                let fp_memory = engine.fingerprints.len() * std::mem::size_of::<u16>();
                let hot_memory = engine.hot.as_ref().map(|h| h.memory_usage()).unwrap_or(0);
                HybridStats {
                    engine: "pgm",
                    total_keys: self.key_count,
                    mph_memory,
                    pgm_memory,
                    total_memory: mph_memory + pgm_memory + xor_memory + fp_memory + hot_memory,
                }
            }
            Engine::Mph(engine) => {
                let mph_memory = std::mem::size_of_val(&engine.mph)
                    + engine.mph.g.len() * std::mem::size_of::<u32>();
                let xor_memory = engine.xor.memory_usage();
                let fp_memory = engine.fingerprints.len() * std::mem::size_of::<u16>();
                HybridStats {
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
        println!("Hybrid Index Statistics:");
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

    pub fn to_bytes(&self) -> Result<Vec<u8>, HybridError> {
        let mut out = Vec::new();
        match &self.engine {
            Engine::Mph(engine) => {
                write_u8(&mut out, 0);
                write_u64(&mut out, self.key_count as u64);
                write_mph(&mut out, &engine.mph);
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

    pub fn serialize(&self) -> Result<Vec<u8>, HybridError> {
        self.to_bytes()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HybridError> {
        let mut cursor = Cursor::new(bytes);
        let tag = cursor.read_u8().ok_or(HybridError::CorruptData)?;
        let key_count = cursor.read_u64().ok_or(HybridError::CorruptData)? as usize;
        match tag {
            0 => {
                let mph = read_mph(&mut cursor)?;
                let mut xor_cursor = XorCursor::new(bytes);
                xor_cursor.pos = cursor.pos;
                let xor = Xor8::read_from(&mut xor_cursor).ok_or(HybridError::CorruptData)?;
                cursor.pos = xor_cursor.pos;
                let fingerprints = read_fingerprints(&mut cursor)?;
                Ok(HybridIndex {
                    engine: Engine::Mph(MphEngine {
                        mph,
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
                let xor = Xor8::read_from(&mut xor_cursor).ok_or(HybridError::CorruptData)?;
                cursor.pos = xor_cursor.pos;
                let mph = read_mph(&mut cursor)?;
                let fingerprints = read_fingerprints(&mut cursor)?;
                let hot_flag = cursor.read_u8().ok_or(HybridError::CorruptData)?;
                let hot = if hot_flag == 1 {
                    let mut pos = cursor.pos;
                    let hot =
                        HotTierIndex::read_from(bytes, &mut pos).ok_or(HybridError::CorruptData)?;
                    Some(hot)
                } else {
                    None
                };
                Ok(HybridIndex {
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
            _ => Err(HybridError::CorruptData),
        }
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, HybridError> {
        Self::from_bytes(bytes)
    }

    fn lookup_mph(&self, engine: &MphEngine, key: &[u8]) -> Result<usize, HybridError> {
        let hash = engine.xor.hash_bytes(key);
        if !engine.xor.contains_hash(hash) {
            return Err(HybridError::KeyNotFound);
        }
        let idx = engine.mph.index(key) as usize;
        let fp = fingerprint16(hash);
        // SAFETY: mph index is in [0..n), fingerprints.len() == n
        let ok = unsafe { *engine.fingerprints.get_unchecked(idx) == fp };
        if ok {
            Ok(idx)
        } else {
            Err(HybridError::KeyNotFound)
        }
    }

    fn lookup_pgm(&self, engine: &PgmEngine, key: u64) -> Result<usize, HybridError> {
        if let Some(hot) = engine.hot.as_ref() {
            if let Some(idx) = hot.lookup_u64(key) {
                return Ok(idx as usize);
            }
        }
        if !engine.pgm.filter_allows(key) {
            return Err(HybridError::KeyNotFound);
        }
        let hash = engine.xor.hash_u64(key);
        if !engine.xor.contains_hash(hash) {
            return Err(HybridError::KeyNotFound);
        }
        let global_idx = engine.pgm.index(key)?;
        let seg_id = engine
            .pgm
            .segment_for_key(key)
            .ok_or(HybridError::KeyNotFound)?;
        let remap_id = remap_id_from_index(&engine.pgm, seg_id, global_idx);
        let idx = engine.mph.index(&remap_id.to_le_bytes()) as usize;
        let fp = fingerprint16(hash_u64_det(remap_id));
        // SAFETY: mph index is in [0..n), fingerprints.len() == n
        let ok = unsafe { *engine.fingerprints.get_unchecked(idx) == fp };
        if ok {
            Ok(global_idx)
        } else {
            Err(HybridError::KeyNotFound)
        }
    }
}

/// Builder for hybrid index
pub struct HybridBuilder {
    config: HybridConfig,
}

impl HybridBuilder {
    pub fn new() -> Self {
        Self {
            config: HybridConfig::default(),
        }
    }

    pub fn with_config(mut self, config: HybridConfig) -> Self {
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

    pub fn auto_detect_numeric(mut self, enabled: bool) -> Self {
        self.config.auto_detect_numeric = enabled;
        self
    }

    pub fn build_index<K>(self, keys: Vec<K>) -> Result<HybridIndex, HybridError>
    where
        K: AsRef<[u8]>,
    {
        HybridIndex::build_index(keys, self.config)
    }
}

impl Default for HybridBuilder {
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

fn dedup_unique_keys(keys: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>, HybridError> {
    let mut uniq = Vec::with_capacity(keys.len());
    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::with_capacity(keys.len());
    for key in keys {
        let h = crate::build_hasher::fast_hash_bytes(&key);
        if let Some(indices) = buckets.get(&h) {
            for &idx in indices {
                if uniq[idx] == key {
                    return Err(MphError::DuplicateKey.into());
                }
            }
        }
        let idx = uniq.len();
        uniq.push(key);
        buckets.entry(h).or_default().push(idx);
    }
    Ok(uniq)
}

fn permute_keys_for_builder(keys: &mut [Vec<u8>], seed: u64) {
    if keys.len() <= 1 {
        return;
    }
    let mut s = seed ^ (keys.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let sample = keys.len().min(8);
    for i in 0..sample {
        s ^= crate::build_hasher::fast_hash_bytes(&keys[i]);
        s = xorshift64(s);
    }
    for i in (1..keys.len()).rev() {
        s = xorshift64(s);
        let j = (s % (i as u64 + 1)) as usize;
        keys.swap(i, j);
    }
}

#[inline]
fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn build_xor_bytes(keys: &[Vec<u8>]) -> Result<Xor8, HybridError> {
    let mut seed = 0xD1B5_4A32_D192_ED03u64;
    for _ in 0..16 {
        if let Ok(xor) = Xor8::build_from_bytes(keys, seed) {
            return Ok(xor);
        }
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    }
    Err(HybridError::CorruptData)
}

fn build_xor_u64(keys: &[u64]) -> Result<Xor8, HybridError> {
    let mut seed = 0xD1B5_4A32_D192_ED03u64;
    for _ in 0..16 {
        if let Ok(xor) = Xor8::build_from_u64(keys, seed) {
            return Ok(xor);
        }
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    }
    Err(HybridError::CorruptData)
}

fn build_fingerprints_bytes(mph: &Mphf, keys: &[Vec<u8>]) -> Vec<u16> {
    let mut fps = vec![0u16; keys.len()];
    for key in keys {
        let idx = mph.index(key) as usize;
        let fp = fingerprint16(hash_bytes(key));
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

fn read_mph(cursor: &mut Cursor<'_>) -> Result<Mphf, HybridError> {
    let n = cursor.read_u64().ok_or(HybridError::CorruptData)?;
    let m = cursor.read_u32().ok_or(HybridError::CorruptData)?;
    let salt = cursor.read_u64().ok_or(HybridError::CorruptData)?;
    let g_len = cursor.read_u64().ok_or(HybridError::CorruptData)? as usize;
    let mut g = Vec::with_capacity(g_len);
    for _ in 0..g_len {
        g.push(cursor.read_u32().ok_or(HybridError::CorruptData)?);
    }
    Ok(Mphf { n, m, salt, g })
}

fn write_fingerprints(out: &mut Vec<u8>, fps: &[u16]) {
    write_u64(out, fps.len() as u64);
    for &fp in fps {
        write_u16(out, fp);
    }
}

fn read_fingerprints(cursor: &mut Cursor<'_>) -> Result<Box<[u16]>, HybridError> {
    let len = cursor.read_u64().ok_or(HybridError::CorruptData)? as usize;
    let mut fps = Vec::with_capacity(len);
    for _ in 0..len {
        fps.push(cursor.read_u16().ok_or(HybridError::CorruptData)?);
    }
    Ok(fps.into_boxed_slice())
}
