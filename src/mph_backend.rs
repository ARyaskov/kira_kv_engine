//! MPH backend dispatch.
//!
//! Historically this module hosted several MPH algorithms (PTHash, CHD, RecSplit,
//! BBHash, plus the legacy PtrHash2025). They were removed in v0.5 — PtrHash25
//! (`ptrhash25.rs`, u8 pilots + 2-level bucketing + CompressedPilotsV2) consistently
//! beat all the alternatives on both build (4× faster) and lookup (1.5-3× faster)
//! while using 2.5× less memory. The other backends added complexity and confusion
//! without any workload where they won.
//!
//! `BackendKind` is kept as a 1-variant enum to preserve the public API shape; future
//! algorithms (e.g., true paper-style PtrHash 2025 with cuckoo relocation) would slot
//! in as new variants here.

#![allow(dead_code)]

use crate::ptrhash25::{self, Partitioned, PtrHash25Error};
use hashbrown::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// The only MPH algorithm. u8 pilots + 2-level bucketing + CompressedPilotsV2
    /// (3-tier zero/nibble/overflow) + hugepage-backed storage + partitioned,
    /// parallel, AVX2-vectorized build. See `src/ptrhash25.rs`.
    PtrHash25,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildProfile {
    Balanced,
    Fast,
}

#[derive(Debug, Clone)]
pub struct BuildConfig {
    pub backend: BackendKind,
    pub enable_parallel_build: bool,
    pub seed: u64,
    pub gamma: f64,
    pub rehash_limit: u32,
    pub build_profile: BuildProfile,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            backend: BackendKind::PtrHash25,
            enable_parallel_build: true,
            seed: 0xC0FF_EE00_D15E_A5E,
            gamma: 0.5, // sparse layout — required for u8-pilot convergence
            rehash_limit: 16,
            build_profile: BuildProfile::Fast,
        }
    }
}

impl BuildConfig {
    /// The PtrHash25 configuration derived from this backend configuration. The outer
    /// `Index` layer adds its own fingerprint table, so inner fingerprints are off.
    pub(crate) fn mph_config(&self) -> ptrhash25::BuildConfig {
        ptrhash25::BuildConfig {
            gamma: self.gamma,
            max_rehash: self.rehash_limit.max(8),
            with_fingerprints: false,
            seed: self.seed,
            use_aes_hash: false,
        }
    }
}

pub trait MphBackend {
    fn build(keys: &[u64], config: &BuildConfig) -> Self
    where
        Self: Sized;

    fn lookup(&self, key: u64) -> Option<u32>;

    fn memory_usage_bytes(&self) -> usize;
}

/// PtrHash 2025 backend — u8 pilots, 2-level bucketing, 3-tier compressed pilots,
/// hugepage-backed storage, partitioned parallel build. The default and currently
/// only backend.
///
/// Lookup wraps `crate::ptrhash25::PtrHash25Mphf` directly without going through a
/// generic dispatch enum, keeping the hot path branchless.
#[derive(Debug, Clone)]
pub struct PtrHash25Backend {
    pub(crate) storage: PtrHash25Storage,
}

#[derive(Debug, Clone)]
pub(crate) enum PtrHash25Storage {
    Mph(ptrhash25::PtrHash25Mphf),
    /// Fallback for the unbuildable-after-rehash case (vanishingly rare).
    Map(HashMap<u64, u32>),
}

impl MphBackend for PtrHash25Backend {
    fn build(keys: &[u64], config: &BuildConfig) -> Self {
        let cfg = config.mph_config();
        let part = ptrhash25::partition_keys(keys, &cfg);
        match Self::build_from_partitioned(&part, config, false) {
            Ok((backend, _)) => backend,
            // Duplicate u64 keys: keep the historical behaviour of this low-level entry
            // point (a map that keeps the last occurrence) — `Index` handles duplicates
            // explicitly before ever reaching here.
            Err(_) => Self {
                storage: PtrHash25Storage::Map(build_fallback_map_original(keys)),
            },
        }
    }

    #[inline]
    fn lookup(&self, key: u64) -> Option<u32> {
        match &self.storage {
            PtrHash25Storage::Mph(mph) => Some(mph.index_u64(key)),
            PtrHash25Storage::Map(map) => map.get(&key).copied(),
        }
    }

    fn memory_usage_bytes(&self) -> usize {
        match &self.storage {
            PtrHash25Storage::Mph(mph) => mph.memory_usage(),
            PtrHash25Storage::Map(map) => {
                std::mem::size_of::<HashMap<u64, u32>>()
                    + map.len() * (std::mem::size_of::<u64>() + std::mem::size_of::<u32>())
            }
        }
    }
}

impl PtrHash25Backend {
    /// Build from already-partitioned keys. With `outer_fp16 = true` the u16
    /// fingerprint table (`original_key & 0xFFFF` per slot) is filled inside the
    /// per-part build — no slot array, no second pass.
    ///
    /// `Err(DuplicateKey)` is reported to the caller; `Unresolvable` (astronomically
    /// rare) falls back to a hash map keyed by the original u64 keys.
    pub(crate) fn build_from_partitioned(
        part: &Partitioned,
        config: &BuildConfig,
        outer_fp16: bool,
    ) -> Result<(Self, Option<Vec<u16>>), PtrHash25Error> {
        let cfg = config.mph_config();
        let outputs = ptrhash25::BuildOutputs {
            slots: false,
            fp16: outer_fp16,
        };
        match ptrhash25::build_partitioned_with(part, &cfg, outputs) {
            Ok((mph, _, fp16)) => Ok((
                Self {
                    storage: PtrHash25Storage::Mph(mph),
                },
                fp16,
            )),
            Err(PtrHash25Error::DuplicateKey) => Err(PtrHash25Error::DuplicateKey),
            Err(PtrHash25Error::Unresolvable) => {
                let n = part.len();
                let mut map = HashMap::with_capacity(n * 2);
                let mut fp16 = if outer_fp16 { Some(vec![0u16; n]) } else { None };
                for i in 0..n {
                    let key = part.original_key(i);
                    map.insert(key, i as u32);
                    if let Some(fp) = fp16.as_mut() {
                        fp[i] = (key & 0xFFFF) as u16;
                    }
                }
                Ok((
                    Self {
                        storage: PtrHash25Storage::Map(map),
                    },
                    fp16,
                ))
            }
        }
    }

    /// Upper bound on `lookup`'s return value.
    pub fn slot_capacity(&self) -> usize {
        match &self.storage {
            PtrHash25Storage::Mph(mph) => mph.slot_capacity(),
            PtrHash25Storage::Map(map) => {
                map.values().copied().max().map_or(0, |m| m as usize + 1)
            }
        }
    }
}

/// Single-variant dispatch wrapper. Kept as an enum so future algorithms can be
/// added as additional variants without breaking callers that match on it.
#[derive(Debug)]
pub enum BackendDispatch {
    PtrHash25(PtrHash25Backend),
}

impl BackendDispatch {
    pub fn kind(&self) -> BackendKind {
        match self {
            Self::PtrHash25(_) => BackendKind::PtrHash25,
        }
    }

    #[inline]
    pub fn lookup(&self, key: u64) -> Option<u32> {
        match self {
            Self::PtrHash25(b) => b.lookup(key),
        }
    }

    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::PtrHash25(b) => b.memory_usage_bytes(),
        }
    }

    pub fn slot_capacity(&self) -> usize {
        match self {
            Self::PtrHash25(b) => b.slot_capacity(),
        }
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        match self {
            Self::PtrHash25(b) => {
                out.push(0);
                write_ptrhash25_storage(&b.storage, out);
            }
        }
    }

    pub fn read_from(buf: &[u8], pos: &mut usize) -> Option<Self> {
        let tag = read_u8(buf, pos)?;
        match tag {
            0 => {
                let storage = read_ptrhash25_storage(buf, pos)?;
                Some(Self::PtrHash25(PtrHash25Backend { storage }))
            }
            _ => None,
        }
    }
}

pub fn build_dispatch(keys: &[u64], cfg: &BuildConfig) -> BackendDispatch {
    match cfg.backend {
        BackendKind::PtrHash25 => BackendDispatch::PtrHash25(PtrHash25Backend::build(keys, cfg)),
    }
}

fn write_ptrhash25_storage(storage: &PtrHash25Storage, out: &mut Vec<u8>) {
    match storage {
        PtrHash25Storage::Mph(mph) => {
            out.push(0);
            ptrhash25::write_ptrhash25(mph, out);
        }
        PtrHash25Storage::Map(map) => {
            out.push(1);
            out.extend_from_slice(&(map.len() as u64).to_le_bytes());
            for (k, v) in map {
                out.extend_from_slice(&k.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

fn read_ptrhash25_storage(buf: &[u8], pos: &mut usize) -> Option<PtrHash25Storage> {
    let tag = read_u8(buf, pos)?;
    match tag {
        0 => ptrhash25::read_ptrhash25(buf, pos).map(PtrHash25Storage::Mph),
        1 => {
            let len = read_u64(buf, pos)? as usize;
            let mut map = HashMap::with_capacity(len * 2);
            for _ in 0..len {
                let k = read_u64(buf, pos)?;
                let v = read_u32(buf, pos)?;
                map.insert(k, v);
            }
            Some(PtrHash25Storage::Map(map))
        }
        _ => None,
    }
}

fn build_fallback_map_original(keys: &[u64]) -> HashMap<u64, u32> {
    let mut map = HashMap::with_capacity(keys.len() * 2);
    for (i, &k) in keys.iter().enumerate() {
        map.insert(k, i as u32);
    }
    map
}

fn read_u8(buf: &[u8], pos: &mut usize) -> Option<u8> {
    if *pos + 1 > buf.len() {
        return None;
    }
    let v = buf[*pos];
    *pos += 1;
    Some(v)
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Option<u32> {
    if *pos + 4 > buf.len() {
        return None;
    }
    let mut a = [0u8; 4];
    a.copy_from_slice(&buf[*pos..*pos + 4]);
    *pos += 4;
    Some(u32::from_le_bytes(a))
}

fn read_u64(buf: &[u8], pos: &mut usize) -> Option<u64> {
    if *pos + 8 > buf.len() {
        return None;
    }
    let mut a = [0u8; 8];
    a.copy_from_slice(&buf[*pos..*pos + 8]);
    *pos += 8;
    Some(u64::from_le_bytes(a))
}
