use hashbrown::HashSet;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::hash::BuildHasherDefault;
use thiserror::Error;

/// Minimal perfect hash using PtrHash-style pilot assignment.
#[cfg_attr(
    feature = "serde",
    derive(
        Serialize,
        Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize
    )
)]
#[derive(Debug, Clone)]
pub struct Mphf {
    pub n: u64,
    pub m: u32,
    pub salt: u64,
    pub g: Vec<u32>,
}

const MODE_SHARDED_PTRHASH: u64 = 1u64 << 63;
const SHARD_BITS_SHIFT: u64 = 56;
const SHARD_BITS_MASK: u64 = 0x3F << SHARD_BITS_SHIFT;
const SALT_META_MASK: u64 = MODE_SHARDED_PTRHASH | SHARD_BITS_MASK;
const MAX_BUILD_SHARDS: usize = 16;

#[allow(dead_code)]
impl Mphf {
    #[inline]
    pub fn index(&self, key: &[u8]) -> u64 {
        let hash_salt = self.salt & !SALT_META_MASK;
        let kh = hash_key(key, hash_salt);
        let b = bucket_index(kh.h1, self.m as usize);
        let pilot = unsafe { *self.g.get_unchecked(b) } as u64;
        let shards = decode_shard_count(self.salt);
        if shards == 1 {
            return slot_index(kh.h2, kh.h3, pilot, self.n as usize) as u64;
        }
        let meta_base = self.m as usize;
        let shard = bucket_to_shard(b, shards);
        let start = unsafe { *self.g.get_unchecked(meta_base + shard) } as usize;
        let end = unsafe { *self.g.get_unchecked(meta_base + shard + 1) } as usize;
        let local = slot_index(kh.h2, kh.h3, pilot, end - start);
        (start + local) as u64
    }

    #[inline]
    #[allow(dead_code)]
    pub fn index_str(&self, s: &str) -> u64 {
        self.index(s.as_bytes())
    }

    #[inline]
    pub fn index_u64(&self, key: u64) -> u64 {
        let hash_salt = self.salt & !SALT_META_MASK;
        let base = crate::simd_hash::hash_u64_one(key, hash_salt ^ 0xD6E8_FD9B_D6E8_FD9B);
        let (h1, h2, h3) = derive_hash_triple(base);
        let b = bucket_index(h1, self.m as usize);
        let pilot = unsafe { *self.g.get_unchecked(b) } as u64;
        let shards = decode_shard_count(self.salt);
        if shards == 1 {
            return slot_index(h2, h3, pilot, self.n as usize) as u64;
        }
        let meta_base = self.m as usize;
        let shard = bucket_to_shard(b, shards);
        let start = unsafe { *self.g.get_unchecked(meta_base + shard) } as usize;
        let end = unsafe { *self.g.get_unchecked(meta_base + shard + 1) } as usize;
        let local = slot_index(h2, h3, pilot, end - start);
        (start + local) as u64
    }

    #[cfg(feature = "serde")]
    #[allow(dead_code)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, MphError> {
        let bytes = rkyv::to_bytes::<_, 1024>(self).map_err(|e| MphError::Serde(e.to_string()))?;
        Ok(bytes.to_vec())
    }

    #[cfg(feature = "serde")]
    #[allow(dead_code)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MphError> {
        let archived = unsafe { rkyv::archived_root::<Self>(bytes) };
        rkyv::Deserialize::deserialize(archived, &mut rkyv::Infallible)
            .map_err(|e| MphError::Serde(e.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct BuildConfig {
    pub gamma: f64,
    pub rehash_limit: u32,
    pub salt: u64,
}

impl Default for BuildConfig {
    fn default() -> Self {
        crate::cpu::detect_features().optimal_config()
    }
}

#[derive(Debug, Error)]
pub enum MphError {
    #[error("duplicate key detected during build")]
    DuplicateKey,
    #[error("could not place all keys after rehash attempts")]
    Unresolvable,
    #[cfg(feature = "serde")]
    #[error("serialization error: {0}")]
    Serde(String),
}

pub struct Builder {
    cfg: BuildConfig,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            cfg: BuildConfig::default(),
        }
    }

    pub fn with_config(mut self, cfg: BuildConfig) -> Self {
        self.cfg = cfg;
        self
    }

    pub fn build<K, I>(self, keys: I) -> Result<Mphf, MphError>
    where
        K: Borrow<[u8]>,
        I: IntoIterator<Item = K>,
    {
        let mut uniq = Vec::<Vec<u8>>::new();
        uniq.reserve(1024);
        let mut seen: HashSet<Vec<u8>, BuildHasherDefault<crate::build_hasher::FastBuildHasher>> =
            HashSet::default();
        for k in keys {
            let v = k.borrow().to_vec();
            if !seen.insert(v.clone()) {
                return Err(MphError::DuplicateKey);
            }
            uniq.push(v);
        }
        self.build_unique_vec(uniq)
    }

    pub fn build_unique_vec(self, uniq: Vec<Vec<u8>>) -> Result<Mphf, MphError> {
        self.build_unique_ref(&uniq)
    }

    pub fn build_unique_ref(self, uniq: &[Vec<u8>]) -> Result<Mphf, MphError> {
        let n = uniq.len();
        assert!(n > 0, "empty key set is not supported");

        for gamma in gamma_candidates(self.cfg.gamma) {
            for round in 0..=self.cfg.rehash_limit {
                let salt = mix_salt(self.cfg.salt ^ gamma.to_bits(), round);
                match try_build_ptrhash(uniq, n, salt, gamma) {
                    Ok(mut mph) => {
                        mph.salt = salt;
                        return Ok(mph);
                    }
                    Err(MphError::Unresolvable) => continue,
                    Err(e) => return Err(e),
                }
            }
        }

        for gamma in gamma_candidates(self.cfg.gamma) {
            for round in 0..=self.cfg.rehash_limit {
                let salt = mix_salt(
                    self.cfg.salt ^ gamma.to_bits() ^ 0xD1B5_4A32_D192_ED03,
                    round,
                );
                match try_build_ptrhash_sharded(uniq, n, salt, gamma) {
                    Ok(mph) => return Ok(mph),
                    Err(MphError::Unresolvable) => continue,
                    Err(e) => return Err(e),
                }
            }
        }

        Err(MphError::Unresolvable)
    }

    pub fn build_unique_u64_ref(self, uniq: &[u64]) -> Result<Mphf, MphError> {
        let n = uniq.len();
        assert!(n > 0, "empty key set is not supported");

        self.build_unique_u64_ref_fast(uniq)
    }

    pub fn build_unique_u64_ref_fast(self, uniq: &[u64]) -> Result<Mphf, MphError> {
        let n = uniq.len();
        assert!(n > 0, "empty key set is not supported");

        let mut gammas = vec![
            self.cfg.gamma.max(0.80),
            (self.cfg.gamma * 0.9).max(0.70),
            (self.cfg.gamma * 0.8).max(0.66),
        ];
        gammas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        gammas.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let rounds = self.cfg.rehash_limit.min(1).max(1);
        let quick_cap = 32_768u32;
        let full_cap = 65_536u32;

        for &gamma in &gammas {
            for round in 0..=rounds {
                let salt = mix_salt(self.cfg.salt ^ gamma.to_bits(), round);
                if let Ok(mph) = try_build_ptrhash_sharded_u64_cap(uniq, n, salt, gamma, quick_cap)
                {
                    return Ok(mph);
                }
            }
        }

        for &gamma in &gammas {
            for round in 0..=rounds {
                let salt = mix_salt(
                    self.cfg.salt ^ gamma.to_bits() ^ 0xD1B5_4A32_D192_ED03,
                    round,
                );
                if let Ok(mut mph) = try_build_ptrhash_u64_cap(uniq, n, salt, gamma, full_cap) {
                    mph.salt = salt;
                    return Ok(mph);
                }
            }
        }

        Err(MphError::Unresolvable)
    }

    pub fn build_unique_u64_ref_sharded_first(self, uniq: &[u64]) -> Result<Mphf, MphError> {
        let n = uniq.len();
        assert!(n > 0, "empty key set is not supported");

        let mut gammas = vec![
            self.cfg.gamma.max(0.55),
            (self.cfg.gamma * 0.9).max(0.55),
            (self.cfg.gamma * 0.8).max(0.55),
            0.55,
        ];
        gammas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        gammas.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let rounds = self.cfg.rehash_limit.max(1);

        for &gamma in &gammas {
            for round in 0..=rounds {
                let salt = mix_salt(
                    self.cfg.salt ^ gamma.to_bits() ^ 0xD1B5_4A32_D192_ED03,
                    round,
                );
                if let Ok(mph) = try_build_ptrhash_sharded_u64(uniq, n, salt, gamma) {
                    return Ok(mph);
                }
            }
        }

        for &gamma in &gammas {
            for round in 0..=rounds {
                let salt = mix_salt(self.cfg.salt ^ gamma.to_bits(), round);
                if let Ok(mut mph) = try_build_ptrhash_u64(uniq, n, salt, gamma) {
                    mph.salt = salt;
                    return Ok(mph);
                }
            }
        }

        Err(MphError::Unresolvable)
    }
}

fn try_build_ptrhash(keys: &[Vec<u8>], n: usize, salt: u64, gamma: f64) -> Result<Mphf, MphError> {
    let bucket_load = gamma.max(1.1);
    let buckets = ((n as f64 / bucket_load).ceil() as usize).max(1);

    let mut hashes = hash_keys(keys, salt);
    for h in &mut hashes {
        h.bucket = bucket_index(h.h1, buckets) as u32;
    }

    let mut counts = vec![0u32; buckets];
    for h in &hashes {
        unsafe { *counts.get_unchecked_mut(h.bucket as usize) += 1 };
    }

    let mut offsets = vec![0usize; buckets + 1];
    for i in 0..buckets {
        offsets[i + 1] = offsets[i] + counts[i] as usize;
    }

    let mut cur = offsets.clone();
    let mut items = vec![0u32; n];
    for (idx, h) in hashes.iter().enumerate() {
        let b = h.bucket as usize;
        unsafe {
            let pos = *cur.get_unchecked(b);
            *items.get_unchecked_mut(pos) = idx as u32;
            *cur.get_unchecked_mut(b) = pos + 1;
        }
    }

    let mut counts_usize = vec![0usize; buckets];
    for b in 0..buckets {
        counts_usize[b] = offsets[b + 1] - offsets[b];
    }
    let order = build_bucket_order_by_counts_usize(&counts_usize);

    let entries = assign_pilots_single_domain(&order, &offsets, &items, &hashes, salt, n)?;
    let mut pilots = vec![0u32; buckets];
    for (bucket, pilot) in entries {
        unsafe {
            *pilots.get_unchecked_mut(bucket as usize) = pilot;
        }
    }

    Ok(Mphf {
        n: n as u64,
        m: buckets as u32,
        salt,
        g: pilots,
    })
}

fn try_build_ptrhash_u64(keys: &[u64], n: usize, salt: u64, gamma: f64) -> Result<Mphf, MphError> {
    try_build_ptrhash_u64_cap(keys, n, salt, gamma, u32::MAX)
}

fn try_build_ptrhash_u64_cap(
    keys: &[u64],
    n: usize,
    salt: u64,
    gamma: f64,
    attempt_cap: u32,
) -> Result<Mphf, MphError> {
    let bucket_load = gamma.max(1.1);
    let buckets = ((n as f64 / bucket_load).ceil() as usize).max(1);

    let (h2, h3, bucket_idx) = hash_u64_arrays(keys, salt, buckets);
    let (offsets, items, order) = build_u64_bucket_layout(&bucket_idx, buckets);

    let entries = assign_pilots_single_domain_u64_cap(
        &order,
        &offsets,
        &items,
        &h2,
        &h3,
        salt,
        n,
        attempt_cap,
    )?;
    let mut pilots = vec![0u32; buckets];
    for (bucket, pilot) in entries {
        unsafe {
            *pilots.get_unchecked_mut(bucket as usize) = pilot;
        }
    }

    Ok(Mphf {
        n: n as u64,
        m: buckets as u32,
        salt,
        g: pilots,
    })
}

fn try_build_ptrhash_sharded(
    keys: &[Vec<u8>],
    n: usize,
    salt: u64,
    gamma: f64,
) -> Result<Mphf, MphError> {
    let bucket_load = gamma.max(1.1);
    let buckets = ((n as f64 / bucket_load).ceil() as usize).max(1);
    let shard_count = build_shard_count(buckets, n);

    let mut hashes = hash_keys(keys, salt);
    for h in &mut hashes {
        h.bucket = bucket_index(h.h1, buckets) as u32;
    }

    let mut counts = vec![0u32; buckets];
    for h in &hashes {
        unsafe { *counts.get_unchecked_mut(h.bucket as usize) += 1 };
    }

    let mut offsets = vec![0usize; buckets + 1];
    for i in 0..buckets {
        offsets[i + 1] = offsets[i] + counts[i] as usize;
    }

    let mut cur = offsets.clone();
    let mut items = vec![0u32; n];
    for (idx, h) in hashes.iter().enumerate() {
        let b = h.bucket as usize;
        unsafe {
            let pos = *cur.get_unchecked(b);
            *items.get_unchecked_mut(pos) = idx as u32;
            *cur.get_unchecked_mut(b) = pos + 1;
        }
    }

    let mut counts_usize = vec![0usize; buckets];
    for b in 0..buckets {
        counts_usize[b] = offsets[b + 1] - offsets[b];
    }
    let order = build_bucket_order_by_counts_usize(&counts_usize);

    let mut buckets_by_shard = vec![Vec::<u32>::new(); shard_count];
    let mut keys_per_shard = vec![0usize; shard_count];
    for &b in &order {
        let shard = bucket_to_shard(b as usize, shard_count);
        buckets_by_shard[shard].push(b);
        keys_per_shard[shard] += offsets[b as usize + 1] - offsets[b as usize];
    }
    let mut shard_prefix = vec![0usize; shard_count + 1];
    for i in 0..shard_count {
        shard_prefix[i + 1] = shard_prefix[i] + keys_per_shard[i];
    }
    if shard_prefix[shard_count] != n {
        return Err(MphError::Unresolvable);
    }

    let mut pilots = vec![0u32; buckets];
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let shard_results = (0..shard_count)
            .into_par_iter()
            .map(|shard| {
                assign_shard_pilots(
                    shard,
                    &buckets_by_shard[shard],
                    &offsets,
                    &items,
                    &hashes,
                    salt,
                    shard_prefix[shard],
                    shard_prefix[shard + 1],
                    u32::MAX,
                )
            })
            .collect::<Vec<_>>();
        for shard in shard_results {
            let entries = shard?;
            for (bucket, pilot) in entries {
                unsafe {
                    *pilots.get_unchecked_mut(bucket as usize) = pilot;
                }
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        for shard in 0..shard_count {
            let entries = assign_shard_pilots_u64(
                shard,
                &buckets_by_shard[shard],
                &offsets,
                &items,
                &hashes,
                salt,
                shard_prefix[shard],
                shard_prefix[shard + 1],
                u32::MAX,
            )?;
            for (bucket, pilot) in entries {
                pilots[bucket as usize] = pilot;
            }
        }
    }

    for &p in &shard_prefix {
        pilots.push(p as u32);
    }

    Ok(Mphf {
        n: n as u64,
        m: buckets as u32,
        salt: encode_sharded_salt(salt, shard_count),
        g: pilots,
    })
}

fn try_build_ptrhash_sharded_u64(
    keys: &[u64],
    n: usize,
    salt: u64,
    gamma: f64,
) -> Result<Mphf, MphError> {
    try_build_ptrhash_sharded_u64_cap(keys, n, salt, gamma, u32::MAX)
}

fn try_build_ptrhash_sharded_u64_cap(
    keys: &[u64],
    n: usize,
    salt: u64,
    gamma: f64,
    attempt_cap: u32,
) -> Result<Mphf, MphError> {
    let bucket_load = gamma.max(1.1);
    let buckets = ((n as f64 / bucket_load).ceil() as usize).max(1);
    let shard_count = build_shard_count(buckets, n);

    let (h2, h3, bucket_idx) = hash_u64_arrays(keys, salt, buckets);
    let (offsets, items, order) = build_u64_bucket_layout(&bucket_idx, buckets);

    let mut buckets_by_shard = vec![Vec::<u32>::new(); shard_count];
    let mut keys_per_shard = vec![0usize; shard_count];
    for &b in &order {
        let shard = bucket_to_shard(b as usize, shard_count);
        buckets_by_shard[shard].push(b);
        keys_per_shard[shard] += (offsets[b as usize + 1] - offsets[b as usize]) as usize;
    }
    let mut shard_prefix = vec![0usize; shard_count + 1];
    for i in 0..shard_count {
        shard_prefix[i + 1] = shard_prefix[i] + keys_per_shard[i];
    }
    if shard_prefix[shard_count] != n {
        return Err(MphError::Unresolvable);
    }

    let mut pilots = vec![0u32; buckets];
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let shard_results = (0..shard_count)
            .into_par_iter()
            .map(|shard| {
                assign_shard_pilots_u64(
                    shard,
                    &buckets_by_shard[shard],
                    &offsets,
                    &items,
                    &h2,
                    &h3,
                    salt,
                    shard_prefix[shard],
                    shard_prefix[shard + 1],
                    attempt_cap,
                )
            })
            .collect::<Vec<_>>();
        for shard in shard_results {
            let entries = shard?;
            for (bucket, pilot) in entries {
                unsafe {
                    *pilots.get_unchecked_mut(bucket as usize) = pilot;
                }
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        for shard in 0..shard_count {
            let entries = assign_shard_pilots(
                shard,
                &buckets_by_shard[shard],
                &offsets,
                &items,
                &h2,
                &h3,
                salt,
                shard_prefix[shard],
                shard_prefix[shard + 1],
                attempt_cap,
            )?;
            for (bucket, pilot) in entries {
                pilots[bucket as usize] = pilot;
            }
        }
    }

    for &p in &shard_prefix {
        pilots.push(p as u32);
    }

    Ok(Mphf {
        n: n as u64,
        m: buckets as u32,
        salt: encode_sharded_salt(salt, shard_count),
        g: pilots,
    })
}

#[inline]
fn hash_keys(keys: &[Vec<u8>], salt: u64) -> Vec<KeyHash> {
    if keys.is_empty() {
        return Vec::new();
    }

    if keys.iter().all(|k| k.len() == 8) {
        let s = salt ^ 0xD6E8_FD9B_D6E8_FD9B;
        let mut words = vec![0u64; keys.len()];
        for (i, key) in keys.iter().enumerate() {
            let v = unsafe { std::ptr::read_unaligned(key.as_ptr() as *const u64) };
            words[i] = u64::from_le(v);
        }
        let mut base = vec![0u64; keys.len()];
        crate::simd_hash::hash_u64(&words, s, &mut base);
        let mut out = Vec::with_capacity(keys.len());
        for b in base {
            let (h1, h2, h3) = derive_hash_triple(b);
            out.push(KeyHash {
                h1,
                h2,
                h3,
                bucket: 0,
            });
        }
        return out;
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        return keys
            .par_iter()
            .map(|k| hash_key(k, salt))
            .collect::<Vec<KeyHash>>();
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            out.push(hash_key(key, salt));
        }
        out
    }
}

#[inline]
fn hash_u64_keys(keys: &[u64], salt: u64) -> Vec<KeyHash> {
    if keys.is_empty() {
        return Vec::new();
    }
    let s = salt ^ 0xD6E8_FD9B_D6E8_FD9B;
    let mut base = vec![0u64; keys.len()];
    crate::simd_hash::hash_u64(keys, s, &mut base);
    let mut out = Vec::with_capacity(keys.len());
    for b in base {
        let (h1, h2, h3) = derive_hash_triple(b);
        out.push(KeyHash {
            h1,
            h2,
            h3,
            bucket: 0,
        });
    }
    out
}

#[inline]
fn hash_u64_arrays(keys: &[u64], salt: u64, buckets: usize) -> (Vec<u64>, Vec<u64>, Vec<u32>) {
    let n = keys.len();
    let s = salt ^ 0xD6E8_FD9B_D6E8_FD9B;
    let mut h2 = vec![0u64; n];
    crate::simd_hash::hash_u64(keys, s, &mut h2);
    let mut h3 = vec![0u64; n];
    let mut bucket_idx = vec![0u32; n];
    for i in 0..n {
        let (h1, h2v, h3v) = derive_hash_triple(unsafe { *h2.get_unchecked(i) });
        unsafe {
            *h2.get_unchecked_mut(i) = h2v;
            *h3.get_unchecked_mut(i) = h3v;
            *bucket_idx.get_unchecked_mut(i) = bucket_index(h1, buckets) as u32;
        }
    }
    (h2, h3, bucket_idx)
}

#[inline]
fn build_u64_bucket_layout(bucket_idx: &[u32], buckets: usize) -> (Vec<usize>, Vec<u32>, Vec<u32>) {
    let n = bucket_idx.len();
    let mut counts = vec![0usize; buckets];
    for &b in bucket_idx {
        unsafe { *counts.get_unchecked_mut(b as usize) += 1usize };
    }
    let order = build_bucket_order_by_counts_usize(&counts);

    let mut offsets = vec![0usize; buckets + 1];
    for i in 0..buckets {
        unsafe {
            *offsets.get_unchecked_mut(i + 1) =
                *offsets.get_unchecked(i) + *counts.get_unchecked(i);
        }
    }

    for i in 0..buckets {
        unsafe {
            *counts.get_unchecked_mut(i) = *offsets.get_unchecked(i);
        }
    }

    let mut items = vec![0u32; n];
    for (idx, &b_u32) in bucket_idx.iter().enumerate() {
        let b = b_u32 as usize;
        unsafe {
            let pos = *counts.get_unchecked(b);
            *items.get_unchecked_mut(pos) = idx as u32;
            *counts.get_unchecked_mut(b) = pos + 1;
        }
    }
    (offsets, items, order)
}

#[inline]
fn build_bucket_order_by_counts_usize(counts: &[usize]) -> Vec<u32> {
    if counts.is_empty() {
        return Vec::new();
    }
    let max_len = counts.iter().copied().max().unwrap_or(0usize);
    let mut freq = vec![0usize; max_len + 1];
    for &c in counts {
        freq[c] += 1;
    }
    let mut start = vec![0usize; max_len + 1];
    let mut acc = 0usize;
    for len in (0..=max_len).rev() {
        start[len] = acc;
        acc += freq[len];
    }
    let mut next = start;
    let mut order = vec![0u32; counts.len()];
    for (bucket, &c) in counts.iter().enumerate() {
        let pos = next[c];
        order[pos] = bucket as u32;
        next[c] = pos + 1;
    }
    order
}

#[inline]
fn hash_key(key: &[u8], salt: u64) -> KeyHash {
    let base = if key.len() == 8 {
        let word = unsafe { std::ptr::read_unaligned(key.as_ptr() as *const u64) };
        let word = u64::from_le(word);
        crate::simd_hash::hash_u64_one(word, salt ^ 0xD6E8_FD9B_D6E8_FD9B)
    } else {
        wyhash::wyhash(key, salt ^ 0xA076_1D64_78BD_642F)
    };
    let (h1, h2, h3) = derive_hash_triple(base);
    KeyHash {
        h1,
        h2,
        h3,
        bucket: 0,
    }
}

#[inline]
fn derive_hash_triple(base: u64) -> (u64, u64, u64) {
    let h1 = mix64(base ^ 0x9E37_79B9_7F4A_7C15);
    let h2 = mix64(base ^ 0xA24B_1F6F_DA39_2B31);
    let h3 = mix64(base ^ 0xE703_7ED1_A0B4_28DB) | 1;
    (h1, h2, h3)
}

fn assign_pilots_single_domain(
    order: &[u32],
    offsets: &[usize],
    items: &[u32],
    hashes: &[KeyHash],
    salt: u64,
    n: usize,
) -> Result<Vec<(u32, u32)>, MphError> {
    assign_pilots_single_domain_cap(order, offsets, items, hashes, salt, n, u32::MAX)
}

fn assign_pilots_single_domain_cap(
    order: &[u32],
    offsets: &[usize],
    items: &[u32],
    hashes: &[KeyHash],
    salt: u64,
    n: usize,
    attempt_cap: u32,
) -> Result<Vec<(u32, u32)>, MphError> {
    let mut occupied = vec![false; n];
    let mut seen_epoch = vec![0u32; n];
    let mut epoch = 1u32;

    let mut max_bucket_len = 0usize;
    for &b in order {
        let len = (offsets[b as usize + 1] - offsets[b as usize]) as usize;
        if len > max_bucket_len {
            max_bucket_len = len;
        }
    }

    let mut trial_slots = vec![0usize; max_bucket_len.max(1)];
    let mut out = Vec::with_capacity(order.len());

    for &b_u32 in order {
        let b = b_u32 as usize;
        let start = offsets[b];
        let end = offsets[b + 1];
        let len = end - start;
        if len == 0 {
            out.push((b_u32, 0));
            continue;
        }

        let max_attempts = pilot_attempts_for_bucket(len).min(attempt_cap.max((len as u32) * 8));
        let mut found = false;
        for attempt in 0..max_attempts {
            epoch = epoch.wrapping_add(1);
            if epoch == 0 {
                seen_epoch.fill(0);
                epoch = 1;
            }
            let pilot = pick_pilot(salt, b_u32, attempt);
            let mut valid = true;

            for i in 0..len {
                let idx = unsafe { *items.get_unchecked(start + i) } as usize;
                let h = unsafe { *hashes.get_unchecked(idx) };
                let slot = slot_index(h.h2, h.h3, pilot as u64, n);
                if unsafe { *occupied.get_unchecked(slot) } {
                    valid = false;
                    break;
                }
                let mark = unsafe { seen_epoch.get_unchecked_mut(slot) };
                if *mark == epoch {
                    valid = false;
                    break;
                }
                *mark = epoch;
                unsafe {
                    *trial_slots.get_unchecked_mut(i) = slot;
                }
            }

            if !valid {
                continue;
            }

            for i in 0..len {
                let slot = unsafe { *trial_slots.get_unchecked(i) };
                unsafe {
                    *occupied.get_unchecked_mut(slot) = true;
                }
            }

            out.push((b_u32, pilot));
            found = true;
            break;
        }

        if !found {
            return Err(MphError::Unresolvable);
        }
    }

    Ok(out)
}

fn assign_pilots_single_domain_u64_cap(
    order: &[u32],
    offsets: &[usize],
    items: &[u32],
    h2: &[u64],
    h3: &[u64],
    salt: u64,
    n: usize,
    attempt_cap: u32,
) -> Result<Vec<(u32, u32)>, MphError> {
    let mut occupied = vec![0u64; n.div_ceil(64)];
    let mut seen_epoch = vec![0u32; n];
    let mut epoch = 1u32;

    let mut max_bucket_len = 0usize;
    for &b in order {
        let len = offsets[b as usize + 1] - offsets[b as usize];
        if len > max_bucket_len {
            max_bucket_len = len;
        }
    }

    let mut trial_slots = vec![0usize; max_bucket_len.max(1)];
    let mut out = Vec::with_capacity(order.len());

    for &b_u32 in order {
        let b = b_u32 as usize;
        let start = offsets[b];
        let end = offsets[b + 1];
        let len = end - start;
        if len == 0 {
            out.push((b_u32, 0));
            continue;
        }

        let max_attempts =
            pilot_attempts_for_bucket(len).min(attempt_cap.max((len as u32).saturating_mul(8)));
        let mut found = false;
        for attempt in 0..max_attempts {
            epoch = epoch.wrapping_add(1);
            if epoch == 0 {
                seen_epoch.fill(0);
                epoch = 1;
            }
            let pilot = pick_pilot(salt, b_u32, attempt);
            let mut valid = true;

            for i in 0..len {
                let idx = unsafe { *items.get_unchecked(start + i) } as usize;
                let slot = slot_index(
                    unsafe { *h2.get_unchecked(idx) },
                    unsafe { *h3.get_unchecked(idx) },
                    pilot as u64,
                    n,
                );
                if bit_test(&occupied, slot) {
                    valid = false;
                    break;
                }
                let mark = unsafe { seen_epoch.get_unchecked_mut(slot) };
                if *mark == epoch {
                    valid = false;
                    break;
                }
                *mark = epoch;
                unsafe { *trial_slots.get_unchecked_mut(i) = slot };
            }

            if !valid {
                continue;
            }

            for i in 0..len {
                let slot = unsafe { *trial_slots.get_unchecked(i) };
                bit_set(&mut occupied, slot);
            }

            out.push((b_u32, pilot));
            found = true;
            break;
        }

        if !found {
            return Err(MphError::Unresolvable);
        }
    }

    Ok(out)
}

fn assign_shard_pilots(
    shard: usize,
    shard_buckets: &[u32],
    offsets: &[usize],
    items: &[u32],
    hashes: &[KeyHash],
    salt: u64,
    shard_start: usize,
    shard_end: usize,
    attempt_cap: u32,
) -> Result<Vec<(u32, u32)>, MphError> {
    let local_n = shard_end.saturating_sub(shard_start);
    if local_n == 0 {
        return Ok(Vec::new());
    }

    let mut occupied = vec![false; local_n];
    let mut seen_epoch = vec![0u32; local_n];
    let mut epoch = 1u32;
    let mut max_bucket_len = 0usize;
    for &b in shard_buckets {
        let len = (offsets[b as usize + 1] - offsets[b as usize]) as usize;
        if len > max_bucket_len {
            max_bucket_len = len;
        }
    }
    let mut trial_slots = vec![0usize; max_bucket_len.max(1)];
    let mut out = Vec::with_capacity(shard_buckets.len());

    for &b_u32 in shard_buckets {
        let b = b_u32 as usize;
        let start = offsets[b];
        let end = offsets[b + 1];
        let len = end - start;
        if len == 0 {
            out.push((b_u32, 0));
            continue;
        }

        let max_attempts =
            pilot_attempts_for_bucket(len).min(attempt_cap.max((len as u32).saturating_mul(8)));
        let mut found = false;
        for attempt in 0..max_attempts {
            epoch = epoch.wrapping_add(1);
            if epoch == 0 {
                seen_epoch.fill(0);
                epoch = 1;
            }
            let pilot = pick_pilot(salt ^ ((shard as u64) << 17), b_u32, attempt);
            let mut valid = true;
            for i in 0..len {
                let idx = unsafe { *items.get_unchecked(start + i) } as usize;
                let h = unsafe { *hashes.get_unchecked(idx) };
                let local_slot = slot_index(h.h2, h.h3, pilot as u64, local_n);
                if unsafe { *occupied.get_unchecked(local_slot) } {
                    valid = false;
                    break;
                }
                let mark = unsafe { seen_epoch.get_unchecked_mut(local_slot) };
                if *mark == epoch {
                    valid = false;
                    break;
                }
                *mark = epoch;
                unsafe {
                    *trial_slots.get_unchecked_mut(i) = local_slot;
                }
            }
            if !valid {
                continue;
            }
            for i in 0..len {
                let slot = unsafe { *trial_slots.get_unchecked(i) };
                unsafe { *occupied.get_unchecked_mut(slot) = true };
            }
            out.push((b_u32, pilot));
            found = true;
            break;
        }
        if !found {
            return Err(MphError::Unresolvable);
        }
    }

    Ok(out)
}

fn assign_shard_pilots_u64(
    shard: usize,
    shard_buckets: &[u32],
    offsets: &[usize],
    items: &[u32],
    h2: &[u64],
    h3: &[u64],
    salt: u64,
    shard_start: usize,
    shard_end: usize,
    attempt_cap: u32,
) -> Result<Vec<(u32, u32)>, MphError> {
    let local_n = shard_end.saturating_sub(shard_start);
    if local_n == 0 {
        return Ok(Vec::new());
    }

    let mut occupied = vec![0u64; local_n.div_ceil(64)];
    let mut seen_epoch = vec![0u32; local_n];
    let mut epoch = 1u32;
    let mut max_bucket_len = 0usize;
    for &b in shard_buckets {
        let len = offsets[b as usize + 1] - offsets[b as usize];
        if len > max_bucket_len {
            max_bucket_len = len;
        }
    }
    let mut trial_slots = vec![0usize; max_bucket_len.max(1)];
    let mut out = Vec::with_capacity(shard_buckets.len());

    for &b_u32 in shard_buckets {
        let b = b_u32 as usize;
        let start = offsets[b];
        let end = offsets[b + 1];
        let len = end - start;
        if len == 0 {
            out.push((b_u32, 0));
            continue;
        }

        let max_attempts = pilot_attempts_for_bucket(len).min(attempt_cap.max((len as u32) * 8));
        let mut found = false;
        for attempt in 0..max_attempts {
            epoch = epoch.wrapping_add(1);
            if epoch == 0 {
                seen_epoch.fill(0);
                epoch = 1;
            }
            let pilot = pick_pilot(salt ^ ((shard as u64) << 17), b_u32, attempt);
            let mut valid = true;
            for i in 0..len {
                let idx = unsafe { *items.get_unchecked(start + i) } as usize;
                let local_slot = slot_index(
                    unsafe { *h2.get_unchecked(idx) },
                    unsafe { *h3.get_unchecked(idx) },
                    pilot as u64,
                    local_n,
                );
                if bit_test(&occupied, local_slot) {
                    valid = false;
                    break;
                }
                let mark = unsafe { seen_epoch.get_unchecked_mut(local_slot) };
                if *mark == epoch {
                    valid = false;
                    break;
                }
                *mark = epoch;
                unsafe {
                    *trial_slots.get_unchecked_mut(i) = local_slot;
                }
            }
            if !valid {
                continue;
            }
            for i in 0..len {
                let slot = unsafe { *trial_slots.get_unchecked(i) };
                bit_set(&mut occupied, slot);
            }
            out.push((b_u32, pilot));
            found = true;
            break;
        }
        if !found {
            return Err(MphError::Unresolvable);
        }
    }

    Ok(out)
}

#[inline]
fn build_shard_count(buckets: usize, n: usize) -> usize {
    if buckets == 0 || n <= 1 {
        return 1;
    }
    let mut shards = 1usize;
    #[cfg(feature = "parallel")]
    {
        if let Some(v) = std::env::var_os("KIRA_PTRHASH_THREADS") {
            if let Ok(s) = v.to_string_lossy().parse::<usize>() {
                shards = s.max(1);
            }
        } else if cfg!(target_arch = "aarch64") {
            // On Apple M-series, limiting to P-cores usually improves build throughput.
            shards = 4;
        } else if cfg!(target_arch = "x86_64") {
            // On hybrid x86 (like i7-12700), memory-bound build tends to scale
            // better with a subset of cores than with all SMT threads.
            let t = rayon::current_num_threads();
            shards = (t / 2).clamp(4, 8);
        }
        let threads = rayon::current_num_threads();
        shards = shards.min(threads).min(MAX_BUILD_SHARDS).max(1);
    }
    shards.min(buckets).min(n).max(1)
}

#[inline]
fn bucket_to_shard(bucket: usize, shard_count: usize) -> usize {
    bucket % shard_count
}

#[inline]
fn decode_shard_count(salt: u64) -> usize {
    if (salt & MODE_SHARDED_PTRHASH) == 0 {
        return 1;
    }
    let c = ((salt & SHARD_BITS_MASK) >> SHARD_BITS_SHIFT) as usize;
    c.max(1)
}

#[inline]
fn encode_sharded_salt(salt: u64, shard_count: usize) -> u64 {
    let c = (shard_count as u64) & 0x3F;
    (salt & !SALT_META_MASK) | MODE_SHARDED_PTRHASH | (c << SHARD_BITS_SHIFT)
}

#[inline]
fn bucket_index(hash: u64, buckets: usize) -> usize {
    fast_reduce64(mix64(hash ^ 0x9E37_79B9_7F4A_7C15), buckets)
}

fn build_bucket_order_by_counts(counts: &[u32]) -> Vec<u32> {
    if counts.is_empty() {
        return Vec::new();
    }
    let max_len = counts.iter().copied().max().unwrap_or(0) as usize;
    let mut freq = vec![0usize; max_len + 1];
    for &c in counts {
        freq[c as usize] += 1;
    }
    let mut start = vec![0usize; max_len + 1];
    let mut acc = 0usize;
    for len in (0..=max_len).rev() {
        start[len] = acc;
        acc += freq[len];
    }
    let mut next = start;
    let mut order = vec![0u32; counts.len()];
    for (bucket, &c) in counts.iter().enumerate() {
        let pos = next[c as usize];
        order[pos] = bucket as u32;
        next[c as usize] = pos + 1;
    }
    order
}

#[inline]
fn bit_test(bits: &[u64], idx: usize) -> bool {
    let w = unsafe { *bits.get_unchecked(idx >> 6) };
    (w & (1u64 << (idx & 63))) != 0
}

#[inline]
fn bit_set(bits: &mut [u64], idx: usize) {
    let w = unsafe { bits.get_unchecked_mut(idx >> 6) };
    *w |= 1u64 << (idx & 63);
}

#[inline]
fn slot_index(h2: u64, h3: u64, pilot: u64, n: usize) -> usize {
    let mixed = mix64(
        h2 ^ h3.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ pilot.wrapping_mul(0xA24B_1F6F_DA39_2B31),
    );
    fast_reduce64(mixed, n)
}

#[inline]
fn pilot_attempts_for_bucket(len: usize) -> u32 {
    let base = match len {
        0..=2 => 512,
        3..=4 => 2_048,
        5..=8 => 8_192,
        9..=16 => 32_768,
        _ => 262_144,
    };
    base + (len as u32 * 256)
}

#[inline]
fn pick_pilot(salt: u64, bucket: u32, attempt: u32) -> u32 {
    if attempt == 0 {
        return 0;
    }
    let x = salt
        ^ ((bucket as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ ((attempt as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    (mix64(x) & 0xFFFF_FFFF) as u32
}

#[inline]
fn fast_reduce64(hash: u64, n: usize) -> usize {
    ((hash as u128 * n as u128) >> 64) as usize
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[derive(Copy, Clone)]
struct KeyHash {
    h1: u64,
    h2: u64,
    h3: u64,
    bucket: u32,
}

#[inline]
fn mix_salt(base: u64, round: u32) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET ^ base;
    h ^= round as u64;
    h = h.wrapping_mul(FNV_PRIME);
    h ^ (h >> 33)
}

#[inline]
fn gamma_candidates(base: f64) -> Vec<f64> {
    let mut out = vec![
        base.max(0.55),
        (base * 0.9).max(0.55),
        (base * 0.8).max(0.55),
        (base * 0.7).max(0.55),
        0.55,
        0.40,
        0.30,
        0.20,
        0.10,
    ];
    out.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    out
}
