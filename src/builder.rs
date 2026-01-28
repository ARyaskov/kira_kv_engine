use crate::hash::KeyHash;
use crate::util::BitSet;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use hashbrown::HashSet;
use std::hash::BuildHasherDefault;
use thiserror::Error;

/// Complete MPH structure: stores size, number of buckets, salt and displacement vector per bucket.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Mphf {
    pub n: u64,
    pub buckets: u64,
    pub salt: u64,
    pub disps: Vec<u64>, // len == buckets
}

impl Mphf {
    /// O(1) lookup. Uses the same formula as build.
    #[inline]
    pub fn index(&self, key: &[u8]) -> u64 {
        let kh = KeyHash::from_key(key, self.salt);
        let b = kh.bucket(self.buckets);
        let d = unsafe { *self.disps.get_unchecked(b) };
        kh.place(self.n, d) as u64
    }

    #[inline]
    pub fn index_str(&self, s: &str) -> u64 {
        self.index(s.as_bytes())
    }

    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, MphError> {
        Ok(bincode::serialize(self)?)
    }

    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MphError> {
        Ok(bincode::deserialize(bytes)?)
    }
}

/// Build parameters.
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Average bucket size. Smaller makes placement easier, but increases overhead buckets.
    pub target_bucket_size: f64,
    /// How many seeds/displacements to try per bucket before giving up and rehashing.
    pub max_seed_attempts: u32,
    /// Base salt (re-hash will deterministically add rounds to it).
    pub salt: u64,
    /// How many different salts (rounds) to try before giving up.
    pub rehash_limit: u32,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            target_bucket_size: 4.0,
            max_seed_attempts: 50_000,
            salt: 0xC0FF_EE00_D15E_A5E,
            rehash_limit: 6,
        }
    }
}

#[derive(Debug, Error)]
pub enum MphError {
    #[error("duplicate key detected during build")]
    DuplicateKey,
    #[error("could not place all buckets after rehash attempts")]
    Unresolvable,
    #[cfg(feature = "serde")]
    #[error("serialization error: {0}")]
    Serde(#[from] Box<bincode::ErrorKind>),
}

pub struct Builder {
    cfg: BuildConfig,
}

impl Builder {
    pub fn new() -> Self {
        Self { cfg: BuildConfig::default() }
    }

    pub fn with_config(mut self, cfg: BuildConfig) -> Self {
        self.cfg = cfg;
        self
    }

    /// Build MPH. Requires **unique** keys.
    pub fn build<K, I>(self, keys: I) -> Result<Mphf, MphError>
    where
        K: Borrow<[u8]>,
        I: IntoIterator<Item = K>,
    {
        // 0) Collect and verify uniqueness by actual bytes (without probabilistic hashes).
        let mut uniq = Vec::<Vec<u8>>::new();
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

    /// Build MPH. Assumes **unique** keys (skips duplicate checks).
    pub fn build_unique_vec(self, uniq: Vec<Vec<u8>>) -> Result<Mphf, MphError> {
        self.build_unique_ref(&uniq)
    }

    /// Build MPH. Assumes **unique** keys (skips duplicate checks).
    pub fn build_unique_ref(self, uniq: &[Vec<u8>]) -> Result<Mphf, MphError> {
        let n = uniq.len();
        assert!(n > 0, "empty key set is not supported");

        // 1) Several attempts with different salts.
        for round in 0..=self.cfg.rehash_limit {
            let salt = mix_salt(self.cfg.salt, round);
            match try_build_once(&uniq, n, salt, &self.cfg) {
                Ok(mut mph) => {
                    mph.salt = salt;
                    return Ok(mph);
                }
                Err(MphError::Unresolvable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(MphError::Unresolvable)
    }
}

/// One build attempt for a specific salt.
fn try_build_once(keys: &[Vec<u8>], n: usize, salt: u64, cfg: &BuildConfig) -> Result<Mphf, MphError> {
    let n_u64 = n as u64;

    // 1) Preliminary hashing and bucketing.
    let buckets_cnt = ( (n as f64 / cfg.target_bucket_size).ceil() as usize ).max(1);
    let mut buckets: Vec<Vec<KeyHash>> = vec![Vec::new(); buckets_cnt];
    for k in keys {
        let kh = KeyHash::from_key(k, salt);
        let b = kh.bucket(buckets_cnt as u64);
        buckets[b].push(kh);
    }

    // 2) Process buckets in descending size order. (smaller ones are easier to place later)
    let mut order: Vec<usize> = (0..buckets_cnt).collect();
    order.sort_by_key(|&b| -(buckets[b].len() as isize));

    // 3) Global occupancy and displacements.
    let mut occupied = BitSet::new(n);
    let mut disps = vec![0u64; buckets_cnt];

    // Simple PRNG for choosing next displacement.
    let mut prng = XorShift64::seeded(0x9E37_79B9_7F4A_7C15 ^ salt);

    // 4) Place buckets.
    for &b in &order {
        let items = &buckets[b];
        if items.is_empty() {
            disps[b] = 0;
            continue;
        }

        // Try displacements (including 0), order determined by PRNG (but deterministic from salt).
        let mut attempts = 0u32;
        'find_disp: loop {
            if attempts >= cfg.max_seed_attempts {
                return Err(MphError::Unresolvable);
            }
            attempts += 1;

            // Different strategies for robustness: some attempts use small displacements,
            // some use pseudo-random ones from PRNG.
            let d = if attempts <= 256 {
                (attempts as u64 - 1) // 0,1,2,...255 — cheap enumeration
            } else {
                prng.next()
            };

            // Check positions
            let mut ok = true;
            let mut positions = Vec::with_capacity(items.len());
            for kh in items {
                let p = kh.place(n_u64, d);
                if occupied.test(p) {
                    ok = false;
                    break;
                }
                positions.push(p);
            }
            if !ok {
                continue;
            }
            // Check for duplicates inside bucket
            positions.sort_unstable();
            if positions.windows(2).any(|w| w[0] == w[1]) {
                continue;
            }

            // Success - mark slots
            for p in positions {
                occupied.set(p);
            }
            disps[b] = d;
            break 'find_disp;
        }
    }

    Ok(Mphf {
        n: n_u64,
        buckets: buckets_cnt as u64,
        salt,
        disps,
    })
}

/// Minimal xorshift
struct XorShift64(u64);
impl XorShift64 {
    fn seeded(mut s: u64) -> Self {
        if s == 0 { s = 0x9E37_79B9_7F4A_7C15 }
        Self(s)
    }
    #[inline]
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Shift base salt deterministically
fn mix_salt(base: u64, round: u32) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME:  u64 = 0x100000001b3;
    let mut h = FNV_OFFSET ^ base;
    h ^= round as u64;
    h = h.wrapping_mul(FNV_PRIME);
    h ^ (h >> 33)
}
