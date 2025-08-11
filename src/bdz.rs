use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashSet;
use thiserror::Error;

/// Minimal perfect hash by BDZ (3-hypergraph peeling) with:
/// - wyhash-based vertex derivation (1×wyhash + splitmix64),
/// - CSR adjacency (offsets + flat edges),
/// - optional parallel hashing via rayon ("rayon" feature),
/// - u32 everywhere and cache-friendly data layout.
///
/// Query: f(k) = (g[v0] + g[v1] + g[v2]) % n
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Mphf {
    pub n: u64,      // number of keys
    pub m: u32,      // graph vertices (m = ceil(gamma * n))
    pub salt: u64,   // effective salt used to derive vertices
    pub g: Vec<u32>, // length == m, values in [0..n)
}

impl Mphf {
    #[inline]
    pub fn index(&self, key: &[u8]) -> u64 {
        let (a, b, c) = vertices(key, self.salt, self.m as u64);
        // Safety: a,b,c < m; g.len() == m
        let ga = unsafe { *self.g.get_unchecked(a as usize) };
        let gb = unsafe { *self.g.get_unchecked(b as usize) };
        let gc = unsafe { *self.g.get_unchecked(c as usize) };
        ((ga + gb + gc) % (self.n as u32)) as u64
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

/// Builder configuration.
/// For huge datasets (e.g., 100M keys) set `gamma ≈ 1.27` to reduce rehash retries.
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Vertex ratio m/n; BDZ classic ~1.23. For 100M keys recommend 1.27.
    pub gamma: f64,
    /// Maximum rehash attempts if graph is not peelable.
    pub rehash_limit: u32,
    /// Base salt. Effective salts derive deterministically.
    pub salt: u64,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            gamma: 1.27,                // Good default; for 100M use 1.27
            rehash_limit: 16,           // plenty; on good salts often 0-1
            salt: 0xC0FF_EE00_D15E_A5E, // arbitrary base
        }
    }
}

#[derive(Debug, Error)]
pub enum MphError {
    #[error("duplicate key detected during build")]
    DuplicateKey,
    #[error("graph was not peelable after rehash attempts")]
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
        Self {
            cfg: BuildConfig::default(),
        }
    }
    pub fn with_config(mut self, cfg: BuildConfig) -> Self {
        self.cfg = cfg;
        self
    }

    /// Build MPH from **unique** keys.
    pub fn build<K, I>(self, keys: I) -> Result<Mphf, MphError>
    where
        K: Borrow<[u8]>,
        I: IntoIterator<Item = K>,
    {
        // Collect and verify true uniqueness (no probabilistic dedup).
        let mut uniq = Vec::<Vec<u8>>::new();
        uniq.reserve(1024);
        let mut seen = HashSet::<Vec<u8>>::new();
        for k in keys {
            let v = k.borrow().to_vec();
            if !seen.insert(v.clone()) {
                return Err(MphError::DuplicateKey);
            }
            uniq.push(v);
        }
        let n = uniq.len();
        assert!(n > 0, "empty key set is not supported");

        // Rehash loop: try different effective salts until the hypergraph peels fully.
        for round in 0..=self.cfg.rehash_limit {
            let salt = mix_salt(self.cfg.salt, round);
            match try_build_bdz(&uniq, n, salt, self.cfg.gamma) {
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

/// One BDZ build attempt.
/// Steps:
/// 1) derive (v0,v1,v2) per edge,
/// 2) build CSR (deg/off/edges),
/// 3) peel (queue vertices of degree 1),
/// 4) assign g[] in reverse peel order.
fn try_build_bdz(keys: &[Vec<u8>], n: usize, salt: u64, gamma: f64) -> Result<Mphf, MphError> {
    let n_u32 = n as u32;
    let m = ((gamma * n as f64).ceil() as u32).max(1);

    // 1) derive vertices
    let (v0, v1, v2) = derive_vertices(keys, salt, m as u64);

    // 2) degrees and CSR
    let mut deg = vec![0u32; m as usize];
    for i in 0..n {
        // SAFETY: vX[i] < m by construction
        unsafe {
            *deg.get_unchecked_mut(v0[i] as usize) += 1;
            *deg.get_unchecked_mut(v1[i] as usize) += 1;
            *deg.get_unchecked_mut(v2[i] as usize) += 1;
        }
    }

    // prefix sums -> offsets
    let mut off = vec![0usize; (m as usize) + 1];
    for i in 0..m as usize {
        off[i + 1] = off[i] + deg[i] as usize;
    }
    let mut cur = off.clone();
    let mut edges = vec![0u32; off[m as usize]];

    for eid in 0..n as u32 {
        let a = v0[eid as usize] as usize;
        let b = v1[eid as usize] as usize;
        let c = v2[eid as usize] as usize;
        unsafe {
            let ia = *cur.get_unchecked(a);
            edges[ia] = eid;
            *cur.get_unchecked_mut(a) = ia + 1;

            let ib = *cur.get_unchecked(b);
            edges[ib] = eid;
            *cur.get_unchecked_mut(b) = ib + 1;

            let ic = *cur.get_unchecked(c);
            edges[ic] = eid;
            *cur.get_unchecked_mut(c) = ic + 1;
        }
    }

    // 3) peeling
    // queue of vertices with degree == 1 (as a simple array + head index)
    let mut q = Vec::with_capacity(m as usize);
    for (vid, &d) in deg.iter().enumerate() {
        if d == 1 {
            q.push(vid as u32);
        }
    }
    let mut q_head = 0usize;

    #[derive(Copy, Clone)]
    struct Peel {
        edge: u32,
        pivot: u8,
    } // pivot ∈ {0,1,2}
    let mut peel_order = Vec::<Peel>::with_capacity(n);
    let mut removed = vec![false; n]; // removed edges

    while q_head < q.len() {
        let u = q[q_head];
        q_head += 1;

        // iterate incident edges via CSR
        let (start, end) = unsafe {
            (
                *off.get_unchecked(u as usize),
                *off.get_unchecked(u as usize + 1),
            )
        };

        // Collect live incident edges
        let mut inc_buf: Vec<u32> = Vec::with_capacity(8);
        for i in start..end {
            let e = unsafe { *edges.get_unchecked(i) };
            if !unsafe { *removed.get_unchecked(e as usize) } {
                inc_buf.push(e);
            }
        }

        for e in inc_buf {
            if unsafe { *removed.get_unchecked(e as usize) } {
                continue;
            }
            let a = v0[e as usize];
            let b = v1[e as usize];
            let c = v2[e as usize];

            // pivot is the current degree-1 endpoint of this edge
            let pivot = if unsafe { *deg.get_unchecked(a as usize) } == 1 {
                0
            } else if unsafe { *deg.get_unchecked(b as usize) } == 1 {
                1
            } else if unsafe { *deg.get_unchecked(c as usize) } == 1 {
                2
            } else {
                continue;
            };

            peel_order.push(Peel { edge: e, pivot });
            unsafe {
                *removed.get_unchecked_mut(e as usize) = true;
            }

            match pivot {
                0 => {
                    dec_deg(&mut deg, b, &mut q);
                    dec_deg(&mut deg, c, &mut q);
                }
                1 => {
                    dec_deg(&mut deg, a, &mut q);
                    dec_deg(&mut deg, c, &mut q);
                }
                _ => {
                    dec_deg(&mut deg, a, &mut q);
                    dec_deg(&mut deg, b, &mut q);
                }
            }
        }
    }

    if peel_order.len() != n {
        return Err(MphError::Unresolvable);
    }

    // 4) assign g[] in reverse peel order
    let mut g = vec![u32::MAX; m as usize]; // MAX => unassigned
    for rec in peel_order.iter().rev() {
        let e = rec.edge as usize;
        // vertices of edge e
        let a = v0[e] as usize;
        let b = v1[e] as usize;
        let c = v2[e] as usize;

        // put unknown vertex first (pivot)
        let (x, y, z) = match rec.pivot {
            0 => (a, b, c),
            1 => (b, a, c),
            _ => (c, a, b),
        };
        let gy = if unsafe { *g.get_unchecked(y) } == u32::MAX {
            0
        } else {
            unsafe { *g.get_unchecked(y) }
        };
        let gz = if unsafe { *g.get_unchecked(z) } == u32::MAX {
            0
        } else {
            unsafe { *g.get_unchecked(z) }
        };
        let sum = (gy + gz) % n_u32;
        let want = ((rec.edge % n_u32) + n_u32 - sum) % n_u32;
        unsafe {
            *g.get_unchecked_mut(x) = want;
        }
    }
    for v in &mut g {
        if *v == u32::MAX {
            *v = 0;
        }
    }

    Ok(Mphf {
        n: n as u64,
        m,
        salt,
        g,
    })
}

#[inline]
fn dec_deg(deg: &mut [u32], v: u32, q: &mut Vec<u32>) {
    // SAFETY: v < deg.len()
    let d = unsafe { deg.get_unchecked_mut(v as usize) };
    if *d > 0 {
        *d -= 1;
        if *d == 1 {
            q.push(v);
        }
    }
}

/// Derive 3 vertices for each key (possibly in parallel if the "rayon" feature is enabled).
fn derive_vertices(keys: &[Vec<u8>], salt: u64, m: u64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let verts: Vec<(u32, u32, u32)> = keys.par_iter().map(|k| vertices(k, salt, m)).collect();
        let n = verts.len();
        let mut v0 = Vec::with_capacity(n);
        let mut v1 = Vec::with_capacity(n);
        let mut v2 = Vec::with_capacity(n);
        for (a, b, c) in verts {
            v0.push(a);
            v1.push(b);
            v2.push(c);
        }
        (v0, v1, v2)
    }
    #[cfg(not(feature = "rayon"))]
    {
        let n = keys.len();
        let mut v0 = Vec::with_capacity(n);
        let mut v1 = Vec::with_capacity(n);
        let mut v2 = Vec::with_capacity(n);
        for k in keys {
            let (a, b, c) = vertices(k, salt, m);
            v0.push(a);
            v1.push(b);
            v2.push(c);
        }
        (v0, v1, v2)
    }
}

/// 1× wyhash + splitmix64 → 3 independent vertex indexes
#[inline]
fn vertices(key: &[u8], salt: u64, m: u64) -> (u32, u32, u32) {
    let base = wyhash1(key, salt);
    let a = splitmix64(base ^ 0x9E37_79B9_7F4A_7C15) % m;
    let b = splitmix64(base.wrapping_add(0xA24B_1F6F)) % m;
    let c = splitmix64(base ^ 0x853C_49E6_0A6C_9D39) % m;
    (a as u32, b as u32, c as u32)
}

#[inline]
fn wyhash1(data: &[u8], seed: u64) -> u64 {
    wyhash::wyhash(data, seed)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministically tweak base salt by round (FNV-like).
#[inline]
fn mix_salt(base: u64, round: u32) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET ^ base;
    h ^= round as u64;
    h = h.wrapping_mul(FNV_PRIME);
    h ^ (h >> 33)
}
