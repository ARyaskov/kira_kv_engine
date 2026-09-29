//! PtrHash 2025 implementation: u8 pilots, 2-level bucketing, cuckoo-style eviction,
//! tail remapping for minimality, and **partitioned (sharded) construction**.
//!
//! Key design points (after Groot Koerkamp 2024 / Pibiri & Trani 2023):
//!
//! 1. **Pilots are small integers in [0, 255]**, stored as one byte per bucket. The slot
//!    for a key is computed by mixing `(h2, pilot)` and reducing to the part's slot
//!    range — pilots act as a per-bucket displacement seed.
//! 2. **λ ≈ 3 keys per bucket.** Pilot storage is therefore ~2.7 bits/key with no
//!    compression, and one byte load per lookup. (Earlier versions used 2.2 buckets
//!    per key and needed a rank/select-compressed pilot table to get below 5 bits.)
//! 3. **2-level bucket assignment.** Keys are split into a "large" zone (first
//!    `ALPHA_BUCKETS` ≈ 30% of buckets, receiving ~60% of keys) and a "small" zone
//!    (remaining ~70% of buckets, receiving ~40% of keys). This skews the bucket-size
//!    distribution so big buckets are placed early into an empty table and tiny tail
//!    buckets fill the gaps.
//! 4. **Largest-bucket-first pilot search with eviction.** Buckets are placed in
//!    descending size order. When no pilot places a bucket collision-free, the pilot
//!    with the cheapest set of colliding buckets is chosen, those buckets are evicted
//!    (their slots freed, pushed back on the work stack) and placed again later —
//!    exactly the cuckoo-hashing displacement of the PtrHash paper. This is what lets
//!    the table run at α = 0.98 load with only 256 pilot values.
//! 5. **Minimal output.** Each part hashes into `⌈m/α⌉` slots but returns positions in
//!    `[0, m)`: the ~2% of keys that land in the tail `[m, ⌈m/α⌉)` are redirected by a
//!    small per-part remap table to the free slots below `m`. Globally the index is a
//!    bijection onto `[0, n)` — side arrays are sized to `len()`, not `1.1 × len()`.
//! 6. **Partitioned build.** Keys are first split into independent *parts* of
//!    ~`PART_TARGET_KEYS` keys by a dedicated part hash. Every part owns a contiguous
//!    range of slots and buckets and is built completely independently on a working
//!    set of ~1 MB that stays in L2; parts are built in parallel with rayon.
//! 7. **Per-part rehash.** If a part exceeds its eviction budget it is retried with a
//!    different salt — only that part, not the whole index.
//!
//! Lookup is: base hash, with the part selector (xor + mul + mulhi) and the per-part
//! table load (L1-resident) computed in parallel with it → ~5-cycle per-part remix →
//! one pilot load → slot compute → (2% of the time) one remap load.
//!
//! Indexes with a single part (≤ `PART_TARGET_KEYS` keys) use the pre-partition hash
//! formulas, and files written by 0.6 (1.1× slot padding, no remap) still load.

use thiserror::Error;

/// Fraction of buckets assigned to the "large" zone (high-density region).
/// Empirically 0.30 works well across 1M..1B keys.
const ALPHA_BUCKETS: f64 = 0.30;
/// Fraction of keys hashed into the "large" zone (matches Pibiri/Trani PTHash3).
const BETA_KEYS: f64 = 0.60;
/// `BETA_KEYS` expressed as a threshold on the top 16 bits of `h1`.
const BETA_THRESHOLD: u32 = (BETA_KEYS * 65536.0) as u32;

/// Target number of keys per part. 32K keys → ~33K slots, ~11K buckets; the per-part
/// working set (h1, bucket ids, offsets, items, order, slot ownership) is well under
/// 1 MB, which fits the L2 of any current core.
pub const PART_TARGET_KEYS: usize = 1 << 15;

/// Default keys per bucket. 3.0 gives ~2.7 bits/key of pilots; the eviction search
/// converges in one salt round at this density for uniformly hashed keys.
pub const DEFAULT_LAMBDA: f64 = 3.0;
/// Default load factor (keys / slots). The 2% tail is remapped, see module docs.
pub const DEFAULT_ALPHA: f64 = 0.98;

/// Evictions allowed per key before a part gives up on its salt.
const EVICTION_BUDGET_PER_KEY: usize = 32;
/// Buckets placed most recently are protected from re-eviction (cycle breaker).
const RECENT: usize = 16;
/// Free-slot marker in the slot → bucket ownership table.
const FREE: u32 = u32::MAX;

const H2_XOR: u64 = 0xA24B_1F6F_DA39_2B31;
const ROUND_MUL: u64 = 0x9E37_79B9_7F4A_7C15;
const PART_ID_MUL: u64 = 0xD1B5_4A32_D192_ED03;
const PART_SALT_MIX: u64 = 0x5851_F42D_4C95_7F2D;
/// Odd multiplier of the part selector (golden-ratio Fibonacci hashing).
const PART_MUL: u64 = 0x9E37_79B9_7F4A_7C15;
/// Odd multiplier of the per-part remix.
const REMIX_MUL: u64 = 0xBF58_476D_1CE4_E5B9;

/// Hash mixing helper (splitmix finalizer). Used for salt derivation only.
#[inline(always)]
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline(always)]
fn fast_reduce(hash: u64, n: usize) -> usize {
    ((hash as u128 * n as u128) >> 64) as usize
}

/// Base per-key hash inside a part. `rotated` is the key after the data-driven
/// prerotation. Build and lookup must use the same `use_aes` flag.
#[inline(always)]
fn base_hash(rotated: u64, salt: u64, use_aes: bool) -> u64 {
    if use_aes {
        crate::aes_hash::hash_u64(rotated, salt)
    } else {
        crate::simd_hash::hash_u64_one(rotated, salt)
    }
}

#[inline(always)]
fn h2_from_h1(h1: u64) -> u64 {
    h1.rotate_left(23) ^ H2_XOR
}

/// Part selector: multiplicative (Fibonacci) hash → top bits. One xor + one multiply,
/// independent of the base hash, so on lookup the CPU computes it *in parallel* with
/// `base_hash` instead of adding a second full mix to the dependent chain. Good part
/// balance for both uniform inputs (canonical hashes) and structured ones (sequential
/// ids) — the top bits of `k × odd` are well spread for any arithmetic progression.
#[inline(always)]
fn part_of(rotated: u64, part_salt: u64, parts: usize) -> usize {
    fast_reduce((rotated ^ part_salt).wrapping_mul(PART_MUL), parts)
}

/// Per-part remix applied on top of the base hash in multi-part indexes:
/// `h1 = (base ^ part_salt) × odd`. Both steps are bijections, so equal `h1` still means
/// equal key; the multiply propagates every input bit into the top bits that pick the
/// zone and bucket, so a different salt yields an unrelated bucket/slot assignment
/// (that's what a per-part rehash needs). Costs ~5 cycles vs ~14 for a second `mix64`.
#[inline(always)]
fn remix(base: u64, part_salt: u64) -> u64 {
    (base ^ part_salt).wrapping_mul(REMIX_MUL)
}

/// Batch hash of already-rotated keys with a part salt. AVX2 4-wide for mix64.
#[inline]
fn hash_keys_batch(keys: &[u64], salt: u64, use_aes: bool, out: &mut [u64]) {
    if use_aes {
        for (o, &k) in out.iter_mut().zip(keys) {
            *o = crate::aes_hash::hash_u64(k, salt);
        }
    } else {
        crate::simd_hash::hash_u64(keys, salt, out);
    }
}

/// Salt for `(part, rehash round)`. Part 0 / round 0 equals the pre-partition salt so
/// single-part indexes are bit-identical with v0.6 builds.
#[inline(always)]
fn part_round_salt(seed: u64, part_id: usize, round: u32) -> u64 {
    mix64(
        seed ^ (round as u64).wrapping_mul(ROUND_MUL) ^ (part_id as u64).wrapping_mul(PART_ID_MUL),
    )
}

#[inline(always)]
fn parts_for(n: usize) -> usize {
    if n <= PART_TARGET_KEYS { 1 } else { n.div_ceil(PART_TARGET_KEYS) }
}

#[inline(always)]
fn large_buckets_of(num_buckets: usize) -> usize {
    ((num_buckets as f64) * ALPHA_BUCKETS) as usize
}

/// Slots searched for `m` keys at load factor `alpha`. At least one slot even for an
/// empty part so a foreign key selecting that part still computes a valid index.
#[inline(always)]
fn slots_for_keys(m: usize, alpha: f64) -> usize {
    (((m as f64) / alpha).ceil() as usize).max(m).max(1)
}

#[inline(always)]
fn buckets_for_keys(m: usize, lambda: f64) -> usize {
    (((m as f64) / lambda).ceil() as usize).max(1)
}

/// Bucket index inside a part, combining h1 with 2-level skew. Keys with hash in the
/// "large zone" (top BETA_KEYS fraction by high bits) map to the first ALPHA_BUCKETS
/// fraction of buckets; the rest map to the tail buckets.
///
/// Result: large zone has BETA_KEYS / ALPHA_BUCKETS ≈ 2x density vs uniform,
/// small zone has (1-BETA_KEYS) / (1-ALPHA_BUCKETS) ≈ 0.57x density.
#[inline(always)]
fn bucket_in_part(h1: u64, large_buckets: usize, small_buckets: usize) -> usize {
    let zone_decider = (h1 >> 48) as u32;
    let h_low = (h1 & 0x0000_FFFF_FFFF_FFFF) << 16;
    if zone_decider < BETA_THRESHOLD {
        fast_reduce(h_low, large_buckets.max(1))
    } else {
        large_buckets + fast_reduce(h_low, small_buckets.max(1))
    }
}

/// Bucket index for a single-partition layout with `num_buckets` buckets.
#[inline(always)]
fn bucket_for(h1: u64, num_buckets: usize) -> usize {
    let large = large_buckets_of(num_buckets);
    bucket_in_part(h1, large, num_buckets - large)
}

/// Slot computation given bucket's chosen pilot. The pilot acts as a per-bucket "salt"
/// that perturbs the slot mapping. Even though pilot is in [0, 255], multiplying by a
/// large odd constant spreads it across the full u64 range before reduction.
#[inline(always)]
fn slot_for(h2: u64, pilot: u8, n: usize) -> usize {
    let pilot_mix = (pilot as u64).wrapping_mul(0xA24B_1F6F_DA39_2B31);
    let mixed = (h2 ^ pilot_mix)
        .rotate_left(31)
        .wrapping_mul(0xD6E8_FEB8_6659_FD93);
    fast_reduce(mixed, n)
}

#[inline(always)]
fn fingerprint_u8(h2: u64) -> u8 {
    // High byte of h2 (not used for slot derivation, so independent collisions).
    (h2 >> 56) as u8
}

/// Learn the best bit-rotation for `keys` that flattens bucket-size variance.
///
/// **Why this helps**: many real-world key distributions have non-uniform bit entropy
/// (sequential IDs, time-stamped records, hash-of-hash chains). Universal hash functions
/// mix bits well *on average* but a small per-data rotation can squeeze 5-15% extra
/// flatness, which lowers the eviction count of the pilot search.
///
/// Method: sample 0.5% of keys (capped at 32K), evaluate chi-squared bucket-size
/// uniformity for each candidate rotation in 0..64, return the best. ~3 ms at 10M keys.
fn learn_prerotation(keys: &[u64], num_buckets: usize, salt: u64, use_aes: bool) -> u8 {
    if keys.len() < 2048 {
        return 0; // sample too small — skip the search, default rotation 0
    }
    let sample_size = (keys.len() / 200).clamp(2048, 32_768);
    let stride = (keys.len() / sample_size).max(1);
    let sample: Vec<u64> = (0..sample_size).map(|i| keys[i * stride]).collect();

    const HIST: usize = 256;
    let mut best_rot = 0u8;
    let mut best_score = f64::MAX;
    for rot in (0..64u8).step_by(4) {
        let mut hist = [0u32; HIST];
        for &k in &sample {
            let base = base_hash(k.rotate_left(rot as u32), salt, use_aes);
            let b = bucket_for(base, num_buckets);
            hist[b % HIST] += 1;
        }
        let expected = sample.len() as f64 / HIST as f64;
        let chi: f64 = hist
            .iter()
            .map(|&o| {
                let d = o as f64 - expected;
                d * d / expected
            })
            .sum();
        if chi < best_score {
            best_score = chi;
            best_rot = rot;
        }
    }
    best_rot
}

/// Geometry + salt of one part. 40 bytes; the whole table stays L1/L2-resident
/// (1 entry per ~32K keys).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct PartInfo {
    /// First global slot of this part (its keys occupy `[slot_off, slot_off + num_keys)`).
    pub slot_off: u64,
    /// Salt that successfully placed this part.
    pub salt: u64,
    /// First global bucket of this part.
    pub bucket_off: u32,
    /// Keys in this part = size of its output range.
    pub num_keys: u32,
    /// Slots the pilot search hashes into (`⌈num_keys / α⌉`). Local slots
    /// `≥ num_keys` are redirected through the remap table.
    pub num_slots: u32,
    /// Buckets owned by this part (`⌈num_keys / λ⌉`).
    pub num_buckets: u32,
    /// Cached `num_buckets × ALPHA_BUCKETS` (large-zone size).
    pub large_buckets: u32,
    /// Offset of this part's `num_slots - num_keys` entries in the global remap table.
    pub remap_off: u32,
}

impl PartInfo {
    #[inline]
    fn new(
        slot_off: u64,
        salt: u64,
        bucket_off: u32,
        num_keys: u32,
        num_slots: u32,
        num_buckets: u32,
        remap_off: u32,
    ) -> Self {
        Self {
            slot_off,
            salt,
            bucket_off,
            num_keys,
            num_slots,
            num_buckets,
            large_buckets: large_buckets_of(num_buckets as usize) as u32,
            remap_off,
        }
    }

    #[inline]
    fn remap_len(&self) -> usize {
        (self.num_slots - self.num_keys) as usize
    }
}

/// Final MPHF structure. Lookup = (part hash) + hash + 1 byte load + slot compute
/// (+ 1 remap load for the ~2% of keys in a part's tail).
#[derive(Debug, Clone)]
pub struct PtrHash25Mphf {
    /// Number of keys = size of the output range: `index_u64` returns values in `[0, n)`.
    pub n: u64,
    /// Total number of buckets across all parts.
    pub num_buckets: u32,
    /// Multi-part: salt of the global base hash (`h1 = remix(base_hash(key, salt),
    /// parts[p].salt)`). Single part: the hash salt itself (`parts[0].salt`, legacy).
    pub salt: u64,
    /// One pilot byte per bucket. Hugepage-backed when ≥ 1 MB.
    pub pilots: crate::hugepage::HugeVec<u8>,
    /// Tail redirection: for part `p`, local slot `s ≥ num_keys` maps to
    /// `remap[p.remap_off + s - num_keys]`, a free slot `< num_keys`.
    pub remap: Box<[u32]>,
    /// Optional fingerprint table (u8 per output position) for negative-query rejection.
    /// Empty if the index doesn't need negative-query support.
    pub fingerprints: Vec<u8>,
    /// True iff this MPHF was built with the AES hash. Lookups must use the same path.
    pub use_aes_hash: bool,
    /// Data-driven bit rotation applied to keys before hashing (0..63). Learned
    /// during build to flatten bucket-size variance for the specific key distribution.
    pub prerotate: u8,
    /// Salt of the part-selector hash. Unused when `parts.len() == 1`.
    pub part_salt: u64,
    /// Per-part geometry. Always at least one entry.
    pub parts: Box<[PartInfo]>,
}

impl PtrHash25Mphf {
    /// Size of the output range: `lookup_u64` returns values in `[0..slot_capacity())`.
    /// Equal to the number of keys for indexes built by this version; `1.1 × keys`
    /// for files written by 0.6.
    #[inline(always)]
    pub fn slot_capacity(&self) -> usize {
        self.n as usize
    }

    #[inline(always)]
    pub fn num_parts(&self) -> usize {
        self.parts.len()
    }

    /// Part index of a key.
    #[inline(always)]
    pub fn part_of(&self, key: u64) -> usize {
        if self.parts.len() > 1 {
            part_of(key.rotate_left(self.prerotate as u32), self.part_salt, self.parts.len())
        } else {
            0
        }
    }

    /// O(1) lookup. Caller must check fingerprint separately for negative-query safety.
    #[inline(always)]
    pub fn index_u64(&self, key: u64) -> u32 {
        self.slot_and_h2(key).0 as u32
    }

    /// Lookup with built-in fingerprint check. Returns Some(idx) if the key was in the
    /// build set, None if it's a foreign key that happens to collide.
    #[inline(always)]
    pub fn lookup_u64(&self, key: u64) -> Option<u32> {
        let (slot, h2) = self.slot_and_h2(key);
        let fp_expected = fingerprint_u8(h2);
        if self.fingerprints.is_empty()
            || unsafe { *self.fingerprints.get_unchecked(slot) } == fp_expected
        {
            Some(slot as u32)
        } else {
            None
        }
    }

    #[inline(always)]
    fn slot_and_h2(&self, key: u64) -> (usize, u64) {
        let rotated = key.rotate_left(self.prerotate as u32);
        let (p, h1) = if self.parts.len() > 1 {
            // Part selection (xor + mul + mulhi) and the base hash are independent
            // chains — the OoO core overlaps them; only the ~5-cycle remix is serial.
            let pi = part_of(rotated, self.part_salt, self.parts.len());
            // SAFETY: `pi < parts.len()` by construction of `fast_reduce`.
            let p = unsafe { self.parts.get_unchecked(pi) };
            let base = base_hash(rotated, self.salt, self.use_aes_hash);
            (p, remix(base, p.salt))
        } else {
            // Single part: exactly the pre-partition formula (v0.6-compatible).
            let p = unsafe { self.parts.get_unchecked(0) };
            (p, base_hash(rotated, p.salt, self.use_aes_hash))
        };
        let h2 = h2_from_h1(h1);
        let large = p.large_buckets as usize;
        let bucket = p.bucket_off as usize
            + bucket_in_part(h1, large, p.num_buckets as usize - large);
        // SAFETY: `bucket < bucket_off + num_buckets ≤ pilots.len()` — validated on
        // load, guaranteed by construction on build.
        let pilot = unsafe { *self.pilots.as_slice().get_unchecked(bucket) };
        let mut local = slot_for(h2, pilot, p.num_slots as usize);
        let num_keys = p.num_keys as usize;
        if local >= num_keys {
            // SAFETY: `local - num_keys < num_slots - num_keys = remap_len(p)` and the
            // part's remap range lies inside `remap` (validated on load).
            local = unsafe {
                *self.remap.get_unchecked(p.remap_off as usize + (local - num_keys))
            } as usize;
        }
        (p.slot_off as usize + local, h2)
    }

    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.pilots.memory_usage()
            + self.remap.len() * std::mem::size_of::<u32>()
            + self.fingerprints.len()
            + self.parts.len() * std::mem::size_of::<PartInfo>()
    }
}

#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Average keys per bucket. Higher = fewer pilots (less memory) but a harder
    /// pilot search; 3.0 (default) needs ~2.7 bits/key and converges reliably,
    /// 4.0 is the practical ceiling for u8 pilots.
    pub lambda: f64,
    /// Load factor: keys / searched slots. The `(1 - alpha)` tail is remapped so the
    /// output stays minimal. 0.98 (default) keeps the remap table at ~0.1 B/key.
    pub alpha: f64,
    /// Maximum salt-rehash rounds (per part) before declaring the keyset unbuildable.
    pub max_rehash: u32,
    /// Whether to build the per-slot fingerprint table (adds 1 byte/key, allows negative
    /// lookups). Skip for hit-only workloads.
    pub with_fingerprints: bool,
    /// Initial salt.
    pub seed: u64,
    /// If true, use the AES round hash (`aes_hash::hash_u64`) instead of mix64 for the
    /// base hash. Both build and lookup must use the same variant — this flag affects
    /// both, and it is recorded in the serialized form. The AES hash is stronger
    /// against adversarial inputs but loses the AVX2 4-wide batch speedup of mix64.
    pub use_aes_hash: bool,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            lambda: DEFAULT_LAMBDA,
            alpha: DEFAULT_ALPHA,
            max_rehash: 16,
            with_fingerprints: true,
            seed: 0xC0FF_EE00_D15E_A5E,
            use_aes_hash: false,
        }
    }
}

#[derive(Debug, Error)]
pub enum PtrHash25Error {
    #[error("could not place all keys after max rehash rounds")]
    Unresolvable,
    /// Two identical u64 keys were found in the input. Equal keys always collide in
    /// the same bucket for every pilot, so this is detected exactly (not probabilistically).
    #[error("duplicate key in input")]
    DuplicateKey,
}

pub struct Builder {
    cfg: BuildConfig,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn config(&self) -> &BuildConfig {
        &self.cfg
    }

    /// Build an MPHF over `keys` (must be unique). Partitions, then builds every part
    /// in parallel (when the `parallel` feature is on and there is more than one part).
    pub fn build(self, keys: &[u64]) -> Result<PtrHash25Mphf, PtrHash25Error> {
        assert!(!keys.is_empty(), "empty key set");
        let part = partition_keys(keys, &self.cfg);
        build_partitioned(&part, &self.cfg).map(|(mph, _)| mph)
    }

    /// Like [`Builder::build`] but also returns the slot of every key, in the order of
    /// the returned [`Partitioned`] key array.
    pub fn build_with_slots(
        self,
        keys: &[u64],
    ) -> Result<(PtrHash25Mphf, Partitioned, Vec<u32>), PtrHash25Error> {
        assert!(!keys.is_empty(), "empty key set");
        let part = partition_keys(keys, &self.cfg);
        let (mph, slots) = build_partitioned(&part, &self.cfg)?;
        Ok((mph, part, slots))
    }
}

/// Keys grouped by part, ready for [`build_partitioned`].
///
/// `keys` holds the **prerotated** keys (see [`Partitioned::prerotate`]) ordered by
/// part; part `p` occupies `keys[part_offsets[p]..part_offsets[p+1]]`. The original
/// key of entry `i` is `keys[i].rotate_right(prerotate)`.
#[derive(Debug, Clone)]
pub struct Partitioned {
    /// Hugepage-backed when available (≥ 1 MB): 800 MB at 100M keys.
    pub keys: crate::hugepage::HugeVec<u64>,
    pub part_offsets: Vec<usize>,
    pub part_salt: u64,
    pub prerotate: u8,
}

impl Partitioned {
    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[inline]
    pub fn num_parts(&self) -> usize {
        self.part_offsets.len() - 1
    }

    #[inline]
    pub fn part_keys(&self, p: usize) -> &[u64] {
        &self.keys.as_slice()[self.part_offsets[p]..self.part_offsets[p + 1]]
    }

    /// Original (un-rotated) key at position `i`.
    #[inline(always)]
    pub fn original_key(&self, i: usize) -> u64 {
        self.keys.as_slice()[i].rotate_right(self.prerotate as u32)
    }
}

/// Raw pointer that can be shared across rayon workers. Every worker writes a disjoint
/// set of positions, which is what makes the scatter sound.
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// Accessor on purpose: closures must capture the whole `SyncPtr` (which is
    /// `Sync`), not the raw `*mut T` field that precise closure capture would
    /// otherwise pick when the field is named directly.
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

fn partition_ranges(n: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        let threads = rayon::current_num_threads().max(1);
        // ≥ 64K keys per range so the per-range histograms amortize.
        (threads * 2).min(n / 65_536).max(1)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = n;
        1
    }
}

/// Split `keys` into parts (see module docs). Learns the prerotation, rotates every key
/// and groups the rotated keys by part with a two-pass radix partition.
pub fn partition_keys(keys: &[u64], cfg: &BuildConfig) -> Partitioned {
    let out = crate::hugepage::HugeVec::<u64>::zeroed(keys.len());
    partition_keys_into(keys, cfg, out)
}

/// [`partition_keys`] writing into a caller-provided buffer of exactly `keys.len()`
/// elements (its previous contents are irrelevant). Lets the caller recycle a buffer it
/// already faulted in instead of allocating a fresh 8 B/key array.
pub fn partition_keys_into(
    keys: &[u64],
    cfg: &BuildConfig,
    mut out: crate::hugepage::HugeVec<u64>,
) -> Partitioned {
    let n = keys.len();
    assert_eq!(out.len(), n, "output buffer must have one slot per key");
    let parts = parts_for(n);
    let total_buckets = buckets_for_keys(n, cfg.lambda);
    let round0_salt = part_round_salt(cfg.seed, 0, 0);
    let prerotate = learn_prerotation(keys, total_buckets, round0_salt, cfg.use_aes_hash);
    let part_salt = mix64(cfg.seed ^ PART_SALT_MIX);
    let rot = prerotate as u32;

    if parts == 1 {
        for (o, &k) in out.as_mut_slice().iter_mut().zip(keys) {
            *o = k.rotate_left(rot);
        }
        return Partitioned {
            keys: out,
            part_offsets: vec![0, n],
            part_salt,
            prerotate,
        };
    }

    let ranges = partition_ranges(n);
    let range_len = n.div_ceil(ranges);
    let mut hist = vec![0usize; ranges * parts];

    // Pass 1: per-range histogram of part ids.
    let count_range = |r: usize, row: &mut [usize]| {
        let lo = r * range_len;
        let hi = (lo + range_len).min(n);
        for &k in &keys[lo..hi] {
            row[part_of(k.rotate_left(rot), part_salt, parts)] += 1;
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        hist.par_chunks_mut(parts)
            .enumerate()
            .for_each(|(r, row)| count_range(r, row));
    }
    #[cfg(not(feature = "parallel"))]
    {
        hist.chunks_mut(parts)
            .enumerate()
            .for_each(|(r, row)| count_range(r, row));
    }

    // Prefix sums: part offsets, then per-range write cursors (in place of `hist`).
    let mut part_offsets = vec![0usize; parts + 1];
    for p in 0..parts {
        let mut s = 0usize;
        for r in 0..ranges {
            s += hist[r * parts + p];
        }
        part_offsets[p + 1] = part_offsets[p] + s;
    }
    for p in 0..parts {
        let mut running = part_offsets[p];
        for r in 0..ranges {
            let c = hist[r * parts + p];
            hist[r * parts + p] = running;
            running += c;
        }
    }

    // Pass 2: scatter rotated keys to their part ranges. First-touch the output
    // sequentially so the parallel scatter doesn't fault pages from eight threads.
    crate::hugepage::prefault(out.as_mut_slice());
    let out_ptr = SyncPtr(out.as_mut_slice().as_mut_ptr());
    // Software write-combining: every worker stages keys in one 64-byte line per part
    // and flushes full lines. With thousands of parts (100M keys → 3052) a naive scatter
    // keeps thousands of write streams live per thread and misses the STLB on nearly
    // every store (measured 18× the 10M cost for 10× the keys); flushing whole lines cuts
    // the random stores — and the page walks — by 8×.
    const WC: usize = 8;
    let scatter_range = |r: usize, cursors: &mut [usize]| {
        let lo = r * range_len;
        let hi = (lo + range_len).min(n);
        let mut wc = vec![0u64; parts * WC];
        let mut fill = vec![0u8; parts];
        let out = out_ptr.get();
        for &k in &keys[lo..hi] {
            let k = k.rotate_left(rot);
            let p = part_of(k, part_salt, parts);
            let f = fill[p] as usize;
            wc[p * WC + f] = k;
            if f + 1 == WC {
                let pos = cursors[p];
                cursors[p] = pos + WC;
                fill[p] = 0;
                // SAFETY: `[pos, pos + WC)` is a unique range in `out` (disjoint cursors).
                unsafe { std::ptr::copy_nonoverlapping(wc.as_ptr().add(p * WC), out.add(pos), WC) };
            } else {
                fill[p] = (f + 1) as u8;
            }
        }
        for p in 0..parts {
            let f = fill[p] as usize;
            if f > 0 {
                let pos = cursors[p];
                cursors[p] = pos + f;
                // SAFETY: as above, partial line.
                unsafe { std::ptr::copy_nonoverlapping(wc.as_ptr().add(p * WC), out.add(pos), f) };
            }
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        hist.par_chunks_mut(parts)
            .enumerate()
            .for_each(|(r, row)| scatter_range(r, row));
    }
    #[cfg(not(feature = "parallel"))]
    {
        hist.chunks_mut(parts)
            .enumerate()
            .for_each(|(r, row)| scatter_range(r, row));
    }

    Partitioned {
        keys: out,
        part_offsets,
        part_salt,
        prerotate,
    }
}

/// Per-thread scratch buffers for one part build. Grown on demand and reused across
/// parts so the hot loop never allocates.
#[derive(Default)]
struct Scratch {
    h1: Vec<u64>,
    bidx: Vec<u32>,
    offsets: Vec<u32>,
    cursor: Vec<u32>,
    items_h2: Vec<u64>,
    order: Vec<u32>,
    /// Owning bucket per slot, `FREE` when empty.
    slot_bucket: Vec<u32>,
    /// Buckets still to be placed (largest first on top initially).
    stack: Vec<u32>,
    trial: Vec<u32>,
    freq: Vec<u32>,
    start: Vec<u32>,
}

#[inline]
fn grow<T: Clone + Default>(v: &mut Vec<T>, len: usize) {
    if v.len() < len {
        v.resize(len, T::default());
    }
}

/// One part's inputs and output views. Output slices are disjoint sub-slices of the
/// global pilot / remap / slot / fingerprint arrays.
struct PartJob<'a> {
    part_id: usize,
    keys: &'a [u64],
    info: PartInfo,
    /// `Some(global_salt)` for multi-part indexes (base hash + per-part remix),
    /// `None` for the single-part legacy formula.
    global_salt: Option<u64>,
    prerotate: u8,
    pilots: &'a mut [u8],
    /// `num_slots - num_keys` entries: free slot below `num_keys` for every tail slot.
    remap: &'a mut [u32],
    /// Local output position per key (part-key order), if the caller wants them.
    slots: Option<&'a mut [u32]>,
    /// Inner u8 fingerprints (`fingerprint_u8(h2)`) per local position.
    fp8: Option<&'a mut [u8]>,
    /// Outer u16 fingerprints (`original_key & 0xFFFF`) per local position — written
    /// here so `Index` needs neither a slot array nor a second pass.
    fp16: Option<&'a mut [u16]>,
}

/// Which side tables [`build_partitioned_with`] should fill.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuildOutputs {
    /// Return the local slot of every key (in `Partitioned::keys` order).
    pub slots: bool,
    /// Return a `u16` fingerprint table over the global output range, keyed by the low
    /// 16 bits of the original (un-rotated) key.
    pub fp16: bool,
}

/// Tiny xorshift for tie-breaking in the eviction search — deterministic per salt.
#[inline(always)]
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Try to place `bucket` with `pilot` into free slots only. On success the slots are
/// recorded in `trial[..len]` and `true` is returned; nothing is written to the table.
#[inline(always)]
fn place_free(bucket: &[u64], pilot: u8, num_slots: usize, slot_bucket: &[u32], trial: &mut [u32]) -> bool {
    for (i, &h2) in bucket.iter().enumerate() {
        let slot = slot_for(h2, pilot, num_slots);
        if slot_bucket[slot] != FREE {
            return false;
        }
        for &t in &trial[..i] {
            if t as usize == slot {
                return false;
            }
        }
        trial[i] = slot as u32;
    }
    true
}

/// Cost of placing `bucket` with `pilot` when occupied slots may be evicted: the
/// summed size of the colliding buckets, or `None` if the pilot self-collides or would
/// evict a protected (recent) bucket. Stops early once `cost ≥ limit`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn eviction_cost(
    bucket: &[u64],
    pilot: u8,
    num_slots: usize,
    slot_bucket: &[u32],
    offsets: &[u32],
    recent: &[u32; RECENT],
    limit: usize,
    trial: &mut [u32],
) -> Option<usize> {
    let mut cost = 0usize;
    for (i, &h2) in bucket.iter().enumerate() {
        let slot = slot_for(h2, pilot, num_slots);
        for &t in &trial[..i] {
            if t as usize == slot {
                return None;
            }
        }
        trial[i] = slot as u32;
        let owner = slot_bucket[slot];
        if owner != FREE {
            if recent.contains(&owner) {
                return None;
            }
            let ob = owner as usize;
            cost += (offsets[ob + 1] - offsets[ob]) as usize;
            if cost >= limit {
                return None;
            }
        }
    }
    Some(cost)
}

/// Build one part. On success returns the salt that placed it and fills `job.pilots`,
/// `job.remap`, `job.slots` (local position per key), `job.fp8` and `job.fp16` (if present).
fn build_part(job: &mut PartJob<'_>, sc: &mut Scratch, cfg: &BuildConfig) -> Result<u64, PtrHash25Error> {
    let keys = job.keys;
    let m = keys.len();
    let num_slots = job.info.num_slots as usize;
    let num_buckets = job.info.num_buckets as usize;
    let large = job.info.large_buckets as usize;
    let small = num_buckets - large;
    if m == 0 {
        // Nothing to place; pilots stay 0 and the tail (every slot) maps to position 0
        // of the part's (empty) range — see `build_partitioned_with` for its offset.
        job.remap.fill(0);
        return Ok(part_round_salt(cfg.seed, job.part_id, 0));
    }

    grow(&mut sc.h1, m);
    grow(&mut sc.bidx, m);
    grow(&mut sc.items_h2, m);
    grow(&mut sc.offsets, num_buckets + 1);
    grow(&mut sc.cursor, num_buckets + 1);
    grow(&mut sc.order, num_buckets);
    grow(&mut sc.slot_bucket, num_slots);
    let Scratch {
        h1,
        bidx,
        offsets,
        cursor,
        items_h2,
        order,
        slot_bucket,
        stack,
        trial,
        freq,
        start,
    } = sc;
    let h1 = &mut h1[..m];
    let bidx = &mut bidx[..m];
    let items_h2 = &mut items_h2[..m];
    let offsets = &mut offsets[..num_buckets + 1];
    let cursor = &mut cursor[..num_buckets + 1];
    let order = &mut order[..num_buckets];
    let slot_bucket = &mut slot_bucket[..num_slots];
    let eviction_budget = m * EVICTION_BUDGET_PER_KEY + 1024;

    for round in 0..=cfg.max_rehash {
        let salt = part_round_salt(cfg.seed, job.part_id, round);

        // Step 1: hash all keys (AVX2 4-wide), assign buckets, count bucket sizes.
        match job.global_salt {
            // Multi-part: base hash with the global salt, then the per-part remix.
            Some(global_salt) => {
                hash_keys_batch(keys, global_salt, cfg.use_aes_hash, h1);
                for x in h1.iter_mut() {
                    *x = remix(*x, salt);
                }
            }
            // Single part: legacy formula, the part salt is the hash salt.
            None => hash_keys_batch(keys, salt, cfg.use_aes_hash, h1),
        }
        offsets.fill(0);
        let mut max_size = 0u32;
        for i in 0..m {
            let b = bucket_in_part(h1[i], large, small);
            bidx[i] = b as u32;
            let c = offsets[b + 1] + 1;
            offsets[b + 1] = c;
            max_size = max_size.max(c);
        }
        let max_size = max_size as usize;

        // Step 2: exclusive prefix sum → bucket offsets; scatter h2 into bucket order.
        for b in 1..=num_buckets {
            offsets[b] += offsets[b - 1];
        }
        cursor.copy_from_slice(offsets);
        for i in 0..m {
            let b = bidx[i] as usize;
            let pos = cursor[b] as usize;
            cursor[b] += 1;
            items_h2[pos] = h2_from_h1(h1[i]);
        }

        // Step 3: bucket order by descending size (stable counting sort across size classes,
        // so buckets of equal size are visited in ascending index order → near-sequential
        // reads of `offsets` / `items_h2`).
        grow(freq, max_size + 1);
        grow(start, max_size + 1);
        let freq = &mut freq[..max_size + 1];
        let start = &mut start[..max_size + 1];
        freq.fill(0);
        for b in 0..num_buckets {
            freq[(offsets[b + 1] - offsets[b]) as usize] += 1;
        }
        let mut acc = 0u32;
        for size in (0..=max_size).rev() {
            start[size] = acc;
            acc += freq[size];
        }
        for b in 0..num_buckets {
            let size = (offsets[b + 1] - offsets[b]) as usize;
            let pos = start[size] as usize;
            start[size] += 1;
            order[pos] = b as u32;
        }

        // Duplicate check: equal h2 ⇔ equal key (h1 is a bijection of the key for a
        // fixed salt); such a pair can never be separated by any pilot. Report exactly.
        // Done once per bucket here so re-placements after eviction don't repeat it.
        let non_empty = freq[1..].iter().sum::<u32>() as usize;
        for &b in &order[..non_empty] {
            let b = b as usize;
            let bucket = &items_h2[offsets[b] as usize..offsets[b + 1] as usize];
            for i in 1..bucket.len() {
                for j in 0..i {
                    if bucket[i] == bucket[j] {
                        return Err(PtrHash25Error::DuplicateKey);
                    }
                }
            }
        }

        // Step 4: pilot search, largest buckets first, with eviction.
        slot_bucket.fill(FREE);
        grow(trial, max_size.max(1));
        let trial = &mut trial[..max_size.max(1)];
        job.pilots.fill(0);
        stack.clear();
        stack.extend(order[..non_empty].iter().rev());
        let mut recent = [FREE; RECENT];
        let mut recent_pos = 0usize;
        let mut rng = salt | 1;
        let mut evictions = 0usize;
        let mut failed = false;
        while let Some(b) = stack.pop() {
            let b = b as usize;
            let s = offsets[b] as usize;
            let e = offsets[b + 1] as usize;
            let len = e - s;
            let bucket = &items_h2[s..e];

            // 4a. Collision-free pilot, lowest first.
            let mut placed = false;
            for pilot in 0..=255u8 {
                if place_free(bucket, pilot, num_slots, slot_bucket, trial) {
                    for &t in &trial[..len] {
                        slot_bucket[t as usize] = b as u32;
                    }
                    job.pilots[b] = pilot;
                    placed = true;
                    break;
                }
            }
            if placed {
                continue;
            }

            // 4b. Cheapest eviction. Random start pilot breaks ties differently for
            // buckets that keep colliding; recently placed buckets are protected so a
            // pair cannot evict each other forever.
            let start_pilot = (xorshift(&mut rng) >> 56) as u8;
            let mut best: Option<(usize, u8)> = None;
            for pass in 0..2 {
                let protect = if pass == 0 { recent } else { [FREE; RECENT] };
                for i in 0..=255u8 {
                    let pilot = start_pilot.wrapping_add(i);
                    let limit = best.map_or(usize::MAX, |(c, _)| c);
                    if let Some(cost) = eviction_cost(
                        bucket, pilot, num_slots, slot_bucket, offsets, &protect, limit, trial,
                    ) {
                        best = Some((cost, pilot));
                        if cost == 0 {
                            break;
                        }
                    }
                }
                if best.is_some() {
                    break;
                }
            }
            let Some((_, pilot)) = best else {
                // Every pilot self-collides: not reachable for distinct keys with
                // len ≤ 255, but a fresh salt is the only sensible reaction.
                failed = true;
                break;
            };
            // Evict the owners of the slots this bucket takes, then take them.
            for (i, &h2) in bucket.iter().enumerate() {
                let slot = slot_for(h2, pilot, num_slots);
                trial[i] = slot as u32;
                let owner = slot_bucket[slot];
                if owner != FREE {
                    let ob = owner as usize;
                    let opilot = job.pilots[ob];
                    for &oh2 in &items_h2[offsets[ob] as usize..offsets[ob + 1] as usize] {
                        slot_bucket[slot_for(oh2, opilot, num_slots)] = FREE;
                    }
                    stack.push(owner);
                    evictions += 1;
                }
            }
            for &t in &trial[..len] {
                slot_bucket[t as usize] = b as u32;
            }
            job.pilots[b] = pilot;
            recent[recent_pos] = b as u32;
            recent_pos = (recent_pos + 1) % RECENT;
            if evictions > eviction_budget {
                failed = true;
                break;
            }
        }
        if failed {
            continue;
        }

        // Step 5: tail remap. Occupied slots ≥ m take the free slots < m in order, so
        // the final positions are a bijection onto [0, m).
        let mut next_free = 0usize;
        for tail in m..num_slots {
            if slot_bucket[tail] != FREE {
                while slot_bucket[next_free] != FREE {
                    next_free += 1;
                }
                job.remap[tail - m] = next_free as u32;
                next_free += 1;
            } else {
                job.remap[tail - m] = 0;
            }
        }

        // Step 6: side tables in key order — local positions, inner u8 fingerprints
        // and/or outer u16 fingerprints. Each is a separate tight loop so the common
        // case (exactly one table) stays branch-free.
        let pilots: &[u8] = job.pilots;
        let remap: &[u32] = job.remap;
        let pos_of = |i: usize| -> (usize, u64) {
            let h2 = h2_from_h1(h1[i]);
            let mut local = slot_for(h2, pilots[bidx[i] as usize], num_slots);
            if local >= m {
                local = remap[local - m] as usize;
            }
            (local, h2)
        };
        if let Some(sl) = job.slots.as_deref_mut() {
            for (i, s) in sl.iter_mut().enumerate() {
                *s = pos_of(i).0 as u32;
            }
        }
        if let Some(fp) = job.fp8.as_deref_mut() {
            for i in 0..m {
                let (slot, h2) = pos_of(i);
                fp[slot] = fingerprint_u8(h2);
            }
        }
        if let Some(fp) = job.fp16.as_deref_mut() {
            let rot = job.prerotate as u32;
            for i in 0..m {
                let slot = pos_of(i).0;
                fp[slot] = (keys[i].rotate_right(rot) & 0xFFFF) as u16;
            }
        }
        return Ok(salt);
    }
    Err(PtrHash25Error::Unresolvable)
}

fn run_jobs(jobs: Vec<PartJob<'_>>, cfg: &BuildConfig) -> Result<Vec<u64>, PtrHash25Error> {
    if jobs.len() == 1 {
        // Single part: stay on the caller's thread (also keeps nested builds inside
        // other rayon jobs — e.g. HybridIndex mini-MPHs — free of scheduling overhead).
        let mut jobs = jobs;
        let mut sc = Scratch::default();
        return Ok(vec![build_part(&mut jobs[0], &mut sc, cfg)?]);
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        jobs.into_par_iter()
            .map_init(Scratch::default, |sc, mut job| build_part(&mut job, sc, cfg))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut jobs = jobs;
        let mut sc = Scratch::default();
        jobs.iter_mut().map(|job| build_part(job, &mut sc, cfg)).collect()
    }
}

/// Build the MPHF from partitioned keys. Returns the structure plus the local position
/// of every key in `part.keys` order.
pub fn build_partitioned(
    part: &Partitioned,
    cfg: &BuildConfig,
) -> Result<(PtrHash25Mphf, Vec<u32>), PtrHash25Error> {
    let outputs = BuildOutputs { slots: true, fp16: false };
    let (mph, slots, _) = build_partitioned_with(part, cfg, outputs)?;
    Ok((mph, slots.expect("slots requested")))
}

/// Build the MPHF from partitioned keys, filling only the side tables requested in
/// `outputs`. Every table is written per part, in parallel, from the positions the
/// pilot search already knows — no second lookup pass and no intermediate slot array
/// unless the caller asks for one.
pub fn build_partitioned_with(
    part: &Partitioned,
    cfg: &BuildConfig,
    outputs: BuildOutputs,
) -> Result<(PtrHash25Mphf, Option<Vec<u32>>, Option<Vec<u16>>), PtrHash25Error> {
    let n = part.len();
    assert!(n > 0, "empty key set");
    assert!(n <= u32::MAX as usize, "PtrHash25 addresses at most 2^32 - 1 keys");
    assert!(cfg.lambda > 0.0 && cfg.alpha > 0.0 && cfg.alpha <= 1.0, "invalid lambda/alpha");
    let parts = part.num_parts();

    // Per-part geometry from the actual part sizes.
    let mut infos: Vec<PartInfo> = Vec::with_capacity(parts);
    let mut slot_off = 0u64;
    let mut bucket_off = 0u64;
    let mut remap_off = 0u64;
    for p in 0..parts {
        let m = part.part_offsets[p + 1] - part.part_offsets[p];
        let slots = slots_for_keys(m, cfg.alpha);
        let buckets = buckets_for_keys(m, cfg.lambda);
        // An empty part owns no output positions; point its (foreign-key-only) range at
        // position 0 so a lookup landing there stays inside [0, n).
        let off = if m == 0 { 0 } else { slot_off };
        infos.push(PartInfo::new(
            off,
            0,
            bucket_off as u32,
            m as u32,
            slots as u32,
            buckets as u32,
            remap_off as u32,
        ));
        slot_off += m as u64;
        bucket_off += buckets as u64;
        remap_off += (slots - m) as u64;
        assert!(bucket_off <= u32::MAX as u64, "too many buckets for u32 addressing");
        assert!(remap_off <= u32::MAX as u64, "too many remap entries for u32 addressing");
    }
    let total_buckets = bucket_off as usize;
    let total_remap = remap_off as usize;

    let mut pilots_buf = crate::hugepage::HugeVec::<u8>::zeroed(total_buckets);
    let mut remap = vec![0u32; total_remap];
    let mut slots = if outputs.slots { vec![0u32; n] } else { Vec::new() };
    let mut fp8 = if cfg.with_fingerprints { vec![0u8; n] } else { Vec::new() };
    let mut fp16 = if outputs.fp16 { vec![0u16; n] } else { Vec::new() };
    // The per-part jobs write these at random offsets from every worker; first-touch
    // them sequentially instead (see `hugepage::prefault`).
    crate::hugepage::prefault(pilots_buf.as_mut_slice());
    crate::hugepage::prefault(&mut slots);
    crate::hugepage::prefault(&mut fp8);
    crate::hugepage::prefault(&mut fp16);
    let pilots: &mut [u8] = pilots_buf.as_mut_slice();
    // Multi-part: one global base-hash salt, per-part remix salts. Single part: the
    // part salt *is* the hash salt (legacy formula).
    let global_salt = if parts > 1 { Some(part_round_salt(cfg.seed, 0, 0)) } else { None };

    // Carve disjoint output views per part.
    let mut jobs = Vec::with_capacity(parts);
    {
        let mut pil_rest: &mut [u8] = pilots;
        let mut rm_rest: &mut [u32] = &mut remap;
        let mut sl_rest: &mut [u32] = &mut slots;
        let mut fp8_rest: &mut [u8] = &mut fp8;
        let mut fp16_rest: &mut [u16] = &mut fp16;
        for (p, info) in infos.iter().enumerate() {
            let m = info.num_keys as usize;
            let (pil, r) = std::mem::take(&mut pil_rest).split_at_mut(info.num_buckets as usize);
            pil_rest = r;
            let (rm, r) = std::mem::take(&mut rm_rest).split_at_mut(info.remap_len());
            rm_rest = r;
            let sl = outputs.slots.then(|| {
                let (s, r) = std::mem::take(&mut sl_rest).split_at_mut(m);
                sl_rest = r;
                s
            });
            let f8 = cfg.with_fingerprints.then(|| {
                let (f, r) = std::mem::take(&mut fp8_rest).split_at_mut(m);
                fp8_rest = r;
                f
            });
            let f16 = outputs.fp16.then(|| {
                let (f, r) = std::mem::take(&mut fp16_rest).split_at_mut(m);
                fp16_rest = r;
                f
            });
            jobs.push(PartJob {
                part_id: p,
                keys: part.part_keys(p),
                info: *info,
                global_salt,
                prerotate: part.prerotate,
                pilots: pil,
                remap: rm,
                slots: sl,
                fp8: f8,
                fp16: f16,
            });
        }
    }

    let salts = run_jobs(jobs, cfg)?;
    for (info, salt) in infos.iter_mut().zip(salts) {
        info.salt = salt;
    }

    Ok((
        PtrHash25Mphf {
            n: n as u64,
            num_buckets: total_buckets as u32,
            salt: global_salt.unwrap_or(infos[0].salt),
            pilots: pilots_buf,
            remap: remap.into_boxed_slice(),
            fingerprints: fp8,
            use_aes_hash: cfg.use_aes_hash,
            prerotate: part.prerotate,
            part_salt: part.part_salt,
            parts: infos.into_boxed_slice(),
        },
        outputs.slots.then_some(slots),
        outputs.fp16.then_some(fp16),
    ))
}

/// Set in the prerotate byte of files written by this version: a full per-part
/// geometry table (with `num_keys` and remap offsets) and the remap table follow.
const FLAG_V3_GEOMETRY: u8 = 0x40;
/// Set by 0.6 for multi-part files: the legacy per-part table follows.
const FLAG_LEGACY_MULTI: u8 = 0x80;

/// Wire-format writer for index serialization.
///
/// Layout: `[n u64][num_buckets u32][salt u64][plen u64][pilots][flen u64][fps]
/// [use_aes u8][flags|prerotate u8]`, then with `FLAG_V3_GEOMETRY`:
/// `[part_salt u64][parts u32]{[slot_off u64][salt u64][bucket_off u32][num_keys u32]
/// [num_slots u32][num_buckets u32][remap_off u32]}*[remap_len u64][remap u32]*`.
/// Files written by 0.6 carry either no table (single part, 1.1× padded slots) or the
/// `FLAG_LEGACY_MULTI` table without `num_keys`/remap; both still load.
pub fn write_ptrhash25(mph: &PtrHash25Mphf, out: &mut Vec<u8>) {
    out.extend_from_slice(&mph.n.to_le_bytes());
    out.extend_from_slice(&mph.num_buckets.to_le_bytes());
    out.extend_from_slice(&mph.salt.to_le_bytes());
    let flat = mph.pilots.as_slice();
    out.extend_from_slice(&(flat.len() as u64).to_le_bytes());
    out.extend_from_slice(flat);
    out.extend_from_slice(&(mph.fingerprints.len() as u64).to_le_bytes());
    out.extend_from_slice(&mph.fingerprints);
    out.push(if mph.use_aes_hash { 1 } else { 0 });
    out.push((mph.prerotate & 0x3F) | FLAG_V3_GEOMETRY);
    out.extend_from_slice(&mph.part_salt.to_le_bytes());
    out.extend_from_slice(&(mph.parts.len() as u32).to_le_bytes());
    for p in mph.parts.iter() {
        out.extend_from_slice(&p.slot_off.to_le_bytes());
        out.extend_from_slice(&p.salt.to_le_bytes());
        out.extend_from_slice(&p.bucket_off.to_le_bytes());
        out.extend_from_slice(&p.num_keys.to_le_bytes());
        out.extend_from_slice(&p.num_slots.to_le_bytes());
        out.extend_from_slice(&p.num_buckets.to_le_bytes());
        out.extend_from_slice(&p.remap_off.to_le_bytes());
    }
    out.extend_from_slice(&(mph.remap.len() as u64).to_le_bytes());
    for &r in mph.remap.iter() {
        out.extend_from_slice(&r.to_le_bytes());
    }
}

pub fn read_ptrhash25(buf: &[u8], pos: &mut usize) -> Option<PtrHash25Mphf> {
    fn rd_u32(buf: &[u8], pos: &mut usize) -> Option<u32> {
        if *pos + 4 > buf.len() {
            return None;
        }
        let mut a = [0u8; 4];
        a.copy_from_slice(&buf[*pos..*pos + 4]);
        *pos += 4;
        Some(u32::from_le_bytes(a))
    }
    fn rd_u64(buf: &[u8], pos: &mut usize) -> Option<u64> {
        if *pos + 8 > buf.len() {
            return None;
        }
        let mut a = [0u8; 8];
        a.copy_from_slice(&buf[*pos..*pos + 8]);
        *pos += 8;
        Some(u64::from_le_bytes(a))
    }
    let n = rd_u64(buf, pos)?;
    let num_buckets = rd_u32(buf, pos)?;
    let salt = rd_u64(buf, pos)?;
    let plen = rd_u64(buf, pos)? as usize;
    if *pos + plen > buf.len() {
        return None;
    }
    let pilots = crate::hugepage::HugeVec::from_slice(&buf[*pos..*pos + plen]);
    *pos += plen;
    let flen = rd_u64(buf, pos)? as usize;
    if *pos + flen > buf.len() {
        return None;
    }
    let fingerprints = buf[*pos..*pos + flen].to_vec();
    *pos += flen;
    // Trailing flags (v0.4+/v0.5). Older indexes lack them → default to 0.
    let use_aes_hash = if *pos < buf.len() {
        let v = buf[*pos] != 0;
        *pos += 1;
        v
    } else {
        false
    };
    let rot_byte = if *pos < buf.len() {
        let v = buf[*pos];
        *pos += 1;
        v
    } else {
        0
    };
    let prerotate = rot_byte & 0x3F;
    let (part_salt, parts, remap) = if rot_byte & FLAG_V3_GEOMETRY != 0 {
        let part_salt = rd_u64(buf, pos)?;
        let count = rd_u32(buf, pos)? as usize;
        if count == 0 {
            return None;
        }
        let mut parts = Vec::with_capacity(count.min(buf.len() / 40));
        for _ in 0..count {
            let slot_off = rd_u64(buf, pos)?;
            let psalt = rd_u64(buf, pos)?;
            let bucket_off = rd_u32(buf, pos)?;
            let num_keys = rd_u32(buf, pos)?;
            let num_slots = rd_u32(buf, pos)?;
            let nb = rd_u32(buf, pos)?;
            let remap_off = rd_u32(buf, pos)?;
            if num_slots < num_keys {
                return None;
            }
            parts.push(PartInfo::new(slot_off, psalt, bucket_off, num_keys, num_slots, nb, remap_off));
        }
        let remap_len = rd_u64(buf, pos)? as usize;
        if remap_len.checked_mul(4)? > buf.len() - *pos {
            return None;
        }
        let mut remap = Vec::with_capacity(remap_len);
        for _ in 0..remap_len {
            remap.push(rd_u32(buf, pos)?);
        }
        (part_salt, parts, remap)
    } else if rot_byte & FLAG_LEGACY_MULTI != 0 {
        let part_salt = rd_u64(buf, pos)?;
        let count = rd_u32(buf, pos)? as usize;
        if count == 0 {
            return None;
        }
        let mut parts = Vec::with_capacity(count.min(buf.len() / 28));
        for _ in 0..count {
            let slot_off = rd_u64(buf, pos)?;
            let psalt = rd_u64(buf, pos)?;
            let bucket_off = rd_u32(buf, pos)?;
            let num_slots = rd_u32(buf, pos)?;
            let nb = rd_u32(buf, pos)?;
            // Legacy geometry: no tail, every searched slot is an output position.
            parts.push(PartInfo::new(slot_off, psalt, bucket_off, num_slots, num_slots, nb, 0));
        }
        (part_salt, parts, Vec::new())
    } else {
        let n32 = u32::try_from(n).ok()?;
        (0, vec![PartInfo::new(0, salt, 0, n32, n32, num_buckets, 0)], Vec::new())
    };
    let mph = PtrHash25Mphf {
        n,
        num_buckets,
        salt,
        pilots,
        remap: remap.into_boxed_slice(),
        fingerprints,
        use_aes_hash,
        prerotate,
        part_salt,
        parts: parts.into_boxed_slice(),
    };
    mph.validate().then_some(mph)
}

impl PtrHash25Mphf {
    /// Check every invariant the unchecked lookup path relies on. Called on every
    /// deserialization so a corrupt or hostile file can only be rejected, never read
    /// out of bounds.
    pub fn validate(&self) -> bool {
        let n = self.n;
        let pilots = self.pilots.len() as u64;
        if n == 0 || n > u32::MAX as u64 || self.parts.is_empty() {
            return false;
        }
        if self.num_buckets as u64 != pilots {
            return false;
        }
        if !self.fingerprints.is_empty() && self.fingerprints.len() as u64 != n {
            return false;
        }
        if self.prerotate >= 64 {
            return false;
        }
        let remap_len = self.remap.len() as u64;
        for p in self.parts.iter() {
            if p.num_slots < p.num_keys || p.num_slots == 0 || p.num_buckets == 0 {
                return false;
            }
            if p.large_buckets > p.num_buckets {
                return false;
            }
            if p.bucket_off as u64 + p.num_buckets as u64 > pilots {
                return false;
            }
            // Every reachable output position must be < n: the keys' own range and,
            // through the remap, the tail.
            if p.slot_off + p.num_keys as u64 > n || p.slot_off >= n {
                return false;
            }
            let tail = (p.num_slots - p.num_keys) as u64;
            if p.remap_off as u64 + tail > remap_len {
                return false;
            }
            let lo = p.remap_off as usize;
            let hi = lo + tail as usize;
            let limit = if p.num_keys == 0 { 1 } else { p.num_keys };
            if self.remap[lo..hi].iter().any(|&r| r >= limit) {
                return false;
            }
        }
        true
    }
}
