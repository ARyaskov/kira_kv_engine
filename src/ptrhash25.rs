//! PtrHash 2025 implementation with u8 pilots, 2-level bucketing and **partitioned
//! (sharded) construction**.
//!
//! Key design points (after Groot Koerkamp 2024 / Pibiri & Trani 2023):
//!
//! 1. **Pilots are small integers in [0, 255]**, stored as `Vec<u8>`. The slot for a key
//!    is computed by mixing `(h2, pilot_as_seed)` and reducing modulo `n` — pilots act
//!    as a per-bucket displacement seed, not as random u32 values.
//! 2. **2-level bucket assignment.** Keys are split into a "large" zone (first
//!    `ALPHA_BUCKETS` ≈ 30% of buckets, receiving ~60% of keys) and a "small" zone
//!    (remaining ~70% of buckets, receiving ~40% of keys). This skews the bucket-size
//!    distribution so the worst-case bucket is much smaller — critical for u8 pilots.
//! 3. **Largest-bucket-first pilot search.** Buckets sorted descending by size are placed
//!    when the slot map is mostly empty; tiny tail buckets get placed when it's dense.
//! 4. **Partitioned build.** Keys are first split into independent *parts* of
//!    ~`PART_TARGET_KEYS` keys by a dedicated part hash. Every part owns a contiguous
//!    range of slots and buckets and is built completely independently: hashing,
//!    bucket scatter, size sort, pilot search and fingerprints all run on a working set
//!    of ~1 MB that stays in L2. Parts are built in parallel with rayon. The old
//!    single-partition build was 86% single-threaded and, at 10M keys, every pilot
//!    probe was a DRAM miss into an 80 MB `h2` array; the partitioned build turns that
//!    into cache-resident work spread over all cores.
//! 5. **Per-part rehash.** If a part cannot be placed with u8 pilots it is retried with a
//!    different salt — only that part, not the whole index.
//! 6. **Memory layout**: pilots = one byte per bucket (compressed for large indexes).
//!    The per-part table costs 32 bytes per part (~1 byte per 1000 keys).
//!
//! Lookup is: base hash, with the part selector (xor + mul + mulhi) and the per-part
//! table load (L1-resident) computed in parallel with it → ~5-cycle per-part remix →
//! one pilot load → slot compute. Single-part indexes skip the selector and the remix
//! entirely. With pilots in LLC this is ~15-25 ns on Alder Lake, within noise of the
//! pre-partition lookup.
//!
//! Indexes with a single part (≤ `PART_TARGET_KEYS` keys) use exactly the pre-partition
//! formulas and wire format, so small indexes are bit-compatible with v0.6.

#![allow(dead_code)]

use thiserror::Error;

/// Fraction of buckets assigned to the "large" zone (high-density region).
/// Empirically 0.30 works well across 1M..1B keys.
const ALPHA_BUCKETS: f64 = 0.30;
/// Fraction of keys hashed into the "large" zone (matches Pibiri/Trani PTHash3).
const BETA_KEYS: f64 = 0.60;
/// `BETA_KEYS` expressed as a threshold on the top 16 bits of `h1`.
const BETA_THRESHOLD: u32 = (BETA_KEYS * 65536.0) as u32;

/// Target number of keys per part. 32K keys → ~35K slots, ~70K buckets; the per-part
/// working set (h1, bucket ids, offsets, items, order, occupancy bitmap) is ~1 MB,
/// which fits the 1.25 MB L2 of a Golden Cove P-core.
pub const PART_TARGET_KEYS: usize = 1 << 15;

/// Slot over-provisioning. u8 pilots can't reliably find a slot for the last bucket
/// (only 256 attempts vs 1/N success probability). 10% padding gives the late buckets
/// enough room. The resulting MPHF is "near-minimal": slots are in `[0, 1.1×N)`.
const SLOT_PADDING: f64 = 1.10;

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
/// flatness, which shifts the pilot distribution further toward zero — multiplying the
/// CompressedPilotsV2 win.
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

/// Pilot table storage. Three variants:
///
/// 1. **Flat(`HugeVec<u8>`)** — one byte per bucket. Hugepage-backed when ≥ 1 MB,
///    which on i7-12700 with 100M-key indexes means ~99% TLB hit rate vs ~1% with
///    4 KB pages. Fast random access (1 cache line per lookup).
/// 2. **Compressed(`CompressedPilots`)** — 4-bit nibbles + overflow with rank-select.
///    ~50% memory savings vs Flat at the cost of one extra rank query for the
///    overflow buckets (~5% of accesses).
/// 3. **CompressedV2** — 3-tier zero/nibble/overflow, ~2× smaller than `Compressed`.
#[derive(Debug, Clone)]
pub enum PilotTable {
    /// Flat byte-per-bucket (no compression). Used for small indexes (<256k buckets).
    Flat(crate::hugepage::HugeVec<u8>),
    /// 4-bit nibbles + u8 overflow. Used historically; kept for compatibility.
    Compressed(crate::compressed_pilots::CompressedPilots),
    /// 3-tier: zero_bitmap + 4-bit nibbles + u8 overflow. Exploits zero-skew of pilot
    /// distribution under sparse gamma. ~2× compression vs `Compressed` on typical
    /// 100M-key workloads.
    CompressedV2(crate::compressed_pilots::CompressedPilotsV2),
}

impl PilotTable {
    #[inline(always)]
    pub fn get(&self, bucket: usize) -> u8 {
        match self {
            PilotTable::Flat(v) => unsafe { *v.as_slice().get_unchecked(bucket) },
            PilotTable::Compressed(c) => c.get(bucket),
            PilotTable::CompressedV2(c) => c.get(bucket),
        }
    }

    pub fn memory_usage(&self) -> usize {
        match self {
            PilotTable::Flat(v) => v.memory_usage(),
            PilotTable::Compressed(c) => c.memory_usage(),
            PilotTable::CompressedV2(c) => c.memory_usage(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            PilotTable::Flat(v) => v.len(),
            PilotTable::Compressed(c) => c.num_buckets as usize,
            PilotTable::CompressedV2(c) => c.num_buckets as usize,
        }
    }

    fn from_flat(pilots: &[u8]) -> Self {
        // Pilot table format selection:
        //  - <256k buckets: Flat (best lookup latency, fits in L1/L2 for sure)
        //  - ≥256k buckets: CompressedV2 (3-tier zero/nibble/overflow) — exploits the
        //    fact that ~80% of pilots are 0 under sparse gamma.
        if pilots.len() >= 256 * 1024 {
            PilotTable::CompressedV2(crate::compressed_pilots::CompressedPilotsV2::from_flat(pilots))
        } else {
            PilotTable::Flat(crate::hugepage::HugeVec::from_slice(pilots))
        }
    }
}

/// Geometry + salt of one part. 32 bytes; the whole table stays L1/L2-resident
/// (1 entry per ~32K keys).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct PartInfo {
    /// First global slot of this part.
    pub slot_off: u64,
    /// Salt that successfully placed this part.
    pub salt: u64,
    /// First global bucket of this part.
    pub bucket_off: u32,
    /// Slots owned by this part (`≈ 1.10 × keys`).
    pub num_slots: u32,
    /// Buckets owned by this part (`≈ num_slots / gamma`).
    pub num_buckets: u32,
    /// Cached `num_buckets × ALPHA_BUCKETS` (large-zone size).
    pub large_buckets: u32,
}

impl PartInfo {
    #[inline]
    fn new(slot_off: u64, salt: u64, bucket_off: u32, num_slots: u32, num_buckets: u32) -> Self {
        Self {
            slot_off,
            salt,
            bucket_off,
            num_slots,
            num_buckets,
            large_buckets: large_buckets_of(num_buckets as usize) as u32,
        }
    }
}

/// Final MPHF structure. Lookup = (part hash) + hash + 1 byte load + slot compute.
#[derive(Debug, Clone)]
pub struct PtrHash25Mphf {
    /// Total number of slots across all parts (≥ original key count due to padding).
    pub n: u64,
    /// Total number of buckets across all parts.
    pub num_buckets: u32,
    /// Multi-part: salt of the global base hash (`h1 = remix(base_hash(key, salt),
    /// parts[p].salt)`). Single part: the hash salt itself (`parts[0].salt`, legacy).
    pub salt: u64,
    /// Pilot table — flat for small indexes, compressed for large.
    pub pilots: PilotTable,
    /// Optional fingerprint table (u8 per slot) for negative-query rejection.
    /// Empty if the index doesn't need negative-query support.
    pub fingerprints: Vec<u8>,
    /// True iff this MPHF was built with AES-NI hash. Lookups must use the same path.
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
    /// Slot-space size. `lookup_u64` returns values in `[0..slot_capacity())`;
    /// this is `~1.1 * n_input_keys` (over-provisioning by `SLOT_PADDING`).
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
        let pilot = self.pilots.get(bucket);
        let slot = p.slot_off as usize + slot_for(h2, pilot, p.num_slots as usize);
        (slot, h2)
    }

    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.pilots.memory_usage()
            + self.fingerprints.len()
            + self.parts.len() * std::mem::size_of::<PartInfo>()
    }
}

#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Target keys-per-bucket. Lower = sparser buckets, easier pilot search, more memory.
    /// 0.7-1.0 works well; default 0.85.
    pub gamma: f64,
    /// Maximum salt-rehash rounds (per part) before declaring the keyset unbuildable.
    pub max_rehash: u32,
    /// Whether to build the per-slot fingerprint table (adds 1 byte/key, allows negative
    /// lookups). Skip for hit-only workloads.
    pub with_fingerprints: bool,
    /// Initial salt.
    pub seed: u64,
    /// If true, use AES-NI (`hash_u64_aes`) instead of mix64 for the base hash.
    /// Both build and lookup must use the same variant — this flag affects both.
    /// AES-NI gives stronger distribution against adversarial inputs but loses the
    /// AVX2 4-wide batch speedup of mix64. Use only if you've measured a gain on
    /// your specific workload.
    pub use_aes_hash: bool,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            // 0.5 = sparse layout, 2 slots per bucket on average. Conservative for the
            // naive u8-pilot search; the paper's "true" PtrHash 2025 uses tighter
            // bucketing (~0.85) thanks to cuckoo relocation that this implementation
            // doesn't have. With 0.5, build always converges in 1-2 salt attempts.
            gamma: 0.5,
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
    /// `Sync`), not the raw `*mut T` field that edition-2021 precise capture would
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
    let total_slots = ((n as f64) * SLOT_PADDING).ceil() as usize;
    let total_buckets = (((total_slots as f64) / cfg.gamma).ceil() as usize).max(1);
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
    occupied: Vec<u64>,
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
/// global pilot / slot / fingerprint arrays.
struct PartJob<'a> {
    part_id: usize,
    keys: &'a [u64],
    info: PartInfo,
    /// `Some(global_salt)` for multi-part indexes (base hash + per-part remix),
    /// `None` for the single-part legacy formula.
    global_salt: Option<u64>,
    prerotate: u8,
    pilots: &'a mut [u8],
    /// Local slot per key (part-key order), if the caller wants them.
    slots: Option<&'a mut [u32]>,
    /// Inner u8 fingerprints (`fingerprint_u8(h2)`) per local slot.
    fp8: Option<&'a mut [u8]>,
    /// Outer u16 fingerprints (`original_key & 0xFFFF`) per local slot — written here
    /// so `Index` needs neither a slot array nor a second pass.
    fp16: Option<&'a mut [u16]>,
}

/// Which side tables [`build_partitioned_with`] should fill.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuildOutputs {
    /// Return the local slot of every key (in `Partitioned::keys` order).
    pub slots: bool,
    /// Return a `u16` fingerprint table over the global slot space, keyed by the low
    /// 16 bits of the original (un-rotated) key.
    pub fp16: bool,
}

/// Build one part. On success returns the salt that placed it and fills `job.pilots`,
/// `job.slots` (local slot per key) and `job.fp8` (if present).
fn build_part(job: &mut PartJob<'_>, sc: &mut Scratch, cfg: &BuildConfig) -> Result<u64, PtrHash25Error> {
    let keys = job.keys;
    let m = keys.len();
    let num_slots = job.info.num_slots as usize;
    let num_buckets = job.info.num_buckets as usize;
    let large = job.info.large_buckets as usize;
    let small = num_buckets - large;

    grow(&mut sc.h1, m);
    grow(&mut sc.bidx, m);
    grow(&mut sc.items_h2, m);
    grow(&mut sc.offsets, num_buckets + 1);
    grow(&mut sc.cursor, num_buckets + 1);
    grow(&mut sc.order, num_buckets);
    grow(&mut sc.occupied, num_slots.div_ceil(64));
    let Scratch {
        h1,
        bidx,
        offsets,
        cursor,
        items_h2,
        order,
        occupied,
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
    let occupied = &mut occupied[..num_slots.div_ceil(64)];

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

        // Step 4: pilot search, largest buckets first.
        occupied.fill(0);
        grow(trial, max_size.max(1));
        let trial = &mut trial[..max_size.max(1)];
        job.pilots.fill(0);
        let mut failed = false;
        for &b in order.iter() {
            let b = b as usize;
            let s = offsets[b] as usize;
            let e = offsets[b + 1] as usize;
            let len = e - s;
            if len == 0 {
                // Sizes are descending: every remaining bucket is empty.
                break;
            }
            let bucket = &items_h2[s..e];
            if len >= 2 {
                // Equal h2 ⇔ equal key (h1 is a bijection of the key for a fixed salt):
                // such a pair can never be separated by any pilot. Report exactly.
                for i in 1..len {
                    for j in 0..i {
                        if bucket[i] == bucket[j] {
                            return Err(PtrHash25Error::DuplicateKey);
                        }
                    }
                }
            }

            let mut placed = false;
            'pilots: for pilot in 0..=255u8 {
                for (i, &h2) in bucket.iter().enumerate() {
                    let slot = slot_for(h2, pilot, num_slots);
                    if (occupied[slot >> 6] >> (slot & 63)) & 1 != 0 {
                        continue 'pilots;
                    }
                    for &t in &trial[..i] {
                        if t as usize == slot {
                            continue 'pilots;
                        }
                    }
                    trial[i] = slot as u32;
                }
                for &t in &trial[..len] {
                    occupied[(t >> 6) as usize] |= 1u64 << (t & 63);
                }
                job.pilots[b] = pilot;
                placed = true;
                break;
            }
            if !placed {
                failed = true;
                break;
            }
        }
        if failed {
            continue;
        }

        // Step 5: side tables in key order — local slots, inner u8 fingerprints and/or
        // outer u16 fingerprints. Each is a separate tight loop so the common case
        // (exactly one table) stays branch-free.
        let pilots: &[u8] = job.pilots;
        let slot_of = |i: usize| -> (usize, u64) {
            let h2 = h2_from_h1(h1[i]);
            (slot_for(h2, pilots[bidx[i] as usize], num_slots), h2)
        };
        if let Some(sl) = job.slots.as_deref_mut() {
            for (i, s) in sl.iter_mut().enumerate() {
                *s = slot_of(i).0 as u32;
            }
        }
        if let Some(fp) = job.fp8.as_deref_mut() {
            for i in 0..m {
                let (slot, h2) = slot_of(i);
                fp[slot] = fingerprint_u8(h2);
            }
        }
        if let Some(fp) = job.fp16.as_deref_mut() {
            let rot = job.prerotate as u32;
            for i in 0..m {
                let slot = slot_of(i).0;
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

/// Build the MPHF from partitioned keys. Returns the structure plus the slot of every
/// key in `part.keys` order.
pub fn build_partitioned(
    part: &Partitioned,
    cfg: &BuildConfig,
) -> Result<(PtrHash25Mphf, Vec<u32>), PtrHash25Error> {
    let outputs = BuildOutputs { slots: true, fp16: false };
    let (mph, slots, _) = build_partitioned_with(part, cfg, outputs)?;
    Ok((mph, slots.expect("slots requested")))
}

/// Build the MPHF from partitioned keys, filling only the side tables requested in
/// `outputs`. Every table is written per part, in parallel, from the slots the pilot
/// search already knows — no second lookup pass and no intermediate slot array unless
/// the caller asks for one.
pub fn build_partitioned_with(
    part: &Partitioned,
    cfg: &BuildConfig,
    outputs: BuildOutputs,
) -> Result<(PtrHash25Mphf, Option<Vec<u32>>, Option<Vec<u16>>), PtrHash25Error> {
    let n = part.len();
    assert!(n > 0, "empty key set");
    let parts = part.num_parts();

    // Per-part geometry from the actual part sizes.
    let mut infos: Vec<PartInfo> = Vec::with_capacity(parts);
    let mut slot_off = 0u64;
    let mut bucket_off = 0u64;
    for p in 0..parts {
        let m = part.part_offsets[p + 1] - part.part_offsets[p];
        let slots = (((m as f64) * SLOT_PADDING).ceil() as usize).max(1);
        let buckets = (((slots as f64) / cfg.gamma).ceil() as usize).max(1);
        infos.push(PartInfo::new(slot_off, 0, bucket_off as u32, slots as u32, buckets as u32));
        slot_off += slots as u64;
        bucket_off += buckets as u64;
        assert!(bucket_off <= u32::MAX as u64, "too many buckets for u32 addressing");
    }
    let total_slots = slot_off as usize;
    let total_buckets = bucket_off as usize;

    let mut pilots_buf = crate::hugepage::HugeVec::<u8>::zeroed(total_buckets);
    let mut slots = if outputs.slots { vec![0u32; n] } else { Vec::new() };
    let mut fp8 = if cfg.with_fingerprints { vec![0u8; total_slots] } else { Vec::new() };
    let mut fp16 = if outputs.fp16 { vec![0u16; total_slots] } else { Vec::new() };
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
        let mut sl_rest: &mut [u32] = &mut slots;
        let mut fp8_rest: &mut [u8] = &mut fp8;
        let mut fp16_rest: &mut [u16] = &mut fp16;
        for (p, info) in infos.iter().enumerate() {
            let m = part.part_offsets[p + 1] - part.part_offsets[p];
            let ns = info.num_slots as usize;
            let (pil, r) = std::mem::take(&mut pil_rest).split_at_mut(info.num_buckets as usize);
            pil_rest = r;
            let sl = outputs.slots.then(|| {
                let (s, r) = std::mem::take(&mut sl_rest).split_at_mut(m);
                sl_rest = r;
                s
            });
            let f8 = cfg.with_fingerprints.then(|| {
                let (f, r) = std::mem::take(&mut fp8_rest).split_at_mut(ns);
                fp8_rest = r;
                f
            });
            let f16 = outputs.fp16.then(|| {
                let (f, r) = std::mem::take(&mut fp16_rest).split_at_mut(ns);
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

    let pilot_table = PilotTable::from_flat(pilots_buf.as_slice());
    drop(pilots_buf);

    Ok((
        PtrHash25Mphf {
            n: total_slots as u64,
            num_buckets: total_buckets as u32,
            salt: global_salt.unwrap_or(infos[0].salt),
            pilots: pilot_table,
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

/// Wire-format writer for index serialization. We always persist the flat byte form;
/// the compressed view is reconstructed on load when the table is large enough to
/// benefit from it.
///
/// Layout: `[n u64][num_buckets u32][salt u64][plen u64][pilots][flen u64][fps]
/// [use_aes u8][prerotate u8]` — identical to v0.6 for single-part indexes. Multi-part
/// indexes set bit 7 of the prerotate byte and append
/// `[part_salt u64][parts u32]{[slot_off u64][salt u64][bucket_off u32][num_slots u32]
/// [num_buckets u32]}*`. v0.6 readers never see the flag because v0.6 never produced
/// multi-part tables; v0.6 files load here as a single part.
pub fn write_ptrhash25(mph: &PtrHash25Mphf, out: &mut Vec<u8>) {
    out.extend_from_slice(&mph.n.to_le_bytes());
    out.extend_from_slice(&mph.num_buckets.to_le_bytes());
    out.extend_from_slice(&mph.salt.to_le_bytes());
    let flat: Vec<u8> = match &mph.pilots {
        PilotTable::Flat(v) => v.as_slice().to_vec(),
        PilotTable::Compressed(c) => (0..c.num_buckets as usize).map(|b| c.get(b)).collect(),
        PilotTable::CompressedV2(c) => (0..c.num_buckets as usize).map(|b| c.get(b)).collect(),
    };
    out.extend_from_slice(&(flat.len() as u64).to_le_bytes());
    out.extend_from_slice(&flat);
    out.extend_from_slice(&(mph.fingerprints.len() as u64).to_le_bytes());
    out.extend_from_slice(&mph.fingerprints);
    // Trailing bytes (v0.5):
    //  [0] hash variant flag (0 = mix64, 1 = AES)
    //  [1] data-driven prerotation (0..63); bit 7 = multi-part table follows
    out.push(if mph.use_aes_hash { 1 } else { 0 });
    let multi = mph.parts.len() > 1;
    out.push((mph.prerotate & 0x3F) | if multi { 0x80 } else { 0 });
    if multi {
        out.extend_from_slice(&mph.part_salt.to_le_bytes());
        out.extend_from_slice(&(mph.parts.len() as u32).to_le_bytes());
        for p in mph.parts.iter() {
            out.extend_from_slice(&p.slot_off.to_le_bytes());
            out.extend_from_slice(&p.salt.to_le_bytes());
            out.extend_from_slice(&p.bucket_off.to_le_bytes());
            out.extend_from_slice(&p.num_slots.to_le_bytes());
            out.extend_from_slice(&p.num_buckets.to_le_bytes());
        }
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
    let pilots = PilotTable::from_flat(&buf[*pos..*pos + plen]);
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
    let (part_salt, parts) = if rot_byte & 0x80 != 0 {
        let part_salt = rd_u64(buf, pos)?;
        let count = rd_u32(buf, pos)? as usize;
        if count == 0 {
            return None;
        }
        let mut parts = Vec::with_capacity(count);
        for _ in 0..count {
            let slot_off = rd_u64(buf, pos)?;
            let psalt = rd_u64(buf, pos)?;
            let bucket_off = rd_u32(buf, pos)?;
            let num_slots = rd_u32(buf, pos)?;
            let nb = rd_u32(buf, pos)?;
            parts.push(PartInfo::new(slot_off, psalt, bucket_off, num_slots, nb));
        }
        (part_salt, parts)
    } else {
        (0, vec![PartInfo::new(0, salt, 0, n as u32, num_buckets)])
    };
    Some(PtrHash25Mphf {
        n,
        num_buckets,
        salt,
        pilots,
        fingerprints,
        use_aes_hash,
        prerotate,
        part_salt,
        parts: parts.into_boxed_slice(),
    })
}
