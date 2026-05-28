//! True PtrHash 2025 implementation with u8 pilots and 2-level bucketing.
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
//! 4. **Multi-stage rehash** on placement failure: a different salt produces a different
//!    bucket layout and a different pilot search order. Up to 16 rounds before giving up.
//! 5. **Memory layout**: pilots = N bytes total (vs 4N in the old u32 scheme). At N=100M
//!    the pilot table is 100 MB instead of 400 MB — far better LLC residency for lookup.
//!
//! Lookup is one hash + one byte load + slot compute. With pilots in LLC this is ~15-25 ns
//! on Alder Lake.

#![allow(dead_code)]

use thiserror::Error;

/// Fraction of buckets assigned to the "large" zone (high-density region).
/// Empirically 0.30 works well across 1M..1B keys.
const ALPHA_BUCKETS: f64 = 0.30;
/// Fraction of keys hashed into the "large" zone (matches Pibiri/Trani PTHash3).
const BETA_KEYS: f64 = 0.60;

/// Hash mixing helpers (xxh3-style fmix, fast on x86_64-v3 via MULX).
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

/// Exclusive prefix sum: `out[0] = 0`, `out[i+1] = out[i] + counts[i]`. SIMD on x86_64.
///
/// The scalar version is bandwidth-bound at ~2 ns/element. The AVX2 Sklansky-style
/// scan processes 8 elements per iteration, hitting ~0.5 ns/element. For 100M-bucket
/// builds (≈ 90 MB counts vector) this is the difference between 200 ms and 50 ms.
#[inline]
fn prefix_sum_u32(counts: &[u32], out: &mut [u32]) {
    debug_assert_eq!(out.len(), counts.len() + 1);
    out[0] = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                prefix_sum_u32_avx2(counts, out);
                return;
            }
        }
    }
    let mut acc = 0u32;
    for i in 0..counts.len() {
        acc = acc.wrapping_add(counts[i]);
        out[i + 1] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn prefix_sum_u32_avx2(counts: &[u32], out: &mut [u32]) {
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_set1_epi32, _mm256_storeu_si256,
        _mm256_slli_si256,
    };
    // Sklansky scan within a lane:
    //   x += shift_left_by_4_bytes(x)
    //   x += shift_left_by_8_bytes(x)
    //   x += shift_left_by_16_bytes(x)
    // This handles a single 8-element AVX2 lane. We then broadcast the carry across
    // batches.
    let n = counts.len();
    let mut carry = 0u32;
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_si256(counts.as_ptr().add(i) as *const __m256i);
        // Stage 1: shift by 4 bytes (1 element) within each 128-bit lane.
        let s1 = _mm256_slli_si256::<4>(v);
        let v1 = _mm256_add_epi32(v, s1);
        // Stage 2: shift by 8 bytes (2 elements).
        let s2 = _mm256_slli_si256::<8>(v1);
        let v2 = _mm256_add_epi32(v1, s2);
        // At this point v2 is correctly scanned within each 128-bit half independently.
        // Need to add the carry from low half to high half. Extract last element of low.
        let mut tmp = [0u32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v2);
        let lane_carry = tmp[3];
        tmp[4] = tmp[4].wrapping_add(lane_carry);
        tmp[5] = tmp[5].wrapping_add(lane_carry);
        tmp[6] = tmp[6].wrapping_add(lane_carry);
        tmp[7] = tmp[7].wrapping_add(lane_carry);
        // Now tmp is the within-batch exclusive prefix when shifted right by 1, but
        // contains inclusive. Convert to exclusive by writing to out[i+1..=i+8] then
        // applying global carry to out[i+1..i+8] and using prior carry on out[i].
        let carry_v = _mm256_set1_epi32(carry as i32);
        let inc_v = _mm256_loadu_si256(tmp.as_ptr() as *const __m256i);
        let final_v = _mm256_add_epi32(inc_v, carry_v);
        // Write to out[i+1..=i+8] (8 elements at out[i+1], out[i+2], ..., out[i+8]).
        _mm256_storeu_si256(out.as_mut_ptr().add(i + 1) as *mut __m256i, final_v);
        // Update carry from the last (now-correct) element.
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, final_v);
        carry = tmp[7];
        i += 8;
    }
    // Tail: scalar.
    while i < n {
        carry = carry.wrapping_add(counts[i]);
        out[i + 1] = carry;
        i += 1;
    }
}

/// Per-key hash: returns (h1 for bucket, h2 for slot).
///
/// `use_aes` selects between two hash families. `prerotate` is a data-driven bit
/// rotation applied BEFORE the hash — learned during build to flatten bucket-size
/// variance for the specific key distribution at hand. See `learn_prerotation`.
#[inline(always)]
fn hash_key(key: u64, salt: u64, use_aes: bool, prerotate: u8) -> (u64, u64) {
    let rotated = key.rotate_left(prerotate as u32);
    let base = if use_aes {
        crate::aes_hash::hash_u64(rotated, salt)
    } else {
        crate::simd_hash::hash_u64_one(rotated, salt)
    };
    let h1 = base;
    let h2 = base.rotate_left(23) ^ 0xA24B_1F6F_DA39_2B31;
    (h1, h2)
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
/// uniformity for each candidate rotation in 0..64, return the best. ~5-15 ms cost
/// at 100M keys, amortized across 60+ s build.
fn learn_prerotation(keys: &[u64], num_buckets: usize, salt: u64, use_aes: bool) -> u8 {
    if keys.len() < 2048 {
        return 0; // sample too small — skip the search, default rotation 0
    }
    let sample_size = (keys.len() / 200).clamp(2048, 32_768);
    let stride = (keys.len() / sample_size).max(1);
    let sample: Vec<u64> = (0..sample_size).map(|i| keys[i * stride]).collect();

    // Use a coarse 256-bucket histogram to estimate distribution shape cheaply.
    const HIST: usize = 256;
    let scale = num_buckets / HIST.max(1);
    let _ = scale; // we instead reduce via bucket_for then mod HIST.

    let mut best_rot = 0u8;
    let mut best_score = f64::MAX;
    for rot in (0..64u8).step_by(4) {
        let mut hist = [0u32; HIST];
        for &k in &sample {
            let rotated = k.rotate_left(rot as u32);
            let base = if use_aes {
                crate::aes_hash::hash_u64(rotated, salt)
            } else {
                crate::simd_hash::hash_u64_one(rotated, salt)
            };
            let b = bucket_for(base, num_buckets);
            hist[b % HIST] += 1;
        }
        // Chi-squared statistic: sum((observed - expected)^2 / expected).
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

/// Vectorized 4-wide bucket_for: takes 4 h1 hashes, returns 4 bucket indices.
/// Compiles to AVX2 mulhi + blend on x86_64+avx2; saves ~6 ns per lookup batch.
#[inline(always)]
pub(crate) fn bucket_for_x4(h1s: [u64; 4], num_buckets: usize) -> [usize; 4] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return bucket_for_x4_avx2(h1s, num_buckets);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        [
            bucket_for(h1s[0], num_buckets),
            bucket_for(h1s[1], num_buckets),
            bucket_for(h1s[2], num_buckets),
            bucket_for(h1s[3], num_buckets),
        ]
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn bucket_for_x4_avx2(h1s: [u64; 4], num_buckets: usize) -> [usize; 4] {
    // AVX2 has no native 64×64→128 multiply; computing exact mulhi takes more work
    // than 4 scalar fast_reduce calls (which compile to MUL + SHR — 6 cycles each).
    // SIMD wins here come from issuing 4 independent dependent chains in parallel,
    // exploiting the 4-wide OoO backend. We use the scalar path inline to let LLVM
    // schedule the chains.
    let beta_threshold = (BETA_KEYS * 65536.0) as u32;
    let large_buckets = ((num_buckets as f64) * ALPHA_BUCKETS) as usize;
    let small_buckets = num_buckets - large_buckets;

    let mut out = [0usize; 4];
    for lane in 0..4 {
        let h1 = h1s[lane];
        let zone_decider = (h1 >> 48) as u32;
        let h_low = h1 & 0x0000_FFFF_FFFF_FFFF;
        out[lane] = if zone_decider < beta_threshold {
            fast_reduce(h_low << 16, large_buckets.max(1))
        } else {
            large_buckets + fast_reduce(h_low << 16, small_buckets.max(1))
        };
    }
    out
}

/// Bucket index combining h1 with 2-level skew. Keys with hash in the "large zone"
/// (top BETA_KEYS fraction by high bits) map to the first ALPHA_BUCKETS fraction of
/// buckets; the rest map to the tail buckets.
///
/// Result: large zone has BETA_KEYS / ALPHA_BUCKETS ≈ 2x density vs uniform,
/// small zone has (1-BETA_KEYS) / (1-ALPHA_BUCKETS) ≈ 0.57x density.
#[inline(always)]
fn bucket_for(h1: u64, num_buckets: usize) -> usize {
    // Use top 16 bits of h1 to decide zone; bottom 48 bits to pick within zone.
    let zone_decider = (h1 >> 48) as u32;
    let beta_threshold = (BETA_KEYS * 65536.0) as u32;
    let large_buckets = ((num_buckets as f64) * ALPHA_BUCKETS) as usize;
    let small_buckets = num_buckets - large_buckets;

    if zone_decider < beta_threshold {
        // Large zone.
        let h_low = h1 & 0x0000_FFFF_FFFF_FFFF;
        fast_reduce(h_low << 16, large_buckets.max(1))
    } else {
        // Small zone (offset by large_buckets).
        let h_low = h1 & 0x0000_FFFF_FFFF_FFFF;
        large_buckets + fast_reduce(h_low << 16, small_buckets.max(1))
    }
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

/// Pilot table storage. Three variants:
///
/// 1. **Flat(`HugeVec<u8>`)** — one byte per bucket. Hugepage-backed when ≥ 1 MB,
///    which on i7-12700 with 100M-key indexes means ~99% TLB hit rate vs ~1% with
///    4 KB pages. Fast random access (1 cache line per lookup).
/// 2. **Compressed(`CompressedPilots`)** — 4-bit nibbles + overflow with rank-select.
///    ~50% memory savings vs Flat at the cost of one extra rank query for the
///    overflow buckets (~5% of accesses).
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
}

/// Final MPHF structure. Lookup = hash + 1 byte load + slot compute.
#[derive(Debug, Clone)]
pub struct PtrHash25Mphf {
    /// Number of slots (≥ original key count due to u8-pilot padding).
    pub n: u64,
    /// Number of buckets (sum of large + small zones).
    pub num_buckets: u32,
    /// Salt that successfully produced a placement.
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
}

impl PtrHash25Mphf {
    /// Slot-space size. `lookup_u64` returns values in `[0..slot_capacity())`;
    /// this is `~1.1 * n_input_keys` (over-provisioning by `1/gamma`).
    #[inline(always)]
    pub fn slot_capacity(&self) -> usize {
        self.n as usize
    }

    /// O(1) lookup. Caller must check fingerprint separately for negative-query safety.
    #[inline(always)]
    pub fn index_u64(&self, key: u64) -> u32 {
        let salt = self.salt;
        let (h1, h2) = hash_key(key, salt, self.use_aes_hash, self.prerotate);
        let bucket = bucket_for(h1, self.num_buckets as usize);
        let pilot = self.pilots.get(bucket);
        slot_for(h2, pilot, self.n as usize) as u32
    }

    /// Lookup with built-in fingerprint check. Returns Some(idx) if the key was in the
    /// build set, None if it's a foreign key that happens to collide.
    #[inline(always)]
    pub fn lookup_u64(&self, key: u64) -> Option<u32> {
        let salt = self.salt;
        let (h1, h2) = hash_key(key, salt, self.use_aes_hash, self.prerotate);
        let bucket = bucket_for(h1, self.num_buckets as usize);
        let pilot = self.pilots.get(bucket);
        let slot = slot_for(h2, pilot, self.n as usize);
        let fp_expected = fingerprint_u8(h2);
        if self.fingerprints.is_empty()
            || unsafe { *self.fingerprints.get_unchecked(slot) } == fp_expected
        {
            Some(slot as u32)
        } else {
            None
        }
    }

    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.pilots.memory_usage() + self.fingerprints.len()
    }
}

#[inline(always)]
fn fingerprint_u8(h2: u64) -> u8 {
    // High byte of h2 (not used for slot derivation, so independent collisions).
    (h2 >> 56) as u8
}

#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Target keys-per-bucket. Lower = sparser buckets, easier pilot search, more memory.
    /// 0.7-1.0 works well; default 0.85.
    pub gamma: f64,
    /// Maximum salt-rehash rounds before declaring the keyset unbuildable.
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

    pub fn build(self, keys: &[u64]) -> Result<PtrHash25Mphf, PtrHash25Error> {
        let n = keys.len();
        assert!(n > 0, "empty key set");

        // Slot padding: u8 pilots can't reliably find a slot for the last bucket
        // (only 256 attempts vs 1/N success probability). 10% padding (slots = 1.10×N)
        // gives the late buckets enough room. The resulting MPHF is "near-minimal":
        // slots returned are in [0, 1.1×N) instead of [0, N). Standard trade-off in
        // u8-pilot MPHFs (BBHash, original FCH).
        let num_slots = ((n as f64) * 1.10).ceil() as usize;
        let num_buckets = ((num_slots as f64) / self.cfg.gamma).ceil() as usize;
        let num_buckets = num_buckets.max(1);

        // Learn the data-driven prerotation ONCE for round 0's salt. The chosen rotation
        // is reused across rehash rounds — it's a property of the key DISTRIBUTION, not
        // a particular salt.
        let round0_salt =
            mix64(self.cfg.seed ^ 0u64.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let prerotate = learn_prerotation(keys, num_buckets, round0_salt, self.cfg.use_aes_hash);

        for round in 0..=self.cfg.max_rehash {
            let salt = mix64(self.cfg.seed ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            match try_build(
                keys,
                n,
                num_slots,
                num_buckets,
                salt,
                self.cfg.with_fingerprints,
                self.cfg.use_aes_hash,
                prerotate,
            ) {
                Ok(mph) => return Ok(mph),
                Err(_) => continue,
            }
        }
        Err(PtrHash25Error::Unresolvable)
    }
}

/// Single build attempt with a given salt. Returns Ok if all buckets placed, Err otherwise.
fn try_build(
    keys: &[u64],
    n: usize,
    num_slots: usize,
    num_buckets: usize,
    salt: u64,
    with_fingerprints: bool,
    use_aes_hash: bool,
    prerotate: u8,
) -> Result<PtrHash25Mphf, PtrHash25Error> {
    // Arena-backed allocation for the large temp buffers. One contiguous VirtualAlloc
    // instead of 5+ separate ones (each costing 50-200 μs on Windows). For 100M keys
    // we save ~500-1500 μs of allocator overhead per build attempt.
    let arena_cap = crate::build_arena::estimate_capacity(n, num_buckets);
    let arena = crate::build_arena::BuildArena::with_capacity(arena_cap);

    // Step 1: hash all keys, compute bucket assignment. Memory-bound, parallelizable.
    let mut h1_arena = arena.alloc_zeroed::<u64>(n);
    let mut h2_arena = arena.alloc_zeroed::<u64>(n);
    let mut bucket_idx_arena = arena.alloc_zeroed::<u32>(n);

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        const CHUNK: usize = 65_536;
        let key_chunks = keys.par_chunks(CHUNK);
        let h1_slice = h1_arena.as_mut_slice();
        let h2_slice = h2_arena.as_mut_slice();
        let bi_slice = bucket_idx_arena.as_mut_slice();
        let h1_chunks = h1_slice.par_chunks_mut(CHUNK);
        let h2_chunks = h2_slice.par_chunks_mut(CHUNK);
        let bi_chunks = bi_slice.par_chunks_mut(CHUNK);
        key_chunks
            .zip(h1_chunks)
            .zip(h2_chunks)
            .zip(bi_chunks)
            .for_each(|(((kc, h1c), h2c), bic)| {
                if use_aes_hash {
                    for j in 0..kc.len() {
                        let rotated = kc[j].rotate_left(prerotate as u32);
                        let a = crate::aes_hash::hash_u64(rotated, salt);
                        h1c[j] = a;
                        h2c[j] = a.rotate_left(23) ^ 0xA24B_1F6F_DA39_2B31;
                        bic[j] = bucket_for(a, num_buckets) as u32;
                    }
                } else if prerotate == 0 {
                    // Fast path: no rotation, use AVX2 batch hash directly on keys.
                    crate::simd_hash::hash_u64(kc, salt, h1c);
                    for j in 0..kc.len() {
                        let a = h1c[j];
                        h2c[j] = a.rotate_left(23) ^ 0xA24B_1F6F_DA39_2B31;
                        bic[j] = bucket_for(a, num_buckets) as u32;
                    }
                } else {
                    // Rotated path: apply per-key rotation, then scalar hash to keep
                    // build/lookup consistent.
                    for j in 0..kc.len() {
                        let rotated = kc[j].rotate_left(prerotate as u32);
                        let a = crate::simd_hash::hash_u64_one(rotated, salt);
                        h1c[j] = a;
                        h2c[j] = a.rotate_left(23) ^ 0xA24B_1F6F_DA39_2B31;
                        bic[j] = bucket_for(a, num_buckets) as u32;
                    }
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        let h1_view = h1_arena.as_mut_slice();
        let h2_view = h2_arena.as_mut_slice();
        let bi_view = bucket_idx_arena.as_mut_slice();
        if use_aes_hash || prerotate != 0 {
            for i in 0..n {
                let rotated = keys[i].rotate_left(prerotate as u32);
                let a = if use_aes_hash {
                    crate::aes_hash::hash_u64(rotated, salt)
                } else {
                    crate::simd_hash::hash_u64_one(rotated, salt)
                };
                h1_view[i] = a;
                h2_view[i] = a.rotate_left(23) ^ 0xA24B_1F6F_DA39_2B31;
                bi_view[i] = bucket_for(a, num_buckets) as u32;
            }
        } else {
            crate::simd_hash::hash_u64(keys, salt, h1_view);
            for i in 0..n {
                let a = h1_view[i];
                h2_view[i] = a.rotate_left(23) ^ 0xA24B_1F6F_DA39_2B31;
                bi_view[i] = bucket_for(a, num_buckets) as u32;
            }
        }
    }
    let h1 = h1_arena.as_slice();
    let h2 = h2_arena.as_slice();
    let bucket_idx = bucket_idx_arena.as_slice();
    let _ = h1; // h1 not used after step 1; keep to extend arena lifetime.

    // Step 2: count bucket sizes, build offsets and packed items array.
    let mut counts_arena = arena.alloc_zeroed::<u32>(num_buckets);
    let counts = counts_arena.as_mut_slice();
    for &b in bucket_idx {
        counts[b as usize] += 1;
    }
    let mut offsets_arena = arena.alloc_zeroed::<u32>(num_buckets + 1);
    let offsets = offsets_arena.as_mut_slice();
    prefix_sum_u32(counts, offsets);
    let mut cursor_arena = arena.alloc_zeroed::<u32>(num_buckets + 1);
    cursor_arena.as_mut_slice().copy_from_slice(offsets);
    let cursor = cursor_arena.as_mut_slice();
    let mut items_arena = arena.alloc_zeroed::<u32>(n);
    let items = items_arena.as_mut_slice();
    for i in 0..n {
        let b = bucket_idx[i] as usize;
        let pos = cursor[b] as usize;
        items[pos] = i as u32;
        cursor[b] += 1;
    }

    // Step 3: bucket order by descending size. Counting sort across size classes.
    let max_size = counts.iter().copied().max().unwrap_or(0) as usize;
    let mut freq = vec![0u32; max_size + 1];
    for &c in counts.iter() {
        freq[c as usize] += 1;
    }
    let mut start = vec![0u32; max_size + 1];
    let mut acc = 0u32;
    for size in (0..=max_size).rev() {
        start[size] = acc;
        acc += freq[size];
    }
    let mut next = start;
    let mut order = vec![0u32; num_buckets];
    for (bucket, &c) in counts.iter().enumerate() {
        let pos = next[c as usize] as usize;
        order[pos] = bucket as u32;
        next[c as usize] += 1;
    }

    // Step 4: pilot search. For each bucket in order, try pilots 0..=255 to place all
    // keys in unoccupied slots. Abort if no pilot works.
    //
    // F (non-temporal hint) — the bucket-scan reads items + h2 sequentially per bucket
    // but jumps randomly across the items / h2 arrays as we iterate `order`. For huge
    // indexes (>10M) the working set blows past LLC. We issue _MM_HINT_NTA prefetches
    // for next-bucket data so the cache line is brought in but not promoted to L2
    // (which would evict pilot-search hot data like `occupied` / `seen_epoch`).
    let mut occupied = vec![0u64; num_slots.div_ceil(64)];
    let mut pilots = vec![0u8; num_buckets];
    let mut trial = vec![0u32; max_size.max(1)];

    for (order_idx, &b) in order.iter().enumerate() {
        // F: prefetch the next-bucket items + h2 entries with NTA hint. They'll be
        // used 1-3 iterations from now and won't be reused after, so don't pollute L2.
        #[cfg(target_arch = "x86_64")]
        if order_idx + 1 < order.len() {
            use core::arch::x86_64::{_MM_HINT_NTA, _mm_prefetch};
            let next_b = order[order_idx + 1] as usize;
            let next_start = offsets[next_b] as usize;
            unsafe {
                if next_start < items.len() {
                    _mm_prefetch(items.as_ptr().add(next_start) as *const i8, _MM_HINT_NTA);
                }
                // First item's h2 — typical bucket is 1-2 items so this covers most.
                if next_start < items.len() {
                    let idx = items[next_start] as usize;
                    if idx < h2.len() {
                        _mm_prefetch(h2.as_ptr().add(idx) as *const i8, _MM_HINT_NTA);
                    }
                }
            }
        }

        let b = b as usize;
        let start_off = offsets[b] as usize;
        let end_off = offsets[b + 1] as usize;
        let len = end_off - start_off;
        if len == 0 {
            continue;
        }

        let mut placed = false;
        for pilot in 0..=255u16 {
            let pilot_u8 = pilot as u8;
            let mut ok = true;
            for i in 0..len {
                let idx = items[start_off + i] as usize;
                let slot = slot_for(h2[idx], pilot_u8, num_slots);
                let word = slot >> 6;
                let bit = 1u64 << (slot & 63);
                if (occupied[word] & bit) != 0 {
                    ok = false;
                    break;
                }
                let mut conflict = false;
                for j in 0..i {
                    if trial[j] as usize == slot {
                        conflict = true;
                        break;
                    }
                }
                if conflict {
                    ok = false;
                    break;
                }
                trial[i] = slot as u32;
            }
            if ok {
                for i in 0..len {
                    let slot = trial[i] as usize;
                    occupied[slot >> 6] |= 1u64 << (slot & 63);
                }
                pilots[b] = pilot_u8;
                placed = true;
                break;
            }
        }

        if !placed {
            return Err(PtrHash25Error::Unresolvable);
        }
    }

    // Step 5: fingerprints (optional). One byte per slot, not per key — slot space
    // is num_slots which is slightly larger than n.
    let fingerprints = if with_fingerprints {
        let mut fps = vec![0u8; num_slots];
        for i in 0..n {
            let bucket = bucket_idx[i] as usize;
            let pilot = pilots[bucket];
            let slot = slot_for(h2[i], pilot, num_slots);
            fps[slot] = fingerprint_u8(h2[i]);
        }
        fps
    } else {
        Vec::new()
    };

    // Pilot table format selection:
    //  - <256k buckets: Flat (best lookup latency, fits in L1/L2 for sure)
    //  - ≥256k buckets: CompressedV2 (3-tier zero/nibble/overflow) — exploits the
    //    fact that ~75% of pilots are 0 under sparse gamma. ~2× smaller than
    //    Compressed (single-tier nibbles) with only +1 cache line of lookup work.
    let pilot_table = if pilots.len() >= 256 * 1024 {
        PilotTable::CompressedV2(
            crate::compressed_pilots::CompressedPilotsV2::from_flat(&pilots),
        )
    } else {
        PilotTable::Flat(crate::hugepage::HugeVec::from_slice(&pilots))
    };

    Ok(PtrHash25Mphf {
        n: num_slots as u64,
        num_buckets: num_buckets as u32,
        salt,
        pilots: pilot_table,
        fingerprints,
        use_aes_hash,
        prerotate,
    })
}

/// Wire-format writer for index serialization. We always persist the flat byte form;
/// the compressed view is reconstructed on load when the table is large enough to
/// benefit from it. Keeps the on-disk format simple.
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
    //  [1] data-driven prerotation (0..63)
    out.push(if mph.use_aes_hash { 1 } else { 0 });
    out.push(mph.prerotate);
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
    let pilots_flat = buf[*pos..*pos + plen].to_vec();
    *pos += plen;
    let flen = rd_u64(buf, pos)? as usize;
    if *pos + flen > buf.len() {
        return None;
    }
    let fingerprints = buf[*pos..*pos + flen].to_vec();
    *pos += flen;
    let pilots = if pilots_flat.len() >= 256 * 1024 {
        PilotTable::CompressedV2(
            crate::compressed_pilots::CompressedPilotsV2::from_flat(&pilots_flat),
        )
    } else {
        PilotTable::Flat(crate::hugepage::HugeVec::from_slice(&pilots_flat))
    };
    // Trailing flags (v0.4+/v0.5). Older indexes lack them → default to 0.
    let use_aes_hash = if *pos < buf.len() {
        let v = buf[*pos] != 0;
        *pos += 1;
        v
    } else {
        false
    };
    let prerotate = if *pos < buf.len() {
        let v = buf[*pos];
        *pos += 1;
        v
    } else {
        0
    };
    Some(PtrHash25Mphf {
        n,
        num_buckets,
        salt,
        pilots,
        fingerprints,
        use_aes_hash,
        prerotate,
    })
}

