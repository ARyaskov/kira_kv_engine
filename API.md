# Kira KV Engine — Public API

The crate exposes **six engines** plus shared primitives. All are accessible
from the crate root:

```rust
use kira_kv_engine::{
    // Static MPH (PtrHash25)
    Index, IndexBuilder, IndexConfig, IndexError, IndexStats,
    // Dynamic MPH (LSM-style)
    DynamicIndex, DynamicConfig, StableId,
    // PGM learned index
    PgmIndex, PgmBuilder, PgmStats, PgmError,
    PgmIndexU128, PgmU128Error,
    // Hybrid (PGM + per-segment MPH)
    HybridIndex, HybridBuilder, HybridError, HybridStorageStats,
    // Compact key storage
    EliasFano,
    // Workload-aware caching
    HotTierIndex, DynamicHotTier, SpaceSaving,
    // GPU export (PtrHash25 snapshot for off-host lookup)
    GpuExport, GpuPart, BloomExport,
    // Diagnostics
    hugepages_available,
    // Deprecated remnants of the multi-backend era
    BackendKind, BackendBuildConfig, BuildProfile, MphBackend,
};
```

Everything marked *deprecated* below compiles with a warning and is removed in
the next breaking release.

---

## `Index` (PtrHash25 static MPH)

Top-level engine for any byte key. Returns a stable `usize` id in `[0..len())`
— the MPH is minimal. (Indexes loaded from 0.6 files keep their `[0..1.1·n)`
range; `slot_capacity()` is the bound either way.)

### `IndexBuilder`

```rust
pub struct IndexBuilder { /* opaque */ }

impl IndexBuilder {
    pub fn new() -> Self;
    pub fn with_config(self, config: IndexConfig) -> Self;
    pub fn with_mph_config(self, mph_config: ptrhash25::BuildConfig) -> Self;
    pub fn with_parallel_build(self, enabled: bool) -> Self;

    /// Drop Bloom + fingerprints: 0.42 B/key, ~4x faster lookups.
    /// Foreign keys return arbitrary in-range positions — closed-world only.
    pub fn with_lean_mph(self, enabled: bool) -> Self;

    // deprecated, no effect: with_pgm_epsilon, with_backend, with_hot_fraction,
    // with_build_fast_profile, auto_detect_numeric, with_pgm_bloom,
    // with_pgm_elias_fano, with_pgm_target_lookup_ns

    /// Owned keys: same build as `build_index_ref`, then `keys` is dropped in
    /// parallel on the build pool.
    pub fn build_index<K: AsRef<[u8]> + Send + Sync>(self, keys: Vec<K>)
        -> Result<Index, IndexError>;

    /// Borrowed keys. No clone on the caller side, no deallocation inside the
    /// build. Duplicates → `Err(IndexError::DuplicateKey)`.
    pub fn build_index_ref<K: AsRef<[u8]> + Sync>(self, keys: &[K])
        -> Result<Index, IndexError>;
}
```

Build diagnostics: `KIRA_BUILD_TRACE=1` prints per-phase timings to stderr. Build
pool: every core by default, P-cores (pinned) on hybrid CPUs; `KIRA_BUILD_THREADS`,
`KIRA_BUILD_CORE_IDS`, `KIRA_BUILD_PIN=0` override.

### `IndexConfig`

```rust
pub struct IndexConfig {
    pub mph_config: ptrhash25::BuildConfig,   // lambda (keys/bucket, 3.0), alpha (load, 0.98),
                                              // max_rehash, seed, with_fingerprints, use_aes_hash
    pub enable_parallel_build: bool,
    pub lean_mph: bool,
    // deprecated, no effect: pgm_epsilon, auto_detect_numeric, backend, hot_fraction,
    // build_fast_profile, pgm_enable_bloom, pgm_enable_elias_fano, pgm_target_lookup_ns
}
```

Default: `lean_mph=false`, `enable_parallel_build=true` (with the `parallel` feature).

### `Index` lookups

```rust
impl Index {
    pub fn empty() -> Self;
    pub fn is_empty(&self) -> bool;
    pub fn len(&self) -> usize;
    /// Exclusive upper bound of every lookup result (== len() for 0.7 builds).
    pub fn slot_capacity(&self) -> usize;

    // Point lookups (Err(KeyNotFound) on a miss)
    pub fn lookup(&self, key: &[u8]) -> Result<usize, IndexError>;
    pub fn lookup_str(&self, key: &str) -> Result<usize, IndexError>;
    pub fn lookup_u64(&self, key: u64) -> Result<usize, IndexError>;

    // Membership: Bloom answer — definite `false`, probabilistic `true`.
    // Lean indexes have no filter and always answer `true`.
    pub fn supports_negative_lookups(&self) -> bool;
    pub fn contains(&self, key: &[u8]) -> bool;
    pub fn contains_batch(&self, keys: &[&[u8]]) -> Vec<bool>;

    // Batched (prefetch pipelines; AVX2 gather dispatched at runtime)
    pub fn lookup_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>>;
    pub fn lookup_batch_pipelined(&self, keys: &[&[u8]]) -> Vec<Option<usize>>;
    pub fn lookup_batch_u64_simd(&self, keys: &[u64]) -> Vec<Option<usize>>;
    /// Zero-allocation variant: caller-owned `canon` and `out` scratch (>= keys.len()).
    pub fn lookup_batch_u64_simd_into(&self, keys: &[u64], canon: &mut [u64], out: &mut [Option<usize>]);

    // Stats / debug
    pub fn stats(&self) -> IndexStats;
    pub fn print_detailed_stats(&self);
    pub fn gpu_export(&self) -> Option<GpuExport>;

    // Serialization (checksummed container, validated on load)
    pub fn to_bytes(&self) -> Result<Vec<u8>, IndexError>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IndexError>;
    pub fn write_to<W: std::io::Write>(&self, w: W) -> std::io::Result<()>;   // streaming
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()>;
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, IndexError>;
    // Section container (same payload; reserved for a future zero-copy loader)
    pub fn save_mmap<P: AsRef<Path>>(&self, path: P) -> Result<(), IndexError>;
    pub fn open_mmap<P: AsRef<Path>>(path: P) -> Result<Self, IndexError>;

    // deprecated aliases: get, get_str, get_u64, get_batch, has, exists, has_batch,
    // exists_batch, serialize, deserialize, lookup_u64_fast, range, get_all
}
```

### `IndexStats`

```rust
pub struct IndexStats {
    pub engine: &'static str,    // "mph" / "mph-empty"
    pub total_keys: usize,
    pub mph_memory: usize,       // pilots + remap + parts
    pub pgm_memory: usize,       // always 0
    pub total_memory: usize,     // + Bloom + fingerprints
}
```

### `IndexError`

```rust
#[non_exhaustive]
pub enum IndexError {
    DuplicateKey,                // exact, never probabilistic
    Unresolvable,                // pilot search gave up (change the seed)
    KeyNotFound,
    InvalidKey,
    CorruptData,                 // checksum or structural check failed
    Io(std::io::Error),
    Unsupported(&'static str),
    Mph(String),                 // other MPH failures
    Pgm(String),                 // other PGM failures
}
```

---

## `DynamicIndex` (LSM on MPH tiers)

Mutable key→stable-id store. Insert/delete supported; stable IDs survive flushes
and compactions. Every tier hit is verified against the stored key bytes, so a
tier never answers with another key's id (also with `lean_tiers`).

```rust
pub struct DynamicIndex { /* opaque */ }

impl DynamicIndex {
    pub fn new() -> Self;
    pub fn with_config(cfg: DynamicConfig) -> Self;

    pub fn insert(&mut self, key: Vec<u8>) -> StableId;      // existing key → its id
    pub fn delete(&mut self, key: &[u8]) -> Option<StableId>; // prior id, if any
    pub fn lookup(&self, key: &[u8]) -> Option<StableId>;

    /// buffer → new tier. On Err the buffer is untouched.
    pub fn flush(&mut self) -> Result<(), IndexError>;
    /// all tiers + buffer → one tier. On Err the index is unchanged.
    pub fn compact(&mut self) -> Result<(), IndexError>;

    pub fn len(&self) -> usize;                 // exact live count
    pub fn is_empty(&self) -> bool;
    pub fn tier_count(&self) -> usize;
    pub fn buffer_len(&self) -> usize;
    pub fn tombstone_count(&self) -> usize;
    pub fn memory_usage(&self) -> usize;

    /// Live entries + id counter + config; loads into one compacted tier.
    pub fn to_bytes(&self) -> Vec<u8>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IndexError>;
}

pub struct DynamicConfig {
    pub flush_threshold: usize,  // default 64K
    pub max_tiers: usize,        // default 8
    pub lean_tiers: bool,        // default false; memory only, answers stay exact
    pub parallel_build: bool,    // default true
}

pub type StableId = u32;
```

---

## `PgmIndex` (sorted u64 learned index)

Range queries + lookups on u64 keys. Linear-time segmentation; every key is
within `epsilon` positions of its prediction (`stats().max_error` is the exact
value; f32 rounding can add 1 at very large N).

```rust
pub struct PgmBuilder { /* opaque */ }

impl PgmBuilder {
    pub fn new() -> Self;                                   // epsilon 64
    pub fn with_epsilon(self, epsilon: u32) -> Self;
    pub fn with_bloom_filter(self, enabled: bool) -> Self;
    pub fn with_elias_fano(self, enabled: bool) -> Self;
    pub fn with_parallel(self, enabled: bool) -> Self;
    pub fn with_target_lookup_ns(self, ns: u32) -> Self;
    /// Sorts `keys`; duplicates → `Err(PgmError::DuplicateKeys)`; empty is fine.
    pub fn build(self, keys: Vec<u64>) -> Result<PgmIndex, PgmError>;
}

pub struct PgmIndex { /* opaque */ }

impl PgmIndex {
    pub fn build(keys: Vec<u64>, epsilon: u32) -> Result<Self, PgmError>;

    pub fn index(&self, key: u64) -> Result<usize, PgmError>;
    /// lower_bound(min)..upper_bound(max) — O(1), never materialized.
    pub fn range(&self, min_key: u64, max_key: u64) -> std::ops::Range<usize>;
    pub fn lower_bound(&self, target: u64) -> usize;
    pub fn upper_bound(&self, target: u64) -> usize;

    pub fn has_bloom(&self) -> bool;
    pub fn num_segments(&self) -> usize;
    /// Convert in-place Vec<u64> keys to Elias-Fano. Returns bytes saved
    /// (positive) or negative if EF would have been bigger.
    pub fn compact_keys(&mut self) -> isize;

    pub fn stats(&self) -> PgmStats;
    pub fn to_bytes(&self) -> Result<Vec<u8>, PgmError>;      // format v3; v2 files load
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PgmError>;
}

#[non_exhaustive]
pub enum PgmError { DuplicateKeys, UnsortedKeys /* deprecated */, EmptyKeys, KeyNotFound, CorruptData }

pub struct PgmStats {
    pub total_keys: usize,
    pub total_segments: usize,
    pub avg_segment_size: f64,
    pub max_error: u32,
    pub memory_usage: usize,
    pub epsilon: u32,
}
```

---

## `PgmIndexU128` (16-byte keys: UUID/SHA-128/IPv6)

Segments are fitted on `key - segment_min_key`, so random 128-bit keys get
hundreds of keys per segment (16 B/key total, i.e. the keys themselves).

```rust
pub struct PgmIndexU128 { /* opaque */ }

impl PgmIndexU128 {
    /// Sorts `keys`; duplicates → `Err(DuplicateKeys)`; empty is fine.
    pub fn build(keys: Vec<u128>, epsilon: u32) -> Result<Self, PgmU128Error>;
    pub fn build_from_bytes16(keys: &[[u8; 16]], epsilon: u32) -> Result<Self, PgmU128Error>;

    pub fn index(&self, key: u128) -> Result<usize, PgmU128Error>;
    pub fn index_bytes16(&self, key: &[u8; 16]) -> Result<usize, PgmU128Error>;
    pub fn range(&self, min_key: u128, max_key: u128) -> std::ops::Range<usize>;
    pub fn lower_bound(&self, target: u128) -> usize;
    pub fn upper_bound(&self, target: u128) -> usize;

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn segments_count(&self) -> usize;
    pub fn memory_usage(&self) -> usize;
    pub fn epsilon(&self) -> u32;

    pub fn to_bytes(&self) -> Vec<u8>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PgmU128Error>;
}

#[non_exhaustive]
pub enum PgmU128Error { DuplicateKeys, UnsortedKeys /* deprecated */, EmptyKeys /* deprecated */, KeyNotFound, CorruptData }
```

---

## `HybridIndex` (PGM-bucketed mini-MPHs)

Universal byte-key engine. PGM segments the hash space (locator only — the
hashes are not kept twice); each segment is a linear array, a MiniChd, or a full
PtrHash25 depending on size.

```rust
pub struct HybridBuilder { /* opaque */ }

impl HybridBuilder {
    pub fn new() -> Self;
    pub fn with_seed(self, seed: u64) -> Self;
    pub fn with_pgm_epsilon(self, epsilon: u32) -> Self;      // default 2048
    pub fn with_linear_threshold(self, n: usize) -> Self;     // default 64
    pub fn with_chd_threshold(self, n: usize) -> Self;        // default 4096
    pub fn with_parallel(self, enabled: bool) -> Self;
    /// Skip Bloom + inner PtrHash25 fingerprints (~35% less memory); foreign keys
    /// return arbitrary positions instead of `None`.
    pub fn with_lean(self, enabled: bool) -> Self;

    /// Empty input builds an always-miss index.
    pub fn build<K: AsRef<[u8]>>(self, keys: &[K]) -> Result<HybridIndex, HybridError>;
    /// SIMD-accelerated build path for u64 keys.
    pub fn build_from_u64(self, keys: &[u64]) -> Result<HybridIndex, HybridError>;
}

pub struct HybridIndex { /* opaque */ }

impl HybridIndex {
    pub fn empty(seed: u64) -> Self;
    pub fn lookup(&self, key: &[u8]) -> Option<u32>;
    pub fn lookup_u64(&self, key: u64) -> Option<u32>;
    pub fn lookup_hash(&self, hash: u64) -> Option<u32>;
    pub fn lookup_batch<K: AsRef<[u8]>>(&self, keys: &[K]) -> Vec<Option<u32>>;
    pub fn lookup_batch_u64_simd(&self, keys: &[u64]) -> Vec<Option<u32>>;
    pub fn lookup_batch_hashes(&self, hashes: &[u64]) -> Vec<Option<u32>>;

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn num_segments(&self) -> usize;
    pub fn memory_usage(&self) -> usize;
    pub fn seed(&self) -> u64;
    pub fn storage_stats(&self) -> HybridStorageStats;

    pub fn to_bytes(&self) -> Vec<u8>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HybridError>;
}

#[non_exhaustive]
pub enum HybridError { EmptyKeys /* deprecated */, HashCollision, Mph(String), Pgm(PgmError), CorruptData }

pub struct HybridStorageStats {
    pub linear_segments: usize,
    pub chd_segments: usize,
    pub mph_segments: usize,
    pub linear_keys: usize,
    pub chd_keys: usize,
    pub mph_keys: usize,
    pub total_segments: usize,
}
```

---

## `EliasFano` (compact sorted u64 storage)

Standalone — also used internally by `PgmIndex::compact_keys`.

```rust
pub struct EliasFano { /* opaque */ }

impl EliasFano {
    pub fn from_sorted(keys: &[u64]) -> Option<Self>;
    pub fn get(&self, i: usize) -> u64;
    pub fn materialize_range(&self, from: usize, count: usize, out: &mut Vec<u64>);
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn universe(&self) -> u64;
    pub fn memory_usage(&self) -> usize;
    pub fn write_to(&self, out: &mut Vec<u8>);
    pub fn read_from(bytes: &[u8], pos: &mut usize) -> Option<Self>;   // validated
}
```

---

## `HotTierIndex`, `DynamicHotTier`, `SpaceSaving`

```rust
pub struct HotTierIndex { /* opaque */ }

impl HotTierIndex {
    /// keys[i] → indices[i]; Bloom + u16 fingerprints, foreign keys rejected.
    pub fn build_from_u64(keys: &[u64], indices: &[u32], seed: u64) -> Option<Self>;
    pub fn lookup_u64(&self, key: u64) -> Option<u32>;
    pub fn memory_usage(&self) -> usize;
    pub fn write_to(&self, out: &mut Vec<u8>);
    pub fn read_from(bytes: &[u8], pos: &mut usize) -> Option<Self>;
}

pub struct SpaceSaving { /* opaque */ }   // O(log K) per observation

impl SpaceSaving {
    pub fn new(capacity: usize) -> Self;
    pub fn observe(&mut self, key: u64);
    pub fn top_k(&self, k: usize) -> Vec<(u64, u64)>;
    pub fn take_top_k_and_reset(&mut self, k: usize) -> Vec<(u64, u64)>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn total_observed(&self) -> u64;
}

pub struct DynamicHotTier { /* opaque; share via Arc */ }

impl DynamicHotTier {
    pub fn new(initial: Option<HotTierIndex>, top_k_capacity: usize, rebuild_every: u64) -> Self;
    /// Read-locks the index; try-locks the tracker (contended observations are dropped).
    pub fn lookup_u64(&self, key: u64) -> Option<u32>;
    pub fn observe(&self, key: u64);
    pub fn dropped_observations(&self) -> u64;
    pub fn should_rebuild(&self) -> bool;
    pub fn take_top_k(&self, k: usize) -> Vec<(u64, u64)>;
    pub fn install(&self, new_index: HotTierIndex);
    pub fn current_memory(&self) -> usize;
}
```

---

## `GpuExport` / `GpuPart` / `BloomExport` (off-host lookup)

POD snapshot of a built `Index` for GPU/SIMD pipeline consumers.

```rust
pub struct GpuExport {
    pub prehash_seed: u64,
    pub mph_salt: u64,          // base-hash salt (multi-part) / part 0 salt (single part)
    pub num_buckets: u32,       // == pilots.len()
    pub num_slots: u64,         // size of the output range (== key count for 0.7 builds)
    pub prerotate: u8,
    pub pilots: Vec<u8>,
    pub remap: Vec<u32>,        // tail redirection table, shared by all parts
    pub bloom: Option<BloomExport>,
    pub fingerprints: Option<Vec<u16>>,   // exactly num_slots entries
    pub part_salt: u64,
    pub parts: Vec<GpuPart>,    // >= 1 entry
}

/// Lookup inside part p (rotated = key.rotate_left(prerotate)):
///   p      = parts.len() > 1 ? mulhi((rotated ^ part_salt) * 0x9E3779B97F4A7C15, parts.len()) : 0
///   base   = mix64(rotated ^ mph_salt)             // single part: mix64(rotated ^ parts[0].salt)
///   h1     = parts.len() > 1 ? (base ^ parts[p].salt) * 0xBF58476D1CE4E5B9 : base
///   h2     = rotl(h1, 23) ^ 0xA24B1F6FDA392B31
///   bucket = parts[p].bucket_off + bucket_for(h1, parts[p].num_buckets, parts[p].large_buckets)
///   local  = slot_for(h2, pilots[bucket], parts[p].num_slots)
///   if local >= parts[p].num_keys { local = remap[parts[p].remap_off + local - parts[p].num_keys] }
///   slot   = parts[p].slot_off + local
pub struct GpuPart {
    pub slot_off: u64,
    pub salt: u64,
    pub bucket_off: u32,
    pub num_keys: u32,         // output positions of this part (== num_slots for 0.6 files)
    pub num_slots: u32,        // searched slots
    pub num_buckets: u32,
    pub large_buckets: u32,    // floor(num_buckets * 0.30)
    pub remap_off: u32,
}

pub struct BloomExport {
    pub bit_shift: u32,       // 26 (6-bit lane index); 27 for filters from 0.6 files
    pub blocks: usize,        // multiple of 256
    pub words: Vec<u64>,      // length = blocks * 8
    //   block = ((hash >> 32) * blocks) >> 32
    //   bit_w = (((hash & 0xFFFFFFFF) * SALT[w]) >> bit_shift) & 0x3F
}
```

---

## Serialized formats

Every `to_bytes` form is `[4]["KIRA"][format u16][kind u8][0][body][checksum u64]`
with `kind` 1 = `Index`, 2 = `HybridIndex`, 3 = `PgmIndexU128`, 4 =
`DynamicIndex`; `PgmIndex` keeps its own versioned layout. Loading verifies the
checksum, then every structural invariant (array lengths, offsets, remap targets,
sortedness) before any unchecked lookup can run. Legacy `Index` tags 0/2/3 and
PGM format v2 still load, without a checksum.

---

## Deprecated

`BackendKind`, `BackendBuildConfig`, `BuildProfile` and the `MphBackend` trait
are remnants of the multi-algorithm era (PtrHash25 has been the only backend
since 0.5). They remain exported for source compatibility only.
