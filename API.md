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
    PgmIndex, PgmBuilder, PgmStats,
    PgmIndexU128, PgmU128Error,
    // Hybrid (PGM + per-segment MPH)
    HybridIndex, HybridBuilder, HybridError, HybridStorageStats,
    // Compact key storage
    EliasFano,
    // Workload-aware caching
    DynamicHotTier, SpaceSaving,
    // GPU export (PtrHash25 snapshot for off-host lookup)
    GpuExport, BloomExport,
    // Backend config
    BackendKind, BackendBuildConfig, BuildProfile, MphBackend,
};
```

---

## `Index` (PtrHash25 static MPH)

Top-level engine for any byte key. Returns a stable `usize` id in `[0..n)` (or
`[0..1.10·n)` for the near-minimal u8-pilot variant).

### `IndexBuilder`

```rust
pub struct IndexBuilder { /* opaque */ }

impl IndexBuilder {
    pub fn new() -> Self;
    pub fn with_config(self, config: IndexConfig) -> Self;
    pub fn with_mph_config(self, mph_config: ptrhash25::BuildConfig) -> Self;
    pub fn with_pgm_epsilon(self, epsilon: u32) -> Self;
    pub fn with_backend(self, backend: BackendKind) -> Self;
    pub fn with_hot_fraction(self, hot_fraction: f32) -> Self;
    pub fn with_parallel_build(self, enabled: bool) -> Self;
    pub fn with_build_fast_profile(self, enabled: bool) -> Self;
    pub fn auto_detect_numeric(self, enabled: bool) -> Self;

    /// Drop Bloom + fingerprints. -87% memory, +10% lookup speed.
    /// Returns garbage positions for keys NOT in the build set — use only
    /// for closed-world workloads (preloaded dictionary, vocab, etc.).
    pub fn with_lean_mph(self, enabled: bool) -> Self;

    pub fn with_pgm_bloom(self, enabled: bool) -> Self;
    pub fn with_pgm_elias_fano(self, enabled: bool) -> Self;
    pub fn with_pgm_target_lookup_ns(self, ns: u32) -> Self;

    pub fn build_index<K: AsRef<[u8]>>(self, keys: Vec<K>) -> Result<Index, IndexError>;
}
```

### `IndexConfig`

```rust
pub struct IndexConfig {
    pub mph_config: ptrhash25::BuildConfig,
    pub pgm_epsilon: u32,
    pub auto_detect_numeric: bool,
    pub backend: BackendKind,
    pub hot_fraction: f32,
    pub enable_parallel_build: bool,
    pub build_fast_profile: bool,
    pub pgm_enable_bloom: bool,
    pub pgm_enable_elias_fano: bool,
    pub pgm_target_lookup_ns: Option<u32>,
    pub lean_mph: bool,
}
```

Default: `lean_mph=false`, `auto_detect_numeric=false`, `enable_parallel_build=true`.

### `Index` lookups

```rust
impl Index {
    // Point lookup variants
    pub fn lookup(&self, key: &[u8]) -> Result<usize, IndexError>;
    pub fn get(&self, key: &[u8]) -> Result<usize, IndexError>;
    pub fn lookup_str(&self, key: &str) -> Result<usize, IndexError>;
    pub fn get_str(&self, key: &str) -> Result<usize, IndexError>;
    pub fn lookup_u64(&self, key: u64) -> Result<usize, IndexError>;
    pub fn get_u64(&self, key: u64) -> Result<usize, IndexError>;

    /// Skip enum dispatch for PtrHash25-backed engines. Returns None if the
    /// backend can't be specialized; callers fall back to `lookup_u64`.
    pub fn lookup_u64_fast(&self, key: u64) -> Option<Result<usize, IndexError>>;

    // Membership
    pub fn contains(&self, key: &[u8]) -> bool;
    pub fn has(&self, key: &[u8]) -> bool;
    pub fn exists(&self, key: &[u8]) -> bool;

    // Batched (cache-prefetch pipelines)
    pub fn lookup_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>>;
    pub fn get_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>>;
    pub fn lookup_batch_pipelined(&self, keys: &[&[u8]]) -> Vec<Option<usize>>;
    pub fn lookup_batch_u64_simd(&self, keys: &[u64]) -> Vec<Option<usize>>;
    pub fn contains_batch(&self, keys: &[&[u8]]) -> Vec<bool>;
    pub fn has_batch(&self, keys: &[&[u8]]) -> Vec<bool>;
    pub fn exists_batch(&self, keys: &[&[u8]]) -> Vec<bool>;

    // Range (no-op — use PgmIndex for u64 ranges; HybridIndex for byte ranges)
    pub fn range(&self, _min_key: u64, _max_key: u64) -> Vec<usize>;
    pub fn get_all(&self, min_key: u64, max_key: u64) -> Vec<usize>;

    // Stats / debug
    pub fn len(&self) -> usize;
    pub fn stats(&self) -> IndexStats;
    pub fn print_detailed_stats(&self);

    // Serialization
    pub fn to_bytes(&self) -> Result<Vec<u8>, IndexError>;
    pub fn serialize(&self) -> Result<Vec<u8>, IndexError>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IndexError>;
    pub fn deserialize(bytes: &[u8]) -> Result<Self, IndexError>;

    // mmap zero-copy load
    pub fn save_mmap<P: AsRef<Path>>(&self, path: P) -> Result<(), IndexError>;
    pub fn open_mmap<P: AsRef<Path>>(path: P) -> Result<Self, IndexError>;
}
```

### `IndexStats`

```rust
pub struct IndexStats {
    pub engine: &'static str,    // "mph"
    pub total_keys: usize,
    pub mph_memory: usize,
    pub pgm_memory: usize,       // 0 for the MPH engine
    pub total_memory: usize,
}
```

### `IndexError`

```rust
pub enum IndexError {
    Mph(String),
    Pgm(String),
    KeyNotFound,
    InvalidKey,
    CorruptData,
}
```

---

## `DynamicIndex` (LSM on MPH tiers)

Mutable key→stable-id store. Insert/delete supported; stable IDs survive flushes
and compactions.

```rust
pub struct DynamicIndex { /* opaque */ }

impl DynamicIndex {
    pub fn new() -> Self;
    pub fn with_config(cfg: DynamicConfig) -> Self;

    pub fn insert(&mut self, key: Vec<u8>) -> StableId;
    pub fn delete(&mut self, key: &[u8]) -> Option<StableId>;
    pub fn lookup(&self, key: &[u8]) -> Option<StableId>;

    pub fn flush(&mut self);     // buffer → new tier
    pub fn compact(&mut self);   // all tiers + buffer → one tier

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn tier_count(&self) -> usize;
    pub fn buffer_len(&self) -> usize;
    pub fn tombstone_count(&self) -> usize;
    pub fn memory_usage(&self) -> usize;
}

pub struct DynamicConfig {
    pub flush_threshold: usize,  // default 64K
    pub max_tiers: usize,        // default 8
    pub lean_tiers: bool,        // default false
    pub parallel_build: bool,    // default true
}

pub type StableId = u32;
```

---

## `PgmIndex` (sorted u64 learned index)

Range queries + lookups on sorted u64 keys.

```rust
pub struct PgmBuilder { /* opaque */ }

impl PgmBuilder {
    pub fn new() -> Self;
    pub fn with_epsilon(self, epsilon: u32) -> Self;
    pub fn with_bloom_filter(self, enabled: bool) -> Self;
    pub fn with_elias_fano(self, enabled: bool) -> Self;
    pub fn with_parallel(self, enabled: bool) -> Self;
    pub fn with_target_lookup_ns(self, ns: u32) -> Self;
    pub fn build(self, keys: Vec<u64>) -> Result<PgmIndex, PgmError>;
}

pub struct PgmIndex { /* opaque */ }

impl PgmIndex {
    pub fn build(keys: Vec<u64>, epsilon: u32) -> Result<Self, PgmError>;

    pub fn index(&self, key: u64) -> Result<usize, PgmError>;
    pub fn range(&self, min_key: u64, max_key: u64) -> Vec<usize>;
    pub fn lower_bound(&self, target: u64) -> usize;
    pub fn upper_bound(&self, target: u64) -> usize;

    pub fn has_bloom(&self) -> bool;
    /// Convert in-place Vec<u64> keys to Elias-Fano. Returns bytes saved
    /// (positive) or negative if EF would have been bigger.
    pub fn compact_keys(&mut self) -> isize;

    pub fn stats(&self) -> PgmStats;
    pub fn to_bytes(&self) -> Result<Vec<u8>, PgmError>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PgmError>;
}

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

```rust
pub struct PgmIndexU128 { /* opaque */ }

impl PgmIndexU128 {
    pub fn build(keys: Vec<u128>, epsilon: u32) -> Result<Self, PgmU128Error>;
    pub fn build_from_bytes16(keys: &[[u8; 16]], epsilon: u32) -> Result<Self, PgmU128Error>;

    pub fn index(&self, key: u128) -> Result<usize, PgmU128Error>;
    pub fn index_bytes16(&self, key: &[u8; 16]) -> Result<usize, PgmU128Error>;
    pub fn range(&self, min_key: u128, max_key: u128) -> Vec<usize>;
    pub fn lower_bound(&self, target: u128) -> usize;
    pub fn upper_bound(&self, target: u128) -> usize;

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn segments_count(&self) -> usize;
    pub fn memory_usage(&self) -> usize;
    pub fn epsilon(&self) -> u32;
}
```

---

## `HybridIndex` (PGM-bucketed mini-MPHs)

Universal byte-key engine. PGM segments the hash space; each segment is either
a linear array, a MiniChd, or a full PtrHash25 depending on size.

```rust
pub struct HybridBuilder { /* opaque */ }

impl HybridBuilder {
    pub fn new() -> Self;
    pub fn with_seed(self, seed: u64) -> Self;
    pub fn with_pgm_epsilon(self, epsilon: u32) -> Self;
    pub fn with_linear_threshold(self, n: usize) -> Self;
    pub fn with_chd_threshold(self, n: usize) -> Self;
    pub fn with_parallel(self, enabled: bool) -> Self;
    /// Skip Bloom + inner PtrHash25 fingerprints. -25% memory; foreign keys
    /// return garbage instead of `None`.
    pub fn with_lean(self, enabled: bool) -> Self;

    pub fn build<K: AsRef<[u8]>>(self, keys: &[K]) -> Result<HybridIndex, HybridError>;
    /// SIMD-accelerated build path for u64 keys.
    pub fn build_from_u64(self, keys: &[u64]) -> Result<HybridIndex, HybridError>;
}

pub struct HybridIndex { /* opaque */ }

impl HybridIndex {
    pub fn lookup(&self, key: &[u8]) -> Option<u32>;
    pub fn lookup_u64(&self, key: u64) -> Option<u32>;
    pub fn lookup_hash(&self, hash: u64) -> Option<u32>;
    pub fn lookup_batch<K: AsRef<[u8]>>(&self, keys: &[K]) -> Vec<Option<u32>>;

    /// SIMD batch lookup with 16-deep prefetch chain. 2-3× faster than scalar.
    pub fn lookup_batch_u64_simd(&self, keys: &[u64]) -> Vec<Option<u32>>;
    pub fn lookup_batch_hashes(&self, hashes: &[u64]) -> Vec<Option<u32>>;

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn num_segments(&self) -> usize;
    pub fn memory_usage(&self) -> usize;
    pub fn seed(&self) -> u64;
    pub fn storage_stats(&self) -> HybridStorageStats;
}

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
    pub fn read_from(bytes: &[u8], pos: &mut usize) -> Option<Self>;
}
```

---

## `DynamicHotTier` + `SpaceSaving`

```rust
pub struct SpaceSaving { /* opaque */ }

impl SpaceSaving {
    pub fn new(capacity: usize) -> Self;
    pub fn observe(&mut self, key: u64);
    pub fn top_k(&self, k: usize) -> Vec<(u64, u64)>;
    pub fn take_top_k_and_reset(&mut self, k: usize) -> Vec<(u64, u64)>;
    pub fn len(&self) -> usize;
    pub fn total_observed(&self) -> u64;
}

pub struct DynamicHotTier { /* opaque */ }

impl DynamicHotTier {
    pub fn new(initial: Option<HotTierIndex>, top_k_capacity: usize, rebuild_every: u64) -> Self;
    pub fn lookup_u64(&self, key: u64) -> Option<u32>;
    pub fn should_rebuild(&self) -> bool;
    pub fn take_top_k(&self, k: usize) -> Vec<(u64, u64)>;
    pub fn install(&self, new_index: HotTierIndex);
    pub fn current_memory(&self) -> usize;
}
```

---

## `GpuExport` / `BloomExport` (off-host lookup)

POD snapshot of a built `Index` for GPU/SIMD pipeline consumers.

```rust
pub struct GpuExport {
    pub prehash_seed: u64,
    pub mph_salt: u64,
    pub num_buckets: u32,
    pub num_slots: u64,
    pub prerotate: u8,
    pub pilots: Vec<u8>,
    pub bloom: Option<BloomExport>,
    pub fingerprints: Option<Vec<u16>>,
}

pub struct BloomExport {
    pub blocks: usize,        // power-of-two
    pub words: Vec<u64>,      // length = blocks * 8
}
```

---

## `BackendKind` + `BackendBuildConfig`

```rust
pub enum BackendKind {
    PtrHash25,        // currently the only variant
}

pub enum BuildProfile {
    Balanced,
    Fast,
}

pub struct BackendBuildConfig {
    pub backend: BackendKind,
    pub enable_parallel_build: bool,
    pub seed: u64,
    pub gamma: f64,
    pub rehash_limit: u32,
    pub build_profile: BuildProfile,
}
```

`MphBackend` trait is exported but is currently a 1-impl marker (room for new
algorithms without breaking callers).
