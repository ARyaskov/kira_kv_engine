# Kira KV Engine — Public API

The crate exposes **six engines** plus shared primitives. All are accessible
from the crate root:

```rust
use kira_kv_engine::{
    // Static MPH (PtrHash25)
    Index, IndexBuilder, IndexConfig, IndexError, IndexStats, MphConfig,
    // One-pass, bounded-working-memory u64 MPH construction
    MphExternalBuilder, MphExternalBuildError, MphIndex,
    // Dynamic MPH (LSM-style)
    DynamicIndex, DynamicConfig, DynamicInsertError, CompactionPolicy, StableId,
    // PGM learned index
    PgmIndex, PgmBuilder, PgmStats, PgmError,
    PgmIndexU128, PgmU128Error,
    // Hybrid (PGM + per-segment MPH)
    HybridIndex, HybridBuilder, HybridError, HybridStorageStats,
    // Compact key storage
    EliasFano,
    // Workload-aware caching
    HotTierIndex, HotTierError, DynamicHotTier, SpaceSaving, MembershipMode,
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
    pub fn with_mph_config(self, mph_config: MphConfig) -> Self;
    pub fn with_parallel_build(self, enabled: bool) -> Self;
    pub fn with_bloom_filter(self, enabled: bool) -> Self;
    pub fn with_fingerprint_bits(self, width: FingerprintBits) -> Self;

    /// Drop Bloom + fingerprints: about 0.42 B/key; speed depends on workload.
    /// Foreign keys return arbitrary in-range positions — closed-world only.
    pub fn with_lean_mph(self, enabled: bool) -> Self;

    // deprecated, no effect: with_pgm_epsilon, with_backend, with_hot_fraction,
    // with_build_fast_profile, auto_detect_numeric, with_pgm_bloom,
    // with_pgm_elias_fano, with_pgm_target_lookup_ns

    /// Owned keys: same build as `build_index_ref`, then `keys` is dropped in
    /// parallel on the build pool when parallel building is enabled.
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
    pub mph_config: MphConfig,              // lambda (keys/bucket, 3.0), alpha (load, 0.98),
                                              // max_rehash, seed, with_fingerprints, use_aes_hash
    pub enable_parallel_build: bool,
    pub lean_mph: bool,
    pub bloom_filter: bool,                  // default true
    pub fingerprint_bits: FingerprintBits,    // Bits16 (default), Bits24, Bits32
    // deprecated, no effect: pgm_epsilon, auto_detect_numeric, backend, hot_fraction,
    // build_fast_profile, pgm_enable_bloom, pgm_enable_elias_fano, pgm_target_lookup_ns
}
```

Default: `lean_mph=false`, `enable_parallel_build=true` (with the `parallel` feature).

`MphConfig` is a public alias of the existing MPH `BuildConfig`; existing types
and builder signatures remain compatible. `lambda` must be finite and positive,
and `alpha` finite and in `(0, 1]`. Invalid options return `InvalidConfig` before
hashing, including on empty input; unrepresentable table sizes return
`CapacityExceeded` before narrowing or table allocation. These checks do not
guarantee that the host has enough free memory for a representable table.

`with_parallel_build(false)` keeps hashing, partitioning, Bloom construction,
pilot search and owned-key destruction on the caller thread. The equivalent
`with_parallel(false)` options on PGM and Hybrid also suppress nested Rayon work.
These builds do not initialize either the dedicated or global Rayon pool.

### `Index` lookups

```rust
impl Index {
    pub fn empty() -> Self;
    pub fn is_empty(&self) -> bool;
    pub fn len(&self) -> usize;
    /// Exclusive upper bound of every lookup result (== len() for 0.7 builds).
    pub fn slot_capacity(&self) -> usize;

    // Point lookups: configured checks may reject a key; positive candidates
    // remain probabilistic unless verified against authoritative full keys.
    pub fn lookup(&self, key: &[u8]) -> Result<usize, IndexError>;
    pub fn lookup_str(&self, key: &str) -> Result<usize, IndexError>;
    pub fn lookup_u64(&self, key: u64) -> Result<usize, IndexError>;

    // Membership: Bloom answer — definite `false`, probabilistic `true`.
    // Lean indexes have no filter and always answer `true`.
    pub fn supports_negative_lookups(&self) -> bool;
    pub fn contains(&self, key: &[u8]) -> bool;
    pub fn contains_batch(&self, keys: &[&[u8]]) -> Vec<bool>;
    pub fn membership_mode(&self) -> MembershipMode;
    pub fn may_contain(&self, key: &[u8]) -> bool;
    pub fn may_contain_batch(&self, keys: &[&[u8]]) -> Vec<bool>;
    pub fn lookup_verified(&self, key: &[u8], verify: impl FnOnce(usize, &[u8]) -> bool)
        -> Result<usize, IndexError>;

    // Batched (SIMD hashing and a three-stage pilot/Bloom/fingerprint pipeline)
    pub fn lookup_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>>;
    pub fn lookup_batch_pipelined(&self, keys: &[&[u8]]) -> Vec<Option<usize>>;
    /// Zero-allocation byte pipeline; out.len() must be >= keys.len().
    pub fn lookup_batch_pipelined_into(&self, keys: &[&[u8]], out: &mut [Option<usize>]);
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
    // Aligned, checksummed section container; Windows safe opener maps tables.
    pub fn save_mmap<P: AsRef<Path>>(&self, path: P) -> Result<(), IndexError>;
    pub fn open_mmap<P: AsRef<Path>>(path: P) -> Result<Self, IndexError>;
    /// Caller guarantees the backing file stays immutable while the index lives.
    pub unsafe fn open_mmap_unchecked<P: AsRef<Path>>(path: P) -> Result<Self, IndexError>;
    pub fn is_memory_mapped(&self) -> bool;

    // deprecated aliases: get, get_str, get_u64, get_batch, has, exists, has_batch,
    // exists_batch, serialize, deserialize, lookup_u64_fast, range, get_all
}
```

`lookup_batch_pipelined_into` preserves query order and overwrites every element
of `out[..keys.len()]`, including misses. Any extra output suffix is unchanged;
an undersized output slice panics. It uses bounded 64-query stack scratch. The
existing allocating methods retain their signatures and behavior.

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
    InvalidConfig(&'static str), // invalid finite/range configuration
    CapacityExceeded(&'static str), // table exceeds representable addressing
    CorruptData,                 // checksum or structural check failed
    Io(std::io::Error),
    Unsupported(&'static str),
    Mph(String),                 // other MPH failures
    Pgm(String),                 // other PGM failures
}
```

---

## `MphExternalBuilder` / `MphIndex` (streamed u64 MPH)

An additive low-level construction option for a one-pass iterator of unique
`u64` keys. Byte-key callers must canonicalize/prehash their keys themselves;
this API does not perform `Index` byte hashing.

```rust
pub struct MphExternalBuilder { /* opaque */ }
pub struct MphIndex { /* opaque */ }

impl MphExternalBuilder {
    pub fn new() -> Self; // default auxiliary budget: 8 MiB
    pub fn with_config(self, config: MphConfig) -> Self;
    pub fn with_memory_budget_bytes(self, bytes: usize) -> Self;
    pub fn build<I: IntoIterator<Item = u64>, P: AsRef<Path>>(
        self, keys: I, expected_len: usize, scratch_dir: P,
    ) -> Result<MphIndex, MphExternalBuildError>;
}

impl MphIndex {
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn index_u64(&self, key: u64) -> u32;
    pub fn lookup_u64(&self, key: u64) -> Option<u32>;
    pub fn memory_usage(&self) -> usize;
    pub fn to_bytes(&self) -> Vec<u8>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MphExternalBuildError>;
}
```

The iterator must yield exactly `expected_len` keys; zero keys return
`EmptyInput`. The budget bounds auxiliary key/sort/scratch arrays, I/O buffers
and temporary bookkeeping. It excludes final resident index tables and storage
already owned by the caller or iterator, so it is not a total-process RAM cap.
Construction spools temporary files beneath `scratch_dir`, partitions with
fanout at most 64, and builds parts sequentially. For `N` keys and `P` parts,
partition I/O is `O(N * (1 + ceil(log_64 P)))` records; resident output is `O(N)`.
A skewed part that exceeds the budget returns
`MphExternalBuildError::BudgetExceeded { required, budget }` before loading its
key array. This error enum is non-exhaustive; other typed errors are `InvalidConfig`, `CapacityExceeded`,
`KeyCountMismatch { expected, observed }`, `EmptyInput`, `Build`, `Io` and
`CorruptData`. Owned temporary files are removed before successful return;
failure/unwind cleanup is best effort, and existing caller files are untouched.

`index_u64` is closed-world: foreign keys receive an arbitrary in-range slot.
`lookup_u64` applies the configured inner fp8 check; positive answers remain
probabilistic, and disabling fingerprints removes membership checking.
Prerotation is fixed at zero, so mappings can differ from the ordinary builder.
Serialization uses the existing raw PtrHash25 payload with structural validation
and full input consumption, **without an outer checksum**. It is distinct from
the checksummed `Index` container. Existing `IndexConfig` fields and formats are
unchanged.

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
    pub fn try_insert(&mut self, key: Vec<u8>) -> Result<StableId, DynamicInsertError>;
    pub fn last_maintenance_error(&self) -> Option<&str>;
    pub fn delete(&mut self, key: &[u8]) -> Option<StableId>; // prior id, if any
    pub fn lookup(&self, key: &[u8]) -> Option<StableId>;

    /// Buffer to new tier, then schedule size-tiered maintenance; bindings survive errors.
    pub fn flush(&mut self) -> Result<(), IndexError>;
    /// All tiers and buffer to one tier; bindings survive errors.
    pub fn compact(&mut self) -> Result<(), IndexError>;
    pub fn finish_background_compaction(&mut self) -> Result<(), IndexError>;
    pub fn set_background_compaction(&mut self, enabled: bool) -> Result<(), IndexError>;
    pub fn set_compaction_ratio(&mut self, ratio: usize) -> Result<(), IndexError>;
    pub fn set_compaction_policy(&mut self, policy: CompactionPolicy) -> Result<(), IndexError>;
    pub fn compaction_policy(&self) -> CompactionPolicy;
    pub fn tier_limit(&self) -> usize;
    pub fn compaction_stats(&self) -> (u64, u64); // automatic merges and input keys rebuilt

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
    pub max_tiers: usize,        // default 8; hard cap for Full/SizeTiered
    pub lean_tiers: bool,        // default false; memory only, answers stay exact
    pub parallel_build: bool,    // default true
}

pub type StableId = u32;

pub enum CompactionPolicy { Full, SizeTiered, Geometric }

#[non_exhaustive]
pub enum DynamicInsertError {
    KeyTooLong,
    IdExhausted,
    Backpressure(String),
    Maintenance { committed_id: StableId, source: IndexError },
}

impl DynamicInsertError {
    pub fn committed_id(&self) -> Option<StableId>;
}
```

`try_insert` checks the maximum key length (`u32::MAX` bytes), ID exhaustion and maintenance backpressure before changing
bindings, including revival of a deleted key. A `Maintenance` error means the key
was inserted and its ID remains readable; retry `flush` or `compact`, not the key.
After a maintenance failure new writes are blocked until a successful explicit
retry. Existing-key inserts still return their ID and do not rewrite a tier.
The original `insert` signature is unchanged: it returns a committed ID after a
maintenance failure and exposes its diagnostic through `last_maintenance_error`;
an oversized key, an uncommitted blocked write or exhausted ID space panics before mutation.

---

## `PgmIndex` (sorted u64 learned index)

Range queries + lookups on u64 keys. Linear-time segmentation; new builds keep
every key within saturating `epsilon + 1` positions of its integer prediction.
`stats().max_error` reports the measured value. Loaded absolute models retain
their original, potentially larger error budget.

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
    pub fn to_bytes(&self) -> Result<Vec<u8>, PgmError>;      // checksummed; raw v2/v3/v4 files load
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PgmError>;
}

#[non_exhaustive]
pub enum PgmError { DuplicateKeys, UnsortedKeys /* deprecated */, EmptyKeys, KeyNotFound, TooManyKeys, CorruptData }

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
    /// New builds compare the original u64 key and reject foreign keys exactly.
    pub fn build_from_u64(keys: &[u64], indices: &[u32], seed: u64) -> Option<Self>;
    pub fn try_build_from_u64(keys: &[u64], indices: &[u32], seed: u64)
        -> Result<Self, HotTierError>;
    pub fn membership_mode(&self) -> MembershipMode;
    pub fn lookup_u64(&self, key: u64) -> Option<u32>;
    pub fn lookup_exact_u64(&self, key: u64) -> Option<u32>;
    pub fn memory_usage(&self) -> usize;
    pub fn write_to(&self, out: &mut Vec<u8>);
    pub fn read_from(bytes: &[u8], pos: &mut usize) -> Option<Self>;
}

pub struct SpaceSaving { /* opaque */ }   // expected O(log K) per observation

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

New hot tiers retain the original u64 in MPH-slot order, adding 8 bytes per slot.
The checked constructor rejects mismatched lengths, empty/duplicate key sets and
the reserved `u32::MAX` position. The original `Option` constructor remains.
Legacy hot payloads contain only fingerprints: they load as `Probabilistic` and
preserve their old `lookup_u64` behavior and serialized bytes. Their
`lookup_exact_u64` returns `None`; `DynamicHotTier` uses this conservative path so
an unverifiable legacy cache hit falls back to the backing index.

Space-Saving uses backward-shift deletion in its probe table. An observation is
expected `O(log K)` with ordinary hashing and worst-case `O(K + log K)` under
collisions, rather than repeated quadratic reinsertion of a probe cluster.
`top_k(k)` selects in `O(K + k log k)` time with `O(K)` scratch, retaining the
count-descending, key-ascending order. A zero request returns without allocation.
`DynamicHotTier::take_top_k` allocates a replacement tracker before locking,
swaps trackers and clears the old rebuild hint under the mutex, then processes
and drops the detached snapshot after releasing the lock. Observations after
the swap belong to the next epoch and can set its rebuild hint independently.
Extraction temporarily retains an additional `O(K)` tracker. Frequency bounds
apply to accepted observations; contention drops remain counted and can bias
the observed stream. This tracker has no persisted wire representation.

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
    pub blocks: usize,        // positive; tiny builds need not be a multiple of 256
    pub words: Vec<u64>,      // length = blocks * 8
    //   block = ((hash >> 32) * blocks) >> 32
    //   bit_w = (((hash & 0xFFFFFFFF) * SALT[w]) >> bit_shift) & 0x3F
}
```

Bloom consumers must accept every positive block count. New tiny filters use
the required number of complete 64-byte blocks (minimum one), while large
filters retain the existing geometry rounded to 256-block groups. Existing
serialized filters retain their stored block count and bit shift on load and
resave; fresh tiny builds can therefore differ in size and bytes from an older
build of the same keys without changing the wire interpretation or lookup
positions.

---

## Serialized formats

The checksummed container is `[tag]["KIRA"][format u16][kind u8][0][body][checksum u64]`.
New `Index` and `HybridIndex` builds use tag 4 and kinds 6 and 7 respectively;
legacy kinds 1 and 2 retain their original byte-key hash when loaded or saved
again. Kinds 3 and 4 identify `PgmIndexU128` and `DynamicIndex`.
Standalone `PgmIndex` uses tag 5 and kind 5 around its raw payload. New models
use raw v6: one relative f64 slope replaces the absolute f32 slope/intercept
pair, preserving eight numeric bytes per segment. Key offsets are subtracted as
integers before conversion; prediction adds the integer segment start. The
stored integer prediction error is measured and bounded by `epsilon + 1`
(saturating at `u32::MAX`). Input is limited to `u32::MAX` keys and larger builds
return `PgmError::TooManyKeys`. Raw versions 2/3/4 and old tag-5 containers remain
readable; loaded absolute models retain raw v4 on resave, including locator-only
models inside Hybrid. New Hybrid payloads embed raw v6. Section metadata byte 5
is 0 for legacy f32 arrays and 1 for relative f64 slopes with empty intercepts.
Legacy `Index` tags 0/2/3 also remain readable.
Legacy raw formats have no checksum; new containers verify the checksum before
validating lengths, contiguous part geometry, remap targets and sortedness.
Stored PGM keys additionally verify segment bounds, model errors and Bloom
containment. Dynamic loads reject repeated stable IDs and allocate their write
buffer on demand rather than reserving the persisted flush threshold.

Elias-Fano retains its ordinary u32-sample byte layout. A reserved zero count
followed by `KIRAEF02` introduces the wide u64-sample extension only when needed.
Loading rebuilds a derived select directory: groups of 64 ranks either span
fewer than 2048 high bits or store their positions directly. Random access visits
at most 33 high words and clears at most 63 set bits; range decoding is `O(k + 1)`.
On 64-bit targets the directory costs 16 bytes per group, plus sparse positions,
versus the old 4-byte samples. Failed reads leave the caller's cursor unchanged.
PGM compressed queries use at most 24 stack keys for small windows and binary
search for larger ones: `O(log S + log W)` time, `O(1)` extra space and no heap
allocation, for `S` segments and window width `W`. Fitting and measuring models
are linear passes; sorting input remains `O(n log n)` in the worst case.

The measured PGM/Elias-Fano speed and memory comparison is in
[`performance/PGM_EF_REPORT.md`](performance/PGM_EF_REPORT.md).

New aligned mmap metadata includes the canonical hash version (88 bytes);
the previous 80-byte metadata remains readable with the legacy hash.

New hot tiers use a length-bounded, checksummed frame starting with `KIRA_HOT`
and version word 1, followed by the legacy payload and original slot keys.
That second word is an invalid legacy Bloom length, making detection unambiguous.
Legacy hot payloads are still readable and reserialize in their original layout.

---

## Deprecated

`BackendKind`, `BackendBuildConfig`, `BuildProfile` and the `MphBackend` trait
are remnants of the multi-algorithm era (PtrHash25 has been the only backend
since 0.5). They remain exported for source compatibility only.

## Membership modes and mapped file lifetime

`FingerprintBits::{Bits16, Bits24, Bits32}` selects 2, 3 or 4 bytes per slot.
Bloom and fingerprint width are independent; lean mode disables both. Without
Bloom, `contains` checks fingerprints. Membership stays probabilistic in every
non-lean mode; a wider fingerprint reduces false acceptance. `GpuExport` carries
`fingerprint_bits` and packed `fingerprint_bytes`; the older `fingerprints`
field is populated only for fp16.

`MembershipMode::{ClosedWorld, Probabilistic, Exact}` makes these guarantees
explicit. For compatibility, `contains` still returns the Bloom-only answer when
Bloom exists; `may_contain` uses all the checks used by `lookup`. Neither makes
a nonempty static Index exact. `lookup_verified` adds a full-key comparison
against authoritative caller-owned storage, including for lean indexes.

The aligned mmap container has a checksum covering its header and sections.
Windows `open_mmap` retains a read handle that denies write/delete sharing.
Safe opens on other platforms retain an owned byte buffer; immutable-file
applications can use `unsafe open_mmap_unchecked` for OS-backed mapping there.
Big-endian and unaligned tables fall back to owned, endian-converted arrays.
Old payload-based mmap containers remain readable through the copying decoder.

`Index::save_mmap` streams borrowed tables and updates the checksum during the
same output pass. For `M` bytes and `S` sections it takes `O(M)` time and
`O(S + fixed buffers)` auxiliary space, without rereading the output. Aligned
section order, padding, fingerprints, canonical hash metadata and checksums
retain their previous bytes when resaving a loaded index; the rare map fallback streams its nested sealed
payload too. The existing `MmapIndexWriter` API continues to own the `Vec`
sections supplied by callers, while its finalization also avoids the reread.

Hybrid builds and loads derive a radix directory from existing PGM segment
bounds. It is enabled for at least 32 segments, uses four to eight prefix buckets
per segment until the 65,536-bucket cap, and stores one u32 boundary per bucket
plus a sentinel (at most 262,148 heap bytes). One-candidate prefixes route
directly; intersecting prefixes use local binary search with exact bound checks.
For `S` segments and `P` buckets, directory construction is `O(S + P)`, storage
`O(P)`, and routing `O(1 + log S_bucket)`, worst-case `O(log S)` under skew.
Small indexes keep the original locator. This directory is rebuilt on load,
included in `memory_usage`, and omitted from serialization, preserving Hybrid
bytes and original input positions across scalar, hash and batch APIs.

Measured storage, tracker and Hybrid comparisons, including temporary memory
costs and unchanged small-index cases, are in
[`performance/STORAGE_HOT_HYBRID_REPORT.md`](performance/STORAGE_HOT_HYBRID_REPORT.md).

Dynamic tiers use one byte arena and slot-ordered metadata. The default full-merge
policy is preserved. `set_background_compaction(true)` opts into capped size tiering:
four similarly sized tiers merge on one worker and publication occurs at flush
boundaries. Disabling the worker keeps size-tiered merges synchronous;
`set_compaction_ratio` also enables size tiering. `max_tiers` applies writer
backpressure. Explicit `compact` restores single-tier reads. Multiple tiers can
increase lookup latency. `finish_background_compaction` waits for eligible merges
and belongs in end-to-end write timing. When deleted key/metadata bytes reach
one quarter of resident tier bytes, or no live keys remain, `flush` or
`finish_background_compaction` performs a full deletion collection. It joins
workers, filters dead keys before building the temporary map, and releases
tombstone storage only after successful publication.
The serialized key/id/config layout stays compatible; restored indexes use the
default full-compaction policy. `memory_usage` excludes in-flight worker scratch
and unpublished output.

For logarithmic rebuilding, explicitly select
`set_compaction_policy(CompactionPolicy::Geometric)`. With fanout `r`, this uses
up to `(r-1)` runs per level and an adaptive `tier_limit()` whose minimum is
`max_tiers`. Every merge promotes a level, including tiny explicit flushes.
For `F` flush runs, cumulative rebuilt records are `O(n log_r F)`; with full
`B`-key flushes, `F` is approximately `n/B`. Lookup can probe `O(r log_r F)` runs.
These bounds assume expected linear MPH construction and bounded key width;
hashing/comparison additionally depends on key bytes. Existing Full and
SizeTiered modes retain the hard cap and do not promise these bounds.

The measured compaction comparison and memory tradeoffs are in
[`performance/COMPACTION_REPORT.md`](performance/COMPACTION_REPORT.md).
