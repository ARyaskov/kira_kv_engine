# Kira KV Engine API

## Public Exports

From crate root:

- `IndexBuilder`
- `IndexConfig`
- `Index`
- `IndexStats`
- `IndexError`
- `BackendKind`
- `BackendBuildConfig` (alias of internal MPH backend config)
- `BuildProfile`
- `MphBackend` trait

---

## `IndexBuilder`

```rust
pub struct IndexBuilder { /* opaque */ }

impl IndexBuilder {
    pub fn new() -> Self;
    pub fn with_config(self, config: IndexConfig) -> Self;
    pub fn with_mph_config(self, mph_config: /* same type as IndexConfig::mph_config */) -> Self;
    pub fn with_pgm_epsilon(self, epsilon: u32) -> Self;
    pub fn with_backend(self, backend: BackendKind) -> Self;
    pub fn with_hot_fraction(self, hot_fraction: f32) -> Self;
    pub fn with_hot_backend(self, backend: BackendKind) -> Self;
    pub fn with_cold_backend(self, backend: BackendKind) -> Self;
    pub fn with_parallel_build(self, enabled: bool) -> Self;
    pub fn with_build_fast_profile(self, enabled: bool) -> Self;
    pub fn auto_detect_numeric(self, enabled: bool) -> Self;
    pub fn build_index<K>(self, keys: Vec<K>) -> Result<Index, IndexError>
    where
        K: AsRef<[u8]>;
}
```

Notes:

- Default backend is `BackendKind::PtrHash2025`.
- `with_hot_fraction`/`with_hot_backend`/`with_cold_backend` are compatibility knobs and currently do not change behavior.

---

## `IndexConfig`

```rust
pub struct IndexConfig {
    pub mph_config: /* PtrHash build config: gamma, rehash_limit, salt */,
    pub pgm_epsilon: u32,
    pub auto_detect_numeric: bool,
    pub backend: BackendKind,
    pub hot_fraction: f32,
    pub hot_backend: BackendKind,
    pub cold_backend: BackendKind,
    pub enable_parallel_build: bool,
    pub build_fast_profile: bool,
}
```

Defaults:

- `backend = BackendKind::PtrHash2025`
- `auto_detect_numeric = false`
- `enable_parallel_build = true`
- `build_fast_profile = true`

---

## `BackendKind`

```rust
pub enum BackendKind {
    PTHash,
    PtrHash2025,
    CHD,
    RecSplit,
    #[cfg(feature = "bbhash")]
    BBHash,
}
```

---

## `Index`

```rust
pub struct Index { /* opaque */ }

impl Index {
    pub fn lookup(&self, key: &[u8]) -> Result<usize, IndexError>;
    pub fn get(&self, key: &[u8]) -> Result<usize, IndexError>;

    pub fn lookup_str(&self, key: &str) -> Result<usize, IndexError>;
    pub fn get_str(&self, key: &str) -> Result<usize, IndexError>;

    pub fn lookup_u64(&self, key: u64) -> Result<usize, IndexError>;
    pub fn get_u64(&self, key: u64) -> Result<usize, IndexError>;

    pub fn range(&self, min_key: u64, max_key: u64) -> Vec<usize>;
    pub fn get_all(&self, min_key: u64, max_key: u64) -> Vec<usize>;

    pub fn contains(&self, key: &[u8]) -> bool;
    pub fn has(&self, key: &[u8]) -> bool;
    pub fn exists(&self, key: &[u8]) -> bool;

    pub fn contains_batch(&self, keys: &[&[u8]]) -> Vec<bool>;
    pub fn has_batch(&self, keys: &[&[u8]]) -> Vec<bool>;
    pub fn exists_batch(&self, keys: &[&[u8]]) -> Vec<bool>;

    pub fn lookup_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>>;
    pub fn get_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>>;

    pub fn len(&self) -> usize;
    pub fn stats(&self) -> IndexStats;
    pub fn print_detailed_stats(&self);

    pub fn to_bytes(&self) -> Result<Vec<u8>, IndexError>;
    pub fn serialize(&self) -> Result<Vec<u8>, IndexError>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IndexError>;
    pub fn deserialize(bytes: &[u8]) -> Result<Self, IndexError>;
}
```

Behavior notes:

- `range/get_all` return results only when engine is numeric PGM mode (`auto_detect_numeric = true` and all input keys are 8-byte LE `u64`).
- In MPH mode, `range/get_all` return empty vector.
- `lookup_u64` works in both modes.

---

## `IndexStats`

```rust
pub struct IndexStats {
    pub engine: &'static str, // "mph" or "pgm"
    pub total_keys: usize,
    pub mph_memory: usize,
    pub pgm_memory: usize,
    pub total_memory: usize,
}
```

---

## `IndexError`

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

## `BackendBuildConfig` and `BuildProfile`

`BackendBuildConfig` configures MPH backend internals exposed from `mph_backend`:

```rust
pub enum BuildProfile {
    Balanced,
    Fast,
}

pub struct BackendBuildConfig {
    pub backend: BackendKind,
    pub hot_fraction: f32,
    pub hot_backend: BackendKind,
    pub cold_backend: BackendKind,
    pub enable_parallel_build: bool,
    pub seed: u64,
    pub gamma: f64,
    pub rehash_limit: u32,
    pub max_pilot_attempts: u32,
    pub build_profile: BuildProfile,
    pub fast_fail_rounds: u32,
    pub frequencies: Option<Vec<(u64, u32)>>,
}
```

This is mostly useful for advanced/internal tuning; `IndexBuilder` is the primary user-facing API.
