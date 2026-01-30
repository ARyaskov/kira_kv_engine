# Kira KV Engine API Documentation

## Overview

Kira KV Engine exposes a single public index: **HybridIndex**.

- By default it builds an MPH index.
- If `auto_detect_numeric` is enabled and all keys are 8-byte little-endian integers, it builds a PGM index.

The internal engines are not part of the public API.


## API Reference

### `HybridBuilder`

Constructs a hybrid index. By default it builds MPH; PGM selection requires enabling `auto_detect_numeric`.

```rust
pub struct HybridBuilder {
    config: HybridConfig,
}

impl HybridBuilder {
    pub fn new() -> Self
    pub fn with_config(self, config: HybridConfig) -> Self
    pub fn with_mph_config(self, mph_config: BuildConfig) -> Self
    pub fn with_pgm_epsilon(self, epsilon: u32) -> Self
    pub fn auto_detect_numeric(self, enabled: bool) -> Self
    pub fn build_index<K>(self, keys: Vec<K>) -> Result<HybridIndex, HybridError>
    where
        K: AsRef<[u8]>
}
```

### `HybridConfig`

```rust
pub struct HybridConfig {
    pub mph_config: BuildConfig,
    pub pgm_epsilon: u32,
    pub auto_detect_numeric: bool,
}
```

### `HybridIndex`

```rust
pub struct HybridIndex { /* opaque */ }

impl HybridIndex {
    pub fn lookup(&self, key: &[u8]) -> Result<usize, HybridError>
    pub fn get(&self, key: &[u8]) -> Result<usize, HybridError>

    pub fn lookup_str(&self, key: &str) -> Result<usize, HybridError>
    pub fn get_str(&self, key: &str) -> Result<usize, HybridError>

    pub fn lookup_u64(&self, key: u64) -> Result<usize, HybridError>
    pub fn get_u64(&self, key: u64) -> Result<usize, HybridError>

    pub fn lookup_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>>
    pub fn get_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>>

    pub fn contains(&self, key: &[u8]) -> bool
    pub fn has(&self, key: &[u8]) -> bool
    pub fn exists(&self, key: &[u8]) -> bool

    pub fn contains_batch(&self, keys: &[&[u8]]) -> Vec<bool>
    pub fn has_batch(&self, keys: &[&[u8]]) -> Vec<bool>
    pub fn exists_batch(&self, keys: &[&[u8]]) -> Vec<bool>

    pub fn range(&self, min_key: u64, max_key: u64) -> Vec<usize>
    pub fn get_all(&self, min_key: u64, max_key: u64) -> Vec<usize>

    pub fn len(&self) -> usize
    pub fn stats(&self) -> HybridStats

    pub fn to_bytes(&self) -> Result<Vec<u8>, HybridError>
    pub fn serialize(&self) -> Result<Vec<u8>, HybridError>

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HybridError>
    pub fn deserialize(bytes: &[u8]) -> Result<Self, HybridError>
}
```

Notes:
- `range()` returns results only when the engine is PGM; otherwise it returns an empty vector.
- `lookup_u64()` works for both engines (numbers are encoded as little-endian bytes for MPH).

### `HybridStats`

```rust
pub struct HybridStats {
    pub engine: &'static str, // "pgm" or "mph"
    pub total_keys: usize,
    pub mph_memory: usize,
    pub pgm_memory: usize,
    pub total_memory: usize,
}
```

### `HybridError`

```rust
pub enum HybridError {
    Mph(String),
    Pgm(String),
    KeyNotFound,
    InvalidKey,
    CorruptData,
}
```

## Examples

### Numeric keys (PGM when `auto_detect_numeric` is enabled)

```rust
use kira_kv_engine::HybridBuilder;

let keys: Vec<Vec<u8>> = (0..1_000_000u64)
    .map(|v| v.to_le_bytes().to_vec())
    .collect();

let index = HybridBuilder::new()
    .auto_detect_numeric(true)
    .build_index(keys)?;
let pos = index.lookup_u64(42)?;
```

### Mixed keys (MPH default)

```rust
use kira_kv_engine::HybridBuilder;

let mut keys = Vec::new();
keys.push(b"alpha".to_vec());
keys.push(123u64.to_le_bytes().to_vec());
keys.push(b"beta".to_vec());

let index = HybridBuilder::new().build_index(keys)?;
let pos = index.lookup(b"alpha")?;
```

### Batch lookup

```rust
use kira_kv_engine::HybridIndex;

fn batch_lookup(index: &HybridIndex, keys: &[&[u8]]) -> Vec<Option<usize>> {
    index.lookup_batch(keys)
}
```
