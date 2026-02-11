# Kira KV Engine

![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/kira_kv_engine)

`kira_kv_engine` builds a static key -> id index for unique keys.

- Default MPH backend: `PtrHash2025` (fast build + fast lookup)
- Optional numeric mode: `PGM` + MPH remap (`auto_detect_numeric = true`)
- Stable lookups after build: index in `[0..n)`

## Install

```toml
[dependencies]
kira_kv_engine = ">=0.3"
```

## Quick Start

```rust
use kira_kv_engine::IndexBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let keys = vec![
        b"user:1".to_vec(),
        b"user:2".to_vec(),
        42u64.to_le_bytes().to_vec(),
    ];

    // Default backend is PtrHash2025
    let index = IndexBuilder::new().build_index(keys)?;

    let id1 = index.lookup_str("user:1")?;
    let id2 = index.lookup_u64(42)?;

    println!("{} {}", id1, id2);
    Ok(())
}
```

## Backends

Selectable via `IndexBuilder::with_backend(...)`:

- `BackendKind::PtrHash2025` (default)
- `BackendKind::PTHash`
- `BackendKind::CHD`
- `BackendKind::RecSplit`
- `BackendKind::BBHash` (only with `bbhash` feature)

Backends are selected via `BackendKind`; only currently supported variants are available.

## Numeric Auto-Detect (PGM)

If every key is exactly 8 bytes (`u64::to_le_bytes()`), you can enable numeric mode:

```rust
use kira_kv_engine::IndexBuilder;

let keys: Vec<Vec<u8>> = (0..100_000u64).map(|v| v.to_le_bytes().to_vec()).collect();
let index = IndexBuilder::new()
    .auto_detect_numeric(true)
    .build_index(keys)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

In this mode:

- `lookup_u64` is the native path
- `range(min, max)` returns matching index ids

## Build/Perf Controls

`IndexBuilder` supports:

- `.with_parallel_build(bool)`
- `.with_build_fast_profile(bool)`
- `.with_mph_config(...)` (`gamma`, `rehash_limit`, `salt`)
- `.with_backend(...)`

## Serialization

```rust
let bytes = index.to_bytes()?;
let restored = kira_kv_engine::Index::from_bytes(&bytes)?;
```

## Benchmarks

Run:

```bash
cargo run --release --example million_build
cargo run --release --example ptrhash_bench
```

The benchmark outputs include:

- build time / build rate
- cold & warm lookup latency
- hit/miss ratios
- memory usage (bytes per key)

## Notes

- Input keys must be unique.
- Indexes are static (rebuild on keyset change).
- Public API reference: `API.md`.

## License

MIT
