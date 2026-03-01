# Kira KV Engine

![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/kira_kv_engine)

`kira_kv_engine` builds a static key -> id index for unique keys.

- Default MPH backend: `PtrHash2025` (fast build + fast lookup)
- Optional numeric mode: `PGM` + MPH remap (`auto_detect_numeric = true`)
- Stable lookups after build: index in `[0..n)`

## Install

```toml
[dependencies]
kira_kv_engine = ">=0.3.2"
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

### Benchmark Tables (MacBook Air 2020, M1, 8GB RAM)

#### Dataset: 1,000,000 keys (`runs=7`, `threads=8`, `core_ids=0..7`)

| Struct | Workload | Cache | Runs | Build m (ms) | Build p95 (ms) | Rate m (keys/s) | Lookup m (ns) | Lookup p95 (ns) | Thr m (ops/s) | Hit % | Miss % | B/key |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtrHash25Default | positive | cold | 7 | 867.45 | 920.04 | 1,152,798 | 26.79 | 32.08 | 37,327,361 | 100.0 | 0.0 | 15.23 |
| PtrHash25Default | positive | warm | 7 | 867.45 | 920.04 | 1,152,798 | 18.76 | 24.21 | 53,309,621 | 100.0 | 0.0 | 15.23 |
| PtrHash25Default | negative | cold | 7 | 867.45 | 920.04 | 1,152,798 | 23.75 | 28.51 | 42,099,343 | 70.0 | 30.0 | 15.23 |
| PtrHash25Default | negative | warm | 7 | 867.45 | 920.04 | 1,152,798 | 14.34 | 26.08 | 69,751,253 | 70.0 | 30.0 | 15.23 |
| PtrHash25Default | zipf | cold | 7 | 867.45 | 920.04 | 1,152,798 | 26.25 | 29.69 | 38,097,676 | 100.0 | 0.0 | 15.23 |
| PtrHash25Default | zipf | warm | 7 | 867.45 | 920.04 | 1,152,798 | 17.87 | 27.67 | 55,954,449 | 100.0 | 0.0 | 15.23 |
| PTHash | positive | cold | 7 | 1,256.36 | 1,289.27 | 795,952 | 32.23 | 37.44 | 31,030,190 | 100.0 | 0.0 | 15.23 |
| PTHash | positive | warm | 7 | 1,256.36 | 1,289.27 | 795,952 | 24.42 | 26.86 | 40,947,224 | 100.0 | 0.0 | 15.23 |
| PTHash | negative | cold | 7 | 1,256.36 | 1,289.27 | 795,952 | 27.26 | 30.25 | 36,688,254 | 70.0 | 30.0 | 15.23 |
| PTHash | negative | warm | 7 | 1,256.36 | 1,289.27 | 795,952 | 15.42 | 23.28 | 64,840,331 | 70.0 | 30.0 | 15.23 |
| PTHash | zipf | cold | 7 | 1,256.36 | 1,289.27 | 795,952 | 25.98 | 31.07 | 38,497,341 | 100.0 | 0.0 | 15.23 |
| PTHash | zipf | warm | 7 | 1,256.36 | 1,289.27 | 795,952 | 17.72 | 26.01 | 56,446,724 | 100.0 | 0.0 | 15.23 |
| PtrHash25 | positive | cold | 7 | 843.30 | 849.89 | 1,185,818 | 26.39 | 30.97 | 37,900,322 | 100.0 | 0.0 | 15.23 |
| PtrHash25 | positive | warm | 7 | 843.30 | 849.89 | 1,185,818 | 18.05 | 24.33 | 55,388,897 | 100.0 | 0.0 | 15.23 |
| PtrHash25 | negative | cold | 7 | 843.30 | 849.89 | 1,185,818 | 23.05 | 25.54 | 43,379,243 | 70.0 | 30.0 | 15.23 |
| PtrHash25 | negative | warm | 7 | 843.30 | 849.89 | 1,185,818 | 16.69 | 20.46 | 59,916,117 | 70.0 | 30.0 | 15.23 |
| PtrHash25 | zipf | cold | 7 | 843.30 | 849.89 | 1,185,818 | 27.43 | 32.84 | 36,454,228 | 100.0 | 0.0 | 15.23 |
| PtrHash25 | zipf | warm | 7 | 843.30 | 849.89 | 1,185,818 | 18.92 | 23.39 | 52,847,140 | 100.0 | 0.0 | 15.23 |
| CHD | positive | cold | 7 | 888.05 | 914.11 | 1,126,065 | 27.59 | 44.54 | 36,242,809 | 100.0 | 0.0 | 15.23 |
| CHD | positive | warm | 7 | 888.05 | 914.11 | 1,126,065 | 18.01 | 36.15 | 55,534,946 | 100.0 | 0.0 | 15.23 |
| CHD | negative | cold | 7 | 888.05 | 914.11 | 1,126,065 | 24.49 | 32.87 | 40,828,825 | 70.0 | 30.0 | 15.23 |
| CHD | negative | warm | 7 | 888.05 | 914.11 | 1,126,065 | 15.07 | 19.49 | 66,345,994 | 70.0 | 30.0 | 15.23 |
| CHD | zipf | cold | 7 | 888.05 | 914.11 | 1,126,065 | 25.94 | 30.87 | 38,548,034 | 100.0 | 0.0 | 15.23 |
| CHD | zipf | warm | 7 | 888.05 | 914.11 | 1,126,065 | 18.49 | 30.85 | 54,078,375 | 100.0 | 0.0 | 15.23 |
| RecSplit | positive | cold | 7 | 1,238.34 | 1,281.61 | 807,536 | 27.21 | 35.35 | 36,744,442 | 100.0 | 0.0 | 15.23 |
| RecSplit | positive | warm | 7 | 1,238.34 | 1,281.61 | 807,536 | 17.66 | 25.53 | 56,609,114 | 100.0 | 0.0 | 15.23 |
| RecSplit | negative | cold | 7 | 1,238.34 | 1,281.61 | 807,536 | 22.86 | 27.86 | 43,750,924 | 70.0 | 30.0 | 15.23 |
| RecSplit | negative | warm | 7 | 1,238.34 | 1,281.61 | 807,536 | 15.02 | 18.50 | 66,592,704 | 70.0 | 30.0 | 15.23 |
| RecSplit | zipf | cold | 7 | 1,238.34 | 1,281.61 | 807,536 | 25.37 | 30.41 | 39,412,750 | 100.0 | 0.0 | 15.23 |
| RecSplit | zipf | warm | 7 | 1,238.34 | 1,281.61 | 807,536 | 18.41 | 24.00 | 54,310,930 | 100.0 | 0.0 | 15.23 |

#### Dataset: 10,000,000 keys (`runs=7`, `threads=8`, `core_ids=0..7`)

| Struct | Workload | Cache | Runs | Build m (ms) | Build p95 (ms) | Rate m (keys/s) | Lookup m (ns) | Lookup p95 (ns) | Thr m (ops/s) | Hit % | Miss % | B/key |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtrHash25Default | positive | cold | 7 | 15,325.54 | 15,999.80 | 652,505 | 98.64 | 128.74 | 10,137,618 | 100.0 | 0.0 | 15.23 |
| PtrHash25Default | positive | warm | 7 | 15,325.54 | 15,999.80 | 652,505 | 35.40 | 69.97 | 28,247,917 | 100.0 | 0.0 | 15.23 |
| PtrHash25Default | negative | cold | 7 | 15,325.54 | 15,999.80 | 652,505 | 37.16 | 72.44 | 26,913,670 | 70.0 | 30.0 | 15.23 |
| PtrHash25Default | negative | warm | 7 | 15,325.54 | 15,999.80 | 652,505 | 29.24 | 68.77 | 34,198,744 | 70.0 | 30.0 | 15.23 |
| PtrHash25Default | zipf | cold | 7 | 15,325.54 | 15,999.80 | 652,505 | 40.27 | 97.88 | 24,833,923 | 100.0 | 0.0 | 15.23 |
| PtrHash25Default | zipf | warm | 7 | 15,325.54 | 15,999.80 | 652,505 | 37.98 | 58.30 | 26,329,647 | 100.0 | 0.0 | 15.23 |
| PTHash | positive | cold | 7 | 23,693.30 | 25,045.76 | 422,060 | 127.64 | 222.49 | 7,834,790 | 100.0 | 0.0 | 15.23 |
| PTHash | positive | warm | 7 | 23,693.30 | 25,045.76 | 422,060 | 39.00 | 62.08 | 25,638,843 | 100.0 | 0.0 | 15.23 |
| PTHash | negative | cold | 7 | 23,693.30 | 25,045.76 | 422,060 | 42.25 | 54.97 | 23,670,970 | 70.0 | 30.0 | 15.23 |
| PTHash | negative | warm | 7 | 23,693.30 | 25,045.76 | 422,060 | 29.82 | 47.36 | 33,532,674 | 70.0 | 30.0 | 15.23 |
| PTHash | zipf | cold | 7 | 23,693.30 | 25,045.76 | 422,060 | 50.53 | 58.00 | 19,790,545 | 100.0 | 0.0 | 15.23 |
| PTHash | zipf | warm | 7 | 23,693.30 | 25,045.76 | 422,060 | 37.04 | 48.11 | 26,999,663 | 100.0 | 0.0 | 15.23 |
| PtrHash25 | positive | cold | 7 | 15,320.16 | 17,225.71 | 652,735 | 105.16 | 307.64 | 9,509,695 | 100.0 | 0.0 | 15.23 |
| PtrHash25 | positive | warm | 7 | 15,320.16 | 17,225.71 | 652,735 | 38.07 | 55.99 | 26,264,518 | 100.0 | 0.0 | 15.23 |
| PtrHash25 | negative | cold | 7 | 15,320.16 | 17,225.71 | 652,735 | 36.87 | 43.74 | 27,119,865 | 70.0 | 30.0 | 15.23 |
| PtrHash25 | negative | warm | 7 | 15,320.16 | 17,225.71 | 652,735 | 32.78 | 60.40 | 30,504,080 | 70.0 | 30.0 | 15.23 |
| PtrHash25 | zipf | cold | 7 | 15,320.16 | 17,225.71 | 652,735 | 47.87 | 78.33 | 20,889,544 | 100.0 | 0.0 | 15.23 |
| PtrHash25 | zipf | warm | 7 | 15,320.16 | 17,225.71 | 652,735 | 43.70 | 91.16 | 22,880,677 | 100.0 | 0.0 | 15.23 |
| CHD | positive | cold | 7 | 16,287.62 | 16,898.75 | 613,963 | 94.35 | 289.34 | 10,598,648 | 100.0 | 0.0 | 15.23 |
| CHD | positive | warm | 7 | 16,287.62 | 16,898.75 | 613,963 | 39.77 | 50.59 | 25,143,532 | 100.0 | 0.0 | 15.23 |
| CHD | negative | cold | 7 | 16,287.62 | 16,898.75 | 613,963 | 36.62 | 45.91 | 27,303,754 | 70.0 | 30.0 | 15.23 |
| CHD | negative | warm | 7 | 16,287.62 | 16,898.75 | 613,963 | 31.16 | 47.01 | 32,087,277 | 70.0 | 30.0 | 15.23 |
| CHD | zipf | cold | 7 | 16,287.62 | 16,898.75 | 613,963 | 46.71 | 61.58 | 21,407,161 | 100.0 | 0.0 | 15.23 |
| CHD | zipf | warm | 7 | 16,287.62 | 16,898.75 | 613,963 | 35.94 | 46.37 | 27,827,388 | 100.0 | 0.0 | 15.23 |
| RecSplit | positive | cold | 7 | 23,565.25 | 24,363.35 | 424,354 | 137.81 | 303.90 | 7,256,499 | 100.0 | 0.0 | 15.23 |
| RecSplit | positive | warm | 7 | 23,565.25 | 24,363.35 | 424,354 | 43.79 | 49.96 | 22,837,568 | 100.0 | 0.0 | 15.23 |
| RecSplit | negative | cold | 7 | 23,565.25 | 24,363.35 | 424,354 | 36.48 | 46.12 | 27,414,791 | 70.0 | 30.0 | 15.23 |
| RecSplit | negative | warm | 7 | 23,565.25 | 24,363.35 | 424,354 | 29.55 | 38.72 | 33,835,222 | 70.0 | 30.0 | 15.23 |
| RecSplit | zipf | cold | 7 | 23,565.25 | 24,363.35 | 424,354 | 42.32 | 70.12 | 23,629,490 | 100.0 | 0.0 | 15.23 |
| RecSplit | zipf | warm | 7 | 23,565.25 | 24,363.35 | 424,354 | 38.86 | 47.13 | 25,731,747 | 100.0 | 0.0 | 15.23 |

## Notes

- Input keys must be unique.
- Indexes are static (rebuild on keyset change).
- Public API reference: `API.md`.

## License

MIT
