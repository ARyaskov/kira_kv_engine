# Kira KV Engine

![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/kira_kv_engine)

`kira_kv_engine` is a high-performance key→id index toolkit for Rust.
**Six engines** under one roof, each tuned for a specific workload shape — static
MPH (lean & default), dynamic LSM-on-MPH, hybrid byte-key, learned PGM (u64 and
u128). Edition 2024, Rust 1.95+.

| Engine | Memory | Lookup | Insert/Delete | Range | Best for |
|---|---:|---:|:---:|:---:|---|
| **Index (lean)** | **0.42 B/key** | **13 ns** | ❌ static | ❌ | Closed-world point lookups (top pick) |
| **Index** (default) | 3.8 B/key | 23 ns miss / 56 ns hit | ❌ static | ❌ | Open-world point lookups (Bloom-rejected misses) |
| **DynamicIndex** | keys + ~30 B/key | 39 ns | ✅ ~540 ns | ❌ | Mutable byte-key sets (LSM on top of MPH) |
| **HybridIndex** | 4.4–6.8 B/key | 60–118 ns | ❌ static | hash-space | Universal byte keys + batch queries |
| **PgmIndex** | 8 B/key | 138 ns | ❌ static | ✅ semantic | u64 range queries (timestamps, IDs) |
| **PgmIndexU128** | 16 B/key | 144 ns | ❌ static | ✅ semantic | UUID/SHA range queries |

10M random keys, single thread, Apple M-series (aarch64); memory is the whole
index, lookups are random hits (`cargo bench --bench engines` prints this table
for your machine). The static MPH is **minimal**: ids are exactly `[0, n)`.

---

## Choosing an index

> **The flowchart**: do you need **range queries**? If yes → PgmIndex (u64) or HybridIndex (bytes).
> Otherwise → Index (use `lean_mph` if your queries are always valid keys).

### Decision table

| Question | Yes | No |
|---|---|---|
| All queries from the build set? (closed world: dictionary, vocab, deduped IDs) | **Index (lean)** | Index |
| **Need frequent insert/delete?** | **DynamicIndex** (LSM-tree on MPH tiers) | use a static engine |
| Need `range(min, max)` over byte keys? | HybridIndex (hash-space) | … |
| Need `range(min, max)` over **u64** with proper ordering? | **PgmIndex** | … |
| 16-byte keys (UUID, SHA-128, IPv6) + range? | **PgmIndexU128** | use Index |

### Use-case recipes

**Bioinformatics: VCF indexer (600M variants, 20K genes, range queries)**

```rust
use kira_kv_engine::{IndexBuilder, PgmBuilder};

// 600M rsIDs → file offset (closed set after build → lean)
let rs_index = IndexBuilder::new()
    .with_lean_mph(true)               // 0.42 B/key, ~13 ns warm
    .build_index(rs_keys)?;

// Variant positions (chrom<<56 | pos) → file offset, with range queries
let pos_index = PgmBuilder::new()
    .with_epsilon(64)
    .with_bloom_filter(true)
    .build(packed_positions)?;

// 20K gene names → coordinate region (lean, ~8 KB total)
let gene_index = IndexBuilder::new()
    .with_lean_mph(true)
    .build_index(gene_names)?;

// "All variants in chr1:1M-2M" — semantic range over u64 keys, O(1) space.
let variants_in_region: std::ops::Range<usize> = pos_index.range(start, end);
```

**LLM token vocabulary (closed set, ~100K tokens)**

```rust
let token_index = IndexBuilder::new()
    .with_lean_mph(true)        // tokens fixed at training time
    .build_index(tokens)?;       // ~40 KB, ~13 ns lookup
```

**URL shortener (open world — invalid lookups happen)**

```rust
let url_index = IndexBuilder::new()
    // lean_mph=false (default) — returns KeyNotFound for foreign IDs
    .build_index(short_codes)?;  // 3.8 B/key, misses rejected by the Bloom filter
```

**Real-time analytics with skewed access (Zipfian)**

```rust
use kira_kv_engine::{DynamicHotTier, HotTierIndex};
use std::sync::Arc;

// Shared by every query thread; lookups take a read lock, the frequency
// tracker is only try-locked (contended observations are dropped, not queued).
let tier = Arc::new(DynamicHotTier::new(None, 2048 /* tracker slots */, 1_000_000 /* observations per rebuild */));

// On the query path:
if let Some(pos) = tier.lookup_u64(key) { /* hot hit */ } else { /* static index */ }

// On a background thread, whenever tier.should_rebuild():
let top: Vec<(u64, u64)> = tier.take_top_k(1024);
let keys: Vec<u64> = top.iter().map(|&(k, _)| k).collect();
let positions: Vec<u32> = keys.iter().map(|k| position_in_static_index(*k)).collect();
if let Some(hot) = HotTierIndex::build_from_u64(&keys, &positions, seed) {
    tier.install(hot);          // readers in flight keep the old tier until they finish
}
```

**Mutable keyset with insert/delete (online dictionary, session table, etc.)**

```rust
use kira_kv_engine::{DynamicIndex, DynamicConfig};

let mut idx = DynamicIndex::with_config(DynamicConfig {
    flush_threshold: 64 * 1024,    // batch ~64K inserts before tier-flush
    max_tiers: 8,                  // compact when >8 tiers accumulate
    lean_tiers: false,
    parallel_build: true,
});

let id_alice = idx.insert(b"alice".to_vec());    // stable u32 id, ~540 ns
let id_bob   = idx.insert(b"bob".to_vec());

assert_eq!(idx.lookup(b"alice"), Some(id_alice));
idx.delete(b"alice");
assert_eq!(idx.lookup(b"alice"), None);

// Explicit compact merges all tiers + buffer into one → restores
// single-tier lookup latency (~40 ns). Returns Err and changes nothing if the
// rebuild fails.
idx.compact()?;
```

Stable IDs **never** change across flush/compact — safe to store as external
pointers (e.g. file offsets, row indexes). Every tier hit is verified against the
stored key bytes, so a tier can never answer with another key's id.

**General byte/string keys with range queries**

```rust
use kira_kv_engine::HybridBuilder;

let hybrid = HybridBuilder::new()
    .with_pgm_epsilon(2048)
    .with_lean(true)            // 4.4 B/key
    .build_from_u64(&u64_keys)?; // SIMD-accelerated build path
let positions = hybrid.lookup_batch_u64_simd(&query_keys);
```

---

## Install

```toml
[dependencies]
kira_kv_engine = "0.7"
```

The on-disk formats and parts of the API changed in 0.7 (see *Notes*); pin the
minor version.

## Quick Start

```rust
use kira_kv_engine::IndexBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let keys = vec![
        b"user:1".to_vec(),
        b"user:2".to_vec(),
        b"user:3".to_vec(),
    ];

    // Default — Bloom + fingerprints, safe for foreign keys.
    let index = IndexBuilder::new().build_index(keys)?;

    let id = index.lookup_str("user:1")?;
    assert_eq!(id, index.lookup(b"user:1")?);
    Ok(())
}
```

## API overview

All static engines are built once from a unique key set. `Index` returns a
`usize` id in `[0..len())`; the other engines return `u32`/`usize` positions in
the same range.

### `IndexBuilder` (PtrHash25 — point lookups, any key type)

```rust
IndexBuilder::new()
    .with_lean_mph(true)               // ★ -89% memory (closed-world only)
    .with_parallel_build(true)
    .build_index_ref(&keys)?           // borrows: no clone, no 10M frees in the build
    // or .build_index(keys)?          // owned: same build + parallel drop of `keys`
```

Keys are hashed in place in parallel (no arena copy), split into ~32K-key **parts**,
and every part is built independently on its own core with an L2-resident working
set: λ = 3 keys per bucket, cuckoo-style eviction in the pilot search, and a small
per-part remap so the output is a bijection onto `[0, n)`. Duplicate keys are
detected exactly and reported as `IndexError::DuplicateKey`. Key types need
`AsRef<[u8]> + Sync` (`+ Send` for the owned variant) — true for `Vec<u8>`,
`String`, `&[u8]`, `&str`, arrays.

Lookups: `index.lookup(&[u8])`, `lookup_u64(u64)`, `lookup_batch_pipelined(&[&[u8]])`,
`lookup_batch_u64_simd(&[u64])` / `lookup_batch_u64_simd_into` (zero allocation).

`contains()` is the Bloom-filter answer (definite "no", probabilistic "yes"); in
lean mode there is no filter and it always returns `true` — check
`supports_negative_lookups()`.

### `HybridBuilder` (PGM + per-segment mini-MPH)

```rust
use kira_kv_engine::HybridBuilder;

HybridBuilder::new()
    .with_pgm_epsilon(2048)
    .with_lean(true)
    .with_linear_threshold(64)
    .build_from_u64(&u64_keys)?        // SIMD-fast path for u64
    // or .build(&[byte_slices])?
```

Lookups: `hybrid.lookup(&[u8])`, `lookup_u64(u64)`, `lookup_batch_u64_simd(&[u64])`,
`lookup_batch_hashes(&[u64])`.

### `PgmBuilder` (sorted u64 with semantic range queries)

```rust
use kira_kv_engine::PgmBuilder;

PgmBuilder::new()
    .with_epsilon(64)                  // every key within 64 positions of its prediction
    .with_bloom_filter(true)           // negative-fast
    .with_elias_fano(true)             // -40% key memory
    .with_target_lookup_ns(50)         // auto-tune ε
    .build(u64_keys)?                  // sorted here; duplicates are an error
```

Lookups: `pgm.index(u64)`, `range(min, max) -> Range<usize>`, `lower_bound`,
`upper_bound`. Segmentation is linear-time (10M keys in ~40 ms single-threaded).

### `PgmIndexU128` (16-byte keys: UUID/SHA-128/IPv6)

```rust
use kira_kv_engine::PgmIndexU128;

let idx = PgmIndexU128::build_from_bytes16(&uuids_be, 64)?;
let pos = idx.index_bytes16(&query_uuid)?;
let range: std::ops::Range<usize> = idx.range(uuid_a, uuid_b);
```

## Serialization

Every engine has a self-contained byte form with a magic, a format version and a
64-bit checksum; loading verifies the checksum and every structural invariant, so
a corrupt or hostile file is rejected instead of being read out of bounds.
Files are portable between x86_64, aarch64 and other targets.

```rust
// Index: streaming save/load, or in-memory bytes
index.save("keys.idx")?;                       // streams, ~0.5 B/key transient
let restored = kira_kv_engine::Index::load("keys.idx")?;
let bytes = index.to_bytes()?;
let restored = kira_kv_engine::Index::from_bytes(&bytes)?;

// The other engines
let bytes = pgm.to_bytes()?;      let pgm = kira_kv_engine::PgmIndex::from_bytes(&bytes)?;
let bytes = hybrid.to_bytes();    let hybrid = kira_kv_engine::HybridIndex::from_bytes(&bytes)?;
let bytes = u128_idx.to_bytes();  let u128_idx = kira_kv_engine::PgmIndexU128::from_bytes(&bytes)?;
let bytes = dynamic.to_bytes();   let dynamic = kira_kv_engine::DynamicIndex::from_bytes(&bytes)?;
```

`Index::save_mmap` / `open_mmap` write the same payload inside a section container
(64-byte-aligned sections, reserved for a future zero-copy loader); today they
read the file into memory like `load`. Indexes written by 0.6 still load (without
checksum verification); rebuild them to get the minimal geometry.

## Benchmarks

```bash
cargo bench --bench engines                       # every engine, 10M keys → Markdown table
KIRA_BENCH_N=100000000 cargo bench --bench engines
cargo run --release --example million_build       # Index build/lookup, multi-threaded lookups
# KIRA_BENCH_N / KIRA_BENCH_OPS / KIRA_BENCH_RUNS → key count, lookups, repetitions;
# KIRA_BUILD_TRACE=1 prints build phases.
```

### 10M random keys, Apple M-series, single thread (`cargo bench --bench engines`)

| Engine | Build s | B/key | Lookup ns | Batch ns | Miss ns |
|---|---:|---:|---:|---:|---:|
| Index (default) | 0.17 | 3.79 | 55.7 | 58.3 | 22.7 |
| Index (lean) | 0.14 | 0.42 | 12.7 | 11.9 | — |
| HybridIndex | 0.64 | 6.80 | 118.2 | 110.5 | 31.7 |
| HybridIndex (lean) | 0.43 | 4.42 | 59.6 | 58.0 | — |
| PgmIndex ε=64 | 0.03 | 8.00 | 138.2 | — | 113.2 |
| PgmIndexU128 ε=64 (2M keys) | 0.02 | 16.00 | 143.7 | — | — |
| DynamicIndex (1M keys, insert 536 ns) | 0.82 | 51.5 | 38.8 | — | — |

Build = `build_index_ref(&keys)` on the library's build pool. The default `Index`
touches three random cache lines per hit (Bloom block, pilot byte, fingerprint); at
10M keys that working set (~38 MB) no longer fits in cache on this machine, which
is the gap between the hit and miss columns. The lean index is 4 MB and stays
cache-resident.

Pilot storage is ~2.7 bits/key (λ = 3, flat bytes, no rank/select on the lookup
path) plus ~0.1 B/key of remap table, against ~2.4 bits/key for the PtrHash paper.

## Performance tuning checklist

1. **Closed key set?** Enable `lean_mph(true)` → -89% memory, 4× faster lookups.
2. **Hugepages on Linux/Windows?** -10–20 ns per lookup at 100M scale. See
   [Hugepages section](#hugepages-windows-linux) below.
3. **Read-heavy after build?** `Index::save` / `Index::load`: a 10M-key index
   serializes in ~10 ms and loads (checksum + validation included) in ~12 ms.
4. **Batched lookups?** Use `lookup_batch_u64_simd_into` (Index) or
   `lookup_batch_u64_simd` (Hybrid) — prefetch pipelines on both x86_64 and
   aarch64, AVX2 gather dispatched at runtime (no `-C target-cpu` needed).
5. **u64 keys to Hybrid?** Use `build_from_u64()` instead of `build()` — SIMD
   hash path.
6. **Building from keys you keep around?** Use `build_index_ref(&keys)` — the
   owned `build_index(keys)` costs an extra clone on your side plus millions of
   frees inside the build.
7. **Build looks slow?** `KIRA_BUILD_TRACE=1` prints per-phase timings. The build
   pool uses every core (`available_parallelism`), or the P-cores on hybrid CPUs
   where the workers are pinned; `KIRA_BUILD_THREADS`, `KIRA_BUILD_CORE_IDS` and
   `KIRA_BUILD_PIN=0` override that.

## Hugepages (Windows / Linux)

Tables ≥ 1 MB are placed in 2 MB hugepages when the process can allocate them;
`kira_kv_engine::hugepages_available()` tells you whether that is the case
(nothing is printed either way).

On Linux reserve pages first, e.g. `sudo sysctl vm.nr_hugepages=1024`.

On Windows you need explicit privilege:
1. `Win+R` → `secpol.msc`
2. *Local Policies* → *User Rights Assignment* → *Lock pages in memory*
3. Add your user (or `BUILTIN\Administrators`)
4. **Log out and log back in** — only new sessions inherit the privilege

## Notes

- Input keys must be **unique**. `Index` detects duplicates exactly (equal keys can
  never be separated by any pilot) and returns `IndexError::DuplicateKey`;
  `PgmBuilder` / `PgmIndexU128` sort their input and return `DuplicateKeys`.
- Static indexes are immutable — modifications require a rebuild.
- **Hashing is platform-independent.** Byte keys use an AES-round hash with
  bit-identical AES-NI, ARMv8-crypto and software backends; 8-byte keys use a
  64-bit mixer. 0.6 fell back to a different hash on hosts without AES-NI, so its
  files with byte keys are only readable on x86_64 with AES.
- Empty inputs are accepted by every engine and build always-miss indexes.
- `IndexConfig` still carries the option fields of the removed in-`Index` PGM
  engine; they are deprecated no-ops and go away in the next breaking release.
- Legacy Cargo features (`serde`, `simd`, `pgm`, `avx512`, …) are accepted and do
  nothing; SIMD paths are dispatched at runtime.
- Public API reference: `API.md`.

## License

MIT
