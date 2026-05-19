# Kira KV Engine

![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/kira_kv_engine)

`kira_kv_engine` is a high-performance key→id index toolkit for Rust.
**Six engines** under one roof, each tuned for a specific workload shape — static
MPH (lean & default), dynamic LSM-on-MPH, hybrid byte-key, learned PGM (u64 and
u128). Edition 2024, Rust 1.95+.

| Engine | Memory | Lookup warm | Insert/Delete | Range | Best for |
|---|---:|---:|:---:|:---:|---|
| **PtrHash25-lean** | **0.6 B/key** | **13 ns** | ❌ static | ❌ | Closed-world point lookups (top pick) |
| **PtrHash25** (default) | 4.5 B/key | 21 ns | ❌ static | ❌ | Open-world point lookups (Bloom-rejected misses) |
| **DynamicIndex** | ~6 B/key | 25–40 ns | ✅ ~700 ns | ❌ | Mutable byte-key sets (LSM on top of MPH) |
| **HybridIndex** | 13–16 B/key | 45–60 ns | ❌ static | hash-space | Universal byte keys + batch queries |
| **PgmIndex** | ~5 B/key | 150–700 ns | ❌ static | ✅ semantic | u64 range queries (timestamps, IDs) |
| **PgmIndexU128** | ~30 B/key | 80 ns | ❌ static | ✅ semantic | UUID/SHA range queries |

Numbers from 10M-key bench on i7-12700, AVX2 enabled, hugepages off.

---

## Choosing an index

> **The flowchart**: do you need **range queries**? If yes → PgmIndex (u64) or HybridIndex (bytes).
> Otherwise → PtrHash25 (use `lean_mph` if your queries are always valid keys).

### Decision table

| Question | Yes | No |
|---|---|---|
| All queries from the build set? (closed world: dictionary, vocab, deduped IDs) | **PtrHash25-lean** | PtrHash25 |
| **Need frequent insert/delete?** | **DynamicIndex** (LSM-tree on MPH tiers) | use a static engine |
| Need `range(min, max)` over byte keys? | HybridIndex (hash-space) | … |
| Need `range(min, max)` over **u64** with proper ordering? | **PgmIndex** | … |
| 16-byte keys (UUID, SHA-128, IPv6) + range? | **PgmIndexU128** | use PtrHash25 |

### Use-case recipes

**Bioinformatics: VCF indexer (600M variants, 20K genes, range queries)**

```rust
use kira_kv_engine::{IndexBuilder, PgmBuilder};

// 600M rsIDs → file offset (closed set after build → lean)
let rs_index = IndexBuilder::new()
    .with_lean_mph(true)               // 0.6 B/key, 14 ns warm
    .build_index(rs_keys)?;

// Variant positions (chrom<<56 | pos) → file offset, with range queries
let pos_index = PgmBuilder::new()
    .with_epsilon(64)
    .with_bloom_filter(true)
    .build(packed_positions)?;

// 20K gene names → coordinate region (lean, ~12 KB total)
let gene_index = IndexBuilder::new()
    .with_lean_mph(true)
    .build_index(gene_names)?;

// "All variants in chr1:1M-2M" — semantic range over u64 keys.
let variants_in_region: Vec<usize> = pos_index.range(start, end);
```

**LLM token vocabulary (closed set, ~100K tokens)**

```rust
let token_index = IndexBuilder::new()
    .with_lean_mph(true)        // tokens fixed at training time
    .build_index(tokens)?;       // ~60 KB, 14 ns lookup
```

**URL shortener (open world — invalid lookups happen)**

```rust
let url_index = IndexBuilder::new()
    // lean_mph=false (default) — returns KeyNotFound for foreign IDs
    .build_index(short_codes)?;  // 4.5 B/key, 21 ns warm + miss safety
```

**Real-time analytics with skewed access (Zipfian)**

```rust
use kira_kv_engine::DynamicHotTier;

let static_index = /* build base index */;
let dynamic_tier = DynamicHotTier::new(None, 1024 /* hot capacity */, 100_000 /* rebuild every */);
// Periodically: dynamic_tier.take_top_k() → build new HotTier → dynamic_tier.install()
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

let id_alice = idx.insert(b"alice".to_vec());    // stable u32 id, ~700 ns
let id_bob   = idx.insert(b"bob".to_vec());

assert_eq!(idx.lookup(b"alice"), Some(id_alice));
idx.delete(b"alice");
assert_eq!(idx.lookup(b"alice"), None);

// Explicit compact merges all tiers + buffer into one → restores
// single-tier lookup latency (~25 ns).
idx.compact();
```

Stable IDs **never** change across flush/compact — safe to store as external
pointers (e.g. file offsets, row indexes).

**General byte/string keys with range queries**

```rust
use kira_kv_engine::HybridBuilder;

let hybrid = HybridBuilder::new()
    .with_pgm_epsilon(2048)
    .with_lean(true)            // 13 B/key
    .build_from_u64(&u64_keys)?; // SIMD-accelerated build path
let positions = hybrid.lookup_batch_u64_simd(&query_keys);  // 30-50 ns/key
```

---

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

All engines are static (built once from a unique key set). Each returns a stable
`u32`/`usize` id in `[0..n)` (or `[0..1.10·n)` for u8-pilot near-minimal variants).

### `IndexBuilder` (PtrHash25 — point lookups, any key type)

```rust
IndexBuilder::new()
    .with_lean_mph(true)               // ★ -87% memory (closed-world only)
    .with_parallel_build(true)
    .with_build_fast_profile(true)
    .build_index(keys)?
```

Lookups: `index.lookup(&[u8])`, `lookup_u64(u64)`, `lookup_batch_pipelined(&[&[u8]])`,
`lookup_batch_u64_simd(&[u64])`.

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
    .with_epsilon(64)
    .with_bloom_filter(true)           // negative-fast
    .with_elias_fano(true)             // -40% key memory
    .with_target_lookup_ns(50)         // auto-tune ε
    .build(sorted_u64_keys)?
```

Lookups: `pgm.index(u64)`, `range(min, max)`, `lower_bound`, `upper_bound`.

### `PgmIndexU128` (16-byte keys: UUID/SHA-128/IPv6)

```rust
use kira_kv_engine::PgmIndexU128;

let idx = PgmIndexU128::build_from_bytes16(&uuids_be, 64)?;
let pos = idx.index_bytes16(&query_uuid)?;
let range = idx.range(uuid_a, uuid_b);
```

## Serialization

```rust
// PtrHash25 (Index): zero-copy mmap or to_bytes/from_bytes
let bytes = index.to_bytes()?;
let restored = kira_kv_engine::Index::from_bytes(&bytes)?;

index.save_mmap("path.bin")?;
let restored = kira_kv_engine::Index::open_mmap("path.bin")?;

// PgmIndex
let bytes = pgm.to_bytes()?;
let restored = kira_kv_engine::PgmIndex::from_bytes(&bytes)?;
```

## Benchmarks

```bash
cargo run --release --example million_build   # PtrHash25 + lean comparison
cargo run --release --example pgm_bench       # PGM + HybridIndex variants
```

### 10M keys on i7-12700 (Alder Lake, AVX2, no hugepages)

**`million_build` (mixed byte keys: 40% numeric + 40% random strings + 20% shared-prefix)**

| Variant | Build ms | B/key | Warm ns | Cold ns | Throughput |
|---|---:|---:|---:|---:|---:|
| **PtrHash25-lean** ⭐ | **3884** | **0.59** | **13.49** | 15.44 | **74 M/s** |
| PtrHash25 (default) | 6184 | 4.46 | 21.24 | 19.05 | 47 M/s |

PtrHash25-lean approaches paper-PTHash 2025 memory (~4.7 bits/key vs ~2.6 bits/key in
the paper) while being 1.4× faster than default and 1.6× faster to build.

**`pgm_bench` (1M u64 keys, 60% clustered + 30% uniform + 10% sequential)**

| Variant | Build ms | B/key | Warm ns | Mix ns |
|---|---:|---:|---:|---:|
| Hybrid ε=2048 (u64-build + SIMD batch) | 18300 | 16.32 | **46** | **34** |
| Hybrid ε=8192 lean (u64+SIMD) | 19900 | 13.12 | 52 | 42 |
| PGM baseline | 6850 | 27.72 | 947 | 632 |
| PGM + Bloom | 5010 | 29.40 | 1010 | 648 |

## Performance tuning checklist

1. **Closed key set?** Enable `lean_mph(true)` → -87% memory, +10% speed.
2. **Hugepages on Linux/Windows?** -10–20 ns per lookup at 100M scale. See
   [Hugepages section](#hugepages-windows-linux) below.
3. **Read-heavy after build?** Use `save_mmap`/`open_mmap` — ~50 ms load vs
   ~5 s deserialize on 100M index.
4. **Batched lookups?** Use `lookup_batch_pipelined` (Mph) or
   `lookup_batch_u64_simd` (Hybrid) — 3-stream cache prefetch ladder.
5. **u64 keys to Hybrid?** Use `build_from_u64()` instead of `build()` — SIMD
   hash path.

## Hugepages (Windows / Linux)

On Linux: `echo always > /sys/kernel/mm/transparent_hugepage/enabled` or use
`madvise(MADV_HUGEPAGE)` (the engine does this automatically when supported).

On Windows you need explicit privilege:
1. `Win+R` → `secpol.msc`
2. *Local Policies* → *User Rights Assignment* → *Lock pages in memory*
3. Add your user (or `BUILTIN\Administrators`)
4. **Log out and log back in** — only new sessions inherit the privilege

The engine prints a guidance message at startup if hugepages are unavailable.

## Notes

- Input keys must be **unique**.
- Indexes are static — modifications require a rebuild.
- Public API reference: `API.md`.

## License

MIT
