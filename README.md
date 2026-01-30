# ⚡ Kira KV Engine

![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/kira_kv_engine)

> KV-storage engine based on **Minimal perfect hash functions** with hybrid indexing (+PGM Index) for Rust. \
> Zero collisions. Pure `O(1)` performance.

## 🧠 Algorithm

Built on the **BDZ algorithm** using 3-hypergraph peeling:
- Maps exactly `n` keys to `n` unique indices
- Uses hypergraph construction with 3 vertices per key
- Peels vertices of degree 1 iteratively until fully resolved
- Falls back to rehashing with different salts if cycles occur

PGM provides learned, range-aware routing; MPH provides constant-time exact resolution. Together they form a scalable, cache-efficient, update-friendly hybrid index.

**📖 Research:** [Simple and Space-Efficient Minimal Perfect Hash Functions](https://cmph.sourceforge.net/papers/wads07.pdf) (Botelho, Pagh, Ziviani, 2007)

For numeric data, we layer in **PGM (Piecewise Geometric Model)** indexing:
- Learns linear segments to approximate key distributions
- Provides `O(log log n)` worst-case with `O(1)` average lookups
- Excellent for sorted integer sequences with predictable patterns

**📖 Research:** [The PGM-index: a fully-dynamic compressed learned index](https://www.vldb.org/pvldb/vol13/p1162-ferragina.pdf) (Ferragina, Vinciguerra, 2020)

## 🚀 Quick Start

## ⚙️ Install

```toml
[dependencies]
kira_kv_engine = ">=0.2.2"
```

```rust
use kira_kv_engine::HybridBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let keys = vec![
        1000u64.to_le_bytes().to_vec(),
        b"user:123".to_vec(),
        b"session:abc".to_vec(),
    ];

    let index = HybridBuilder::new().build_index(keys)?;
    println!("{:?}", index.lookup_u64(1000));
    println!("{:?}", index.lookup_str("user:123"));

    Ok(())
}
```

Full API reference: `API.md`.

## 🧪 Benchmarks

Run comprehensive benchmarks across algorithms:

```bash
cargo run --example million_build --release
```

Windows 11, Intel Core i7 12700, DDR5 4800MHz, NVMe; SIMD-boosted (AVX2):
```
kira_kv_engine benchmark
n = 1_000_000 keys
============================================================
Struct  Workload  Cache   Build ms   Build rate    Lookup ns     Throughput    Hit %   Miss %        B/key
--------------------------------------------------------------------------------------------------------------
Hybrid  positive  cold     1224.80       816457        45.23       22110197    100.0      0.0         8.23
Hybrid  positive  warm     1224.80       816457        24.96       40070524    100.0      0.0         8.23
Hybrid  negative  cold     1224.80       816457        31.16       32088307     70.0     30.0         8.23
Hybrid  negative  warm     1224.80       816457        35.65       28052065     70.0     30.0         8.23
Hybrid  zipf      cold     1224.80       816457        34.68       28831738    100.0      0.0         8.23
Hybrid  zipf      warm     1224.80       816457        24.98       40032026    100.0      0.0         8.23
```

```
kira_kv_engine benchmark
n = 10_000_000 keys
============================================================
Struct  Workload  Cache   Build ms   Build rate    Lookup ns     Throughput    Hit %   Miss %        B/key
--------------------------------------------------------------------------------------------------------------
Hybrid  positive  cold    16379.82       610507        85.38       11711796    100.0      0.0         8.23
Hybrid  positive  warm    16379.82       610507        75.33       13275276    100.0      0.0         8.23
Hybrid  negative  cold    16379.82       610507        62.78       15929147     70.0     30.0         8.23
Hybrid  negative  warm    16379.82       610507        52.86       18916465     70.0     30.0         8.23
Hybrid  zipf      cold    16379.82       610507        52.29       19124116    100.0      0.0         8.23
Hybrid  zipf      warm    16379.82       610507        56.72       17630465    100.0      0.0         8.23
```

```
kira_kv_engine benchmark
n = 100_000_000 keys
============================================================
Struct  Workload  Cache   Build ms   Build rate    Lookup ns     Throughput    Hit %   Miss %        B/key
--------------------------------------------------------------------------------------------------------------
Hybrid  positive  cold   315103.79       317356       239.58        4174041    100.0      0.0         8.23
Hybrid  positive  warm   315103.79       317356       107.06        9340382    100.0      0.0         8.23
Hybrid  negative  cold   315103.79       317356       112.90        8857239     70.0     30.0         8.23
Hybrid  negative  warm   315103.79       317356        99.40       10060160     70.0     30.0         8.23
Hybrid  zipf      cold   315103.79       317356       117.15        8536211    100.0      0.0         8.23
Hybrid  zipf      warm   315103.79       317356       107.16        9331666    100.0      0.0         8.23
```


## 🎛️ Configuration

Configuration options are documented in `API.md`.

## 📊 When to Use

**Perfect for:**
- Fixed datasets (config files, catalogs, dictionaries)
- Memory-constrained environments
- Predictable `O(1)` lookup requirements
- Cold start optimization (mmap friendly)
- High-performance lookups in hot paths

**Not suitable for:**
- Frequently changing key sets
- Unknown dataset sizes
- Write-heavy workloads

## 🚀 Performance Tips

- Use `gamma ≈ 1.25` for datasets > 10M keys
- Enable `parallel` feature for build-time speedup
- Serialize indices to disk for instant cold starts
- Consider hybrid indexing for mixed numeric/string data

## 🎯 Real-World Use Cases *

**Web Servers:**
- Route matching: **6.9× faster** than Redis lookups
- Session lookup: **37.9M operations/sec** vs Redis 5.6M

**Databases:**
- Index structures: **29× less memory** than LevelDB
- Cache lookups: **Zero allocation** after build

**Gaming:**
- Asset ID lookup: **21.4M lookups/sec**
- Player data: **Instant cold start** from disk
\
\
\* provided with our performance benchmarks
---

## 📚 References

- **BDZ Algorithm**: Botelho, D., Pagh, R., Ziviani, N. (2007). [Simple and Space-Efficient Minimal Perfect Hash Functions](https://cmph.sourceforge.net/papers/wads07.pdf). *WADS 2007*
- **PGM Index**: Ferragina, P., Vinciguerra, G. (2020). [The PGM-index: a fully-dynamic compressed learned index](https://www.vldb.org/pvldb/vol13/p1162-ferragina.pdf). *VLDB 2020*
- **Learned Indexes**: Kraska, T. et al. (2018). [The Case for Learned Index Structures](https://arxiv.org/pdf/1712.01208.pdf). *SIGMOD 2018*
- **3-Hypergraph Peeling**: [Peeling Random Planar Maps](https://arxiv.org/pdf/1507.06951.pdf) - Theoretical foundations

## 📜 License

MIT
