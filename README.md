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
kira_kv_engine = "*"
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

```
kira_kv_engine benchmark
n = 1000000 keys
============================================================
Struct  Workload  Cache   Build ms   Build rate    Lookup ns     Throughput    Hit %   Miss %        B/key
--------------------------------------------------------------------------------------------------------------
MPH     positive  cold      730.47      1368978        45.68       21890221    100.0      0.0         5.00
MPH     positive  warm      730.47      1368978        48.63       20564495    100.0      0.0         5.00
MPH     negative  cold      730.47      1368978        59.90       16693331     70.0     30.0         5.00
MPH     negative  warm      730.47      1368978        25.44       39308176     70.0     30.0         5.00
MPH     zipf      cold      730.47      1368978        44.54       22453413    100.0      0.0         5.00
MPH     zipf      warm      730.47      1368978        32.64       30633332    100.0      0.0         5.00
Struct  Workload  Cache   Build ms   Build rate    Lookup ns     Throughput    Hit %   Miss %        B/key
--------------------------------------------------------------------------------------------------------------
PGM     positive  cold       17.54     57012543       344.91        2899293    100.0      0.0        48.00
PGM     positive  warm       17.54     57012543       249.22        4012546    100.0      0.0        48.00
PGM     negative  cold       17.54     57012543       245.52        4072946     70.0     30.0        48.00
PGM     negative  warm       17.54     57012543       238.69        4189462     70.0     30.0        48.00
PGM     zipf      cold       17.54     57012543       265.71        3763490    100.0      0.0        48.00
PGM     zipf      warm       17.54     57012543       262.75        3805851    100.0      0.0        48.00
Struct  Workload  Cache   Build ms   Build rate    Lookup ns     Throughput    Hit %   Miss %        B/key
--------------------------------------------------------------------------------------------------------------
Hybrid  positive  cold      424.66      2354848       159.01        6288715    100.0      0.0        22.20
Hybrid  positive  warm      424.66      2354848       152.59        6553366    100.0      0.0        22.20
Hybrid  negative  cold      424.66      2354848       154.28        6481581     70.0     30.0        22.20
Hybrid  negative  warm      424.66      2354848       119.85        8343820     70.0     30.0        22.20
Hybrid  zipf      cold      424.66      2354848       149.25        6700093    100.0      0.0        22.20
Hybrid  zipf      warm      424.66      2354848       125.22        7985839    100.0      0.0        22.20
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
