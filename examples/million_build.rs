use kira_kv_engine::BackendKind;
use kira_kv_engine::index::{IndexBuilder, IndexConfig};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use std::collections::HashSet;
use std::thread;
use std::time::Instant;

const N_KEYS: usize = 1_000_000;
const GEN_SEED: u64 = 42;
const QUERY_SEED: u64 = 1337;
const MISSING_POOL_FRACTION: f64 = 0.01;
const QUERY_OPS: usize = 50_000;
const USE_PARALLEL: bool = true;
const BUILD_FAST_PROFILE: bool = true;

#[derive(Clone)]
struct QueryBytes {
    key: Vec<u8>,
    is_hit: bool,
}

#[derive(Clone, Copy)]
struct QueryU64 {
    key: u64,
    is_hit: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("kira_kv_engine benchmark");
    println!("n = {} keys", N_KEYS);
    println!("{}", "=".repeat(60));

    run_index_bench()?;

    Ok(())
}

fn run_index_bench() -> Result<(), Box<dyn std::error::Error>> {
    print_table_header();

    let mixed_keys = gen_mixed_keys(N_KEYS, GEN_SEED ^ 0x6666_6666_6666_6666);
    let mut rng = StdRng::seed_from_u64(QUERY_SEED ^ 0x7777_7777_7777_7777);

    let mut key_set = HashSet::with_capacity(N_KEYS * 2);
    for k in &mixed_keys {
        key_set.insert(k.clone());
    }

    let positive_queries =
        make_positive_queries_bytes(&mixed_keys[..QUERY_OPS.min(mixed_keys.len())]);

    let missing_total = (QUERY_OPS as f64 * MISSING_POOL_FRACTION).ceil() as usize;
    let missing_keys = gen_mixed_keys_missing(missing_total, &mut rng, &key_set);
    let negative_queries =
        make_negative_queries_bytes(&mixed_keys, &missing_keys, QUERY_OPS, 0.70, &mut rng);

    let zipf_queries = make_zipfian_queries_bytes(&mixed_keys, QUERY_OPS, &mut rng);

    #[cfg(feature = "bbhash")]
    let mut backends = vec![
        (BackendKind::PtrHash2025, "PtrHash25Default"),
        (BackendKind::PTHash, "PTHash"),
        (BackendKind::PtrHash2025, "PtrHash25"),
        (BackendKind::CHD, "CHD"),
        (BackendKind::RecSplit, "RecSplit"),
    ];
    #[cfg(not(feature = "bbhash"))]
    let backends = vec![
        (BackendKind::PtrHash2025, "PtrHash25Default"),
        (BackendKind::PTHash, "PTHash"),
        (BackendKind::PtrHash2025, "PtrHash25"),
        (BackendKind::CHD, "CHD"),
        (BackendKind::RecSplit, "RecSplit"),
    ];
    #[cfg(feature = "bbhash")]
    backends.push((BackendKind::BBHash, "BBHash"));

    for (backend_kind, backend_name) in backends {
        let mut cfg = IndexConfig::default();
        cfg.auto_detect_numeric = false;
        cfg.backend = backend_kind;
        cfg.hot_fraction = 0.15;
        cfg.hot_backend = BackendKind::CHD;
        cfg.cold_backend = BackendKind::RecSplit;
        cfg.enable_parallel_build = true;
        cfg.build_fast_profile = BUILD_FAST_PROFILE;

        let t_build = Instant::now();
        let index = IndexBuilder::new()
            .with_config(cfg)
            .build_index(mixed_keys.clone())?;
        let build_s = t_build.elapsed().as_secs_f64();

        let stats = index.stats();
        let bytes_per_key = stats.total_memory as f64 / N_KEYS as f64;

        run_lookup_bytes_index(
            backend_name,
            "positive",
            &index,
            build_s,
            bytes_per_key,
            positive_queries.clone(),
        );

        run_lookup_bytes_index(
            backend_name,
            "negative",
            &index,
            build_s,
            bytes_per_key,
            negative_queries.clone(),
        );

        run_lookup_bytes_index(
            backend_name,
            "zipf",
            &index,
            build_s,
            bytes_per_key,
            zipf_queries.clone(),
        );
    }

    Ok(())
}

fn print_table_header() {
    println!(
        "{:<7} {:<9} {:<5} {:>10} {:>12} {:>12} {:>14} {:>8} {:>8} {:>12}",
        "Struct",
        "Workload",
        "Cache",
        "Build ms",
        "Build rate",
        "Lookup ns",
        "Throughput",
        "Hit %",
        "Miss %",
        "B/key"
    );
    println!("{}", "-".repeat(110));
}

fn run_lookup_bytes_mph(
    structure: &str,
    workload: &str,
    mph: &kira_kv_engine::index::Index,
    build_s: f64,
    bytes_per_key: f64,
    mut queries: Vec<QueryBytes>,
) {
    let (hits, misses) = count_hits_bytes(&queries);
    let mut rng = StdRng::seed_from_u64(QUERY_SEED ^ 0x8888_8888_8888_8888);

    let (cold_s, cold_acc) = measure_bytes_queries_batch(&mut queries, &mut rng, false, mph);
    std::hint::black_box(cold_acc);
    print_row(
        structure,
        workload,
        "cold",
        build_s,
        N_KEYS,
        cold_s,
        queries.len(),
        hits,
        misses,
        bytes_per_key,
    );

    let (warm_s, warm_acc) = measure_bytes_queries_batch(&mut queries, &mut rng, true, mph);
    std::hint::black_box(warm_acc);
    print_row(
        structure,
        workload,
        "warm",
        build_s,
        N_KEYS,
        warm_s,
        queries.len(),
        hits,
        misses,
        bytes_per_key,
    );
}

fn run_lookup_u64(
    structure: &str,
    workload: &str,
    pgm: &kira_kv_engine::index::Index,
    build_s: f64,
    bytes_per_key: f64,
    mut queries: Vec<QueryU64>,
) {
    let (hits, misses) = count_hits_u64(&queries);
    let mut rng = StdRng::seed_from_u64(QUERY_SEED ^ 0x9999_9999_9999_9999);

    let (cold_s, cold_acc) = measure_u64_queries(&mut queries, &mut rng, false, |k| {
        if let Ok(idx) = pgm.lookup_u64(k) {
            idx as u64
        } else {
            0
        }
    });
    std::hint::black_box(cold_acc);
    print_row(
        structure,
        workload,
        "cold",
        build_s,
        N_KEYS,
        cold_s,
        queries.len(),
        hits,
        misses,
        bytes_per_key,
    );

    let (warm_s, warm_acc) = measure_u64_queries(&mut queries, &mut rng, true, |k| {
        if let Ok(idx) = pgm.lookup_u64(k) {
            idx as u64
        } else {
            0
        }
    });
    std::hint::black_box(warm_acc);
    print_row(
        structure,
        workload,
        "warm",
        build_s,
        N_KEYS,
        warm_s,
        queries.len(),
        hits,
        misses,
        bytes_per_key,
    );
}

fn run_lookup_bytes_index(
    structure: &str,
    workload: &str,
    index: &kira_kv_engine::index::Index,
    build_s: f64,
    bytes_per_key: f64,
    mut queries: Vec<QueryBytes>,
) {
    let (hits, misses) = count_hits_bytes(&queries);
    let mut rng = StdRng::seed_from_u64(QUERY_SEED ^ 0xaaaa_aaaa_aaaa_aaaa);

    let (cold_s, cold_acc) = measure_bytes_queries_batch(&mut queries, &mut rng, false, index);
    std::hint::black_box(cold_acc);
    print_row(
        structure,
        workload,
        "cold",
        build_s,
        N_KEYS,
        cold_s,
        queries.len(),
        hits,
        misses,
        bytes_per_key,
    );

    let (warm_s, warm_acc) = measure_bytes_queries_batch(&mut queries, &mut rng, true, index);
    std::hint::black_box(warm_acc);
    print_row(
        structure,
        workload,
        "warm",
        build_s,
        N_KEYS,
        warm_s,
        queries.len(),
        hits,
        misses,
        bytes_per_key,
    );
}

fn print_row(
    structure: &str,
    workload: &str,
    cache: &str,
    build_s: f64,
    n_keys: usize,
    lookup_s: f64,
    ops: usize,
    hits: usize,
    misses: usize,
    bytes_per_key: f64,
) {
    let build_ms = build_s * 1000.0;
    let build_rate = n_keys as f64 / build_s;
    let lookup_ns = (lookup_s * 1e9) / ops as f64;
    let throughput = ops as f64 / lookup_s;
    let hit_rate = hits as f64 / ops as f64 * 100.0;
    let miss_rate = misses as f64 / ops as f64 * 100.0;

    println!(
        "{:<7} {:<9} {:<5} {:>10.2} {:>12.0} {:>12.2} {:>14.0} {:>8.1} {:>8.1} {:>12.2}",
        structure,
        workload,
        cache,
        build_ms,
        build_rate,
        lookup_ns,
        throughput,
        hit_rate,
        miss_rate,
        bytes_per_key
    );
}

fn measure_bytes_queries<F>(
    queries: &mut [QueryBytes],
    rng: &mut StdRng,
    warm: bool,
    mut f: F,
) -> (f64, u64)
where
    F: FnMut(&[u8]) -> u64,
{
    if warm {
        queries.shuffle(rng);
        let mut warm_acc = 0u64;
        for q in queries.iter() {
            let key = std::hint::black_box(q.key.as_slice());
            warm_acc ^= f(key);
        }
        std::hint::black_box(warm_acc);
    }

    queries.shuffle(rng);
    let t0 = Instant::now();
    let mut acc = 0u64;
    for q in queries.iter() {
        let key = std::hint::black_box(q.key.as_slice());
        acc ^= f(key);
    }
    let elapsed = t0.elapsed().as_secs_f64();

    (elapsed, acc)
}

fn measure_bytes_queries_batch(
    queries: &mut [QueryBytes],
    rng: &mut StdRng,
    warm: bool,
    index: &kira_kv_engine::index::Index,
) -> (f64, u64) {
    if warm {
        queries.shuffle(rng);
        let warm_acc = if USE_PARALLEL {
            parallel_batch_lookup(index, queries)
        } else {
            batch_lookup(index, queries)
        };
        std::hint::black_box(warm_acc);
    }

    queries.shuffle(rng);
    let t0 = Instant::now();
    let acc = if USE_PARALLEL {
        parallel_batch_lookup(index, queries)
    } else {
        batch_lookup(index, queries)
    };
    let elapsed = t0.elapsed().as_secs_f64();

    (elapsed, acc)
}

fn batch_lookup(index: &kira_kv_engine::index::Index, queries: &[QueryBytes]) -> u64 {
    let refs: Vec<&[u8]> = queries.iter().map(|q| q.key.as_slice()).collect();
    let mut acc = 0u64;
    for opt in index.lookup_batch(&refs) {
        if let Some(idx) = opt {
            acc ^= idx as u64;
        }
    }
    acc
}

fn parallel_batch_lookup(index: &kira_kv_engine::index::Index, queries: &[QueryBytes]) -> u64 {
    let threads = thread_count();
    let per = (queries.len() + threads - 1) / threads;
    let mut accs = Vec::new();
    thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..threads {
            let start = t * per;
            if start >= queries.len() {
                continue;
            }
            let end = (start + per).min(queries.len());
            let slice = &queries[start..end];
            let idx_ref = index;
            handles.push(s.spawn(move || {
                let refs: Vec<&[u8]> = slice.iter().map(|q| q.key.as_slice()).collect();
                let mut local = 0u64;
                for opt in idx_ref.lookup_batch(&refs) {
                    if let Some(idx) = opt {
                        local ^= idx as u64;
                    }
                }
                local
            }));
        }
        for h in handles {
            if let Ok(v) = h.join() {
                accs.push(v);
            }
        }
    });
    accs.into_iter().fold(0u64, |a, b| a ^ b)
}

fn measure_u64_queries<F>(
    queries: &mut [QueryU64],
    rng: &mut StdRng,
    warm: bool,
    mut f: F,
) -> (f64, u64)
where
    F: FnMut(u64) -> u64,
{
    if warm {
        queries.shuffle(rng);
        let mut warm_acc = 0u64;
        for q in queries.iter() {
            let key = std::hint::black_box(q.key);
            warm_acc ^= f(key);
        }
        std::hint::black_box(warm_acc);
    }

    queries.shuffle(rng);
    let t0 = Instant::now();
    let mut acc = 0u64;
    for q in queries.iter() {
        let key = std::hint::black_box(q.key);
        acc ^= f(key);
    }
    let elapsed = t0.elapsed().as_secs_f64();

    (elapsed, acc)
}

fn count_hits_bytes(queries: &[QueryBytes]) -> (usize, usize) {
    let hits = queries.iter().filter(|q| q.is_hit).count();
    (hits, queries.len() - hits)
}

fn count_hits_u64(queries: &[QueryU64]) -> (usize, usize) {
    let hits = queries.iter().filter(|q| q.is_hit).count();
    (hits, queries.len() - hits)
}

fn make_positive_queries_bytes(keys: &[Vec<u8>]) -> Vec<QueryBytes> {
    keys.iter()
        .map(|k| QueryBytes {
            key: k.clone(),
            is_hit: true,
        })
        .collect()
}

fn make_positive_queries_u64(keys: &[u64]) -> Vec<QueryU64> {
    keys.iter()
        .map(|&k| QueryU64 {
            key: k,
            is_hit: true,
        })
        .collect()
}

fn make_negative_queries_bytes(
    keys: &[Vec<u8>],
    missing: &[Vec<u8>],
    total: usize,
    hit_ratio: f64,
    rng: &mut StdRng,
) -> Vec<QueryBytes> {
    let mut queries = Vec::with_capacity(total);
    let hit_count = (total as f64 * hit_ratio) as usize;
    let miss_count = total - hit_count;

    for _ in 0..hit_count {
        let idx = rng.gen_range(0..keys.len());
        queries.push(QueryBytes {
            key: keys[idx].clone(),
            is_hit: true,
        });
    }
    for _ in 0..miss_count {
        let idx = rng.gen_range(0..missing.len());
        queries.push(QueryBytes {
            key: missing[idx].clone(),
            is_hit: false,
        });
    }

    queries
}

fn make_negative_queries_u64(
    keys: &[u64],
    missing: &[u64],
    total: usize,
    hit_ratio: f64,
    rng: &mut StdRng,
) -> Vec<QueryU64> {
    let mut queries = Vec::with_capacity(total);
    let hit_count = (total as f64 * hit_ratio) as usize;
    let miss_count = total - hit_count;

    for _ in 0..hit_count {
        let idx = rng.gen_range(0..keys.len());
        queries.push(QueryU64 {
            key: keys[idx],
            is_hit: true,
        });
    }
    for _ in 0..miss_count {
        let idx = rng.gen_range(0..missing.len());
        queries.push(QueryU64 {
            key: missing[idx],
            is_hit: false,
        });
    }

    queries
}

fn make_zipfian_queries_bytes(keys: &[Vec<u8>], total: usize, rng: &mut StdRng) -> Vec<QueryBytes> {
    let hot_count = (keys.len() / 5).max(1);
    let mut indices: Vec<usize> = (0..keys.len()).collect();
    indices.shuffle(rng);
    let hot_indices = &indices[..hot_count];

    let mut queries = Vec::with_capacity(total);
    for _ in 0..total {
        let use_hot = rng.gen_bool(0.80);
        let idx = if use_hot {
            let h = rng.gen_range(0..hot_indices.len());
            hot_indices[h]
        } else {
            rng.gen_range(0..keys.len())
        };
        queries.push(QueryBytes {
            key: keys[idx].clone(),
            is_hit: true,
        });
    }

    queries
}

fn make_zipfian_queries_u64(keys: &[u64], total: usize, rng: &mut StdRng) -> Vec<QueryU64> {
    let hot_count = (keys.len() / 5).max(1);
    let mut indices: Vec<usize> = (0..keys.len()).collect();
    indices.shuffle(rng);
    let hot_indices = &indices[..hot_count];

    let mut queries = Vec::with_capacity(total);
    for _ in 0..total {
        let use_hot = rng.gen_bool(0.80);
        let idx = if use_hot {
            let h = rng.gen_range(0..hot_indices.len());
            hot_indices[h]
        } else {
            rng.gen_range(0..keys.len())
        };
        queries.push(QueryU64 {
            key: keys[idx],
            is_hit: true,
        });
    }

    queries
}

fn gen_random_strings(n: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut seen = HashSet::with_capacity(n * 2);
    if USE_PARALLEL {
        gen_random_strings_parallel(n, &mut rng, &mut seen, 8, 64)
    } else {
        gen_random_strings_with_range(n, &mut rng, &mut seen, 8, 64)
    }
}

fn gen_shared_prefix_strings(n: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut seen = HashSet::with_capacity(n * 2);
    if USE_PARALLEL {
        gen_shared_prefix_strings_parallel(n, &mut rng, &mut seen, 8, 64)
    } else {
        gen_shared_prefix_strings_with_range(n, &mut rng, &mut seen, 8, 64)
    }
}

fn gen_numeric_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(n);

    let n_uniform = n * 50 / 100;
    let n_zipf = n * 30 / 100;
    let n_cluster = n - n_uniform - n_zipf;

    if USE_PARALLEL {
        keys.extend(gen_numeric_uniform_parallel(n_uniform, rng.next_u64()));
        keys.extend(gen_numeric_zipf_parallel(n_zipf, rng.next_u64()));
        keys.extend(gen_numeric_cluster_parallel(n_cluster, &mut rng));
    } else {
        for _ in 0..n_uniform {
            keys.push(rng.next_u64());
        }
        let zipf_domain = (n_zipf / 4).max(5_000).min(50_000) as u64;
        let zipf_seed = rng.next_u64();
        for _ in 0..n_zipf {
            let rank = sample_zipf_fast(&mut rng, zipf_domain, 1.07);
            let v = splitmix64(rank ^ zipf_seed);
            keys.push(v);
        }
        let cluster_count = (n_cluster / 32_768).max(4);
        let clusters = build_clusters(cluster_count, &mut rng);
        for _ in 0..n_cluster {
            let (start, len) = clusters[rng.gen_range(0..clusters.len())];
            let offset = rng.gen_range(0..len);
            let v = start.wrapping_add(offset);
            keys.push(v);
        }
    }

    keys.sort_unstable();
    keys.dedup();
    if keys.len() < n {
        let extra = n - keys.len();
        for _ in 0..extra {
            keys.push(rng.next_u64());
        }
        keys.sort_unstable();
        keys.dedup();
    }
    keys.truncate(n);
    keys
}

fn gen_mixed_keys(n: usize, seed: u64) -> Vec<Vec<u8>> {
    let n_numeric = n * 40 / 100;
    let n_random = n * 40 / 100;
    let n_shared = n - n_numeric - n_random;

    let numeric_keys = gen_numeric_keys(n_numeric, seed ^ 0xdead_beef_cafe_babe);
    let mut keys = Vec::with_capacity(n);

    for num in numeric_keys {
        keys.push(num.to_le_bytes().to_vec());
    }

    let mut rng = StdRng::seed_from_u64(seed ^ 0x1234_5678_9abc_def0);
    let mut seen_strings = HashSet::with_capacity((n_random + n_shared) * 2);

    let random_strings = if USE_PARALLEL {
        gen_random_strings_parallel(n_random, &mut rng, &mut seen_strings, 9, 64)
    } else {
        gen_random_strings_with_range(n_random, &mut rng, &mut seen_strings, 9, 64)
    };
    let shared_strings = if USE_PARALLEL {
        gen_shared_prefix_strings_parallel(n_shared, &mut rng, &mut seen_strings, 9, 64)
    } else {
        gen_shared_prefix_strings_with_range(n_shared, &mut rng, &mut seen_strings, 9, 64)
    };

    keys.extend(random_strings);
    keys.extend(shared_strings);

    keys.shuffle(&mut rng);
    keys
}

fn gen_random_strings_with_range(
    n: usize,
    rng: &mut StdRng,
    seen: &mut HashSet<Vec<u8>>,
    min_len: usize,
    max_len: usize,
) -> Vec<Vec<u8>> {
    let mut keys = Vec::with_capacity(n);
    while keys.len() < n {
        let len = rng.gen_range(min_len..=max_len);
        let mut key = vec![0u8; len];
        rng.fill_bytes(&mut key);
        if seen.insert(key.clone()) {
            keys.push(key);
        }
    }
    keys
}

fn gen_shared_prefix_strings_with_range(
    n: usize,
    rng: &mut StdRng,
    seen: &mut HashSet<Vec<u8>>,
    min_len: usize,
    max_len: usize,
) -> Vec<Vec<u8>> {
    let mut keys = Vec::with_capacity(n);
    let prefix_len = rng.gen_range(16..=32);
    let mut prefix = vec![0u8; prefix_len];
    rng.fill_bytes(&mut prefix);

    let shared_count = rng.gen_range((n as f64 * 0.60) as usize..=(n as f64 * 0.80) as usize);
    let mut shared_done = 0;

    while shared_done < shared_count {
        let len = rng.gen_range(prefix_len.max(min_len)..=max_len);
        let mut key = vec![0u8; len];
        key[..prefix_len].copy_from_slice(&prefix);
        rng.fill_bytes(&mut key[prefix_len..]);
        if seen.insert(key.clone()) {
            keys.push(key);
            shared_done += 1;
        }
    }

    while keys.len() < n {
        let len = rng.gen_range(min_len..=max_len);
        let mut key = vec![0u8; len];
        rng.fill_bytes(&mut key);
        if seen.insert(key.clone()) {
            keys.push(key);
        }
    }

    keys
}

fn gen_random_strings_parallel(
    n: usize,
    rng: &mut StdRng,
    seen: &mut HashSet<Vec<u8>>,
    min_len: usize,
    max_len: usize,
) -> Vec<Vec<u8>> {
    let threads = thread_count();
    let per = (n + threads - 1) / threads;
    let mut handles = Vec::new();
    for t in 0..threads {
        let count = per.min(n.saturating_sub(t * per));
        if count == 0 {
            continue;
        }
        let seed = rng.next_u64() ^ ((t as u64) << 32);
        handles.push(thread::spawn(move || {
            let mut local_rng = StdRng::seed_from_u64(seed);
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let len = local_rng.gen_range(min_len..=max_len);
                let mut key = vec![0u8; len];
                local_rng.fill_bytes(&mut key);
                out.push(key);
            }
            out
        }));
    }

    let mut keys = Vec::with_capacity(n);
    for h in handles {
        if let Ok(part) = h.join() {
            for key in part {
                if seen.insert(key.clone()) {
                    keys.push(key);
                }
            }
        }
    }

    if keys.len() < n {
        let missing = n - keys.len();
        keys.extend(gen_random_strings_with_range(
            missing, rng, seen, min_len, max_len,
        ));
    }

    keys
}

fn gen_shared_prefix_strings_parallel(
    n: usize,
    rng: &mut StdRng,
    seen: &mut HashSet<Vec<u8>>,
    min_len: usize,
    max_len: usize,
) -> Vec<Vec<u8>> {
    let prefix_len = rng.gen_range(16..=32);
    let mut prefix = vec![0u8; prefix_len];
    rng.fill_bytes(&mut prefix);

    let shared_count = rng.gen_range((n as f64 * 0.60) as usize..=(n as f64 * 0.80) as usize);
    let threads = thread_count();
    let per = (shared_count + threads - 1) / threads;
    let mut handles = Vec::new();
    for t in 0..threads {
        let count = per.min(shared_count.saturating_sub(t * per));
        if count == 0 {
            continue;
        }
        let seed = rng.next_u64() ^ (0x9E37_79B9_7F4A_7C15 ^ (t as u64));
        let prefix_clone = prefix.clone();
        handles.push(thread::spawn(move || {
            let mut local_rng = StdRng::seed_from_u64(seed);
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let len = local_rng.gen_range(prefix_clone.len().max(min_len)..=max_len);
                let mut key = vec![0u8; len];
                key[..prefix_clone.len()].copy_from_slice(&prefix_clone);
                local_rng.fill_bytes(&mut key[prefix_clone.len()..]);
                out.push(key);
            }
            out
        }));
    }

    let mut keys = Vec::with_capacity(n);
    for h in handles {
        if let Ok(part) = h.join() {
            for key in part {
                if seen.insert(key.clone()) {
                    keys.push(key);
                }
            }
        }
    }

    let mut shared_done = keys.len();
    while shared_done < shared_count {
        let len = rng.gen_range(prefix_len.max(min_len)..=max_len);
        let mut key = vec![0u8; len];
        key[..prefix_len].copy_from_slice(&prefix);
        rng.fill_bytes(&mut key[prefix_len..]);
        if seen.insert(key.clone()) {
            keys.push(key);
            shared_done += 1;
        }
    }

    while keys.len() < n {
        let len = rng.gen_range(min_len..=max_len);
        let mut key = vec![0u8; len];
        rng.fill_bytes(&mut key);
        if seen.insert(key.clone()) {
            keys.push(key);
        }
    }

    keys
}

fn gen_random_strings_missing(
    n: usize,
    rng: &mut StdRng,
    existing: &HashSet<Vec<u8>>,
    min_len: usize,
    max_len: usize,
) -> Vec<Vec<u8>> {
    let mut keys = Vec::with_capacity(n);
    while keys.len() < n {
        let len = rng.gen_range(min_len..=max_len);
        let mut key = vec![0u8; len];
        rng.fill_bytes(&mut key);
        if !existing.contains(&key) {
            keys.push(key);
        }
    }
    keys
}

fn gen_shared_prefix_strings_missing(
    n: usize,
    rng: &mut StdRng,
    existing: &HashSet<Vec<u8>>,
    min_len: usize,
    max_len: usize,
) -> Vec<Vec<u8>> {
    let mut keys = Vec::with_capacity(n);
    let prefix_len = rng.gen_range(16..=32);
    let mut prefix = vec![0u8; prefix_len];
    rng.fill_bytes(&mut prefix);

    let shared_count = rng.gen_range((n as f64 * 0.60) as usize..=(n as f64 * 0.80) as usize);
    let mut shared_done = 0;

    while shared_done < shared_count {
        let len = rng.gen_range(prefix_len.max(min_len)..=max_len);
        let mut key = vec![0u8; len];
        key[..prefix_len].copy_from_slice(&prefix);
        rng.fill_bytes(&mut key[prefix_len..]);
        if !existing.contains(&key) {
            keys.push(key);
            shared_done += 1;
        }
    }

    while keys.len() < n {
        let len = rng.gen_range(min_len..=max_len);
        let mut key = vec![0u8; len];
        rng.fill_bytes(&mut key);
        if !existing.contains(&key) {
            keys.push(key);
        }
    }

    keys
}

fn gen_numeric_keys_missing(
    n: usize,
    rng: &mut StdRng,
    existing: &HashSet<u64>,
    seed: u64,
) -> Vec<u64> {
    let mut keys = Vec::with_capacity(n);

    let n_uniform = n * 50 / 100;
    let n_zipf = n * 30 / 100;
    let n_cluster = n - n_uniform - n_zipf;

    let mut local_rng = StdRng::seed_from_u64(seed);

    let mut count = 0;
    while count < n_uniform {
        let v = local_rng.next_u64();
        if !existing.contains(&v) {
            keys.push(v);
            count += 1;
        }
    }

    let zipf_domain = (n_zipf / 4).max(5_000).min(50_000) as u64;
    let zipf_seed = local_rng.next_u64();
    count = 0;
    while count < n_zipf {
        let rank = sample_zipf_fast(rng, zipf_domain, 1.07);
        let v = splitmix64(rank ^ zipf_seed);
        if !existing.contains(&v) {
            keys.push(v);
            count += 1;
        }
    }

    let cluster_count = (n_cluster / 32_768).max(4);
    let clusters = build_clusters(cluster_count, rng);
    count = 0;
    while count < n_cluster {
        let (start, len) = clusters[rng.gen_range(0..clusters.len())];
        let offset = rng.gen_range(0..len);
        let v = start.wrapping_add(offset);
        if !existing.contains(&v) {
            keys.push(v);
            count += 1;
        }
    }

    keys
}

fn gen_mixed_keys_missing(n: usize, rng: &mut StdRng, existing: &HashSet<Vec<u8>>) -> Vec<Vec<u8>> {
    let n_numeric = n * 40 / 100;
    let n_random = n * 40 / 100;
    let n_shared = n - n_numeric - n_random;

    let mut missing = Vec::with_capacity(n);

    let mut numeric_existing = HashSet::new();
    for key in existing {
        if key.len() == 8 {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(key);
            numeric_existing.insert(u64::from_le_bytes(arr));
        }
    }

    let numeric_seed = rng.next_u64() ^ 0xface_cafe_dead_beef;
    let numeric_missing = gen_numeric_keys_missing(n_numeric, rng, &numeric_existing, numeric_seed);
    for num in numeric_missing {
        let key = num.to_le_bytes().to_vec();
        if !existing.contains(&key) {
            missing.push(key);
        }
    }

    let random_missing = gen_random_strings_missing(n_random, rng, existing, 9, 64);
    let shared_missing = gen_shared_prefix_strings_missing(n_shared, rng, existing, 9, 64);

    for k in random_missing {
        missing.push(k);
    }
    for k in shared_missing {
        missing.push(k);
    }

    missing
}

fn sample_zipf_fast(rng: &mut StdRng, max_rank: u64, s: f64) -> u64 {
    let u = rng.r#gen::<f64>().max(f64::MIN_POSITIVE);
    let inv = u.powf(-1.0 / (s - 1.0));
    let rank = inv as u64;
    rank.clamp(1, max_rank)
}

fn build_clusters(n: usize, rng: &mut StdRng) -> Vec<(u64, u64)> {
    let mut clusters = Vec::with_capacity(n);
    let mut base = rng.next_u64();
    for _ in 0..n {
        let len = rng.gen_range(256..=4096) as u64;
        clusters.push((base, len));
        let gap = rng.gen_range(1_000_000..=10_000_000) as u64;
        base = base.wrapping_add(len + gap);
    }
    clusters
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn gen_numeric_uniform_parallel(n: usize, seed: u64) -> Vec<u64> {
    let threads = thread_count();
    let per = (n + threads - 1) / threads;
    let mut handles = Vec::new();
    for t in 0..threads {
        let count = per.min(n.saturating_sub(t * per));
        if count == 0 {
            continue;
        }
        let seed_t = seed ^ (0xA24B_1F6F_1234_5678 ^ (t as u64));
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(seed_t);
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                out.push(rng.next_u64());
            }
            out
        }));
    }

    let mut keys = Vec::with_capacity(n);
    for h in handles {
        if let Ok(part) = h.join() {
            keys.extend(part);
        }
    }
    keys
}

fn gen_numeric_zipf_parallel(n: usize, seed: u64) -> Vec<u64> {
    let threads = thread_count();
    let per = (n + threads - 1) / threads;
    let mut handles = Vec::new();
    let zipf_domain = (n / 4).max(5_000).min(50_000) as u64;
    for t in 0..threads {
        let count = per.min(n.saturating_sub(t * per));
        if count == 0 {
            continue;
        }
        let seed_t = seed ^ (0x9E37_79B9_7F4A_7C15 ^ (t as u64));
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(seed_t);
            let zipf_seed = rng.next_u64();
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let rank = sample_zipf_fast(&mut rng, zipf_domain, 1.07);
                out.push(splitmix64(rank ^ zipf_seed));
            }
            out
        }));
    }

    let mut keys = Vec::with_capacity(n);
    for h in handles {
        if let Ok(part) = h.join() {
            keys.extend(part);
        }
    }
    keys
}

fn gen_numeric_cluster_parallel(n: usize, rng: &mut StdRng) -> Vec<u64> {
    let threads = thread_count();
    let per = (n + threads - 1) / threads;
    let cluster_count = (n / 32_768).max(4);
    let clusters = build_clusters(cluster_count, rng);
    let mut handles = Vec::new();
    for t in 0..threads {
        let count = per.min(n.saturating_sub(t * per));
        if count == 0 {
            continue;
        }
        let seed_t = rng.next_u64() ^ (0xD1B5_4A32_D192_ED03 ^ (t as u64));
        let clusters_clone = clusters.clone();
        handles.push(thread::spawn(move || {
            let mut local_rng = StdRng::seed_from_u64(seed_t);
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let (start, len) = clusters_clone[local_rng.gen_range(0..clusters_clone.len())];
                let offset = local_rng.gen_range(0..len);
                out.push(start.wrapping_add(offset));
            }
            out
        }));
    }

    let mut keys = Vec::with_capacity(n);
    for h in handles {
        if let Ok(part) = h.join() {
            keys.extend(part);
        }
    }
    keys
}

fn thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
}
