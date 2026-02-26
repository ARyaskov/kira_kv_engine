use kira_kv_engine::BackendKind;
use kira_kv_engine::index::{IndexBuilder, IndexConfig};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use std::collections::HashSet;
use std::env;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const N_KEYS: usize = 1_000_000;
const GEN_SEED: u64 = 42;
const QUERY_SEED: u64 = 1337;
const MISSING_POOL_FRACTION: f64 = 0.01;
const QUERY_OPS: usize = 50_000;
const USE_PARALLEL: bool = true;
const BUILD_FAST_PROFILE: bool = true;
const DEFAULT_BENCH_RUNS: usize = 7;

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

#[derive(Clone)]
struct BenchSettings {
    runs: usize,
    threads: usize,
    core_ids: Option<Vec<usize>>,
}

#[derive(Clone, Copy)]
struct SummaryStats {
    median: f64,
    p95: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_bench_settings();
    apply_bench_settings(&settings);

    println!("kira_kv_engine benchmark");
    println!("n = {} keys", N_KEYS);
    println!(
        "bench: runs={}, threads={}, core_ids={}",
        settings.runs,
        settings.threads,
        format_core_ids(settings.core_ids.as_ref())
    );
    println!("{}", "=".repeat(60));

    run_index_bench(&settings)?;

    Ok(())
}

fn run_index_bench(settings: &BenchSettings) -> Result<(), Box<dyn std::error::Error>> {
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
        let mut build_samples = Vec::with_capacity(settings.runs);
        let mut bpk_samples = Vec::with_capacity(settings.runs);
        let mut pos_cold = Vec::with_capacity(settings.runs);
        let mut pos_warm = Vec::with_capacity(settings.runs);
        let mut neg_cold = Vec::with_capacity(settings.runs);
        let mut neg_warm = Vec::with_capacity(settings.runs);
        let mut zipf_cold = Vec::with_capacity(settings.runs);
        let mut zipf_warm = Vec::with_capacity(settings.runs);

        for run in 0..settings.runs {
            let mut cfg = IndexConfig::default();
            cfg.auto_detect_numeric = false;
            cfg.backend = backend_kind;
            cfg.hot_fraction = 0.15;
            cfg.hot_backend = BackendKind::CHD;
            cfg.cold_backend = BackendKind::RecSplit;
            cfg.enable_parallel_build = true;
            cfg.build_fast_profile = BUILD_FAST_PROFILE;
            cfg.mph_config.gamma = 1.2;

            let t_build = Instant::now();
            let index = IndexBuilder::new()
                .with_config(cfg)
                .build_index(mixed_keys.clone())?;
            let build_s = t_build.elapsed().as_secs_f64();
            build_samples.push(build_s);
            bpk_samples.push(index.stats().total_memory as f64 / N_KEYS as f64);

            let mut run_rng = StdRng::seed_from_u64(
                QUERY_SEED
                    ^ 0xaaaa_aaaa_aaaa_aaaa
                    ^ (run as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );

            let mut q = positive_queries.clone();
            let (c, w) = measure_lookup_pair(&mut q, &mut run_rng, &index);
            pos_cold.push(c);
            pos_warm.push(w);

            let mut q = negative_queries.clone();
            let (c, w) = measure_lookup_pair(&mut q, &mut run_rng, &index);
            neg_cold.push(c);
            neg_warm.push(w);

            let mut q = zipf_queries.clone();
            let (c, w) = measure_lookup_pair(&mut q, &mut run_rng, &index);
            zipf_cold.push(c);
            zipf_warm.push(w);
        }

        let build_stats = summarize(&mut build_samples);
        let build_rate_median = N_KEYS as f64 / build_stats.median;
        if backend_kind == BackendKind::CHD && build_rate_median < 500_000.0 {
            eprintln!(
                "PERFORMANCE REGRESSION: {:.0} keys/s median (< 500k) for {}",
                build_rate_median, backend_name
            );
            std::process::exit(1);
        }

        let bytes_per_key = summarize(&mut bpk_samples).median;
        let (hits_pos, misses_pos) = count_hits_bytes(&positive_queries);
        let (hits_neg, misses_neg) = count_hits_bytes(&negative_queries);
        let (hits_zipf, misses_zipf) = count_hits_bytes(&zipf_queries);

        print_row(
            backend_name,
            "positive",
            "cold",
            settings.runs,
            build_stats,
            summarize(&mut pos_cold),
            QUERY_OPS,
            hits_pos,
            misses_pos,
            bytes_per_key,
        );
        print_row(
            backend_name,
            "positive",
            "warm",
            settings.runs,
            build_stats,
            summarize(&mut pos_warm),
            QUERY_OPS,
            hits_pos,
            misses_pos,
            bytes_per_key,
        );
        print_row(
            backend_name,
            "negative",
            "cold",
            settings.runs,
            build_stats,
            summarize(&mut neg_cold),
            QUERY_OPS,
            hits_neg,
            misses_neg,
            bytes_per_key,
        );
        print_row(
            backend_name,
            "negative",
            "warm",
            settings.runs,
            build_stats,
            summarize(&mut neg_warm),
            QUERY_OPS,
            hits_neg,
            misses_neg,
            bytes_per_key,
        );
        print_row(
            backend_name,
            "zipf",
            "cold",
            settings.runs,
            build_stats,
            summarize(&mut zipf_cold),
            QUERY_OPS,
            hits_zipf,
            misses_zipf,
            bytes_per_key,
        );
        print_row(
            backend_name,
            "zipf",
            "warm",
            settings.runs,
            build_stats,
            summarize(&mut zipf_warm),
            QUERY_OPS,
            hits_zipf,
            misses_zipf,
            bytes_per_key,
        );
    }

    Ok(())
}

fn print_table_header() {
    println!(
        "{:<7} {:<9} {:<5} {:>4} {:>10} {:>10} {:>11} {:>11} {:>11} {:>11} {:>7} {:>7} {:>10}",
        "Struct",
        "Workload",
        "Cache",
        "Runs",
        "Build m",
        "Build p95",
        "Rate m",
        "Lookup m",
        "Lookup p95",
        "Thr m",
        "Hit %",
        "Miss %",
        "B/key"
    );
    println!("{}", "-".repeat(146));
}

fn print_row(
    structure: &str,
    workload: &str,
    cache: &str,
    runs: usize,
    build_stats: SummaryStats,
    lookup_stats: SummaryStats,
    ops: usize,
    hits: usize,
    misses: usize,
    bytes_per_key: f64,
) {
    let build_ms_m = build_stats.median * 1000.0;
    let build_ms_p95 = build_stats.p95 * 1000.0;
    let build_rate_m = N_KEYS as f64 / build_stats.median;
    let lookup_ns_m = (lookup_stats.median * 1e9) / ops as f64;
    let lookup_ns_p95 = (lookup_stats.p95 * 1e9) / ops as f64;
    let throughput_m = ops as f64 / lookup_stats.median;
    let hit_rate = hits as f64 / ops as f64 * 100.0;
    let miss_rate = misses as f64 / ops as f64 * 100.0;

    println!(
        "{:<7} {:<9} {:<5} {:>4} {:>10.2} {:>10.2} {:>11.0} {:>11.2} {:>11.2} {:>11.0} {:>7.1} {:>7.1} {:>10.2}",
        structure,
        workload,
        cache,
        runs,
        build_ms_m,
        build_ms_p95,
        build_rate_m,
        lookup_ns_m,
        lookup_ns_p95,
        throughput_m,
        hit_rate,
        miss_rate,
        bytes_per_key
    );
}

fn measure_lookup_pair(
    queries: &mut [QueryBytes],
    rng: &mut StdRng,
    index: &kira_kv_engine::index::Index,
) -> (f64, f64) {
    let (cold_s, cold_acc) = measure_bytes_queries_batch(queries, rng, false, index);
    std::hint::black_box(cold_acc);
    let (warm_s, warm_acc) = measure_bytes_queries_batch(queries, rng, true, index);
    std::hint::black_box(warm_acc);
    (cold_s, warm_s)
}

fn summarize(samples: &mut [f64]) -> SummaryStats {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    SummaryStats {
        median: percentile_sorted(samples, 0.5),
        p95: percentile_sorted(samples, 0.95),
    }
}

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    sorted[idx]
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
    let mut refs: Vec<&[u8]> = queries.iter().map(|q| q.key.as_slice()).collect();

    if warm {
        refs.shuffle(rng);
        let warm_acc = if USE_PARALLEL {
            parallel_batch_lookup(index, &refs)
        } else {
            batch_lookup(index, &refs)
        };
        std::hint::black_box(warm_acc);
    }

    refs.shuffle(rng);
    let t0 = Instant::now();
    let acc = if USE_PARALLEL {
        parallel_batch_lookup(index, &refs)
    } else {
        batch_lookup(index, &refs)
    };
    let elapsed = t0.elapsed().as_secs_f64();

    (elapsed, acc)
}

fn batch_lookup(index: &kira_kv_engine::index::Index, refs: &[&[u8]]) -> u64 {
    let mut acc = 0u64;
    for opt in index.lookup_batch(refs) {
        if let Some(idx) = opt {
            acc ^= idx as u64;
        }
    }
    acc
}

fn parallel_batch_lookup(index: &kira_kv_engine::index::Index, refs: &[&[u8]]) -> u64 {
    let threads = thread_count();
    let per = (refs.len() + threads - 1) / threads;
    let mut accs = Vec::new();
    let core_ids = Arc::new(configured_core_ids());
    thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..threads {
            let start = t * per;
            if start >= refs.len() {
                continue;
            }
            let end = (start + per).min(refs.len());
            let slice = &refs[start..end];
            let idx_ref = index;
            let core_ids_cloned = Arc::clone(&core_ids);
            handles.push(s.spawn(move || {
                pin_thread_to_core(core_ids_cloned.as_ref(), t);
                let mut local = 0u64;
                for opt in idx_ref.lookup_batch(slice) {
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
    env::var("KIRA_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(1)
        })
}

fn load_bench_settings() -> BenchSettings {
    let threads = thread_count();
    let runs = env::var("KIRA_BENCH_RUNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_BENCH_RUNS);
    let core_ids = parse_core_ids(env::var("KIRA_BENCH_CORE_IDS").ok().as_deref())
        .or_else(|| default_core_ids(threads));
    BenchSettings {
        runs,
        threads,
        core_ids,
    }
}

fn apply_bench_settings(settings: &BenchSettings) {
    unsafe {
        env::set_var("KIRA_BUILD_THREADS", settings.threads.to_string());
        env::set_var("KIRA_BENCH_THREADS", settings.threads.to_string());
    }
    if let Some(core_ids) = settings.core_ids.as_ref() {
        let ids = format_core_ids(Some(core_ids));
        unsafe {
            env::set_var("KIRA_BUILD_CORE_IDS", &ids);
            env::set_var("KIRA_BENCH_CORE_IDS", &ids);
        }
        pin_thread_to_core(core_ids, 0);
    }
}

fn parse_core_ids(v: Option<&str>) -> Option<Vec<usize>> {
    let text = v?;
    let ids = text
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    if ids.is_empty() { None } else { Some(ids) }
}

fn default_core_ids(threads: usize) -> Option<Vec<usize>> {
    let all = core_affinity::get_core_ids()?;
    if all.is_empty() {
        return None;
    }
    let n = threads.min(all.len());
    Some(all.into_iter().take(n).map(|c| c.id).collect())
}

fn configured_core_ids() -> Vec<usize> {
    parse_core_ids(env::var("KIRA_BENCH_CORE_IDS").ok().as_deref())
        .or_else(|| default_core_ids(thread_count()))
        .unwrap_or_default()
}

fn format_core_ids(core_ids: Option<&Vec<usize>>) -> String {
    match core_ids {
        Some(ids) if !ids.is_empty() => ids
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(","),
        _ => "none".to_string(),
    }
}

fn pin_thread_to_core(core_ids: &[usize], thread_idx: usize) {
    if core_ids.is_empty() {
        return;
    }
    let core_id = core_ids[thread_idx % core_ids.len()];
    let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
}
