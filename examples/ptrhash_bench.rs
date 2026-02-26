use kira_kv_engine::BackendKind;
use kira_kv_engine::index::{IndexBuilder, IndexConfig};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use std::collections::HashSet;
use std::time::Instant;

const DATASET_KEYS: usize = 1_000_000;
const QUERY_OPS: usize = 50_000;
const MISSING_POOL_FRACTION: f64 = 0.01;
const NEGATIVE_HIT_RATIO: f64 = 0.70;

const NUMERIC_SHARE_PCT: usize = 40;
const RANDOM_BYTES_SHARE_PCT: usize = 40;
const SHARED_PREFIX_SHARE_PCT: usize = 20;

const MIN_STR_LEN: usize = 9;
const MAX_STR_LEN: usize = 64;
const SHARED_PREFIX_MIN: usize = 16;
const SHARED_PREFIX_MAX: usize = 32;
const ZIPF_HOT_RATIO: f64 = 0.80;
const ZIPF_HOT_DIVISOR: usize = 5;

const GEN_SEED: u64 = 42;
const QUERY_SEED: u64 = 1337;
const MIN_BUILD_RATE_KEYS_PER_SEC: f64 = 0.0;

#[derive(Clone)]
struct QueryBytes {
    key: Vec<u8>,
    is_hit: bool,
}

#[derive(Clone, Copy)]
struct WorkloadMetrics {
    cold_ns: f64,
    warm_ns: f64,
    hit_rate: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let keys = gen_mixed_keys(DATASET_KEYS, GEN_SEED ^ 0x6666_6666_6666_6666);
    let mut query_rng = StdRng::seed_from_u64(QUERY_SEED ^ 0x7777_7777_7777_7777);

    let mut key_set = HashSet::with_capacity(keys.len() * 2);
    for k in &keys {
        key_set.insert(k.clone());
    }

    let positive_queries = make_positive_queries_bytes(&keys[..QUERY_OPS.min(keys.len())]);
    let missing_total = (QUERY_OPS as f64 * MISSING_POOL_FRACTION).ceil() as usize;
    let missing_keys = gen_mixed_keys_missing(missing_total, &mut query_rng, &key_set);
    let negative_queries = make_negative_queries_bytes(
        &keys,
        &missing_keys,
        QUERY_OPS,
        NEGATIVE_HIT_RATIO,
        &mut query_rng,
    );
    let zipf_queries = make_zipfian_queries_bytes(&keys, QUERY_OPS, &mut query_rng);

    println!("backend,build_ms,keys_per_sec,bits_per_key,pos_warm_ns,neg_warm_ns,zipf_warm_ns");

    let backends = [
        BackendKind::PtrHash2025,
        BackendKind::PTHash,
        BackendKind::PtrHash2025,
        BackendKind::CHD,
        BackendKind::RecSplit,
    ];

    for backend in backends {
        let cfg = IndexConfig {
            auto_detect_numeric: false,
            backend,
            hot_fraction: 0.15,
            hot_backend: BackendKind::CHD,
            cold_backend: BackendKind::RecSplit,
            enable_parallel_build: true,
            build_fast_profile: true,
            ..IndexConfig::default()
        };

        let t0 = Instant::now();
        let index = IndexBuilder::new()
            .with_config(cfg)
            .build_index(keys.clone())?;
        let build_s = t0.elapsed().as_secs_f64();
        let build_ms = build_s * 1000.0;
        let keys_per_sec = DATASET_KEYS as f64 / build_s;

        let pos = run_workload(
            &index,
            positive_queries.clone(),
            QUERY_SEED ^ 0x1111_1111_1111_1111,
        );
        let neg = run_workload(
            &index,
            negative_queries.clone(),
            QUERY_SEED ^ 0x2222_2222_2222_2222,
        );
        let zipf = run_workload(
            &index,
            zipf_queries.clone(),
            QUERY_SEED ^ 0x3333_3333_3333_3333,
        );

        let stats = index.stats();
        let bits_per_key = if DATASET_KEYS > 0 {
            (stats.mph_memory as f64 * 8.0) / DATASET_KEYS as f64
        } else {
            0.0
        };

        println!(
            "{:?},{:.2},{:.0},{:.3},{:.2},{:.2},{:.2}",
            backend, build_ms, keys_per_sec, bits_per_key, pos.warm_ns, neg.warm_ns, zipf.warm_ns
        );

        assert!(
            keys_per_sec >= MIN_BUILD_RATE_KEYS_PER_SEC,
            "backend {:?} build rate {:.0} below {:.0}",
            backend,
            keys_per_sec,
            MIN_BUILD_RATE_KEYS_PER_SEC
        );
    }

    Ok(())
}

fn run_workload(
    index: &kira_kv_engine::index::Index,
    mut queries: Vec<QueryBytes>,
    seed: u64,
) -> WorkloadMetrics {
    let mut rng = StdRng::seed_from_u64(seed);

    queries.shuffle(&mut rng);
    let cold_t0 = Instant::now();
    let mut cold_hits = 0usize;
    let mut sink = 0usize;
    for q in &queries {
        if let Ok(v) = index.lookup(&q.key) {
            cold_hits += 1;
            sink ^= v;
        }
    }
    let cold_s = cold_t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);

    queries.shuffle(&mut rng);
    let mut warm_sink = 0usize;
    for q in &queries {
        if let Ok(v) = index.lookup(&q.key) {
            warm_sink ^= v;
        }
    }
    std::hint::black_box(warm_sink);

    queries.shuffle(&mut rng);
    let warm_t0 = Instant::now();
    let mut warm_hits = 0usize;
    let mut wsink = 0usize;
    for q in &queries {
        if let Ok(v) = index.lookup(&q.key) {
            warm_hits += 1;
            wsink ^= v;
        }
    }
    let warm_s = warm_t0.elapsed().as_secs_f64();
    std::hint::black_box(wsink);

    WorkloadMetrics {
        cold_ns: cold_s * 1e9 / queries.len() as f64,
        warm_ns: warm_s * 1e9 / queries.len() as f64,
        hit_rate: (cold_hits.max(warm_hits) as f64 / queries.len() as f64) * 100.0,
    }
}

fn make_positive_queries_bytes(keys: &[Vec<u8>]) -> Vec<QueryBytes> {
    keys.iter()
        .map(|k| QueryBytes {
            key: k.clone(),
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

fn make_zipfian_queries_bytes(keys: &[Vec<u8>], total: usize, rng: &mut StdRng) -> Vec<QueryBytes> {
    let hot_count = (keys.len() / ZIPF_HOT_DIVISOR).max(1);
    let mut indices: Vec<usize> = (0..keys.len()).collect();
    indices.shuffle(rng);
    let hot_indices = &indices[..hot_count];

    let mut queries = Vec::with_capacity(total);
    for _ in 0..total {
        let use_hot = rng.gen_bool(ZIPF_HOT_RATIO);
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

fn gen_mixed_keys(n: usize, seed: u64) -> Vec<Vec<u8>> {
    let n_numeric = n * NUMERIC_SHARE_PCT / 100;
    let n_random = n * RANDOM_BYTES_SHARE_PCT / 100;
    let n_shared = n - n_numeric - n_random;

    let numeric_keys = gen_numeric_keys(n_numeric, seed ^ 0xdead_beef_cafe_babe);
    let mut keys = Vec::with_capacity(n);

    for num in numeric_keys {
        keys.push(num.to_le_bytes().to_vec());
    }

    let mut rng = StdRng::seed_from_u64(seed ^ 0x1234_5678_9abc_def0);
    let mut seen_strings = HashSet::with_capacity((n_random + n_shared) * 2);

    let random_strings = gen_random_strings_with_range(
        n_random,
        &mut rng,
        &mut seen_strings,
        MIN_STR_LEN,
        MAX_STR_LEN,
    );
    let shared_strings = gen_shared_prefix_strings_with_range(
        n_shared,
        &mut rng,
        &mut seen_strings,
        MIN_STR_LEN,
        MAX_STR_LEN,
    );

    keys.extend(random_strings);
    keys.extend(shared_strings);
    keys.shuffle(&mut rng);
    keys
}

fn gen_numeric_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut seen = HashSet::with_capacity(n * 2);
    let mut keys = Vec::with_capacity(n);
    while keys.len() < n {
        let x = rng.next_u64();
        if seen.insert(x) {
            keys.push(x);
        }
    }
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
    let prefix_len = rng.gen_range(SHARED_PREFIX_MIN..=SHARED_PREFIX_MAX);
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

fn gen_mixed_keys_missing(n: usize, rng: &mut StdRng, existing: &HashSet<Vec<u8>>) -> Vec<Vec<u8>> {
    let n_numeric = n * NUMERIC_SHARE_PCT / 100;
    let n_random = n * RANDOM_BYTES_SHARE_PCT / 100;
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

    while missing.len() < n_numeric {
        let v = rng.next_u64();
        if !numeric_existing.contains(&v) {
            missing.push(v.to_le_bytes().to_vec());
        }
    }

    let random_missing =
        gen_random_strings_missing(n_random, rng, existing, MIN_STR_LEN, MAX_STR_LEN);
    let shared_missing =
        gen_shared_prefix_strings_missing(n_shared, rng, existing, MIN_STR_LEN, MAX_STR_LEN);

    missing.extend(random_missing);
    missing.extend(shared_missing);
    missing
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
    let prefix_len = rng.gen_range(SHARED_PREFIX_MIN..=SHARED_PREFIX_MAX);
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
