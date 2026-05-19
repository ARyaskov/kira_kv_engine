//! PGM benchmark — exercises the learned-index path with all the new options
//! wired in `pgm.rs`.
//!
//! Forces `auto_detect_numeric = true` and feeds pure 8-byte u64 keys so the
//! Index actually picks the PGM engine (otherwise it falls back to PtrHash25
//! and the PGM code never runs — that's what `million_build` measures).
//!
//! Comparisons (controlled by the `variants` table):
//!   - PGM baseline (no Bloom, no EF, no auto-tune)
//!   - PGM + Bloom        (fast negative path)
//!   - PGM + Elias-Fano   (compact keys)
//!   - PGM + Bloom + EF   (memory + negative-lookup wins combined)
//!   - PGM + auto-tune    (ε auto-pick for target latency)
//!
//! Run: `cargo run --release --example pgm_bench`

#![allow(clippy::too_many_arguments)]

use kira_kv_engine::HybridBuilder;
use kira_kv_engine::index::{IndexBuilder, IndexConfig};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use std::collections::HashSet;
use std::time::Instant;

const N_KEYS: usize = 10_000_00;
const QUERY_OPS: usize = 1_000_000;
const RUNS: usize = 1;
const GEN_SEED: u64 = 0xCAFE_BABE_DEAD_BEEF;
const QUERY_SEED: u64 = 0x1234_5678_9ABC_DEF0;

#[derive(Clone, Copy)]
struct VariantCfg {
    name: &'static str,
    bloom: bool,
    ef: bool,
    target_ns: Option<u32>,
}

fn main() {
    println!("PGM benchmark");
    println!("n = {} u64 keys, queries = {}, runs = {}", N_KEYS, QUERY_OPS, RUNS);
    println!("workload: 60% clustered + 30% uniform + 10% sequential (realistic timestamp-like)");
    println!("{}", "=".repeat(110));

    // Generate sorted unique u64 keys (clustered to favor PGM segmentation).
    let mut keys = gen_clustered_u64(N_KEYS, GEN_SEED);
    keys.sort_unstable();
    keys.dedup();
    while keys.len() < N_KEYS {
        // Pad with random unique to reach N exactly.
        let extra = N_KEYS - keys.len();
        let mut rng = StdRng::seed_from_u64(GEN_SEED ^ extra as u64);
        for _ in 0..extra * 2 {
            keys.push(rng.next_u64());
        }
        keys.sort_unstable();
        keys.dedup();
    }
    keys.truncate(N_KEYS);

    // Queries: half positive (existing key), half negative (random).
    let mut rng = StdRng::seed_from_u64(QUERY_SEED);
    let pos_count = QUERY_OPS / 2;
    let neg_count = QUERY_OPS - pos_count;
    let mut positives: Vec<u64> = (0..pos_count)
        .map(|_| keys[rng.gen_range(0..keys.len())])
        .collect();
    let key_set: HashSet<u64> = keys.iter().copied().collect();
    let mut negatives: Vec<u64> = Vec::with_capacity(neg_count);
    while negatives.len() < neg_count {
        let v = rng.next_u64();
        if !key_set.contains(&v) {
            negatives.push(v);
        }
    }
    drop(key_set);

    let mut mixed: Vec<u64> = positives.iter().chain(negatives.iter()).copied().collect();
    mixed.shuffle(&mut rng);
    positives.shuffle(&mut rng);

    let variants = &[
        VariantCfg { name: "PGM baseline",     bloom: false, ef: false, target_ns: None },
        VariantCfg { name: "PGM + Bloom",      bloom: true,  ef: false, target_ns: None },
        VariantCfg { name: "PGM + EF",         bloom: false, ef: true,  target_ns: None },
        VariantCfg { name: "PGM + Bloom + EF", bloom: true,  ef: true,  target_ns: None },
        VariantCfg { name: "PGM auto-tune 30ns", bloom: false, ef: false, target_ns: Some(30) },
    ];

    print_header();
    // Convert u64 keys to byte vec form expected by IndexBuilder.
    let keys_bytes: Vec<[u8; 8]> = keys.iter().map(|k| k.to_le_bytes()).collect();
    let pos_bytes: Vec<[u8; 8]> = positives.iter().map(|k| k.to_le_bytes()).collect();
    let mix_bytes: Vec<[u8; 8]> = mixed.iter().map(|k| k.to_le_bytes()).collect();

    for &v in variants {
        let mut build_samples = Vec::with_capacity(RUNS);
        let mut mem_samples = Vec::with_capacity(RUNS);
        let mut pos_warm = Vec::with_capacity(RUNS);
        let mut pos_cold = Vec::with_capacity(RUNS);
        let mut mix_warm = Vec::with_capacity(RUNS);

        for _ in 0..RUNS {
            let mut cfg = IndexConfig::default();
            cfg.auto_detect_numeric = true;
            cfg.pgm_epsilon = 64;
            cfg.pgm_enable_bloom = v.bloom;
            cfg.pgm_enable_elias_fano = v.ef;
            cfg.pgm_target_lookup_ns = v.target_ns;
            cfg.enable_parallel_build = true;

            let t = Instant::now();
            let owned: Vec<Vec<u8>> = keys_bytes.iter().map(|b| b.to_vec()).collect();
            let index = IndexBuilder::new()
                .with_config(cfg)
                .build_index(owned)
                .expect("build");
            let build_s = t.elapsed().as_secs_f64();
            build_samples.push(build_s);
            mem_samples.push(index.stats().total_memory as f64 / N_KEYS as f64);

            // Cold: shuffle once, run once.
            let q_pos: Vec<&[u8]> = pos_bytes.iter().map(|b| b.as_slice()).collect();
            let cold = run_batch(&index, &q_pos);
            pos_cold.push(cold);

            // Warm: pre-run once to fill caches, then measure.
            let _ = run_batch(&index, &q_pos);
            let warm = run_batch(&index, &q_pos);
            pos_warm.push(warm);

            // Mixed (50/50 positive/negative): warm.
            let q_mix: Vec<&[u8]> = mix_bytes.iter().map(|b| b.as_slice()).collect();
            let _ = run_batch(&index, &q_mix);
            let warm_mix = run_batch(&index, &q_mix);
            mix_warm.push(warm_mix);

            drop(index);
        }

        let build_med = median(&build_samples) * 1000.0; // ms
        let bpk = median(&mem_samples);
        let pos_cold_ns = ns_per_op(median(&pos_cold), QUERY_OPS / 2);
        let pos_warm_ns = ns_per_op(median(&pos_warm), QUERY_OPS / 2);
        let mix_warm_ns = ns_per_op(median(&mix_warm), QUERY_OPS);

        println!(
            "{:<22} {:>9.0} {:>10.2} {:>10.2} {:>11.2} {:>10.2}",
            v.name, build_med, bpk, pos_cold_ns, pos_warm_ns, mix_warm_ns
        );
    }

    // ---- Bonus: HybridIndex sweep (PGM-bucketed MPH) ----
    println!();
    println!("{}", "=".repeat(110));
    println!("HybridIndex (PGM + per-segment mini-MPH)");
    println!("{:<22} {:>9} {:>10} {:>10} {:>11} {:>10}", "Variant", "Build ms", "B/key", "pos cold ns", "pos warm ns", "mix ns");
    println!("{}", "-".repeat(80));

    let hybrid_variants = &[
        ("Hybrid ε=2048 (bytes-build)", 2048u32, false, false),
        ("Hybrid ε=2048 (u64-build)",    2048u32, false, true),
        ("Hybrid ε=2048 lean (u64)",     2048u32, true,  true),
        ("Hybrid ε=8192 lean (u64)",     8192u32, true,  true),
    ];
    for (name, eps, lean, use_u64_build) in hybrid_variants {
        let mut build_samples = Vec::with_capacity(RUNS);
        let mut mem_samples = Vec::with_capacity(RUNS);
        let mut pos_cold = Vec::with_capacity(RUNS);
        let mut pos_warm = Vec::with_capacity(RUNS);
        let mut mix_warm = Vec::with_capacity(RUNS);

        for _ in 0..RUNS {
            let t = Instant::now();
            let idx = if *use_u64_build {
                // SIMD-accelerated u64 build path.
                HybridBuilder::new()
                    .with_pgm_epsilon(*eps)
                    .with_linear_threshold(64)
                    .with_lean(*lean)
                    .build_from_u64(&keys)
                    .expect("hybrid u64 build")
            } else {
                let owned: Vec<Vec<u8>> = keys_bytes.iter().map(|b| b.to_vec()).collect();
                HybridBuilder::new()
                    .with_pgm_epsilon(*eps)
                    .with_linear_threshold(64)
                    .with_lean(*lean)
                    .build(&owned)
                    .expect("hybrid build")
            };
            let build_s = t.elapsed().as_secs_f64();
            build_samples.push(build_s);
            mem_samples.push(idx.memory_usage() as f64 / N_KEYS as f64);

            // SIMD-batch lookup on positives + mixed.
            let cold = run_hybrid_simd_batch(&idx, &positives);
            pos_cold.push(cold);
            let _ = run_hybrid_simd_batch(&idx, &positives);
            let warm = run_hybrid_simd_batch(&idx, &positives);
            pos_warm.push(warm);

            let _ = run_hybrid_simd_batch(&idx, &mixed);
            let warm_mix = run_hybrid_simd_batch(&idx, &mixed);
            mix_warm.push(warm_mix);

            if !lean && eps == &hybrid_variants[0].1 {
                idx.storage_stats().print_summary();
            }
            drop(idx);
        }

        let build_med = median(&build_samples) * 1000.0;
        let bpk = median(&mem_samples);
        let pos_cold_ns = ns_per_op(median(&pos_cold), QUERY_OPS / 2);
        let pos_warm_ns = ns_per_op(median(&pos_warm), QUERY_OPS / 2);
        let mix_warm_ns = ns_per_op(median(&mix_warm), QUERY_OPS);
        println!(
            "{:<22} {:>9.0} {:>10.2} {:>10.2} {:>11.2} {:>10.2}",
            name, build_med, bpk, pos_cold_ns, pos_warm_ns, mix_warm_ns
        );
    }

    println!();
    println!("Notes:");
    println!(" - 'B/key' = total index bytes / N. EF/Bloom variants trade ~1 byte per key for fewer wide queries.");
    println!(" - 'pos cold' = first lookup pass after fresh build (cold L1/L2/L3).");
    println!(" - 'pos warm' = second/third pass (L2/L3 resident).");
    println!(" - 'mix warm' = 50/50 positive+negative; Bloom variant should shine here.");
    println!(" - Hybrid compares vs PtrHash25 baseline (run `cargo run --release --example million_build`).");
}

fn run_hybrid_batch(idx: &kira_kv_engine::HybridIndex, q: &[&[u8]]) -> f64 {
    let t = Instant::now();
    let mut acc = 0u64;
    for k in q {
        if let Some(p) = idx.lookup(k) {
            acc ^= p as u64;
        }
    }
    let elapsed = t.elapsed().as_secs_f64();
    std::hint::black_box(acc);
    elapsed
}

fn run_hybrid_simd_batch(idx: &kira_kv_engine::HybridIndex, q: &[u64]) -> f64 {
    let t = Instant::now();
    let res = idx.lookup_batch_u64_simd(q);
    let elapsed = t.elapsed().as_secs_f64();
    std::hint::black_box(res);
    elapsed
}

fn print_header() {
    println!(
        "{:<22} {:>9} {:>10} {:>10} {:>11} {:>10}",
        "Variant", "Build ms", "B/key", "pos cold ns", "pos warm ns", "mix ns"
    );
    println!("{}", "-".repeat(80));
}

fn run_batch(index: &kira_kv_engine::index::Index, q: &[&[u8]]) -> f64 {
    let t = Instant::now();
    let res = index.lookup_batch_pipelined(q);
    let elapsed = t.elapsed().as_secs_f64();
    std::hint::black_box(res);
    elapsed
}

fn ns_per_op(secs: f64, ops: usize) -> f64 {
    (secs * 1e9) / ops as f64
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
}

/// 60% clustered + 30% uniform + 10% sequential. Mimics timestamp/sequential-ID
/// workloads where PGM is genuinely useful.
fn gen_clustered_u64(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_cluster = n * 60 / 100;
    let n_uniform = n * 30 / 100;
    let n_seq = n - n_cluster - n_uniform;

    let mut keys = Vec::with_capacity(n);

    // Sequential block: monotonic timestamps.
    let mut t = 1_700_000_000_000u64;
    for _ in 0..n_seq {
        t += rng.gen_range(1..=100);
        keys.push(t);
    }
    // Clustered: many bursts around random anchors.
    let cluster_count = (n_cluster / 256).max(16);
    for _ in 0..cluster_count {
        let base = rng.next_u64();
        let len = rng.gen_range(64..=512);
        for i in 0..len {
            keys.push(base.wrapping_add(i as u64 * rng.gen_range(1..=8)));
        }
    }
    // Uniform random.
    for _ in 0..n_uniform {
        keys.push(rng.next_u64());
    }

    keys
}
