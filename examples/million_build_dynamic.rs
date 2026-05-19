//! DynamicIndex benchmark — exercises the insert/delete/lookup/flush/compact
//! API on a realistic byte-key workload.
//!
//! Scenarios:
//!   1. **Bulk insert**: insert 1M keys from scratch. Measures sustained
//!      insert throughput including periodic auto-flushes.
//!   2. **Steady-state ops**: after bulk, perform mixed insert/delete on a
//!      live index — measures the cost of "ongoing maintenance" ops.
//!   3. **Lookup vs tier count**: lookup before and after compact() to show
//!      the cost of multi-tier traversal vs single-tier.
//!   4. **Mixed read/write**: 80% read, 15% insert, 5% delete (typical
//!      online-cache shape).
//!   5. **HashMap baseline**: same ops on `hashbrown::HashMap` for comparison.
//!   6. **Memory overhead**: actual bytes/key for both engines.
//!
//! Run: `cargo run --release --example million_build_dynamic`

#![allow(clippy::too_many_arguments)]

use hashbrown::HashMap;
use kira_kv_engine::{DynamicConfig, DynamicIndex};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use std::time::Instant;

const N_KEYS: usize = 1_000_000;
const STEADY_OPS: usize = 100_000;
const LOOKUP_OPS: usize = 500_000;
const MIXED_OPS: usize = 200_000;
const GEN_SEED: u64 = 0xCAFE_BABE_DEAD_BEEF;

fn main() {
    println!("DynamicIndex benchmark");
    println!(
        "n = {} initial keys, steady = {}, lookup = {}, mixed = {}",
        N_KEYS, STEADY_OPS, LOOKUP_OPS, MIXED_OPS
    );
    println!("{}", "=".repeat(110));

    let mut rng = StdRng::seed_from_u64(GEN_SEED);
    let keys: Vec<Vec<u8>> = (0..N_KEYS)
        .map(|i| format!("key-{:09}-{}", i, rng.next_u64()).into_bytes())
        .collect();
    let extra_keys: Vec<Vec<u8>> = (0..STEADY_OPS.max(MIXED_OPS))
        .map(|i| format!("extra-{:09}-{}", i, rng.next_u64()).into_bytes())
        .collect();

    // Pre-generate lookup workload: 50/50 hit/miss, shuffled.
    let lookup_keys: Vec<Vec<u8>> = build_lookup_workload(&keys, LOOKUP_OPS, &mut rng);

    println!();
    println!("--- Scenario 1: Bulk insert ---");
    bench_bulk_insert(&keys);

    println!();
    println!("--- Scenario 2: Steady-state insert/delete ---");
    bench_steady_state(&keys, &extra_keys);

    println!();
    println!("--- Scenario 3: Lookup latency vs tier count ---");
    bench_lookup_tiered(&keys, &extra_keys, &lookup_keys);

    println!();
    println!("--- Scenario 4: Mixed 80% read / 15% insert / 5% delete ---");
    bench_mixed_workload(&keys, &extra_keys, &mut rng);

    println!();
    println!("--- Scenario 5: HashMap baseline ---");
    bench_hashmap_baseline(&keys, &extra_keys, &lookup_keys);

    println!();
    println!("Done.");
}

/// Builds a query list of `total` keys: half drawn from `existing` (positive),
/// half are unique strings guaranteed to miss.
fn build_lookup_workload(existing: &[Vec<u8>], total: usize, rng: &mut StdRng) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(total);
    for _ in 0..total / 2 {
        out.push(existing[rng.gen_range(0..existing.len())].clone());
    }
    for i in 0..total - total / 2 {
        out.push(format!("missing-{:09}-{}", i, rng.next_u64()).into_bytes());
    }
    out.shuffle(rng);
    out
}

fn bench_bulk_insert(keys: &[Vec<u8>]) {
    let mut idx = DynamicIndex::with_config(DynamicConfig {
        flush_threshold: 64 * 1024,
        max_tiers: 8,
        lean_tiers: false,
        parallel_build: true,
    });
    let t = Instant::now();
    for k in keys {
        idx.insert(k.clone());
    }
    let elapsed = t.elapsed().as_secs_f64();
    let throughput = keys.len() as f64 / elapsed;
    let per_op = (elapsed * 1e9) / keys.len() as f64;
    let mem = idx.memory_usage();
    println!(
        "  bulk insert: {} keys in {:.2} s → {:.0} K ops/s, {:.0} ns/op",
        keys.len(),
        elapsed,
        throughput / 1000.0,
        per_op,
    );
    println!(
        "  tiers: {}, buffer: {}, tombstones: {}",
        idx.tier_count(),
        idx.buffer_len(),
        idx.tombstone_count()
    );
    println!(
        "  memory: {:.2} MB ({:.2} bytes/key)",
        mem as f64 / 1_048_576.0,
        mem as f64 / keys.len() as f64
    );

    // Forced flush — show timing.
    let t = Instant::now();
    idx.flush();
    let flush_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("  forced flush: {:.1} ms → {} tiers", flush_ms, idx.tier_count());

    // Compact — measure cost of merging all tiers.
    let t = Instant::now();
    idx.compact();
    let compact_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  compact: {:.1} ms → {} tiers, memory now {:.2} MB",
        compact_ms,
        idx.tier_count(),
        idx.memory_usage() as f64 / 1_048_576.0
    );
}

fn bench_steady_state(initial: &[Vec<u8>], extra: &[Vec<u8>]) {
    let mut idx = DynamicIndex::with_config(DynamicConfig {
        flush_threshold: 64 * 1024,
        max_tiers: 8,
        lean_tiers: false,
        parallel_build: true,
    });
    // Pre-load initial set.
    for k in initial {
        idx.insert(k.clone());
    }
    idx.compact();

    // Insert STEADY_OPS new keys.
    let n = STEADY_OPS.min(extra.len());
    let t = Instant::now();
    for k in &extra[..n] {
        idx.insert(k.clone());
    }
    let ins_secs = t.elapsed().as_secs_f64();
    println!(
        "  steady insert: {} ops in {:.2} s → {:.0} K ops/s, {:.0} ns/op",
        n,
        ins_secs,
        n as f64 / ins_secs / 1000.0,
        ins_secs * 1e9 / n as f64,
    );
    println!("  tiers after inserts: {}", idx.tier_count());

    // Delete the same N keys.
    let t = Instant::now();
    for k in &extra[..n] {
        idx.delete(k);
    }
    let del_secs = t.elapsed().as_secs_f64();
    println!(
        "  steady delete: {} ops in {:.2} s → {:.0} K ops/s, {:.0} ns/op",
        n,
        del_secs,
        n as f64 / del_secs / 1000.0,
        del_secs * 1e9 / n as f64,
    );
    println!(
        "  tombstones: {}, memory: {:.2} MB",
        idx.tombstone_count(),
        idx.memory_usage() as f64 / 1_048_576.0
    );
}

fn bench_lookup_tiered(initial: &[Vec<u8>], extra: &[Vec<u8>], lookup_keys: &[Vec<u8>]) {
    let mut idx = DynamicIndex::with_config(DynamicConfig {
        flush_threshold: 32 * 1024, // smaller threshold → more tiers
        max_tiers: 32,              // allow many tiers before forced compact
        lean_tiers: false,
        parallel_build: true,
    });
    for k in initial {
        idx.insert(k.clone());
    }
    // Force several flushes by mixing in extra inserts.
    for k in extra.iter().take(STEADY_OPS) {
        idx.insert(k.clone());
    }
    idx.flush();
    let pre_tiers = idx.tier_count();

    // Warm: pre-iterate once.
    let _ = lookup_pass(&idx, lookup_keys);
    // Measure.
    let warm = lookup_pass(&idx, lookup_keys);
    let warm_ns = warm.0 * 1e9 / lookup_keys.len() as f64;
    println!(
        "  lookup (pre-compact, {} tiers): {:.2} ns/op, {} hits",
        pre_tiers, warm_ns, warm.1
    );

    // Compact and re-measure.
    let t = Instant::now();
    idx.compact();
    let compact_ms = t.elapsed().as_secs_f64() * 1000.0;
    let post_tiers = idx.tier_count();
    let _ = lookup_pass(&idx, lookup_keys);
    let post_warm = lookup_pass(&idx, lookup_keys);
    let post_ns = post_warm.0 * 1e9 / lookup_keys.len() as f64;
    println!(
        "  compact took {:.1} ms → {} tier",
        compact_ms, post_tiers
    );
    println!(
        "  lookup (post-compact, {} tier): {:.2} ns/op, {} hits — speedup {:.2}×",
        post_tiers,
        post_ns,
        post_warm.1,
        warm_ns / post_ns
    );
}

fn bench_mixed_workload(initial: &[Vec<u8>], extra: &[Vec<u8>], rng: &mut StdRng) {
    let mut idx = DynamicIndex::with_config(DynamicConfig {
        flush_threshold: 64 * 1024,
        max_tiers: 8,
        lean_tiers: false,
        parallel_build: true,
    });
    for k in initial {
        idx.insert(k.clone());
    }
    idx.compact();
    let baseline_mem = idx.memory_usage();
    let baseline_keys = initial.len();

    // Mixed workload: 80% read, 15% insert, 5% delete.
    let mut alive: Vec<Vec<u8>> = initial.to_vec();
    let t = Instant::now();
    let mut hits = 0usize;
    let mut inserts = 0usize;
    let mut deletes = 0usize;
    let mut extra_cursor = 0usize;
    for _ in 0..MIXED_OPS {
        let r = rng.gen_range(0..100u32);
        if r < 80 {
            // Read.
            let k = &alive[rng.gen_range(0..alive.len())];
            if idx.lookup(k).is_some() {
                hits += 1;
            }
        } else if r < 95 {
            // Insert.
            if extra_cursor < extra.len() {
                let k = extra[extra_cursor].clone();
                extra_cursor += 1;
                idx.insert(k.clone());
                alive.push(k);
                inserts += 1;
            }
        } else {
            // Delete.
            if !alive.is_empty() {
                let i = rng.gen_range(0..alive.len());
                let k = alive.swap_remove(i);
                idx.delete(&k);
                deletes += 1;
            }
        }
    }
    let elapsed = t.elapsed().as_secs_f64();
    let per_op = elapsed * 1e9 / MIXED_OPS as f64;
    println!(
        "  mixed: {} ops in {:.2} s → {:.0} K ops/s, {:.0} ns/op",
        MIXED_OPS,
        elapsed,
        MIXED_OPS as f64 / elapsed / 1000.0,
        per_op
    );
    println!(
        "  breakdown: {} reads ({} hits), {} inserts, {} deletes",
        MIXED_OPS - inserts - deletes,
        hits,
        inserts,
        deletes
    );
    println!(
        "  final: {} tiers, {} tombstones, memory {:.2} MB ({:+.2} MB vs baseline)",
        idx.tier_count(),
        idx.tombstone_count(),
        idx.memory_usage() as f64 / 1_048_576.0,
        (idx.memory_usage() as f64 - baseline_mem as f64) / 1_048_576.0
    );
    let _ = baseline_keys;
}

fn lookup_pass(idx: &DynamicIndex, keys: &[Vec<u8>]) -> (f64, usize) {
    let t = Instant::now();
    let mut hits = 0usize;
    for k in keys {
        if idx.lookup(k).is_some() {
            hits += 1;
        }
    }
    let elapsed = t.elapsed().as_secs_f64();
    std::hint::black_box(&hits);
    (elapsed, hits)
}

fn bench_hashmap_baseline(initial: &[Vec<u8>], extra: &[Vec<u8>], lookup_keys: &[Vec<u8>]) {
    let mut map: HashMap<Vec<u8>, u32> = HashMap::with_capacity(initial.len() + extra.len());

    // Bulk insert.
    let t = Instant::now();
    for (i, k) in initial.iter().enumerate() {
        map.insert(k.clone(), i as u32);
    }
    let bulk_secs = t.elapsed().as_secs_f64();
    println!(
        "  HashMap bulk insert: {:.2} s, {:.0} K ops/s",
        bulk_secs,
        initial.len() as f64 / bulk_secs / 1000.0
    );

    // Steady inserts.
    let n = STEADY_OPS.min(extra.len());
    let t = Instant::now();
    for (i, k) in extra[..n].iter().enumerate() {
        map.insert(k.clone(), (initial.len() + i) as u32);
    }
    let ins_secs = t.elapsed().as_secs_f64();
    println!(
        "  HashMap steady insert: {} ops, {:.0} ns/op",
        n,
        ins_secs * 1e9 / n as f64
    );

    // Lookup.
    let t = Instant::now();
    let mut hits = 0usize;
    for k in lookup_keys {
        if map.get(k).is_some() {
            hits += 1;
        }
    }
    let lk_secs = t.elapsed().as_secs_f64();
    println!(
        "  HashMap lookup: {} ops, {:.2} ns/op, {} hits",
        lookup_keys.len(),
        lk_secs * 1e9 / lookup_keys.len() as f64,
        hits
    );

    // Memory estimate: HashMap is `(K, V)` pairs + ~25% bucket overhead.
    let entries = map.len();
    let avg_key_len: usize =
        map.iter().take(1000).map(|(k, _)| k.capacity()).sum::<usize>() / 1000.max(1);
    let est_mem = entries * (avg_key_len + std::mem::size_of::<u32>() + 8 /* pointer */)
        + map.capacity() * std::mem::size_of::<(Vec<u8>, u32)>();
    println!(
        "  HashMap memory: ~{:.2} MB ({:.2} bytes/key, estimate)",
        est_mem as f64 / 1_048_576.0,
        est_mem as f64 / entries as f64
    );
}
