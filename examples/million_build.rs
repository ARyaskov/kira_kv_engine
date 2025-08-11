use kira_kv_engine::{Builder, MphError};

use kira_kv_engine::hybrid::{HybridBuilder, HybridConfig};
#[cfg(feature = "pgm")]
use kira_kv_engine::pgm::PgmBuilder;

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::collections::HashSet;
use std::time::Instant;

const N_KEYS: usize = 1_000_000;
const GEN_SEED: u64 = 42;

fn main() -> Result<(), MphError> {
    println!("=== kira_kv_engine :: Hybrid MPH+PGM Benchmark ===");
    println!("n = {} keys", N_KEYS);
    println!("{}", "=".repeat(50));

    if let Err(e) = run_traditional_mph_benchmark() {
        eprintln!("[traditional] error: {e}");
    }

    #[cfg(feature = "pgm")]
    {
        if let Err(e) = run_hybrid_benchmark() {
            eprintln!("[hybrid] error: {e}");
        }
        if let Err(e) = run_pgm_only_benchmark() {
            eprintln!("[pgm-only] error: {e}");
        }
        if let Err(e) = run_comparison_benchmark() {
            eprintln!("[comparison] error: {e}");
        }
    }

    Ok(())
}

fn run_traditional_mph_benchmark() -> Result<(), MphError> {
    println!("🔥 TRADITIONAL MPH BENCHMARK");
    println!("{}", "=".repeat(50));

    // -------------------------
    // 1) Key Generation (optimized)
    // -------------------------
    let t0 = Instant::now();
    let keys = gen_string_keys_optimized(N_KEYS, GEN_SEED);
    let gen_s = t0.elapsed().as_secs_f64();
    println!(
        "gen:    {:>8.3} s   ({:.1} M keys/s)",
        gen_s,
        N_KEYS as f64 / gen_s / 1e6
    );

    // -------------------------
    // 2) Build MPH (with all optimizations)
    // -------------------------
    let t1 = Instant::now();
    let mph = Builder::new().build(keys.iter().map(|v| v.as_slice()))?;
    let build_s = t1.elapsed().as_secs_f64();
    println!(
        "build:  {:>8.3} s   ({:.1} M keys/s)",
        build_s,
        N_KEYS as f64 / build_s / 1e6
    );

    // -------------------------
    // 3) Warm-up runs to stabilize performance
    // -------------------------
    println!("Warming up caches...");
    for _ in 0..3 {
        let mut acc: u64 = 0;
        for chunk in keys.chunks(32_768) {
            for k in chunk {
                acc ^= mph.index(k);
            }
        }
        std::hint::black_box(acc);
    }

    // -------------------------
    // 4) Single-threaded lookups
    // -------------------------
    let t2 = Instant::now();
    let mut acc: u64 = 0;

    // Optimized loop with unrolling
    let mut i = 0;
    while i + 8 <= keys.len() {
        acc ^= mph.index(&keys[i]);
        acc ^= mph.index(&keys[i + 1]);
        acc ^= mph.index(&keys[i + 2]);
        acc ^= mph.index(&keys[i + 3]);
        acc ^= mph.index(&keys[i + 4]);
        acc ^= mph.index(&keys[i + 5]);
        acc ^= mph.index(&keys[i + 6]);
        acc ^= mph.index(&keys[i + 7]);
        i += 8;
    }
    while i < keys.len() {
        acc ^= mph.index(&keys[i]);
        i += 1;
    }

    let lookup_s = t2.elapsed().as_secs_f64();
    println!(
        "lookup: {:>8.3} s   ({:.1} M lookups/s)   (acc={})",
        lookup_s,
        N_KEYS as f64 / lookup_s / 1e6,
        acc
    );

    // -------------------------
    // 5) Memory usage analysis
    // -------------------------
    let mph_size = std::mem::size_of_val(&mph) + mph.g.len() * std::mem::size_of::<u32>();
    let bytes_per_key = mph_size as f64 / N_KEYS as f64;
    let total_mb = mph_size as f64 / 1_048_576.0;

    println!();
    println!("Memory Analysis:");
    println!("  Total MPH size: {:.2} MB", total_mb);
    println!("  Bytes per key:  {:.2} B", bytes_per_key);
    println!("  Compression:    {:.1}x vs HashMap", 120.0 / bytes_per_key);

    let total_time = gen_s + build_s + lookup_s;
    println!();
    println!("Performance Summary:");
    println!("  Total time:     {:.3} s", total_time);
    println!(
        "  Build rate:     {:.1} M keys/s",
        N_KEYS as f64 / build_s / 1e6
    );
    println!(
        "  Lookup rate:    {:.1} M lookups/s",
        N_KEYS as f64 / lookup_s / 1e6
    );
    println!();

    Ok(())
}

#[cfg(feature = "pgm")]
fn run_hybrid_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 HYBRID MPH+PGM BENCHMARK");
    println!("{}", "=".repeat(50));

    let t0 = Instant::now();
    let mixed_keys = gen_mixed_keys(N_KEYS, GEN_SEED);
    let gen_s = t0.elapsed().as_secs_f64();
    println!(
        "gen:    {:>8.3} s   ({:.1} M keys/s)",
        gen_s,
        N_KEYS as f64 / gen_s / 1e6
    );

    let t1 = Instant::now();
    let hybrid = HybridBuilder::new()
        .with_config(HybridConfig::default())
        .build(mixed_keys.clone())?;
    let build_s = t1.elapsed().as_secs_f64();
    println!(
        "build:  {:>8.3} s   ({:.1} M keys/s)",
        build_s,
        N_KEYS as f64 / build_s / 1e6
    );

    // Warm-up
    println!("Warming up hybrid caches...");
    for _ in 0..3 {
        let mut acc: u64 = 0;
        for chunk in mixed_keys.chunks(32_768) {
            for k in chunk {
                if let Ok(idx) = hybrid.index(k) {
                    acc ^= idx as u64;
                }
            }
        }
        std::hint::black_box(acc);
    }

    // Benchmark lookups
    let t2 = Instant::now();
    let mut acc: u64 = 0;
    let mut found = 0;

    for key in &mixed_keys {
        if let Ok(idx) = hybrid.index(key) {
            acc ^= idx as u64;
            found += 1;
        }
    }

    let lookup_s = t2.elapsed().as_secs_f64();
    println!(
        "lookup: {:>8.3} s   ({:.1} M lookups/s)   (found={}, acc={})",
        lookup_s,
        found as f64 / lookup_s / 1e6,
        found,
        acc
    );

    println!();
    Ok(())
}

#[cfg(feature = "pgm")]
fn run_pgm_only_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 PGM-ONLY BENCHMARK (Numeric Keys)");
    println!("{}", "=".repeat(50));

    let t0 = Instant::now();
    let numeric_keys = gen_numeric_keys(N_KEYS, GEN_SEED);
    let gen_s = t0.elapsed().as_secs_f64();
    println!(
        "gen:    {:>8.3} s   ({:.1} M keys/s)",
        gen_s,
        N_KEYS as f64 / gen_s / 1e6
    );

    let t1 = Instant::now();
    let pgm = PgmBuilder::new()
        .with_epsilon(32)
        .build(numeric_keys.clone())?;
    let build_s = t1.elapsed().as_secs_f64();
    println!(
        "build:  {:>8.3} s   ({:.1} M keys/s)",
        build_s,
        N_KEYS as f64 / build_s / 1e6
    );

    // Warm-up
    for _ in 0..3 {
        let mut acc: u64 = 0;
        for &key in numeric_keys.iter().take(10000) {
            if let Ok(idx) = pgm.index(key) {
                acc ^= idx as u64;
            }
        }
        std::hint::black_box(acc);
    }

    // Point lookups
    let t2 = Instant::now();
    let mut acc: u64 = 0;
    let mut found = 0;

    for &key in &numeric_keys {
        if let Ok(idx) = pgm.index(key) {
            acc ^= idx as u64;
            found += 1;
        }
    }

    let lookup_s = t2.elapsed().as_secs_f64();
    println!(
        "lookup: {:>8.3} s   ({:.1} M lookups/s)   (found={}, acc={})",
        lookup_s,
        found as f64 / lookup_s / 1e6,
        found,
        acc
    );

    // Range queries benchmark
    let t3 = Instant::now();
    let mut total_results = 0;

    // 1000 range queries
    for i in 0..1000 {
        let base = numeric_keys[i * (numeric_keys.len() / 1000)];
        let results = pgm.range(base, base + 1000);
        total_results += results.len();
    }

    let range_s = t3.elapsed().as_secs_f64();
    println!(
        "range:  {:>8.3} s   ({:.1} queries/s)   (total_results={})",
        range_s,
        1000.0 / range_s,
        total_results
    );

    println!();
    Ok(())
}

#[cfg(feature = "pgm")]
fn run_comparison_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚔️  COMPARISON BENCHMARK");
    println!("{}", "=".repeat(50));

    let n_test = N_KEYS / 2; // Smaller set for faster comparison

    // Test data
    let string_keys = gen_string_keys_optimized(n_test, GEN_SEED);
    let numeric_keys = gen_numeric_keys(n_test, GEN_SEED);
    let mixed_keys = gen_mixed_keys(n_test, GEN_SEED);

    println!("Testing with {} keys each...\n", n_test);

    // ===== MPH vs PGM vs Hybrid =====
    println!("📊 Build Performance:");
    println!(
        "{:<20} {:>10} {:>15}",
        "Method", "Time (ms)", "Rate (M keys/s)"
    );
    println!("{}", "-".repeat(50));

    // MPH build
    let t_start = Instant::now();
    let mph = Builder::new().build(string_keys.iter().map(|k| k.as_slice()))?;
    let mph_build_ms = t_start.elapsed().as_millis();
    println!(
        "{:<20} {:>10} {:>15.1}",
        "MPH (strings)",
        mph_build_ms,
        n_test as f64 / t_start.elapsed().as_secs_f64() / 1e6
    );

    // PGM build
    let t_start = Instant::now();
    let pgm = PgmBuilder::new().build(numeric_keys.clone())?;
    let pgm_build_ms = t_start.elapsed().as_millis();
    println!(
        "{:<20} {:>10} {:>15.1}",
        "PGM (numbers)",
        pgm_build_ms,
        n_test as f64 / t_start.elapsed().as_secs_f64() / 1e6
    );

    // Hybrid build
    let t_start = Instant::now();
    let hybrid = HybridBuilder::new().build(mixed_keys.clone())?;
    let hybrid_build_ms = t_start.elapsed().as_millis();
    println!(
        "{:<20} {:>10} {:>15.1}",
        "Hybrid (mixed)",
        hybrid_build_ms,
        n_test as f64 / t_start.elapsed().as_secs_f64() / 1e6
    );

    println!();

    // ===== Lookup Performance =====
    println!("🔍 Lookup Performance (1M operations):");
    println!(
        "{:<20} {:>10} {:>15} {:>10}",
        "Method", "Time (ms)", "Rate (M ops/s)", "Hit Rate"
    );
    println!("{}", "-".repeat(60));

    // MPH lookups
    let t_start = Instant::now();
    let mut acc = 0u64;
    for _ in 0..1000 {
        for key in string_keys.iter().take(1000) {
            acc ^= mph.index(key);
        }
    }
    let mph_lookup_ms = t_start.elapsed().as_millis();
    std::hint::black_box(acc);
    println!(
        "{:<20} {:>10} {:>15.1} {:>10}",
        "MPH",
        mph_lookup_ms,
        1_000_000.0 / t_start.elapsed().as_secs_f64() / 1e6,
        "100%"
    );

    // PGM lookups
    let t_start = Instant::now();
    let mut acc = 0u64;
    let mut hits = 0;
    for _ in 0..1000 {
        for &key in numeric_keys.iter().take(1000) {
            if let Ok(idx) = pgm.index(key) {
                acc ^= idx as u64;
                hits += 1;
            }
        }
    }
    let pgm_lookup_ms = t_start.elapsed().as_millis();
    std::hint::black_box(acc);
    println!(
        "{:<20} {:>10} {:>15.1} {:>9.1}%",
        "PGM",
        pgm_lookup_ms,
        hits as f64 / t_start.elapsed().as_secs_f64() / 1e6,
        hits as f64 / 1_000_000.0 * 100.0
    );

    // Hybrid lookups
    let t_start = Instant::now();
    let mut acc = 0u64;
    let mut hits = 0;
    for _ in 0..1000 {
        for key in mixed_keys.iter().take(1000) {
            if let Ok(idx) = hybrid.index(key) {
                acc ^= idx as u64;
                hits += 1;
            }
        }
    }
    let hybrid_lookup_ms = t_start.elapsed().as_millis();
    std::hint::black_box(acc);
    println!(
        "{:<20} {:>10} {:>15.1} {:>9.1}%",
        "Hybrid",
        hybrid_lookup_ms,
        hits as f64 / t_start.elapsed().as_secs_f64() / 1e6,
        hits as f64 / 1_000.0,
    );

    println!("\n🏆 Performance Winner Summary:");
    println!(
        "  Build Speed: {} (PGM usually wins for numbers)",
        if pgm_build_ms < mph_build_ms.min(hybrid_build_ms) {
            "PGM 🥇"
        } else if mph_build_ms < hybrid_build_ms {
            "MPH 🥈"
        } else {
            "Hybrid 🥉"
        }
    );

    println!(
        "  Lookup Speed: {} (MPH usually wins)",
        if mph_lookup_ms < pgm_lookup_ms.min(hybrid_lookup_ms) {
            "MPH 🥇"
        } else if pgm_lookup_ms < hybrid_lookup_ms {
            "PGM 🥈"
        } else {
            "Hybrid 🥉"
        }
    );

    Ok(())
}

fn gen_string_keys_optimized(n: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(n);
    let mut seen = HashSet::with_capacity(n * 2);

    const KEY_SIZE: usize = 20;
    const BATCH_SIZE: usize = 10000;
    let mut batch_buffer = vec![0u8; BATCH_SIZE * KEY_SIZE];

    while keys.len() < n {
        let remaining = n - keys.len();
        let this_batch = remaining.min(BATCH_SIZE);

        rng.fill_bytes(&mut batch_buffer[..this_batch * KEY_SIZE]);

        for i in 0..this_batch {
            let start = i * KEY_SIZE;
            let end = start + KEY_SIZE;
            let mut key = batch_buffer[start..end].to_vec();

            let counter = keys.len() as u64;
            key[0..8].copy_from_slice(&counter.to_le_bytes());

            if seen.insert(key.clone()) {
                keys.push(key);
            }
        }
    }

    keys.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    keys
}

#[cfg(feature = "pgm")]
fn gen_numeric_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(n);

    for i in 0..n {
        let key = match i % 4 {
            0 => i as u64 * 2,                  // Linear pattern
            1 => (i as u64).pow(2) % 1_000_000, // Quadratic pattern
            2 => rng.next_u64() % 1_000_000,    // Random pattern
            _ => 1_000_000 + i as u64,          // Dense pattern
        };
        keys.push(key);
    }

    keys.sort_unstable();
    keys.dedup();
    keys.truncate(n);
    keys
}

#[cfg(feature = "pgm")]
fn gen_mixed_keys(n: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(n);

    let numeric_count = n / 2;
    for i in 0..numeric_count {
        let num = (i as u64 * 1000 + rng.next_u64() % 1000) as u64;
        keys.push(num.to_le_bytes().to_vec());
    }

    let string_count = n - numeric_count;
    for i in 0..string_count {
        let mut key = vec![0u8; 16];
        rng.fill_bytes(&mut key);
        key[0..8].copy_from_slice(&(i as u64 + 1_000_000).to_le_bytes());
        keys.push(key);
    }

    keys
}
