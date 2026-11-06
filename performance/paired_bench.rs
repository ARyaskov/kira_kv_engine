//! Same-process A/B benchmark. Prepared and run by run_paired.ps1.
use std::hint::black_box;
use std::time::Instant;
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn time(n: usize, f: &mut impl FnMut()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_nanos() as f64 / n as f64
}
fn row(name: &str, old: f64, new: f64, unit: &str) {
    println!("{name},{old:.6},{new:.6},{unit},{:.3}", 100. * (1. - new / old));
}
fn pair(name: &str, n: usize, mut old: impl FnMut(), mut new: impl FnMut()) {
    old();
    new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for i in 0..9 {
        let (x, y) = if i % 2 == 0 {
            (time(n, &mut old), time(n, &mut new))
        } else {
            let y = time(n, &mut new);
            (time(n, &mut old), y)
        };
        a.push(x);
        b.push(y);
    }
    row(name, median(a), median(b), if name == "mmap.load" { "ns/op" } else { "ns/key" });
}
fn main() {
    let n =
        std::env::var("KIRA_STAGE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000_000usize);
    let core = std::env::var("KIRA_BENCH_CORE").ok().and_then(|s| s.parse().ok()).unwrap_or(2usize);
    assert!(core_affinity::set_for_current(core_affinity::CoreId { id: core }));
    let q = n.min(1_000_000);
    assert!(n > 0);
    let keys: Vec<[u8; 8]> = (0..n as u64).map(|i| mix(i).to_le_bytes()).collect();
    let ordinals: Vec<usize> = (0..q as u64).map(|i| (mix(i ^ 0x77) % n as u64) as usize).collect();
    let hits: Vec<u64> = ordinals.iter().map(|&i| u64::from_le_bytes(keys[i])).collect();
    let refs: Vec<&[u8]> = ordinals.iter().map(|&i| keys[i].as_slice()).collect();
    let misses: Vec<u64> = (n as u64..n as u64 + q as u64).map(mix).collect();
    println!("metric,before,after,unit,reduction_percent");
    row("keys", n as f64, n as f64, "keys");
    let old = before::IndexBuilder::new().build_index_ref(&keys).unwrap();
    let new = after::IndexBuilder::new().build_index_ref(&keys).unwrap();
    for key in &keys {
        assert_eq!(old.lookup(key).ok(), new.lookup(key).ok());
    }
    row(
        "default.memory",
        old.stats().total_memory as f64 / n as f64,
        new.stats().total_memory as f64 / n as f64,
        "B/key",
    );
    pair(
        "default.hit",
        q,
        || {
            black_box(
                hits.iter()
                    .fold(0usize, |s, &k| s.wrapping_add(old.lookup_u64(black_box(k)).unwrap())),
            );
        },
        || {
            black_box(
                hits.iter()
                    .fold(0usize, |s, &k| s.wrapping_add(new.lookup_u64(black_box(k)).unwrap())),
            );
        },
    );
    pair(
        "default.miss",
        q,
        || {
            black_box(misses.iter().filter(|&&k| old.lookup_u64(black_box(k)).is_ok()).count());
        },
        || {
            black_box(misses.iter().filter(|&&k| new.lookup_u64(black_box(k)).is_ok()).count());
        },
    );
    let mut oa = vec![None; q];
    let mut ob = vec![None; q];
    let mut ca = vec![0; q];
    let mut cb = vec![0; q];
    pair(
        "default.batch_u64",
        q,
        || {
            old.lookup_batch_u64_simd_into(black_box(&hits), &mut ca, &mut oa);
            black_box(&oa);
        },
        || {
            new.lookup_batch_u64_simd_into(black_box(&hits), &mut cb, &mut ob);
            black_box(&ob);
        },
    );
    assert_eq!(oa, ob);
    pair(
        "default.batch_bytes",
        q,
        || {
            black_box(old.lookup_batch_pipelined(black_box(&refs)));
        },
        || {
            black_box(new.lookup_batch_pipelined(black_box(&refs)));
        },
    );
    old.save_mmap("performance/paired-old.idx").unwrap();
    new.save_mmap("performance/paired-new.idx").unwrap();
    {
        let a = before::Index::open_mmap("performance/paired-old.idx").unwrap();
        let b = after::Index::open_mmap("performance/paired-new.idx").unwrap();
        for key in &keys {
            assert_eq!(a.lookup(key).ok(), b.lookup(key).ok());
        }
    }
    pair(
        "mmap.load",
        1,
        || {
            black_box(before::Index::open_mmap("performance/paired-old.idx").unwrap());
        },
        || {
            black_box(after::Index::open_mmap("performance/paired-new.idx").unwrap());
        },
    );
    std::fs::remove_file("performance/paired-old.idx").unwrap();
    std::fs::remove_file("performance/paired-new.idx").unwrap();
    for bits in [
        after::FingerprintBits::Bits16,
        after::FingerprintBits::Bits24,
        after::FingerprintBits::Bits32,
    ] {
        let candidate = after::IndexBuilder::new()
            .with_bloom_filter(false)
            .with_fingerprint_bits(bits)
            .build_index_ref(&keys)
            .unwrap();
        for key in &keys {
            assert!(candidate.lookup(key).is_ok());
        }
        row(
            &format!("fp{}_no_bloom.memory", bits.bits()),
            old.stats().total_memory as f64 / n as f64,
            candidate.stats().total_memory as f64 / n as f64,
            "B/key",
        );
        pair(
            &format!("fp{}_no_bloom.hit", bits.bits()),
            q,
            || {
                black_box(
                    hits.iter().fold(0usize, |s, &k| {
                        s.wrapping_add(old.lookup_u64(black_box(k)).unwrap())
                    }),
                );
            },
            || {
                black_box(hits.iter().fold(0usize, |s, &k| {
                    s.wrapping_add(candidate.lookup_u64(black_box(k)).unwrap())
                }));
            },
        );
    }
    if std::env::var_os("KIRA_PAIRED_STATIC_ONLY").is_some() {
        row("correctness", 1., 1., "pass");
        return;
    }
    drop((old, new, oa, ob, ca, cb));
    let mut old = before::DynamicIndex::new();
    let mut new = after::DynamicIndex::new();
    let mut a = 0u128;
    let mut b = 0u128;
    // Interleave write chunks so old/new grow under comparable clock/load conditions.
    for block in (0..n).step_by(65536) {
        let end = (block + 65536).min(n);
        if block / 65536 % 2 == 0 {
            let t = Instant::now();
            for (i, k) in keys[block..end].iter().enumerate() {
                assert_eq!(old.insert(k.to_vec()), (block + i) as u32);
            }
            a += t.elapsed().as_nanos();
            let t = Instant::now();
            for (i, k) in keys[block..end].iter().enumerate() {
                assert_eq!(new.insert(k.to_vec()), (block + i) as u32);
            }
            b += t.elapsed().as_nanos();
        } else {
            let t = Instant::now();
            for (i, k) in keys[block..end].iter().enumerate() {
                assert_eq!(new.insert(k.to_vec()), (block + i) as u32);
            }
            b += t.elapsed().as_nanos();
            let t = Instant::now();
            for (i, k) in keys[block..end].iter().enumerate() {
                assert_eq!(old.insert(k.to_vec()), (block + i) as u32);
            }
            a += t.elapsed().as_nanos();
        }
    }
    let t = Instant::now();
    old.flush().unwrap();
    a += t.elapsed().as_nanos();
    let t = Instant::now();
    new.flush().unwrap();
    new.finish_background_compaction().unwrap();
    b += t.elapsed().as_nanos();
    row("dynamic.insert", a as f64 / n as f64, b as f64 / n as f64, "ns/key");
    row(
        "dynamic.memory",
        old.memory_usage() as f64 / n as f64,
        new.memory_usage() as f64 / n as f64,
        "B/key",
    );
    pair(
        "dynamic.hit",
        q,
        || {
            black_box(refs.iter().fold(0u64, |s, k| s + old.lookup(black_box(k)).unwrap() as u64));
        },
        || {
            black_box(refs.iter().fold(0u64, |s, k| s + new.lookup(black_box(k)).unwrap() as u64));
        },
    );
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(old.lookup(k), Some(i as u32));
        assert_eq!(new.lookup(k), Some(i as u32));
    }
    drop(new);
    let mut background = after::DynamicIndex::new();
    background.set_background_compaction(true).unwrap();
    let t = Instant::now();
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(background.insert(k.to_vec()), i as u32);
    }
    background.flush().unwrap();
    background.finish_background_compaction().unwrap();
    row(
        "background.insert",
        a as f64 / n as f64,
        t.elapsed().as_nanos() as f64 / n as f64,
        "ns/key",
    );
    pair(
        "background.hit_tiered",
        q,
        || {
            black_box(refs.iter().fold(0u64, |s, k| s + old.lookup(black_box(k)).unwrap() as u64));
        },
        || {
            black_box(
                refs.iter().fold(0u64, |s, k| s + background.lookup(black_box(k)).unwrap() as u64),
            );
        },
    );
    background.compact().unwrap();
    pair(
        "background.hit_compacted",
        q,
        || {
            black_box(refs.iter().fold(0u64, |s, k| s + old.lookup(black_box(k)).unwrap() as u64));
        },
        || {
            black_box(
                refs.iter().fold(0u64, |s, k| s + background.lookup(black_box(k)).unwrap() as u64),
            );
        },
    );
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(background.lookup(k), Some(i as u32));
    }
    row("correctness", 1., 1., "pass");
}
