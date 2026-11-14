//! Diagnostic phase timing for instrumented copies, not a production API.
use kira_kv_engine::{GpuExport, Index, IndexBuilder};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
// Keep the isolated dependency graph equal when the engine revision uses
// tempfile only for tests versus external construction in normal code.
use tempfile as _;

#[path = "million_data.rs"]
mod million_data;

struct CountAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn allocated(bytes: usize) {
    ALLOCATIONS.fetch_add(1, Relaxed);
    let current = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(current, Relaxed);
}

// SAFETY: requests are forwarded unchanged to System. Atomic counters do not
// allocate or modify storage ownership, alignment or initialization.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, bytes: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, bytes) };
        if !replacement.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(bytes);
        }
        replacement
    }
}

#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

fn measured<T>(queries: usize, operation: impl FnOnce() -> T) -> (f64, usize, usize, T) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let allocations = ALLOCATIONS.load(Relaxed);
    let started = Instant::now();
    let result = operation();
    let ns = started.elapsed().as_nanos() as f64 / queries as f64;
    let allocations = ALLOCATIONS.load(Relaxed) - allocations;
    let peak = PEAK.load(Relaxed).saturating_sub(base);
    (ns, allocations, peak, result)
}

// Independent scalar arithmetic for the documented off-host MPH formula.
fn mix(mut value: u64) -> u64 {
    value ^= value >> 32;
    value = value.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    value ^= value >> 32;
    value = value.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    value ^ (value >> 32)
}

fn reduce(value: u64, count: usize) -> usize {
    ((value as u128 * count as u128) >> 64) as usize
}

fn canonical(key: &[u8], seed: u64) -> u64 {
    if key.len() == 8 {
        return mix(u64::from_le_bytes(key.try_into().unwrap()) ^ seed);
    }
    // The frozen million_build generator uses only eight-byte numbers and
    // strings of 9..64 bytes; the legacy scalar AES function is the reference
    // for that length domain, independent of the new canonical batch helper.
    assert!((9..=64).contains(&key.len()));
    kira_kv_engine::__internal::aes_hash::hash_bytes(key, seed)
}

fn oracle(export: &GpuExport, hash: u64) -> Option<usize> {
    if let Some(bloom) = &export.bloom {
        const SALTS: [u32; 8] = [
            0x47B6_137B,
            0x4476_8924,
            0x1820_5237,
            0x2384_8965,
            0x8E6E_2354,
            0x0F7C_C9B6,
            0xE43D_5FA5,
            0xA4D5_2DC1,
        ];
        let block = (((hash >> 32) as u128 * bloom.blocks as u128) >> 32) as usize;
        for (lane, salt) in SALTS.into_iter().enumerate() {
            let bit = ((hash as u32).wrapping_mul(salt) >> bloom.bit_shift) & 63;
            if bloom.words[block * 8 + lane] & (1u64 << bit) == 0 {
                return None;
            }
        }
    }
    let rotated = hash.rotate_left(export.prerotate as u32);
    let part_id = if export.parts.len() == 1 {
        0
    } else {
        reduce((rotated ^ export.part_salt).wrapping_mul(0x9E37_79B9_7F4A_7C15), export.parts.len())
    };
    let part = &export.parts[part_id];
    let h1 = if export.parts.len() == 1 {
        mix(rotated ^ part.salt)
    } else {
        let base = mix(rotated ^ export.mph_salt);
        (base ^ part.salt).wrapping_mul(0xBF58_476D_1CE4_E5B9)
    };
    let low = (h1 & 0x0000_FFFF_FFFF_FFFF) << 16;
    let large = part.large_buckets as usize;
    let local_bucket = if h1 >> 48 < 39_321 {
        reduce(low, large.max(1))
    } else {
        large + reduce(low, (part.num_buckets as usize - large).max(1))
    };
    let pilot = export.pilots[part.bucket_off as usize + local_bucket];
    let h2 = h1.rotate_left(23) ^ 0xA24B_1F6F_DA39_2B31;
    let pilot_mix = (pilot as u64).wrapping_mul(0xA24B_1F6F_DA39_2B31);
    let value = (h2 ^ pilot_mix).rotate_left(31).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    let mut local = reduce(value, part.num_slots as usize);
    if local >= part.num_keys as usize {
        local = export.remap[part.remap_off as usize + local - part.num_keys as usize] as usize;
    }
    let slot = part.slot_off as usize + local;
    if export.fingerprint_bits != 0 {
        let width = export.fingerprint_bits as usize / 8;
        let bytes = export.fingerprint_bytes.as_ref().unwrap();
        let mut stored = 0u32;
        for (offset, &byte) in bytes[slot * width..slot * width + width].iter().enumerate() {
            stored |= (byte as u32) << (offset * 8);
        }
        let mask = u32::MAX >> (32 - export.fingerprint_bits);
        if stored != hash as u32 & mask {
            return None;
        }
    }
    Some(slot)
}

fn digest(values: impl IntoIterator<Item = u8>) -> u64 {
    values.into_iter().fold(0xCBF2_9CE4_8422_2325, |value, byte| {
        (value ^ byte as u64).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

struct Case<'a> {
    label: &'a str,
    mode: &'a str,
    workload: &'a str,
    keys: usize,
    queries: usize,
    build_ms: f64,
    index_bytes: usize,
    known_hits: usize,
    known_misses: usize,
    false_accepts: usize,
    data_digest: u64,
    hash_digest: u64,
}

impl Case<'_> {
    fn print(&self, phase: &str, ns: f64, allocations: usize, peak: usize) {
        println!(
            "{},{},{},{phase},{},{},{:.3},{},{ns:.3},{allocations},{peak},{},{},{},{:016x},{:016x},pass",
            self.label,
            self.mode,
            self.workload,
            self.keys,
            self.queries,
            self.build_ms,
            self.index_bytes,
            self.known_hits,
            self.known_misses,
            self.false_accepts,
            self.data_digest,
            self.hash_digest
        );
    }
}

fn check_universe(index: &Index, export: &GpuExport, keys: &[Vec<u8>]) {
    assert_eq!(export.num_slots as usize, keys.len());
    let mut seen = vec![false; keys.len()];
    for key in keys {
        let slot = oracle(export, canonical(key, export.prehash_seed)).expect("known key rejected");
        assert!(slot < seen.len() && !seen[slot], "export must be a bijection over known keys");
        seen[slot] = true;
        assert_eq!(index.lookup(key).unwrap(), slot);
    }
    assert!(seen.into_iter().all(|value| value));
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let label = &args[1];
    let n: usize = args[2].parse().unwrap();
    let queries: usize = args[3].parse().unwrap();
    assert!((1000..=1_000_000).contains(&n));
    assert!((1000..=n).contains(&queries));
    let core: usize =
        std::env::var("KIRA_BENCH_CORE").unwrap_or_else(|_| "2".into()).parse().unwrap();
    assert!(core_affinity::set_for_current(core_affinity::CoreId { id: core }));
    let keys = million_data::gen_mixed_keys(n, 42 ^ 0x6666_6666_6666_6666);
    assert_eq!(keys.len(), n);
    let existing: HashSet<Vec<u8>> = keys.iter().cloned().collect();
    assert_eq!(existing.len(), n);
    let mut generator = StdRng::seed_from_u64(1337 ^ 0x7777_7777_7777_7777);
    let missing =
        million_data::gen_mixed_keys_missing(queries.div_ceil(100), &mut generator, &existing);
    assert!(missing.iter().all(|key| !existing.contains(key)));
    let positive = million_data::make_positive_queries_bytes(&keys[..queries]);
    let negative =
        million_data::make_negative_queries_bytes(&keys, &missing, queries, 0.70, &mut generator);
    println!(
        "label,mode,workload,phase,keys,queries,build_ms,index_bytes,ns_per_query,phase_allocations,phase_peak_heap_bytes,known_hits,known_misses,false_accepts,data_digest,hash_digest,correctness"
    );
    for lean in [false, true] {
        let started = Instant::now();
        let index = IndexBuilder::new()
            .with_parallel_build(false)
            .with_lean_mph(lean)
            .build_index_ref(&keys)
            .unwrap();
        let build_ms = started.elapsed().as_secs_f64() * 1000.0;
        let export = index.gpu_export().expect("default mix64 geometry exports its scalar oracle");
        check_universe(&index, &export, &keys);
        for (workload, query) in [("positive", &positive), ("negative", &negative)] {
            let mut refs: Vec<&[u8]> = query.iter().map(|query| query.key.as_slice()).collect();
            let mut order = StdRng::seed_from_u64(1337 ^ 0xAAAA_AAAA_AAAA_AAAA);
            refs.shuffle(&mut order);
            let hashes: Vec<u64> =
                refs.iter().map(|key| canonical(key, export.prehash_seed)).collect();
            let expected: Vec<_> = hashes.iter().map(|&hash| oracle(&export, hash)).collect();
            let known: Vec<bool> = refs.iter().map(|key| existing.contains(*key)).collect();
            let known_hits = known.iter().filter(|&&value| value).count();
            if workload == "positive" {
                assert_eq!(known_hits, queries);
            } else {
                assert_eq!(known_hits, (queries as f64 * 0.70) as usize);
            }
            let false_accepts = expected
                .iter()
                .zip(&known)
                .filter(|(result, known)| !**known && result.is_some())
                .count();
            for ((key, expected), known) in refs.iter().zip(&expected).zip(&known) {
                assert_eq!(index.lookup(key).ok(), *expected);
                if *known {
                    assert!(expected.is_some());
                }
            }
            let case = Case {
                label,
                mode: if lean { "lean" } else { "default" },
                workload,
                keys: n,
                queries,
                build_ms,
                index_bytes: index.stats().total_memory,
                known_hits,
                known_misses: queries - known_hits,
                false_accepts,
                data_digest: digest(refs.iter().flat_map(|key| {
                    (key.len() as u64).to_le_bytes().into_iter().chain(key.iter().copied())
                })),
                hash_digest: digest(hashes.iter().flat_map(|hash| hash.to_le_bytes())),
            };
            let mut computed = vec![0u64; queries];
            let (ns, allocations, peak, ()) = measured(queries, || {
                index.__diag_hash_stage(black_box(&refs), black_box(&mut computed));
            });
            assert_eq!(computed, hashes, "production hash fragment differs from scalar oracle");
            assert_eq!(allocations, 0);
            case.print("hash", ns, allocations, peak);
            let mut out = vec![None; queries];
            let (ns, allocations, peak, ()) = measured(queries, || {
                index.__diag_probe_stage(black_box(&hashes), black_box(&mut out));
            });
            assert_eq!(out, expected, "production probe differs from exported-table oracle");
            assert_eq!(allocations, 0);
            case.print("probe", ns, allocations, peak);
            let (ns, allocations, peak, out) =
                measured(queries, || index.lookup_batch_pipelined(black_box(&refs)));
            assert_eq!(out, expected, "public byte pipeline differs from exported-table oracle");
            case.print("pipeline", ns, allocations, peak);
        }
    }
}
