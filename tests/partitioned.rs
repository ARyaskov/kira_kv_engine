//! Partitioned (multi-part) PtrHash25 build: bijectivity, serialization, duplicate
//! detection, and `Index` behaviour above `PART_TARGET_KEYS`.

use kira_kv_engine::__internal::{
    BuildConfig, Builder, PART_TARGET_KEYS, PtrHash25Error, read_ptrhash25, write_ptrhash25,
};
use kira_kv_engine::index::{Index, IndexBuilder, IndexConfig};

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn keys_u64(n: usize, seed: u64) -> Vec<u64> {
    (0..n as u64).map(|i| splitmix(i ^ seed)).collect()
}

/// Keys that cannot collide with `keys_u64(_, _)` / `byte_keys(_, _)` outputs: splitmix
/// is a bijection and the inputs live in a disjoint range (`i + 2^40`).
const FOREIGN_BASE: u64 = 1 << 40;

fn foreign_u64(n: usize, seed: u64) -> Vec<u64> {
    (0..n as u64).map(|i| splitmix((i + FOREIGN_BASE) ^ seed)).collect()
}

fn byte_keys(n: usize, seed: u64) -> Vec<Vec<u8>> {
    byte_keys_from(0, n, seed)
}

fn foreign_bytes(n: usize, seed: u64) -> Vec<Vec<u8>> {
    byte_keys_from(FOREIGN_BASE, n, seed)
}

fn byte_keys_from(start: u64, n: usize, seed: u64) -> Vec<Vec<u8>> {
    (start..start + n as u64)
        .map(|i| {
            let h = splitmix(i ^ seed);
            match i % 3 {
                0 => h.to_le_bytes().to_vec(),
                1 => format!("user:{h:x}").into_bytes(),
                _ => {
                    let mut v = format!("prefix-shared-{}", h & 0xFF).into_bytes();
                    v.extend_from_slice(&h.to_be_bytes());
                    v.extend_from_slice(&splitmix(h).to_le_bytes());
                    v
                }
            }
        })
        .collect()
}

fn assert_bijective(mph: &kira_kv_engine::__internal::PtrHash25Mphf, keys: &[u64]) {
    let cap = mph.slot_capacity();
    let mut seen = vec![false; cap];
    for &k in keys {
        let slot = mph.index_u64(k) as usize;
        assert!(slot < cap, "slot {slot} out of range {cap}");
        assert!(!seen[slot], "collision at slot {slot}");
        seen[slot] = true;
    }
}

#[test]
fn multi_part_is_bijective() {
    let n = PART_TARGET_KEYS * 5 + 123;
    let keys = keys_u64(n, 0xA1);
    let mph = Builder::new().build(&keys).expect("build failed");
    assert!(mph.num_parts() > 1, "expected a multi-part build for {n} keys");
    assert_eq!(mph.slot_capacity(), n, "the MPH must be minimal");
    assert_bijective(&mph, &keys);
    for &k in &keys {
        assert!(mph.lookup_u64(k).is_some());
    }
}

#[test]
fn single_part_below_threshold() {
    let n = PART_TARGET_KEYS;
    let keys = keys_u64(n, 0xB2);
    let mph = Builder::new().build(&keys).expect("build failed");
    assert_eq!(mph.num_parts(), 1);
    assert_bijective(&mph, &keys);
}

#[test]
fn multi_part_serialize_round_trip() {
    let n = PART_TARGET_KEYS * 3 + 7;
    let keys = keys_u64(n, 0xC3);
    let mph = Builder::new().build(&keys).expect("build failed");
    assert!(mph.num_parts() > 1);
    let mut bytes = Vec::new();
    write_ptrhash25(&mph, &mut bytes);
    let mut pos = 0;
    let restored = read_ptrhash25(&bytes, &mut pos).expect("read failed");
    assert_eq!(pos, bytes.len());
    assert_eq!(restored.num_parts(), mph.num_parts());
    assert_eq!(restored.slot_capacity(), mph.slot_capacity());
    for &k in &keys {
        assert_eq!(mph.index_u64(k), restored.index_u64(k));
        assert_eq!(mph.lookup_u64(k), restored.lookup_u64(k));
    }
}

#[test]
fn build_with_slots_matches_lookup() {
    let n = PART_TARGET_KEYS * 2 + 11;
    let keys = keys_u64(n, 0xD4);
    let (mph, part, slots) = Builder::new().build_with_slots(&keys).expect("build failed");
    assert_eq!(part.len(), n);
    assert_eq!(slots.len(), n);
    for p in 0..part.num_parts() {
        let off = mph.parts[p].slot_off as usize;
        for i in part.part_offsets[p]..part.part_offsets[p + 1] {
            let key = part.original_key(i);
            assert_eq!(mph.index_u64(key) as usize, off + slots[i] as usize);
        }
    }
}

#[test]
fn build_partitioned_with_fp16_matches_keys() {
    use kira_kv_engine::__internal::{BuildOutputs, build_partitioned_with, partition_keys};
    let n = PART_TARGET_KEYS * 3 + 5;
    let keys = keys_u64(n, 0x8E);
    let cfg = BuildConfig::default();
    let part = partition_keys(&keys, &cfg);
    let outputs = BuildOutputs { slots: true, fp16: true };
    let (mph, slots, fp16) = build_partitioned_with(&part, &cfg, outputs).expect("build");
    let slots = slots.expect("slots");
    let fp16 = fp16.expect("fp16");
    assert_eq!(fp16.len(), mph.slot_capacity());
    for p in 0..part.num_parts() {
        let off = mph.parts[p].slot_off as usize;
        for i in part.part_offsets[p]..part.part_offsets[p + 1] {
            let key = part.original_key(i);
            let slot = off + slots[i] as usize;
            assert_eq!(mph.index_u64(key) as usize, slot);
            assert_eq!(fp16[slot], (key & 0xFFFF) as u16);
        }
    }
    // No side tables requested → none returned.
    let (_, s, f) = build_partitioned_with(&part, &cfg, BuildOutputs::default()).expect("build");
    assert!(s.is_none() && f.is_none());
}

#[test]
fn duplicate_key_detected_multi_part() {
    let n = PART_TARGET_KEYS * 2;
    let mut keys = keys_u64(n, 0xE5);
    keys[n - 1] = keys[17];
    let err = Builder::new().build(&keys).expect_err("duplicate must fail");
    assert!(matches!(err, PtrHash25Error::DuplicateKey), "got {err:?}");
}

#[test]
fn duplicate_key_detected_single_part() {
    let mut keys = keys_u64(1000, 0xF6);
    keys[999] = keys[3];
    let err = Builder::new().build(&keys).expect_err("duplicate must fail");
    assert!(matches!(err, PtrHash25Error::DuplicateKey), "got {err:?}");
}

#[test]
fn multi_part_with_inner_fingerprints_rejects_foreign() {
    let n = PART_TARGET_KEYS * 2;
    let keys = keys_u64(n, 0x17);
    let cfg = BuildConfig {
        with_fingerprints: true,
        ..BuildConfig::default()
    };
    let mph = Builder::new().with_config(cfg).build(&keys).expect("build failed");
    assert!(mph.num_parts() > 1);
    assert_eq!(mph.fingerprints.len(), mph.slot_capacity());
    for &k in &keys {
        assert!(mph.lookup_u64(k).is_some());
    }
    let mut false_hits = 0usize;
    let probes = 20_000u64;
    for i in 0..probes {
        if mph.lookup_u64(splitmix(i ^ 0xDEAD_BEEF)).is_some() {
            false_hits += 1;
        }
    }
    let rate = false_hits as f64 / probes as f64;
    assert!(rate < 0.02, "false hit rate too high: {rate}");
}

#[test]
fn index_ref_and_owned_agree_multi_part() {
    let n = 200_000;
    let keys = byte_keys(n, 0x28);
    let by_ref = IndexBuilder::new().build_index_ref(&keys).expect("ref build");
    let owned = IndexBuilder::new().build_index(keys.clone()).expect("owned build");
    assert_eq!(by_ref.len(), n);
    assert_eq!(by_ref.slot_capacity(), owned.slot_capacity());
    let mut seen = vec![false; by_ref.slot_capacity()];
    for k in &keys {
        let a = by_ref.lookup(k).expect("hit");
        let b = owned.lookup(k).expect("hit");
        assert_eq!(a, b);
        assert!(!seen[a], "collision");
        seen[a] = true;
    }
    let foreign = foreign_bytes(20_000, 0x99_99);
    let false_hits = foreign.iter().filter(|k| by_ref.lookup(k).is_ok()).count();
    assert!(false_hits < 40, "too many false hits: {false_hits}");
}

#[test]
fn index_duplicate_bytes_rejected() {
    let mut keys = byte_keys(100_000, 0x39);
    keys[99_999] = keys[42].clone();
    let err = IndexBuilder::new().build_index_ref(&keys).err().expect("must fail");
    assert!(matches!(err, kira_kv_engine::IndexError::DuplicateKey), "got {err}");
}

#[test]
fn index_serialize_round_trip_multi_part() {
    let keys = byte_keys(150_000, 0x4A);
    let idx = IndexBuilder::new().build_index_ref(&keys).expect("build");
    let bytes = idx.to_bytes().expect("to_bytes");
    let back = Index::from_bytes(&bytes).expect("from_bytes");
    assert_eq!(back.len(), idx.len());
    for k in &keys {
        assert_eq!(idx.lookup(k).ok(), back.lookup(k).ok());
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.idx");
    idx.save_mmap(&path).unwrap();
    let mm = Index::open_mmap(&path).unwrap();
    for k in keys.iter().take(5000) {
        assert_eq!(idx.lookup(k).ok(), mm.lookup(k).ok());
    }
}

#[test]
fn index_lean_multi_part_is_bijective() {
    let keys = byte_keys(150_000, 0x5B);
    let mut cfg = IndexConfig::default();
    cfg.lean_mph = true;
    let idx = IndexBuilder::new().with_config(cfg).build_index_ref(&keys).expect("build");
    let cap = idx.slot_capacity();
    let mut seen = vec![false; cap];
    for k in &keys {
        let s = idx.lookup(k).expect("hit");
        assert!(s < cap);
        assert!(!seen[s]);
        seen[s] = true;
    }
    let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
    let batch = idx.lookup_batch_pipelined(&refs);
    for (k, b) in keys.iter().zip(batch) {
        assert_eq!(b, idx.lookup(k).ok());
    }
}

#[test]
fn index_u64_keys_multi_part_simd_batch() {
    let n = 120_000;
    let raw = keys_u64(n, 0x6C);
    let keys: Vec<Vec<u8>> = raw.iter().map(|k| k.to_le_bytes().to_vec()).collect();
    let idx = IndexBuilder::new().build_index_ref(&keys).expect("build");
    let batch = idx.lookup_batch_u64_simd(&raw);
    for (k, b) in raw.iter().zip(batch) {
        assert_eq!(b, Some(idx.lookup_u64(*k).expect("hit")));
    }
    let foreign = foreign_u64(10_000, 0x7777);
    let misses = idx.lookup_batch_u64_simd(&foreign).iter().filter(|r| r.is_none()).count();
    assert!(misses > 9_900, "expected nearly all misses, got {misses}");
}

#[test]
fn gpu_export_carries_parts() {
    let keys = byte_keys(100_000, 0x7D);
    let idx = IndexBuilder::new().build_index_ref(&keys).expect("build");
    let export = idx.gpu_export().expect("export");
    assert!(export.parts.len() > 1);
    assert_eq!(export.pilots.len() as u32, export.num_buckets);
    let total_keys: u64 = export.parts.iter().map(|p| p.num_keys as u64).sum();
    assert_eq!(total_keys, export.num_slots);
    assert_eq!(export.num_slots as usize, idx.len());
    let total_remap: usize = export.parts.iter().map(|p| (p.num_slots - p.num_keys) as usize).sum();
    assert_eq!(total_remap, export.remap.len());
    assert!(export.remap.iter().all(|&r| (r as u64) < export.num_slots));
}

/// `contains()` is a filter-level answer. Without a filter (lean mode) it cannot
/// reject anything, and the API must say so instead of pretending to be exact.
#[test]
fn contains_semantics_lean_vs_full() {
    let keys = byte_keys(20_000, 0x8E);
    let foreign = byte_keys(20_000, 0x8F);
    let mut lean_cfg = IndexConfig::default();
    lean_cfg.lean_mph = true;
    let lean = IndexBuilder::new().with_config(lean_cfg).build_index_ref(&keys).expect("build");
    assert!(!lean.supports_negative_lookups());
    assert!(keys.iter().all(|k| lean.contains(k)));
    // Documented behaviour: no membership information at all in lean mode.
    assert!(foreign.iter().all(|k| lean.contains(k)));

    let full = IndexBuilder::new().build_index_ref(&keys).expect("build");
    assert!(full.supports_negative_lookups());
    assert!(keys.iter().all(|k| full.contains(k)));
    let fp = foreign.iter().filter(|k| full.contains(k)).count();
    assert!(fp < 400, "Bloom false positives too high: {fp}/20000");
    let refs: Vec<&[u8]> = foreign.iter().map(|k| k.as_slice()).collect();
    assert_eq!(full.contains_batch(&refs).iter().filter(|&&b| b).count(), fp);
    assert!(Index::empty().supports_negative_lookups());
}

#[test]
fn duplicate_keys_are_a_typed_error() {
    let mut keys = byte_keys(5_000, 0x9F);
    keys[4_999] = keys[42].clone();
    let err = IndexBuilder::new().build_index_ref(&keys).err().expect("duplicate must fail");
    assert!(matches!(err, kira_kv_engine::IndexError::DuplicateKey), "{err:?}");
    let io = Index::load("/definitely/not/a/path/kira.idx").err().expect("missing file");
    assert!(matches!(io, kira_kv_engine::IndexError::Io(_)), "{io:?}");
}
