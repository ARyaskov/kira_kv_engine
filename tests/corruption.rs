//! Corrupt and truncated serialized data must be rejected (or, for legacy formats
//! without a checksum, at least never crash): every deserializer bounds its
//! allocations by the bytes present and validates the invariants the unchecked
//! lookup paths rely on.

use kira_kv_engine::__internal::{
    BlockBloom, Builder, HugepageBuf, MmapIndex, read_ptrhash25, write_ptrhash25,
};
use kira_kv_engine::{EliasFano, Index, IndexBuilder, IndexConfig, PgmBuilder, PgmIndex};

fn splitmix(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn byte_keys(n: usize) -> Vec<Vec<u8>> {
    let mut s = 0x1234u64;
    (0..n)
        .map(|i| {
            let h = splitmix(&mut s);
            if i % 2 == 0 { h.to_le_bytes().to_vec() } else { format!("k:{h:x}").into_bytes() }
        })
        .collect()
}

/// Apply `rounds` single-byte mutations (each on a fresh copy) and hand the result
/// to `check`, which must not panic.
fn mutate_each(bytes: &[u8], rounds: usize, seed: u64, mut check: impl FnMut(&[u8])) {
    let mut s = seed;
    for _ in 0..rounds {
        let mut b = bytes.to_vec();
        let pos = (splitmix(&mut s) % b.len() as u64) as usize;
        let how = splitmix(&mut s) % 4;
        let orig = b[pos];
        b[pos] = match how {
            0 => orig ^ (1 << (splitmix(&mut s) % 8)),
            1 => 0xFF,
            2 => 0x00,
            _ => orig.wrapping_add(1),
        };
        if b[pos] == orig {
            b[pos] = orig ^ 0x55;
        }
        check(&b);
    }
}

fn truncations(bytes: &[u8], mut check: impl FnMut(&[u8])) {
    let step = (bytes.len() / 200).max(1);
    let mut len = 0;
    while len < bytes.len() {
        check(&bytes[..len]);
        len += step;
    }
    // Also the last few bytes one at a time (checksum / trailer region).
    for cut in 1..=16.min(bytes.len()) {
        check(&bytes[..bytes.len() - cut]);
    }
}

#[test]
fn index_v3_rejects_every_mutation_and_truncation() {
    let keys = byte_keys(20_000);
    for lean in [false, true] {
        let mut cfg = IndexConfig::default();
        cfg.lean_mph = lean;
        let idx = IndexBuilder::new().with_config(cfg).build_index_ref(&keys).unwrap();
        let bytes = idx.to_bytes().unwrap();
        let back = Index::from_bytes(&bytes).unwrap();
        assert_eq!(back.len(), idx.len());
        for k in &keys {
            assert_eq!(back.lookup(k).ok(), idx.lookup(k).ok());
        }
        mutate_each(&bytes, 2_000, 0xA5, |b| {
            assert!(Index::from_bytes(b).is_err(), "mutation accepted (lean={lean})");
        });
        truncations(&bytes, |b| {
            assert!(Index::from_bytes(b).is_err(), "truncation to {} accepted", b.len());
        });
    }
    let empty = Index::empty().to_bytes().unwrap();
    assert!(Index::from_bytes(&empty).unwrap().is_empty());
    mutate_each(&empty, 200, 0x11, |b| assert!(Index::from_bytes(b).is_err()));
}

/// Legacy (tag 2) payloads carry no checksum. Rebuild one from the v3 body and
/// make sure corrupt variants either fail to load or load into something whose
/// lookups stay in bounds and never panic.
#[test]
fn index_legacy_tag2_never_panics_on_corruption() {
    let keys = byte_keys(10_000);
    let foreign = byte_keys(10_000)
        .into_iter()
        .map(|mut k| {
            k.push(b'!');
            k
        })
        .collect::<Vec<_>>();
    let idx = IndexBuilder::new().build_index_ref(&keys).unwrap();
    let v3 = idx.to_bytes().unwrap();
    // v3: [tag][magic 4][version 2][hash 1][reserved 1][key_count 8][payload...][checksum 8]
    // tag 2: [2][key_count 8][payload...]
    let mut legacy = vec![2u8];
    legacy.extend_from_slice(&v3[9..v3.len() - 8]);
    let back = Index::from_bytes(&legacy).expect("legacy layout must load");
    for k in &keys {
        assert_eq!(back.lookup(k).ok(), idx.lookup(k).ok());
    }
    let exercise = |b: &[u8]| {
        if let Ok(ix) = Index::from_bytes(b) {
            let cap = ix.slot_capacity();
            for k in keys.iter().chain(foreign.iter()) {
                if let Ok(s) = ix.lookup(k) {
                    assert!(s < cap);
                }
                let _ = ix.contains(k);
            }
            let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
            let _ = ix.lookup_batch_pipelined(&refs);
            let u: Vec<u64> = (0..1000u64).collect();
            let _ = ix.lookup_batch_u64_simd(&u);
        }
    };
    mutate_each(&legacy, 1_500, 0xB6, exercise);
    truncations(&legacy, exercise);
}

#[test]
fn ptrhash_reader_validates_geometry() {
    let mut s = 7u64;
    let keys: Vec<u64> = (0..70_000).map(|_| splitmix(&mut s)).collect();
    let mph = Builder::new().build(&keys).unwrap();
    let mut bytes = Vec::new();
    write_ptrhash25(&mph, &mut bytes);
    let exercise = |b: &[u8]| {
        let mut pos = 0;
        if let Some(m) = read_ptrhash25(b, &mut pos) {
            let cap = m.slot_capacity();
            for &k in &keys {
                assert!((m.index_u64(k) as usize) < cap);
                let _ = m.lookup_u64(k);
            }
        }
    };
    mutate_each(&bytes, 3_000, 0xC7, exercise);
    truncations(&bytes, exercise);
}

#[test]
fn pgm_reader_never_panics_on_corruption() {
    let mut keys: Vec<u64> = (0..20_000u64).map(|i| i * 1_000_003 % 0xFFFF_FFFF).collect();
    keys.sort_unstable();
    keys.dedup();
    let pgm =
        PgmBuilder::new().with_epsilon(32).with_bloom_filter(true).build(keys.clone()).unwrap();
    let bytes = pgm.to_bytes().unwrap();
    let exercise = |b: &[u8]| {
        if let Ok(p) = PgmIndex::from_bytes(b) {
            for &k in keys.iter().step_by(7) {
                if let Ok(pos) = p.index(k) {
                    assert!(pos < keys.len());
                }
                let _ = p.lower_bound(k);
                let _ = p.index(k.wrapping_add(1));
            }
        }
    };
    mutate_each(&bytes, 2_000, 0xD8, exercise);
    truncations(&bytes, exercise);
}

#[test]
fn elias_fano_reader_never_panics_on_corruption() {
    let keys: Vec<u64> = (0..5_000u64).map(|i| i * 977 + (i % 5)).collect();
    let ef = EliasFano::from_sorted(&keys).unwrap();
    let mut bytes = Vec::new();
    ef.write_to(&mut bytes);
    let exercise = |b: &[u8]| {
        let mut pos = 0;
        if let Some(e) = EliasFano::read_from(b, &mut pos) {
            let mut out = Vec::new();
            for i in (0..e.len()).step_by(13) {
                let _ = e.get(i);
                e.materialize_range(i, 70, &mut out);
            }
        }
    };
    mutate_each(&bytes, 2_000, 0xE9, exercise);
    truncations(&bytes, exercise);
}

#[test]
fn bloom_reader_bounds_its_allocation() {
    let hashes: Vec<u64> = (0..1000u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let bf = BlockBloom::build_from_prehashed(&hashes);
    let mut bytes = Vec::new();
    bf.write_to(&mut bytes);
    // Length word claiming 2^60 words must be rejected, not allocated.
    let mut huge = bytes.clone();
    huge[8..16].copy_from_slice(&((1u64 << 60) | (1 << 63)).to_le_bytes());
    let mut pos = 0;
    assert!(BlockBloom::read_from(&huge, &mut pos).is_none());
    // Zero words: the lookup would index block 0 of nothing.
    let mut zero = bytes.clone();
    zero[8..16].copy_from_slice(&(1u64 << 63).to_le_bytes());
    let mut pos = 0;
    assert!(BlockBloom::read_from(&zero, &mut pos).is_none());
    let exercise = |b: &[u8]| {
        let mut pos = 0;
        if let Some(f) = BlockBloom::read_from(b, &mut pos) {
            for &h in &hashes {
                let _ = f.contains_hash(h);
            }
        }
    };
    mutate_each(&bytes, 500, 0xFA, exercise);
    truncations(&bytes, exercise);
    let _ = HugepageBuf::alloc_zeroed(16); // keep the import meaningful across features
    let _ = MmapIndex::open("/definitely/not/here").is_err();
}

#[test]
fn save_and_load_roundtrip_streaming() {
    let keys = byte_keys(30_000);
    let idx = IndexBuilder::new().build_index_ref(&keys).unwrap();
    let path = std::env::temp_dir().join(format!("kira_save_{}.idx", std::process::id()));
    idx.save(&path).unwrap();
    let streamed = std::fs::read(&path).unwrap();
    assert_eq!(streamed, idx.to_bytes().unwrap(), "streaming and in-memory forms differ");
    let back = Index::load(&path).unwrap();
    for k in &keys {
        assert_eq!(back.lookup(k).ok(), idx.lookup(k).ok());
    }
    let _ = std::fs::remove_file(&path);
}
