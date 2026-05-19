use kira_kv_engine::__internal::{MmapIndex, MmapIndexWriter};
use kira_kv_engine::{PgmBuilder, PgmIndex};

fn build_keys(n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| i * 997).collect()
}

#[test]
fn end_to_end_lookup() {
    let mut keys = build_keys(50_000);
    keys.sort_unstable();
    let pgm = PgmBuilder::new().with_epsilon(32).build(keys.clone()).unwrap();
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(pgm.index(k).unwrap(), i, "miss at key #{i}");
    }
    let miss = 42u64;
    if !keys.contains(&miss) {
        assert!(pgm.index(miss).is_err());
    }
}

#[test]
fn lookup_with_bloom() {
    let mut keys = build_keys(20_000);
    keys.sort_unstable();
    let pgm = PgmBuilder::new()
        .with_epsilon(32)
        .with_bloom_filter(true)
        .build(keys.clone())
        .unwrap();
    assert!(pgm.has_bloom());
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(pgm.index(k).unwrap(), i);
    }
}

#[test]
fn lookup_with_elias_fano() {
    let mut keys = build_keys(20_000);
    keys.sort_unstable();
    let pgm = PgmBuilder::new()
        .with_epsilon(32)
        .with_elias_fano(true)
        .build(keys.clone())
        .unwrap();
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(pgm.index(k).unwrap(), i, "ef miss at {i}");
    }
    let lb = pgm.lower_bound(keys[100]);
    assert_eq!(lb, 100);
}

#[test]
fn lookup_with_all_options() {
    let mut keys = build_keys(20_000);
    keys.sort_unstable();
    let pgm = PgmBuilder::new()
        .with_epsilon(64)
        .with_bloom_filter(true)
        .with_elias_fano(true)
        .with_target_lookup_ns(50)
        .build(keys.clone())
        .unwrap();
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(pgm.index(k).unwrap(), i);
    }
}

#[test]
fn range_returns_consecutive_positions() {
    let keys: Vec<u64> = (100..2_000).collect();
    let pgm = PgmBuilder::new().with_epsilon(16).build(keys).unwrap();
    let r = pgm.range(500, 510);
    assert_eq!(r.first(), Some(&400));
    assert_eq!(r.last(), Some(&410));
}

#[test]
fn compact_keys_preserves_lookups() {
    let mut keys = build_keys(3_000);
    keys.sort_unstable();
    let mut pgm = PgmBuilder::new().with_epsilon(32).build(keys.clone()).unwrap();
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(pgm.index(k).unwrap(), i);
    }
    let saved = pgm.compact_keys();
    assert!(saved > 0, "EF should save bytes for 3k keys, saved={saved}");
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(pgm.index(k).unwrap(), i, "post-compact miss at {i}");
    }
}

#[test]
fn mmap_sections_roundtrip() {
    let mut keys: Vec<u64> = (0..2_000u64).map(|i| i * 997).collect();
    keys.sort_unstable();
    let pgm = PgmBuilder::new()
        .with_epsilon(16)
        .with_bloom_filter(true)
        .build(keys.clone())
        .unwrap();
    let tmp = std::env::temp_dir().join(format!("kira_pgm_mmap_{}.bin", std::process::id()));
    {
        let mut w = MmapIndexWriter::create(&tmp, keys.len() as u64).unwrap();
        pgm.write_to_sections(&mut w);
        w.finalize().unwrap();
    }
    let mmap = MmapIndex::open(&tmp).unwrap();
    let header = mmap.parse_header().unwrap();
    let pgm2 = PgmIndex::read_from_sections(&mmap, &header).unwrap();
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(pgm2.index(k).unwrap(), i);
    }
    assert!(pgm2.has_bloom());
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn write_read_v2_roundtrip() {
    let mut keys = build_keys(5_000);
    keys.sort_unstable();
    let pgm = PgmBuilder::new()
        .with_epsilon(32)
        .with_bloom_filter(true)
        .build(keys.clone())
        .unwrap();
    let bytes = pgm.to_bytes().unwrap();
    let pgm2 = PgmIndex::from_bytes(&bytes).unwrap();
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(pgm2.index(k).unwrap(), i);
    }
    assert!(pgm2.has_bloom());
}
