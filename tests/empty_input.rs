//! Empty-input safety for `Index` and `PgmIndex` (0.6.1+).

use kira_kv_engine::index::{Index, IndexBuilder};
use kira_kv_engine::{PgmBuilder, PgmIndex};

#[test]
fn empty_index_build_and_lookups_are_misses() {
    let idx = IndexBuilder::new().build_index(Vec::<Vec<u8>>::new()).unwrap();
    assert_eq!(idx.len(), 0);
    assert!(idx.is_empty());
    assert!(idx.lookup(b"anything").is_err());
    assert!(idx.lookup_str("anything").is_err());
    assert!(idx.lookup_u64(42).is_err());
    assert!(!idx.contains(b"anything"));
    assert_eq!(idx.lookup_batch(&[b"a", b"b", b"c"]), vec![None, None, None]);
    assert_eq!(idx.lookup_batch_u64_simd(&[1, 2, 3]), vec![None, None, None]);
}

#[test]
fn empty_index_explicit_constructor() {
    let idx = Index::empty();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
    assert!(idx.lookup_u64(0).is_err());
}

#[test]
fn empty_index_to_bytes_roundtrip() {
    let idx = Index::empty();
    let bytes = idx.to_bytes().unwrap();
    let round = Index::from_bytes(&bytes).unwrap();
    assert!(round.is_empty());
    assert!(round.lookup(b"foo").is_err());
}

#[test]
fn empty_index_save_open_mmap_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.idx");
    Index::empty().save_mmap(&path).unwrap();
    let round = Index::open_mmap(&path).unwrap();
    assert!(round.is_empty());
    assert!(round.lookup_u64(99).is_err());
}

#[test]
fn empty_pgm_build_and_query() {
    let pgm = PgmBuilder::new().with_epsilon(16).build(Vec::new()).unwrap();
    assert!(pgm.index(1).is_err());
    assert_eq!(pgm.range(0, u64::MAX), Vec::<usize>::new());
    assert_eq!(pgm.lower_bound(0), 0);
    assert_eq!(pgm.upper_bound(0), 0);
}

#[test]
fn empty_pgm_default_factory() {
    let pgm = PgmIndex::build(Vec::new(), 32).unwrap();
    assert!(pgm.index(7).is_err());
    assert!(pgm.range(0, 1_000_000).is_empty());
}

#[test]
fn empty_pgm_to_bytes_roundtrip() {
    let pgm = PgmBuilder::new().with_epsilon(8).build(Vec::new()).unwrap();
    let bytes = pgm.to_bytes().unwrap();
    let round = PgmIndex::from_bytes(&bytes).unwrap();
    assert!(round.range(0, u64::MAX).is_empty());
    assert_eq!(round.lower_bound(10), 0);
}
