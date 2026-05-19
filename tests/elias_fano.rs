use kira_kv_engine::EliasFano;

fn roundtrip(keys: &[u64]) {
    let ef = EliasFano::from_sorted(keys).unwrap();
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(ef.get(i), k, "mismatch at {i}");
    }
}

#[test]
fn dense_keys() {
    let keys: Vec<u64> = (0..1000).collect();
    roundtrip(&keys);
}

#[test]
fn sparse_keys() {
    let keys: Vec<u64> = (0..1000).map(|i| (i as u64) * 1_000_003).collect();
    roundtrip(&keys);
}

#[test]
fn very_sparse_keys() {
    let keys: Vec<u64> = (0..256).map(|i| (i as u64) << 50).collect();
    roundtrip(&keys);
}

#[test]
fn materialize_range_basic() {
    let keys: Vec<u64> = (10..200).collect();
    let ef = EliasFano::from_sorted(&keys).unwrap();
    let mut out = Vec::new();
    ef.materialize_range(5, 8, &mut out);
    assert_eq!(out, vec![15, 16, 17, 18, 19, 20, 21, 22]);
}

#[test]
fn write_read_roundtrip() {
    let keys: Vec<u64> = (0..500).map(|i| (i * 13 + 7) as u64).collect();
    let ef = EliasFano::from_sorted(&keys).unwrap();
    let mut buf = Vec::new();
    ef.write_to(&mut buf);
    let mut pos = 0;
    let ef2 = EliasFano::read_from(&buf, &mut pos).unwrap();
    assert_eq!(pos, buf.len());
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(ef2.get(i), k);
    }
}
