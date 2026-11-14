use kira_kv_engine::EliasFano;

fn clustered_keys(n: usize) -> Vec<u64> {
    let last = (n as u64) * n as u64;
    std::iter::once(0).chain(last - n as u64 + 2..=last).collect()
}

#[test]
fn sparse_cluster_random_access_preserves_every_key() {
    let keys = clustered_keys(4096);
    let index = EliasFano::from_sorted(&keys).unwrap();
    for (rank, &key) in keys.iter().enumerate() {
        assert_eq!(index.get(rank), key);
    }
    for rank in [1, 63, 64, 65, keys.len() - 1].into_iter().cycle().take(1000) {
        assert_eq!(index.get(rank), keys[rank]);
    }
    let mut bytes = Vec::new();
    index.write_to(&mut bytes);
    let restored = EliasFano::read_from(&bytes, &mut 0).unwrap();
    for (rank, &key) in keys.iter().enumerate() {
        assert_eq!(restored.get(rank), key);
    }
}

#[test]
fn materialization_jumps_gaps_within_and_between_rank_blocks() {
    let n = 4096;
    let end = (n * n) as u64;
    let boundary_gap: Vec<u64> =
        (0..64).map(|key| key * 2).chain(end - (n as u64 - 64) + 1..=end).collect();
    for keys in [clustered_keys(n), boundary_gap] {
        let index = EliasFano::from_sorted(&keys).unwrap();
        let mut output = Vec::new();
        for (from, count) in [(0, 129), (1, 63), (60, 70), (63, 3), (64, 65), (n - 2, 9)] {
            index.materialize_range(from, count, &mut output);
            assert_eq!(output, keys[from..keys.len().min(from + count)]);
        }
    }
}

#[test]
fn materialization_clamps_huge_counts_before_reserving_or_adding() {
    let keys = [0, 2, 9, 1000];
    let index = EliasFano::from_sorted(&keys).unwrap();
    let mut output = vec![99];
    index.materialize_range(1, usize::MAX, &mut output);
    assert_eq!(output, keys[1..]);
    index.materialize_range(usize::MAX, usize::MAX, &mut output);
    assert!(output.is_empty());
    index.materialize_range(index.len(), usize::MAX, &mut output);
    assert!(output.is_empty());
}

#[test]
fn sorted_constructor_rejects_non_unique_or_descending_input() {
    assert!(EliasFano::from_sorted(&[]).is_none());
    assert!(EliasFano::from_sorted(&[1, 1]).is_none());
    assert!(EliasFano::from_sorted(&[2, 1]).is_none());
    assert!(EliasFano::from_sorted(&[0, 9, 3]).is_none());
    for keys in [vec![0], vec![u64::MAX], vec![0, u64::MAX]] {
        let index = EliasFano::from_sorted(&keys).unwrap();
        for (rank, &key) in keys.iter().enumerate() {
            assert_eq!(index.get(rank), key);
        }
    }
}

#[test]
fn random_access_rejects_out_of_bounds_indices_in_every_profile() {
    let index = EliasFano::from_sorted(&[7, 11, 19]).unwrap();
    for rank in [index.len(), usize::MAX] {
        assert!(std::panic::catch_unwind(|| index.get(rank)).is_err());
    }
}

#[test]
fn frozen_legacy_samples_preserve_bytes_and_cursor_boundaries() {
    let fixture = include_bytes!("fixtures/elias_fano_v081.bin");
    let mut bytes = vec![7, 11];
    bytes.extend_from_slice(fixture);
    let end = bytes.len();
    bytes.extend_from_slice(b"following value");
    let mut cursor = 2;
    let index = EliasFano::read_from(&bytes, &mut cursor).unwrap();
    assert_eq!(cursor, end);
    assert_eq!(index.len(), 256);
    for rank in 0..256 {
        assert_eq!(index.get(rank), rank as u64 * 13 + 7);
    }
    let mut rewritten = Vec::new();
    index.write_to(&mut rewritten);
    assert_eq!(rewritten, fixture);
}

#[test]
fn invalid_input_never_advances_the_public_cursor() {
    let fixture = include_bytes!("fixtures/elias_fano_v081.bin");
    for length in [0, 1, 7, 8, 16, fixture.len() - 1] {
        let mut cursor = 0;
        assert!(EliasFano::read_from(&fixture[..length], &mut cursor).is_none());
        assert_eq!(cursor, 0);
    }
    let mut cursor = usize::MAX;
    assert!(EliasFano::read_from(fixture, &mut cursor).is_none());
    assert_eq!(cursor, usize::MAX);
    let mut changed = fixture.to_vec();
    let last = changed.len() - 4;
    changed[last..].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(EliasFano::read_from(&changed, &mut 0).is_none());
}
