use kira_kv_engine::__internal::{PartInfo, PtrHash25Mphf, read_ptrhash25, write_ptrhash25};
use kira_kv_engine::{DynamicIndex, Index, IndexError};

fn legacy_index(mph: &[u8]) -> Vec<u8> {
    let mut bytes = vec![2];
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&[0, 0]);
    bytes.extend_from_slice(mph);
    bytes.extend_from_slice(&[0, 0]);
    bytes
}

fn mph_prefix(pilots: &[u8], fingerprint_len: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&(pilots.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(pilots.len() as u64).to_le_bytes());
    bytes.extend_from_slice(pilots);
    bytes.extend_from_slice(&fingerprint_len.to_le_bytes());
    bytes
}

#[test]
fn legacy_index_rejects_overflowing_pilot_and_fingerprint_lengths() {
    let mut pilots = mph_prefix(&[0], 0);
    pilots[20..28].copy_from_slice(&u64::MAX.to_le_bytes());
    let fingerprints = mph_prefix(&[0], u64::MAX);
    for mph in [pilots, fingerprints] {
        assert!(matches!(Index::from_bytes(&legacy_index(&mph)), Err(IndexError::CorruptData)));
    }
}

#[test]
fn ptrhash_reader_rejects_an_overflowing_initial_cursor() {
    let mut pos = usize::MAX;
    assert!(read_ptrhash25(&[0; 64], &mut pos).is_none());
}

fn part(slot_off: u64, num_keys: u32, num_slots: u32, bucket_off: u32, remap_off: u32) -> PartInfo {
    PartInfo {
        slot_off,
        salt: 0,
        bucket_off,
        num_keys,
        num_slots,
        num_buckets: 1,
        large_buckets: 0,
        remap_off,
    }
}

fn mph(n: u64, parts: Vec<PartInfo>, remap: Vec<u32>) -> PtrHash25Mphf {
    PtrHash25Mphf {
        n,
        num_buckets: parts.len() as u32,
        salt: 0,
        pilots: vec![0; parts.len()].into(),
        remap: remap.into(),
        fingerprints: Vec::new(),
        use_aes_hash: false,
        prerotate: 0,
        part_salt: 0,
        parts: parts.into_boxed_slice(),
    }
}

fn assert_geometry_rejected(mph: PtrHash25Mphf) {
    assert!(!mph.validate());
    let mut bytes = Vec::new();
    write_ptrhash25(&mph, &mut bytes);
    assert!(read_ptrhash25(&bytes, &mut 0).is_none());
}

#[test]
fn ptrhash_rejects_overlapping_or_uncovered_tables_and_output_ranges() {
    assert_geometry_rejected(mph(2, vec![part(0, 1, 3, 0, 0), part(1, 1, 3, 1, 0)], vec![0, 0]));
    assert_geometry_rejected(mph(2, vec![part(0, 1, 2, 0, 0), part(0, 1, 2, 1, 1)], vec![0, 0]));
    assert_geometry_rejected(mph(2, vec![part(0, 1, 2, 0, 0), part(1, 1, 2, 0, 1)], vec![0, 0]));
    assert_geometry_rejected(mph(3, vec![part(0, 1, 1, 0, 0), part(2, 1, 1, 1, 0)], Vec::new()));
    assert_geometry_rejected(mph(2, vec![part(0, 1, 1, 0, 0)], Vec::new()));
    assert_geometry_rejected(mph(1, vec![part(0, 1, 2, 0, 0)], vec![0, 0]));
}

#[test]
fn ptrhash_rejects_many_parts_sharing_one_large_remap_range() {
    let parts = (0..512).map(|_| part(0, 1, 4097, 0, 0)).collect();
    assert_geometry_rejected(mph(1, parts, vec![0; 4096]));
}

#[test]
fn ptrhash_preserves_contiguous_geometry_with_empty_parts() {
    let original = mph(
        3,
        vec![part(0, 0, 1, 0, 0), part(0, 1, 2, 1, 1), part(0, 0, 1, 2, 2), part(1, 2, 3, 3, 3)],
        vec![0, 0, 0, 1],
    );
    assert!(original.validate());
    let mut bytes = Vec::new();
    write_ptrhash25(&original, &mut bytes);
    let mut pos = 0;
    let restored = read_ptrhash25(&bytes, &mut pos).unwrap();
    assert_eq!(pos, bytes.len());
    assert_eq!(restored.slot_capacity(), 3);
    assert_eq!(restored.num_parts(), 4);
    for key in [0, 1, 2, u64::MAX] {
        assert!(restored.index_u64(key) < 3);
    }
}

#[test]
fn ptrhash_preserves_legacy_padded_single_and_multiple_parts() {
    let mut single = mph_prefix(&[0], 0);
    single[..8].copy_from_slice(&3u64.to_le_bytes());
    single.extend_from_slice(&[0, 0]);
    let mut pos = 0;
    let restored = read_ptrhash25(&single, &mut pos).unwrap();
    assert_eq!(pos, single.len());
    assert_eq!(restored.slot_capacity(), 3);

    let mut multiple = mph_prefix(&[0, 0], 0);
    multiple[..8].copy_from_slice(&5u64.to_le_bytes());
    multiple.extend_from_slice(&[0, 0x80]);
    multiple.extend_from_slice(&0u64.to_le_bytes());
    multiple.extend_from_slice(&2u32.to_le_bytes());
    for (slot_off, bucket_off, slots) in [(0u64, 0u32, 2u32), (2, 1, 3)] {
        multiple.extend_from_slice(&slot_off.to_le_bytes());
        multiple.extend_from_slice(&0u64.to_le_bytes());
        multiple.extend_from_slice(&bucket_off.to_le_bytes());
        multiple.extend_from_slice(&slots.to_le_bytes());
        multiple.extend_from_slice(&1u32.to_le_bytes());
    }
    let mut pos = 0;
    let restored = read_ptrhash25(&multiple, &mut pos).unwrap();
    assert_eq!(pos, multiple.len());
    assert_eq!(restored.slot_capacity(), 5);
    assert_eq!(restored.num_parts(), 2);
    for key in [0, 1, 2, u64::MAX] {
        assert!(restored.index_u64(key) < 5);
    }
}

fn dynamic_bytes(threshold: u64, next_id: u32, entries: &[(&[u8], u32)]) -> Vec<u8> {
    let mut bytes = vec![4];
    bytes.extend_from_slice(b"KIRA");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&[4, 0]);
    bytes.extend_from_slice(&threshold.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&next_id.to_le_bytes());
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for &(key, id) in entries {
        bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    let mut state = 0x4B49_5241_5F43_4B53;
    for chunk in bytes.chunks(64 * 1024) {
        state = wyhash::wyhash(chunk, state);
    }
    let sum = wyhash::wyhash(&state.to_le_bytes(), state);
    bytes.extend_from_slice(&sum.to_le_bytes());
    bytes
}

#[test]
fn dynamic_load_does_not_allocate_the_persisted_flush_threshold() {
    let threshold = usize::MAX as u64;
    let bytes = dynamic_bytes(threshold, 0, &[]);
    assert_eq!(bytes.len(), 47);
    let mut idx = DynamicIndex::from_bytes(&bytes).unwrap();
    assert!(idx.is_empty());
    assert!(idx.memory_usage() < 4096);
    let id = idx.insert(b"new".to_vec());
    assert_eq!(idx.lookup(b"new"), Some(id));
    assert_eq!(u64::from_le_bytes(idx.to_bytes()[9..17].try_into().unwrap()), threshold);

    let idx =
        DynamicIndex::from_bytes(&dynamic_bytes(threshold, 9, &[(b"alpha", 3), (b"omega", 8)]))
            .unwrap();
    assert_eq!(idx.lookup(b"alpha"), Some(3));
    assert_eq!(idx.lookup(b"omega"), Some(8));
}

#[test]
fn dynamic_load_rejects_duplicate_stable_ids() {
    let bytes = dynamic_bytes(64, 1, &[(b"a", 0), (b"b", 0)]);
    assert!(matches!(DynamicIndex::from_bytes(&bytes), Err(IndexError::CorruptData)));
}

#[test]
fn dynamic_load_rejects_a_truncated_key_with_a_large_length() {
    let mut bytes = dynamic_bytes(64, 1, &[(b"a", 0)]);
    bytes[39..43].copy_from_slice(&u32::MAX.to_le_bytes());
    let len = bytes.len() - 8;
    let state = wyhash::wyhash(&bytes[..len], 0x4B49_5241_5F43_4B53);
    let sum = wyhash::wyhash(&state.to_le_bytes(), state);
    bytes[len..].copy_from_slice(&sum.to_le_bytes());
    assert!(matches!(DynamicIndex::from_bytes(&bytes), Err(IndexError::CorruptData)));
}
