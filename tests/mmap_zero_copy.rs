use kira_kv_engine::{Index, IndexBuilder};

#[test]
fn mapped_tables_keep_owner_and_roundtrip_all_keys() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let keys: Vec<[u8; 8]> = (0..100_000u64).map(u64::to_le_bytes).collect();
    for lean in [false, true] {
        let idx = IndexBuilder::new().with_lean_mph(lean).build_index_ref(&keys).unwrap();
        idx.save_mmap(&path).unwrap();
        let mapped = Index::open_mmap(&path).unwrap();
        assert!(!idx.is_memory_mapped());
        #[cfg(windows)]
        assert!(mapped.is_memory_mapped());
        for key in &keys {
            assert_eq!(idx.lookup(key).ok(), mapped.lookup(key).ok());
        }
        let restored = Index::from_bytes(&mapped.to_bytes().unwrap()).unwrap();
        assert_eq!(restored.len(), keys.len());
        #[cfg(windows)]
        assert!(std::fs::OpenOptions::new().write(true).open(&path).is_err());
        drop(mapped);
    }
}

#[test]
fn corruption_and_truncation_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let idx = IndexBuilder::new().build_index(vec![b"abc".to_vec(), b"def".to_vec()]).unwrap();
    idx.save_mmap(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let mut downgraded = bytes.clone();
    downgraded[8..12].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&path, downgraded).unwrap();
    assert!(Index::open_mmap(&path).is_err());
    for i in [0, 8, 12, 24, 64, bytes.len() / 2, bytes.len() - 1] {
        let mut bad = bytes.clone();
        bad[i] ^= 0x80;
        std::fs::write(&path, bad).unwrap();
        assert!(Index::open_mmap(&path).is_err(), "corruption at {i}");
    }
    for len in [0, 1, 63, 64, bytes.len() - 1] {
        std::fs::write(&path, &bytes[..len]).unwrap();
        assert!(Index::open_mmap(&path).is_err());
    }
}

#[test]
fn legacy_section_payload_remains_readable() {
    use kira_kv_engine::__internal::{MmapIndexWriter, SectionKind};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let idx = IndexBuilder::new().build_index(vec![b"key".to_vec()]).unwrap();
    let mut w = MmapIndexWriter::create(&path, 1).unwrap();
    w.add_section(SectionKind::LegacyPayload, idx.to_bytes().unwrap());
    w.finalize().unwrap();
    // Historical v1 section containers had no outer checksum.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.truncate(bytes.len() - 8);
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&path, bytes).unwrap();
    assert_eq!(Index::open_mmap(&path).unwrap().lookup(b"key").ok(), idx.lookup(b"key").ok());
}

#[cfg(windows)]
#[test]
fn safe_open_rejects_preexisting_write_mapping_after_handle_is_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    IndexBuilder::new().build_index(vec![b"key".to_vec()]).unwrap().save_mmap(&path).unwrap();
    let file = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
    // SAFETY: this test exclusively owns the file; no typed index view exists.
    let writable = unsafe { memmap2::MmapOptions::new().map_mut(&file).unwrap() };
    drop(file);
    assert!(Index::open_mmap(&path).is_err());
    drop(writable);
    assert!(Index::open_mmap(&path).unwrap().is_memory_mapped());
}

#[test]
fn checksummed_but_invalid_geometry_is_rejected() {
    use kira_kv_engine::__internal::{MmapIndex, MmapIndexWriter, SectionKind};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let keys: Vec<_> = (0..100u64).map(u64::to_le_bytes).collect();
    IndexBuilder::new().build_index_ref(&keys).unwrap().save_mmap(&path).unwrap();
    let reader = MmapIndex::open(&path).unwrap();
    let header = reader.parse_header().unwrap();
    let kinds = [
        SectionKind::PtrHash25Meta,
        SectionKind::PtrHash25Pilots,
        SectionKind::PtrHash25Parts,
        SectionKind::PtrHash25Remap,
        SectionKind::BlockBloomWords,
        SectionKind::OuterFingerprints,
    ];
    let sections: Vec<_> = kinds
        .into_iter()
        .map(|kind| (kind, reader.section(&header, kind).unwrap().to_vec()))
        .collect();
    drop(reader);
    for bad_offset in [false, true] {
        let mut writer = MmapIndexWriter::create(&path, 100).unwrap();
        for (kind, source) in &sections {
            let mut bytes = source.clone();
            if *kind == SectionKind::PtrHash25Parts {
                if bad_offset {
                    bytes[..8].fill(0xff);
                } else {
                    let buckets = bytes[28..32].to_vec();
                    bytes[32..36].copy_from_slice(&buckets);
                }
            }
            writer.add_section(*kind, bytes);
        }
        writer.finalize().unwrap();
        assert!(Index::open_mmap(&path).is_err());
    }
}
