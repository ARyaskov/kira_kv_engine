use kira_kv_engine::__internal::{MmapIndex, MmapIndexWriter, SectionKind};

#[test]
fn round_trip_sections() {
    let tmp = std::env::temp_dir().join(format!("kira_mmap_test_{}.bin", std::process::id()));
    {
        let mut w = MmapIndexWriter::create(&tmp, 12345).expect("create");
        w.add_section(SectionKind::PtrHash25Pilots, vec![1u8, 2, 3, 4]);
        w.add_section(SectionKind::BlockBloomWords, vec![0xAB; 128]);
        w.finalize().expect("finalize");
    }
    let reader = MmapIndex::open(&tmp).expect("open");
    let header = reader.parse_header().expect("header");
    assert_eq!(header.key_count, 12345);
    assert_eq!(header.sections.len(), 2);
    let pilots = reader
        .section(&header, SectionKind::PtrHash25Pilots)
        .expect("pilots");
    assert_eq!(pilots, &[1, 2, 3, 4]);
    let bloom = reader
        .section(&header, SectionKind::BlockBloomWords)
        .expect("bloom");
    assert_eq!(bloom.len(), 128);
    assert!(bloom.iter().all(|&b| b == 0xAB));
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn rejects_bad_magic() {
    let tmp = std::env::temp_dir().join(format!("kira_mmap_bad_{}.bin", std::process::id()));
    std::fs::write(&tmp, b"NOT_A_KIRA_INDEX_HEADER........").expect("write");
    let reader = MmapIndex::open(&tmp).expect("open");
    let err = reader.parse_header().unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = std::fs::remove_file(&tmp);
}
