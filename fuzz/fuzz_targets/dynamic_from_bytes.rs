#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(mut idx) = kira_kv_engine::DynamicIndex::from_bytes(data) {
        let before = idx.len();
        let _ = idx.lookup(b"probe");
        let id = idx.insert(b"probe".to_vec());
        assert_eq!(idx.lookup(b"probe"), Some(id));
        assert_eq!(idx.delete(b"probe"), Some(id));
        assert_eq!(idx.len(), before);
        let _ = kira_kv_engine::DynamicIndex::from_bytes(&idx.to_bytes()).expect("re-serialized");
    }
});
