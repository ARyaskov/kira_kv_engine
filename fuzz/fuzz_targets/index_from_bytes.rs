#![no_main]
//! Deserialize arbitrary bytes as an `Index` and exercise every lookup path.
//! Must never panic or read out of bounds (run under ASan for the latter).
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(idx) = kira_kv_engine::Index::from_bytes(data) {
        let cap = idx.slot_capacity();
        for k in 0..64u64 {
            if let Ok(s) = idx.lookup_u64(k) {
                assert!(s < cap);
            }
            let _ = idx.contains(&k.to_le_bytes());
            let _ = idx.lookup(&data[..data.len().min(k as usize)]);
        }
        let keys: Vec<u64> = (0..40).collect();
        let _ = idx.lookup_batch_u64_simd(&keys);
        let refs: Vec<&[u8]> = (0..70).map(|i| &data[..data.len().min(i)]).collect();
        let _ = idx.lookup_batch_pipelined(&refs);
        let _ = idx.lookup_batch(&refs);
        let _ = idx.stats();
        let _ = idx.gpu_export();
        let again = idx.to_bytes().unwrap();
        let _ = kira_kv_engine::Index::from_bytes(&again).expect("re-serialized index must load");
    }
});
