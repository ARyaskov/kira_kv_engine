#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(idx) = kira_kv_engine::HybridIndex::from_bytes(data) {
        let n = idx.len() as u32;
        for k in 0..64u64 {
            if let Some(p) = idx.lookup_u64(k) {
                assert!(p < n);
            }
            let _ = idx.lookup(&data[..data.len().min(k as usize)]);
        }
        let keys: Vec<u64> = (0..40).collect();
        let _ = idx.lookup_batch_u64_simd(&keys);
        let _ = idx.storage_stats();
        let _ = kira_kv_engine::HybridIndex::from_bytes(&idx.to_bytes()).expect("re-serialized");
    }
});
