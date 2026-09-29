#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(idx) = kira_kv_engine::PgmIndexU128::from_bytes(data) {
        let n = idx.len();
        for k in [0u128, 1, 1 << 70, u128::MAX / 3, u128::MAX] {
            if let Ok(p) = idx.index(k) {
                assert!(p < n);
            }
            assert!(idx.lower_bound(k) <= n);
        }
        let _ = kira_kv_engine::PgmIndexU128::from_bytes(&idx.to_bytes()).expect("re-serialized");
    }
});
