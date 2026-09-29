#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(pgm) = kira_kv_engine::PgmIndex::from_bytes(data) {
        let n = pgm.stats().total_keys;
        for k in [0u64, 1, 7, 1 << 20, u64::MAX / 2, u64::MAX] {
            if let Ok(p) = pgm.index(k) {
                assert!(p < n);
            }
            assert!(pgm.lower_bound(k) <= n);
            let r = pgm.range(k, k.wrapping_add(1000));
            assert!(r.end <= n);
        }
        let _ = kira_kv_engine::PgmIndex::from_bytes(&pgm.to_bytes().unwrap()).expect("re-serialized");
    }
});
