use kira_kv_engine::SpaceSaving;

#[test]
fn space_saving_tracks_top_k() {
    let mut ss = SpaceSaving::new(8);
    for _ in 0..50 {
        for &hot in &[1u64, 2, 3, 4, 5] {
            ss.observe(hot);
        }
    }
    for cold in 100u64..130 {
        ss.observe(cold);
    }
    let top = ss.top_k(5);
    let hot_keys: std::collections::HashSet<u64> = top.iter().map(|(k, _)| *k).collect();
    for k in &[1u64, 2, 3, 4, 5] {
        assert!(hot_keys.contains(k), "missing hot key {k}");
    }
}

#[test]
fn take_and_reset() {
    let mut ss = SpaceSaving::new(4);
    ss.observe(10);
    ss.observe(20);
    ss.observe(10);
    let top = ss.take_top_k_and_reset(2);
    assert_eq!(top.len(), 2);
    assert_eq!(ss.len(), 0);
    assert_eq!(ss.total_observed(), 0);
}
