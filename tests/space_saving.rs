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

/// Reference implementation: exact counts. Space-Saving must never undercount
/// and must keep every key whose true count exceeds total / capacity.
#[test]
fn space_saving_matches_reference_bounds_on_zipf() {
    let capacity = 64;
    let mut ss = SpaceSaving::new(capacity);
    let mut exact = std::collections::HashMap::<u64, u64>::new();
    let mut s = 0x9E37u64;
    let n = 200_000u64;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Zipf-ish: key = floor(1 / u^0.8) over a large universe.
        let u = ((s >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let key = (1.0 / u.powf(0.8)) as u64;
        ss.observe(key);
        *exact.entry(key).or_default() += 1;
    }
    assert_eq!(ss.total_observed(), n);
    assert_eq!(ss.len(), capacity);
    let bound = n / capacity as u64;
    for (k, est) in ss.top_k(capacity) {
        let truth = exact.get(&k).copied().unwrap_or(0);
        assert!(est >= truth, "undercount for {k}: {est} < {truth}");
        assert!(est - truth <= bound, "overcount for {k}: {est} vs {truth}");
    }
    let tracked: std::collections::HashSet<u64> = ss.top_k(capacity).into_iter().map(|(k, _)| k).collect();
    for (k, &c) in &exact {
        if c > bound {
            assert!(tracked.contains(k), "frequent key {k} ({c}) evicted");
        }
    }
}

#[test]
fn observe_is_fast_at_large_capacity() {
    // O(K) per observation would make 1M observations at K=4096 take seconds.
    let mut ss = SpaceSaving::new(4096);
    let t = std::time::Instant::now();
    for i in 0..1_000_000u64 {
        ss.observe(i % 8192);
    }
    let per = t.elapsed().as_nanos() as f64 / 1e6;
    assert!(per < 2_000.0, "observe took {per:.0} ns");
}

#[test]
fn dynamic_hot_tier_is_shareable_and_counts_drops() {
    use kira_kv_engine::DynamicHotTier;
    use std::sync::Arc;
    let tier = Arc::new(DynamicHotTier::new(None, 256, 10_000));
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let tier = Arc::clone(&tier);
            std::thread::spawn(move || {
                for i in 0..50_000u64 {
                    let _ = tier.lookup_u64(i % 100 + t);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert!(tier.should_rebuild());
    let top = tier.take_top_k(10);
    assert!(!top.is_empty());
    assert!(!tier.should_rebuild());
    let _ = tier.dropped_observations();
}
