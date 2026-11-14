use std::collections::HashMap;

use kira_kv_engine::{DynamicHotTier, SpaceSaving};

fn reference_top_k(counts: &HashMap<u64, u64>, k: usize) -> Vec<(u64, u64)> {
    let mut result: Vec<_> = counts.iter().map(|(&key, &count)| (key, count)).collect();
    result.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    result.truncate(k);
    result
}

#[test]
fn top_k_selection_matches_exact_counts_including_ties_and_limits() {
    let mut tracker = SpaceSaving::new(257);
    let mut exact = HashMap::new();
    for key in (0..257u64).rev() {
        for _ in 0..key % 13 + 1 {
            tracker.observe(key);
            *exact.entry(key).or_insert(0) += 1;
        }
    }
    for k in [0, 1, 2, 7, 63, 256, 257, usize::MAX] {
        assert_eq!(tracker.top_k(k), reference_top_k(&exact, k), "k={k}");
    }
    assert_eq!(tracker.total_observed(), exact.values().sum::<u64>());
    assert_eq!(tracker.len(), exact.len());
}

#[test]
fn resets_start_independent_epochs_for_every_requested_k() {
    let mut tracker = SpaceSaving::new(31);
    for k in [0, 1, 7, 31, usize::MAX] {
        let mut exact = HashMap::new();
        for key in 0..31u64 {
            for _ in 0..key % 5 + 1 {
                tracker.observe(key);
                *exact.entry(key).or_insert(0) += 1;
            }
        }
        assert_eq!(tracker.take_top_k_and_reset(k), reference_top_k(&exact, k));
        assert!(tracker.is_empty());
        assert_eq!(tracker.total_observed(), 0);
        assert!(tracker.top_k(usize::MAX).is_empty());
    }
    tracker.observe(u64::MAX);
    assert_eq!(tracker.top_k(1), vec![(u64::MAX, 1)]);
}

#[test]
fn repeated_evictions_preserve_estimate_bounds_and_frequent_candidates() {
    let capacity = 32;
    let mut tracker = SpaceSaving::new(capacity);
    let mut exact = HashMap::<u64, u64>::new();
    let mut total = 0u64;
    for i in 0..40_000u64 {
        let key = if i % 4 != 0 { i % 7 } else { 1000 + i / 4 };
        tracker.observe(key);
        *exact.entry(key).or_default() += 1;
        total += 1;
        if i % 997 == 0 {
            let result: HashMap<_, _> = tracker.top_k(usize::MAX).into_iter().collect();
            let bound = total / capacity as u64;
            for (&key, &estimate) in &result {
                let truth = exact[&key];
                assert!(estimate >= truth, "undercount for {key}");
                assert!(estimate - truth <= bound, "overcount for {key}");
            }
            for (&key, &count) in &exact {
                if count > bound {
                    assert!(result.contains_key(&key), "frequent key {key} evicted");
                }
            }
        }
    }
    assert_eq!(tracker.total_observed(), total);
}

#[test]
fn dynamic_extraction_resets_hint_and_preserves_the_next_epoch() {
    let tier = DynamicHotTier::new(None, 32, 3);
    for _ in 0..3 {
        tier.observe(7);
    }
    assert!(tier.should_rebuild());
    assert_eq!(tier.take_top_k(1), vec![(7, 3)]);
    assert!(!tier.should_rebuild());
    tier.observe(9);
    assert!(!tier.should_rebuild());
    tier.observe(9);
    tier.observe(11);
    assert!(tier.should_rebuild());
    assert_eq!(tier.take_top_k(usize::MAX), vec![(9, 2), (11, 1)]);
    assert!(!tier.should_rebuild());
    assert_eq!(tier.dropped_observations(), 0);
    tier.observe(13);
    assert!(tier.take_top_k(0).is_empty());
    assert!(tier.take_top_k(usize::MAX).is_empty());
}
