use kira_kv_engine::__internal::radix_sort_u64_pairs;

#[test]
fn radix_pairs_matches_std_sort() {
    let mut a: Vec<(u64, u32)> =
        (0..5000u32).map(|i| (i.wrapping_mul(0x9E37) as u64, i)).collect();
    let mut b = a.clone();
    radix_sort_u64_pairs(&mut a);
    b.sort_unstable_by_key(|&(h, _)| h);
    assert_eq!(a, b);
}

#[test]
fn radix_pairs_small_path() {
    let mut a: Vec<(u64, u32)> = (0..32u32).map(|i| ((100 - i as u64), i)).collect();
    radix_sort_u64_pairs(&mut a);
    for w in a.windows(2) {
        assert!(w[0].0 <= w[1].0);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn pool_initializes_once() {
    use kira_kv_engine::__internal::pool;
    let p1 = pool();
    let p2 = pool();
    assert_eq!(p1.current_num_threads(), p2.current_num_threads());
    assert!(p1.current_num_threads() >= 1);
}
