use kira_kv_engine::__internal::HugepageBuf;

#[test]
fn alloc_zero_init_small() {
    let mut buf = HugepageBuf::alloc_zeroed(1024);
    assert_eq!(buf.len(), 1024);
    for b in buf.as_slice() {
        assert_eq!(*b, 0);
    }
    buf.as_mut_slice()[0] = 0xAB;
    assert_eq!(buf.as_slice()[0], 0xAB);
}

#[test]
fn alloc_large_falls_back_gracefully() {
    let buf = HugepageBuf::alloc_zeroed(16 * 1024 * 1024);
    assert!(buf.len() >= 16 * 1024 * 1024);
}
