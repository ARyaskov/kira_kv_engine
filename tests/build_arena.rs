use kira_kv_engine::__internal::BuildArena;

#[test]
fn alloc_and_use() {
    let arena = BuildArena::with_capacity(4096);
    let mut a = arena.alloc_zeroed::<u64>(10);
    let mut b = arena.alloc_zeroed::<u32>(20);
    for (i, x) in a.as_mut_slice().iter_mut().enumerate() {
        *x = i as u64 * 7;
    }
    for (i, x) in b.as_mut_slice().iter_mut().enumerate() {
        *x = i as u32 * 11;
    }
    assert_eq!(a[3], 21);
    assert_eq!(b[5], 55);
}

#[test]
fn reset_zeroes_on_realloc() {
    let arena = BuildArena::with_capacity(1024);
    {
        let mut a = arena.alloc_zeroed::<u32>(50);
        for x in a.as_mut_slice().iter_mut() {
            *x = 0xDEAD;
        }
    }
    arena.reset();
    let b = arena.alloc_zeroed::<u32>(50);
    for x in b.iter() {
        assert_eq!(*x, 0, "must be re-zeroed");
    }
}

#[test]
#[should_panic(expected = "BuildArena out of space")]
fn over_capacity_panics() {
    let arena = BuildArena::with_capacity(64);
    let _ = arena.alloc_zeroed::<u64>(3 * 1024 * 1024 / 8);
}
