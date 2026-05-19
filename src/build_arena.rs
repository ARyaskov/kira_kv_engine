//! Slab arena for build-time temporary buffers.
//!
//! The PtrHash25 build path allocates ~5 large `Vec`s (h1, h2, bucket_idx, items, offsets)
//! plus several smaller ones per build attempt. On Windows each `Vec::new + push` chain
//! hits `HeapAlloc → VirtualAlloc` for big sizes, which costs ~50-200 μs per allocation.
//! Across 5 gamma stages × 5 buffers = 25 allocations per build, that's 1-5 ms wasted
//! per 100M build before any real work happens.
//!
//! This arena owns one contiguous backing buffer that all build-temp `Vec`-like views
//! borrow from. After each gamma stage we reset the arena (cursor → 0) and reuse the
//! same physical pages. Net effect: one upfront allocation per build instead of dozens.
//!
//! Usage:
//! ```ignore
//! let mut arena = BuildArena::with_capacity(n * 32); // overestimate
//! let h1 = arena.alloc_zeroed::<u64>(n);
//! let h2 = arena.alloc_zeroed::<u64>(n);
//! // ... build work ...
//! arena.reset(); // reuse for next gamma stage
//! ```

#![allow(dead_code)]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::Cell;
use std::marker::PhantomData;
use std::ptr::NonNull;

const DEFAULT_ALIGN: usize = 64; // cache-line aligned for all sub-allocations

/// Owned slab backing store. Drop releases via the global allocator.
pub struct BuildArena {
    ptr: NonNull<u8>,
    capacity: usize,
    cursor: Cell<usize>,
    layout: Layout,
}

// SAFETY: BuildArena owns a unique allocation; pointers handed out via `alloc_*`
// are non-aliasing slices, enforced by the &mut returned signature.
unsafe impl Send for BuildArena {}

impl BuildArena {
    /// Allocate a slab of `capacity` bytes, cache-line aligned. Rounds up to a
    /// 2 MB boundary (hugepage-friendly) but NOT to a power of two — at 4+ GB the
    /// next-power-of-two jump (e.g. 4 → 8 GB) often fails to find a contiguous
    /// virtual address range on Windows even when total free RAM is plentiful.
    pub fn with_capacity(capacity: usize) -> Self {
        let raw = capacity.max(64);
        // Round up to 2 MB (hugepage size). Keeps allocations aligned for potential
        // future hugepage backing without inflating size like next_power_of_two would.
        let cap = (raw + (2 * 1024 * 1024 - 1)) & !(2 * 1024 * 1024 - 1);
        let layout = Layout::from_size_align(cap, DEFAULT_ALIGN).expect("layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("BuildArena alloc failed (out of memory)");
        Self {
            ptr,
            capacity: cap,
            cursor: Cell::new(0),
            layout,
        }
    }

    /// Carve out a zero-initialized `&mut [T]` of `count` elements. Bumps the cursor.
    /// Panics if the request exceeds remaining capacity.
    pub fn alloc_zeroed<'a, T: Copy + Default>(&'a self, count: usize) -> ArenaSlice<'a, T> {
        let elem_size = std::mem::size_of::<T>();
        let bytes = count.checked_mul(elem_size).expect("size overflow");
        let aligned_cursor = align_up(self.cursor.get(), std::mem::align_of::<T>().max(64));
        let end = aligned_cursor + bytes;
        assert!(
            end <= self.capacity,
            "BuildArena out of space: requested {bytes} more, capacity {}, cursor {}",
            self.capacity,
            self.cursor.get()
        );
        self.cursor.set(end);
        // SAFETY: range [aligned_cursor, end) is within our buffer, T is Copy + Default.
        unsafe {
            let base = self.ptr.as_ptr().add(aligned_cursor) as *mut T;
            // Zero out (we hand zero-initialized memory regardless of prior content
            // from a previous reset cycle).
            std::ptr::write_bytes(base, 0, count);
            ArenaSlice {
                ptr: base,
                len: count,
                _marker: PhantomData,
            }
        }
    }

    /// Reset cursor for reuse (e.g., between gamma stages). Memory is NOT freed —
    /// next `alloc_*` will zero on demand.
    pub fn reset(&self) {
        self.cursor.set(0);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn used(&self) -> usize {
        self.cursor.get()
    }
}

impl Drop for BuildArena {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// `Vec`-like view into the arena. Drops without freeing (the arena owns the memory).
pub struct ArenaSlice<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> ArenaSlice<'a, T> {
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a, T> std::ops::Deref for ArenaSlice<'a, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> std::ops::DerefMut for ArenaSlice<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[inline]
fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// Estimate the arena capacity needed for a PtrHash25 build given N keys and
/// `num_buckets`. Sized to fit h1 + h2 + bucket_idx + items + offsets + small extras
/// with 20% slack.
pub fn estimate_capacity(n: usize, num_buckets: usize) -> usize {
    let h1 = n * 8;
    let h2 = n * 8;
    let bucket_idx = n * 4;
    let items = n * 4;
    let counts = num_buckets * 4;
    let offsets = (num_buckets + 1) * 4;
    let cursor = (num_buckets + 1) * 4;
    let total = h1 + h2 + bucket_idx + items + counts + offsets + cursor;
    // 20% slack for alignment padding + freq/start/order/etc.
    total + total / 5
}

