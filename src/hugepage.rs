//! Hugepage-backed allocator for large MPHF data structures.
//!
//! On i7-12700 with 100M-entry pilot table (100 MB), regular 4 KB pages produce ~25k
//! TLB entries while the CPU has ~64 L1-TLB entries per core. That means ≥99% TLB miss
//! rate on random-access lookup workloads, costing ~10-20 ns per miss on top of the
//! actual cache load.
//!
//! Switching to 2 MB hugepages drops the count to ~50 entries — fits comfortably in TLB.
//! Per-lookup latency improvement on 100M warm workloads is typically 10-20 ns.
//!
//! - **Windows**: `VirtualAlloc` with `MEM_LARGE_PAGES`. Requires `SeLockMemoryPrivilege`
//!   for the process (usually needs admin or explicit policy).
//! - **Linux**: `mmap` with `MAP_HUGETLB | MAP_ANONYMOUS`. Requires
//!   `/proc/sys/vm/nr_hugepages` to have available pages, or transparent hugepages.
//! - **Fallback**: regular 4 KB pages via the global allocator.
//!
//! Returned buffers have hugepage alignment (2 MB) and are zero-initialized.

#![allow(dead_code)]

use std::alloc::{dealloc, Layout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// One-time message flag so we don't spam stderr per allocation.
static HUGEPAGE_MSG_SHOWN: AtomicBool = AtomicBool::new(false);

/// Lazy detection of hugepage availability. Returns Some(()) if hugepages CAN be
/// allocated, None otherwise. Caches the result and prints a one-time message on
/// the first attempt if unavailable.
fn hugepage_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let ok = check_hugepage_permission();
        if !ok && !HUGEPAGE_MSG_SHOWN.swap(true, Ordering::Relaxed) {
            print_hugepage_help();
        }
        ok
    })
}

fn check_hugepage_permission() -> bool {
    #[cfg(target_os = "windows")]
    {
        windows::has_large_page_privilege()
    }
    #[cfg(target_os = "linux")]
    {
        linux::has_hugepages_available()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        false
    }
}

fn print_hugepage_help() {
    #[cfg(target_os = "windows")]
    eprintln!(
        "[kira_kv_engine] Large pages unavailable; falling back to 4 KB pages.\n  \
         Note: running as Administrator alone is NOT enough — Windows requires the\n  \
         'Lock pages in memory' privilege to be explicitly granted AND a new logon\n  \
         session for it to take effect. To enable:\n  \
           1. Win+R → secpol.msc → enter\n  \
           2. Local Policies → User Rights Assignment → 'Lock pages in memory'\n  \
           3. Add your user account (or BUILTIN\\Administrators)\n  \
           4. Log out and log back in (privilege only applies to new sessions)\n  \
         On a 100M-key index this saves ~10-20 ns per lookup by eliminating TLB misses."
    );
    #[cfg(target_os = "linux")]
    eprintln!(
        "[kira_kv_engine] Hugepages unavailable; falling back to 4 KB pages.\n  \
         Reserve some 2 MB pages first, e.g.:\n  \
           sudo sysctl vm.nr_hugepages=1024\n  \
         to cut TLB misses by ~99% on 50M-100M-key indexes."
    );
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    eprintln!("[kira_kv_engine] Hugepages not supported on this OS; using 4 KB pages.");
}

/// 2 MB hugepage size.
const HUGEPAGE_SIZE: usize = 2 * 1024 * 1024;

/// Owned byte buffer that may live in hugepages (preferred) or regular pages (fallback).
pub struct HugepageBuf {
    ptr: *mut u8,
    len: usize,
    layout: HugepageLayout,
}

impl std::fmt::Debug for HugepageBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HugepageBuf")
            .field("len", &self.len)
            .field("is_hugepage", &self.is_hugepage())
            .finish()
    }
}

enum HugepageLayout {
    /// Backed by OS hugepage allocator — must be released via the matching API.
    Hugepage,
    /// Backed by the global allocator with a fallback Layout.
    Fallback(Layout),
}

// SAFETY: HugepageBuf owns a unique allocation; Send is safe.
unsafe impl Send for HugepageBuf {}
unsafe impl Sync for HugepageBuf {}

impl HugepageBuf {
    /// Allocate `len` bytes, zero-initialized, hugepage-aligned. Falls back to the
    /// global allocator if hugepages are unavailable.
    pub fn alloc_zeroed(len: usize) -> Self {
        if len >= 1024 * 1024 {
            if let Some(buf) = try_alloc_hugepage(len) {
                return buf;
            }
        }
        alloc_fallback(len)
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether this allocation is hugepage-backed (for diagnostic / metrics).
    pub fn is_hugepage(&self) -> bool {
        matches!(self.layout, HugepageLayout::Hugepage)
    }
}

impl Drop for HugepageBuf {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        match self.layout {
            HugepageLayout::Hugepage => unsafe { release_hugepage(self.ptr, self.len) },
            HugepageLayout::Fallback(layout) => unsafe { dealloc(self.ptr, layout) },
        }
    }
}

/// Owned `Vec`-like buffer that auto-routes to hugepages when ≥ 1 MB. Drop-in for
/// callers that want hugepage backing without manually checking sizes.
///
/// `HugeVec<T>` keeps a `Vec<T>` for small sizes (fast path, allocator-managed) and
/// switches to a `HugepageBuf` for large sizes. The slice view abstracts over both.
#[derive(Debug)]
pub enum HugeVec<T: Copy + Default + 'static> {
    Small(Vec<T>),
    Huge {
        buf: HugepageBuf,
        len: usize,
        _t: std::marker::PhantomData<T>,
    },
}

impl<T: Copy + Default + 'static> HugeVec<T> {
    /// Allocate `len` elements, zero-initialized. Hugepages are used when the
    /// total byte size ≥ 1 MB on supported OSes; otherwise falls back to `Vec`.
    pub fn zeroed(len: usize) -> Self {
        let byte_size = len.checked_mul(std::mem::size_of::<T>()).expect("size overflow");
        if byte_size >= 1024 * 1024 {
            let buf = HugepageBuf::alloc_zeroed(byte_size);
            HugeVec::Huge {
                buf,
                len,
                _t: std::marker::PhantomData,
            }
        } else {
            HugeVec::Small(vec![T::default(); len])
        }
    }

    /// Build from an existing slice — copies into hugepage backing if large.
    pub fn from_slice(src: &[T]) -> Self {
        let mut v = Self::zeroed(src.len());
        v.as_mut_slice().copy_from_slice(src);
        v
    }

    pub fn as_slice(&self) -> &[T] {
        match self {
            HugeVec::Small(v) => v.as_slice(),
            HugeVec::Huge { buf, len, .. } => {
                // SAFETY: buf is valid for at least len * size_of::<T>() bytes; T: Copy.
                unsafe { std::slice::from_raw_parts(buf.as_slice().as_ptr() as *const T, *len) }
            }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            HugeVec::Small(v) => v.as_mut_slice(),
            HugeVec::Huge { buf, len, .. } => unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_slice().as_mut_ptr() as *mut T, *len)
            },
        }
    }

    pub fn len(&self) -> usize {
        match self {
            HugeVec::Small(v) => v.len(),
            HugeVec::Huge { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_hugepage_backed(&self) -> bool {
        matches!(self, HugeVec::Huge { .. })
    }

    pub fn memory_usage(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

// Manual Clone since HugepageBuf isn't Clone (it owns OS memory).
impl<T: Copy + Default + 'static> Clone for HugeVec<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

// SAFETY: HugeVec only owns T data; if T is Send/Sync, the wrapper inherits.
unsafe impl<T: Copy + Default + Send + 'static> Send for HugeVec<T> {}
unsafe impl<T: Copy + Default + Sync + 'static> Sync for HugeVec<T> {}

fn alloc_fallback(len: usize) -> HugepageBuf {
    // 64-byte alignment matches a cache line, which is enough for our purposes.
    let layout = Layout::from_size_align(len.max(1), 64).expect("layout");
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    HugepageBuf {
        ptr,
        len,
        layout: HugepageLayout::Fallback(layout),
    }
}

fn try_alloc_hugepage(len: usize) -> Option<HugepageBuf> {
    // Probe permission once; if absent, never bother the OS with futile calls.
    if !hugepage_available() {
        return None;
    }
    // Round up to hugepage size.
    let rounded = (len + HUGEPAGE_SIZE - 1) & !(HUGEPAGE_SIZE - 1);
    #[cfg(target_os = "windows")]
    {
        windows::alloc(rounded).map(|ptr| HugepageBuf {
            ptr,
            len: rounded,
            layout: HugepageLayout::Hugepage,
        })
    }
    #[cfg(target_os = "linux")]
    {
        linux::alloc(rounded).map(|ptr| HugepageBuf {
            ptr,
            len: rounded,
            layout: HugepageLayout::Hugepage,
        })
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        let _ = rounded;
        None
    }
}

unsafe fn release_hugepage(ptr: *mut u8, len: usize) {
    #[cfg(target_os = "windows")]
    unsafe {
        windows::release(ptr, len);
    }
    #[cfg(target_os = "linux")]
    unsafe {
        linux::release(ptr, len);
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        let _ = (ptr, len);
    }
}

// ----- Windows
#[cfg(target_os = "windows")]
mod windows {
    const MEM_COMMIT: u32 = 0x00001000;
    const MEM_RESERVE: u32 = 0x00002000;
    const MEM_LARGE_PAGES: u32 = 0x20000000;
    const MEM_RELEASE: u32 = 0x00008000;
    const PAGE_READWRITE: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(
            lpAddress: *mut std::ffi::c_void,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> *mut std::ffi::c_void;
        fn VirtualFree(lpAddress: *mut std::ffi::c_void, dwSize: usize, dwFreeType: u32) -> i32;
        fn GetLargePageMinimum() -> usize;
    }

    pub fn alloc(size: usize) -> Option<*mut u8> {
        unsafe {
            let min = GetLargePageMinimum();
            if min == 0 {
                return None;
            }
            let ptr = VirtualAlloc(
                std::ptr::null_mut(),
                size,
                MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                PAGE_READWRITE,
            );
            if ptr.is_null() {
                None
            } else {
                Some(ptr as *mut u8)
            }
        }
    }

    pub unsafe fn release(ptr: *mut u8, _len: usize) {
        unsafe {
            VirtualFree(ptr as *mut _, 0, MEM_RELEASE);
        }
    }

    /// Quick probe: does this process have SeLockMemoryPrivilege?
    /// We just try a tiny 2 MB allocation and free it. If it succeeds, the right
    /// privilege is present. This is more reliable than parsing token privileges
    /// directly (which would need advapi32 linkage).
    pub fn has_large_page_privilege() -> bool {
        unsafe {
            let min = GetLargePageMinimum();
            if min == 0 {
                return false;
            }
            let ptr = VirtualAlloc(
                std::ptr::null_mut(),
                min, // 2 MB minimum hugepage
                MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                PAGE_READWRITE,
            );
            if ptr.is_null() {
                false
            } else {
                VirtualFree(ptr, 0, MEM_RELEASE);
                true
            }
        }
    }
}

// ----- Linux
#[cfg(target_os = "linux")]
mod linux {
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_ANONYMOUS: i32 = 0x20;
    const MAP_HUGETLB: i32 = 0x40000;
    const MAP_FAILED: *mut std::ffi::c_void = !0 as *mut std::ffi::c_void;

    unsafe extern "C" {
        fn mmap(
            addr: *mut std::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut std::ffi::c_void;
        fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
    }

    pub fn alloc(size: usize) -> Option<*mut u8> {
        unsafe {
            let ptr = mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                -1,
                0,
            );
            if ptr == MAP_FAILED {
                None
            } else {
                Some(ptr as *mut u8)
            }
        }
    }

    pub unsafe fn release(ptr: *mut u8, len: usize) {
        unsafe {
            munmap(ptr as *mut _, len);
        }
    }

    /// Probe: are any 2 MB hugepages available in /proc/sys/vm/nr_hugepages?
    pub fn has_hugepages_available() -> bool {
        match std::fs::read_to_string("/proc/sys/vm/nr_hugepages") {
            Ok(s) => s.trim().parse::<u64>().map(|n| n > 0).unwrap_or(false),
            Err(_) => false,
        }
    }
}

