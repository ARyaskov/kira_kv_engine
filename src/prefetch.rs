//! Portable software prefetch. A hint only — never a memory access with side effects.

/// Prefetch the cache line containing `p` for a read (into L1).
#[inline(always)]
pub fn prefetch_read<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: PREFETCHT0 is a hint; it never faults and SSE is baseline on x86_64.
        unsafe { core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: PRFM is a hint; it never faults, touches no registers except
        // reading `p`, and leaves memory and flags untouched.
        unsafe {
            core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p as *const u8,
                             options(nostack, preserves_flags, readonly));
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = p;
    }
}
