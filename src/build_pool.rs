//! Shared build-time resources: a persistent rayon thread pool and a
//! u64-specialized radix sort.
//!
//! ## Persistent rayon pool
//!
//! The library previously created a fresh rayon pool inside every Index::build
//! call. Each pool created on Windows costs ~150 ms of CreateThread overhead
//! for the worker fan-out — dominant cost on small (<10M) indexes. With
//! `pool()` we initialize a single hybrid-aware pool once and reuse it for
//! every subsequent build/flush.
//!
//! On Intel 12th-gen+ hybrid hardware we pin the workers to the P-cores
//! (returned by `Topology::detect`) so the build hot loop doesn't get
//! scheduled onto the lower-IPC E-cores.
//!
//! ## Radix sort
//!
//! `std::sort_unstable_by_key(|&(h, _)| h)` on `(u64, u32)` is a comparison
//! sort: O(N log N) with branch-heavy compares. For >1M elements LSD radix
//! sort on the 64-bit hash beats it by 2-3× because every byte-pass is a
//! pure linear sweep over memory (~3 GB/s).

#[cfg(feature = "parallel")]
use std::sync::OnceLock;

#[cfg(feature = "parallel")]
static GLOBAL_BUILD_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Get the global build thread pool. Initialized lazily on first call;
/// thereafter shared by every Index/PgmIndex/HybridIndex build.
#[cfg(feature = "parallel")]
pub fn pool() -> &'static rayon::ThreadPool {
    GLOBAL_BUILD_POOL.get_or_init(|| {
        let threads = pick_thread_count();
        let core_ids = pick_pinning_cores();
        let mut builder = rayon::ThreadPoolBuilder::new().num_threads(threads);
        if let Some(cores) = core_ids {
            let cores = std::sync::Arc::new(cores);
            let cores_handler = std::sync::Arc::clone(&cores);
            builder = builder.start_handler(move |idx| {
                if !cores_handler.is_empty() {
                    let core_id = cores_handler[idx % cores_handler.len()];
                    let _ =
                        core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                }
            });
        }
        builder.build().expect("build_pool: rayon pool init failed")
    })
}

#[cfg(feature = "parallel")]
fn pick_thread_count() -> usize {
    if let Some(v) = std::env::var_os("KIRA_BUILD_THREADS") {
        if let Ok(parsed) = v.to_string_lossy().parse::<usize>() {
            return parsed.max(1);
        }
    }
    let topo = crate::hybrid_topology::Topology::detect();
    if topo.is_hybrid && !topo.performance_cores.is_empty() {
        return topo.performance_cores.len();
    }
    std::thread::available_parallelism()
        .map(|n| {
            let t = n.get();
            #[cfg(target_arch = "x86_64")]
            {
                (t / 2).clamp(4, 16)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                t.clamp(2, 8)
            }
        })
        .unwrap_or(4)
}

#[cfg(feature = "parallel")]
fn pick_pinning_cores() -> Option<Vec<usize>> {
    if let Some(v) = std::env::var_os("KIRA_BUILD_CORE_IDS") {
        let ids: Vec<usize> = v
            .to_string_lossy()
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        if !ids.is_empty() {
            return Some(ids);
        }
    }
    let topo = crate::hybrid_topology::Topology::detect();
    if topo.is_hybrid && !topo.performance_cores.is_empty() {
        return Some(topo.performance_cores);
    }
    let cores = core_affinity::get_core_ids()?;
    if cores.is_empty() {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let half = (cores.len() / 2).clamp(1, 16);
        return Some(cores.iter().take(half).map(|c| c.id).collect());
    }
    #[allow(unreachable_code)]
    Some(cores.iter().map(|c| c.id).collect())
}

/// LSD radix sort for `(u64 key, u32 payload)` pairs, sorted by the u64 key.
///
/// Faster than `sort_unstable_by_key` for N > ~1024 because every pass is a
/// linear sweep with branch-free address generation. 8 byte-passes × O(N) =
/// O(8N), vs O(N log N) for comparison sort.
///
/// For small N falls back to the standard library sort to amortize the
/// scratch-buffer allocation.
pub fn radix_sort_u64_pairs(arr: &mut Vec<(u64, u32)>) {
    if arr.len() < 1024 {
        arr.sort_unstable_by_key(|&(h, _)| h);
        return;
    }
    let n = arr.len();
    let mut buf: Vec<(u64, u32)> = vec![(0, 0); n];
    // Ping-pong between `arr` and `buf` across 8 passes (one per byte). After
    // an even number of passes (8 == even) the sorted data lands back in `arr`.
    let mut src: &mut [(u64, u32)] = arr.as_mut_slice();
    let mut dst: &mut [(u64, u32)] = buf.as_mut_slice();
    for shift in (0..64u32).step_by(8) {
        let mut counts = [0usize; 256];
        for &(h, _) in src.iter() {
            counts[((h >> shift) & 0xff) as usize] += 1;
        }
        // Convert to prefix-sum offsets.
        let mut running = 0usize;
        for c in counts.iter_mut() {
            let here = *c;
            *c = running;
            running += here;
        }
        // Scatter.
        for &(h, p) in src.iter() {
            let bucket = ((h >> shift) & 0xff) as usize;
            unsafe {
                *dst.get_unchecked_mut(counts[bucket]) = (h, p);
            }
            counts[bucket] += 1;
        }
        std::mem::swap(&mut src, &mut dst);
    }
}
