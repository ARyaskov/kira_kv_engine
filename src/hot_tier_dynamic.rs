//! Dynamic hot-tier with LFU eviction.
//!
//! Wraps a [`HotTierIndex`] (built once for the initial top-K keys) with an
//! online frequency tracker so the hot set can drift as the workload changes.
//!
//! Frequency estimation uses **Space-Saving** (Metwally, Agrawal, El Abbadi 2005)
//! rather than Count-Min-Sketch because:
//!   - Space-Saving directly stores the *candidate* top-K keys — no separate
//!     reservoir is needed to enumerate them at rebuild time.
//!   - It has tight error bounds (overestimation only) and `O(K)` memory.
//!   - Every operation is `O(log K)` here (a binary min-heap over the counters;
//!     the paper's Stream-Summary list is O(1) but pointer-heavy).
//!
//! At `rebuild_threshold` observations, callers can request a rebuild via
//! [`DynamicHotTier::take_top_k`] which atomically swaps out the inner tracker,
//! returns its top-K (key, est_count) pairs, and resets the counters to zero.
//! Building a new [`HotTierIndex`] from those keys is the caller's
//! responsibility — that way the rebuild can be off-loaded to a background thread.
//!
//! ## Concurrency
//!
//! Lookups take a *read* lock on the installed index, so any number of threads
//! query concurrently; `install` takes the write lock for a pointer swap. The
//! frequency tracker is a single mutable structure behind a `Mutex`, but lookups
//! only `try_lock` it: under contention the observation is dropped (and counted in
//! [`DynamicHotTier::dropped_observations`]) instead of serializing the readers.
//! Space-Saving is a sampling estimator anyway, so dropped observations only cost
//! a little accuracy.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::hot_tier::HotTierIndex;

/// Space-Saving frequency counter for u64 keys. Tracks up to `capacity` most
/// frequent keys with bounded error: estimated count never undershoots true
/// count, and overshoots by at most `total_observations / capacity`.
#[derive(Debug)]
pub struct SpaceSaving {
    capacity: usize,
    /// (key, estimated_count, error_bound) per slot; SoA for locality.
    keys: Vec<u64>,
    counts: Vec<u64>,
    errors: Vec<u64>,
    /// Binary min-heap of slot indices ordered by `counts`, so the eviction
    /// victim is `heap[0]` and a counter increment is one sift-down.
    heap: Vec<u32>,
    /// Position of every slot in `heap`.
    heap_pos: Vec<u32>,
    /// key → slot: linear-probed open addressing (no per-observe allocation).
    probe: Vec<i32>, // -1 = empty, else slot index
    probe_keys: Vec<u64>,
    probe_mask: usize,
    total_observed: u64,
}

impl SpaceSaving {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let probe_size = (capacity * 4).next_power_of_two().max(64);
        Self {
            capacity,
            keys: Vec::with_capacity(capacity),
            counts: Vec::with_capacity(capacity),
            errors: Vec::with_capacity(capacity),
            heap: Vec::with_capacity(capacity),
            heap_pos: Vec::with_capacity(capacity),
            probe: vec![-1; probe_size],
            probe_keys: vec![0u64; probe_size],
            probe_mask: probe_size - 1,
            total_observed: 0,
        }
    }

    pub fn total_observed(&self) -> u64 {
        self.total_observed
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[inline]
    fn hash(key: u64) -> u64 {
        // splitmix64
        let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Probe slot for `key`: `(probe index, Some(slot))` if present, else the
    /// empty probe index where it would go.
    #[inline]
    fn probe_for(&self, key: u64) -> (usize, Option<usize>) {
        let mut idx = (Self::hash(key) as usize) & self.probe_mask;
        loop {
            let v = self.probe[idx];
            if v == -1 {
                return (idx, None);
            }
            if self.probe_keys[idx] == key {
                return (idx, Some(v as usize));
            }
            idx = (idx + 1) & self.probe_mask;
        }
    }

    /// Record one observation of `key`. O(log K).
    pub fn observe(&mut self, key: u64) {
        self.total_observed = self.total_observed.saturating_add(1);
        let (probe_idx, occ) = self.probe_for(key);
        if let Some(slot) = occ {
            self.counts[slot] += 1;
            let pos = self.heap_pos[slot] as usize;
            self.sift_down(pos);
            return;
        }
        if self.keys.len() < self.capacity {
            // Free slot available — admit the new key with count=1.
            let slot = self.keys.len();
            self.keys.push(key);
            self.counts.push(1);
            self.errors.push(0);
            self.heap.push(slot as u32);
            self.heap_pos.push(self.heap.len() as u32 - 1);
            self.probe[probe_idx] = slot as i32;
            self.probe_keys[probe_idx] = key;
            self.sift_up(self.heap.len() - 1);
            return;
        }
        // Evict the minimum; the new key inherits its count (+1) and the old
        // count becomes its error bound.
        let victim = self.heap[0] as usize;
        let old_key = self.keys[victim];
        let old_count = self.counts[victim];
        self.remove_from_probe(old_key);
        self.keys[victim] = key;
        self.counts[victim] = old_count + 1;
        self.errors[victim] = old_count;
        // Re-probe: `remove_from_probe` may have shifted the chain.
        let (new_probe_idx, _) = self.probe_for(key);
        self.probe[new_probe_idx] = victim as i32;
        self.probe_keys[new_probe_idx] = key;
        self.sift_down(0);
    }

    #[inline]
    fn heap_swap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.heap_pos[self.heap[a] as usize] = a as u32;
        self.heap_pos[self.heap[b] as usize] = b as u32;
    }

    fn sift_down(&mut self, mut pos: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * pos + 1;
            if l >= n {
                return;
            }
            let r = l + 1;
            let mut m = l;
            if r < n && self.counts[self.heap[r] as usize] < self.counts[self.heap[l] as usize] {
                m = r;
            }
            if self.counts[self.heap[m] as usize] < self.counts[self.heap[pos] as usize] {
                self.heap_swap(m, pos);
                pos = m;
            } else {
                return;
            }
        }
    }

    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if self.counts[self.heap[pos] as usize] < self.counts[self.heap[parent] as usize] {
                self.heap_swap(pos, parent);
                pos = parent;
            } else {
                return;
            }
        }
    }

    fn remove_from_probe(&mut self, key: u64) {
        let mut idx = (Self::hash(key) as usize) & self.probe_mask;
        loop {
            if self.probe[idx] != -1 && self.probe_keys[idx] == key {
                self.probe[idx] = -1;
                self.probe_keys[idx] = 0;
                // Re-insert the rest of the chain.
                let mut next = (idx + 1) & self.probe_mask;
                while self.probe[next] != -1 {
                    let k = self.probe_keys[next];
                    let v = self.probe[next];
                    self.probe[next] = -1;
                    self.probe_keys[next] = 0;
                    let (slot, _) = self.probe_for(k);
                    self.probe[slot] = v;
                    self.probe_keys[slot] = k;
                    next = (next + 1) & self.probe_mask;
                }
                return;
            }
            idx = (idx + 1) & self.probe_mask;
        }
    }

    /// Return the current top-K (key, estimated_count) sorted by count descending.
    /// `k` is clamped to `len()`.
    pub fn top_k(&self, k: usize) -> Vec<(u64, u64)> {
        let mut all: Vec<(u64, u64)> = self
            .keys
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .collect();
        all.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }

    /// Atomically extract top-K and reset all counters back to zero. Use this
    /// when starting a hot-tier rebuild — observations made between the call
    /// and the new tier being installed won't be lost (they'll just contribute
    /// to the *next* rebuild window).
    pub fn take_top_k_and_reset(&mut self, k: usize) -> Vec<(u64, u64)> {
        let out = self.top_k(k);
        self.keys.clear();
        self.counts.clear();
        self.errors.clear();
        self.heap.clear();
        self.heap_pos.clear();
        self.probe.fill(-1);
        self.probe_keys.fill(0);
        self.total_observed = 0;
        out
    }
}

/// Dynamic hot-tier holding a `HotTierIndex` plus a frequency tracker. The
/// inner index is replaced atomically by [`DynamicHotTier::install`] after a
/// rebuild. Share it between threads behind an `Arc`.
pub struct DynamicHotTier {
    index: RwLock<Option<Arc<HotTierIndex>>>,
    tracker: Mutex<SpaceSaving>,
    rebuild_every: u64,
    pending_rebuild: AtomicBool,
    dropped: AtomicU64,
}

impl DynamicHotTier {
    /// Construct a new dynamic hot-tier seeded with `initial` (typically the
    /// statically-built tier from the index's first build). `top_k_capacity` is
    /// the Space-Saving counter capacity (typical: 2× expected hot-set size).
    /// `rebuild_every` is the number of observations between automatic rebuild
    /// hints (0 = never).
    pub fn new(initial: Option<HotTierIndex>, top_k_capacity: usize, rebuild_every: u64) -> Self {
        Self {
            index: RwLock::new(initial.map(Arc::new)),
            tracker: Mutex::new(SpaceSaving::new(top_k_capacity.max(16))),
            rebuild_every,
            pending_rebuild: AtomicBool::new(false),
            dropped: AtomicU64::new(0),
        }
    }

    /// Look up a key in the current hot tier. Returns `Some(idx)` on hit. Records
    /// the observation in the tracker (hit or miss) unless another thread holds
    /// the tracker at this instant, in which case the observation is dropped.
    pub fn lookup_u64(&self, key: u64) -> Option<u32> {
        self.observe(key);
        let guard = self.index.read().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().and_then(|h| h.lookup_u64(key))
    }

    /// Record an observation without a lookup (e.g. for keys served from the
    /// static index). Non-blocking.
    pub fn observe(&self, key: u64) {
        match self.tracker.try_lock() {
            Ok(mut t) => {
                t.observe(key);
                if self.rebuild_every > 0 && t.total_observed() >= self.rebuild_every {
                    self.pending_rebuild.store(true, Ordering::Relaxed);
                }
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(std::sync::TryLockError::Poisoned(p)) => {
                let mut t = p.into_inner();
                t.observe(key);
            }
        }
    }

    /// Observations dropped because the tracker was busy. Diagnostic only.
    pub fn dropped_observations(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// `true` if the tracker has observed enough lookups to suggest a rebuild.
    /// Polled by background workers.
    pub fn should_rebuild(&self) -> bool {
        self.pending_rebuild.load(Ordering::Relaxed)
    }

    /// Drain top-K observed keys and clear the rebuild flag. Caller uses these
    /// keys to build a new `HotTierIndex`, then installs it via [`DynamicHotTier::install`].
    pub fn take_top_k(&self, k: usize) -> Vec<(u64, u64)> {
        self.pending_rebuild.store(false, Ordering::Relaxed);
        let mut t = self.tracker.lock().unwrap_or_else(|e| e.into_inner());
        t.take_top_k_and_reset(k)
    }

    /// Install a freshly-built `HotTierIndex`. Readers in flight keep the old one
    /// until they finish; it is dropped when the last reference goes away.
    pub fn install(&self, new_index: HotTierIndex) {
        let mut g = self.index.write().unwrap_or_else(|e| e.into_inner());
        *g = Some(Arc::new(new_index));
    }

    pub fn current_memory(&self) -> usize {
        let g = self.index.read().unwrap_or_else(|e| e.into_inner());
        g.as_ref().map(|h| h.memory_usage()).unwrap_or(0)
    }
}
