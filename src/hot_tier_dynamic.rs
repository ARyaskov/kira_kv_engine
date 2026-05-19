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
//!   - All operations are O(1) amortized.
//!
//! At `rebuild_threshold` observations, callers can request a rebuild via
//! [`take_top_k`] which atomically swaps out the inner tracker, returns its
//! top-K (key, est_count) pairs, and resets the counters to zero. Building a
//! new [`HotTierIndex`] from those keys is the caller's responsibility — that
//! way the rebuild can be off-loaded to a background thread.

use std::sync::Mutex;

use crate::hot_tier::HotTierIndex;

/// Space-Saving frequency counter for u64 keys. Tracks up to `capacity` most
/// frequent keys with bounded error: estimated count never undershoots true
/// count, and overshoots by at most `total_observations / capacity`.
#[derive(Debug)]
pub struct SpaceSaving {
    capacity: usize,
    /// (key, estimated_count, error_bound). Kept as parallel vectors for SoA
    /// cache locality.
    keys: Vec<u64>,
    counts: Vec<u64>,
    errors: Vec<u64>,
    /// Index of the slot with the minimum count (maintained lazily).
    min_idx: usize,
    /// For O(1) lookups: maps key → slot index. We use a simple linear-probed
    /// open-addressed hash table to avoid the overhead of HashMap allocation
    /// on every observe.
    probe: Vec<i32>,        // -1 = empty, else slot index
    probe_keys: Vec<u64>,   // mirrors probe; lets us check key without indirection
    probe_mask: usize,
    total_observed: u64,
}

impl SpaceSaving {
    pub fn new(capacity: usize) -> Self {
        let probe_size = (capacity * 4).next_power_of_two().max(64);
        Self {
            capacity,
            keys: Vec::with_capacity(capacity),
            counts: Vec::with_capacity(capacity),
            errors: Vec::with_capacity(capacity),
            min_idx: 0,
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

    #[inline]
    fn hash(key: u64) -> u64 {
        // splitmix64
        let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Find probe slot for `key`. Returns `(slot, occupant)` where `occupant`
    /// is `Some(slot_idx)` if the slot already holds `key`, else `None`.
    #[inline]
    fn probe_for(&self, key: u64) -> (usize, Option<i32>) {
        let mut idx = (Self::hash(key) as usize) & self.probe_mask;
        loop {
            let v = self.probe[idx];
            if v == -1 {
                return (idx, None);
            }
            if self.probe_keys[idx] == key {
                return (idx, Some(v));
            }
            idx = (idx + 1) & self.probe_mask;
        }
    }

    /// Record one observation of `key`. O(1) amortized for the common case
    /// (`key` already in slots); O(K) on eviction due to a full `recompute_min`.
    /// At K ≤ 2× hot_set this is well under 1 µs even for K=1024.
    pub fn observe(&mut self, key: u64) {
        self.total_observed = self.total_observed.saturating_add(1);
        let (probe_idx, occ) = self.probe_for(key);
        if let Some(slot) = occ {
            // Increment existing counter — always recompute_min because in the
            // dense case (lots of duplicate counts) lazy update would miss
            // valid min changes.
            let s = slot as usize;
            self.counts[s] += 1;
            self.recompute_min();
            return;
        }
        if self.keys.len() < self.capacity {
            // Free slot available — admit the new key with count=1.
            let slot = self.keys.len() as i32;
            self.keys.push(key);
            self.counts.push(1);
            self.errors.push(0);
            self.probe[probe_idx] = slot;
            self.probe_keys[probe_idx] = key;
            self.recompute_min();
            return;
        }
        // Evict the minimum slot; the new key inherits its count.
        let evict_idx = self.min_idx;
        let old_key = self.keys[evict_idx];
        let old_count = self.counts[evict_idx];
        // Remove old key from probe table.
        self.remove_from_probe(old_key);
        // Insert new key into probe table — re-probe because `probe_idx` was
        // computed for an empty path and remove_from_probe shifted entries.
        self.keys[evict_idx] = key;
        self.counts[evict_idx] = old_count + 1;
        self.errors[evict_idx] = old_count; // error bound = pre-eviction count
        let (new_probe_idx, _) = self.probe_for(key);
        self.probe[new_probe_idx] = evict_idx as i32;
        self.probe_keys[new_probe_idx] = key;
        self.recompute_min();
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

    fn recompute_min(&mut self) {
        let mut m = self.counts[0];
        let mut mi = 0usize;
        for i in 1..self.counts.len() {
            if self.counts[i] < m {
                m = self.counts[i];
                mi = i;
            }
        }
        self.min_idx = mi;
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
        all.sort_by(|a, b| b.1.cmp(&a.1));
        all.truncate(k);
        all
    }

    /// Atomically extract top-K and reset all counters back to zero. Use this
    /// when starting a hot-tier rebuild — observations made between the call
    /// and the new tier being installed won't be lost (they'll just contribute
    /// to the *next* rebuild window).
    pub fn take_top_k_and_reset(&mut self, k: usize) -> Vec<(u64, u64)> {
        let out = self.top_k(k);
        // Reset.
        self.keys.clear();
        self.counts.clear();
        self.errors.clear();
        self.probe.fill(-1);
        self.probe_keys.fill(0);
        self.min_idx = 0;
        self.total_observed = 0;
        out
    }
}

/// Dynamic hot-tier holding a `HotTierIndex` plus a frequency tracker. The
/// inner index is replaced atomically by [`install`] after a rebuild.
///
/// Concurrency: `observe` takes a read-style lock on the inner index (None
/// blocking — uses `Mutex`); `install` takes a write lock. Callers driving
/// many concurrent lookups should clone the wrapper into an `Arc` and call
/// `observe` per lookup.
pub struct DynamicHotTier {
    inner: Mutex<HotTierState>,
}

struct HotTierState {
    index: Option<HotTierIndex>,
    tracker: SpaceSaving,
    rebuild_every: u64,
    pending_rebuild: bool,
}

impl DynamicHotTier {
    /// Construct a new dynamic hot-tier seeded with `initial` (typically the
    /// statically-built tier from the index's first build). `top_k_capacity` is
    /// the Space-Saving counter capacity (typical: 2× expected hot-set size).
    /// `rebuild_every` is the number of observations between automatic rebuild
    /// hints.
    pub fn new(initial: Option<HotTierIndex>, top_k_capacity: usize, rebuild_every: u64) -> Self {
        Self {
            inner: Mutex::new(HotTierState {
                index: initial,
                tracker: SpaceSaving::new(top_k_capacity.max(16)),
                rebuild_every,
                pending_rebuild: false,
            }),
        }
    }

    /// Look up a key in the current hot tier. Returns `Some(idx)` on hit. Always
    /// records the observation in the tracker, regardless of hit/miss — the
    /// frequency information is what drives the next rebuild.
    pub fn lookup_u64(&self, key: u64) -> Option<u32> {
        let mut guard = self.inner.lock().ok()?;
        guard.tracker.observe(key);
        let total = guard.tracker.total_observed();
        let rebuild_every = guard.rebuild_every;
        if rebuild_every > 0 && total >= rebuild_every {
            guard.pending_rebuild = true;
        }
        guard.index.as_ref().and_then(|h| h.lookup_u64(key))
    }

    /// `true` if the tracker has observed enough lookups to suggest a rebuild.
    /// Polled by background workers.
    pub fn should_rebuild(&self) -> bool {
        self.inner.lock().map(|g| g.pending_rebuild).unwrap_or(false)
    }

    /// Drain top-K observed keys and clear the rebuild flag. Caller uses these
    /// keys to build a new `HotTierIndex`, then installs it via [`install`].
    pub fn take_top_k(&self, k: usize) -> Vec<(u64, u64)> {
        match self.inner.lock() {
            Ok(mut g) => {
                g.pending_rebuild = false;
                g.tracker.take_top_k_and_reset(k)
            }
            Err(_) => Vec::new(),
        }
    }

    /// Install a freshly-built `HotTierIndex`. The old one is dropped.
    pub fn install(&self, new_index: HotTierIndex) {
        if let Ok(mut g) = self.inner.lock() {
            g.index = Some(new_index);
        }
    }

    pub fn current_memory(&self) -> usize {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.index.as_ref().map(|h| h.memory_usage()))
            .unwrap_or(0)
    }
}

