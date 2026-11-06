//! Dynamic key→id index with LSM-tree structure on top of static MPH tiers.
//!
//! ## Why
//!
//! Pure MPH (`PtrHash25`) is bijective: every `lookup(key)` returns a stable
//! slot in `[0..n)` — but any insert/delete reshuffles the whole table. So MPH
//! is rebuild-only.
//!
//! `DynamicIndex` adds insert/delete on top by adopting the **LSM-tree**
//! pattern:
//!
//! ```text
//!                 Insert / Delete
//!                       │
//!                       ▼
//!              ┌──────────────────┐
//!              │ Write Buffer     │  hashbrown::HashMap, mutable
//!              │ (in-memory)      │  O(1) ops
//!              └────────┬─────────┘
//!                       │ flush at `flush_threshold` keys
//!                       ▼
//!              ┌──────────────────┐    youngest tier on top
//!              │ L1: small Index  │    ~flush_threshold keys
//!              ├──────────────────┤
//!              │ L2: medium Index │    ~N · ratio keys
//!              ├──────────────────┤
//!              │ L3: large Index  │    full base set
//!              └──────────────────┘    oldest at bottom
//! ```
//!
//! Tombstones (deleted keys) live in a separate `HashSet` and are checked
//! before any tier. They are dropped by `compact()`, which materialises the
//! live set.
//!
//! ### Stable IDs and compaction
//!
//! IDs survive flushes and compactions. Each immutable tier stores key bytes in
//! one arena and 16-byte metadata directly in MPH-slot order. Every lookup verifies
//! the key bytes, including lean tiers.
//!
//! Default automatic compaction preserves the original full-merge policy.
//! `set_background_compaction(true)` opts into size-tiered worker merges: four
//! similarly sized tiers merge while the old generation stays readable. The writer
//! publishes completed work at flush boundaries. `max_tiers` applies backpressure.
//! Size-tiered writes are cheaper, but multiple levels increase read amplification.
//! `compact()` restores a single tier and discards tombstones.
//!
//! A failed build keeps all key/id bindings readable. A flush may already have
//! installed its new tier before a subsequent maintenance failure is reported.
//! Worker lifetime is bounded by the index: dropping it joins outstanding work.
//!
use crate::index::{Index, IndexBuilder, IndexConfig, IndexError};
use hashbrown::{HashMap, HashSet};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

/// A stable external id assigned by [`DynamicIndex`] on insert. Never
/// invalidated by flushes or compactions.
pub type StableId = u32;

/// Configuration for [`DynamicIndex`].
#[derive(Debug, Clone)]
pub struct DynamicConfig {
    /// Flush the write buffer to a new tier when it reaches this size.
    /// 64K is a good default (~10 ms flush cost, ~1 MB index size).
    pub flush_threshold: usize,
    /// Maximum number of immutable tiers; writers wait for selected merges above it.
    /// More tiers = slower lookups (linear in tier count) but cheaper writes.
    pub max_tiers: usize,
    /// Lean mode for tier indexes (no Bloom, no fingerprints): ~4 B/key less
    /// per tier. Tier hits are always verified against the stored key, so this
    /// only trades lookup speed on misses (every tier is probed) for memory.
    pub lean_tiers: bool,
    /// Use parallel build for tier construction.
    pub parallel_build: bool,
}

impl Default for DynamicConfig {
    fn default() -> Self {
        Self {
            flush_threshold: 64 * 1024,
            max_tiers: 8,
            lean_tiers: false,
            parallel_build: cfg!(feature = "parallel"),
        }
    }
}

/// Dynamic key→id index. Supports `insert`, `delete`, `lookup` with stable
/// ids while delegating heavy lookups to compacted static `Index` tiers.
pub struct DynamicIndex {
    /// Mutable write buffer for recent inserts.
    buffer: HashMap<Vec<u8>, StableId>,
    /// Deleted keys that still exist in some tier (must be checked before any tier).
    tombstones: HashSet<Vec<u8>>,
    /// Immutable tiers, youngest at index 0. Lookup checks them in order.
    tiers: Vec<Arc<Tier>>,
    pending: Option<CompactionJob>,
    background_compaction: bool,
    size_tiering: bool,
    compaction_ratio: usize,
    compaction_count: u64,
    rebuilt_keys: u64,
    /// Next stable id to assign.
    next_id: StableId,
    /// Exact number of live keys.
    live: usize,
    cfg: DynamicConfig,
}

#[derive(Clone, Copy, Default)]
struct Entry {
    offset: usize,
    len: u32,
    id: StableId,
}

struct Tier {
    index: Index,
    /// Direct MPH-slot metadata: no slot-to-entry indirection or per-key allocation.
    entries: Box<[Entry]>,
    arena: Box<[u8]>,
}
impl Tier {
    #[inline]
    fn get(&self, key: &[u8]) -> Option<StableId> {
        let slot = self.index.lookup(key).ok()?;
        let e = self.entries.get(slot)?;
        if &self.arena[e.offset..e.offset + e.len as usize] == key { Some(e.id) } else { None }
    }
    fn iter(&self) -> impl Iterator<Item = (&[u8], StableId)> {
        self.entries.iter().map(|e| (&self.arena[e.offset..e.offset + e.len as usize], e.id))
    }
    fn memory_usage(&self) -> usize {
        self.index.stats().total_memory
            + self.arena.len()
            + std::mem::size_of_val(self.entries.as_ref())
    }
}

impl DynamicIndex {
    pub fn new() -> Self {
        Self::with_config(DynamicConfig::default())
    }

    pub fn with_config(mut cfg: DynamicConfig) -> Self {
        cfg.flush_threshold = cfg.flush_threshold.max(1);
        cfg.max_tiers = cfg.max_tiers.max(1);
        Self {
            buffer: HashMap::with_capacity(cfg.flush_threshold),
            tombstones: HashSet::new(),
            tiers: Vec::new(),
            pending: None,
            background_compaction: false,
            size_tiering: false,
            compaction_ratio: 4,
            compaction_count: 0,
            rebuilt_keys: 0,
            next_id: 0,
            live: 0,
            cfg,
        }
    }

    /// Insert a new key. Returns its stable id. If the key is already present
    /// (in the buffer or a tier), the existing id is returned and the binding
    /// is promoted into the buffer so later writes see it first.
    ///
    /// A key deleted earlier gets a fresh id — deletion is final for the old id.
    pub fn insert(&mut self, key: Vec<u8>) -> StableId {
        if let Some(&id) = self.buffer.get(&key) {
            return id;
        }
        // A revived key gets a fresh id; the stale tier copy is shadowed by the
        // buffer now and by the younger tier after the next flush.
        let revived = self.tombstones.remove(&key);
        if !revived && let Some(id) = self.lookup_in_tiers(&key) {
            // Promote into the buffer so future writes/deletes see the live
            // value first.
            self.buffer.insert(key, id);
            self.maybe_flush();
            return id;
        }
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).expect("DynamicIndex: id space exhausted");
        self.buffer.insert(key, id);
        self.live += 1;
        self.maybe_flush();
        id
    }

    #[inline]
    fn maybe_flush(&mut self) {
        if self.buffer.len() >= self.cfg.flush_threshold {
            // A failed flush keeps every entry in the buffer; the next insert
            // retries. Nothing is lost, so the error is safe to drop here.
            let _ = self.flush();
        }
    }

    /// Delete a key. Returns its prior id if it existed, else `None`.
    pub fn delete(&mut self, key: &[u8]) -> Option<StableId> {
        if self.tombstones.contains(key) {
            return None;
        }
        let in_buffer = self.buffer.remove(key);
        let in_tiers = self.lookup_in_tiers(key);
        let prior = in_buffer.or(in_tiers);
        if prior.is_some() {
            self.live -= 1;
            // A tombstone is only needed while some tier still holds the key.
            if in_tiers.is_some() {
                self.tombstones.insert(key.to_vec());
            }
        }
        prior
    }

    /// Look up a key. Checks: buffer → tombstones → tiers (youngest first).
    pub fn lookup(&self, key: &[u8]) -> Option<StableId> {
        if let Some(&id) = self.buffer.get(key) {
            return Some(id);
        }
        if self.tombstones.contains(key) {
            return None;
        }
        self.lookup_in_tiers(key)
    }

    fn lookup_in_tiers(&self, key: &[u8]) -> Option<StableId> {
        self.tiers.iter().find_map(|tier| tier.get(key))
    }

    /// Exact number of live (non-deleted) key→id mappings.
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Flush the buffer and schedule a size-tiered merge. On a tier-build error
    /// the buffer stays intact; maintenance errors never lose key/id bindings.
    pub fn flush(&mut self) -> Result<(), IndexError> {
        self.poll_compaction(false)?;
        if !self.buffer.is_empty() {
            let entries: Vec<(&[u8], StableId)> =
                self.buffer.iter().map(|(k, &id)| (k.as_slice(), id)).collect();
            let tier = Arc::new(self.build_tier(entries)?);
            self.buffer.clear();
            self.tiers.insert(0, tier);
        }
        if !self.size_tiering {
            if self.tiers.len() > self.cfg.max_tiers {
                let rebuilt = self.tiers.iter().map(|t| t.entries.len() as u64).sum::<u64>();
                self.compact()?;
                self.compaction_count += 1;
                self.rebuilt_keys += rebuilt;
            }
            return Ok(());
        }
        self.schedule_compaction()?;
        // Backpressure bounds the read amplification and retained generations.
        // Only an already-running merge is awaited; never rebuild the full set here.
        while self.tiers.len() > self.cfg.max_tiers {
            if self.pending.is_some() {
                self.poll_compaction(true)?;
            }
            self.schedule_compaction()?;
        }
        Ok(())
    }

    /// Enable size-tiered worker compaction (opt-in). Disabling joins work and
    /// keeps size-tiered merging synchronous; default indexes use full merges.
    pub fn set_background_compaction(&mut self, enabled: bool) -> Result<(), IndexError> {
        if !enabled {
            self.finish_background_compaction()?;
        }
        self.background_compaction = enabled;
        if enabled {
            self.size_tiering = true;
        }
        Ok(())
    }

    /// Size-tier fanout, at least two. The serialized key/id format is unchanged;
    /// restored indexes start with the default fanout of four.
    pub fn set_compaction_ratio(&mut self, ratio: usize) -> Result<(), IndexError> {
        if ratio < 2 {
            return Err(IndexError::Unsupported("compaction ratio must be at least two"));
        }
        self.finish_background_compaction()?;
        self.compaction_ratio = ratio;
        self.size_tiering = true;
        Ok(())
    }

    /// Wait for all currently eligible merges. Includes background work in timings.
    pub fn finish_background_compaction(&mut self) -> Result<(), IndexError> {
        loop {
            self.poll_compaction(true)?;
            self.schedule_compaction()?;
            if self.pending.is_none() {
                return Ok(());
            }
        }
    }

    /// Number of automatic merges and number of input keys rebuilt by those merges.
    pub fn compaction_stats(&self) -> (u64, u64) {
        (self.compaction_count, self.rebuilt_keys)
    }

    fn merge_range(&self) -> Option<std::ops::Range<usize>> {
        let level = |len: usize| {
            let mut n = len / self.cfg.flush_threshold;
            let mut l = 0;
            while n >= self.compaction_ratio {
                n /= self.compaction_ratio;
                l += 1;
            }
            l
        };
        let mut start = 0;
        while start < self.tiers.len() {
            let current = level(self.tiers[start].entries.len());
            let mut end = start + 1;
            while end < self.tiers.len() && level(self.tiers[end].entries.len()) == current {
                end += 1;
            }
            if end - start >= self.compaction_ratio {
                return Some(start..start + self.compaction_ratio);
            }
            start = end;
        }
        if self.tiers.len() > self.cfg.max_tiers && self.tiers.len() >= 2 {
            Some(self.tiers.len() - 2..self.tiers.len())
        } else {
            None
        }
    }

    fn schedule_compaction(&mut self) -> Result<(), IndexError> {
        if !self.size_tiering || self.pending.is_some() {
            return Ok(());
        }
        while let Some(range) = self.merge_range() {
            let inputs = self.tiers[range.clone()].to_vec();
            if self.background_compaction {
                let work = inputs.clone();
                let cfg = self.cfg.clone();
                let (tx, rx) = mpsc::sync_channel(1);
                let handle = std::thread::Builder::new().name("kira-compaction".into()).spawn(
                    move || {
                        let result = merge_tiers(&cfg, &work).map(Arc::new);
                        let _ = tx.send(result);
                    },
                )?;
                self.pending = Some(CompactionJob { inputs, receiver: Mutex::new(rx), handle });
                return Ok(());
            }
            let replacement = Arc::new(merge_tiers(&self.cfg, &inputs)?);
            self.compaction_count += 1;
            self.rebuilt_keys += inputs.iter().map(|t| t.entries.len() as u64).sum::<u64>();
            self.tiers.splice(range, [replacement]);
        }
        Ok(())
    }

    fn poll_compaction(&mut self, wait: bool) -> Result<(), IndexError> {
        let Some(job) = &self.pending else {
            return Ok(());
        };
        let received =
            {
                let receiver = job.receiver.lock().unwrap_or_else(|p| p.into_inner());
                if wait {
                    Some(receiver.recv().map_err(|_| {
                        IndexError::Mph("background compaction worker stopped".into())
                    }))
                } else {
                    match receiver.try_recv() {
                        Ok(v) => Some(Ok(v)),
                        Err(mpsc::TryRecvError::Empty) => None,
                        Err(mpsc::TryRecvError::Disconnected) => Some(Err(IndexError::Mph(
                            "background compaction worker stopped".into(),
                        ))),
                    }
                }
            };
        let Some(received) = received else {
            return Ok(());
        };
        let job = self.pending.take().unwrap();
        let _ = job.handle.join();
        let replacement = received??;
        let start = self
            .tiers
            .iter()
            .position(|t| Arc::ptr_eq(t, &job.inputs[0]))
            .expect("pending input stays installed");
        debug_assert!(
            self.tiers[start..start + job.inputs.len()]
                .iter()
                .zip(&job.inputs)
                .all(|(a, b)| Arc::ptr_eq(a, b))
        );
        self.compaction_count += 1;
        self.rebuilt_keys += job.inputs.iter().map(|t| t.entries.len() as u64).sum::<u64>();
        self.tiers.splice(start..start + job.inputs.len(), [replacement]);
        Ok(())
    }

    /// Merge all tiers and the buffer into a single tier. Use this after a
    /// burst of inserts to bring lookup latency back to single-tier cost.
    ///
    /// On error all logical key/id bindings remain unchanged.
    pub fn compact(&mut self) -> Result<(), IndexError> {
        self.finish_background_compaction()?;
        // Collect all live entries: tiers oldest-first so newer wins, then the
        // buffer, then drop tombstoned keys.
        let mut merged: HashMap<&[u8], StableId> = HashMap::with_capacity(self.live);
        for tier in self.tiers.iter().rev() {
            for (k, id) in tier.iter() {
                merged.insert(k, id);
            }
        }
        for (k, id) in self.buffer.iter() {
            merged.insert(k.as_slice(), *id);
        }
        for tomb in self.tombstones.iter() {
            merged.remove(tomb.as_slice());
        }
        debug_assert_eq!(merged.len(), self.live);
        let new_tier = if merged.is_empty() {
            drop(merged);
            None
        } else {
            let entries: Vec<(&[u8], StableId)> = merged.into_iter().collect();
            Some(self.build_tier(entries)?)
        };
        // Only now is the old state replaced.
        self.tiers.clear();
        self.buffer.clear();
        self.tombstones.clear();
        if let Some(t) = new_tier {
            self.tiers.push(Arc::new(t));
        }
        Ok(())
    }

    fn build_tier(&self, entries: Vec<(&[u8], StableId)>) -> Result<Tier, IndexError> {
        build_tier(&self.cfg, entries)
    }

    /// Approximate memory usage in bytes (buffer + tombstones + tier indexes
    /// including arena metadata and stored keys. Excludes in-flight worker scratch
    /// and unpublished output. Useful for tuning `flush_threshold`.
    pub fn memory_usage(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        total += self.buffer.capacity() * std::mem::size_of::<(Vec<u8>, StableId)>();
        for (k, _) in &self.buffer {
            total += k.capacity();
        }
        total += self.tombstones.capacity() * std::mem::size_of::<Vec<u8>>();
        for k in &self.tombstones {
            total += k.capacity();
        }
        for tier in &self.tiers {
            total += tier.memory_usage();
        }
        total
    }
}

impl DynamicIndex {
    /// Serialize the live key→id set, the id counter and the configuration into
    /// a self-contained, checksummed byte vector. Tiers are not stored; they are
    /// rebuilt on load (one compacted tier).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(self.memory_usage() / 2 + 64);
        body.extend_from_slice(&(self.cfg.flush_threshold as u64).to_le_bytes());
        body.extend_from_slice(&(self.cfg.max_tiers as u64).to_le_bytes());
        body.push(self.cfg.lean_tiers as u8);
        body.push(self.cfg.parallel_build as u8);
        body.extend_from_slice(&self.next_id.to_le_bytes());
        body.extend_from_slice(&(self.live as u64).to_le_bytes());
        let mut write = |k: &[u8], id: StableId| {
            body.extend_from_slice(&(k.len() as u32).to_le_bytes());
            body.extend_from_slice(k);
            body.extend_from_slice(&id.to_le_bytes());
        };
        for (k, &id) in &self.buffer {
            write(k, id);
        }
        // Oldest tier last so a key shadowed by a younger tier or the buffer is
        // skipped; tombstoned keys are dead.
        let mut seen: HashSet<&[u8]> = self.buffer.keys().map(|k| k.as_slice()).collect();
        for tier in &self.tiers {
            for (k, id) in tier.iter() {
                if self.tombstones.contains(k) || !seen.insert(k) {
                    continue;
                }
                write(k, id);
            }
        }
        crate::wire::seal(crate::wire::KIND_DYNAMIC, &body)
    }

    /// Restore an index written by [`DynamicIndex::to_bytes`]. The entries are
    /// loaded into one compacted tier; stable ids are preserved.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IndexError> {
        let body =
            crate::wire::unseal(bytes, crate::wire::KIND_DYNAMIC).ok_or(IndexError::CorruptData)?;
        let bad = || IndexError::CorruptData;
        let mut pos = 0usize;
        let rd = |pos: &mut usize, n: usize| -> Result<&[u8], IndexError> {
            let v = body.get(*pos..*pos + n).ok_or(IndexError::CorruptData)?;
            *pos += n;
            Ok(v)
        };
        let u64_at = |v: &[u8]| u64::from_le_bytes(v.try_into().unwrap());
        let flush_threshold = usize::try_from(u64_at(rd(&mut pos, 8)?)).map_err(|_| bad())?;
        let max_tiers = usize::try_from(u64_at(rd(&mut pos, 8)?)).map_err(|_| bad())?;
        let lean_tiers = rd(&mut pos, 1)?[0] != 0;
        let parallel_build = rd(&mut pos, 1)?[0] != 0;
        let next_id = u32::from_le_bytes(rd(&mut pos, 4)?.try_into().unwrap());
        let live = usize::try_from(u64_at(rd(&mut pos, 8)?)).map_err(|_| bad())?;
        if live > (body.len() - pos) / 8 || flush_threshold == 0 {
            return Err(bad());
        }
        let mut entries: Vec<(&[u8], StableId)> = Vec::with_capacity(live);
        let mut seen: HashSet<&[u8]> = HashSet::with_capacity(live);
        for _ in 0..live {
            let len = u32::from_le_bytes(rd(&mut pos, 4)?.try_into().unwrap()) as usize;
            let key = rd(&mut pos, len)?;
            let id = u32::from_le_bytes(rd(&mut pos, 4)?.try_into().unwrap());
            if id >= next_id || !seen.insert(key) {
                return Err(bad());
            }
            entries.push((key, id));
        }
        if pos != body.len() {
            return Err(bad());
        }
        let mut idx = Self::with_config(DynamicConfig {
            flush_threshold,
            max_tiers,
            lean_tiers,
            parallel_build,
        });
        idx.next_id = next_id;
        idx.live = live;
        if !entries.is_empty() {
            let tier = idx.build_tier(entries)?;
            idx.tiers.push(Arc::new(tier));
        }
        Ok(idx)
    }
}

impl Default for DynamicIndex {
    fn default() -> Self {
        Self::new()
    }
}

fn build_tier(cfg: &DynamicConfig, input: Vec<(&[u8], StableId)>) -> Result<Tier, IndexError> {
    let keys: Vec<&[u8]> = input.iter().map(|&(k, _)| k).collect();
    let mut index_cfg = IndexConfig::default();
    index_cfg.lean_mph = cfg.lean_tiers;
    index_cfg.enable_parallel_build = cfg.parallel_build;
    let index = IndexBuilder::new().with_config(index_cfg).build_index_ref(&keys)?;
    let bytes = input
        .iter()
        .try_fold(0usize, |sum, (key, _)| sum.checked_add(key.len()))
        .ok_or(IndexError::InvalidKey)?;
    let mut arena = Vec::with_capacity(bytes);
    let mut entries = vec![Entry::default(); index.slot_capacity()];
    for (key, id) in input {
        let len = u32::try_from(key.len()).map_err(|_| IndexError::InvalidKey)?;
        let slot = index.lookup(key)?;
        entries[slot] = Entry { offset: arena.len(), len, id };
        arena.extend_from_slice(key);
    }
    Ok(Tier { index, entries: entries.into_boxed_slice(), arena: arena.into_boxed_slice() })
}

struct CompactionJob {
    inputs: Vec<Arc<Tier>>,
    receiver: Mutex<mpsc::Receiver<Result<Arc<Tier>, IndexError>>>,
    handle: JoinHandle<()>,
}

fn merge_tiers(cfg: &DynamicConfig, tiers: &[Arc<Tier>]) -> Result<Tier, IndexError> {
    let mut merged: HashMap<&[u8], StableId> =
        HashMap::with_capacity(tiers.iter().map(|t| t.entries.len()).sum());
    // Newest wins. Global tombstones stay installed while any older tier can exist.
    for tier in tiers.iter().rev() {
        for (key, id) in tier.iter() {
            merged.insert(key, id);
        }
    }
    build_tier(cfg, merged.into_iter().collect())
}

impl Drop for DynamicIndex {
    fn drop(&mut self) {
        if let Some(job) = self.pending.take() {
            let _ = job.handle.join();
        }
    }
}
