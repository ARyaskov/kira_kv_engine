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
//! ### Stable IDs
//!
//! Each insert assigns a fresh u32 id. That id is the *external* contract — it
//! survives all flushes/compactions. Internally each tier stores
//! `tier_slot → entry index`, and every tier hit is verified against the stored
//! key bytes, so a tier can never answer with another key's id: neither an MPH
//! collision of a foreign key (lean tiers) nor a fingerprint false positive
//! (1/65536 per foreign lookup otherwise) leaks through.
//!
//! ### Failure handling
//!
//! `flush()` and `compact()` build the new tier *before* touching the current
//! state; if the build fails the index is unchanged and the error is returned.
//!
//! ### Performance
//!
//! - Lookup: 25–40 ns (buffer hit) — 50–100 ns (one tier miss, hit deeper)
//! - Insert: ~700 ns amortized (HashMap insert; flush every N inserts)
//! - Delete: ~100 ns (tombstone insert)
//! - Memory: each tier keeps its `(key, id)` entries for compaction and
//!   verification, so the footprint is dominated by the key bytes themselves,
//!   plus ~5 B/key of static index and 4 B/slot of entry map.
//!
//! ### Trade-offs vs pure static
//!
//! - Lookup is ~2× slower (worst case: check buffer + all tiers).
//! - Compaction is amortized — occasional 100ms+ pauses unless you call
//!   `compact()` explicitly at a convenient moment.

use crate::index::{Index, IndexBuilder, IndexConfig, IndexError};
use hashbrown::{HashMap, HashSet};

/// A stable external id assigned by [`DynamicIndex`] on insert. Never
/// invalidated by flushes or compactions.
pub type StableId = u32;

/// Configuration for [`DynamicIndex`].
#[derive(Debug, Clone)]
pub struct DynamicConfig {
    /// Flush the write buffer to a new tier when it reaches this size.
    /// 64K is a good default (~10 ms flush cost, ~1 MB index size).
    pub flush_threshold: usize,
    /// Maximum number of immutable tiers before forced compaction merges them.
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
    tiers: Vec<Tier>,
    /// Next stable id to assign.
    next_id: StableId,
    /// Exact number of live keys.
    live: usize,
    cfg: DynamicConfig,
}

struct Tier {
    /// Static MPH on the tier's keyset.
    index: Index,
    /// `slot → index into entries`. `u32::MAX` for slots no key maps to.
    slot_to_entry: Box<[u32]>,
    /// Source `(key, id)` pairs — verification target and compaction input.
    entries: Box<[(Vec<u8>, StableId)]>,
}

impl Tier {
    #[inline]
    fn get(&self, key: &[u8]) -> Option<StableId> {
        let slot = self.index.lookup(key).ok()?;
        let e = *self.slot_to_entry.get(slot)?;
        let (stored, id) = self.entries.get(e as usize)?;
        if stored.as_slice() == key { Some(*id) } else { None }
    }
}

impl DynamicIndex {
    pub fn new() -> Self {
        Self::with_config(DynamicConfig::default())
    }

    pub fn with_config(cfg: DynamicConfig) -> Self {
        Self {
            buffer: HashMap::with_capacity(cfg.flush_threshold),
            tombstones: HashSet::new(),
            tiers: Vec::new(),
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
        self.tiers.iter().find_map(|t| t.get(key))
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

    /// Flush the write buffer to a new tier. Triggers compaction if the
    /// resulting tier count exceeds `max_tiers`.
    ///
    /// On error the buffer is left untouched and nothing is lost.
    pub fn flush(&mut self) -> Result<(), IndexError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let entries: Vec<(Vec<u8>, StableId)> =
            self.buffer.iter().map(|(k, &id)| (k.clone(), id)).collect();
        let tier = self.build_tier(entries)?;
        self.buffer.clear();
        self.tiers.insert(0, tier);
        if self.tiers.len() > self.cfg.max_tiers {
            self.compact()?;
        }
        Ok(())
    }

    /// Merge all tiers and the buffer into a single tier. Use this after a
    /// burst of inserts to bring lookup latency back to single-tier cost.
    ///
    /// On error the index is left exactly as it was.
    pub fn compact(&mut self) -> Result<(), IndexError> {
        // Collect all live entries: tiers oldest-first so newer wins, then the
        // buffer, then drop tombstoned keys.
        let mut merged: HashMap<Vec<u8>, StableId> = HashMap::with_capacity(self.live);
        for tier in self.tiers.iter().rev() {
            for (k, id) in tier.entries.iter() {
                merged.insert(k.clone(), *id);
            }
        }
        for (k, id) in self.buffer.iter() {
            merged.insert(k.clone(), *id);
        }
        for tomb in self.tombstones.iter() {
            merged.remove(tomb);
        }
        debug_assert_eq!(merged.len(), self.live);
        let new_tier = if merged.is_empty() {
            None
        } else {
            let entries: Vec<(Vec<u8>, StableId)> = merged.into_iter().collect();
            Some(self.build_tier(entries)?)
        };
        // Only now is the old state replaced.
        self.tiers.clear();
        self.buffer.clear();
        self.tombstones.clear();
        if let Some(t) = new_tier {
            self.tiers.push(t);
        }
        Ok(())
    }

    fn build_tier(&self, entries: Vec<(Vec<u8>, StableId)>) -> Result<Tier, IndexError> {
        let keys: Vec<&[u8]> = entries.iter().map(|(k, _)| k.as_slice()).collect();
        let mut cfg = IndexConfig::default();
        cfg.lean_mph = self.cfg.lean_tiers;
        cfg.enable_parallel_build = self.cfg.parallel_build;
        let index = IndexBuilder::new().with_config(cfg).build_index_ref(&keys)?;
        let mut slot_to_entry = vec![u32::MAX; index.slot_capacity()];
        for (i, k) in keys.iter().enumerate() {
            let slot = index.lookup(k)?;
            slot_to_entry[slot] = i as u32;
        }
        Ok(Tier {
            index,
            slot_to_entry: slot_to_entry.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
        })
    }

    /// Approximate memory usage in bytes (buffer + tombstones + tier indexes
    /// + entry maps + stored keys). Useful for tuning `flush_threshold`.
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
            total += tier.index.stats().total_memory;
            total += tier.slot_to_entry.len() * 4;
            total += tier.entries.len() * std::mem::size_of::<(Vec<u8>, StableId)>();
            for (k, _) in tier.entries.iter() {
                total += k.capacity();
            }
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
            for (k, id) in tier.entries.iter() {
                if self.tombstones.contains(k) || !seen.insert(k.as_slice()) {
                    continue;
                }
                write(k, *id);
            }
        }
        crate::wire::seal(crate::wire::KIND_DYNAMIC, &body)
    }

    /// Restore an index written by [`DynamicIndex::to_bytes`]. The entries are
    /// loaded into one compacted tier; stable ids are preserved.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IndexError> {
        let body = crate::wire::unseal(bytes, crate::wire::KIND_DYNAMIC).ok_or(IndexError::CorruptData)?;
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
        let mut entries: Vec<(Vec<u8>, StableId)> = Vec::with_capacity(live);
        let mut seen: HashSet<Vec<u8>> = HashSet::with_capacity(live);
        for _ in 0..live {
            let len = u32::from_le_bytes(rd(&mut pos, 4)?.try_into().unwrap()) as usize;
            let key = rd(&mut pos, len)?.to_vec();
            let id = u32::from_le_bytes(rd(&mut pos, 4)?.try_into().unwrap());
            if id >= next_id || !seen.insert(key.clone()) {
                return Err(bad());
            }
            entries.push((key, id));
        }
        if pos != body.len() {
            return Err(bad());
        }
        let mut idx = Self::with_config(DynamicConfig { flush_threshold, max_tiers, lean_tiers, parallel_build });
        idx.next_id = next_id;
        idx.live = live;
        if !entries.is_empty() {
            let tier = idx.build_tier(entries)?;
            idx.tiers.push(tier);
        }
        Ok(idx)
    }
}

impl Default for DynamicIndex {
    fn default() -> Self {
        Self::new()
    }
}
