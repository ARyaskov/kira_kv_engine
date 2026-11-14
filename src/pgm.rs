//! PGM (Piecewise Geometric Model) learned index for sorted u64 keys.
//!
//! Implementation:
//!   - **Relative f64 slopes**: integer key offsets avoid loss of precision near u64::MAX.
//!   - **Per-segment u8 epsilon** (`max_errors_u8`) with overflow Vec for rare cases >254.
//!   - **Branchless SIMD local search** via `movemask + tzcnt` (no scalar fallback inside SIMD loop).
//!   - **SIMD-aware branchless segment lookup** with cache-line prefetch on each step.
//!   - **Anchored slope-window segmentation** with O(N) fitting and error measurement.
//!   - **Parallel chunked build** via rayon (independent chunks, then merge).
//!   - **Optional Block-Bloom** fast-path for negative lookups (10–100× faster miss).
//!   - **Auto-tune epsilon** via a cost model sampled from the key distribution.
//!
//! Standalone serialization uses a checksummed container around the v6 payload.
//! Legacy raw v2/v3/v4 absolute models remain readable and retain their numeric
//! representation when saved again, including locator-only HybridIndex models.

use thiserror::Error;

use crate::block_bloom::BlockBloom;

use crate::prefetch::prefetch_read;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vceqq_u64, vcgeq_u64, vdupq_n_u64, vgetq_lane_u64, vld1q_u64};
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_movemask_epi8,
    _mm256_or_si256, _mm256_set1_epi64x, _mm256_xor_si256,
};

// --------------------------------------------------------------------------------------
// Supported raw wire versions; standalone container tag 5 is separate.
// --------------------------------------------------------------------------------------
const PGM_FORMAT_V2: u8 = 0x02;
/// v3 adds the fitted key count separately from the stored keys, so a
/// locator-only index (keys released, see `take_keys`) round-trips.
const PGM_FORMAT_V3: u8 = 0x03;
/// v4 drops the per-segment 64-bit filters (never consulted on any lookup path).
const PGM_FORMAT_V4: u8 = 0x04;
/// v6 stores a relative f64 slope instead of an absolute f32 slope/intercept.
/// Tag 5 is reserved for the standalone checksummed container.
const PGM_FORMAT_V6: u8 = 0x06;

/// Segment built in-memory (used during PGM construction only).
/// Both the key and position origins are kept as integers until prediction.
#[derive(Debug, Clone)]
struct SegmentBuild {
    slope: f64,
    min_key: u64,
    max_key: u64,
    start: usize,
    end: usize,
}

/// SoA layout for cache-friendly segment access.
///
/// `max_errors_u8` is the primary epsilon store; `0xFF` is a sentinel meaning
/// "see overflow_errors". `overflow_errors` is a sorted `Vec<(segment_idx, real_error)>`
/// touched only on the rare cold path. With ε ≤ 254 (the typical configuration)
/// the overflow vec is empty.
#[repr(align(64))]
#[derive(Debug, Clone, Default)]
pub(crate) struct SegmentsSoA {
    /// New relative models have no absolute slope/intercept arrays. Keeping the
    /// legacy representation permits lossless reads and resaves without keys.
    pub(crate) relative_slopes: Option<Vec<f64>>,
    pub(crate) slopes: Vec<f32>,
    pub(crate) intercepts: Vec<f32>,
    pub(crate) min_keys: Vec<u64>,
    pub(crate) max_keys: Vec<u64>,
    pub(crate) max_errors_u8: Vec<u8>, // 0xFF sentinel = overflow
    pub(crate) overflow_errors: Vec<(u32, u32)>, // (segment_idx, real_error), sorted
    pub(crate) starts: Vec<u32>,
    pub(crate) ends: Vec<u32>,
}

impl SegmentsSoA {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.max_keys.len()
    }

    #[inline]
    fn get_max_error(&self, seg_idx: usize) -> u32 {
        let e = self.max_errors_u8[seg_idx];
        if e != 0xFF {
            e as u32
        } else {
            // Cold path: rare overflow.
            self.overflow_errors
                .binary_search_by_key(&(seg_idx as u32), |&(s, _)| s)
                .map(|i| self.overflow_errors[i].1)
                .unwrap_or(255)
        }
    }
}

/// PGM Index for sorted integer keys. Lookup takes O(log S + log W) time for S
/// segments and a fitted error window of W keys, with a SIMD scan for small W.
#[derive(Debug, Clone)]
pub struct PgmIndex {
    /// Sorted keys (plain storage). Empty when `keys_ef` is `Some` — that's the
    /// compacted state.
    keys: Vec<u64>,
    /// Optional Elias-Fano compact representation of `keys`. When set, the
    /// plain `keys` Vec is empty and key reads go through `key_at`/
    /// a bounded-select reader. Search windows do not allocate or decompress
    /// unrelated keys.
    keys_ef: Option<crate::elias_fano::EliasFano>,
    /// Learned segments.
    segments: SegmentsSoA,
    /// Number of keys the segments were fitted on. Stays valid after
    /// [`PgmIndex::take_keys`] releases the key array.
    key_count: usize,
    /// Epsilon parameter for error tolerance (max across all segments).
    epsilon: u32,
    /// Optional Block-Bloom filter for fast negative lookup rejection.
    bloom: Option<BlockBloom>,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PgmError {
    /// The input contains the same key twice (the builder sorts, so order is
    /// never the problem).
    #[error("duplicate key in input")]
    DuplicateKeys,
    #[error("PGM supports at most u32::MAX keys")]
    TooManyKeys,
    #[deprecated(
        since = "0.7.0",
        note = "the builder sorts its input; duplicates raise DuplicateKeys"
    )]
    #[error("keys must be sorted and unique")]
    UnsortedKeys,
    #[error("empty key set")]
    EmptyKeys,
    #[error("key not found")]
    KeyNotFound,
    #[error("corrupt data")]
    CorruptData,
}

fn check_key_count(count: usize) -> Result<(), PgmError> {
    if count > u32::MAX as usize { Err(PgmError::TooManyKeys) } else { Ok(()) }
}

impl PgmIndex {
    /// Build a PGM index with the default builder settings, sorting unique keys
    /// before fitting models.
    pub fn build(keys: Vec<u64>, epsilon: u32) -> Result<Self, PgmError> {
        PgmBuilder::new().with_epsilon(epsilon).build(keys)
    }

    /// Give the sorted key array back to the caller and switch this index into
    /// *locator-only* mode: `segment_for_key` / `segment_bounds` keep working from
    /// the segment tables alone, while `index()`/`lower_bound()`/`range()` — which
    /// need the keys — report misses. Used by `HybridIndex`, which only needs the
    /// segment locator and keeps its own per-segment storage, saving 8 B/key.
    pub(crate) fn take_keys(&mut self) -> Vec<u64> {
        self.keys_ef = None;
        std::mem::take(&mut self.keys)
    }

    /// Whether the key array is present (see [`PgmIndex::take_keys`]).
    #[inline]
    fn has_keys(&self) -> bool {
        self.keys_ef.is_some() || self.keys.len() == self.key_count
    }

    /// Compact `self.keys` into an Elias-Fano representation, freeing the plain
    /// `Vec<u64>`. After this call:
    ///   - `keys()` returns an empty slice (the EF storage doesn't expose a
    ///     contiguous `&[u64]`).
    ///   - `key_at(i)`, `materialize_range`, and the internal `index/lower_bound`
    ///     paths continue to work transparently via the EF reader.
    ///
    /// Returns the number of bytes saved (positive = saved, negative = EF was
    /// larger and we did not compact).
    pub fn compact_keys(&mut self) -> isize {
        if self.keys.is_empty() || self.keys_ef.is_some() {
            return 0;
        }
        let original_bytes = self.keys.len() * 8;
        let ef = match crate::elias_fano::EliasFano::from_sorted(&self.keys) {
            Some(ef) => ef,
            None => return 0,
        };
        let ef_bytes = ef.memory_usage();
        if ef_bytes >= original_bytes {
            return -((ef_bytes as isize) - (original_bytes as isize));
        }
        self.keys.clear();
        self.keys.shrink_to_fit();
        self.keys_ef = Some(ef);
        (original_bytes as isize) - (ef_bytes as isize)
    }

    /// Get the i-th key (O(1) plain or bounded-select EF).
    #[inline]
    fn key_at(&self, i: usize) -> u64 {
        if let Some(ef) = &self.keys_ef { ef.get(i) } else { self.keys[i] }
    }

    /// Number of keys in the index.
    #[inline]
    fn keys_count(&self) -> usize {
        self.key_count
    }

    /// Linear-time piecewise-linear segmentation (slope-window / "swing" filter).
    ///
    /// Each segment is anchored at its first key `(x0, y0)`. Every further key
    /// `(x, y)` constrains the slope to `[(y - y0 - ε) / (x - x0), (y - y0 + ε) / (x - x0)]`;
    /// the running intersection of these intervals is the set of lines through
    /// the anchor that keep *every* key of the segment within ε. The segment is
    /// closed at the first key that empties the interval, and the middle of the
    /// interval is emitted as the slope. Adding a key is O(1), so the whole pass
    /// is O(N) — the previous implementation re-validated the entire segment on
    /// every extension, which was O(L²) per segment and took seconds for a
    /// single 200K-key segment.
    ///
    /// Anchoring the intercept costs a few percent more segments than the optimal
    /// convex-hull PLA. Stored predictions are measured afterwards, with a
    /// bounded-error fallback for floating-point rounding (`segments_to_soa`).
    fn build_segments_greedy(keys: &[u64], epsilon: u32) -> Vec<SegmentBuild> {
        let n = keys.len();
        if n == 0 {
            return Vec::new();
        }
        let local_eps = epsilon as f64;
        let mut segments: Vec<SegmentBuild> = Vec::with_capacity(n / 32 + 1);
        let mut start = 0usize;
        while start < n {
            let mut lo = f64::NEG_INFINITY;
            let mut hi = f64::INFINITY;
            let mut end = start + 1;
            while end < n {
                // Subtract before converting: neighbouring u64 keys remain
                // distinguishable even when their absolute values exceed 2^53.
                let dx = (keys[end] - keys[start]) as f64;
                let dy = (end - start) as f64;
                let new_lo = lo.max((dy - local_eps) / dx);
                let new_hi = hi.min((dy + local_eps) / dx);
                if new_lo > new_hi {
                    // This key does not fit; `lo..=hi` still describes the keys before it.
                    break;
                }
                lo = new_lo;
                hi = new_hi;
                end += 1;
            }
            let slope = if end == start + 1 {
                0.0
            } else {
                // Any slope in [lo, hi] is valid; the midpoint maximizes the margin
                // against floating-point rounding.
                0.5 * (lo + hi)
            };
            segments.push(SegmentBuild {
                slope,
                min_key: keys[start],
                max_key: keys[end - 1],
                start,
                end,
            });
            start = end;
        }
        segments
    }

    /// Parallel segment build.
    ///
    /// We split keys into chunks of `CHUNK` elements and run `build_segments_greedy`
    /// on each independently. The result is a vec-of-vecs concatenated in order;
    /// the boundary between two chunks ends at an inter-chunk position so no
    /// segment crosses chunks — that's the cost of cheap parallelism (it loses
    /// ~1 segment per chunk boundary, ≤ 0.1% overhead at CHUNK=128K).
    #[cfg(feature = "parallel")]
    fn build_segments_parallel(keys: &[u64], epsilon: u32) -> Vec<SegmentBuild> {
        use rayon::prelude::*;
        const CHUNK: usize = 128 * 1024;
        let n = keys.len();
        if n < CHUNK * 2 {
            return Self::build_segments_greedy(keys, epsilon);
        }
        let chunks: Vec<(usize, usize)> =
            (0..n).step_by(CHUNK).map(|s| (s, (s + CHUNK).min(n))).collect();

        let per_chunk: Vec<Vec<SegmentBuild>> = chunks
            .par_iter()
            .map(|&(c_start, c_end)| {
                // Build a sub-vector of keys for this chunk. We pass positions
                // as the global index so segments carry true positions.
                let sub = &keys[c_start..c_end];
                let mut local = Self::build_segments_greedy(sub, epsilon);
                // Translate local positions (0..sub.len()) to global (c_start..c_end).
                for seg in &mut local {
                    seg.start += c_start;
                    seg.end += c_start;
                }
                local
            })
            .collect();

        // Concatenate. Boundaries are clean by construction.
        let mut all = Vec::with_capacity(per_chunk.iter().map(|v| v.len()).sum());
        for mut v in per_chunk {
            all.append(&mut v);
        }
        all
    }

    /// Find the position of a key in O(log S + log W) time and O(1) extra space.
    pub fn index(&self, key: u64) -> Result<usize, PgmError> {
        // optional Block-Bloom short-circuit for negative lookups.
        if let Some(bf) = &self.bloom
            && !bf.contains_u64(key)
        {
            return Err(PgmError::KeyNotFound);
        }
        if !self.has_keys() {
            return Err(PgmError::KeyNotFound);
        }

        let segment_idx = find_segment_branchless(&self.segments.max_keys, key);
        if segment_idx >= self.segments.max_keys.len() {
            return Err(PgmError::KeyNotFound);
        }
        let min_key = self.segments.min_keys[segment_idx];
        let max_key = self.segments.max_keys[segment_idx];
        if key < min_key || key > max_key {
            return Err(PgmError::KeyNotFound);
        }
        let n = self.keys_count();
        let (search_start, search_end) = self.scan_window(segment_idx, key, n);

        // Fast path: keys live in plain `Vec<u64>`. A short window is scanned with
        // SIMD (a few vector compares beat any branching); a long one is bisected
        // — the keys are sorted, and 7 dependent L1 loads cost far less than
        // comparing 129 keys. Crossover measured at ~24 keys on both x86 and ARM.
        if self.keys_ef.is_none() {
            let window = search_end - search_start;
            if window <= SCAN_WINDOW {
                if let Some(pos) = find_in_range_simd(&self.keys, search_start, search_end, key) {
                    return Ok(pos);
                }
                return Err(PgmError::KeyNotFound);
            }
            return match bisect_window(&self.keys[search_start..search_end], key) {
                Some(local_pos) => Ok(search_start + local_pos),
                None => Err(PgmError::KeyNotFound),
            };
        }

        let window = search_end - search_start;
        if window <= SCAN_WINDOW {
            let mut buf = [0u64; SCAN_WINDOW];
            let count =
                self.keys_ef.as_ref().unwrap().materialize_into(search_start, &mut buf[..window]);
            return find_in_range_simd(&buf, 0, count, key)
                .map(|local| search_start + local)
                .ok_or(PgmError::KeyNotFound);
        }
        // A large error window is bisected in compressed storage. Bounded EF
        // select keeps this O(log W), with no W-sized temporary allocation.
        let pos = self.lower_bound_compressed(search_start, search_end, key);
        if pos < search_end && self.key_at(pos) == key {
            Ok(pos)
        } else {
            Err(PgmError::KeyNotFound)
        }
    }

    /// `[start, end)` of positions to scan for `key` in `segment_idx`. Clamped to
    /// `[0, n]` with `start ≤ end`, so even a segment with nonsense slope (corrupt
    /// input) yields an empty window instead of a panic.
    #[inline(always)]
    fn scan_window(&self, segment_idx: usize, key: u64, n: usize) -> (usize, usize) {
        let predicted_pos = predict_pos(&self.segments, segment_idx, key);
        let max_error = self.segments.get_max_error(segment_idx) as usize;
        let start = predicted_pos.saturating_sub(max_error).min(n);
        let end = predicted_pos.saturating_add(max_error).saturating_add(1).min(n);
        (start, end.max(start))
    }

    /// Positions of all keys in `[min_key, max_key]`, as the half-open range
    /// `lower_bound(min_key)..upper_bound(max_key)` — O(1) space however large the
    /// range is. Iterate it, take its `len()`, or slice a side array with it.
    pub fn range(&self, min_key: u64, max_key: u64) -> std::ops::Range<usize> {
        let start_pos = self.lower_bound(min_key);
        let end_pos = self.upper_bound(max_key);
        start_pos..end_pos.max(start_pos)
    }

    /// Find first position where key >= target.
    pub fn lower_bound(&self, target: u64) -> usize {
        let n = self.keys_count();
        if !self.has_keys() {
            return n;
        }
        let segment_idx = find_segment_branchless(&self.segments.max_keys, target);
        if segment_idx >= self.segments.max_keys.len() {
            return n;
        }
        let segment_start = self.segments.starts[segment_idx] as usize;
        let segment_end = self.segments.ends[segment_idx] as usize;
        // No line is fitted over gaps between segments. The next segment's first
        // position is the exact lower bound before (or at) its minimum key.
        if target <= self.segments.min_keys[segment_idx] {
            return segment_start;
        }
        let (search_start, search_end) = self.scan_window(segment_idx, target, n);
        let search_start = search_start.clamp(segment_start, segment_end);
        let search_end = search_end.clamp(search_start, segment_end);

        if self.keys_ef.is_none() {
            let window = &self.keys[search_start..search_end];
            if window.len() > SCAN_WINDOW {
                // Sorted window: bisect. `partition_point` is the first key ≥ target;
                // if none, the answer is the segment's (clamped) end.
                let p = window.partition_point(|&k| k < target);
                return if p < window.len() { search_start + p } else { search_end.min(n) };
            }
            if let Some(pos) = find_first_ge_simd(&self.keys, search_start, search_end, target) {
                return pos;
            }
            // Every key of the window is below target: the next position is the answer
            // when the window is clamped at n, otherwise the segment says target is past
            // its keys (then `search_end` is already n or the lookup misses).
            return search_end.min(n);
        }
        let window = search_end - search_start;
        if window <= SCAN_WINDOW {
            let mut buf = [0u64; SCAN_WINDOW];
            let count =
                self.keys_ef.as_ref().unwrap().materialize_into(search_start, &mut buf[..window]);
            return find_first_ge_simd(&buf, 0, count, target)
                .map(|local| search_start + local)
                .unwrap_or(search_end);
        }
        self.lower_bound_compressed(search_start, search_end, target)
    }

    #[inline]
    fn lower_bound_compressed(&self, mut start: usize, mut end: usize, target: u64) -> usize {
        while start < end {
            let middle = start + (end - start) / 2;
            if self.key_at(middle) < target {
                start = middle + 1;
            } else {
                end = middle;
            }
        }
        start
    }

    /// Find first position where key > target.
    pub fn upper_bound(&self, target: u64) -> usize {
        match target.checked_add(1) {
            Some(next) => self.lower_bound(next),
            None => self.keys_count(),
        }
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> PgmStats {
        let total_keys = self.keys_count();
        let total_segments = self.segments.max_keys.len();
        let avg_segment_size =
            if total_segments > 0 { total_keys as f64 / total_segments as f64 } else { 0.0 };

        let max_error =
            (0..total_segments).map(|i| self.segments.get_max_error(i)).max().unwrap_or(0);

        let bloom_mem = self.bloom.as_ref().map(|b| b.memory_usage()).unwrap_or(0);
        let ef_mem = self.keys_ef.as_ref().map(|e| e.memory_usage()).unwrap_or(0);
        let memory_usage = std::mem::size_of_val(&self.keys)
            + self.keys.len() * std::mem::size_of::<u64>()
            + ef_mem
            + std::mem::size_of_val(&self.segments)
            + self.segments.slopes.len() * std::mem::size_of::<f32>()
            + self.segments.intercepts.len() * std::mem::size_of::<f32>()
            + self.segments.relative_slopes.as_ref().map_or(0, |s| s.len() * 8)
            + self.segments.min_keys.len() * std::mem::size_of::<u64>()
            + self.segments.max_keys.len() * std::mem::size_of::<u64>()
            + self.segments.max_errors_u8.len() * std::mem::size_of::<u8>()
            + self.segments.overflow_errors.len() * std::mem::size_of::<(u32, u32)>()
            + self.segments.starts.len() * std::mem::size_of::<u32>()
            + self.segments.ends.len() * std::mem::size_of::<u32>()
            + bloom_mem;

        PgmStats {
            total_keys,
            total_segments,
            avg_segment_size,
            max_error,
            memory_usage,
            epsilon: self.epsilon,
        }
    }

    /// Serialize into a self-contained, checksummed container. New models use a
    /// relative v6 payload; loaded legacy models retain their absolute v4 layout.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PgmError> {
        let mut out = Vec::with_capacity(self.stats().memory_usage);
        self.write_to(&mut out);
        Ok(crate::wire::seal_tagged(crate::wire::PGM_CONTAINER_TAG, crate::wire::KIND_PGM, &out))
    }

    /// Read a standalone checksummed container or a raw v2/v3/v4/v6 payload.
    /// Trailing bytes are rejected; embedded payloads use the separate raw reader.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PgmError> {
        let body = if bytes.first() == Some(&crate::wire::PGM_CONTAINER_TAG) {
            crate::wire::unseal_tagged(bytes, crate::wire::PGM_CONTAINER_TAG, crate::wire::KIND_PGM)
                .ok_or(PgmError::CorruptData)?
        } else {
            bytes
        };
        let mut pos = 0usize;
        let index = Self::read_from(body, &mut pos)?;
        if pos != body.len() {
            return Err(PgmError::CorruptData);
        }
        Ok(index)
    }

    /// Wire format v4/v6:
    /// [u8 ver] [u32 epsilon] [u64 key_count] [u64 stored_keys] [keys×u64]
    /// [u64 seg_len]
    /// segments contiguous SoA: slopes[u32 f32][intercepts][min_keys u64][max_keys u64]
    ///                          [max_errors u8] [u64 overflow_count]
    ///                          [overflow entries (u32 seg_idx, u32 err)×N]
    ///                          [starts u32][ends u32]
    /// [u8 has_bloom] [bloom payload if present]
    /// v2 (single key count) and v3 (with a `[filters u64]` array before `starts`)
    /// are still read. v6 replaces slopes/intercepts (4 + 4 bytes per segment)
    /// with a relative f64 slope (8 bytes per segment).
    pub(crate) fn write_to(&self, out: &mut Vec<u8>) {
        out.push(if self.segments.relative_slopes.is_some() {
            PGM_FORMAT_V6
        } else {
            PGM_FORMAT_V4
        });
        write_u32(out, self.epsilon);
        write_u64(out, self.key_count as u64);
        // Plain or EF — emit the materialized u64 stream either way. EF reload
        // is a post-deserialization choice (call `compact_keys` if desired).
        // Locator-only indexes store no keys at all.
        let stored = if self.has_keys() { self.key_count } else { 0 };
        write_u64(out, stored as u64);
        if stored > 0 {
            if self.keys_ef.is_some() {
                for k in self.keys_iter() {
                    write_u64(out, k);
                }
            } else {
                crate::wire::extend_le(out, &self.keys);
            }
        }
        let s = &self.segments;
        write_u64(out, s.len() as u64);
        if let Some(slopes) = &s.relative_slopes {
            crate::wire::extend_le(out, slopes);
        } else {
            crate::wire::extend_le(out, &s.slopes);
            crate::wire::extend_le(out, &s.intercepts);
        }
        crate::wire::extend_le(out, &s.min_keys);
        crate::wire::extend_le(out, &s.max_keys);
        out.extend_from_slice(&s.max_errors_u8);
        write_u64(out, s.overflow_errors.len() as u64);
        for &(si, er) in &s.overflow_errors {
            write_u32(out, si);
            write_u32(out, er);
        }
        crate::wire::extend_le(out, &s.starts);
        crate::wire::extend_le(out, &s.ends);
        match &self.bloom {
            Some(bf) => {
                out.push(1);
                bf.write_to(out);
            }
            None => {
                out.push(0);
            }
        }
    }

    pub(crate) fn read_from(bytes: &[u8], pos: &mut usize) -> Result<Self, PgmError> {
        let mut cur = Cursor { buf: bytes, pos: *pos };
        let ver = cur.read_u8().ok_or(PgmError::CorruptData)?;
        if !matches!(ver, PGM_FORMAT_V2 | PGM_FORMAT_V3 | PGM_FORMAT_V4 | PGM_FORMAT_V6) {
            return Err(PgmError::CorruptData);
        }
        let epsilon = cur.read_u32().ok_or(PgmError::CorruptData)?;
        // Every count is capped by the bytes actually present before allocating.
        let remaining = |cur: &Cursor<'_>| cur.buf.len().saturating_sub(cur.pos);
        let key_count = usize::try_from(cur.read_u64().ok_or(PgmError::CorruptData)?)
            .map_err(|_| PgmError::CorruptData)?;
        let stored = if ver >= PGM_FORMAT_V3 {
            usize::try_from(cur.read_u64().ok_or(PgmError::CorruptData)?)
                .map_err(|_| PgmError::CorruptData)?
        } else {
            key_count
        };
        if stored != 0 && stored != key_count {
            return Err(PgmError::CorruptData);
        }
        let keys: Vec<u64> = cur.read_le(stored)?;
        let seg_len = usize::try_from(cur.read_u64().ok_or(PgmError::CorruptData)?)
            .map_err(|_| PgmError::CorruptData)?;
        // v4 removed one u64 filter per segment; retain the older payload bound
        // for v2/v3 without rejecting compact multi-segment v4 indexes.
        let segment_bytes = if ver < PGM_FORMAT_V4 { 41 } else { 33 };
        if seg_len > remaining(&cur) / segment_bytes {
            return Err(PgmError::CorruptData);
        }
        let (relative_slopes, slopes, intercepts) = if ver == PGM_FORMAT_V6 {
            (Some(cur.read_le::<f64>(seg_len)?), Vec::new(), Vec::new())
        } else {
            (None, cur.read_le::<f32>(seg_len)?, cur.read_le::<f32>(seg_len)?)
        };
        let min_keys: Vec<u64> = cur.read_le(seg_len)?;
        let max_keys: Vec<u64> = cur.read_le(seg_len)?;
        let max_errors_u8 = cur.read_bytes(seg_len).ok_or(PgmError::CorruptData)?.to_vec();
        let over_n = usize::try_from(cur.read_u64().ok_or(PgmError::CorruptData)?)
            .map_err(|_| PgmError::CorruptData)?;
        if over_n > remaining(&cur) / 8 {
            return Err(PgmError::CorruptData);
        }
        let mut overflow_errors = Vec::with_capacity(over_n);
        for _ in 0..over_n {
            let si = cur.read_u32().ok_or(PgmError::CorruptData)?;
            let er = cur.read_u32().ok_or(PgmError::CorruptData)?;
            overflow_errors.push((si, er));
        }
        if ver < PGM_FORMAT_V4 {
            // Skip the legacy per-segment filter words.
            let _: Vec<u64> = cur.read_le(seg_len)?;
        }
        let starts: Vec<u32> = cur.read_le(seg_len)?;
        let ends: Vec<u32> = cur.read_le(seg_len)?;
        let has_bloom = cur.read_u8().ok_or(PgmError::CorruptData)?;
        if has_bloom > 1 {
            return Err(PgmError::CorruptData);
        }
        *pos = cur.pos;
        let bloom = if has_bloom == 1 {
            let mut bp = *pos;
            let bf = BlockBloom::read_from(bytes, &mut bp).ok_or(PgmError::CorruptData)?;
            *pos = bp;
            Some(bf)
        } else {
            None
        };
        let idx = PgmIndex {
            key_count,
            keys,
            keys_ef: None,
            segments: SegmentsSoA {
                relative_slopes,
                slopes,
                intercepts,
                min_keys,
                max_keys,
                max_errors_u8,
                overflow_errors,
                starts,
                ends,
            },
            epsilon,
            bloom,
        };
        if !idx.validate() {
            return Err(PgmError::CorruptData);
        }
        Ok(idx)
    }

    /// Validate segment geometry and error metadata in linear time. When stored
    /// keys are present, verify their bounds and the fitted line's error as well.
    /// Locator-only payloads still carry complete, ordered segment geometry.
    pub(crate) fn validate(&self) -> bool {
        let s = &self.segments;
        let m = s.max_keys.len();
        let valid_numeric_shape = match &s.relative_slopes {
            Some(slopes) => slopes.len() == m && s.slopes.is_empty() && s.intercepts.is_empty(),
            None => s.slopes.len() == m && s.intercepts.len() == m,
        };
        if !valid_numeric_shape
            || s.min_keys.len() != m
            || s.max_errors_u8.len() != m
            || s.starts.len() != m
            || s.ends.len() != m
        {
            return false;
        }
        let n = self.keys_count();
        if n == 0 {
            return m == 0 && self.keys.is_empty() && s.overflow_errors.is_empty();
        }
        if m == 0 || check_key_count(n).is_err() {
            return false;
        }
        if !self.keys.is_empty() && self.keys.len() != n {
            return false;
        }
        if self.keys.windows(2).any(|w| w[0] >= w[1]) {
            return false;
        }
        if s.overflow_errors.windows(2).any(|w| w[0].0 >= w[1].0)
            || s.overflow_errors.iter().any(|&(si, _)| si as usize >= m)
        {
            return false;
        }
        let mut overflow = s.overflow_errors.iter().peekable();
        let mut prev_end = 0u32;
        for i in 0..m {
            let (st, en) = (s.starts[i], s.ends[i]);
            if st != prev_end || en <= st || en as usize > n {
                return false;
            }
            if s.min_keys[i] > s.max_keys[i] || (i > 0 && s.min_keys[i] <= s.max_keys[i - 1]) {
                return false;
            }
            let valid_line = if let Some(slopes) = &s.relative_slopes {
                slopes[i].is_finite() && slopes[i] >= 0.0
            } else {
                s.slopes[i].is_finite() && s.slopes[i] >= 0.0 && s.intercepts[i].is_finite()
            };
            if !valid_line {
                return false;
            }
            let max_error = if s.max_errors_u8[i] == 0xFF {
                match overflow.next() {
                    Some(&(segment, error)) if segment as usize == i && error >= 255 => error,
                    _ => return false,
                }
            } else {
                if overflow.peek().is_some_and(|&&(segment, _)| segment as usize == i) {
                    return false;
                }
                s.max_errors_u8[i] as u32
            };
            // Legacy absolute models may have a much larger quantization error
            // than their requested epsilon. Only relative v6 promises this cap.
            if s.relative_slopes.is_some() && max_error > self.epsilon.saturating_add(1) {
                return false;
            }
            if !self.keys.is_empty() {
                let start = st as usize;
                let end = en as usize;
                if s.min_keys[i] != self.keys[start] || s.max_keys[i] != self.keys[end - 1] {
                    return false;
                }
                for (local, &key) in self.keys[start..end].iter().enumerate() {
                    let predicted = predict_value(s, i, key);
                    let valid_error = if s.relative_slopes.is_some() {
                        (predicted as usize).abs_diff(start + local) <= max_error as usize
                    } else {
                        (predicted - (start + local) as f64).abs().ceil() <= max_error as f64
                    };
                    if !predicted.is_finite()
                        || !valid_error
                        || self.bloom.as_ref().is_some_and(|bloom| !bloom.contains_u64(key))
                    {
                        return false;
                    }
                }
            }
            prev_end = en;
        }
        if prev_end as usize != n {
            return false;
        }
        overflow.next().is_none()
    }

    /// Store one relative f64 slope in the same eight bytes previously occupied
    /// by an absolute f32 slope/intercept pair. Measure the exact integer
    /// prediction used by lookup. Rare rounding failures fall back to singleton
    /// segments, so the epsilon + 1 cap holds without an unbounded refit loop.
    /// The fitting and verification passes visit each key O(1) times overall.
    fn segments_to_soa(keys: &[u64], segments: Vec<SegmentBuild>, epsilon: u32) -> SegmentsSoA {
        let n = segments.len();
        let mut slopes = Vec::with_capacity(n);
        let mut min_keys = Vec::with_capacity(n);
        let mut max_keys = Vec::with_capacity(n);
        let mut max_errors_u8 = Vec::with_capacity(n);
        let mut overflow_errors: Vec<(u32, u32)> = Vec::new();
        let mut starts = Vec::with_capacity(n);
        let mut ends = Vec::with_capacity(n);

        let mut emit =
            |slope: f64, min_key: u64, max_key: u64, start: usize, end: usize, err: u32| {
                let seg_idx = slopes.len();
                slopes.push(slope);
                min_keys.push(min_key);
                max_keys.push(max_key);
                if err <= 254 {
                    max_errors_u8.push(err as u8);
                } else {
                    max_errors_u8.push(0xFFu8);
                    overflow_errors.push((seg_idx as u32, err));
                }
                starts.push(start as u32);
                ends.push(end as u32);
            };
        let allowed = epsilon.saturating_add(1) as usize;
        for seg in segments {
            let mut err = 0usize;
            let mut fits = true;
            for (local, &key) in keys[seg.start..seg.end].iter().enumerate() {
                let prediction = relative_prediction(seg.slope, seg.min_key, seg.start, key);
                if !prediction.is_finite() || !seg.slope.is_finite() || seg.slope < 0.0 {
                    fits = false;
                    break;
                }
                err = err.max((prediction as usize).abs_diff(seg.start + local));
                if err > allowed {
                    fits = false;
                    break;
                }
            }
            if fits {
                emit(seg.slope, seg.min_key, seg.max_key, seg.start, seg.end, err as u32);
            } else {
                for (local, &key) in keys[seg.start..seg.end].iter().enumerate() {
                    let start = seg.start + local;
                    emit(0.0, key, key, start, start + 1, 0);
                }
            }
        }
        // overflow_errors is built in segment-index order; binary_search_by_key works.
        SegmentsSoA {
            relative_slopes: Some(slopes),
            slopes: Vec::new(),
            intercepts: Vec::new(),
            min_keys,
            max_keys,
            max_errors_u8,
            overflow_errors,
            starts,
            ends,
        }
    }

    pub(crate) fn segment_for_key(&self, key: u64) -> Option<usize> {
        let idx = find_segment_branchless(&self.segments.max_keys, key);
        if idx >= self.segments.max_keys.len() {
            return None;
        }
        let min_key = self.segments.min_keys[idx];
        let max_key = self.segments.max_keys[idx];
        if key < min_key || key > max_key {
            return None;
        }
        Some(idx)
    }

    pub(crate) fn segment_bounds(&self, idx: usize) -> Option<(u32, u32)> {
        if idx >= self.segments.max_keys.len() {
            return None;
        }
        Some((self.segments.starts[idx], self.segments.ends[idx]))
    }

    /// Raw pointer to the segments `max_keys` array — exposed for prefetch
    /// chains in higher-level engines (e.g. HybridIndex batch lookup).
    pub(crate) fn max_keys_ptr(&self) -> *const u64 {
        self.segments.max_keys.as_ptr()
    }

    /// Number of segments — companion to `max_keys_ptr`.
    pub fn num_segments(&self) -> usize {
        self.segments.max_keys.len()
    }

    /// Iterate keys in order. Works whether plain or EF-backed.
    pub(crate) fn keys_iter(&self) -> KeysIter<'_> {
        KeysIter { pgm: self, i: 0, n: self.keys_count() }
    }

    /// Whether this index has the optional Bloom filter for negative-lookup fast path.
    pub fn has_bloom(&self) -> bool {
        self.bloom.is_some()
    }

    /// Write PGM into a `MmapIndexWriter` as separate per-field sections.
    /// This is the input side of zero-copy mmap; the reader (`read_from_sections`)
    /// can pull each field straight from the mapped byte buffer without parsing.
    pub fn write_to_sections(&self, writer: &mut crate::mmap_index::MmapIndexWriter) {
        use crate::mmap_index::SectionKind;
        let s = &self.segments;
        let keys = if self.keys_ef.is_some() {
            bytes_of_u64(&self.keys_iter().collect::<Vec<_>>())
        } else {
            bytes_of_u64(&self.keys)
        };
        writer.add_section(SectionKind::PgmKeys, keys);
        writer.add_section(
            SectionKind::PgmSlopes,
            match &s.relative_slopes {
                Some(slopes) => bytes_of_f64(slopes),
                None => bytes_of_f32(&s.slopes),
            },
        );
        writer.add_section(SectionKind::PgmIntercepts, bytes_of_f32(&s.intercepts));
        writer.add_section(SectionKind::PgmMinKeys, bytes_of_u64(&s.min_keys));
        writer.add_section(SectionKind::PgmMaxKeys, bytes_of_u64(&s.max_keys));
        writer.add_section(SectionKind::PgmMaxErrors, s.max_errors_u8.clone());
        let overflow_bytes = bytes_of_u32_pairs(&s.overflow_errors);
        writer.add_section(SectionKind::PgmOverflowErrors, overflow_bytes);
        writer.add_section(SectionKind::PgmStarts, bytes_of_u32(&s.starts));
        writer.add_section(SectionKind::PgmEnds, bytes_of_u32(&s.ends));
        // PgmMeta: [u32 epsilon][u8 has_bloom][u8 relative][padding..].
        // Legacy writers set all three padding bytes to zero.
        let mut meta = Vec::with_capacity(8);
        meta.extend_from_slice(&self.epsilon.to_le_bytes());
        meta.push(if self.bloom.is_some() { 1 } else { 0 });
        meta.push(u8::from(s.relative_slopes.is_some()));
        // Pad to 8 bytes for alignment.
        while meta.len() < 8 {
            meta.push(0);
        }
        writer.add_section(SectionKind::PgmMeta, meta);
        if let Some(bf) = &self.bloom {
            let mut bf_bytes = Vec::new();
            bf.write_to(&mut bf_bytes);
            writer.add_section(SectionKind::PgmBloom, bf_bytes);
        }
    }

    /// Read PGM back from a parsed `Header`. We still allocate `Vec`s for each
    /// field (true zero-copy would need lifetime-parameterized `PgmIndex<'a>`),
    /// but parsing is now a simple memcpy per section instead of length-prefix
    /// decoding — roughly 10× faster on a 100M-key index.
    pub fn read_from_sections(
        mmap: &crate::mmap_index::MmapIndex,
        header: &crate::mmap_index::Header,
    ) -> Option<Self> {
        use crate::mmap_index::SectionKind;
        let meta = mmap.section(header, SectionKind::PgmMeta)?;
        if !matches!(meta.len(), 5 | 8)
            || meta[4] > 1
            || (meta.len() == 8 && (meta[5] > 1 || meta[6] != 0 || meta[7] != 0))
        {
            return None;
        }
        let relative = meta.len() == 8 && meta[5] == 1;
        let keys = u64_vec_from_section(mmap.section(header, SectionKind::PgmKeys)?)?;
        let slope_bytes = mmap.section(header, SectionKind::PgmSlopes)?;
        let (relative_slopes, slopes) = if relative {
            (Some(f64_vec_from_section(slope_bytes)?), Vec::new())
        } else {
            (None, f32_vec_from_section(slope_bytes)?)
        };
        let intercepts = f32_vec_from_section(mmap.section(header, SectionKind::PgmIntercepts)?)?;
        let min_keys = u64_vec_from_section(mmap.section(header, SectionKind::PgmMinKeys)?)?;
        let max_keys = u64_vec_from_section(mmap.section(header, SectionKind::PgmMaxKeys)?)?;
        let max_errors_u8 = mmap.section(header, SectionKind::PgmMaxErrors)?.to_vec();
        let overflow_errors =
            u32_pairs_from_section(mmap.section(header, SectionKind::PgmOverflowErrors)?)?;
        let starts = u32_vec_from_section(mmap.section(header, SectionKind::PgmStarts)?)?;
        let ends = u32_vec_from_section(mmap.section(header, SectionKind::PgmEnds)?)?;
        let epsilon = u32::from_le_bytes([meta[0], meta[1], meta[2], meta[3]]);
        let has_bloom = meta[4] == 1;
        let bloom = if has_bloom {
            let bf_bytes = mmap.section(header, SectionKind::PgmBloom)?;
            let mut pos = 0usize;
            let bf = crate::block_bloom::BlockBloom::read_from(bf_bytes, &mut pos)?;
            if pos != bf_bytes.len() {
                return None;
            }
            Some(bf)
        } else {
            None
        };
        let idx = PgmIndex {
            key_count: keys.len(),
            keys,
            keys_ef: None,
            segments: SegmentsSoA {
                relative_slopes,
                slopes,
                intercepts,
                min_keys,
                max_keys,
                max_errors_u8,
                overflow_errors,
                starts,
                ends,
            },
            epsilon,
            bloom,
        };
        idx.validate().then_some(idx)
    }
}

/// Iterator over keys (either plain or EF-backed).
pub(crate) struct KeysIter<'a> {
    pgm: &'a PgmIndex,
    i: usize,
    n: usize,
}

impl<'a> Iterator for KeysIter<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.n {
            return None;
        }
        let v = self.pgm.key_at(self.i);
        self.i += 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.n - self.i;
        (rem, Some(rem))
    }
}

impl<'a> ExactSizeIterator for KeysIter<'a> {}

// --------------------------------------------------------------------------------------
// Section serialization helpers.
// --------------------------------------------------------------------------------------

fn bytes_of_u64(v: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    crate::wire::extend_le(&mut out, v);
    out
}

fn bytes_of_u32(v: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    crate::wire::extend_le(&mut out, v);
    out
}

fn bytes_of_f32(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    crate::wire::extend_le(&mut out, v);
    out
}

fn bytes_of_f64(v: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    crate::wire::extend_le(&mut out, v);
    out
}

fn bytes_of_u32_pairs(v: &[(u32, u32)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for &(a, b) in v {
        out.extend_from_slice(&a.to_le_bytes());
        out.extend_from_slice(&b.to_le_bytes());
    }
    out
}

fn u64_vec_from_section(bytes: &[u8]) -> Option<Vec<u64>> {
    if !bytes.len().is_multiple_of(8) {
        return None;
    }
    crate::wire::read_le(bytes, bytes.len() / 8)
}

fn u32_vec_from_section(bytes: &[u8]) -> Option<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    crate::wire::read_le(bytes, bytes.len() / 4)
}

fn f32_vec_from_section(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    crate::wire::read_le(bytes, bytes.len() / 4)
}

fn f64_vec_from_section(bytes: &[u8]) -> Option<Vec<f64>> {
    if !bytes.len().is_multiple_of(8) {
        return None;
    }
    crate::wire::read_le(bytes, bytes.len() / 8)
}

fn u32_pairs_from_section(bytes: &[u8]) -> Option<Vec<(u32, u32)>> {
    if !bytes.len().is_multiple_of(8) {
        return None;
    }
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.as_chunks::<8>().0 {
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        a.copy_from_slice(&chunk[..4]);
        b.copy_from_slice(&chunk[4..]);
        out.push((u32::from_le_bytes(a), u32::from_le_bytes(b)));
    }
    Some(out)
}

// --------------------------------------------------------------------------------------
// SIMD branchless local search. The old code had a scalar `for lane in 0..4`
// inside the SIMD loop that triggered branch mispredicts on each match. We replace
// it with `_mm256_movemask_epi8 + tzcnt`, which gives the matched lane in 2 cycles
// and zero branches.
// --------------------------------------------------------------------------------------

/// Integer key subtraction preserves small local gaps at large absolute keys.
#[inline]
fn relative_prediction(slope: f64, min_key: u64, start: usize, key: u64) -> f64 {
    slope.mul_add(key.saturating_sub(min_key) as f64, start as f64)
}

#[inline]
fn predict_value(segments: &SegmentsSoA, idx: usize, key: u64) -> f64 {
    if let Some(slopes) = &segments.relative_slopes {
        relative_prediction(slopes[idx], segments.min_keys[idx], segments.starts[idx] as usize, key)
    } else {
        // Legacy absolute models must retain their original arithmetic on load.
        (segments.slopes[idx] as f64).mul_add(key as f64, segments.intercepts[idx] as f64)
    }
}

/// The error metadata bounds this exact truncated prediction at every fitted key.
#[inline]
fn predict_pos(segments: &SegmentsSoA, idx: usize, key: u64) -> usize {
    let prediction = predict_value(segments, idx, key);
    if prediction <= 0.0 { 0 } else { prediction as usize }
}

/// Largest scan window searched linearly (SIMD); longer windows are bisected.
const SCAN_WINDOW: usize = 24;

/// Branchless binary search for `target` in a sorted window; returns its offset.
#[inline]
fn bisect_window(w: &[u64], target: u64) -> Option<usize> {
    let mut base = 0usize;
    let mut len = w.len();
    while len > 1 {
        let half = len / 2;
        // SAFETY: `base + half < w.len()` for `len ≤ w.len() - base`.
        let m = unsafe { *w.get_unchecked(base + half) };
        base += if m <= target { half } else { 0 };
        len -= half;
    }
    (len == 1 && w[base] == target).then_some(base)
}

#[inline]
fn find_in_range_simd(keys: &[u64], start: usize, end: usize, target: u64) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            return find_in_range_avx2(keys, start, end, target);
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        if is_aarch64_feature_detected!("neon") {
            return find_in_range_neon(keys, start, end, target);
        }
    }

    let mut i = start;
    while i < end {
        if keys[i] == target {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Branchless AVX2 equality search. `_mm256_cmpeq_epi64` sets 8 bytes per matched
/// lane; `_mm256_movemask_epi8` reduces that to a u32; `trailing_zeros()/8` gives
/// the matched lane in zero branches. We only branch when `mask != 0` (rare-ish:
/// happens at most once per segment lookup, so well-predicted).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn find_in_range_avx2(keys: &[u64], start: usize, end: usize, target: u64) -> Option<usize> {
    let mut i = start;
    let target_vec = _mm256_set1_epi64x(target as i64);
    while i + 4 <= end {
        let ptr = keys.as_ptr().add(i) as *const __m256i;
        let chunk = _mm256_loadu_si256(ptr);
        let eq = _mm256_cmpeq_epi64(chunk, target_vec);
        let mask = _mm256_movemask_epi8(eq) as u32;
        if mask != 0 {
            // Each u64 lane contributes 8 bytes; tzcnt/8 = lane.
            let lane = (mask.trailing_zeros() / 8) as usize;
            return Some(i + lane);
        }
        i += 4;
    }
    while i < end {
        if *keys.get_unchecked(i) == target {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Branchless NEON equality search. NEON has no `movemask`, but we can OR the two
/// lanes of the compare result and zero-test in one shot.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn find_in_range_neon(keys: &[u64], start: usize, end: usize, target: u64) -> Option<usize> {
    let mut i = start;
    let target_vec = vdupq_n_u64(target);
    while i + 2 <= end {
        let ptr = keys.as_ptr().add(i);
        let chunk = vld1q_u64(ptr);
        let eq = vceqq_u64(chunk, target_vec);
        let lo = vgetq_lane_u64(eq, 0);
        let hi = vgetq_lane_u64(eq, 1);
        if lo != 0 {
            return Some(i);
        }
        if hi != 0 {
            return Some(i + 1);
        }
        i += 2;
    }
    while i < end {
        if *keys.get_unchecked(i) == target {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[inline]
fn find_first_ge_simd(keys: &[u64], start: usize, end: usize, target: u64) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            return find_first_ge_avx2(keys, start, end, target);
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        if is_aarch64_feature_detected!("neon") {
            return find_first_ge_neon(keys, start, end, target);
        }
    }

    let mut i = start;
    while i < end {
        if keys[i] >= target {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn find_first_ge_avx2(keys: &[u64], start: usize, end: usize, target: u64) -> Option<usize> {
    let mut i = start;
    let sign = _mm256_set1_epi64x(i64::MIN);
    let target_vec = _mm256_set1_epi64x(target as i64);
    let target_u = _mm256_xor_si256(target_vec, sign);
    while i + 4 <= end {
        let ptr = keys.as_ptr().add(i) as *const __m256i;
        let chunk = _mm256_loadu_si256(ptr);
        let chunk_u = _mm256_xor_si256(chunk, sign);
        let gt = _mm256_cmpgt_epi64(chunk_u, target_u);
        let eq = _mm256_cmpeq_epi64(chunk_u, target_u);
        let ge = _mm256_or_si256(gt, eq);
        let mask = _mm256_movemask_epi8(ge) as u32;
        if mask != 0 {
            let lane = (mask.trailing_zeros() / 8) as usize;
            return Some(i + lane);
        }
        i += 4;
    }
    while i < end {
        if *keys.get_unchecked(i) >= target {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn find_first_ge_neon(keys: &[u64], start: usize, end: usize, target: u64) -> Option<usize> {
    let mut i = start;
    let target_vec = vdupq_n_u64(target);
    while i + 2 <= end {
        let ptr = keys.as_ptr().add(i);
        let chunk = vld1q_u64(ptr);
        let ge = vcgeq_u64(chunk, target_vec);
        let lo = vgetq_lane_u64(ge, 0);
        let hi = vgetq_lane_u64(ge, 1);
        if lo != 0 {
            return Some(i);
        }
        if hi != 0 {
            return Some(i + 1);
        }
        i += 2;
    }
    while i < end {
        if *keys.get_unchecked(i) >= target {
            return Some(i);
        }
        i += 1;
    }
    None
}

// --------------------------------------------------------------------------------------
// Branchless segment lookup. Replaces the old "implicit-heap with leading_zeros
// step" binary search with a branchless cmov binary search that issues prefetches for
// both potential next-mid positions on each iteration. On 1.5M-segment indexes this
// turns the segment lookup from ~21 hard cache misses into ~21 prefetched misses,
// halving the effective latency.
// --------------------------------------------------------------------------------------

#[inline]
fn find_segment_branchless(max_keys: &[u64], key: u64) -> usize {
    let n = max_keys.len();
    if n == 0 {
        return 0;
    }
    let mut base = 0usize;
    let mut len = n;
    let ptr = max_keys.as_ptr();
    while len > 1 {
        let half = len / 2;
        let mid = base + half;
        // Prefetch the two potential next-mid positions. The CPU will load both
        // cache lines while we compute the comparison; only the chosen one is
        // actually used, the other is wasted bandwidth — but at large N the
        // win from hiding the next miss vastly outweighs the wasted prefetch.
        if half > 4 {
            let nq = half / 2;
            // SAFETY: both offsets are below `n`; a prefetch is a hint.
            prefetch_read(unsafe { ptr.add(base + nq) });
            prefetch_read(unsafe { ptr.add(mid + nq) });
        }
        let m = unsafe { *ptr.add(mid) };
        // Branchless update:
        //   if m < key: base = mid + 1; len = len - half - 1
        //   else      : len = half
        // We compute both branches and select via integer-arithmetic (no cmov
        // needed — LLVM lowers the multiply to cmov on x86_64).
        let less = (m < key) as usize;
        let new_base = base + less * (mid + 1 - base);
        let new_len = if less != 0 { len - half - 1 } else { half };
        base = new_base;
        len = new_len;
    }
    if base < n && unsafe { *ptr.add(base) } < key { base + 1 } else { base }
}

// --------------------------------------------------------------------------------------
// Serialization scratch.
// --------------------------------------------------------------------------------------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn read_bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(len)?;
        let bytes = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(bytes)
    }
    fn read_u8(&mut self) -> Option<u8> {
        Some(self.read_bytes(1)?[0])
    }
    fn read_u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.read_bytes(4)?.try_into().ok()?))
    }
    fn read_u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.read_bytes(8)?.try_into().ok()?))
    }
    fn read_le<T: crate::wire::LeNum + Default>(
        &mut self,
        count: usize,
    ) -> Result<Vec<T>, PgmError> {
        crate::wire::read_le_at(self.buf, &mut self.pos, count).ok_or(PgmError::CorruptData)
    }
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

// --------------------------------------------------------------------------------------
// Public stats / builder API.
// --------------------------------------------------------------------------------------

/// Statistics about PGM index.
#[derive(Debug)]
pub struct PgmStats {
    pub total_keys: usize,
    pub total_segments: usize,
    pub avg_segment_size: f64,
    pub max_error: u32,
    pub memory_usage: usize,
    pub epsilon: u32,
}

impl PgmStats {
    pub fn print_summary(&self) {
        println!("PGM Index Statistics:");
        println!("  Total keys: {}", self.total_keys);
        println!("  Segments: {}", self.total_segments);
        println!("  Avg segment size: {:.1}", self.avg_segment_size);
        println!("  Max error: {}", self.max_error);
        println!("  Memory usage: {:.2} MB", self.memory_usage as f64 / 1_048_576.0);
        println!("  Epsilon: {}", self.epsilon);
        println!(
            "  Compression ratio: {:.2}x",
            self.total_keys as f64 / self.total_segments as f64
        );
    }
}

/// Builder for PGM Index with configuration.
///
/// Supports auto-tuning: set `target_lookup_ns(N)` and the builder will
/// sweep ε values, building a small calibration sub-index against a random
/// 16K-key sample, and pick the ε whose simulated cost is closest to N ns.
pub struct PgmBuilder {
    epsilon: u32,
    enable_bloom: bool,
    enable_parallel: bool,
    enable_elias_fano: bool,
    target_lookup_ns: Option<u32>,
}

impl PgmBuilder {
    pub fn new() -> Self {
        Self {
            epsilon: 64,
            enable_bloom: false,
            enable_parallel: cfg!(feature = "parallel"),
            enable_elias_fano: false,
            target_lookup_ns: None,
        }
    }

    /// Error tolerance for fitting each segment. The measured integer prediction
    /// error is at most `epsilon.saturating_add(1)`; the extra position accounts
    /// for floating-point rounding and truncation. `stats().max_error` reports
    /// the actual bound used by lookup, uniformly across all keys and offsets.
    pub fn with_epsilon(mut self, epsilon: u32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Enable Block-Bloom fast-negative-lookup filter. Costs ~10–12 bits per
    /// key but rejects 99%+ of misses in O(1) without touching segments.
    pub fn with_bloom_filter(mut self, enabled: bool) -> Self {
        self.enable_bloom = enabled;
        self
    }

    /// Enable parallel segment build. Default true if `parallel` feature
    /// is enabled at compile time.
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.enable_parallel = enabled;
        self
    }

    /// Compact key storage via Elias-Fano after build when it saves memory.
    /// Small search windows use a stack buffer; larger windows use O(log W)
    /// bounded compressed reads without materializing the complete window.
    pub fn with_elias_fano(mut self, enabled: bool) -> Self {
        self.enable_elias_fano = enabled;
        self
    }

    /// Auto-tune ε to target a specific per-lookup latency in nanoseconds.
    /// Overrides `with_epsilon` if set. Calibration uses ~16K-key sample and a
    /// crude latency model: cost(ε) = base_seg_lookup + α·log₂(N/ε) + β·ε,
    /// fit to a couple of measured points.
    pub fn with_target_lookup_ns(mut self, ns: u32) -> Self {
        self.target_lookup_ns = Some(ns);
        self
    }

    /// Build a PGM index over `keys`. Empty input is permitted and yields an
    /// always-miss instance.
    pub fn build(self, keys: Vec<u64>) -> Result<PgmIndex, PgmError> {
        crate::build_pool::run(self.enable_parallel, || self.build_inner(keys))
    }

    fn build_inner(self, keys: Vec<u64>) -> Result<PgmIndex, PgmError> {
        check_key_count(keys.len())?;
        let mut sorted = keys;
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] >= w[1]) {
            return Err(PgmError::DuplicateKeys);
        }

        if sorted.is_empty() {
            return Ok(PgmIndex {
                key_count: 0,
                keys: Vec::new(),
                keys_ef: None,
                segments: SegmentsSoA {
                    relative_slopes: Some(Vec::new()),
                    ..SegmentsSoA::default()
                },
                epsilon: self.epsilon,
                bloom: None,
            });
        }

        let epsilon = match self.target_lookup_ns {
            Some(target) => Self::auto_tune_epsilon(&sorted, target).unwrap_or(self.epsilon),
            None => self.epsilon,
        };

        let raw_segments = {
            #[cfg(feature = "parallel")]
            {
                if self.enable_parallel && crate::build_pool::parallel_enabled() {
                    PgmIndex::build_segments_parallel(&sorted, epsilon)
                } else {
                    PgmIndex::build_segments_greedy(&sorted, epsilon)
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                PgmIndex::build_segments_greedy(&sorted, epsilon)
            }
        };
        let segments = PgmIndex::segments_to_soa(&sorted, raw_segments, epsilon);

        let bloom = if self.enable_bloom {
            Some(BlockBloom::build_from_u64(&sorted, 0xC1B5_4A32_D192_ED03))
        } else {
            None
        };

        let mut idx = PgmIndex {
            key_count: sorted.len(),
            keys: sorted,
            keys_ef: None,
            segments,
            epsilon,
            bloom,
        };
        if self.enable_elias_fano {
            let _ = idx.compact_keys();
        }
        Ok(idx)
    }

    /// Sweep ε ∈ {16, 32, 64, 128, 256} on a 16K-key sample, measure resulting
    /// segment count and simulate cost. Cost model:
    ///   cost(ε) ≈ C_seg_lookup · log₂(N_segs) + C_simd · (2ε + 1) / 8
    /// where C_seg_lookup ≈ 1 ns per binary step (with prefetch hide),
    ///       C_simd       ≈ 0.5 ns per 8 u64 compared.
    /// Pick the smallest ε whose modeled cost ≤ target_ns; if none, return the
    /// best we found.
    fn auto_tune_epsilon(keys: &[u64], target_ns: u32) -> Option<u32> {
        if keys.len() < 1024 {
            // Too few keys for sweep to mean anything — just return None and let
            // caller use the configured default.
            return None;
        }
        const SAMPLE: usize = 16 * 1024;
        let stride = (keys.len() / SAMPLE).max(1);
        let sample: Vec<u64> = keys.iter().copied().step_by(stride).take(SAMPLE).collect();
        let n_full = keys.len() as f64;
        let n_sample = sample.len() as f64;
        let mut best: Option<(u32, f64)> = None;
        for &eps in &[16u32, 32, 48, 64, 96, 128, 192, 256] {
            let segs = PgmIndex::build_segments_greedy(&sample, eps);
            // Scale segment count from sample to full N: same density.
            let scaled_segs = segs.len() as f64 * (n_full / n_sample);
            let seg_lookup_steps = (scaled_segs.max(2.0)).log2();
            let cost_ns = 1.0 * seg_lookup_steps + 0.5 * ((2 * eps + 1) as f64 / 8.0);
            let diff = (cost_ns - target_ns as f64).abs();
            if best.map(|(_, b)| diff < b).unwrap_or(true) {
                best = Some((eps, diff));
            }
            if cost_ns <= target_ns as f64 {
                // Greedy: smaller ε first. But our list is ascending — the first
                // one satisfying the budget is the tightest reasonable choice.
                return Some(eps);
            }
        }
        best.map(|(e, _)| e)
    }
}

impl Default for PgmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn position_count_bound_does_not_truncate() {
        assert!(super::check_key_count(0).is_ok());
        assert!(super::check_key_count(u32::MAX as usize).is_ok());
        if let Ok(too_many) = usize::try_from(u32::MAX as u64 + 1) {
            assert!(matches!(super::check_key_count(too_many), Err(super::PgmError::TooManyKeys)));
        }
    }
}
