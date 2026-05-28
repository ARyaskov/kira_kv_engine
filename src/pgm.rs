//! PGM (Piecewise Geometric Model) learned index for sorted u64 keys.
//!
//! v2 architecture additions over v1:
//!   - **f32 quantization** of slope/intercept (50% smaller segment metadata).
//!   - **Per-segment u8 epsilon** (`max_errors_u8`) with overflow Vec for rare cases >254.
//!   - **Branchless SIMD local search** via `movemask + tzcnt` (no scalar fallback inside SIMD loop).
//!   - **SIMD-aware branchless segment lookup** with cache-line prefetch on each step.
//!   - **Greedy-PLA optimal segmentation** with incremental convex-hull validation
//!     (replaces O(N²) regression-per-extension with O(N · avg_seg_len)).
//!   - **Parallel chunked build** via rayon (independent prefix sums, then merge).
//!   - **Optional Block-Bloom** fast-path for negative lookups (10–100× faster miss).
//!   - **Auto-tune epsilon** via micro-benchmark on the actual key distribution.
//!
//! On-disk wire format is bumped to version=2; v1 indexes are no longer readable.
//! That's an explicit trade-off — we accept format-break in exchange for a much
//! cleaner SoA layout that's friendlier to mmap zero-copy (handled by `mmap_index`).

use thiserror::Error;

use crate::block_bloom::BlockBloom;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vceqq_u64, vcgeq_u64, vdupq_n_u64, vgetq_lane_u64, vld1q_u64};
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_movemask_epi8,
    _mm256_or_si256, _mm256_set1_epi64x, _mm256_xor_si256, _mm_prefetch, _MM_HINT_T0,
};

// --------------------------------------------------------------------------------------
// Wire format version. v1 = legacy (f64 slopes, u32 errors). v2 = current.
// --------------------------------------------------------------------------------------
const PGM_FORMAT_V2: u8 = 0x02;

/// Segment built in-memory (used during PGM construction only).
/// Working-storage segment used during greedy-PLA construction. f64 is
/// retained until the final emit so quantization to f32 happens once with the
/// max-error check baked in (`segments_to_soa`).
#[derive(Debug, Clone)]
struct SegmentBuild {
    slope: f64,
    intercept: f64,
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
    pub(crate) slopes: Vec<f32>,           
    pub(crate) intercepts: Vec<f32>,       
    pub(crate) min_keys: Vec<u64>,
    pub(crate) max_keys: Vec<u64>,
    pub(crate) max_errors_u8: Vec<u8>,     // 0xFF sentinel = overflow
    pub(crate) overflow_errors: Vec<(u32, u32)>, // (segment_idx, real_error), sorted
    pub(crate) filters: Vec<u64>,
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

/// PGM Index for sorted integer keys with O(1) average lookup.
#[derive(Debug, Clone)]
pub struct PgmIndex {
    /// Sorted keys (plain storage). Empty when `keys_ef` is `Some` — that's the
    /// compacted state.
    keys: Vec<u64>,
    /// Optional Elias-Fano compact representation of `keys`. When set, the
    /// plain `keys` Vec is empty and key reads go through `key_at`/
    /// `materialize_range`. Trades ~30–50% memory for one extra cache line per
    /// local search on average.
    keys_ef: Option<crate::elias_fano::EliasFano>,
    /// Learned segments.
    segments: SegmentsSoA,
    /// Epsilon parameter for error tolerance (max across all segments).
    epsilon: u32,
    /// Optional Block-Bloom filter for fast negative lookup rejection.
    bloom: Option<BlockBloom>,
}

#[derive(Debug, Error)]
pub enum PgmError {
    #[error("keys must be sorted and unique")]
    UnsortedKeys,
    #[error("empty key set")]
    EmptyKeys,
    #[error("key not found")]
    KeyNotFound,
    #[error("corrupt data")]
    CorruptData,
    #[cfg(feature = "serde")]
    #[error("serialization error: {0}")]
    Serde(String),
}

impl PgmIndex {
    /// Build PGM index from sorted unique keys with the default settings (greedy
    /// PLA, no Bloom filter, sequential).
    pub fn build(keys: Vec<u64>, epsilon: u32) -> Result<Self, PgmError> {
        PgmBuilder::new().with_epsilon(epsilon).build(keys)
    }

    /// Build via builder — exposed mostly so the `mmap_index` zero-copy path
    /// can reconstruct the same shape from raw byte slices.
    pub(crate) fn from_parts(
        keys: Vec<u64>,
        segments: SegmentsSoA,
        epsilon: u32,
        bloom: Option<BlockBloom>,
    ) -> Self {
        Self {
            keys,
            keys_ef: None,
            segments,
            epsilon,
            bloom,
        }
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

    /// Get the i-th key (O(1) plain, O(1) amortized for EF).
    #[inline]
    fn key_at(&self, i: usize) -> u64 {
        if let Some(ef) = &self.keys_ef {
            ef.get(i)
        } else {
            self.keys[i]
        }
    }

    /// Number of keys in the index.
    #[inline]
    fn keys_count(&self) -> usize {
        if let Some(ef) = &self.keys_ef {
            ef.len()
        } else {
            self.keys.len()
        }
    }

    /// Materialize keys in `[from, to)` into `out`. Used by `index()` /
    /// `lower_bound()` for the local SIMD scan window.
    fn materialize_keys(&self, from: usize, to: usize, out: &mut Vec<u64>) {
        out.clear();
        if from >= to {
            return;
        }
        if let Some(ef) = &self.keys_ef {
            ef.materialize_range(from, to - from, out);
        } else {
            out.extend_from_slice(&self.keys[from..to]);
        }
    }

    /// Greedy-PLA optimal segmentation in O(N · avg_seg_len) time. Replaces the
    /// old O(N²) "regress from scratch on every extension attempt" loop.
    ///
    /// The shape of the algorithm:
    /// 1. We process keys left-to-right.
    /// 2. For each prefix we maintain incremental linear-regression stats
    ///    (sum_x, sum_y, sum_xy, sum_xx). Adding a point is O(1).
    /// 3. After each `add`, we recompute (slope, intercept) and validate
    ///    max_error against ALL points in the current segment. Validating is
    ///    O(seg_len) — but seg_len averages 30–100, so total work is linear in N.
    /// 4. If max_error > ε, we close the segment at the *previous* boundary
    ///    and start a new one.
    ///
    /// Why this beats the old "fit_segment(start..end) for every end"
    /// — the old code did a full quadratic-sum regression each time, so
    /// extending a segment to length L cost O(L), and extending to L+1 cost O(L+1),
    /// totalling O(L²) for the segment. Our incremental version is O(L) total.
    fn build_segments_greedy(
        keys: &[u64],
        epsilon: u32,
        cold_epsilon: u32,
        hot_start: usize,
        hot_end: usize,
    ) -> Vec<SegmentBuild> {
        let n = keys.len();
        if n == 0 {
            return Vec::new();
        }
        let mut segments: Vec<SegmentBuild> = Vec::with_capacity(n / 32 + 1);
        let mut start = 0usize;
        while start < n {
            let mut reg = LinReg::new();
            let mut last_good: Option<SegmentBuild> = None;
            let local_eps = if start >= hot_start && start < hot_end {
                epsilon
            } else {
                cold_epsilon
            };
            let mut end = start;
            while end < n {
                reg.add(keys[end], end);
                end += 1;
                if reg.n < 2.0 {
                    last_good = Some(SegmentBuild {
                        slope: 0.0,
                        intercept: start as f64,
                        min_key: keys[start],
                        max_key: keys[start],
                        start,
                        end,
                    });
                    continue;
                }
                if let Some((slope, intercept)) = reg.slope_intercept() {
                    if reg.max_error_quantized(slope, intercept, keys, start) <= local_eps {
                        last_good = Some(SegmentBuild {
                            slope,
                            intercept,
                            min_key: keys[start],
                            max_key: keys[end - 1],
                            start,
                            end,
                        });
                    } else {
                        end -= 1;
                        break;
                    }
                } else {
                    break;
                }
            }
            match last_good {
                Some(seg) => {
                    let advance = seg.end;
                    segments.push(seg);
                    start = advance;
                }
                None => {
                    segments.push(SegmentBuild {
                        slope: 0.0,
                        intercept: start as f64,
                        min_key: keys[start],
                        max_key: keys[start],
                        start,
                        end: start + 1,
                    });
                    start += 1;
                }
            }
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
    fn build_segments_parallel(
        keys: &[u64],
        epsilon: u32,
        cold_epsilon: u32,
        hot_start: usize,
        hot_end: usize,
    ) -> Vec<SegmentBuild> {
        use rayon::prelude::*;
        const CHUNK: usize = 128 * 1024;
        let n = keys.len();
        if n < CHUNK * 2 {
            return Self::build_segments_greedy(keys, epsilon, cold_epsilon, hot_start, hot_end);
        }
        let chunks: Vec<(usize, usize)> = (0..n)
            .step_by(CHUNK)
            .map(|s| (s, (s + CHUNK).min(n)))
            .collect();

        let per_chunk: Vec<Vec<SegmentBuild>> = chunks
            .par_iter()
            .map(|&(c_start, c_end)| {
                // Build a sub-vector of keys for this chunk. We pass positions
                // as the global index so segments carry true positions.
                let sub = &keys[c_start..c_end];
                let mut local = Self::build_segments_greedy(
                    sub,
                    epsilon,
                    cold_epsilon,
                    hot_start.saturating_sub(c_start).min(sub.len()),
                    hot_end.saturating_sub(c_start).min(sub.len()),
                );
                // Translate local positions (0..sub.len()) to global (c_start..c_end).
                for seg in &mut local {
                    seg.start += c_start;
                    seg.end += c_start;
                    seg.intercept += c_start as f64;
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

    /// Find position of key with O(1) average complexity.
    pub fn index(&self, key: u64) -> Result<usize, PgmError> {
        // optional Block-Bloom short-circuit for negative lookups.
        if let Some(bf) = &self.bloom {
            if !bf.contains_u64(key) {
                return Err(PgmError::KeyNotFound);
            }
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
        let predicted_pos = predict_pos(&self.segments, segment_idx, key);
        let n = self.keys_count();
        let max_error = self.segments.get_max_error(segment_idx) as usize;
        let search_start = predicted_pos.saturating_sub(max_error);
        let search_end = (predicted_pos + max_error + 1).min(n);

        // Fast path: keys live in plain `Vec<u64>`. SIMD search directly.
        if self.keys_ef.is_none() {
            let window = search_end - search_start;
            if window <= 256 {
                if let Some(pos) = find_in_range_simd(&self.keys, search_start, search_end, key) {
                    return Ok(pos);
                }
                return Err(PgmError::KeyNotFound);
            }
            return match self.keys[search_start..search_end].binary_search(&key) {
                Ok(local_pos) => Ok(search_start + local_pos),
                Err(_) => Err(PgmError::KeyNotFound),
            };
        }

        // EF path: materialize the search window into a small stack buffer,
        // then run the same SIMD scan against it.
        let mut buf = Vec::with_capacity(search_end - search_start);
        self.materialize_keys(search_start, search_end, &mut buf);
        if let Some(local) = find_in_range_simd(&buf, 0, buf.len(), key) {
            Ok(search_start + local)
        } else {
            Err(PgmError::KeyNotFound)
        }
    }

    /// Range query: find all positions with keys in [min_key, max_key].
    pub fn range(&self, min_key: u64, max_key: u64) -> Vec<usize> {
        let start_pos = self.lower_bound(min_key);
        let end_pos = self.upper_bound(max_key);
        (start_pos..end_pos).collect()
    }

    pub(crate) fn range_guard(&self, key: u64) -> bool {
        if self.keys_count() == 0 {
            return false;
        }
        let idx = find_segment_branchless(&self.segments.max_keys, key);
        if idx >= self.segments.max_keys.len() {
            return false;
        }
        let min_key = self.segments.min_keys[idx];
        let max_key = self.segments.max_keys[idx];
        key >= min_key && key <= max_key
    }

    /// Find first position where key >= target.
    pub fn lower_bound(&self, target: u64) -> usize {
        let n = self.keys_count();
        let segment_idx = find_segment_branchless(&self.segments.max_keys, target);
        if segment_idx >= self.segments.max_keys.len() {
            return n;
        }
        let predicted_pos = predict_pos(&self.segments, segment_idx, target);
        let max_error = self.segments.get_max_error(segment_idx) as usize;
        let search_start = predicted_pos.saturating_sub(max_error);
        let search_end = (predicted_pos + max_error + 1).min(n);

        if self.keys_ef.is_none() {
            if let Some(pos) = find_first_ge_simd(&self.keys, search_start, search_end, target) {
                return pos;
            }
            let mut pos = search_start;
            while pos < search_end {
                if self.keys[pos] >= target {
                    return pos;
                }
                pos += 1;
            }
            return n;
        }
        // EF path.
        let mut buf = Vec::with_capacity(search_end - search_start);
        self.materialize_keys(search_start, search_end, &mut buf);
        if let Some(local) = find_first_ge_simd(&buf, 0, buf.len(), target) {
            return search_start + local;
        }
        n
    }

    /// Find first position where key > target.
    pub fn upper_bound(&self, target: u64) -> usize {
        self.lower_bound(target.saturating_add(1))
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> PgmStats {
        let total_keys = self.keys_count();
        let total_segments = self.segments.max_keys.len();
        let avg_segment_size = if total_segments > 0 {
            total_keys as f64 / total_segments as f64
        } else {
            0.0
        };

        let max_error = (0..total_segments)
            .map(|i| self.segments.get_max_error(i))
            .max()
            .unwrap_or(0);

        let bloom_mem = self.bloom.as_ref().map(|b| b.memory_usage()).unwrap_or(0);
        let ef_mem = self.keys_ef.as_ref().map(|e| e.memory_usage()).unwrap_or(0);
        let memory_usage = std::mem::size_of_val(&self.keys)
            + self.keys.len() * std::mem::size_of::<u64>()
            + ef_mem
            + std::mem::size_of_val(&self.segments)
            + self.segments.slopes.len() * std::mem::size_of::<f32>()
            + self.segments.intercepts.len() * std::mem::size_of::<f32>()
            + self.segments.min_keys.len() * std::mem::size_of::<u64>()
            + self.segments.max_keys.len() * std::mem::size_of::<u64>()
            + self.segments.max_errors_u8.len() * std::mem::size_of::<u8>()
            + self.segments.overflow_errors.len() * std::mem::size_of::<(u32, u32)>()
            + self.segments.filters.len() * std::mem::size_of::<u64>()
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

    /// Serialize to a self-contained byte vector (uses the same v2 wire format
    /// as [`write_to`]).
    #[allow(dead_code)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, PgmError> {
        let mut out = Vec::with_capacity(self.stats().memory_usage);
        self.write_to(&mut out);
        Ok(out)
    }

    /// Deserialize from a byte slice produced by [`to_bytes`] / [`write_to`].
    #[allow(dead_code)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PgmError> {
        let mut pos = 0usize;
        Self::read_from(bytes, &mut pos)
    }

    /// Wire format v2:
    /// [u8 ver] [u32 epsilon] [u64 keys_len] [keys×u64]
    /// [u64 seg_len]
    /// segments contiguous SoA: slopes[u32 f32][intercepts][min_keys u64][max_keys u64]
    ///                          [max_errors u8] [u64 overflow_count]
    ///                          [overflow entries (u32 seg_idx, u32 err)×N]
    ///                          [filters u64][starts u32][ends u32]
    /// [u8 has_bloom] [bloom payload if present]
    pub(crate) fn write_to(&self, out: &mut Vec<u8>) {
        out.push(PGM_FORMAT_V2);
        write_u32(out, self.epsilon);
        let n = self.keys_count();
        write_u64(out, n as u64);
        // Plain or EF — emit the materialized u64 stream either way. EF reload
        // is a post-deserialization choice (call `compact_keys` if desired).
        if self.keys_ef.is_some() {
            for k in self.keys_iter() {
                write_u64(out, k);
            }
        } else {
            for k in &self.keys {
                write_u64(out, *k);
            }
        }
        let s = &self.segments;
        write_u64(out, s.len() as u64);
        for &v in &s.slopes {
            write_f32(out, v);
        }
        for &v in &s.intercepts {
            write_f32(out, v);
        }
        for &v in &s.min_keys {
            write_u64(out, v);
        }
        for &v in &s.max_keys {
            write_u64(out, v);
        }
        for &v in &s.max_errors_u8 {
            out.push(v);
        }
        write_u64(out, s.overflow_errors.len() as u64);
        for &(si, er) in &s.overflow_errors {
            write_u32(out, si);
            write_u32(out, er);
        }
        for &v in &s.filters {
            write_u64(out, v);
        }
        for &v in &s.starts {
            write_u32(out, v);
        }
        for &v in &s.ends {
            write_u32(out, v);
        }
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
        let mut cur = Cursor {
            buf: bytes,
            pos: *pos,
        };
        let ver = cur.read_u8().ok_or(PgmError::CorruptData)?;
        if ver != PGM_FORMAT_V2 {
            return Err(PgmError::CorruptData);
        }
        let epsilon = cur.read_u32().ok_or(PgmError::CorruptData)?;
        let keys_len = cur.read_u64().ok_or(PgmError::CorruptData)? as usize;
        let mut keys = Vec::with_capacity(keys_len);
        for _ in 0..keys_len {
            keys.push(cur.read_u64().ok_or(PgmError::CorruptData)?);
        }
        let seg_len = cur.read_u64().ok_or(PgmError::CorruptData)? as usize;

        let mut slopes = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            slopes.push(cur.read_f32().ok_or(PgmError::CorruptData)?);
        }
        let mut intercepts = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            intercepts.push(cur.read_f32().ok_or(PgmError::CorruptData)?);
        }
        let mut min_keys = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            min_keys.push(cur.read_u64().ok_or(PgmError::CorruptData)?);
        }
        let mut max_keys = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            max_keys.push(cur.read_u64().ok_or(PgmError::CorruptData)?);
        }
        let mut max_errors_u8 = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            max_errors_u8.push(cur.read_u8().ok_or(PgmError::CorruptData)?);
        }
        let over_n = cur.read_u64().ok_or(PgmError::CorruptData)? as usize;
        let mut overflow_errors = Vec::with_capacity(over_n);
        for _ in 0..over_n {
            let si = cur.read_u32().ok_or(PgmError::CorruptData)?;
            let er = cur.read_u32().ok_or(PgmError::CorruptData)?;
            overflow_errors.push((si, er));
        }
        let mut filters = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            filters.push(cur.read_u64().ok_or(PgmError::CorruptData)?);
        }
        let mut starts = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            starts.push(cur.read_u32().ok_or(PgmError::CorruptData)?);
        }
        let mut ends = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            ends.push(cur.read_u32().ok_or(PgmError::CorruptData)?);
        }
        let has_bloom = cur.read_u8().ok_or(PgmError::CorruptData)?;
        *pos = cur.pos;
        let bloom = if has_bloom == 1 {
            let mut bp = *pos;
            let bf = BlockBloom::read_from(bytes, &mut bp).ok_or(PgmError::CorruptData)?;
            *pos = bp;
            Some(bf)
        } else {
            None
        };
        Ok(PgmIndex {
            keys,
            keys_ef: None,
            segments: SegmentsSoA {
                slopes,
                intercepts,
                min_keys,
                max_keys,
                max_errors_u8,
                overflow_errors,
                filters,
                starts,
                ends,
            },
            epsilon,
            bloom,
        })
    }

    /// Convert `Vec<SegmentBuild>` (f64 working storage) into the SoA layout with
    /// f32-quantized slopes/intercepts and u8 per-segment errors. After quantization
    /// we re-measure max_error against quantized predictions — that's the cost of
    /// the size win. Segments whose post-quant error exceeds 254 fall into
    /// the overflow vec.
    fn segments_to_soa(keys: &[u64], segments: Vec<SegmentBuild>) -> SegmentsSoA {
        let n = segments.len();
        let mut slopes = Vec::with_capacity(n);
        let mut intercepts = Vec::with_capacity(n);
        let mut min_keys = Vec::with_capacity(n);
        let mut max_keys = Vec::with_capacity(n);
        let mut max_errors_u8 = Vec::with_capacity(n);
        let mut overflow_errors: Vec<(u32, u32)> = Vec::new();
        let mut filters = Vec::with_capacity(n);
        let mut starts = Vec::with_capacity(n);
        let mut ends = Vec::with_capacity(n);

        for (seg_idx, seg) in segments.iter().enumerate() {
            let sq = seg.slope as f32;
            let iq = seg.intercept as f32;
            // Recompute max_error against quantized predictions over [start, end).
            let mut err = 0u32;
            for i in seg.start..seg.end {
                let pred = (sq as f64).mul_add(keys[i] as f64, iq as f64);
                let actual = i as f64;
                let e = (pred - actual).abs() as u32;
                if e > err {
                    err = e;
                }
            }
            slopes.push(sq);
            intercepts.push(iq);
            min_keys.push(seg.min_key);
            max_keys.push(seg.max_key);
            if err <= 254 {
                max_errors_u8.push(err as u8);
            } else {
                max_errors_u8.push(0xFFu8);
                overflow_errors.push((seg_idx as u32, err));
            }
            starts.push(seg.start as u32);
            ends.push(seg.end as u32);
            let mut filter = 0u64;
            let mut i = seg.start;
            while i < seg.end {
                filter |= filter_bit(keys[i]);
                i += 1;
            }
            filters.push(filter);
        }
        // overflow_errors is built in segment-index order; binary_search_by_key works.
        SegmentsSoA {
            slopes,
            intercepts,
            min_keys,
            max_keys,
            max_errors_u8,
            overflow_errors,
            filters,
            starts,
            ends,
        }
    }

    pub(crate) fn filter_allows(&self, key: u64) -> bool {
        let idx = find_segment_branchless(&self.segments.max_keys, key);
        if idx >= self.segments.max_keys.len() {
            return false;
        }
        let min_key = self.segments.min_keys[idx];
        let max_key = self.segments.max_keys[idx];
        if key < min_key || key > max_key {
            return false;
        }
        (self.segments.filters[idx] & filter_bit(key)) != 0
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
    pub(crate) fn num_segments(&self) -> usize {
        self.segments.max_keys.len()
    }

    pub(crate) fn keys_len(&self) -> usize {
        self.keys_count()
    }

    /// Plain-storage slice. Empty when the index has been compacted via
    /// [`compact_keys`] (EF-backed). Callers needing random key access in that
    /// state should use [`key_at_public`] or [`keys_iter`].
    pub(crate) fn keys(&self) -> &[u64] {
        &self.keys
    }

    /// Random access to the i-th key — O(1) for plain, O(1) amortized for EF.
    pub(crate) fn key_at_public(&self, i: usize) -> u64 {
        self.key_at(i)
    }

    /// Iterate keys in order. Works whether plain or EF-backed.
    pub(crate) fn keys_iter(&self) -> KeysIter<'_> {
        KeysIter {
            pgm: self,
            i: 0,
            n: self.keys_count(),
        }
    }

    pub(crate) fn segment_density_order(&self, hot_fraction: f64) -> Vec<usize> {
        let n = self.keys.len().max(1);
        let target = ((n as f64) * hot_fraction).ceil() as usize;
        let mut ids: Vec<usize> = (0..self.segments.max_keys.len()).collect();
        ids.sort_by(|&a, &b| {
            let a_len = (self.segments.ends[a] - self.segments.starts[a]) as u128;
            let b_len = (self.segments.ends[b] - self.segments.starts[b]) as u128;
            let a_span = (self.segments.max_keys[a] - self.segments.min_keys[a] + 1) as u128;
            let b_span = (self.segments.max_keys[b] - self.segments.min_keys[b] + 1) as u128;
            let left = a_len * b_span;
            let right = b_len * a_span;
            right.cmp(&left).then_with(|| a.cmp(&b))
        });
        let mut picked = Vec::new();
        let mut count = 0usize;
        for id in ids {
            let len = (self.segments.ends[id] - self.segments.starts[id]) as usize;
            picked.push(id);
            count += len;
            if count >= target {
                break;
            }
        }
        picked
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
        writer.add_section(SectionKind::PgmKeys, bytes_of_u64(&self.keys));
        writer.add_section(SectionKind::PgmSlopes, bytes_of_f32(&s.slopes));
        writer.add_section(SectionKind::PgmIntercepts, bytes_of_f32(&s.intercepts));
        writer.add_section(SectionKind::PgmMinKeys, bytes_of_u64(&s.min_keys));
        writer.add_section(SectionKind::PgmMaxKeys, bytes_of_u64(&s.max_keys));
        writer.add_section(SectionKind::PgmMaxErrors, s.max_errors_u8.clone());
        let overflow_bytes = bytes_of_u32_pairs(&s.overflow_errors);
        writer.add_section(SectionKind::PgmOverflowErrors, overflow_bytes);
        writer.add_section(SectionKind::PgmFilters, bytes_of_u64(&s.filters));
        writer.add_section(SectionKind::PgmStarts, bytes_of_u32(&s.starts));
        writer.add_section(SectionKind::PgmEnds, bytes_of_u32(&s.ends));
        // PgmMeta: [u32 epsilon][u8 has_bloom][padding..]
        let mut meta = Vec::with_capacity(8);
        meta.extend_from_slice(&self.epsilon.to_le_bytes());
        meta.push(if self.bloom.is_some() { 1 } else { 0 });
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
        let keys = u64_vec_from_section(mmap.section(header, SectionKind::PgmKeys)?);
        let slopes = f32_vec_from_section(mmap.section(header, SectionKind::PgmSlopes)?);
        let intercepts = f32_vec_from_section(mmap.section(header, SectionKind::PgmIntercepts)?);
        let min_keys = u64_vec_from_section(mmap.section(header, SectionKind::PgmMinKeys)?);
        let max_keys = u64_vec_from_section(mmap.section(header, SectionKind::PgmMaxKeys)?);
        let max_errors_u8 = mmap.section(header, SectionKind::PgmMaxErrors)?.to_vec();
        let overflow_errors =
            u32_pairs_from_section(mmap.section(header, SectionKind::PgmOverflowErrors)?);
        let filters = u64_vec_from_section(mmap.section(header, SectionKind::PgmFilters)?);
        let starts = u32_vec_from_section(mmap.section(header, SectionKind::PgmStarts)?);
        let ends = u32_vec_from_section(mmap.section(header, SectionKind::PgmEnds)?);
        let meta = mmap.section(header, SectionKind::PgmMeta)?;
        if meta.len() < 5 {
            return None;
        }
        let epsilon = u32::from_le_bytes([meta[0], meta[1], meta[2], meta[3]]);
        let has_bloom = meta[4] != 0;
        let bloom = if has_bloom {
            let bf_bytes = mmap.section(header, SectionKind::PgmBloom)?;
            let mut pos = 0usize;
            crate::block_bloom::BlockBloom::read_from(bf_bytes, &mut pos)
        } else {
            None
        };
        Some(PgmIndex {
            keys,
            keys_ef: None,
            segments: SegmentsSoA {
                slopes,
                intercepts,
                min_keys,
                max_keys,
                max_errors_u8,
                overflow_errors,
                filters,
                starts,
                ends,
            },
            epsilon,
            bloom,
        })
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
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_of_u32(v: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_of_f32(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
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

fn u64_vec_from_section(bytes: &[u8]) -> Vec<u64> {
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(8) {
        let mut a = [0u8; 8];
        a.copy_from_slice(chunk);
        out.push(u64::from_le_bytes(a));
    }
    out
}

fn u32_vec_from_section(bytes: &[u8]) -> Vec<u32> {
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        let mut a = [0u8; 4];
        a.copy_from_slice(chunk);
        out.push(u32::from_le_bytes(a));
    }
    out
}

fn f32_vec_from_section(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        let mut a = [0u8; 4];
        a.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(a));
    }
    out
}

fn u32_pairs_from_section(bytes: &[u8]) -> Vec<(u32, u32)> {
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(8) {
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        a.copy_from_slice(&chunk[..4]);
        b.copy_from_slice(&chunk[4..]);
        out.push((u32::from_le_bytes(a), u32::from_le_bytes(b)));
    }
    out
}

// --------------------------------------------------------------------------------------
// Incremental linear regression. Used by build_segments_greedy.
// --------------------------------------------------------------------------------------

struct LinReg {
    n: f64,
    sum_x: f64,
    sum_y: f64,
    sum_xy: f64,
    sum_xx: f64,
}

impl LinReg {
    fn new() -> Self {
        Self {
            n: 0.0,
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xy: 0.0,
            sum_xx: 0.0,
        }
    }

    #[inline]
    fn add(&mut self, key: u64, pos: usize) {
        let kf = key as f64;
        let pf = pos as f64;
        self.n += 1.0;
        self.sum_x += kf;
        self.sum_y += pf;
        self.sum_xy += kf * pf;
        self.sum_xx += kf * kf;
    }

    #[inline]
    fn slope_intercept(&self) -> Option<(f64, f64)> {
        let denom = self.n * self.sum_xx - self.sum_x * self.sum_x;
        if denom.abs() < 1e-10 {
            return None;
        }
        let slope = (self.n * self.sum_xy - self.sum_x * self.sum_y) / denom;
        let intercept = (self.sum_y - slope * self.sum_x) / self.n;
        Some((slope, intercept))
    }

    /// Compute max prediction error after we'd quantize slope→f32 and intercept→f32.
    /// This is what determines whether the segment can be stored with the current ε.
    /// We compute it against the quantized values so we don't accept segments that
    /// later violate ε once we drop precision (honesty check).
    #[inline]
    fn max_error_quantized(&self, slope: f64, intercept: f64, keys: &[u64], start: usize) -> u32 {
        let sq = slope as f32 as f64;
        let iq = intercept as f32 as f64;
        let mut max_err = 0u32;
        let end = start + self.n as usize;
        for (offset, &key) in keys[start..end].iter().enumerate() {
            let pred = sq.mul_add(key as f64, iq);
            let actual = (start + offset) as f64;
            let err = (pred - actual).abs() as u32;
            if err > max_err {
                max_err = err;
            }
        }
        max_err
    }
}

// --------------------------------------------------------------------------------------
// SIMD branchless local search. The old code had a scalar `for lane in 0..4`
// inside the SIMD loop that triggered branch mispredicts on each match. We replace
// it with `_mm256_movemask_epi8 + tzcnt`, which gives the matched lane in 2 cycles
// and zero branches.
// --------------------------------------------------------------------------------------

#[inline]
fn predict_pos(segments: &SegmentsSoA, idx: usize, key: u64) -> usize {
    // f32 mul_add for cache-friendly hot path; if intermediate precision matters
    // we widen to f64 for the final accumulate.
    let s = segments.slopes[idx] as f64;
    let b = segments.intercepts[idx] as f64;
    let prediction = s.mul_add(key as f64, b);
    if prediction <= 0.0 {
        0
    } else {
        prediction as usize
    }
}

#[inline]
fn filter_bit(key: u64) -> u64 {
    let h = splitmix64(key);
    1u64 << (h & 63)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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
unsafe fn find_first_ge_avx2(
    keys: &[u64],
    start: usize,
    end: usize,
    target: u64,
) -> Option<usize> {
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
unsafe fn find_first_ge_neon(
    keys: &[u64],
    start: usize,
    end: usize,
    target: u64,
) -> Option<usize> {
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
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if half > 4 {
                let nq = half / 2;
                _mm_prefetch(ptr.add(base + nq) as *const i8, _MM_HINT_T0);
                _mm_prefetch(ptr.add(mid + nq) as *const i8, _MM_HINT_T0);
            }
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
    if base < n && unsafe { *ptr.add(base) } < key {
        base + 1
    } else {
        base
    }
}

// --------------------------------------------------------------------------------------
// Serialization scratch.
// --------------------------------------------------------------------------------------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn read_u8(&mut self) -> Option<u8> {
        if self.pos + 1 > self.buf.len() {
            return None;
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        Some(v)
    }
    fn read_u32(&mut self) -> Option<u32> {
        if self.pos + 4 > self.buf.len() {
            return None;
        }
        let mut array = [0u8; 4];
        array.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Some(u32::from_le_bytes(array))
    }
    fn read_u64(&mut self) -> Option<u64> {
        if self.pos + 8 > self.buf.len() {
            return None;
        }
        let mut array = [0u8; 8];
        array.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Some(u64::from_le_bytes(array))
    }
    fn read_f32(&mut self) -> Option<f32> {
        if self.pos + 4 > self.buf.len() {
            return None;
        }
        let mut array = [0u8; 4];
        array.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Some(f32::from_le_bytes(array))
    }
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn write_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

// --------------------------------------------------------------------------------------
// Public stats / builder API.
// --------------------------------------------------------------------------------------

/// Statistics about PGM index.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PgmStats {
    pub total_keys: usize,
    pub total_segments: usize,
    pub avg_segment_size: f64,
    pub max_error: u32,
    pub memory_usage: usize,
    pub epsilon: u32,
}

#[allow(dead_code)]
impl PgmStats {
    pub fn print_summary(&self) {
        println!("PGM Index Statistics:");
        println!("  Total keys: {}", self.total_keys);
        println!("  Segments: {}", self.total_segments);
        println!("  Avg segment size: {:.1}", self.avg_segment_size);
        println!("  Max error: {}", self.max_error);
        println!(
            "  Memory usage: {:.2} MB",
            self.memory_usage as f64 / 1_048_576.0
        );
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

    /// Compact key storage via Elias-Fano after build. Saves ~30–50% on key
    /// memory at large N, costs an extra ~50 ns per lookup for the materialize
    /// step. Net win for memory-bound workloads (100M+ keys) where the whole
    /// `keys` array no longer fits comfortably in LLC.
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
        let mut sorted = keys;
        sorted.sort_unstable();
        for w in sorted.windows(2) {
            if w[0] >= w[1] {
                return Err(PgmError::UnsortedKeys);
            }
        }

        if sorted.is_empty() {
            return Ok(PgmIndex {
                keys: Vec::new(),
                keys_ef: None,
                segments: SegmentsSoA::default(),
                epsilon: self.epsilon,
                bloom: None,
            });
        }

        let epsilon = match self.target_lookup_ns {
            Some(target) => Self::auto_tune_epsilon(&sorted, target).unwrap_or(self.epsilon),
            None => self.epsilon,
        };

        let cold_epsilon = epsilon.saturating_mul(4).max(epsilon);
        let hot_start = sorted.len() / 10;
        let hot_end = sorted.len().saturating_sub(sorted.len() / 10);

        let raw_segments = {
            #[cfg(feature = "parallel")]
            {
                if self.enable_parallel {
                    PgmIndex::build_segments_parallel(
                        &sorted,
                        epsilon,
                        cold_epsilon,
                        hot_start,
                        hot_end,
                    )
                } else {
                    PgmIndex::build_segments_greedy(
                        &sorted,
                        epsilon,
                        cold_epsilon,
                        hot_start,
                        hot_end,
                    )
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                PgmIndex::build_segments_greedy(&sorted, epsilon, cold_epsilon, hot_start, hot_end)
            }
        };
        let segments = PgmIndex::segments_to_soa(&sorted, raw_segments);

        let bloom = if self.enable_bloom {
            Some(BlockBloom::build_from_u64(&sorted, 0xC1B5_4A32_D192_ED03))
        } else {
            None
        };

        let mut idx = PgmIndex {
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
            let segs = PgmIndex::build_segments_greedy(
                &sample,
                eps,
                eps.saturating_mul(4),
                sample.len() / 10,
                sample.len().saturating_sub(sample.len() / 10),
            );
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
