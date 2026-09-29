//! PGM index for fixed-size 16-byte keys (`u128`) — typically UUID/SHA-128/IPv6
//! addresses.
//!
//! Architecturally similar to `PgmIndex<u64>`: linear segments over the sorted
//! key sequence, predicted position + local search. Each segment's line is
//! fitted on `key - min_key` (the u128 difference converted to f64), so the
//! 53-bit mantissa covers the segment's span rather than the whole 2^128
//! universe — the earlier absolute-key fit lost every bit below 2^75 and
//! produced segments of a handful of keys for random UUIDs. Per segment:
//! 32 B min/max + 8 B slope + 1 B error + 8 B start/end.
//!
//! No SIMD on the local-search step — AVX2 doesn't have 128-bit equality
//! compare and AVX-512 isn't always available. The local-scan window is
//! typically only `2ε+1` ≤ 257 elements anyway, so a tight scalar loop with
//! prefetch lands at ~30–60 ns per lookup with full L1 residency.

use crate::prefetch::prefetch_read;
use thiserror::Error;

/// PGM Index for sorted unique 16-byte keys.
#[derive(Debug, Clone)]
pub struct PgmIndexU128 {
    keys: Vec<u128>,
    segments: SegmentsSoA,
    epsilon: u32,
}

#[derive(Debug, Clone, Default)]
struct SegmentsSoA {
    /// Positions per key unit, relative to the segment's `min_key`:
    /// `pos = slope * (key - min_key) + start`. f64 because for u128 universes the
    /// slope is ∼N/2^128, outside the f32 range.
    slopes: Vec<f64>,
    min_keys: Vec<u128>,
    max_keys: Vec<u128>,
    max_errors_u8: Vec<u8>,
    overflow_errors: Vec<(u32, u32)>,
    starts: Vec<u32>,
    ends: Vec<u32>,
}

impl SegmentsSoA {
    fn len(&self) -> usize {
        self.max_keys.len()
    }

    fn get_max_error(&self, seg_idx: usize) -> u32 {
        let e = self.max_errors_u8[seg_idx];
        if e != 0xFF {
            e as u32
        } else {
            self.overflow_errors
                .binary_search_by_key(&(seg_idx as u32), |&(s, _)| s)
                .map(|i| self.overflow_errors[i].1)
                .unwrap_or(255)
        }
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PgmU128Error {
    /// The input contains the same key twice (the builder sorts, so order is
    /// never the problem).
    #[error("duplicate key in input")]
    DuplicateKeys,
    #[deprecated(since = "0.7.0", note = "the builder sorts its input; duplicates raise DuplicateKeys")]
    #[error("keys must be sorted and unique")]
    UnsortedKeys,
    /// No longer returned: an empty key set builds an always-miss index.
    #[deprecated(since = "0.7.0", note = "empty input is accepted")]
    #[error("empty key set")]
    EmptyKeys,
    #[error("key not found")]
    KeyNotFound,
    #[error("corrupt data")]
    CorruptData,
}

impl PgmIndexU128 {
    /// Build a PGM-U128 from unique u128 keys (sorted here if needed). An empty
    /// key set yields an index on which every lookup misses.
    pub fn build(mut keys: Vec<u128>, epsilon: u32) -> Result<Self, PgmU128Error> {
        keys.sort_unstable();
        if keys.windows(2).any(|w| w[0] >= w[1]) {
            return Err(PgmU128Error::DuplicateKeys);
        }
        let segs = Self::build_segments(&keys, epsilon);
        Ok(Self {
            keys,
            segments: segs,
            epsilon,
        })
    }

    /// Convenience: build from raw 16-byte slices (e.g. `&[[u8; 16]]`). Each slice
    /// is interpreted as big-endian to preserve lexicographic ordering — i.e.
    /// the byte-wise sort order matches the u128 numeric order.
    pub fn build_from_bytes16(keys: &[[u8; 16]], epsilon: u32) -> Result<Self, PgmU128Error> {
        let u128_keys: Vec<u128> = keys.iter().map(|b| u128::from_be_bytes(*b)).collect();
        Self::build(u128_keys, epsilon)
    }

    /// Linear-time slope-window segmentation (same algorithm as the u64 index),
    /// fitted on `key - segment_min_key` so a segment's line keeps full f64
    /// precision however large the absolute keys are.
    fn build_segments(keys: &[u128], epsilon: u32) -> SegmentsSoA {
        let n = keys.len();
        let eps = epsilon as f64;
        let mut raw = Vec::with_capacity(n / 32 + 1);
        let mut start = 0usize;
        while start < n {
            let base = keys[start];
            let mut lo = f64::NEG_INFINITY;
            let mut hi = f64::INFINITY;
            let mut end = start + 1;
            while end < n {
                let dx = (keys[end] - base) as f64;
                if dx <= 0.0 {
                    break;
                }
                let dy = (end - start) as f64;
                let new_lo = lo.max((dy - eps) / dx);
                let new_hi = hi.min((dy + eps) / dx);
                if new_lo > new_hi {
                    break;
                }
                lo = new_lo;
                hi = new_hi;
                end += 1;
            }
            let slope = if end == start + 1 { 0.0 } else { 0.5 * (lo + hi) };
            // Measure the real error of the stored line once (O(L)).
            let mut max_error = 0u32;
            for (i, &k) in keys[start..end].iter().enumerate() {
                let pred = slope * ((k - base) as f64) + start as f64;
                let e = (pred - (start + i) as f64).abs().ceil() as u32;
                max_error = max_error.max(e);
            }
            raw.push(RawSeg {
                slope,
                min_key: base,
                max_key: keys[end - 1],
                max_error,
                start,
                end,
            });
            start = end;
        }
        Self::pack(raw)
    }

    fn pack(raw: Vec<RawSeg>) -> SegmentsSoA {
        let n = raw.len();
        let mut s = SegmentsSoA {
            slopes: Vec::with_capacity(n),
            min_keys: Vec::with_capacity(n),
            max_keys: Vec::with_capacity(n),
            max_errors_u8: Vec::with_capacity(n),
            overflow_errors: Vec::new(),
            starts: Vec::with_capacity(n),
            ends: Vec::with_capacity(n),
        };
        for (i, seg) in raw.into_iter().enumerate() {
            s.slopes.push(seg.slope);
            s.min_keys.push(seg.min_key);
            s.max_keys.push(seg.max_key);
            if seg.max_error <= 254 {
                s.max_errors_u8.push(seg.max_error as u8);
            } else {
                s.max_errors_u8.push(0xFF);
                s.overflow_errors.push((i as u32, seg.max_error));
            }
            s.starts.push(seg.start as u32);
            s.ends.push(seg.end as u32);
        }
        s
    }

    /// Lookup a key — returns its position in the sorted sequence, or
    /// `KeyNotFound`.
    pub fn index(&self, key: u128) -> Result<usize, PgmU128Error> {
        let seg = find_segment_u128(&self.segments.max_keys, key);
        if seg >= self.segments.max_keys.len() {
            return Err(PgmU128Error::KeyNotFound);
        }
        if key < self.segments.min_keys[seg] || key > self.segments.max_keys[seg] {
            return Err(PgmU128Error::KeyNotFound);
        }
        let pred = predict_pos_u128(&self.segments, seg, key);
        let err = self.segments.get_max_error(seg) as usize;
        let s = pred.saturating_sub(err).min(self.keys.len());
        let e = (pred + err + 1).min(self.keys.len()).max(s);
        // Sorted window: bisect it (branchless). 16-byte keys make a linear scan
        // of 2ε+1 entries several cache lines of work; 7 dependent loads are not.
        let w = &self.keys[s..e];
        let mut base = 0usize;
        let mut len = w.len();
        while len > 1 {
            let half = len / 2;
            let m = w[base + half];
            base += if m <= key { half } else { 0 };
            len -= half;
        }
        if len == 1 && w[base] == key { Ok(s + base) } else { Err(PgmU128Error::KeyNotFound) }
    }

    /// Convenience wrapper accepting a 16-byte big-endian slice.
    pub fn index_bytes16(&self, key: &[u8; 16]) -> Result<usize, PgmU128Error> {
        self.index(u128::from_be_bytes(*key))
    }

    /// Positions of all keys in `[min_key, max_key]`, as the half-open range
    /// `lower_bound(min_key)..upper_bound(max_key)` — O(1) space however large the
    /// range is.
    pub fn range(&self, min_key: u128, max_key: u128) -> std::ops::Range<usize> {
        let lo = self.lower_bound(min_key);
        let hi = self.upper_bound(max_key);
        lo..hi.max(lo)
    }

    pub fn lower_bound(&self, target: u128) -> usize {
        let seg = find_segment_u128(&self.segments.max_keys, target);
        if seg >= self.segments.max_keys.len() {
            return self.keys.len();
        }
        let pred = predict_pos_u128(&self.segments, seg, target);
        let err = self.segments.get_max_error(seg) as usize;
        let s = pred.saturating_sub(err).min(self.keys.len());
        let e = (pred + err + 1).min(self.keys.len()).max(s);
        let p = self.keys[s..e].partition_point(|&k| k < target);
        if p < e - s { s + p } else { e }
    }

    pub fn upper_bound(&self, target: u128) -> usize {
        self.lower_bound(target.saturating_add(1))
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn segments_count(&self) -> usize {
        self.segments.len()
    }

    pub fn memory_usage(&self) -> usize {
        let s = &self.segments;
        std::mem::size_of_val(&self.keys)
            + self.keys.len() * std::mem::size_of::<u128>()
            + s.slopes.len() * 8
            + s.min_keys.len() * 16
            + s.max_keys.len() * 16
            + s.max_errors_u8.len()
            + s.overflow_errors.len() * 8
            + s.starts.len() * 4
            + s.ends.len() * 4
    }

    pub fn epsilon(&self) -> u32 {
        self.epsilon
    }

    /// Serialize into a self-contained, checksummed byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let s = &self.segments;
        let mut body = Vec::with_capacity(self.memory_usage() + 64);
        body.extend_from_slice(&self.epsilon.to_le_bytes());
        body.extend_from_slice(&(self.keys.len() as u64).to_le_bytes());
        crate::wire::extend_le(&mut body, &self.keys);
        body.extend_from_slice(&(s.len() as u64).to_le_bytes());
        crate::wire::extend_le(&mut body, &s.slopes);
        crate::wire::extend_le(&mut body, &s.min_keys);
        crate::wire::extend_le(&mut body, &s.max_keys);
        body.extend_from_slice(&s.max_errors_u8);
        body.extend_from_slice(&(s.overflow_errors.len() as u64).to_le_bytes());
        for &(si, er) in &s.overflow_errors {
            body.extend_from_slice(&si.to_le_bytes());
            body.extend_from_slice(&er.to_le_bytes());
        }
        crate::wire::extend_le(&mut body, &s.starts);
        crate::wire::extend_le(&mut body, &s.ends);
        crate::wire::seal(crate::wire::KIND_PGM_U128, &body)
    }

    /// Deserialize [`PgmIndexU128::to_bytes`] output; checksum and structure are verified.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PgmU128Error> {
        let body = crate::wire::unseal(bytes, crate::wire::KIND_PGM_U128).ok_or(PgmU128Error::CorruptData)?;
        let bad = || PgmU128Error::CorruptData;
        let mut pos = 0usize;
        let rd_u64 = |pos: &mut usize| -> Result<usize, PgmU128Error> {
            let v = body.get(*pos..*pos + 8).ok_or(PgmU128Error::CorruptData)?;
            *pos += 8;
            usize::try_from(u64::from_le_bytes(v.try_into().unwrap())).map_err(|_| PgmU128Error::CorruptData)
        };
        let epsilon = u32::from_le_bytes(body.get(0..4).ok_or_else(bad)?.try_into().unwrap());
        pos += 4;
        let n = rd_u64(&mut pos)?;
        let keys: Vec<u128> = crate::wire::read_le_at(body, &mut pos, n).ok_or_else(bad)?;
        let m = rd_u64(&mut pos)?;
        if m > (body.len() - pos) / 49 {
            return Err(bad());
        }
        let slopes: Vec<f64> = crate::wire::read_le_at(body, &mut pos, m).ok_or_else(bad)?;
        let min_keys: Vec<u128> = crate::wire::read_le_at(body, &mut pos, m).ok_or_else(bad)?;
        let max_keys: Vec<u128> = crate::wire::read_le_at(body, &mut pos, m).ok_or_else(bad)?;
        let max_errors_u8 = body.get(pos..pos + m).ok_or_else(bad)?.to_vec();
        pos += m;
        let ov = rd_u64(&mut pos)?;
        if ov > (body.len() - pos) / 8 {
            return Err(bad());
        }
        let mut overflow_errors = Vec::with_capacity(ov);
        for _ in 0..ov {
            let a = body.get(pos..pos + 8).ok_or_else(bad)?;
            overflow_errors.push((
                u32::from_le_bytes(a[..4].try_into().unwrap()),
                u32::from_le_bytes(a[4..].try_into().unwrap()),
            ));
            pos += 8;
        }
        let starts: Vec<u32> = crate::wire::read_le_at(body, &mut pos, m).ok_or_else(bad)?;
        let ends: Vec<u32> = crate::wire::read_le_at(body, &mut pos, m).ok_or_else(bad)?;
        if pos != body.len() {
            return Err(bad());
        }
        let idx = Self {
            keys,
            segments: SegmentsSoA { slopes, min_keys, max_keys, max_errors_u8, overflow_errors, starts, ends },
            epsilon,
        };
        if idx.validate() { Ok(idx) } else { Err(bad()) }
    }

    /// Structural invariants the lookup relies on.
    fn validate(&self) -> bool {
        let s = &self.segments;
        let n = self.keys.len();
        let m = s.len();
        if n == 0 {
            return m == 0;
        }
        if m == 0 || n > u32::MAX as usize || self.keys.windows(2).any(|w| w[0] >= w[1]) {
            return false;
        }
        let mut prev_end = 0u32;
        for i in 0..m {
            if s.starts[i] != prev_end || s.ends[i] <= s.starts[i] || s.ends[i] as usize > n {
                return false;
            }
            if s.min_keys[i] > s.max_keys[i]
                || s.min_keys[i] != self.keys[s.starts[i] as usize]
                || s.max_keys[i] != self.keys[s.ends[i] as usize - 1]
                || !s.slopes[i].is_finite()
            {
                return false;
            }
            prev_end = s.ends[i];
        }
        prev_end as usize == n
            && s.overflow_errors.windows(2).all(|w| w[0].0 < w[1].0)
            && s.overflow_errors.iter().all(|&(si, _)| (si as usize) < m)
    }
}

#[derive(Debug, Clone)]
struct RawSeg {
    slope: f64,
    min_key: u128,
    max_key: u128,
    max_error: u32,
    start: usize,
    end: usize,
}

/// Truncated position prediction; with the error rounded up at build time the
/// `[pos - err, pos + err]` window of the callers always contains the key.
#[inline]
fn predict_pos_u128(seg: &SegmentsSoA, idx: usize, key: u128) -> usize {
    let dx = key.saturating_sub(seg.min_keys[idx]) as f64;
    let p = seg.slopes[idx].mul_add(dx, seg.starts[idx] as f64);
    if p <= 0.0 { 0 } else { p as usize }
}

#[inline]
fn find_segment_u128(max_keys: &[u128], key: u128) -> usize {
    // Branchless binary search for u128. No SIMD: AVX2 lacks 128-bit
    // compare. We rely on branchless cmov + prefetch — the latter helps a lot
    // at large N where each step is a cold cache miss.
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
        if half > 4 {
            let nq = half / 2;
            // SAFETY: both offsets are below `n`; a prefetch is a hint.
            prefetch_read(unsafe { ptr.add(base + nq) });
            prefetch_read(unsafe { ptr.add(mid + nq) });
        }
        let m = unsafe { *ptr.add(mid) };
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

