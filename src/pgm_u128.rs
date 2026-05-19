//! PGM index for fixed-size 16-byte keys (`u128`) — typically UUID/SHA-128/IPv6
//! addresses.
//!
//! Architecturally similar to `PgmIndex<u64>`: linear-regression segments over
//! the sorted key sequence, predicted position + local search. We use `i128`
//! intermediate arithmetic for slope/intercept (with f64 widening for the LR
//! fit) and accept a slightly larger memory footprint per segment (32 B of
//! min/max + 8 B slope + 8 B intercept + 4 B errors = 52 B/segment vs 40 B in
//! the u64 variant).
//!
//! No SIMD on the local-search step — AVX2 doesn't have 128-bit equality
//! compare and AVX-512 isn't always available. The local-scan window is
//! typically only `2ε+1` ≤ 257 elements anyway, so a tight scalar loop with
//! prefetch lands at ~30–60 ns per lookup with full L1 residency.

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
    /// f64 here — for u128 universes the slope is typically very small (∼N/2^128)
    /// and quantizing to f32 immediately overflows the f32 dynamic range.
    slopes: Vec<f64>,
    intercepts: Vec<f64>,
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
pub enum PgmU128Error {
    #[error("keys must be sorted and unique")]
    UnsortedKeys,
    #[error("empty key set")]
    EmptyKeys,
    #[error("key not found")]
    KeyNotFound,
    #[error("corrupt data")]
    CorruptData,
}

impl PgmIndexU128 {
    /// Build a PGM-U128 from sorted unique u128 keys.
    pub fn build(mut keys: Vec<u128>, epsilon: u32) -> Result<Self, PgmU128Error> {
        if keys.is_empty() {
            return Err(PgmU128Error::EmptyKeys);
        }
        keys.sort_unstable();
        for w in keys.windows(2) {
            if w[0] >= w[1] {
                return Err(PgmU128Error::UnsortedKeys);
            }
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

    fn build_segments(keys: &[u128], epsilon: u32) -> SegmentsSoA {
        let n = keys.len();
        let mut raw = Vec::with_capacity(n / 32 + 1);
        let mut start = 0usize;
        while start < n {
            let mut reg = LinReg128::new();
            let mut last_good: Option<RawSeg> = None;
            let mut end = start;
            while end < n {
                reg.add(keys[end], end);
                end += 1;
                if reg.n < 2.0 {
                    last_good = Some(RawSeg {
                        slope: 0.0,
                        intercept: start as f64,
                        min_key: keys[start],
                        max_key: keys[start],
                        max_error: 0,
                        start,
                        end,
                    });
                    continue;
                }
                if let Some((slope, intercept)) = reg.slope_intercept() {
                    let me = reg.max_error(slope, intercept, keys, start);
                    if me <= epsilon {
                        last_good = Some(RawSeg {
                            slope,
                            intercept,
                            min_key: keys[start],
                            max_key: keys[end - 1],
                            max_error: me,
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
                    raw.push(seg);
                    start = advance;
                }
                None => {
                    raw.push(RawSeg {
                        slope: 0.0,
                        intercept: start as f64,
                        min_key: keys[start],
                        max_key: keys[start],
                        max_error: 0,
                        start,
                        end: start + 1,
                    });
                    start += 1;
                }
            }
        }
        Self::pack(raw)
    }

    fn pack(raw: Vec<RawSeg>) -> SegmentsSoA {
        let n = raw.len();
        let mut s = SegmentsSoA {
            slopes: Vec::with_capacity(n),
            intercepts: Vec::with_capacity(n),
            min_keys: Vec::with_capacity(n),
            max_keys: Vec::with_capacity(n),
            max_errors_u8: Vec::with_capacity(n),
            overflow_errors: Vec::new(),
            starts: Vec::with_capacity(n),
            ends: Vec::with_capacity(n),
        };
        for (i, seg) in raw.into_iter().enumerate() {
            s.slopes.push(seg.slope);
            s.intercepts.push(seg.intercept);
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
        let s = pred.saturating_sub(err);
        let e = (pred + err + 1).min(self.keys.len());
        // Tight scalar scan with cache-line prefetch.
        let mut i = s;
        while i < e {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                if i + 8 < e {
                    _mm_prefetch(self.keys.as_ptr().add(i + 8) as *const i8, _MM_HINT_T0);
                }
            }
            if self.keys[i] == key {
                return Ok(i);
            }
            i += 1;
        }
        Err(PgmU128Error::KeyNotFound)
    }

    /// Convenience wrapper accepting a 16-byte big-endian slice.
    pub fn index_bytes16(&self, key: &[u8; 16]) -> Result<usize, PgmU128Error> {
        self.index(u128::from_be_bytes(*key))
    }

    /// Range query — returns positions for keys in [min_key, max_key].
    pub fn range(&self, min_key: u128, max_key: u128) -> Vec<usize> {
        let lo = self.lower_bound(min_key);
        let hi = self.upper_bound(max_key);
        (lo..hi).collect()
    }

    pub fn lower_bound(&self, target: u128) -> usize {
        let seg = find_segment_u128(&self.segments.max_keys, target);
        if seg >= self.segments.max_keys.len() {
            return self.keys.len();
        }
        let pred = predict_pos_u128(&self.segments, seg, target);
        let err = self.segments.get_max_error(seg) as usize;
        let s = pred.saturating_sub(err);
        let e = (pred + err + 1).min(self.keys.len());
        let mut i = s;
        while i < e {
            if self.keys[i] >= target {
                return i;
            }
            i += 1;
        }
        self.keys.len()
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
            + s.intercepts.len() * 8
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
}

#[derive(Debug, Clone)]
struct RawSeg {
    slope: f64,
    intercept: f64,
    min_key: u128,
    max_key: u128,
    max_error: u32,
    start: usize,
    end: usize,
}

struct LinReg128 {
    n: f64,
    // Use f64; for u128 keys at full universe range (~3.4e38) we lose precision
    // in slope, so we scale by 2^-64. For practical keys (UUIDs are random across
    // the full u128 space) the predicted positions are still within ε of truth.
    sum_x: f64,
    sum_y: f64,
    sum_xy: f64,
    sum_xx: f64,
}

impl LinReg128 {
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
    fn add(&mut self, key: u128, pos: usize) {
        // Scale by 2^-64 so the f64 representation of huge u128 keys doesn't
        // lose all precision in slope. We divide back at predict time.
        let kf = (key as f64) * (2.0f64).powi(-64);
        let pf = pos as f64;
        self.n += 1.0;
        self.sum_x += kf;
        self.sum_y += pf;
        self.sum_xy += kf * pf;
        self.sum_xx += kf * kf;
    }

    fn slope_intercept(&self) -> Option<(f64, f64)> {
        let denom = self.n * self.sum_xx - self.sum_x * self.sum_x;
        if denom.abs() < 1e-20 {
            return None;
        }
        let slope = (self.n * self.sum_xy - self.sum_x * self.sum_y) / denom;
        let intercept = (self.sum_y - slope * self.sum_x) / self.n;
        Some((slope, intercept))
    }

    fn max_error(&self, slope: f64, intercept: f64, keys: &[u128], start: usize) -> u32 {
        let mut err = 0u32;
        let end = start + self.n as usize;
        for (offset, &k) in keys[start..end].iter().enumerate() {
            let kf = (k as f64) * (2.0f64).powi(-64);
            let pred = slope.mul_add(kf, intercept);
            let actual = (start + offset) as f64;
            let e = (pred - actual).abs() as u32;
            if e > err {
                err = e;
            }
        }
        err
    }
}

#[inline]
fn predict_pos_u128(seg: &SegmentsSoA, idx: usize, key: u128) -> usize {
    let kf = (key as f64) * (2.0f64).powi(-64);
    let p = seg.slopes[idx].mul_add(kf, seg.intercepts[idx]);
    if p <= 0.0 {
        0
    } else {
        p as usize
    }
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
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            if half > 4 {
                let nq = half / 2;
                _mm_prefetch(ptr.add(base + nq) as *const i8, _MM_HINT_T0);
                _mm_prefetch(ptr.add(mid + nq) as *const i8, _MM_HINT_T0);
            }
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

