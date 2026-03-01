use serde::{Deserialize, Serialize};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vceqq_u64, vcgeq_u64, vdupq_n_u64, vgetq_lane_u64, vld1q_u64};
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_movemask_epi8,
    _mm256_or_si256, _mm256_set1_epi64x, _mm256_xor_si256,
};
use thiserror::Error;

/// Learned index segment with linear approximation (build-only).
#[derive(Debug, Clone)]
struct SegmentBuild {
    slope: f64,
    intercept: f64,
    min_key: u64,
    max_key: u64,
    max_error: u32,
    start: usize,
    end: usize,
}

#[repr(align(64))]
#[cfg_attr(
    feature = "serde",
    derive(
        Serialize,
        Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize
    )
)]
#[derive(Debug, Clone)]
struct SegmentsSoA {
    slopes: Vec<f64>,
    intercepts: Vec<f64>,
    min_keys: Vec<u64>,
    max_keys: Vec<u64>,
    max_errors: Vec<u32>,
    filters: Vec<u64>,
    starts: Vec<u32>,
    ends: Vec<u32>,
}

/// PGM Index for sorted integer keys with O(1) average lookup
#[cfg_attr(
    feature = "serde",
    derive(
        Serialize,
        Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize
    )
)]
#[derive(Debug, Clone)]
pub struct PgmIndex {
    /// Sorted keys
    keys: Vec<u64>,
    /// Learned segments
    segments: SegmentsSoA,
    /// Epsilon parameter for error tolerance
    epsilon: u32,
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
    /// Build PGM index from sorted unique keys
    pub fn build(mut keys: Vec<u64>, epsilon: u32) -> Result<Self, PgmError> {
        if keys.is_empty() {
            return Err(PgmError::EmptyKeys);
        }

        // Verify sorted and unique
        keys.sort_unstable();
        for window in keys.windows(2) {
            if window[0] >= window[1] {
                return Err(PgmError::UnsortedKeys);
            }
        }

        let segments = Self::build_segments(&keys, epsilon);

        Ok(PgmIndex {
            keys,
            segments,
            epsilon,
        })
    }

    /// Build optimal segments using dynamic programming approach
    fn build_segments(keys: &[u64], epsilon: u32) -> SegmentsSoA {
        let mut segments = Vec::new();
        let mut start = 0;
        let hot_start = keys.len() / 10;
        let hot_end = keys.len().saturating_sub(keys.len() / 10);
        let cold_epsilon = epsilon.saturating_mul(4).max(epsilon);

        while start < keys.len() {
            let mut end = start + 1;
            let mut best_segment = None;

            let local_epsilon = if start >= hot_start && start < hot_end {
                epsilon
            } else {
                cold_epsilon
            };
            // Extend segment as far as possible while maintaining error bound
            while end <= keys.len() {
                if let Some(segment) = Self::fit_segment(keys, start, end, local_epsilon) {
                    best_segment = Some(segment);
                    end += 1;
                } else {
                    break;
                }
            }

            if let Some(segment) = best_segment {
                segments.push(segment);
                start = end - 1;
            } else {
                // Fallback: create minimal segment
                let segment = SegmentBuild {
                    slope: 0.0,
                    intercept: start as f64,
                    min_key: keys[start],
                    max_key: keys[start],
                    max_error: 0,
                    start,
                    end: start + 1,
                };
                segments.push(segment);
                start += 1;
            }
        }

        Self::segments_to_soa(keys, segments)
    }

    /// Fit linear segment to key range with error bound
    fn fit_segment(keys: &[u64], start: usize, end: usize, epsilon: u32) -> Option<SegmentBuild> {
        if end <= start + 1 {
            return None;
        }

        let key_range = &keys[start..end];
        let positions: Vec<usize> = (start..end).collect();

        // Linear regression: y = slope * x + intercept
        let n = key_range.len() as f64;
        let sum_x: f64 = key_range.iter().map(|&k| k as f64).sum();
        let sum_y: f64 = positions.iter().map(|&p| p as f64).sum();
        let sum_xy: f64 = key_range
            .iter()
            .zip(&positions)
            .map(|(&k, &p)| k as f64 * p as f64)
            .sum();
        let sum_x2: f64 = key_range.iter().map(|&k| (k as f64).powi(2)).sum();

        // Avoid division by zero
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return None;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        // Check error bound
        let mut max_error = 0u32;
        for (i, &key) in key_range.iter().enumerate() {
            let predicted = slope * (key as f64) + intercept;
            let actual = (start + i) as f64;
            let error = (predicted - actual).abs() as u32;
            max_error = max_error.max(error);
        }

        if max_error <= epsilon {
            Some(SegmentBuild {
                slope,
                intercept,
                min_key: key_range[0],
                max_key: key_range[key_range.len() - 1],
                max_error,
                start,
                end,
            })
        } else {
            None
        }
    }

    /// Find position of key with O(1) average complexity
    pub fn index(&self, key: u64) -> Result<usize, PgmError> {
        let segment_idx = find_segment_by_max_key(&self.segments.max_keys, key);
        if segment_idx >= self.segments.max_keys.len() {
            return Err(PgmError::KeyNotFound);
        }
        let min_key = self.segments.min_keys[segment_idx];
        let max_key = self.segments.max_keys[segment_idx];
        if key < min_key || key > max_key {
            return Err(PgmError::KeyNotFound);
        }
        let predicted_pos = predict_pos(&self.segments, segment_idx, key);

        // Local search around prediction
        let max_error = self.segments.max_errors[segment_idx] as usize;
        let search_start = predicted_pos.saturating_sub(max_error);
        let search_end = (predicted_pos + max_error + 1).min(self.keys.len());

        let window = search_end - search_start;
        if window <= 256 {
            if let Some(pos) = find_in_range_simd(&self.keys, search_start, search_end, key) {
                return Ok(pos);
            }
            return Err(PgmError::KeyNotFound);
        }

        match self.keys[search_start..search_end].binary_search(&key) {
            Ok(local_pos) => Ok(search_start + local_pos),
            Err(_) => Err(PgmError::KeyNotFound),
        }
    }

    /// Range query: find all positions with keys in [min_key, max_key]
    pub fn range(&self, min_key: u64, max_key: u64) -> Vec<usize> {
        let start_pos = self.lower_bound(min_key);
        let end_pos = self.upper_bound(max_key);
        (start_pos..end_pos).collect()
    }

    pub(crate) fn range_guard(&self, key: u64) -> bool {
        if self.keys.is_empty() {
            return false;
        }
        let idx = find_segment_by_max_key(&self.segments.max_keys, key);
        if idx >= self.segments.max_keys.len() {
            return false;
        }
        let min_key = self.segments.min_keys[idx];
        let max_key = self.segments.max_keys[idx];
        key >= min_key && key <= max_key
    }

    /// Find first position where key >= target
    pub fn lower_bound(&self, target: u64) -> usize {
        let segment_idx = find_segment_by_max_key(&self.segments.max_keys, target);
        if segment_idx >= self.segments.max_keys.len() {
            return self.keys.len();
        }

        let predicted_pos = predict_pos(&self.segments, segment_idx, target);
        let max_error = self.segments.max_errors[segment_idx] as usize;
        let search_start = predicted_pos.saturating_sub(max_error);
        let search_end = (predicted_pos + max_error + 1).min(self.keys.len());

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

        self.keys.len()
    }

    /// Find first position where key > target
    pub fn upper_bound(&self, target: u64) -> usize {
        self.lower_bound(target + 1)
    }

    /// Get statistics about the index
    pub fn stats(&self) -> PgmStats {
        let total_keys = self.keys.len();
        let total_segments = self.segments.max_keys.len();
        let avg_segment_size = if total_segments > 0 {
            total_keys as f64 / total_segments as f64
        } else {
            0.0
        };

        let max_error = self.segments.max_errors.iter().copied().max().unwrap_or(0);

        let memory_usage = std::mem::size_of_val(&self.keys)
            + self.keys.len() * std::mem::size_of::<u64>()
            + std::mem::size_of_val(&self.segments)
            + self.segments.slopes.len() * std::mem::size_of::<f64>()
            + self.segments.intercepts.len() * std::mem::size_of::<f64>()
            + self.segments.min_keys.len() * std::mem::size_of::<u64>()
            + self.segments.max_keys.len() * std::mem::size_of::<u64>()
            + self.segments.max_errors.len() * std::mem::size_of::<u32>()
            + self.segments.filters.len() * std::mem::size_of::<u64>()
            + self.segments.starts.len() * std::mem::size_of::<u32>()
            + self.segments.ends.len() * std::mem::size_of::<u32>();

        PgmStats {
            total_keys,
            total_segments,
            avg_segment_size,
            max_error,
            memory_usage,
            epsilon: self.epsilon,
        }
    }

    #[cfg(feature = "serde")]
    #[allow(dead_code)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, PgmError> {
        let bytes = rkyv::to_bytes::<_, 1024>(self).map_err(|e| PgmError::Serde(e.to_string()))?;
        Ok(bytes.to_vec())
    }

    #[cfg(feature = "serde")]
    #[allow(dead_code)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PgmError> {
        let archived = unsafe { rkyv::archived_root::<Self>(bytes) };
        rkyv::Deserialize::deserialize(archived, &mut rkyv::Infallible)
            .map_err(|e| PgmError::Serde(e.to_string()))
    }

    pub(crate) fn write_to(&self, out: &mut Vec<u8>) {
        write_u32(out, self.epsilon);
        write_u64(out, self.keys.len() as u64);
        for k in &self.keys {
            write_u64(out, *k);
        }
        write_u64(out, self.segments.max_keys.len() as u64);
        for i in 0..self.segments.max_keys.len() {
            write_f64(out, self.segments.slopes[i]);
            write_f64(out, self.segments.intercepts[i]);
            write_u64(out, self.segments.min_keys[i]);
            write_u64(out, self.segments.max_keys[i]);
            write_u32(out, self.segments.max_errors[i]);
            write_u64(out, self.segments.filters[i]);
            write_u32(out, self.segments.starts[i]);
            write_u32(out, self.segments.ends[i]);
        }
    }

    pub(crate) fn read_from(bytes: &[u8], pos: &mut usize) -> Result<Self, PgmError> {
        let mut cur = Cursor {
            buf: bytes,
            pos: *pos,
        };
        let epsilon = cur.read_u32().ok_or(PgmError::CorruptData)?;
        let keys_len = cur.read_u64().ok_or(PgmError::CorruptData)? as usize;
        let mut keys = Vec::with_capacity(keys_len);
        for _ in 0..keys_len {
            keys.push(cur.read_u64().ok_or(PgmError::CorruptData)?);
        }
        let seg_len = cur.read_u64().ok_or(PgmError::CorruptData)? as usize;
        let mut slopes = Vec::with_capacity(seg_len);
        let mut intercepts = Vec::with_capacity(seg_len);
        let mut min_keys = Vec::with_capacity(seg_len);
        let mut max_keys = Vec::with_capacity(seg_len);
        let mut max_errors = Vec::with_capacity(seg_len);
        let mut filters = Vec::with_capacity(seg_len);
        let mut starts = Vec::with_capacity(seg_len);
        let mut ends = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            slopes.push(cur.read_f64().ok_or(PgmError::CorruptData)?);
            intercepts.push(cur.read_f64().ok_or(PgmError::CorruptData)?);
            min_keys.push(cur.read_u64().ok_or(PgmError::CorruptData)?);
            max_keys.push(cur.read_u64().ok_or(PgmError::CorruptData)?);
            max_errors.push(cur.read_u32().ok_or(PgmError::CorruptData)?);
            filters.push(cur.read_u64().ok_or(PgmError::CorruptData)?);
            starts.push(cur.read_u32().ok_or(PgmError::CorruptData)?);
            ends.push(cur.read_u32().ok_or(PgmError::CorruptData)?);
        }
        *pos = cur.pos;
        Ok(PgmIndex {
            keys,
            segments: SegmentsSoA {
                slopes,
                intercepts,
                min_keys,
                max_keys,
                max_errors,
                filters,
                starts,
                ends,
            },
            epsilon,
        })
    }

    fn segments_to_soa(keys: &[u64], segments: Vec<SegmentBuild>) -> SegmentsSoA {
        let mut slopes = Vec::with_capacity(segments.len());
        let mut intercepts = Vec::with_capacity(segments.len());
        let mut min_keys = Vec::with_capacity(segments.len());
        let mut max_keys = Vec::with_capacity(segments.len());
        let mut max_errors = Vec::with_capacity(segments.len());
        let mut filters = Vec::with_capacity(segments.len());
        let mut starts = Vec::with_capacity(segments.len());
        let mut ends = Vec::with_capacity(segments.len());

        for seg in segments {
            slopes.push(seg.slope);
            intercepts.push(seg.intercept);
            min_keys.push(seg.min_key);
            max_keys.push(seg.max_key);
            max_errors.push(seg.max_error);
            starts.push(seg.start as u32);
            ends.push(seg.end as u32);
            let mut filter = 0u64;
            let mut i = seg.start;
            while i < seg.end {
                let bit = filter_bit(keys[i]);
                filter |= bit;
                i += 1;
            }
            filters.push(filter);
        }

        SegmentsSoA {
            slopes,
            intercepts,
            min_keys,
            max_keys,
            max_errors,
            filters,
            starts,
            ends,
        }
    }

    pub(crate) fn filter_allows(&self, key: u64) -> bool {
        let idx = find_segment_by_max_key(&self.segments.max_keys, key);
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
        let idx = find_segment_by_max_key(&self.segments.max_keys, key);
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

    pub(crate) fn keys_len(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn keys(&self) -> &[u64] {
        &self.keys
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
}

#[inline]
fn predict_pos(segments: &SegmentsSoA, idx: usize, key: u64) -> usize {
    let prediction = segments.slopes[idx].mul_add(key as f64, segments.intercepts[idx]);
    prediction.max(0.0) as usize
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

fn find_in_range_simd(keys: &[u64], start: usize, end: usize, target: u64) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            let mut i = start;
            let target_vec = _mm256_set1_epi64x(target as i64);
            while i + 4 <= end {
                let ptr = keys.as_ptr().add(i) as *const __m256i;
                let chunk = _mm256_loadu_si256(ptr);
                let eq = _mm256_cmpeq_epi64(chunk, target_vec);
                let mask = _mm256_movemask_epi8(eq);
                if mask != 0 {
                    for lane in 0..4 {
                        if keys[i + lane] == target {
                            return Some(i + lane);
                        }
                    }
                }
                i += 4;
            }
            while i < end {
                if keys[i] == target {
                    return Some(i);
                }
                i += 1;
            }
            return None;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        if is_aarch64_feature_detected!("neon") {
            let mut i = start;
            let target_vec = vdupq_n_u64(target);
            while i + 2 <= end {
                let ptr = keys.as_ptr().add(i);
                let chunk = vld1q_u64(ptr);
                let eq = vceqq_u64(chunk, target_vec);
                if vgetq_lane_u64(eq, 0) != 0 {
                    return Some(i);
                }
                if vgetq_lane_u64(eq, 1) != 0 {
                    return Some(i + 1);
                }
                i += 2;
            }
            while i < end {
                if keys[i] == target {
                    return Some(i);
                }
                i += 1;
            }
            return None;
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

fn find_first_ge_simd(keys: &[u64], start: usize, end: usize, target: u64) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
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
                let mask = _mm256_movemask_epi8(ge);
                if mask != 0 {
                    for lane in 0..4 {
                        if keys[i + lane] >= target {
                            return Some(i + lane);
                        }
                    }
                }
                i += 4;
            }
            while i < end {
                if keys[i] >= target {
                    return Some(i);
                }
                i += 1;
            }
            return None;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        if is_aarch64_feature_detected!("neon") {
            let mut i = start;
            let target_vec = vdupq_n_u64(target);
            while i + 2 <= end {
                let ptr = keys.as_ptr().add(i);
                let chunk = vld1q_u64(ptr);
                let ge = vcgeq_u64(chunk, target_vec);
                if vgetq_lane_u64(ge, 0) != 0 {
                    return Some(i);
                }
                if vgetq_lane_u64(ge, 1) != 0 {
                    return Some(i + 1);
                }
                i += 2;
            }
            while i < end {
                if keys[i] >= target {
                    return Some(i);
                }
                i += 1;
            }
            return None;
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

fn find_segment_by_max_key(max_keys: &[u64], key: u64) -> usize {
    let len = max_keys.len();
    if len == 0 {
        return 0;
    }
    let mut idx = 0usize;
    let msb = (len - 1).leading_zeros();
    let mut step = if len <= 1 {
        0usize
    } else {
        1usize << (usize::BITS - 1 - msb)
    };
    while step > 0 {
        let next = idx + step;
        if next < len && max_keys[next] < key {
            idx = next;
        }
        step >>= 1;
    }
    if max_keys[idx] < key { idx + 1 } else { idx }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
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

    fn read_f64(&mut self) -> Option<f64> {
        if self.pos + 8 > self.buf.len() {
            return None;
        }
        let mut array = [0u8; 8];
        array.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Some(f64::from_le_bytes(array))
    }
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Statistics about PGM index
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

/// Builder for PGM Index with configuration
pub struct PgmBuilder {
    epsilon: u32,
}

impl PgmBuilder {
    pub fn new() -> Self {
        Self { epsilon: 64 }
    }

    pub fn with_epsilon(mut self, epsilon: u32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn build(self, keys: Vec<u64>) -> Result<PgmIndex, PgmError> {
        PgmIndex::build(keys, self.epsilon)
    }
}

impl Default for PgmBuilder {
    fn default() -> Self {
        Self::new()
    }
}
