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

/// Learned index segment with linear approximation
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
struct Segment {
    /// Linear function: y = slope * x + intercept
    slope: f64,
    intercept: f64,
    /// Key range this segment covers
    min_key: u64,
    max_key: u64,
    /// Maximum prediction error in this segment
    max_error: u32,
}

impl Segment {
    /// Predict position for a key using linear function
    #[inline]
    fn predict(&self, key: u64) -> usize {
        let prediction = self.slope * (key as f64) + self.intercept;
        prediction.max(0.0) as usize
    }
}

/// PGM Index for sorted integer keys with O(1) average lookup
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct PgmIndex {
    /// Sorted keys
    keys: Vec<u64>,
    /// Learned segments
    segments: Vec<Segment>,
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
    Serde(#[from] Box<bincode::ErrorKind>),
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
    fn build_segments(keys: &[u64], epsilon: u32) -> Vec<Segment> {
        let mut segments = Vec::new();
        let mut start = 0;

        while start < keys.len() {
            let mut end = start + 1;
            let mut best_segment = None;

            // Extend segment as far as possible while maintaining error bound
            while end <= keys.len() {
                if let Some(segment) = Self::fit_segment(keys, start, end, epsilon) {
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
                let segment = Segment {
                    slope: 0.0,
                    intercept: start as f64,
                    min_key: keys[start],
                    max_key: keys[start],
                    max_error: 0,
                };
                segments.push(segment);
                start += 1;
            }
        }

        segments
    }

    /// Fit linear segment to key range with error bound
    fn fit_segment(keys: &[u64], start: usize, end: usize, epsilon: u32) -> Option<Segment> {
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
            Some(Segment {
                slope,
                intercept,
                min_key: key_range[0],
                max_key: key_range[key_range.len() - 1],
                max_error,
            })
        } else {
            None
        }
    }

    /// Find position of key with O(1) average complexity
    pub fn index(&self, key: u64) -> Result<usize, PgmError> {
        let segment_idx = find_segment_by_max_key(&self.segments, key);
        if segment_idx >= self.segments.len() {
            return Err(PgmError::KeyNotFound);
        }
        let segment = &self.segments[segment_idx];
        if key < segment.min_key || key > segment.max_key {
            return Err(PgmError::KeyNotFound);
        }
        let predicted_pos = segment.predict(key);

        // Local search around prediction
        let search_start = predicted_pos.saturating_sub(segment.max_error as usize);
        let search_end = (predicted_pos + segment.max_error as usize + 1).min(self.keys.len());

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
        let idx = find_segment_by_max_key(&self.segments, key);
        if idx >= self.segments.len() {
            return false;
        }
        let seg = &self.segments[idx];
        key >= seg.min_key && key <= seg.max_key
    }

    /// Find first position where key >= target
    pub fn lower_bound(&self, target: u64) -> usize {
        let segment_idx = find_segment_by_max_key(&self.segments, target);
        if segment_idx >= self.segments.len() {
            return self.keys.len();
        }

        let segment = &self.segments[segment_idx];
        let predicted_pos = segment.predict(target);
        let search_start = predicted_pos.saturating_sub(segment.max_error as usize);
        let search_end = (predicted_pos + segment.max_error as usize + 1).min(self.keys.len());

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
        let total_segments = self.segments.len();
        let avg_segment_size = if total_segments > 0 {
            total_keys as f64 / total_segments as f64
        } else {
            0.0
        };

        let max_error = self.segments.iter().map(|s| s.max_error).max().unwrap_or(0);

        let memory_usage = std::mem::size_of_val(&self.keys)
            + self.keys.len() * std::mem::size_of::<u64>()
            + std::mem::size_of_val(&self.segments)
            + self.segments.len() * std::mem::size_of::<Segment>();

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
        Ok(bincode::serialize(self)?)
    }

    #[cfg(feature = "serde")]
    #[allow(dead_code)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PgmError> {
        Ok(bincode::deserialize(bytes)?)
    }

    pub(crate) fn write_to(&self, out: &mut Vec<u8>) {
        write_u32(out, self.epsilon);
        write_u64(out, self.keys.len() as u64);
        for k in &self.keys {
            write_u64(out, *k);
        }
        write_u64(out, self.segments.len() as u64);
        for s in &self.segments {
            write_f64(out, s.slope);
            write_f64(out, s.intercept);
            write_u64(out, s.min_key);
            write_u64(out, s.max_key);
            write_u32(out, s.max_error);
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
        let mut segments = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            let slope = cur.read_f64().ok_or(PgmError::CorruptData)?;
            let intercept = cur.read_f64().ok_or(PgmError::CorruptData)?;
            let min_key = cur.read_u64().ok_or(PgmError::CorruptData)?;
            let max_key = cur.read_u64().ok_or(PgmError::CorruptData)?;
            let max_error = cur.read_u32().ok_or(PgmError::CorruptData)?;
            segments.push(Segment {
                slope,
                intercept,
                min_key,
                max_key,
                max_error,
            });
        }
        *pos = cur.pos;
        Ok(PgmIndex {
            keys,
            segments,
            epsilon,
        })
    }
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

fn find_segment_by_max_key(segments: &[Segment], key: u64) -> usize {
    let len = segments.len();
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
        if next < len && segments[next].max_key < key {
            idx = next;
        }
        step >>= 1;
    }
    if segments[idx].max_key < key {
        idx + 1
    } else {
        idx
    }
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
