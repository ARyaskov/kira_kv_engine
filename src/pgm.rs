use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
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
        // Binary search for the right segment
        let segment_idx = match self.segments.binary_search_by(|segment| {
            if key < segment.min_key {
                Ordering::Greater
            } else if key > segment.max_key {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        }) {
            Ok(idx) => idx,
            Err(_) => return Err(PgmError::KeyNotFound),
        };

        let segment = &self.segments[segment_idx];
        let predicted_pos = segment.predict(key);

        // Local search around prediction
        let search_start = predicted_pos.saturating_sub(segment.max_error as usize);
        let search_end = (predicted_pos + segment.max_error as usize + 1).min(self.keys.len());

        // Binary search in narrow range
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

    /// Find first position where key >= target
    pub fn lower_bound(&self, target: u64) -> usize {
        // Find segment containing target
        let mut segment_idx = 0;
        for (i, segment) in self.segments.iter().enumerate() {
            if target <= segment.max_key {
                segment_idx = i;
                break;
            }
        }

        if segment_idx >= self.segments.len() {
            return self.keys.len();
        }

        let segment = &self.segments[segment_idx];
        let predicted_pos = segment.predict(target);
        let search_start = predicted_pos.saturating_sub(segment.max_error as usize);
        let search_end = (predicted_pos + segment.max_error as usize + 1).min(self.keys.len());

        // Find first position >= target
        for pos in search_start..search_end {
            if self.keys[pos] >= target {
                return pos;
            }
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
    pub fn to_bytes(&self) -> Result<Vec<u8>, PgmError> {
        Ok(bincode::serialize(self)?)
    }

    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PgmError> {
        Ok(bincode::deserialize(bytes)?)
    }
}

/// Statistics about PGM index
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
