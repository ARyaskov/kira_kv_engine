use crate::bdz::{BuildConfig as MphConfig, Builder as MphBuilder, MphError, Mphf};
use crate::pgm::{PgmBuilder, PgmError, PgmIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Hybrid index: MPH for strings, PGM for numbers
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct HybridIndex {
    /// MPH index for non-numeric keys
    mph_index: Option<Mphf>,
    /// PGM index for numeric keys
    pgm_index: Option<PgmIndex>,
    /// Mapping of numeric keys to positions
    numeric_keys: Vec<u64>,
    /// Mapping of string keys to positions
    string_keys: Vec<Vec<u8>>,
    /// Starting offset for PGM keys
    pgm_offset: usize,
}

#[derive(Debug, Error)]
pub enum HybridError {
    #[error("MPH error: {0}")]
    Mph(#[from] MphError),
    #[error("PGM error: {0}")]
    Pgm(#[from] PgmError),
    #[error("key not found")]
    KeyNotFound,
    #[error("invalid key format")]
    InvalidKey,
    #[cfg(feature = "serde")]
    #[error("serialization error: {0}")]
    Serde(#[from] Box<bincode::ErrorKind>),
}

/// Configuration for hybrid index
#[derive(Debug, Clone)]
pub struct HybridConfig {
    pub mph_config: MphConfig,
    pub pgm_epsilon: u32,
    pub auto_detect_numeric: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            mph_config: MphConfig::default(),
            pgm_epsilon: 64,
            auto_detect_numeric: true,
        }
    }
}

/// Hybrid index statistics
#[derive(Debug)]
pub struct HybridStats {
    pub total_keys: usize,
    pub numeric_keys: usize,
    pub string_keys: usize,
    pub mph_memory: usize,
    pub pgm_memory: usize,
    pub total_memory: usize,
    pub compression_ratio: f64,
}

impl HybridIndex {
    /// Build hybrid index from keys
    pub fn build<K>(keys: Vec<K>, config: HybridConfig) -> Result<Self, HybridError>
    where
        K: AsRef<[u8]>,
    {
        let (numeric_keys, string_keys) = Self::partition_keys(keys, config.auto_detect_numeric);

        let mut hybrid = HybridIndex {
            mph_index: None,
            pgm_index: None,
            numeric_keys: numeric_keys.clone(),
            string_keys: string_keys.clone(),
            pgm_offset: string_keys.len(),
        };

        // Build MPH for string keys
        if !string_keys.is_empty() {
            let mph = MphBuilder::new()
                .with_config(config.mph_config)
                .build(string_keys.iter().map(|k| k.as_slice()))?;
            hybrid.mph_index = Some(mph);
        }

        // Build PGM for numeric keys
        if !numeric_keys.is_empty() {
            let pgm = PgmBuilder::new()
                .with_epsilon(config.pgm_epsilon)
                .build(numeric_keys)?;
            hybrid.pgm_index = Some(pgm);
        }

        Ok(hybrid)
    }

    /// Partition keys into numeric and string types
    fn partition_keys<K>(keys: Vec<K>, auto_detect: bool) -> (Vec<u64>, Vec<Vec<u8>>)
    where
        K: AsRef<[u8]>,
    {
        let mut numeric_keys = Vec::new();
        let mut string_keys = Vec::new();

        for key in keys {
            let key_bytes = key.as_ref();

            if auto_detect && key_bytes.len() == 8 {
                // Try to interpret as u64
                if let Some(num) = Self::try_parse_u64(key_bytes) {
                    numeric_keys.push(num);
                    continue;
                }
            }

            string_keys.push(key_bytes.to_vec());
        }

        // Sort numeric keys for PGM
        numeric_keys.sort_unstable();
        numeric_keys.dedup();

        (numeric_keys, string_keys)
    }

    /// Try to parse bytes as u64 (little endian)
    fn try_parse_u64(bytes: &[u8]) -> Option<u64> {
        if bytes.len() != 8 {
            return None;
        }

        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Some(u64::from_le_bytes(array))
    }

    /// Search for key index
    pub fn index(&self, key: &[u8]) -> Result<usize, HybridError> {
        // First try to find as numeric key
        if key.len() == 8 {
            if let Some(num) = Self::try_parse_u64(key) {
                if let Some(ref pgm) = self.pgm_index {
                    if let Ok(pos) = pgm.index(num) {
                        return Ok(self.pgm_offset + pos);
                    }
                }
            }
        }

        // If not found as number, search as string
        if let Some(ref mph) = self.mph_index {
            let mph_index = mph.index(key);
            if mph_index < self.string_keys.len() as u64 {
                return Ok(mph_index as usize);
            }
        }

        Err(HybridError::KeyNotFound)
    }

    /// Search for string key
    pub fn index_str(&self, key: &str) -> Result<usize, HybridError> {
        self.index(key.as_bytes())
    }

    /// Search for numeric key
    pub fn index_u64(&self, key: u64) -> Result<usize, HybridError> {
        if let Some(ref pgm) = self.pgm_index {
            let pos = pgm.index(key)?;
            Ok(self.pgm_offset + pos)
        } else {
            Err(HybridError::KeyNotFound)
        }
    }

    /// Range query for numeric keys
    pub fn range_u64(&self, min_key: u64, max_key: u64) -> Vec<usize> {
        if let Some(ref pgm) = self.pgm_index {
            pgm.range(min_key, max_key)
                .into_iter()
                .map(|pos| self.pgm_offset + pos)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Batch lookup for multiple keys
    #[cfg(feature = "simd")]
    pub fn index_batch(&self, keys: &[&[u8]]) -> Vec<Option<usize>> {
        keys.iter().map(|&key| self.index(key).ok()).collect()
    }

    /// Index statistics
    pub fn stats(&self) -> HybridStats {
        let mph_memory = if let Some(ref mph) = self.mph_index {
            std::mem::size_of_val(mph) + mph.g.len() * std::mem::size_of::<u32>()
        } else {
            0
        };

        let pgm_memory = if let Some(ref pgm) = self.pgm_index {
            pgm.stats().memory_usage
        } else {
            0
        };

        let total_keys = self.numeric_keys.len() + self.string_keys.len();
        let total_memory = mph_memory
            + pgm_memory
            + std::mem::size_of_val(&self.numeric_keys)
            + std::mem::size_of_val(&self.string_keys);

        let compression_ratio = if total_memory > 0 {
            (total_keys * 32) as f64 / total_memory as f64 // vs approximate HashMap
        } else {
            0.0
        };

        HybridStats {
            total_keys,
            numeric_keys: self.numeric_keys.len(),
            string_keys: self.string_keys.len(),
            mph_memory,
            pgm_memory,
            total_memory,
            compression_ratio,
        }
    }

    /// Print detailed statistics
    pub fn print_detailed_stats(&self) {
        let stats = self.stats();

        println!("🔥 Hybrid Index Statistics:");
        println!("  Total keys: {}", stats.total_keys);
        println!(
            "  Numeric keys (PGM): {} ({:.1}%)",
            stats.numeric_keys,
            stats.numeric_keys as f64 / stats.total_keys as f64 * 100.0
        );
        println!(
            "  String keys (MPH): {} ({:.1}%)",
            stats.string_keys,
            stats.string_keys as f64 / stats.total_keys as f64 * 100.0
        );

        println!("  Memory breakdown:");
        println!(
            "    MPH index: {:.2} MB",
            stats.mph_memory as f64 / 1_048_576.0
        );
        println!(
            "    PGM index: {:.2} MB",
            stats.pgm_memory as f64 / 1_048_576.0
        );
        println!(
            "    Total: {:.2} MB",
            stats.total_memory as f64 / 1_048_576.0
        );

        println!("  Efficiency:");
        println!(
            "    Compression ratio: {:.1}x vs HashMap",
            stats.compression_ratio
        );
        println!(
            "    Bytes per key: {:.1}",
            stats.total_memory as f64 / stats.total_keys as f64
        );

        // PGM details
        if let Some(ref pgm) = self.pgm_index {
            println!("  PGM details:");
            let pgm_stats = pgm.stats();
            pgm_stats.print_summary();
        }
    }

    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, HybridError> {
        Ok(bincode::serialize(self)?)
    }

    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HybridError> {
        Ok(bincode::deserialize(bytes)?)
    }
}

/// Builder for hybrid index
pub struct HybridBuilder {
    config: HybridConfig,
}

impl HybridBuilder {
    pub fn new() -> Self {
        Self {
            config: HybridConfig::default(),
        }
    }

    pub fn with_config(mut self, config: HybridConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_mph_config(mut self, mph_config: MphConfig) -> Self {
        self.config.mph_config = mph_config;
        self
    }

    pub fn with_pgm_epsilon(mut self, epsilon: u32) -> Self {
        self.config.pgm_epsilon = epsilon;
        self
    }

    pub fn auto_detect_numeric(mut self, enabled: bool) -> Self {
        self.config.auto_detect_numeric = enabled;
        self
    }

    pub fn build<K>(self, keys: Vec<K>) -> Result<HybridIndex, HybridError>
    where
        K: AsRef<[u8]>,
    {
        HybridIndex::build(keys, self.config)
    }
}

impl Default for HybridBuilder {
    fn default() -> Self {
        Self::new()
    }
}
