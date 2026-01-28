//! kira_kv_engine — BDZ (3-hypergraph peeling) MPH+PGM Index.
//!
//! - Build once on a set of **unique** keys (bytes/str).
//! - O(1) lookups: key -> unique index in `[0..n)`.
//! - Robust: if a build attempt finds a cycle, we rehash with another salt.

mod bdz;
mod build_hasher;
mod cpu;
mod hot_tier;
pub mod hybrid;
mod pgm;
mod remap;
mod simd_hash;
mod xor_filter;
pub use hybrid::{HybridBuilder, HybridConfig, HybridError, HybridIndex, HybridStats};
