//! kira_kv_engine — BDZ (3-hypergraph peeling) MPH+PGM Index.
//!
//! - Build once on a set of **unique** keys (bytes/str).
//! - O(1) lookups: key -> unique index in `[0..n)`.
//! - Robust: if a build attempt finds a cycle, we rehash with another salt.

mod bdz;
pub use bdz::{BuildConfig, Builder, MphError, Mphf};

pub mod hybrid;
pub mod pgm;
pub use hybrid::{HybridBuilder, HybridConfig};
pub use pgm::PgmBuilder;
