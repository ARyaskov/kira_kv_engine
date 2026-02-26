//! kira_kv_engine — PtrHash-style MPH + PGM index.
//!
//! - Build once on a set of **unique** keys (bytes/str).
//! - O(1) lookups: key -> unique index in `[0..n)`.
//! - Robust: if a build attempt finds a cycle, we rehash with another salt.

mod build_hasher;
mod canonical_hash;
mod cpu;
mod hot_tier;
pub mod index;
mod mph_backend;
mod pgm;
mod ptrhash;
mod remap;
mod simd_hash;
mod xor_filter;
pub use index::{Index, IndexBuilder, IndexConfig, IndexError, IndexStats};
pub use mph_backend::{BackendKind, BuildConfig as BackendBuildConfig, BuildProfile, MphBackend};
