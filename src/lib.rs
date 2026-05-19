//! kira_kv_engine — PtrHash-style MPH + PGM index.
//!
//! - Build once on a set of **unique** keys (bytes/str).
//! - O(1) lookups: key -> unique index in `[0..n)`.
//! - Robust: if a build attempt finds a cycle, we rehash with another salt.

mod aes_hash;
mod block_bloom;
mod build_arena;
mod build_hasher;
mod build_pool;
mod canonical_hash;
mod compressed_pilots;
mod cpu;
mod dynamic_index;
mod elias_fano;
mod hot_tier;
mod hot_tier_dynamic;
mod hugepage;
mod hybrid_engine;
mod hybrid_topology;
pub mod index;
mod mini_chd;
mod mmap_index;
mod mph_backend;
mod pgm;
mod pgm_u128;
mod ptrhash25;
mod simd_hash;
pub use index::{
    BloomExport, GpuExport, Index, IndexBuilder, IndexConfig, IndexError, IndexStats,
};
pub use mph_backend::{BackendKind, BuildConfig as BackendBuildConfig, BuildProfile, MphBackend};

// PGM extensions exposed for advanced users:
pub use dynamic_index::{DynamicConfig, DynamicIndex, StableId};
pub use elias_fano::EliasFano;
pub use hot_tier_dynamic::{DynamicHotTier, SpaceSaving};
pub use hybrid_engine::{HybridBuilder, HybridError, HybridIndex, HybridStorageStats};
pub use pgm::{PgmBuilder, PgmIndex, PgmStats};
pub use pgm_u128::{PgmIndexU128, PgmU128Error};

/// Test-only re-exports of crate-internal items. Hidden from rustdoc; not part
/// of the stable API. Used exclusively by `tests/` for internal-state tests.
#[doc(hidden)]
pub mod __internal {
    pub mod aes_hash {
        pub use crate::aes_hash::{hash_bytes, hash_u64};
    }
    pub use crate::block_bloom::BlockBloom;
    pub use crate::build_arena::BuildArena;
    pub use crate::build_pool::{pool, radix_sort_u64_pairs};
    pub use crate::compressed_pilots::{CompressedPilots, CompressedPilotsV2};
    pub use crate::hugepage::HugepageBuf;
    pub use crate::mini_chd::{MiniChd, MiniChdError};
    pub use crate::mmap_index::{Header, MmapIndex, MmapIndexWriter, SectionKind};
    pub use crate::ptrhash25::{
        BuildConfig, Builder, PtrHash25Error, PtrHash25Mphf, read_ptrhash25, write_ptrhash25,
    };
    pub mod simd_hash {
        pub fn hash_u64_scalar(keys: &[u64], seed: u64, out: &mut [u64]) {
            crate::simd_hash::scalar::hash_u64_scalar(keys, seed, out);
        }
        #[cfg(target_arch = "x86_64")]
        /// # Safety
        /// Caller must be on an AVX2-capable CPU.
        pub unsafe fn hash_u64_avx2(keys: &[u64], seed: u64, out: &mut [u64]) {
            unsafe { crate::simd_hash::x86_64::hash_u64_avx2(keys, seed, out) }
        }
        #[cfg(target_arch = "aarch64")]
        /// # Safety
        /// Caller must be on a NEON-capable CPU.
        pub unsafe fn hash_u64_neon(keys: &[u64], seed: u64, out: &mut [u64]) {
            unsafe { crate::simd_hash::aarch64::hash_u64_neon(keys, seed, out) }
        }
    }
}
