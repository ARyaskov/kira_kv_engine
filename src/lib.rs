//! kira_kv_engine — key→id index toolkit: a minimal perfect hash (`Index`, PtrHash
//! 2025 with eviction), an LSM of such hashes (`DynamicIndex`), a PGM-bucketed
//! byte-key engine (`HybridIndex`) and learned indexes with range queries
//! (`PgmIndex` for u64, `PgmIndexU128` for 16-byte keys).
//!
//! - Build once on a set of **unique** keys (bytes/str); duplicates are an error.
//! - O(1) lookups: key → position in `[0..Index::slot_capacity())`, which equals
//!   `[0..len())` for indexes built by this version (minimal perfect hashing);
//!   indexes written by 0.6 keep their 1.1× padded range. Size parallel side
//!   arrays to `slot_capacity()`.
//! - Empty input is accepted by every engine and yields an always-miss instance.
//! - Every engine serializes into a checksummed, validated, platform-independent
//!   byte form; hashing is bit-identical on x86_64, aarch64 and other targets.
//!
//! See `README.md` for choosing an engine and `API.md` for the full surface.

mod aes_hash;
mod block_bloom;
mod build_pool;
mod canonical_hash;
mod checksum;
mod cpu;
mod dynamic_index;
mod elias_fano;
mod hot_tier;
mod hot_tier_dynamic;
mod hugepage;
mod hybrid_engine;
#[cfg(feature = "parallel")]
mod hybrid_topology;
pub mod index;
mod mini_chd;
mod mmap_index;
mod mph_backend;
mod pgm;
mod prefetch;
mod pgm_u128;
mod ptrhash25;
mod simd_hash;
mod wire;
pub use index::{
    BloomExport, GpuExport, GpuPart, Index, IndexBuilder, IndexConfig, IndexError, IndexStats,
};
#[allow(deprecated)]
pub use mph_backend::{BackendKind, BuildConfig as BackendBuildConfig, BuildProfile, MphBackend};

// PGM extensions exposed for advanced users:
pub use dynamic_index::{DynamicConfig, DynamicIndex, StableId};
pub use elias_fano::EliasFano;
pub use hot_tier::HotTierIndex;
pub use hot_tier_dynamic::{DynamicHotTier, SpaceSaving};
pub use hugepage::hugepages_available;
pub use hybrid_engine::{HybridBuilder, HybridError, HybridIndex, HybridStorageStats};
pub use pgm::{PgmBuilder, PgmIndex, PgmStats};
pub use pgm_u128::{PgmIndexU128, PgmU128Error};

/// Test-only re-exports of crate-internal items. Hidden from rustdoc; not part
/// of the stable API. Used exclusively by `tests/` for internal-state tests.
#[doc(hidden)]
pub mod __internal {
    pub mod aes_hash {
        pub use crate::aes_hash::{aes_round_hw, aes_round_soft, hash_bytes, hash_u64, hw_available};
    }
    pub use crate::block_bloom::BlockBloom;
    #[cfg(feature = "parallel")]
    pub use crate::build_pool::pool;
    pub use crate::build_pool::radix_sort_u64_pairs;
    pub use crate::hugepage::HugepageBuf;
    pub use crate::mini_chd::{MiniChd, MiniChdError};
    pub use crate::mmap_index::{Header, MmapIndex, MmapIndexWriter, SectionKind};
    pub use crate::ptrhash25::{
        BuildConfig, BuildOutputs, Builder, PART_TARGET_KEYS, PartInfo, Partitioned,
        PtrHash25Error, PtrHash25Mphf, build_partitioned, build_partitioned_with, partition_keys,
        read_ptrhash25, write_ptrhash25,
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
