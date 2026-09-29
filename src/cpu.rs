//! CPU feature probe used to pick build-time defaults.

use crate::ptrhash25::BuildConfig;

#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_neon: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self { has_avx2: is_x86_feature_detected!("avx2"), has_neon: false }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self { has_avx2: false, has_neon: std::arch::is_aarch64_feature_detected!("neon") }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self { has_avx2: false, has_neon: false }
        }
    }

    pub fn optimal_config(&self) -> BuildConfig {
        BuildConfig { with_fingerprints: false, ..BuildConfig::default() }
    }

    #[allow(deprecated)]
    pub fn optimal_index_config(&self) -> crate::IndexConfig {
        let has_wide_simd = self.has_avx2 || self.has_neon;
        crate::IndexConfig {
            mph_config: self.optimal_config(),
            pgm_epsilon: if has_wide_simd { 32 } else { 64 },
            auto_detect_numeric: true,
            backend: crate::BackendKind::PtrHash25,
            hot_fraction: 0.15,
            // Parallel build is CPU-bound, not SIMD-bound. Gate only on `parallel` feature.
            enable_parallel_build: cfg!(feature = "parallel"),
            build_fast_profile: true,
            pgm_enable_bloom: false,
            pgm_enable_elias_fano: false,
            pgm_target_lookup_ns: None,
            lean_mph: false,
        }
    }
}

pub fn detect_features() -> CpuFeatures {
    CpuFeatures::detect()
}
