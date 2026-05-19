#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_bmi1: bool,
    pub has_bmi2: bool,
    pub has_popcnt: bool,
    pub has_lzcnt: bool,
    pub has_fma: bool,
    pub has_avx512f: bool,
    pub has_neon: bool,
}

use crate::BackendKind;
use crate::ptrhash25::BuildConfig;
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

#[allow(dead_code)]
impl CpuFeatures {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_bmi1: is_x86_feature_detected!("bmi1"),
                has_bmi2: is_x86_feature_detected!("bmi2"),
                has_popcnt: is_x86_feature_detected!("popcnt"),
                has_lzcnt: is_x86_feature_detected!("lzcnt"),
                has_fma: is_x86_feature_detected!("fma"),
                has_avx512f: is_x86_feature_detected!("avx512f"),
                has_neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                has_avx2: false,
                has_bmi1: false,
                has_bmi2: false,
                has_popcnt: false,
                has_lzcnt: false,
                has_fma: false,
                has_avx512f: false,
                has_neon: is_aarch64_feature_detected!("neon"),
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                has_avx2: false,
                has_bmi1: false,
                has_bmi2: false,
                has_popcnt: false,
                has_lzcnt: false,
                has_fma: false,
                has_avx512f: false,
                has_neon: false,
            }
        }
    }

    pub fn print_summary(&self) {
        println!("🖥️  CPU Features:");
        println!("  AVX2: {}", if self.has_avx2 { "✅" } else { "❌" });
        println!(
            "  BMI1/BMI2: {}/{}",
            if self.has_bmi1 { "✅" } else { "❌" },
            if self.has_bmi2 { "✅" } else { "❌" }
        );
        println!("  POPCNT: {}", if self.has_popcnt { "✅" } else { "❌" });
        println!("  LZCNT: {}", if self.has_lzcnt { "✅" } else { "❌" });
        println!("  FMA: {}", if self.has_fma { "✅" } else { "❌" });
        println!("  AVX512F: {}", if self.has_avx512f { "✅" } else { "❌" });
        println!("  NEON: {}", if self.has_neon { "✅" } else { "❌" });
    }

    pub fn optimal_config(&self) -> BuildConfig {
        BuildConfig {
            gamma: 0.5,
            max_rehash: 16,
            with_fingerprints: false,
            seed: 0xC0FF_EE00_D15E_A5E,
            use_aes_hash: false,
        }
    }

    pub fn optimal_index_config(&self) -> crate::IndexConfig {
        let has_wide_simd = self.has_avx2 || self.has_neon;
        crate::IndexConfig {
            mph_config: self.optimal_config(),
            pgm_epsilon: if has_wide_simd { 32 } else { 64 },
            auto_detect_numeric: true,
            backend: BackendKind::PtrHash25,
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
