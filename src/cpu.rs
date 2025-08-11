#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_bmi1: bool,
    pub has_bmi2: bool,
    pub has_popcnt: bool,
    pub has_lzcnt: bool,
    pub has_fma: bool,
    pub has_avx512f: bool,
}

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
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                has_avx2: false,
                has_bmi1: false,
                has_bmi2: false,
                has_popcnt: false,
                has_lzcnt: false,
                has_fma: false,
                has_avx512f: false,
            }
        }
    }

    pub fn print_summary(&self) {
        println!("🖥️  CPU Features:");
        println!("  AVX2: {}", if self.has_avx2 { "✅" } else { "❌" });
        println!("  BMI1/BMI2: {}/{}",
                if self.has_bmi1 { "✅" } else { "❌" },
                if self.has_bmi2 { "✅" } else { "❌" });
        println!("  POPCNT: {}", if self.has_popcnt { "✅" } else { "❌" });
        println!("  LZCNT: {}", if self.has_lzcnt { "✅" } else { "❌" });
        println!("  FMA: {}", if self.has_fma { "✅" } else { "❌" });
        println!("  AVX512F: {}", if self.has_avx512f { "✅" } else { "❌" });
    }

    pub fn optimal_config(&self) -> crate::BuildConfig {
        crate::BuildConfig {
            gamma: if self.has_avx2 { 1.25 } else { 1.27 },
            rehash_limit: 16,
            salt: 0xC0FF_EE00_D15E_A5E,
        }
    }

    #[cfg(feature = "pgm")]
    pub fn optimal_hybrid_config(&self) -> crate::HybridConfig {
        crate::HybridConfig {
            mph_config: self.optimal_config(),
            pgm_epsilon: if self.has_avx2 { 32 } else { 64 },
            auto_detect_numeric: true,
        }
    }
}

pub fn detect_features() -> CpuFeatures {
    CpuFeatures::detect()
}