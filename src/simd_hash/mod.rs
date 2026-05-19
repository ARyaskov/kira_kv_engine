#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

pub fn hash_u64(keys: &[u64], seed: u64, out: &mut [u64]) {
    assert_eq!(keys.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { x86_64::hash_u64_avx2(keys, seed, out) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { aarch64::hash_u64_neon(keys, seed, out) };
        }
    }
    scalar::hash_u64_scalar(keys, seed, out);
}

#[inline]
pub fn hash_u64_one(key: u64, seed: u64) -> u64 {
    scalar::hash_u64_one(key, seed)
}

/// Alternative AES-NI based hash for single u64 keys. Different formula from
/// `hash_u64_one` — DO NOT mix within one MPHF (build/lookup must use the same
/// hash function). Useful when you want stronger distribution against adversarial
/// inputs, at the cost of ~2-4 cycles extra per call vs splitmix.
#[inline]
#[allow(dead_code)]
pub fn hash_u64_aes(key: u64, seed: u64) -> u64 {
    crate::aes_hash::hash_u64(key, seed)
}

pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
