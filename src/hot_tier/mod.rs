use crate::block_bloom::BlockBloom;
use crate::ptrhash25::{Builder as MphBuilder, PtrHash25Mphf};
use thiserror::Error;

// The second word is not a valid legacy Bloom length (at least eight words,
// divisible by eight), so no valid legacy payload can share this prefix.
const EXACT_PREFIX: &[u8; 16] = b"KIRA_HOT\x01\0\0\0\0\0\0\0";

/// A checked hot-tier construction failure.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum HotTierError {
    #[error("hot-tier key and index lengths differ")]
    LengthMismatch,
    #[error("hot-tier key set is empty")]
    EmptyKeys,
    #[error("duplicate hot-tier key")]
    DuplicateKey,
    #[error("u32::MAX is reserved for an unoccupied hot-tier slot")]
    ReservedIndex,
    #[error("hot-tier key count exceeds u32 addressing")]
    TooManyKeys,
    #[error("hot-tier pilot search could not place all keys")]
    Unresolvable,
}

#[derive(Debug)]
pub struct HotTierIndex {
    filter: BlockBloom,
    mph: PtrHash25Mphf,
    fingerprints: Box<[u16]>,
    indices: Box<[u32]>,
    /// None for legacy payloads, which did not retain the original keys.
    keys: Option<Box<[u64]>>,
}

impl HotTierIndex {
    pub fn build_from_u64(keys: &[u64], indices: &[u32], seed: u64) -> Option<Self> {
        Self::try_build_from_u64(keys, indices, seed).ok()
    }

    /// Build an exact hot tier; the legacy `Option` constructor wraps this method.
    pub fn try_build_from_u64(
        keys: &[u64],
        indices: &[u32],
        seed: u64,
    ) -> Result<Self, HotTierError> {
        if keys.len() != indices.len() {
            return Err(HotTierError::LengthMismatch);
        }
        if keys.is_empty() {
            return Err(HotTierError::EmptyKeys);
        }
        if keys.len() > u32::MAX as usize {
            return Err(HotTierError::TooManyKeys);
        }
        if indices.contains(&u32::MAX) {
            return Err(HotTierError::ReservedIndex);
        }
        let filter = BlockBloom::build_from_u64(keys, 0xC1B5_4A32_D192_ED03);
        let cfg = crate::ptrhash25::BuildConfig {
            lambda: crate::ptrhash25::DEFAULT_LAMBDA,
            alpha: crate::ptrhash25::DEFAULT_ALPHA,
            max_rehash: 16,
            with_fingerprints: false,
            seed,
            use_aes_hash: false,
        };
        let mph = MphBuilder::new().with_config(cfg).build(keys).map_err(|error| match error {
            crate::ptrhash25::PtrHash25Error::DuplicateKey => HotTierError::DuplicateKey,
            crate::ptrhash25::PtrHash25Error::Unresolvable => HotTierError::Unresolvable,
        })?;
        let slot_space = mph.n as usize;
        let fingerprints = build_fingerprints_u64(&mph, keys, slot_space).into_boxed_slice();
        // Map slot → original index. Slot space may be larger than keys.len() due
        // to PtrHash25's 1.10× padding; un-used slots stay at u32::MAX.
        let mut indices_vec = vec![u32::MAX; slot_space];
        let mut slot_keys = vec![0; slot_space];
        for (i, &k) in keys.iter().enumerate() {
            let slot = mph.index_u64(k) as usize;
            indices_vec[slot] = indices[i];
            slot_keys[slot] = k;
        }
        Ok(Self {
            filter,
            mph,
            fingerprints,
            indices: indices_vec.into_boxed_slice(),
            keys: Some(slot_keys.into_boxed_slice()),
        })
    }

    /// New builds compare original u64 keys. Legacy loaded tiers retain their
    /// probabilistic behavior because their files contain only fingerprints.
    pub fn membership_mode(&self) -> crate::MembershipMode {
        if self.keys.is_some() {
            crate::MembershipMode::Exact
        } else {
            crate::MembershipMode::Probabilistic
        }
    }

    /// Return only an exact hit. Legacy payloads conservatively return `None` so
    /// a cache can fall back to its authoritative backing index.
    pub fn lookup_exact_u64(&self, key: u64) -> Option<u32> {
        self.keys.as_ref()?;
        self.lookup_u64(key)
    }

    #[inline]
    pub fn lookup_u64(&self, key: u64) -> Option<u32> {
        if !self.filter.contains_u64(key) {
            return None;
        }
        let slot = self.mph.index_u64(key) as usize;
        if self.keys.as_ref().is_some_and(|keys| keys[slot] != key) {
            return None;
        }
        let fp = fingerprint16(hash_u64_det(key));
        if unsafe { *self.fingerprints.get_unchecked(slot) == fp } {
            let idx = unsafe { *self.indices.get_unchecked(slot) };
            if idx == u32::MAX { None } else { Some(idx) }
        } else {
            None
        }
    }

    pub fn memory_usage(&self) -> usize {
        let mph_mem = self.mph.memory_usage();
        let filter_mem = self.filter.memory_usage();
        let fp_mem = self.fingerprints.len() * std::mem::size_of::<u16>();
        let idx_mem = self.indices.len() * std::mem::size_of::<u32>();
        let key_mem = self.keys.as_ref().map_or(0, |keys| keys.len() * 8);
        mph_mem + filter_mem + fp_mem + idx_mem + key_mem
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        let Some(keys) = &self.keys else {
            self.write_legacy_payload(out);
            return;
        };
        let start = out.len();
        out.extend_from_slice(EXACT_PREFIX);
        let length_offset = out.len();
        write_u64(out, 0);
        let payload_start = out.len();
        self.write_legacy_payload(out);
        let payload_length = (out.len() - payload_start) as u64;
        out[length_offset..length_offset + 8].copy_from_slice(&payload_length.to_le_bytes());
        write_u64(out, keys.len() as u64);
        crate::wire::extend_le(out, keys);
        let checksum = crate::checksum::checksum(&out[start..]);
        write_u64(out, checksum);
    }

    fn write_legacy_payload(&self, out: &mut Vec<u8>) {
        self.filter.write_to(out);
        crate::ptrhash25::write_ptrhash25(&self.mph, out);
        write_fingerprints(out, &self.fingerprints);
        write_u64(out, self.indices.len() as u64);
        crate::wire::extend_le(out, &self.indices);
    }

    pub fn read_from(bytes: &[u8], pos: &mut usize) -> Option<Self> {
        let remaining = bytes.get(*pos..)?;
        let mut consumed = 0;
        let tier = if remaining.starts_with(EXACT_PREFIX) {
            Self::read_exact_frame(remaining, &mut consumed)?
        } else {
            Self::read_legacy_payload(remaining, &mut consumed)?
        };
        *pos = pos.checked_add(consumed)?;
        Some(tier)
    }

    fn read_exact_frame(bytes: &[u8], pos: &mut usize) -> Option<Self> {
        let mut cur = LocalCursor { buf: bytes, pos: EXACT_PREFIX.len() };
        let payload_length = usize::try_from(cur.read_u64()?).ok()?;
        let payload = cur.read_bytes(payload_length)?;
        let key_count = usize::try_from(cur.read_u64()?).ok()?;
        let key_bytes = cur.read_bytes(key_count.checked_mul(8)?)?;
        let checksum_offset = cur.pos;
        let checksum = cur.read_u64()?;
        if crate::checksum::checksum(&bytes[..checksum_offset]) != checksum {
            return None;
        }
        let mut payload_pos = 0;
        let mut tier = Self::read_legacy_payload(payload, &mut payload_pos)?;
        if payload_pos != payload.len() || key_count != tier.mph.slot_capacity() {
            return None;
        }
        let keys: Vec<u64> = crate::wire::read_le(key_bytes, key_count)?;
        for (slot, &key) in keys.iter().enumerate() {
            if tier.indices[slot] == u32::MAX {
                continue;
            }
            if tier.mph.index_u64(key) as usize != slot
                || tier.fingerprints[slot] != fingerprint16(hash_u64_det(key))
                || !tier.filter.contains_u64(key)
            {
                return None;
            }
        }
        tier.keys = Some(keys.into_boxed_slice());
        *pos = cur.pos;
        Some(tier)
    }

    fn read_legacy_payload(bytes: &[u8], pos: &mut usize) -> Option<Self> {
        let mut read_pos = *pos;
        let filter = BlockBloom::read_from(bytes, &mut read_pos)?;
        let mph = crate::ptrhash25::read_ptrhash25(bytes, &mut read_pos)?;
        let mut cur = LocalCursor { buf: bytes, pos: read_pos };
        let fingerprints = read_fingerprints(&mut cur)?;
        let len = usize::try_from(cur.read_u64()?).ok()?;
        let indices: Vec<u32> = crate::wire::read_le_at(bytes, &mut cur.pos, len)?;
        // Both side tables are indexed by the MPH slot without bounds checks.
        if fingerprints.len() != mph.slot_capacity() || indices.len() != mph.slot_capacity() {
            return None;
        }
        *pos = cur.pos;
        Some(Self {
            filter,
            mph,
            fingerprints: fingerprints.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
            keys: None,
        })
    }
}

/// Minimal byte-cursor used during deserialization. Replaces the old
/// `xor_filter::Cursor` after the xor_filter module was retired.
struct LocalCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> LocalCursor<'a> {
    fn read_bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(len)?;
        let bytes = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(bytes)
    }

    fn read_u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.read_bytes(8)?.try_into().ok()?))
    }
}

fn build_fingerprints_u64(mph: &PtrHash25Mphf, keys: &[u64], slot_space: usize) -> Vec<u16> {
    let mut fps = vec![0u16; slot_space];
    for &key in keys {
        let idx = mph.index_u64(key) as usize;
        let fp = fingerprint16(hash_u64(key));
        fps[idx] = fp;
    }
    fps
}

#[inline]
fn hash_u64(key: u64) -> u64 {
    splitmix64(key ^ 0xA24B_1F6F_1234_5678)
}

#[inline]
fn hash_u64_det(key: u64) -> u64 {
    splitmix64(key ^ 0xA24B_1F6F_1234_5678)
}

#[inline]
fn fingerprint16(hash: u64) -> u16 {
    (hash & 0xFFFF) as u16
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_fingerprints(out: &mut Vec<u8>, fps: &[u16]) {
    write_u64(out, fps.len() as u64);
    crate::wire::extend_le(out, fps);
}

fn read_fingerprints(cursor: &mut LocalCursor<'_>) -> Option<Vec<u16>> {
    let len = usize::try_from(cursor.read_u64()?).ok()?;
    crate::wire::read_le_at(cursor.buf, &mut cursor.pos, len)
}
