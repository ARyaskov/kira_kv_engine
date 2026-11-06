use crate::hugepage::HugeVec;
use std::io::Write;
use std::ptr::NonNull;

/// Membership fingerprint width. None of these widths provide exact membership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FingerprintBits {
    #[default]
    Bits16,
    Bits24,
    Bits32,
}
impl FingerprintBits {
    pub fn bits(self) -> u8 {
        match self {
            Self::Bits16 => 16,
            Self::Bits24 => 24,
            Self::Bits32 => 32,
        }
    }
    pub(crate) fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            1 | 16 => Some(Self::Bits16),
            24 => Some(Self::Bits24),
            32 => Some(Self::Bits32),
            _ => None,
        }
    }
}

#[derive(Debug)]
enum Storage {
    U16(HugeVec<u16>),
    Packed(HugeVec<u8>),
}

/// The cached pointer keeps the owned/mapped storage dispatch off the lookup
/// path. Storage is immutable and owns the allocation for the pointer's lifetime.
#[derive(Debug)]
pub(crate) struct Fingerprints {
    pointer: NonNull<u8>,
    count: usize,
    width: FingerprintBits,
    storage: Storage,
}
// SAFETY: the pointer only accesses immutable, owned Send + Sync storage. Moving
// this wrapper cannot move Vec allocations, hugepages, or the shared file map.
unsafe impl Send for Fingerprints {}
unsafe impl Sync for Fingerprints {}
impl Fingerprints {
    pub(crate) fn low16(mut values: Vec<u16>) -> Self {
        // Do not double a large table just to append one padding element.
        values.reserve_exact(1);
        values.push(0);
        Self::u16_table(values.into())
    }
    pub(crate) fn u16_table(values: HugeVec<u16>) -> Self {
        let pointer = NonNull::new(values.as_ptr() as *mut u8).unwrap();
        Self {
            pointer,
            count: values.len().saturating_sub(1),
            width: FingerprintBits::Bits16,
            storage: Storage::U16(values),
        }
    }
    pub(crate) fn packed(data: HugeVec<u8>, width: FingerprintBits, count: usize) -> Self {
        let pointer = NonNull::new(data.as_ptr() as *mut u8).unwrap();
        Self { pointer, count, width, storage: Storage::Packed(data) }
    }
    pub(crate) fn width(&self) -> FingerprintBits {
        self.width
    }
    pub(crate) fn memory_usage(&self) -> usize {
        self.count * (self.width.bits() as usize / 8)
    }
    #[inline(always)]
    pub(crate) fn matches(&self, slot: usize, hash: u64) -> bool {
        debug_assert!(slot < self.count);
        // SAFETY: Index verifies storage length against backend slot capacity
        // before exposing a loaded index; builders establish the same invariant.
        // The 24-bit representation has a padding byte for the final u32 load.
        unsafe {
            match self.width {
                FingerprintBits::Bits16 => {
                    *self.pointer.as_ptr().cast::<u16>().add(slot) == hash as u16
                }
                width => {
                    let got = std::ptr::read_unaligned(
                        self.pointer.as_ptr().add(slot * (width.bits() as usize / 8)).cast::<u32>(),
                    )
                    .to_le();
                    let mask =
                        if width == FingerprintBits::Bits24 { 0x00ff_ffff } else { u32::MAX };
                    got & mask == hash as u32 & mask
                }
            }
        }
    }
    #[inline(always)]
    pub(crate) fn prefetch(&self, slot: usize) {
        debug_assert!(slot < self.count);
        crate::prefetch::prefetch_read(unsafe {
            self.pointer.as_ptr().add(slot * (self.width.bits() as usize / 8))
        });
    }
    pub(crate) fn validate(&self, count: usize) -> bool {
        self.count == count
            && match &self.storage {
                Storage::U16(v) => {
                    self.width == FingerprintBits::Bits16 && v.len() == count.saturating_add(1)
                }
                Storage::Packed(v) => {
                    self.width != FingerprintBits::Bits16
                        && count
                            .checked_mul(self.width.bits() as usize / 8)
                            .and_then(|v| v.checked_add(1))
                            == Some(v.len())
                }
            }
    }
    pub(crate) fn wire_flag(&self) -> u8 {
        self.width.bits() / 8 - 1
    }
    pub(crate) fn write_into<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&(self.count as u64).to_le_bytes())?;
        match &self.storage {
            Storage::U16(v) => crate::wire::write_le(w, &v[..self.count]),
            Storage::Packed(v) => w.write_all(&v[..self.memory_usage()]),
        }
    }
    pub(crate) fn mmap_bytes(&self) -> Vec<u8> {
        match &self.storage {
            Storage::U16(v) => {
                let mut bytes = Vec::new();
                crate::wire::extend_le(&mut bytes, v.as_slice());
                bytes
            }
            Storage::Packed(v) => v.to_vec(),
        }
    }
    pub(crate) fn export_low16(&self) -> Option<Vec<u16>> {
        if let Storage::U16(v) = &self.storage { Some(v[..self.count].to_vec()) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padding_does_not_double_the_table_allocation() {
        let fp = Fingerprints::low16(vec![0; 100_000]);
        let Storage::U16(HugeVec::Small(v)) = &fp.storage else { panic!("expected owned table") };
        assert_eq!(v.len(), 100_001);
        assert_eq!(v.capacity(), 100_001);
    }
}
