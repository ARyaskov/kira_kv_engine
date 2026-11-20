//! Bulk little-endian (de)serialization of plain integer/float arrays.
//!
//! On little-endian targets an array of `u64`/`u32`/`u16`/`f32` *is* its wire
//! form, so writing is one `memcpy` and reading is one allocation plus one
//! `memcpy`. Big-endian targets fall back to per-element conversion. Either way
//! the result is identical, so files are portable.

use std::io::{self, Write};

/// Plain fixed-width numbers with a defined little-endian byte form.
pub trait LeNum: Copy {
    const SIZE: usize;
    /// Per-element form, used on big-endian targets.
    #[allow(dead_code)]
    fn write_le(self, out: &mut [u8]);
    /// Per-element form, used on big-endian targets.
    #[allow(dead_code)]
    fn read_le(src: &[u8]) -> Self;
}

macro_rules! le_num {
    ($t:ty, $n:expr) => {
        impl LeNum for $t {
            const SIZE: usize = $n;
            #[inline(always)]
            fn write_le(self, out: &mut [u8]) {
                out[..$n].copy_from_slice(&self.to_le_bytes());
            }
            #[inline(always)]
            fn read_le(src: &[u8]) -> Self {
                <$t>::from_le_bytes(src[..$n].try_into().unwrap())
            }
        }
    };
}
le_num!(u16, 2);
le_num!(u8, 1);
le_num!(u32, 4);
le_num!(u64, 8);
le_num!(f32, 4);
le_num!(f64, 8);
le_num!(u128, 16);

/// Shared self-describing container for every engine's `to_bytes` form:
/// `[tag][magic "KIRA"][format u16][kind u8][reserved u8][body][checksum u64]`.
/// Tag 4 is shared by hash indexes; standalone PGM uses tag 5 to distinguish
/// its envelope from legacy raw PGM version 4.
/// `kind` identifies the engine (and, for `Index`, its canonical hash).
pub const CONTAINER_TAG: u8 = 4;
/// Separate from every legacy raw PGM version, including version 4.
pub const PGM_CONTAINER_TAG: u8 = 5;
pub const CONTAINER_MAGIC: &[u8; 4] = b"KIRA";
pub const CONTAINER_FORMAT: u16 = 1;
pub const KIND_INDEX: u8 = 1;
pub const KIND_HYBRID: u8 = 2;
pub const KIND_PGM_U128: u8 = 3;
pub const KIND_DYNAMIC: u8 = 4;
pub const KIND_PGM: u8 = 5;
pub const KIND_INDEX_V2: u8 = 6;
pub const KIND_HYBRID_V2: u8 = 7;
pub const HEADER_LEN: usize = 9;

/// Start the output buffer for a container, reserving room for its body and checksum.
pub(crate) fn begin_container(tag: u8, kind: u8, body_capacity: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + body_capacity + 8);
    out.push(tag);
    out.extend_from_slice(CONTAINER_MAGIC);
    out.extend_from_slice(&CONTAINER_FORMAT.to_le_bytes());
    out.push(kind);
    out.push(0);
    out
}

/// Append the checksum to the same buffer used for the header and body.
pub(crate) fn finish_container(mut out: Vec<u8>) -> Vec<u8> {
    let sum = crate::checksum::checksum(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

/// Verify header, kind and checksum; return the body.
pub fn unseal(bytes: &[u8], kind: u8) -> Option<&[u8]> {
    unseal_tagged(bytes, CONTAINER_TAG, kind)
}

pub fn unseal_tagged(bytes: &[u8], tag: u8, kind: u8) -> Option<&[u8]> {
    if bytes.len() < HEADER_LEN + 8 {
        return None;
    }
    let (data, trailer) = bytes.split_at(bytes.len() - 8);
    let stored = u64::from_le_bytes(trailer.try_into().unwrap());
    if crate::checksum::checksum(data) != stored {
        return None;
    }
    if data[0] != tag || &data[1..5] != CONTAINER_MAGIC {
        return None;
    }
    if u16::from_le_bytes([data[5], data[6]]) != CONTAINER_FORMAT || data[7] != kind {
        return None;
    }
    Some(&data[HEADER_LEN..])
}

/// Byte view of a numeric slice on little-endian targets.
#[cfg(target_endian = "little")]
#[inline(always)]
pub(crate) fn as_bytes<T: LeNum>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is a plain number without padding; `v` is a valid slice, so its
    // `len * SIZE` bytes are initialized and readable for the returned lifetime.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Append the little-endian form of `v` to `out`.
pub fn extend_le<T: LeNum>(out: &mut Vec<u8>, v: &[T]) {
    #[cfg(target_endian = "little")]
    {
        out.extend_from_slice(as_bytes(v));
    }
    #[cfg(not(target_endian = "little"))]
    {
        extend_converted_le(out, v);
    }
}

#[cfg(any(test, not(target_endian = "little")))]
fn extend_converted_le<T: LeNum>(out: &mut Vec<u8>, v: &[T]) {
    out.reserve(v.len() * T::SIZE);
    let mut buf = [0u8; 16];
    for &x in v {
        x.write_le(&mut buf);
        out.extend_from_slice(&buf[..T::SIZE]);
    }
}

/// Write the little-endian form of `v` to `w`.
pub fn write_le<T: LeNum, W: Write + ?Sized>(w: &mut W, v: &[T]) -> io::Result<()> {
    #[cfg(target_endian = "little")]
    {
        w.write_all(as_bytes(v))
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut chunk = Vec::with_capacity(64 * 1024);
        for part in v.chunks(8 * 1024) {
            chunk.clear();
            extend_le(&mut chunk, part);
            w.write_all(&chunk)?;
        }
        Ok(())
    }
}

/// Decode `count` numbers from the front of `src`. `None` if `src` is too short.
pub fn read_le<T: LeNum + Default>(src: &[u8], count: usize) -> Option<Vec<T>> {
    let bytes = count.checked_mul(T::SIZE)?;
    let src = src.get(..bytes)?;
    let mut out: Vec<T> = Vec::with_capacity(count);
    #[cfg(target_endian = "little")]
    {
        // SAFETY: the capacity holds `count` elements; `src` has exactly
        // `count * SIZE` bytes and every bit pattern is a valid `T`.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), out.as_mut_ptr() as *mut u8, bytes);
            out.set_len(count);
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        out.extend(src.chunks_exact(T::SIZE).map(T::read_le));
    }
    Some(out)
}

/// Like [`read_le`] but advances `pos`.
pub fn read_le_at<T: LeNum + Default>(src: &[u8], pos: &mut usize, count: usize) -> Option<Vec<T>> {
    let v = read_le::<T>(src.get(*pos..)?, count)?;
    *pos += count * T::SIZE;
    Some(v)
}

/// `Write` adapter that feeds everything through a [`crate::checksum::Checksum`].
pub struct ChecksumWriter<W: Write> {
    inner: W,
    sum: crate::checksum::Checksum,
}

impl<W: Write> ChecksumWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { inner, sum: crate::checksum::Checksum::new() }
    }

    /// Append the checksum of everything written so far and return the writer.
    pub fn finish(mut self) -> io::Result<W> {
        let sum = self.sum.finish();
        self.inner.write_all(&sum.to_le_bytes())?;
        Ok(self.inner)
    }
}

impl<W: Write> Write for ChecksumWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.sum.update(&buf[..n]);
        Ok(n)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.inner.write_all(buf)?;
        self.sum.update(buf);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn finishing_a_reserved_container_preserves_its_payload_allocation() {
        let payload = b"immutable payload";
        let mut out =
            super::begin_container(super::PGM_CONTAINER_TAG, super::KIND_PGM, payload.len());
        out.extend_from_slice(payload);
        let original = out.as_ptr();
        let frame = super::finish_container(out);
        assert_eq!(frame.as_ptr(), original);
        assert_eq!(&frame[..super::HEADER_LEN], b"\x05KIRA\x01\x00\x05\x00");
        assert_eq!(
            super::unseal_tagged(&frame, super::PGM_CONTAINER_TAG, super::KIND_PGM),
            Some(payload.as_slice())
        );
    }

    #[test]
    fn converted_u128_encoding_uses_all_sixteen_bytes() {
        let values = [u128::from_le_bytes(core::array::from_fn(|i| i as u8)), u128::MAX];
        let mut bytes = Vec::new();
        super::extend_converted_le(&mut bytes, &values);
        assert_eq!(&bytes[..16], &(0u8..16).collect::<Vec<_>>());
        assert_eq!(&bytes[16..], &[255; 16]);
    }
}
