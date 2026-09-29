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
le_num!(u32, 4);
le_num!(u64, 8);
le_num!(f32, 4);
le_num!(f64, 8);
le_num!(u128, 16);

/// Shared self-describing container for every engine's `to_bytes` form:
/// `[TAG=4][magic "KIRA"][format u16][kind u8][reserved u8][body][checksum u64]`.
/// `kind` identifies the engine (and, for `Index`, its canonical hash).
pub const CONTAINER_TAG: u8 = 4;
pub const CONTAINER_MAGIC: &[u8; 4] = b"KIRA";
pub const CONTAINER_FORMAT: u16 = 1;
pub const KIND_INDEX: u8 = 1;
pub const KIND_HYBRID: u8 = 2;
pub const KIND_PGM_U128: u8 = 3;
pub const KIND_DYNAMIC: u8 = 4;
pub const HEADER_LEN: usize = 9;

/// Wrap `body` into the sealed container of the given `kind`.
pub fn seal(kind: u8, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + body.len() + 8);
    out.push(CONTAINER_TAG);
    out.extend_from_slice(CONTAINER_MAGIC);
    out.extend_from_slice(&CONTAINER_FORMAT.to_le_bytes());
    out.push(kind);
    out.push(0);
    out.extend_from_slice(body);
    let sum = crate::checksum::checksum(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

/// Verify header, kind and checksum; return the body.
pub fn unseal(bytes: &[u8], kind: u8) -> Option<&[u8]> {
    if bytes.len() < HEADER_LEN + 8 {
        return None;
    }
    let (data, trailer) = bytes.split_at(bytes.len() - 8);
    let stored = u64::from_le_bytes(trailer.try_into().unwrap());
    if crate::checksum::checksum(data) != stored {
        return None;
    }
    if data[0] != CONTAINER_TAG || &data[1..5] != CONTAINER_MAGIC {
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
fn as_bytes<T: LeNum>(v: &[T]) -> &[u8] {
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
        out.reserve(v.len() * T::SIZE);
        let mut buf = [0u8; 8];
        for &x in v {
            x.write_le(&mut buf);
            out.extend_from_slice(&buf[..T::SIZE]);
        }
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
