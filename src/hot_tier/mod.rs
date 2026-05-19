use crate::block_bloom::BlockBloom;
use crate::ptrhash25::{Builder as MphBuilder, PtrHash25Mphf};

#[derive(Debug)]
pub struct HotTierIndex {
    filter: BlockBloom,
    mph: PtrHash25Mphf,
    fingerprints: Box<[u16]>,
    indices: Box<[u32]>,
}

impl HotTierIndex {
    pub fn build_from_u64(keys: &[u64], indices: &[u32], seed: u64) -> Option<Self> {
        if keys.is_empty() {
            return None;
        }
        let filter = BlockBloom::build_from_u64(keys, 0xC1B5_4A32_D192_ED03);
        let cfg = crate::ptrhash25::BuildConfig {
            gamma: 0.5,
            max_rehash: 16,
            with_fingerprints: false,
            seed,
            use_aes_hash: false,
        };
        let mph = MphBuilder::new().with_config(cfg).build(keys).ok()?;
        let slot_space = mph.n as usize;
        let fingerprints =
            build_fingerprints_u64(&mph, keys, slot_space).into_boxed_slice();
        // Map slot → original index. Slot space may be larger than keys.len() due
        // to PtrHash25's 1.10× padding; un-used slots stay at u32::MAX.
        let mut indices_vec = vec![u32::MAX; slot_space];
        for (i, &k) in keys.iter().enumerate() {
            let slot = mph.index_u64(k) as usize;
            indices_vec[slot] = indices[i];
        }
        Some(Self {
            filter,
            mph,
            fingerprints,
            indices: indices_vec.into_boxed_slice(),
        })
    }

    #[inline]
    pub fn lookup_u64(&self, key: u64) -> Option<u32> {
        if !self.filter.contains_u64(key) {
            return None;
        }
        let slot = self.mph.index_u64(key) as usize;
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
        mph_mem + filter_mem + fp_mem + idx_mem
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.filter.write_to(out);
        crate::ptrhash25::write_ptrhash25(&self.mph, out);
        write_fingerprints(out, &self.fingerprints);
        write_u64(out, self.indices.len() as u64);
        for &v in self.indices.iter() {
            write_u32(out, v);
        }
    }

    pub fn read_from(bytes: &[u8], pos: &mut usize) -> Option<Self> {
        let filter = BlockBloom::read_from(bytes, pos)?;
        let mph = crate::ptrhash25::read_ptrhash25(bytes, pos)?;
        let mut cur = LocalCursor { buf: bytes, pos: *pos };
        let fingerprints = read_fingerprints(&mut cur)?;
        let len = cur.read_u64()? as usize;
        let mut indices = Vec::with_capacity(len);
        for _ in 0..len {
            indices.push(read_u32(&mut cur)?);
        }
        *pos = cur.pos;
        Some(Self {
            filter,
            mph,
            fingerprints: fingerprints.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
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
    fn read_u64(&mut self) -> Option<u64> {
        if self.pos + 8 > self.buf.len() {
            return None;
        }
        let mut a = [0u8; 8];
        a.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Some(u64::from_le_bytes(a))
    }

    fn read_bytes(&mut self, out: &mut [u8]) -> Option<()> {
        if self.pos + out.len() > self.buf.len() {
            return None;
        }
        out.copy_from_slice(&self.buf[self.pos..self.pos + out.len()]);
        self.pos += out.len();
        Some(())
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

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_fingerprints(out: &mut Vec<u8>, fps: &[u16]) {
    write_u64(out, fps.len() as u64);
    for &fp in fps {
        out.extend_from_slice(&fp.to_le_bytes());
    }
}

fn read_fingerprints(cursor: &mut LocalCursor<'_>) -> Option<Vec<u16>> {
    let len = cursor.read_u64()? as usize;
    let mut fps = Vec::with_capacity(len);
    for _ in 0..len {
        let mut array = [0u8; 2];
        cursor.read_bytes(&mut array)?;
        fps.push(u16::from_le_bytes(array));
    }
    Some(fps)
}

fn read_u32(cursor: &mut LocalCursor<'_>) -> Option<u32> {
    let mut array = [0u8; 4];
    cursor.read_bytes(&mut array)?;
    Some(u32::from_le_bytes(array))
}
