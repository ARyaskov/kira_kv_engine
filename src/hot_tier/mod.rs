use crate::ptrhash::{Builder as MphBuilder, Mphf};
use crate::xor_filter::Xor8;

#[derive(Debug)]
pub struct HotTierIndex {
    xor: Xor8,
    mph: Mphf,
    fingerprints: Box<[u16]>,
    indices: Box<[u32]>,
}

impl HotTierIndex {
    pub fn build_from_u64(
        keys: &[u64],
        indices: &[u32],
        mph_config: &crate::ptrhash::BuildConfig,
    ) -> Option<Self> {
        if keys.is_empty() {
            return None;
        }
        let bytes: Vec<Vec<u8>> = keys.iter().map(|k| k.to_le_bytes().to_vec()).collect();
        let xor = Xor8::build_from_u64(keys, 0xC1B5_4A32_D192_ED03).ok()?;
        let mph = MphBuilder::new()
            .with_config(mph_config.clone())
            .build(bytes.iter().map(|k| k.as_slice()))
            .ok()?;
        let fingerprints = build_fingerprints_u64(&mph, keys).into_boxed_slice();
        let indices = indices.to_vec().into_boxed_slice();
        Some(Self {
            xor,
            mph,
            fingerprints,
            indices,
        })
    }

    #[inline]
    pub fn lookup_u64(&self, key: u64) -> Option<u32> {
        let hash = self.xor.hash_u64(key);
        if !self.xor.contains_hash(hash) {
            return None;
        }
        let idx = self.mph.index(&key.to_le_bytes()) as usize;
        let fp = fingerprint16(hash);
        if unsafe { *self.fingerprints.get_unchecked(idx) == fp } {
            Some(unsafe { *self.indices.get_unchecked(idx) })
        } else {
            None
        }
    }

    pub fn memory_usage(&self) -> usize {
        let mph_mem =
            std::mem::size_of_val(&self.mph) + self.mph.g.len() * std::mem::size_of::<u32>();
        let xor_mem = self.xor.memory_usage();
        let fp_mem = self.fingerprints.len() * std::mem::size_of::<u16>();
        let idx_mem = self.indices.len() * std::mem::size_of::<u32>();
        mph_mem + xor_mem + fp_mem + idx_mem
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.xor.write_to(out);
        write_mph(out, &self.mph);
        write_fingerprints(out, &self.fingerprints);
        write_u64(out, self.indices.len() as u64);
        for &v in self.indices.iter() {
            write_u32(out, v);
        }
    }

    pub fn read_from(bytes: &[u8], pos: &mut usize) -> Option<Self> {
        let mut cursor = crate::xor_filter::Cursor::new(bytes);
        cursor.pos = *pos;
        let xor = Xor8::read_from(&mut cursor)?;
        let mph = read_mph(&mut cursor)?;
        let fingerprints = read_fingerprints(&mut cursor)?;
        let len = cursor.read_u64()? as usize;
        let mut indices = Vec::with_capacity(len);
        for _ in 0..len {
            indices.push(read_u32(&mut cursor)?);
        }
        *pos = cursor.pos;
        Some(Self {
            xor,
            mph,
            fingerprints: fingerprints.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
        })
    }
}

fn build_fingerprints_u64(mph: &Mphf, keys: &[u64]) -> Vec<u16> {
    let mut fps = vec![0u16; keys.len()];
    for &key in keys {
        let idx = mph.index(&key.to_le_bytes()) as usize;
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

fn write_mph(out: &mut Vec<u8>, mph: &Mphf) {
    write_u64(out, mph.n);
    write_u32(out, mph.m);
    write_u64(out, mph.salt);
    write_u64(out, mph.g.len() as u64);
    for v in &mph.g {
        write_u32(out, *v);
    }
}

fn read_mph(cursor: &mut crate::xor_filter::Cursor<'_>) -> Option<Mphf> {
    let n = cursor.read_u64()?;
    let m = read_u32(cursor)?;
    let salt = cursor.read_u64()?;
    let g_len = cursor.read_u64()? as usize;
    let mut g = Vec::with_capacity(g_len);
    for _ in 0..g_len {
        g.push(read_u32(cursor)?);
    }
    Some(Mphf { n, m, salt, g })
}

fn write_fingerprints(out: &mut Vec<u8>, fps: &[u16]) {
    write_u64(out, fps.len() as u64);
    for &fp in fps {
        out.extend_from_slice(&fp.to_le_bytes());
    }
}

fn read_fingerprints(cursor: &mut crate::xor_filter::Cursor<'_>) -> Option<Vec<u16>> {
    let len = cursor.read_u64()? as usize;
    let mut fps = Vec::with_capacity(len);
    for _ in 0..len {
        let mut array = [0u8; 2];
        cursor.read_bytes(&mut array)?;
        fps.push(u16::from_le_bytes(array));
    }
    Some(fps)
}

fn read_u32(cursor: &mut crate::xor_filter::Cursor<'_>) -> Option<u32> {
    let mut array = [0u8; 4];
    cursor.read_bytes(&mut array)?;
    Some(u32::from_le_bytes(array))
}
