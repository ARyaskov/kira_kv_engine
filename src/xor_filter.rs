use wyhash::wyhash;

#[derive(Debug, Clone)]
pub struct Xor8 {
    seed: u64,
    fingerprints: Vec<u8>,
}

impl Xor8 {
    pub fn build_from_bytes(keys: &[Vec<u8>], seed: u64) -> Result<Self, ()> {
        let mut hashes = Vec::with_capacity(keys.len());
        for k in keys {
            hashes.push(hash_bytes(k.as_slice(), seed));
        }
        Self::build_from_hashes(&hashes, seed)
    }

    pub fn build_from_u64(keys: &[u64], seed: u64) -> Result<Self, ()> {
        let mut hashes = Vec::with_capacity(keys.len());
        for &k in keys {
            hashes.push(hash_u64(k, seed));
        }
        Self::build_from_hashes(&hashes, seed)
    }

    pub fn contains_bytes(&self, key: &[u8]) -> bool {
        let hash = hash_bytes(key, self.seed);
        self.contains_hash(hash)
    }

    pub fn contains_u64(&self, key: u64) -> bool {
        let hash = hash_u64(key, self.seed);
        self.contains_hash(hash)
    }

    pub fn hash_bytes(&self, key: &[u8]) -> u64 {
        hash_bytes(key, self.seed)
    }

    pub fn hash_u64(&self, key: u64) -> u64 {
        hash_u64(key, self.seed)
    }

    pub fn memory_usage(&self) -> usize {
        std::mem::size_of_val(self) + self.fingerprints.len()
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        write_u64(out, self.seed);
        write_u64(out, self.fingerprints.len() as u64);
        out.extend_from_slice(&self.fingerprints);
    }

    pub fn read_from(cursor: &mut Cursor<'_>) -> Option<Self> {
        let seed = cursor.read_u64()?;
        let len = cursor.read_u64()? as usize;
        let mut fingerprints = vec![0u8; len];
        cursor.read_bytes(&mut fingerprints)?;
        Some(Self { seed, fingerprints })
    }

    pub(crate) fn contains_hash(&self, hash: u64) -> bool {
        let fp = fingerprint8(hash);
        let (i0, i1, i2) = indices(hash, self.fingerprints.len());
        fp == (self.fingerprints[i0] ^ self.fingerprints[i1] ^ self.fingerprints[i2])
    }

    fn build_from_hashes(hashes: &[u64], seed: u64) -> Result<Self, ()> {
        if hashes.is_empty() {
            return Err(());
        }
        let size = ((hashes.len() as f64) * 1.23).ceil() as usize + 32;
        let mut counts = vec![0u32; size];
        let mut xors = vec![0u64; size];

        for &h in hashes {
            let (i0, i1, i2) = indices(h, size);
            counts[i0] += 1;
            counts[i1] += 1;
            counts[i2] += 1;
            xors[i0] ^= h;
            xors[i1] ^= h;
            xors[i2] ^= h;
        }

        let mut stack = Vec::with_capacity(hashes.len());
        let mut queue = Vec::new();
        for (i, &c) in counts.iter().enumerate() {
            if c == 1 {
                queue.push(i);
            }
        }

        while let Some(i) = queue.pop() {
            if counts[i] == 0 {
                continue;
            }
            let h = xors[i];
            stack.push((i, h));
            let (i0, i1, i2) = indices(h, size);
            for idx in [i0, i1, i2] {
                if idx == i {
                    continue;
                }
                counts[idx] -= 1;
                xors[idx] ^= h;
                if counts[idx] == 1 {
                    queue.push(idx);
                }
            }
            counts[i] = 0;
        }

        if stack.len() != hashes.len() {
            return Err(());
        }

        let mut fingerprints = vec![0u8; size];
        while let Some((i, h)) = stack.pop() {
            let fp = fingerprint8(h);
            let (i0, i1, i2) = indices(h, size);
            let v = fp ^ fingerprints[i0] ^ fingerprints[i1] ^ fingerprints[i2];
            fingerprints[i] = v;
        }

        Ok(Self { seed, fingerprints })
    }
}

fn hash_bytes(key: &[u8], seed: u64) -> u64 {
    wyhash(key, seed)
}

fn hash_u64(key: u64, seed: u64) -> u64 {
    splitmix64(key ^ seed)
}

fn fingerprint8(hash: u64) -> u8 {
    (hash & 0xFF) as u8
}

fn indices(mut hash: u64, size: usize) -> (usize, usize, usize) {
    let mut i0 = (hash as usize) % size;
    hash = splitmix64(hash);
    let mut i1 = (hash as usize) % size;
    hash = splitmix64(hash);
    let mut i2 = (hash as usize) % size;
    if i0 == i1 || i0 == i2 || i1 == i2 {
        hash = splitmix64(hash ^ 0x9E37_79B9_7F4A_7C15);
        i0 = (hash as usize) % size;
        hash = splitmix64(hash);
        i1 = (hash as usize) % size;
        hash = splitmix64(hash);
        i2 = (hash as usize) % size;
    }
    (i0, i1, i2)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub struct Cursor<'a> {
    buf: &'a [u8],
    pub(crate) pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn read_u64(&mut self) -> Option<u64> {
        if self.pos + 8 > self.buf.len() {
            return None;
        }
        let mut array = [0u8; 8];
        array.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Some(u64::from_le_bytes(array))
    }

    pub fn read_bytes(&mut self, out: &mut [u8]) -> Option<()> {
        if self.pos + out.len() > self.buf.len() {
            return None;
        }
        out.copy_from_slice(&self.buf[self.pos..self.pos + out.len()]);
        self.pos += out.len();
        Some(())
    }
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
