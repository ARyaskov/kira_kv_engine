//! Compact Elias-Fano encoding for sorted u64 sequences.
//!
//! For N sorted unique u64 keys with universe U = max_key + 1, Elias-Fano uses
//! N · (2 + ceil(log2(U/N))) bits — typically 35–40% less than the 8N bytes of
//! a `Vec<u64>` at U ≈ 2^64, N = 10^8.
//!
//! Layout:
//!   - `low`: packed low `l` bits of each key, where l = floor(log2(U/N)).
//!     Total bits = N · l, packed into a `Vec<u64>` little-endian.
//!   - `high`: unary high bits. For key i, write a single `1` at bit position
//!     `(key[i] >> l) + i` of a bitvector of length N + (U >> l). All other bits
//!     are `0`. Decoding recovers `high_i = select1(i) - i`.
//!   - A derived select directory groups 64 consecutive ranks. Dense blocks span
//!     at most 2048 bits; sparse blocks retain their positions explicitly. This
//!     bounds random access independently of gaps between key values.
//!
//! The ordinary wire layout is unchanged. Its rank samples are checked while
//! rebuilding the derived directory on load. A reserved-zero prefix identifies
//! the wide-sample extension for positions that cannot fit in the legacy u32.

/// Compact representation of a sorted u64 sequence.
#[derive(Debug, Clone)]
pub struct EliasFano {
    n: usize,
    universe: u64,
    /// Bits per low component.
    low_bits: u8,
    /// Mask = (1 << low_bits) - 1.
    low_mask: u64,
    low: Vec<u64>,
    /// Length of the high bitvector in bits.
    high_len: u64,
    high: Vec<u64>,
    select_blocks: Vec<SelectBlock>,
    sparse_positions: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::{DENSE_SPAN, EliasFano, SelectBlock, WIDE_MAGIC};

    #[test]
    fn long_gap_select_uses_sparse_positions_and_bounds_dense_spans() {
        let n = 4096u64;
        let keys: Vec<_> = std::iter::once(0).chain(n * n - n + 2..=n * n).collect();
        let index = EliasFano::from_sorted(&keys).unwrap();
        assert!(matches!(index.select_blocks[0], SelectBlock::Sparse { .. }));
        for (block, entry) in index.select_blocks.iter().enumerate() {
            if let SelectBlock::Dense { first } = *entry {
                let last_rank = ((block + 1) * 64).min(index.len()) - 1;
                assert!(index.select1(last_rank) - first < DENSE_SPAN);
            }
        }
        let mut output = [u64::MAX; 130];
        assert_eq!(index.materialize_into(0, &mut output), output.len());
        assert_eq!(output.as_slice(), &keys[..output.len()]);
        let mut tail = [u64::MAX; 4];
        assert_eq!(index.materialize_into(index.len() - 2, &mut tail), 2);
        assert_eq!(&tail[..2], &keys[keys.len() - 2..]);
        assert_eq!(&tail[2..], &[u64::MAX; 2]);
        assert_eq!(index.materialize_into(usize::MAX, &mut tail), 0);
    }

    #[test]
    fn wide_sample_extension_roundtrips_without_large_allocations() {
        let keys = [0, 9, 10, 1000, u64::MAX];
        let index = EliasFano::from_sorted(&keys).unwrap();
        let mut bytes = vec![17];
        index.write_format(&mut bytes, true);
        let end = bytes.len();
        bytes.extend_from_slice(b"next payload");
        assert_eq!(&bytes[9..17], WIDE_MAGIC);
        let mut cursor = 1;
        let restored = EliasFano::read_from(&bytes, &mut cursor).unwrap();
        assert_eq!(cursor, end);
        for (rank, &key) in keys.iter().enumerate() {
            assert_eq!(restored.get(rank), key);
        }
        let mut bad = bytes[..end].to_vec();
        bad[9] ^= 1;
        assert!(EliasFano::read_from(&bad, &mut 1).is_none());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn sample_positions_above_u32_are_serialized_without_truncation() {
        // Exercise the wire-width boundary without allocating the corresponding
        // half-gigabyte high vector. Only serialization reads this synthetic slot.
        for position in [u32::MAX as usize, u32::MAX as usize + 1] {
            let mut index = EliasFano::from_sorted(&[0]).unwrap();
            index.select_blocks[0] = SelectBlock::Dense { first: position };
            let mut bytes = Vec::new();
            index.write_to(&mut bytes);
            if position == u32::MAX as usize {
                assert_eq!(u64::from_le_bytes(bytes[..8].try_into().unwrap()), 1);
                assert_eq!(&bytes[bytes.len() - 4..], &u32::MAX.to_le_bytes());
            } else {
                assert_eq!(u64::from_le_bytes(bytes[..8].try_into().unwrap()), 0);
                assert_eq!(&bytes[8..16], WIDE_MAGIC);
                assert_eq!(&bytes[bytes.len() - 8..], &(position as u64).to_le_bytes());
            }
        }
    }
}

const SAMPLE_RATE: usize = 64;
const DENSE_SPAN: usize = 2048;
const WIDE_MAGIC: &[u8; 8] = b"KIRAEF02";

#[derive(Debug, Clone, Copy)]
enum SelectBlock {
    Dense { first: usize },
    Sparse { offset: usize },
}

impl EliasFano {
    /// Encode a sorted (ascending) unique u64 sequence.
    ///
    /// Returns `None` if the sequence is empty, not strictly ascending, or its
    /// representation exceeds the target's addressable range.
    pub fn from_sorted(keys: &[u64]) -> Option<Self> {
        if keys.is_empty() || keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return None;
        }
        let n = keys.len();
        // Universe is max_key + 1; we need at least 1 to define log2.
        let max_key = *keys.last().unwrap();
        let universe = max_key.saturating_add(1).max(n as u64);
        // l = floor(log2(U / N)); clamp to [0, 56] so the low component fits in 64 bits
        // even when we straddle a u64 boundary.
        let ratio = (universe / n as u64).max(1);
        let low_bits = ratio.ilog2() as u8;
        let low_bits = low_bits.min(56);
        let low_mask = if low_bits == 0 { 0 } else { (1u64 << low_bits) - 1 };

        // Pack low bits.
        let total_low_bits = n.checked_mul(low_bits as usize)?;
        let low_words = total_low_bits.div_ceil(64);
        let mut low = vec![0u64; low_words];
        if low_bits > 0 {
            for (i, &k) in keys.iter().enumerate() {
                let lo = k & low_mask;
                let bit_pos = i * low_bits as usize;
                let word = bit_pos / 64;
                let off = bit_pos % 64;
                low[word] |= lo << off;
                if off + low_bits as usize > 64 {
                    low[word + 1] |= lo >> (64 - off);
                }
            }
        }

        // High bitvector: bit `(k >> low_bits) + i` set for the i-th key.
        let high_len = (max_key >> low_bits as u32).checked_add(n as u64)?;
        let high_words = usize::try_from(high_len.div_ceil(64)).ok()?;
        let mut high = vec![0u64; high_words];
        for (i, &k) in keys.iter().enumerate() {
            let pos = (k >> low_bits as u32) + i as u64;
            let word = (pos / 64) as usize;
            let off = (pos % 64) as u8;
            high[word] |= 1u64 << off;
        }

        let (select_blocks, sparse_positions) =
            Self::build_select(&high, usize::try_from(high_len).ok()?, n, None)?;
        Some(EliasFano {
            n,
            universe,
            low_bits,
            low_mask,
            low,
            high_len,
            high,
            select_blocks,
            sparse_positions,
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn universe(&self) -> u64 {
        self.universe
    }

    /// Decode the i-th key with bounded O(1) work. Dense blocks visit at most 33
    /// high words; sparse blocks use one indexed position. Out-of-range indices
    /// panic in every build profile.
    #[inline]
    pub fn get(&self, i: usize) -> u64 {
        assert!(i < self.n, "EliasFano index out of bounds");
        let low = self.read_low(i);
        let high_bit = self.select1(i) as u64;
        let high = high_bit - i as u64;
        (high << self.low_bits as u32) | low
    }

    /// Decode `count` consecutive keys starting at `from` into `out`. This is the
    /// hot path for PGM local search: ε keys around the predicted position are
    /// materialized at once into a stack-allocated buffer, then scanned with SIMD.
    pub fn materialize_range(&self, from: usize, count: usize, out: &mut Vec<u64>) {
        out.clear();
        if from >= self.n || count == 0 {
            return;
        }
        let count = count.min(self.n - from);
        out.reserve(count);
        self.decode_range(from, count, |key| out.push(key));
    }

    /// Fill existing storage without allocating; only the returned prefix is
    /// overwritten. Dense blocks are walked once, with directory jumps across
    /// sparse gaps and rank-block boundaries.
    pub(crate) fn materialize_into(&self, from: usize, out: &mut [u64]) -> usize {
        let count = out.len().min(self.n.saturating_sub(from));
        let mut written = 0;
        self.decode_range(from, count, |key| {
            out[written] = key;
            written += 1;
        });
        written
    }

    fn decode_range(&self, from: usize, count: usize, mut emit: impl FnMut(u64)) {
        let end = from + count;
        let mut rank = from;
        while rank < end {
            let block_index = rank / SAMPLE_RATE;
            let block_start = block_index * SAMPLE_RATE;
            let block_end = end.min(block_start.saturating_add(SAMPLE_RATE));
            match self.select_blocks[block_index] {
                SelectBlock::Sparse { offset } => {
                    for i in rank..block_end {
                        let position = self.sparse_positions[offset + i - block_start];
                        emit(self.decode_key(i, position));
                    }
                }
                SelectBlock::Dense { .. } => {
                    let mut position = self.select1(rank);
                    emit(self.decode_key(rank, position));
                    for i in rank + 1..block_end {
                        position = self.next_set_bit(position + 1);
                        emit(self.decode_key(i, position));
                    }
                }
            }
            rank = block_end;
        }
    }

    #[inline]
    fn decode_key(&self, rank: usize, position: usize) -> u64 {
        let high = position as u64 - rank as u64;
        (high << self.low_bits as u32) | self.read_low(rank)
    }

    #[inline]
    fn read_low(&self, i: usize) -> u64 {
        if self.low_bits == 0 {
            return 0;
        }
        let bit_pos = i * self.low_bits as usize;
        let word = bit_pos / 64;
        let off = bit_pos % 64;
        let mut v = self.low[word] >> off;
        if off + self.low_bits as usize > 64 && word + 1 < self.low.len() {
            v |= self.low[word + 1] << (64 - off);
        }
        v & self.low_mask
    }

    /// Position of the i-th set bit. The constructor and reader establish the
    /// dense-span bound; sparse blocks bypass the high-vector walk completely.
    #[inline]
    fn select1(&self, i: usize) -> usize {
        debug_assert!(i < self.n);
        let block_index = i / SAMPLE_RATE;
        let mut bits_to_skip = i % SAMPLE_RATE;
        let start_bit = match self.select_blocks[block_index] {
            SelectBlock::Dense { first } => first,
            SelectBlock::Sparse { offset } => return self.sparse_positions[offset + bits_to_skip],
        };
        let mut word = start_bit / 64;
        let off = start_bit % 64;
        // Mask off bits below `off` (already counted in earlier samples). The
        // sampled bit itself stays — it is the 0-th set bit relative to this
        // sample, so when bits_to_skip == 0 we want trailing_zeros to return
        // exactly `start_bit`.
        let mut w = self.high[word] & !((1u64 << off) - 1);
        loop {
            let popcnt = w.count_ones() as usize;
            if popcnt > bits_to_skip {
                // Strip `bits_to_skip` lowest set bits, then trailing_zeros
                // gives the position of the target bit.
                for _ in 0..bits_to_skip {
                    w &= w - 1;
                }
                return word * 64 + w.trailing_zeros() as usize;
            }
            bits_to_skip -= popcnt;
            word += 1;
            w = self.high[word];
        }
    }

    /// Find the next set bit at or after `from`. Used by `materialize_range` to
    /// advance one position cheaply.
    #[inline]
    fn next_set_bit(&self, from: usize) -> usize {
        let mut word = from / 64;
        let off = from % 64;
        let mut w = self.high[word] & !((1u64 << off) - 1);
        loop {
            if w != 0 {
                return word * 64 + w.trailing_zeros() as usize;
            }
            word += 1;
            if word >= self.high.len() {
                return self.high_len as usize;
            }
            w = self.high[word];
        }
    }

    pub fn memory_usage(&self) -> usize {
        self.low.len() * 8
            + self.high.len() * 8
            + self.select_blocks.capacity() * std::mem::size_of::<SelectBlock>()
            + self.sparse_positions.capacity() * std::mem::size_of::<usize>()
            + std::mem::size_of::<Self>()
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        let wide = (0..self.select_blocks.len())
            .any(|block| u32::try_from(self.block_first(block)).is_err());
        self.write_format(out, wide);
    }

    fn block_first(&self, block: usize) -> usize {
        match self.select_blocks[block] {
            SelectBlock::Dense { first } => first,
            SelectBlock::Sparse { offset } => self.sparse_positions[offset],
        }
    }

    fn write_format(&self, out: &mut Vec<u8>, wide: bool) {
        if wide {
            out.extend_from_slice(&0u64.to_le_bytes());
            out.extend_from_slice(WIDE_MAGIC);
        }
        out.extend_from_slice(&(self.n as u64).to_le_bytes());
        out.extend_from_slice(&self.universe.to_le_bytes());
        out.push(self.low_bits);
        out.extend_from_slice(&(self.low.len() as u64).to_le_bytes());
        crate::wire::extend_le(out, &self.low);
        out.extend_from_slice(&self.high_len.to_le_bytes());
        out.extend_from_slice(&(self.high.len() as u64).to_le_bytes());
        crate::wire::extend_le(out, &self.high);
        out.extend_from_slice(&(self.select_blocks.len() as u64).to_le_bytes());
        for block in 0..self.select_blocks.len() {
            let position = self.block_first(block);
            if wide {
                out.extend_from_slice(&(position as u64).to_le_bytes());
            } else {
                out.extend_from_slice(&(position as u32).to_le_bytes());
            }
        }
    }

    fn build_select(
        high: &[u64],
        high_len: usize,
        n: usize,
        samples: Option<(&[u8], bool)>,
    ) -> Option<(Vec<SelectBlock>, Vec<usize>)> {
        let mut blocks = Vec::with_capacity(n.div_ceil(SAMPLE_RATE));
        let mut sparse = Vec::new();
        let mut positions = [0usize; SAMPLE_RATE];
        let mut rank = 0usize;
        for (word_index, &word) in high.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                if rank >= n {
                    return None;
                }
                let position =
                    word_index.checked_mul(64)?.checked_add(bits.trailing_zeros() as usize)?;
                if position >= high_len {
                    return None;
                }
                let local = rank % SAMPLE_RATE;
                if local == 0
                    && let Some((samples, wide)) = samples
                {
                    let width = if wide { 8 } else { 4 };
                    let offset = (rank / SAMPLE_RATE).checked_mul(width)?;
                    let raw = samples.get(offset..offset.checked_add(width)?)?;
                    let expected = if wide {
                        usize::try_from(u64::from_le_bytes(raw.try_into().ok()?)).ok()?
                    } else {
                        u32::from_le_bytes(raw.try_into().ok()?) as usize
                    };
                    if position != expected {
                        return None;
                    }
                }
                positions[local] = position;
                rank += 1;
                if local + 1 == SAMPLE_RATE {
                    Self::push_select_block(&positions, &mut blocks, &mut sparse);
                }
                bits &= bits - 1;
            }
        }
        if rank != n {
            return None;
        }
        let tail = rank % SAMPLE_RATE;
        if tail != 0 {
            Self::push_select_block(&positions[..tail], &mut blocks, &mut sparse);
        }
        Some((blocks, sparse))
    }

    fn push_select_block(
        positions: &[usize],
        blocks: &mut Vec<SelectBlock>,
        sparse: &mut Vec<usize>,
    ) {
        let first = positions[0];
        if positions[positions.len() - 1] - first < DENSE_SPAN {
            blocks.push(SelectBlock::Dense { first });
        } else {
            let offset = sparse.len();
            sparse.extend_from_slice(positions);
            blocks.push(SelectBlock::Sparse { offset });
        }
    }

    pub fn read_from(bytes: &[u8], pos: &mut usize) -> Option<Self> {
        fn rd_bytes<'a>(bytes: &'a [u8], pos: &mut usize, len: usize) -> Option<&'a [u8]> {
            let end = pos.checked_add(len)?;
            let value = bytes.get(*pos..end)?;
            *pos = end;
            Some(value)
        }
        fn rd_u64(b: &[u8], p: &mut usize) -> Option<u64> {
            Some(u64::from_le_bytes(rd_bytes(b, p, 8)?.try_into().ok()?))
        }
        fn rd_u8(b: &[u8], p: &mut usize) -> Option<u8> {
            Some(rd_bytes(b, p, 1)?[0])
        }
        // Allocation sizes are capped by the bytes actually present; structural
        // invariants are checked so decoding can never index out of bounds.
        let mut cursor = *pos;
        let first = rd_u64(bytes, &mut cursor)?;
        let wide = first == 0;
        let n = if wide {
            if rd_bytes(bytes, &mut cursor, WIDE_MAGIC.len())? != WIDE_MAGIC {
                return None;
            }
            usize::try_from(rd_u64(bytes, &mut cursor)?).ok()?
        } else {
            usize::try_from(first).ok()?
        };
        let universe = rd_u64(bytes, &mut cursor)?;
        let low_bits = rd_u8(bytes, &mut cursor)?;
        if low_bits > 56 || n == 0 {
            return None;
        }
        let low_mask = if low_bits == 0 { 0 } else { (1u64 << low_bits) - 1 };
        let low_len = usize::try_from(rd_u64(bytes, &mut cursor)?).ok()?;
        let low: Vec<u64> = crate::wire::read_le_at(bytes, &mut cursor, low_len)?;
        let high_len = rd_u64(bytes, &mut cursor)?;
        let high_count = usize::try_from(rd_u64(bytes, &mut cursor)?).ok()?;
        let high: Vec<u64> = crate::wire::read_le_at(bytes, &mut cursor, high_count)?;
        let sample_count = usize::try_from(rd_u64(bytes, &mut cursor)?).ok()?;
        if sample_count != n.div_ceil(SAMPLE_RATE) {
            return None;
        }
        let samples =
            rd_bytes(bytes, &mut cursor, sample_count.checked_mul(if wide { 8 } else { 4 })?)?;
        // low: n * low_bits bits must fit; high: high_len bits must fit and hold
        // exactly n set bits; samples: one per SAMPLE_RATE keys, each inside `high`.
        let need_low = n.checked_mul(low_bits as usize)?.div_ceil(64);
        if low.len() < need_low || (high.len() as u64) < high_len.div_ceil(64) {
            return None;
        }
        if high_len < n as u64 || universe < n as u64 {
            return None;
        }
        let (select_blocks, sparse_positions) =
            Self::build_select(&high, usize::try_from(high_len).ok()?, n, Some((samples, wide)))?;
        let value = EliasFano {
            n,
            universe,
            low_bits,
            low_mask,
            low,
            high_len,
            high,
            select_blocks,
            sparse_positions,
        };
        let highest = value.select1(n - 1) as u64 - (n - 1) as u64;
        if highest > (u64::MAX >> low_bits) {
            return None;
        }
        let mut previous = None;
        let mut valid = true;
        value.decode_range(0, n, |key| {
            if previous.is_some_and(|prev| prev >= key)
                || (key >= universe && !(universe == u64::MAX && key == u64::MAX))
            {
                valid = false;
            }
            previous = Some(key);
        });
        if !valid {
            return None;
        }
        *pos = cursor;
        Some(value)
    }
}
