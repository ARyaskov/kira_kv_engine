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
//!   - `select_sample`: every 64th set bit's position is stored explicitly so
//!     `select1(i)` is O(1) amortized — find the right sample, then linear scan.
//!
//! The decoder is branchless on the hot path; `select1` is the only non-trivial
//! step and we keep it cheap with a 1-step sample table.

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
    /// Position (in bits) of every 64th set bit in `high`. Length =
    /// ceil(N / SAMPLE).
    select_sample: Vec<u32>,
}

const SAMPLE_RATE: usize = 64;

impl EliasFano {
    /// Encode a sorted (ascending) unique u64 sequence.
    ///
    /// Returns `None` if the sequence is empty.
    pub fn from_sorted(keys: &[u64]) -> Option<Self> {
        if keys.is_empty() {
            return None;
        }
        let n = keys.len();
        // Universe is max_key + 1; we need at least 1 to define log2.
        let max_key = *keys.last().unwrap();
        let universe = max_key.saturating_add(1).max(n as u64);
        // l = floor(log2(U / N)); clamp to [0, 56] so the low component fits in 64 bits
        // even when we straddle a u64 boundary.
        let ratio = (universe / n as u64).max(1);
        let low_bits = (63 - ratio.leading_zeros()) as u8;
        let low_bits = low_bits.min(56);
        let low_mask = if low_bits == 0 { 0 } else { (1u64 << low_bits) - 1 };

        // Pack low bits.
        let total_low_bits = n * low_bits as usize;
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
        let high_len = (max_key >> low_bits as u32).saturating_add(n as u64);
        let high_words = high_len.div_ceil(64) as usize;
        let mut high = vec![0u64; high_words];
        for (i, &k) in keys.iter().enumerate() {
            let pos = (k >> low_bits as u32) + i as u64;
            let word = (pos / 64) as usize;
            let off = (pos % 64) as u8;
            high[word] |= 1u64 << off;
        }

        // Build select1 sample table. select_sample[s] = bit position of the
        // (s * SAMPLE_RATE)-th set bit (0-indexed). For s = 0 the answer is the
        // bit position of the very first '1'.
        let num_samples = n.div_ceil(SAMPLE_RATE);
        let mut select_sample = Vec::with_capacity(num_samples);
        let mut bit_count = 0usize;
        for (word_idx, &w) in high.iter().enumerate() {
            let popcnt = w.count_ones() as usize;
            // We want to record any sample target that lies in this word.
            let mut remaining = w;
            for _ in 0..popcnt {
                if bit_count.is_multiple_of(SAMPLE_RATE) {
                    let bit_in_word = remaining.trailing_zeros() as usize;
                    let abs_bit = word_idx * 64 + bit_in_word;
                    select_sample.push(abs_bit as u32);
                }
                // Clear lowest set bit.
                remaining &= remaining - 1;
                bit_count += 1;
            }
        }

        Some(EliasFano { n, universe, low_bits, low_mask, low, high_len, high, select_sample })
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

    /// Decode the i-th key in O(1) amortized. Out-of-range indices panic in
    /// debug builds; in release the behavior is wrap-around (no UB).
    #[inline]
    pub fn get(&self, i: usize) -> u64 {
        debug_assert!(i < self.n);
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
        out.reserve(count);
        if from >= self.n || count == 0 {
            return;
        }
        let end = (from + count).min(self.n);
        // We could walk the high bitvector incrementally for O(1) per element after
        // the first select1 — for now keep the simple `get`-per-index loop which is
        // already O(1) amortized.
        let mut hi_bit = self.select1(from);
        let mut last_hi = (hi_bit as u64) - from as u64;
        out.push((last_hi << self.low_bits as u32) | self.read_low(from));
        for i in (from + 1)..end {
            hi_bit = self.next_set_bit(hi_bit + 1);
            last_hi = (hi_bit as u64) - i as u64;
            out.push((last_hi << self.low_bits as u32) | self.read_low(i));
        }
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

    /// Position (bit index) of the i-th set bit in `high`. O(1) amortized via
    /// SAMPLE_RATE-th element table + popcount walk.
    #[inline]
    fn select1(&self, i: usize) -> usize {
        debug_assert!(i < self.n);
        let sample_idx = i / SAMPLE_RATE;
        let mut bits_to_skip = i % SAMPLE_RATE;
        let start_bit = self.select_sample[sample_idx] as usize;
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
            + self.select_sample.len() * 4
            + std::mem::size_of::<Self>()
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.n as u64).to_le_bytes());
        out.extend_from_slice(&self.universe.to_le_bytes());
        out.push(self.low_bits);
        out.extend_from_slice(&(self.low.len() as u64).to_le_bytes());
        crate::wire::extend_le(out, &self.low);
        out.extend_from_slice(&self.high_len.to_le_bytes());
        out.extend_from_slice(&(self.high.len() as u64).to_le_bytes());
        crate::wire::extend_le(out, &self.high);
        out.extend_from_slice(&(self.select_sample.len() as u64).to_le_bytes());
        crate::wire::extend_le(out, &self.select_sample);
    }

    pub fn read_from(bytes: &[u8], pos: &mut usize) -> Option<Self> {
        fn rd_u64(b: &[u8], p: &mut usize) -> Option<u64> {
            if *p + 8 > b.len() {
                return None;
            }
            let mut a = [0u8; 8];
            a.copy_from_slice(&b[*p..*p + 8]);
            *p += 8;
            Some(u64::from_le_bytes(a))
        }
        fn rd_u8(b: &[u8], p: &mut usize) -> Option<u8> {
            if *p + 1 > b.len() {
                return None;
            }
            let v = b[*p];
            *p += 1;
            Some(v)
        }
        // Allocation sizes are capped by the bytes actually present; structural
        // invariants are checked so decoding can never index out of bounds.
        let n = rd_u64(bytes, pos)? as usize;
        let universe = rd_u64(bytes, pos)?;
        let low_bits = rd_u8(bytes, pos)?;
        if low_bits > 56 || n == 0 {
            return None;
        }
        let low_mask = if low_bits == 0 { 0 } else { (1u64 << low_bits) - 1 };
        let low_len = rd_u64(bytes, pos)? as usize;
        let low: Vec<u64> = crate::wire::read_le_at(bytes, pos, low_len)?;
        let high_len = rd_u64(bytes, pos)?;
        let high_count = rd_u64(bytes, pos)? as usize;
        let high: Vec<u64> = crate::wire::read_le_at(bytes, pos, high_count)?;
        let sample_count = rd_u64(bytes, pos)? as usize;
        let select_sample: Vec<u32> = crate::wire::read_le_at(bytes, pos, sample_count)?;
        // low: n * low_bits bits must fit; high: high_len bits must fit and hold
        // exactly n set bits; samples: one per SAMPLE_RATE keys, each inside `high`.
        let need_low = (n as u64 * low_bits as u64).div_ceil(64) as usize;
        if low.len() < need_low || (high.len() as u64) < high_len.div_ceil(64) {
            return None;
        }
        let ones: u64 = high.iter().map(|w| w.count_ones() as u64).sum();
        if ones != n as u64 || high_len < n as u64 {
            return None;
        }
        if select_sample.len() != n.div_ceil(SAMPLE_RATE)
            || select_sample.iter().any(|&s| s as u64 >= high_len)
        {
            return None;
        }
        Some(EliasFano { n, universe, low_bits, low_mask, low, high_len, high, select_sample })
    }
}
