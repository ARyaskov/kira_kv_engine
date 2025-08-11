#[derive(Debug)]
pub struct BitSet {
    bits: Vec<u64>,
    n: usize,
}
impl BitSet {
    pub fn new(n: usize) -> Self {
        let words = (n + 63) / 64;
        Self { bits: vec![0; words], n }
    }
    #[inline]
    pub fn test(&self, idx: usize) -> bool {
        let (w, b) = (idx / 64, idx % 64);
        (self.bits[w] >> b) & 1 == 1
    }
    #[inline]
    pub fn set(&mut self, idx: usize) {
        let (w, b) = (idx / 64, idx % 64);
        self.bits[w] |= 1u64 << b;
    }
}
