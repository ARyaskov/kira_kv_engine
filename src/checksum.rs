//! Streaming 64-bit checksum for serialized indexes.
//!
//! wyhash over fixed 64 KiB chunks, chained through the seed. Chunking makes the
//! value independent of how the input was split across `update` calls, so a
//! streaming writer and a one-shot verifier agree. Throughput is ~10 GB/s; a 300 MB
//! index costs ~30 ms to verify on load.

const CHUNK: usize = 64 * 1024;
const SEED: u64 = 0x4B49_5241_5F43_4B53; // "KIRA_CKS"

pub struct Checksum {
    state: u64,
    buf: Vec<u8>,
}

impl Default for Checksum {
    fn default() -> Self {
        Self::new()
    }
}

impl Checksum {
    pub fn new() -> Self {
        Self { state: SEED, buf: Vec::with_capacity(CHUNK) }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let room = CHUNK - self.buf.len();
            let take = room.min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buf.len() == CHUNK {
                self.state = wyhash::wyhash(&self.buf, self.state);
                self.buf.clear();
            }
        }
    }

    pub fn finish(mut self) -> u64 {
        if !self.buf.is_empty() {
            self.state = wyhash::wyhash(&self.buf, self.state);
        }
        // Final avalanche so an empty stream is not the raw seed.
        wyhash::wyhash(&self.state.to_le_bytes(), self.state)
    }
}

/// One-shot checksum of a byte slice.
pub fn checksum(data: &[u8]) -> u64 {
    let mut c = Checksum::new();
    c.update(data);
    c.finish()
}
