//! Section container for on-disk indexes.
//!
//! Layout: a fixed header, a section table, then the sections, each starting at a
//! 64-byte-aligned offset. The alignment is what a future zero-copy loader would
//! need to reference pilots, Bloom words and fingerprints in place.
//!
//! **Today there is no memory mapping**: `MmapIndex::open` reads the whole file
//! into a `Vec<u8>` (no `memmap2` dependency), and `Index::open_mmap` parses the
//! `LegacyPayload` section exactly like `Index::from_bytes`. The names are kept
//! for API compatibility; `Index::save`/`Index::load` are the plain, streaming
//! alternative.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

/// Magic prefix to detect kira_kv_engine on-disk indexes.
pub const MAGIC: &[u8; 8] = b"KIRA_V01";

/// Section identifiers. Each section starts at a 64-byte-aligned offset within the file.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // wire-format ids; not all are produced by this version
pub enum SectionKind {
    /// Whole legacy payload (output of Index::to_bytes). Used by the v0 path.
    LegacyPayload = 0,
    /// u8 pilots array of PtrHash25. Aligned to 64 B.
    PtrHash25Pilots = 1,
    /// u64 words of BlockBloom. Aligned to 64 B.
    BlockBloomWords = 2,
    /// u8 fingerprints of PtrHash25.
    PtrHash25Fingerprints = 3,
    /// u16 fingerprints of the outer MPH Engine.
    OuterFingerprints = 4,
    /// PtrHash25 metadata (n, num_buckets, salt, bloom_seed, prehash_seed, key_count) — POD.
    PtrHash25Meta = 5,

    // ---- PGM zero-copy sections ----
    /// PGM raw sorted keys, `[u64]`.
    PgmKeys = 16,
    /// PGM segment slopes, `[f32]` (one f32 per segment).
    PgmSlopes = 17,
    /// PGM segment intercepts, `[f32]`.
    PgmIntercepts = 18,
    /// PGM per-segment min_key, `[u64]`.
    PgmMinKeys = 19,
    /// PGM per-segment max_key, `[u64]`.
    PgmMaxKeys = 20,
    /// PGM per-segment max_error, `[u8]` (0xFF means see overflow).
    PgmMaxErrors = 21,
    /// PGM overflow errors, `[(u32 seg_idx, u32 err)]`.
    PgmOverflowErrors = 22,
    /// PGM per-segment 64-bit filter, `[u64]`.
    PgmFilters = 23,
    /// PGM per-segment start position, `[u32]`.
    PgmStarts = 24,
    /// PGM per-segment end position, `[u32]`.
    PgmEnds = 25,
    /// PGM metadata: `[u32 epsilon][u8 has_bloom][padding…]`.
    PgmMeta = 26,
    /// PGM optional Block-Bloom (encoded in BlockBloom format).
    PgmBloom = 27,
}

/// On-disk header (POD, native LE).
#[repr(C)]
pub struct MmapHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub section_count: u32,
    pub key_count: u64,
    pub flags: u64,
    pub _reserved: [u64; 4],
}

/// Per-section entry in the header table.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SectionEntry {
    pub kind: u32,
    pub _pad: u32,
    pub offset: u64,
    pub length: u64,
}

/// Reader over a mmap'd index file. Owns the mmap; exposes typed views.
pub struct MmapIndex {
    mmap: Box<dyn AsRef<[u8]> + Send + Sync>,
}

impl MmapIndex {
    /// Open an index file via memory map (read-only). Returns a reader that gives
    /// zero-copy access to sections.
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        // Without the `memmap2` crate (avoiding new deps), we fall back to read-into-Vec.
        // The shape stays identical so a memmap2 swap is a one-line change later.
        let bytes = std::fs::read(path)?;
        Ok(MmapIndex { mmap: Box::new(bytes) })
    }

    pub fn as_bytes(&self) -> &[u8] {
        (*self.mmap).as_ref()
    }

    /// Parse the header and return a view of each section by kind.
    pub fn parse_header(&self) -> std::io::Result<Header> {
        let bytes = self.as_bytes();
        if bytes.len() < std::mem::size_of::<MmapHeader>() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "header too short"));
        }
        // SAFETY: bounds-checked above; the header is plain data and `Vec<u8>` gives no
        // alignment guarantee, so it is copied out with an unaligned read rather than
        // referenced in place.
        let header: MmapHeader =
            unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const MmapHeader) };
        if &header.magic != MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        let table_start = std::mem::size_of::<MmapHeader>();
        let entry_size = std::mem::size_of::<SectionEntry>();
        let count = header.section_count as usize;
        let table_len = count
            .checked_mul(entry_size)
            .filter(|len| table_start + len <= bytes.len())
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "section table overflow")
            })?;
        let mut sections = Vec::with_capacity(count);
        for chunk in bytes[table_start..table_start + table_len].chunks_exact(entry_size) {
            // SAFETY: `chunk` holds exactly one plain-data `SectionEntry`; unaligned read
            // for the same reason as the header.
            sections
                .push(unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const SectionEntry) });
        }
        Ok(Header { key_count: header.key_count, sections })
    }

    /// Get the byte slice for a section by kind. Returns None if not present.
    pub fn section(&self, header: &Header, kind: SectionKind) -> Option<&[u8]> {
        let entry = header.sections.iter().find(|e| e.kind == kind as u32)?;
        let bytes = self.as_bytes();
        let start = usize::try_from(entry.offset).ok()?;
        let len = usize::try_from(entry.length).ok()?;
        let end = start.checked_add(len)?;
        bytes.get(start..end)
    }
}

#[derive(Debug, Clone)]
pub struct Header {
    pub key_count: u64,
    pub sections: Vec<SectionEntry>,
}

/// Writer that builds an index file with multiple sections.
pub struct MmapIndexWriter {
    file: File,
    pending_sections: Vec<(SectionKind, Vec<u8>)>,
    key_count: u64,
}

impl MmapIndexWriter {
    pub fn create<P: AsRef<Path>>(path: P, key_count: u64) -> std::io::Result<Self> {
        let file =
            OpenOptions::new().create(true).truncate(true).read(true).write(true).open(path)?;
        Ok(Self { file, pending_sections: Vec::new(), key_count })
    }

    pub fn add_section(&mut self, kind: SectionKind, bytes: Vec<u8>) {
        self.pending_sections.push((kind, bytes));
    }

    /// Finalize: write header + section table + sections, each 64-byte aligned.
    pub fn finalize(mut self) -> std::io::Result<()> {
        let section_count = self.pending_sections.len();
        let header_size = std::mem::size_of::<MmapHeader>();
        let table_size = section_count * std::mem::size_of::<SectionEntry>();
        let mut cursor = align_up(header_size + table_size, 64);

        let mut entries = Vec::with_capacity(section_count);
        for (kind, bytes) in &self.pending_sections {
            entries.push(SectionEntry {
                kind: *kind as u32,
                _pad: 0,
                offset: cursor as u64,
                length: bytes.len() as u64,
            });
            cursor = align_up(cursor + bytes.len(), 64);
        }

        // Write header.
        let header = MmapHeader {
            magic: *MAGIC,
            version: 1,
            section_count: section_count as u32,
            key_count: self.key_count,
            flags: 0,
            _reserved: [0; 4],
        };
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const MmapHeader as *const u8,
                std::mem::size_of::<MmapHeader>(),
            )
        };
        self.file.write_all(header_bytes)?;
        for entry in &entries {
            let entry_bytes = unsafe {
                std::slice::from_raw_parts(
                    entry as *const SectionEntry as *const u8,
                    std::mem::size_of::<SectionEntry>(),
                )
            };
            self.file.write_all(entry_bytes)?;
        }

        // Write sections with alignment padding.
        let mut written = header_size + table_size;
        for ((_, bytes), entry) in self.pending_sections.iter().zip(entries.iter()) {
            let target = entry.offset as usize;
            if target > written {
                let pad = vec![0u8; target - written];
                self.file.write_all(&pad)?;
                written = target;
            }
            self.file.write_all(bytes)?;
            written += bytes.len();
        }
        self.file.flush()?;
        Ok(())
    }
}

fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}
