//! Section container for on-disk indexes.
//!
//! Layout: a fixed header, a section table, then the sections, each starting at a
//! 64-byte-aligned offset. The alignment is what a future zero-copy loader would
//! need to reference pilots, Bloom words and fingerprints in place.
//!
//! Windows safe opens deny write/delete sharing for the lifetime of the mapping.
//! On other platforms the safe opener retains the owned-byte fallback; callers
//! guaranteeing file immutability can use `open_unchecked` for zero-copy access.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::Path;
use std::sync::Arc;

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
    PtrHash25Parts = 6,
    PtrHash25Remap = 7,

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
    pub(crate) mmap: Arc<MappedFile>,
}

/// Shared mapping owner. Keeping the file handle prevents write/delete opens on Windows.
#[derive(Debug)]
pub struct MappedFile {
    bytes: MappedBytes,
    _file: Option<File>,
}

#[derive(Debug)]
enum MappedBytes {
    Map(memmap2::Mmap),
    Owned(Vec<u8>),
}
impl AsRef<[u8]> for MappedFile {
    fn as_ref(&self) -> &[u8] {
        match &self.bytes {
            MappedBytes::Map(m) => m,
            MappedBytes::Owned(v) => v,
        }
    }
}
impl MappedFile {
    pub(crate) fn is_mapped(&self) -> bool {
        matches!(self.bytes, MappedBytes::Map(_))
    }
}

impl MmapIndex {
    /// Open an index file via memory map (read-only). Returns a reader that gives
    /// zero-copy access to sections.
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            let file =
                OpenOptions::new().read(true).share_mode(1 /* FILE_SHARE_READ */).open(path)?;
            // SAFETY: the retained handle excludes all write/delete handles, including
            // pre-existing ones, until the last mapped table has been dropped.
            unsafe { Self::map_file(file) }
        }
        #[cfg(not(windows))]
        {
            Ok(Self {
                mmap: Arc::new(MappedFile {
                    bytes: MappedBytes::Owned(std::fs::read(path)?),
                    _file: None,
                }),
            })
        }
    }

    /// Map a file without enforcing immutability through Windows sharing rules.
    /// # Safety
    /// The file must not be modified or truncated until this reader and all indexes
    /// derived from it have been dropped. This also applies to other processes.
    pub unsafe fn open_unchecked<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        unsafe { Self::map_file(File::open(path)?) }
    }

    unsafe fn map_file(file: File) -> std::io::Result<Self> {
        let bytes = if file.metadata()?.len() == 0 {
            MappedBytes::Owned(Vec::new())
        } else {
            MappedBytes::Map(unsafe { memmap2::MmapOptions::new().map(&file)? })
        };
        Ok(Self { mmap: Arc::new(MappedFile { bytes, _file: Some(file) }) })
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.mmap.as_ref().as_ref()
    }

    /// Parse the header and return a view of each section by kind.
    pub fn parse_header(&self) -> std::io::Result<Header> {
        let bytes = self.as_bytes();
        if bytes.len() < std::mem::size_of::<MmapHeader>() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "header too short"));
        }
        let header = MmapHeader {
            magic: bytes[..8].try_into().unwrap(),
            version: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            section_count: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            key_count: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            flags: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            _reserved: [0; 4],
        };
        if &header.magic != MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        if !matches!(header.version, 1 | 2) || header.flags != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported mmap format",
            ));
        }
        let payload_end = if header.version == 2 {
            let end = bytes.len().checked_sub(8).ok_or_else(|| invalid("missing checksum"))?;
            let stored = u64::from_le_bytes(bytes[end..].try_into().unwrap());
            if crate::checksum::checksum(&bytes[..end]) != stored {
                return Err(invalid("checksum mismatch"));
            }
            end
        } else {
            bytes.len()
        };
        let table_start = std::mem::size_of::<MmapHeader>();
        let entry_size = std::mem::size_of::<SectionEntry>();
        let count = header.section_count as usize;
        let table_len = count
            .checked_mul(entry_size)
            .filter(|len| table_start.checked_add(*len).is_some_and(|end| end <= payload_end))
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "section table overflow")
            })?;
        let mut sections = Vec::with_capacity(count);
        for chunk in bytes[table_start..table_start + table_len].chunks_exact(entry_size) {
            sections.push(SectionEntry {
                kind: u32::from_le_bytes(chunk[..4].try_into().unwrap()),
                _pad: u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                offset: u64::from_le_bytes(chunk[8..16].try_into().unwrap()),
                length: u64::from_le_bytes(chunk[16..24].try_into().unwrap()),
            });
        }
        let mut end = align_up(table_start + table_len, 64);
        let mut kinds = std::collections::HashSet::new();
        for entry in &sections {
            let start = usize::try_from(entry.offset).map_err(|_| invalid("section offset"))?;
            let len = usize::try_from(entry.length).map_err(|_| invalid("section length"))?;
            if !start.is_multiple_of(64) || start < end || !kinds.insert(entry.kind) {
                return Err(invalid("overlapping or duplicate sections"));
            }
            end = start
                .checked_add(len)
                .filter(|&e| e <= payload_end)
                .ok_or_else(|| invalid("section bounds"))?;
        }
        Ok(Header { version: header.version, key_count: header.key_count, sections })
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

    pub(crate) fn table<T: crate::wire::LeNum + Default + 'static>(
        &self,
        header: &Header,
        kind: SectionKind,
    ) -> Option<crate::hugepage::HugeVec<T>> {
        let entry = header.sections.iter().find(|e| e.kind == kind as u32)?;
        let len = usize::try_from(entry.length).ok()?;
        if !len.is_multiple_of(std::mem::size_of::<T>()) {
            return None;
        }
        let mapped = crate::hugepage::HugeVec::from_mapped(
            self.mmap.clone(),
            usize::try_from(entry.offset).ok()?,
            len / std::mem::size_of::<T>(),
        );
        if mapped.is_some() {
            return mapped;
        }
        // Owned fallback for big-endian hosts or an allocator with insufficient
        // alignment. All on-disk arrays still have little-endian representation.
        let mut pos = usize::try_from(entry.offset).ok()?;
        crate::wire::read_le_at::<T>(self.as_bytes(), &mut pos, len / std::mem::size_of::<T>())
            .map(Into::into)
    }
}

fn invalid(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

#[derive(Debug, Clone)]
pub struct Header {
    pub version: u32,
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
            version: 2,
            section_count: section_count as u32,
            key_count: self.key_count,
            flags: 0,
            _reserved: [0; 4],
        };
        self.file.write_all(&header.magic)?;
        self.file.write_all(&header.version.to_le_bytes())?;
        self.file.write_all(&header.section_count.to_le_bytes())?;
        self.file.write_all(&header.key_count.to_le_bytes())?;
        self.file.write_all(&header.flags.to_le_bytes())?;
        self.file.write_all(&[0; 32])?;
        for entry in &entries {
            self.file.write_all(&entry.kind.to_le_bytes())?;
            self.file.write_all(&entry._pad.to_le_bytes())?;
            self.file.write_all(&entry.offset.to_le_bytes())?;
            self.file.write_all(&entry.length.to_le_bytes())?;
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
        self.file.rewind()?;
        let mut checksum = crate::checksum::Checksum::new();
        let mut buffer = [0; 65536];
        loop {
            let n = self.file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            checksum.update(&buffer[..n]);
        }
        self.file.write_all(&checksum.finish().to_le_bytes())?;
        self.file.flush()?;
        Ok(())
    }
}

fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}
