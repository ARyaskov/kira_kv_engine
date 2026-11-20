//! Bounded-working-memory construction over a one-pass stream of u64 keys.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::{
    BuildConfig, BuildValidationError, CandidateMarks, PART_SALT_MIX, PART_TARGET_KEYS, PartInfo,
    PartJob, PtrHash25Error, PtrHash25Mphf, Scratch, buckets_for_keys, build_part,
    check_array_capacity, mix64, part_of, part_round_salt, read_ptrhash25, slots_for_keys,
    write_ptrhash25,
};

const FANOUT: usize = 64;
const FILE_BUFFER: usize = 4096;
// Covers at most 64 buffered files, reader/record buffers, paths, recursion and
// vector/object bookkeeping. Index output arrays and caller input are excluded.
const FIXED_WORKING_BYTES: usize = 512 * 1024;
const DEFAULT_BUDGET: usize = 8 * 1024 * 1024;

/// Failure to construct or load a streamed low-level MPH index.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MphExternalBuildError {
    #[error("invalid MPH configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("MPH capacity exceeded: {0}")]
    CapacityExceeded(&'static str),
    #[error("working-memory budget exceeded: need {required} bytes, budget is {budget}")]
    BudgetExceeded { required: usize, budget: usize },
    #[error("input key count differs: expected {expected}, observed {observed}")]
    KeyCountMismatch { expected: usize, observed: usize },
    #[error("an MPH requires at least one key")]
    EmptyInput,
    #[error("MPH construction failed: {0}")]
    Build(#[from] PtrHash25Error),
    #[error("temporary file I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("corrupt MPH payload")]
    CorruptData,
}

impl From<BuildValidationError> for MphExternalBuildError {
    fn from(error: BuildValidationError) -> Self {
        match error {
            BuildValidationError::InvalidConfig(message) => Self::InvalidConfig(message),
            BuildValidationError::CapacityExceeded(message) => Self::CapacityExceeded(message),
        }
    }
}

/// An immutable low-level u64 MPH. Its keys must be canonicalized by the caller
/// when they originate as byte strings; this type performs no byte-key hashing.
/// The private tables preserve the validated invariants used by fast lookup.
#[derive(Debug)]
pub struct MphIndex {
    inner: PtrHash25Mphf,
}

impl MphIndex {
    pub fn len(&self) -> usize {
        self.inner.n as usize
    }
    pub fn is_empty(&self) -> bool {
        self.inner.n == 0
    }

    /// Return the minimal slot for a known key. Foreign keys get an arbitrary
    /// in-range slot; this operation does not verify membership.
    pub fn index_u64(&self, key: u64) -> u32 {
        self.inner.index_u64(key)
    }

    /// Apply the configured inner fp8 check. Positive answers remain
    /// probabilistic; without fingerprints every key receives a slot.
    pub fn lookup_u64(&self, key: u64) -> Option<u32> {
        self.inner.lookup_u64(key)
    }

    pub fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    /// Write the existing raw PtrHash25 payload; no new envelope is introduced.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_ptrhash25(&self.inner, &mut bytes);
        bytes
    }

    /// Validate an existing raw PtrHash25 payload and require full consumption.
    /// This raw format has structural validation, not an outer checksum.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MphExternalBuildError> {
        let mut position = 0;
        let inner =
            read_ptrhash25(bytes, &mut position).ok_or(MphExternalBuildError::CorruptData)?;
        if position != bytes.len() {
            return Err(MphExternalBuildError::CorruptData);
        }
        Ok(Self { inner })
    }
}

/// Build from a one-pass u64 iterator using temporary radix partitions.
///
/// The budget bounds auxiliary key/scratch/buffer allocations, excluding the
/// final index arrays and storage already owned by the caller/iterator. Parts
/// are built sequentially. Oversized/skewed parts return `BudgetExceeded` before
/// being loaded instead of silently growing past the budget. Mapping may differ
/// from the ordinary builder because prerotation is fixed at zero.
#[derive(Debug, Clone)]
pub struct MphExternalBuilder {
    config: BuildConfig,
    memory_budget: usize,
}

impl Default for MphExternalBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MphExternalBuilder {
    pub fn new() -> Self {
        Self { config: BuildConfig::default(), memory_budget: DEFAULT_BUDGET }
    }

    pub fn with_config(mut self, config: BuildConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_memory_budget_bytes(mut self, bytes: usize) -> Self {
        self.memory_budget = bytes;
        self
    }

    /// Consume exactly `expected_len` unique keys, with temporary files beneath
    /// `scratch_dir`. Success requires removal of the owned temporary directory;
    /// failure/unwind uses best-effort RAII cleanup. Existing files are untouched.
    /// The input goes straight to the first partition level. Disk partitioning uses
    /// O(N (1 + ceil(log_64 P))) records for P parts, avoiding an
    /// O(N) in-memory copy. Resident output still requires O(N) index memory.
    pub fn build<I: IntoIterator<Item = u64>, P: AsRef<Path>>(
        self,
        keys: I,
        expected_len: usize,
        scratch_dir: P,
    ) -> Result<MphIndex, MphExternalBuildError> {
        self.config.validate_for_len(expected_len)?;
        if expected_len == 0 {
            return Err(MphExternalBuildError::EmptyInput);
        }
        self.max_part_keys(0)?;
        let provided_path_bytes = scratch_dir.as_ref().as_os_str().len();
        if provided_path_bytes > self.memory_budget / 16 {
            return Err(MphExternalBuildError::BudgetExceeded {
                required: provided_path_bytes.saturating_mul(16),
                budget: self.memory_budget,
            });
        }
        let temporary = tempfile::Builder::new().prefix("kira-mph-").tempdir_in(scratch_dir)?;
        // Recursion retains at most six groups of 64 path names (u32 part count).
        // The fixed reserve covers ordinary paths; reserve additional bytes for
        // long prefixes before cloning them or consuming the input stream.
        let path_extra = temporary
            .path()
            .as_os_str()
            .len()
            .saturating_add(64)
            .saturating_sub(128)
            .checked_mul(1024)
            .ok_or(MphExternalBuildError::CapacityExceeded(
                "temporary path bookkeeping overflow",
            ))?;
        let max_keys = self.max_part_keys(path_extra)?;
        let target = PART_TARGET_KEYS.min((max_keys / 2).max(1));
        let part_count = expected_len
            .div_ceil(target)
            .checked_next_power_of_two()
            .ok_or(MphExternalBuildError::CapacityExceeded("partition count overflow"))?;
        if part_count > u32::MAX as usize {
            return Err(MphExternalBuildError::CapacityExceeded("too many partitions"));
        }
        validate_output_capacities(part_count, 0, 0)?;
        let part_salt = mix64(self.config.seed ^ PART_SALT_MIX);
        let directory = temporary.path();
        let (paths, mut outputs, child_leaves) = create_partitions(directory, 0, part_count)?;
        let mut observed = 0usize;
        for key in keys {
            observed = observed
                .checked_add(1)
                .ok_or(MphExternalBuildError::CapacityExceeded("input count overflow"))?;
            if observed > expected_len {
                return Err(MphExternalBuildError::KeyCountMismatch {
                    expected: expected_len,
                    observed,
                });
            }
            let child = part_of(key, part_salt, part_count) / child_leaves;
            outputs[child].write_all(&key.to_le_bytes())?;
        }
        if observed != expected_len {
            return Err(MphExternalBuildError::KeyCountMismatch {
                expected: expected_len,
                observed,
            });
        }
        flush_partitions(outputs)?;
        split_children(directory, paths, 0, child_leaves, part_count, part_salt)?;

        // Geometry is final index metadata. Check every leaf's working set before
        // allocating its key/scratch arrays or any final output tables.
        let mut infos = Vec::with_capacity(part_count);
        let mut total_keys = 0u64;
        let mut total_buckets = 0usize;
        let mut total_remap = 0usize;
        for part in 0..part_count {
            let count = leaf_count(temporary.path(), part)?;
            let count_u32 = u32::try_from(count).map_err(|_| {
                MphExternalBuildError::CapacityExceeded("partition key count exceeds u32")
            })?;
            let required = working_bytes(count, &self.config)?
                .checked_add(path_extra)
                .ok_or(MphExternalBuildError::CapacityExceeded("working set size overflow"))?;
            if required > self.memory_budget {
                return Err(MphExternalBuildError::BudgetExceeded {
                    required,
                    budget: self.memory_budget,
                });
            }
            let slots = slots_for_keys(count, self.config.alpha);
            let buckets = buckets_for_keys(count, self.config.lambda);
            let slots_u32 = u32::try_from(slots)
                .map_err(|_| MphExternalBuildError::CapacityExceeded("part slots exceed u32"))?;
            let buckets_u32 = u32::try_from(buckets)
                .map_err(|_| MphExternalBuildError::CapacityExceeded("part buckets exceed u32"))?;
            let bucket_offset = u32::try_from(total_buckets).map_err(|_| {
                MphExternalBuildError::CapacityExceeded("bucket offsets exceed u32")
            })?;
            let remap_offset = u32::try_from(total_remap)
                .map_err(|_| MphExternalBuildError::CapacityExceeded("remap offsets exceed u32"))?;
            infos.push(PartInfo::new(
                if count == 0 { 0 } else { total_keys },
                0,
                bucket_offset,
                count_u32,
                slots_u32,
                buckets_u32,
                remap_offset,
            ));
            total_keys += count as u64;
            total_buckets = total_buckets
                .checked_add(buckets)
                .ok_or(MphExternalBuildError::CapacityExceeded("bucket count overflow"))?;
            total_remap = total_remap
                .checked_add(slots - count)
                .ok_or(MphExternalBuildError::CapacityExceeded("remap count overflow"))?;
        }
        if total_buckets > u32::MAX as usize || total_remap > u32::MAX as usize {
            return Err(MphExternalBuildError::CapacityExceeded("MPH table offsets exceed u32"));
        }
        if total_keys != expected_len as u64 {
            return Err(MphExternalBuildError::KeyCountMismatch {
                expected: expected_len,
                observed: usize::try_from(total_keys).unwrap_or(usize::MAX),
            });
        }
        validate_output_capacities(part_count, total_buckets, total_remap)?;
        let mut pilots = crate::hugepage::HugeVec::<u8>::zeroed(total_buckets);
        let mut remap = vec![0u32; total_remap];
        let mut fingerprints =
            if self.config.with_fingerprints { vec![0u8; expected_len] } else { Vec::new() };
        let global_salt = (part_count > 1).then(|| part_round_salt(self.config.seed, 0, 0));
        // All three dimensions are monotone in the leaf key count. Allocate once
        // for the largest leaf, whose budget was checked before any build.
        let largest_leaf = infos.iter().max_by_key(|info| info.num_keys).unwrap();
        let mut keys = Vec::with_capacity(largest_leaf.num_keys as usize);
        let mut scratch = bounded_scratch(
            largest_leaf.num_keys as usize,
            largest_leaf.num_buckets as usize,
            largest_leaf.num_slots as usize,
        );
        for (part, info) in infos.iter_mut().enumerate() {
            let count = info.num_keys as usize;
            read_leaf(temporary.path(), part, count, &mut keys)?;
            let pilot_start = info.bucket_off as usize;
            let remap_start = info.remap_off as usize;
            let fingerprint_start = info.slot_off as usize;
            let mut job = PartJob {
                part_id: part,
                keys: &keys,
                info: *info,
                global_salt,
                prerotate: 0,
                pilots: &mut pilots.as_mut_slice()
                    [pilot_start..pilot_start + info.num_buckets as usize],
                remap: &mut remap[remap_start..remap_start + info.remap_len()],
                slots: None,
                fp8: self
                    .config
                    .with_fingerprints
                    .then(|| &mut fingerprints[fingerprint_start..fingerprint_start + count]),
                fp16: None,
                packed: None,
                exact_u64_slots: None,
            };
            info.salt = build_part(&mut job, &mut scratch, &self.config)?;
        }
        let salt = global_salt.unwrap_or(infos[0].salt);
        temporary.close()?;
        Ok(MphIndex {
            inner: PtrHash25Mphf {
                n: expected_len as u64,
                num_buckets: total_buckets as u32,
                salt,
                pilots,
                remap: remap.into(),
                fingerprints,
                use_aes_hash: self.config.use_aes_hash,
                prerotate: 0,
                part_salt,
                parts: infos.into_boxed_slice(),
            },
        })
    }

    fn max_part_keys(&self, path_extra: usize) -> Result<usize, MphExternalBuildError> {
        let minimum = working_bytes(1, &self.config)?
            .checked_add(path_extra)
            .ok_or(MphExternalBuildError::CapacityExceeded("working set size overflow"))?;
        if minimum > self.memory_budget {
            return Err(MphExternalBuildError::BudgetExceeded {
                required: minimum,
                budget: self.memory_budget,
            });
        }
        let (mut low, mut high) = (1, PART_TARGET_KEYS * 2);
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if working_bytes(middle, &self.config).is_ok_and(|required| {
                required
                    .checked_add(path_extra)
                    .is_some_and(|required| required <= self.memory_budget)
            }) {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        Ok(low)
    }
}

fn validate_output_capacities(
    parts: usize,
    buckets: usize,
    remap: usize,
) -> Result<(), MphExternalBuildError> {
    check_array_capacity::<PartInfo>(parts)?;
    check_array_capacity::<u8>(buckets)?;
    check_array_capacity::<u32>(remap)?;
    Ok(())
}

fn working_bytes(keys: usize, config: &BuildConfig) -> Result<usize, MphExternalBuildError> {
    let buckets = buckets_for_keys(keys, config.lambda);
    let slots = slots_for_keys(keys, config.alpha);
    keys.checked_mul(48)
        .and_then(|bytes| bytes.checked_add(buckets.checked_mul(16)?))
        .and_then(|bytes| bytes.checked_add(slots.checked_mul(4)?))
        .and_then(|bytes| bytes.checked_add(slots.div_ceil(64).checked_mul(16)?))
        .and_then(|bytes| bytes.checked_add(FIXED_WORKING_BYTES + 16))
        .ok_or(MphExternalBuildError::CapacityExceeded("working set size overflow"))
}

fn bounded_scratch(keys: usize, buckets: usize, slots: usize) -> Scratch {
    // At most one work-list entry exists per bucket: evicting clears all of an
    // owner's slots before enqueueing it, so a pending owner cannot be requeued.
    // Every other buffer is pre-sized for the worst bucket in this part, and the
    // unstable duplicate sort uses no heap scratch.
    Scratch {
        h1: vec![0; keys],
        bidx: vec![0; keys],
        items_h2: vec![0; keys],
        offsets: vec![0; buckets + 1],
        cursor: vec![0; buckets + 1],
        order: vec![0; buckets],
        slots: super::SlotOwners { occupied: vec![0; slots.div_ceil(64)], owners: vec![0; slots] },
        stack: Vec::with_capacity(buckets),
        trial: vec![0; keys.max(1)],
        freq: vec![0; keys + 1],
        start: vec![0; keys + 1],
        duplicate_keys: Vec::with_capacity(keys),
        candidate_marks: CandidateMarks {
            words: vec![0; slots.div_ceil(64)],
            #[cfg(test)]
            visits: 0,
        },
    }
}

fn leaf_path(directory: &Path, part: usize) -> std::path::PathBuf {
    directory.join(format!("part-{part:08x}.keys"))
}

fn leaf_count(directory: &Path, part: usize) -> Result<usize, MphExternalBuildError> {
    let bytes = std::fs::metadata(leaf_path(directory, part))?.len();
    if !bytes.is_multiple_of(8) {
        return Err(MphExternalBuildError::CorruptData);
    }
    usize::try_from(bytes / 8)
        .map_err(|_| MphExternalBuildError::CapacityExceeded("partition too large"))
}

fn read_leaf(directory: &Path, part: usize, count: usize, keys: &mut Vec<u64>) -> io::Result<()> {
    let mut file = BufReader::with_capacity(FILE_BUFFER, File::open(leaf_path(directory, part))?);
    keys.clear();
    for _ in 0..count {
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes)?;
        keys.push(u64::from_le_bytes(bytes));
    }
    Ok(())
}

fn split_file(
    directory: &Path,
    source: &Path,
    first: usize,
    leaves: usize,
    all_parts: usize,
    salt: u64,
) -> io::Result<()> {
    let (paths, mut outputs, child_leaves) = create_partitions(directory, first, leaves)?;
    let mut input = BufReader::with_capacity(FILE_BUFFER, File::open(source)?);
    let mut distribute = |bytes: &[u8; 8]| {
        let key = u64::from_le_bytes(*bytes);
        let child = (part_of(key, salt, all_parts) - first) / child_leaves;
        outputs[child].write_all(bytes)
    };
    loop {
        let buffer = input.fill_buf()?;
        if buffer.is_empty() {
            break;
        }
        let (records, remainder) = buffer.as_chunks::<8>();
        for record in records {
            distribute(record)?;
        }
        if records.is_empty() {
            // A short read can split a record across buffer fills.
            let mut record = [0; 8];
            let prefix = remainder.len();
            record[..prefix].copy_from_slice(remainder);
            input.consume(prefix);
            input.read_exact(&mut record[prefix..])?;
            distribute(&record)?;
        } else {
            let consumed = records.len() * 8;
            input.consume(consumed);
        }
    }
    flush_partitions(outputs)?;
    drop(input);
    std::fs::remove_file(source)?;
    split_children(directory, paths, first, child_leaves, all_parts, salt)
}

fn create_partitions(
    directory: &Path,
    first: usize,
    leaves: usize,
) -> io::Result<(Vec<PathBuf>, Vec<BufWriter<File>>, usize)> {
    let fanout = FANOUT.min(leaves);
    let child_leaves = leaves / fanout;
    let mut paths = Vec::with_capacity(fanout);
    let mut outputs = Vec::with_capacity(fanout);
    for child in 0..fanout {
        let start = first + child * child_leaves;
        let path = if child_leaves == 1 {
            leaf_path(directory, start)
        } else {
            directory.join(format!("split-{start:08x}-{child_leaves:08x}.keys"))
        };
        outputs.push(BufWriter::with_capacity(FILE_BUFFER, File::create(&path)?));
        paths.push(path);
    }
    Ok((paths, outputs, child_leaves))
}

fn flush_partitions(outputs: Vec<BufWriter<File>>) -> io::Result<()> {
    for mut output in outputs {
        output.flush()?;
    }
    Ok(())
}

fn split_children(
    directory: &Path,
    paths: Vec<PathBuf>,
    first: usize,
    child_leaves: usize,
    all_parts: usize,
    salt: u64,
) -> io::Result<()> {
    if child_leaves > 1 {
        for (child, path) in paths.into_iter().enumerate() {
            split_file(
                directory,
                &path,
                first + child * child_leaves,
                child_leaves,
                all_parts,
                salt,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_output_addressability_is_checked_without_allocating() {
        let addressable = isize::MAX as usize;
        for (parts, buckets, remap) in [
            (addressable / std::mem::size_of::<PartInfo>() + 1, 0, 0),
            (0, addressable + 1, 0),
            (0, 0, addressable / std::mem::size_of::<u32>() + 1),
        ] {
            assert!(matches!(
                validate_output_capacities(parts, buckets, remap),
                Err(MphExternalBuildError::CapacityExceeded(_))
            ));
        }
        assert!(
            validate_output_capacities(
                addressable / std::mem::size_of::<PartInfo>(),
                addressable,
                addressable / std::mem::size_of::<u32>(),
            )
            .is_ok()
        );
    }

    #[test]
    fn a_skewed_unique_partition_is_rejected_before_loading_it() {
        let directory = tempfile::tempdir().unwrap();
        let builder = MphExternalBuilder::new().with_memory_budget_bytes(1024 * 1024);
        let max_keys = builder.max_part_keys(0).unwrap();
        let count = max_keys + 1;
        let salt = mix64(builder.config.seed ^ PART_SALT_MIX);
        // The selector product is exactly `key`, so every distinct input lands
        // in part zero for every feasible partition count in this tiny build.
        let keys = (0..count as u64).map(|key| key.wrapping_mul(0xF1DE_83E1_9937_733D) ^ salt);
        let error = builder.build(keys, count, directory.path()).unwrap_err();
        assert!(
            matches!(error, MphExternalBuildError::BudgetExceeded { required, budget } if required > budget)
        );
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn radix_partitioning_recurses_without_losing_or_reordering_membership() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("root.keys");
        let keys: Vec<u64> = (0..1024).map(mix64).collect();
        let mut file = File::create(&source).unwrap();
        for &key in &keys {
            file.write_all(&key.to_le_bytes()).unwrap();
        }
        drop(file);
        let salt = 17;
        let parts = 128;
        split_file(directory.path(), &source, 0, parts, parts, salt).unwrap();
        let mut observed = Vec::new();
        for part in 0..parts {
            let count = leaf_count(directory.path(), part).unwrap();
            let mut leaf = Vec::new();
            read_leaf(directory.path(), part, count, &mut leaf).unwrap();
            assert!(leaf.iter().all(|&key| part_of(key, salt, parts) == part));
            observed.extend(leaf);
        }
        observed.sort_unstable();
        let mut expected = keys;
        expected.sort_unstable();
        assert_eq!(observed, expected);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), parts);
    }
}
