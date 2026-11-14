# Streaming storage, hot-tracker maintenance and Hybrid routing

The changes stream mmap output without section-sized copies or a second file
read, bound collision-chain deletion in SpaceSaving, move dynamic tracker
extraction outside its mutex, and derive a radix directory for larger Hybrid
segment sets. Existing public signatures and persisted layouts stay compatible.
Temporary memory and ordinary-case timing costs are reported alongside gains.

## Method and correctness

- Baseline: immutable `d10dc11`. Candidate: the change containing this report.
  The table uses one pass of the final candidate. An earlier candidate pass was
  superseded after a mechanical Clippy correction to a constant-size field
  iterator; its receipts are retained privately, with no averaging or selection.
- Windows x86-64, Intel Family 6 Model 151 Stepping 2, Rust 1.100.0
  (`0ef1bc386`, LLVM 23.1.5), x86-64-v3, release opt-level 3, fat LTO and one
  codegen unit. Caller pinned to logical core 2; all engine builds sequential.
  Compiler, core, input count and resolved registry versions match both drivers.
- The System allocator wrapper measures requested live/peak heap bytes and
  allocation calls. These exclude allocator metadata, RSS and mapped file pages.
  Inputs/oracle vectors are prepared outside operation peaks. Counter and assert
  overhead is present in both versions. These are local observations, not
  confidence intervals; CPU frequency and background activity are uncontrolled.
- Storage saves a million numeric byte keys in fp32+Bloom or lean mode. It
  measures one `save_mmap`, including create, output, checksum and flush. This is
  a page-cache save without `sync_all` or a durability guarantee. Every original
  MPH slot is checked for a bijection and equals the restored slot. Both saved
  files have byte-identical SHA-256 hashes across revisions. Construction is
  outside the save timer and scratch peak; no operation heap remains afterward.
- Tracker mixed input: K=4096, 500,000 observations, a separated hot set plus a
  wider cold set. Exact counts verify overestimation bounds and membership of
  returned top16 candidates in the exact top16. Collision input: K=2048, 256
  evictions after filling K counters. Inverse SplitMix64 generates distinct
  same-home keys without expensive search; count sums and bounds are checked.
  Unit tests additionally check every probe/key binding and O(K) visit budget.
- Top16 extraction: K=65,536, each cold key observed once and 64 hot keys 10–20
  times. Full exact 16-element output and tie order are checked independently.
  The input is nearly sorted by the output order, so this is favorable to the
  previous full sort. Dynamic extraction measures replacement allocation,
  snapshot processing and old-tracker destruction; next-epoch behavior is
  checked afterward. This sequential benchmark does not measure concurrent
  dropped-observation rate or full-stream popularity bias. A deterministic
  overlapping-observer test separately verifies lock release and rebuild hints.
- Hybrid uses 262,144 numeric keys at epsilon 0/64/2048, plus a 32-key control.
  Timing covers 100,000 scalar `lookup_u64` hits in deterministic shuffled order.
  Every input position and round-trip lookup is checked; numeric SIMD batch
  correctness is checked outside timing. Batch throughput and miss rates are
  not measured. All ten rows pass their oracles in both revisions.

## Storage

Memory below is additional peak heap during saving, excluding the resident index.

| Dataset | Save ms, before → after | Speedup | Peak bytes, before → after | Allocations |
|---|---:|---:|---:|---:|
| fp32 + Bloom, 1M keys | 5.7772 → 3.6112 | 1.60x | 5,858,605 → 74,000 | 23 → 7 |
| lean, 1M keys | 1.9282 → 0.9192 | 2.10x | 482,172 → 73,888 | 19 → 6 |

The files remain 5,793,033 and 416,684 bytes respectively. fp32+Bloom scratch
falls 98.74%. The old implementation already checksummed with bounded read
buffers; section copies caused its O(file-size) scratch. The new implementation
borrows those sections and streams through ChecksumWriter with fixed 64 KiB
checksum and BufWriter buffers. Generic output requires only Write, not Read or
Seek. A Map fallback streams its nested sealed payload/checksum too.

Time remains O(M) for M output bytes; auxiliary memory becomes O(S + fixed
buffers) for S section descriptors. The public MmapIndexWriter still owns the
Vec sections callers supply, preserving its API; finalization adds only bounded
buffers and descriptor metadata. Opening/validating a checksum still reads
Omega(M) bytes. The pre-existing 16-byte big-endian u128 scratch fix and its
regression test remain unchanged.

## SpaceSaving and dynamic extraction

| Operation | Time, before → after | Retained tracker bytes | Peak extra bytes, before → after |
|---|---:|---:|---:|
| mixed observe, ns/observation | 14.549 → 15.468 | 327,680 → 327,680 | 0 → 0 |
| colliding eviction, ns/observation | 941,073.438 → 4,048.047 | 163,840 → 163,840 | 0 → 0 |
| standalone top16, microseconds | 274.600 → 321.000 | 5,242,880 → 5,242,880 | 2,097,152 → 1,048,576 |
| DynamicHotTier take16, microseconds | 555.600 → 778.700 | 5,242,880 → 5,242,880 | 2,097,152 → 6,291,456 |

Backward-shift deletion visits the probe tail once: O(K) worst case instead of
O(K squared) reinsertion under collisions. The adversarial eviction improves
232.48x without extra operation allocation. Normal probing remains expected
O(1), worst O(K); heap maintenance is O(log K). A complete observation is thus
expected O(log K), worst O(K + log K), with O(K) retained memory.

Top-k now selects a partition in O(K), then sorts only k entries, for
O(K + k log k) time and O(K) snapshot space. It returns the same descending-count,
ascending-key order. On this nearly sorted input it is 16.90% slower despite
halving scratch. Mixed observations are 6.32% slower in this pass. These are
included rather than asserting a universal ordinary-case speedup.

Dynamic take16 is 40.15% slower end-to-end and uses three times the extraction
scratch. Its new replacement tracker temporarily adds O(K) memory; allocation,
selection and destruction occur outside the mutex. The critical section is an
O(1) swap plus hint reset, replacing selection and probe-table reset under lock.
The next epoch can receive observations and set its rebuild hint while the old
snapshot is processed. No measured contention-throughput improvement is claimed.
Frequency guarantees apply to accepted observations; the existing try-lock drop
policy can still bias the observed stream. No tracker wire representation exists.

## Hybrid radix directory

Heap excludes prepared input/query arrays. Peak is total extra build heap; queries
allocate zero in every row. Directory bytes are included in retained heap.

| Dataset / segments | Build ms, before → after | Lookup ns, before → after | Heap bytes, before → after | Peak build bytes |
|---|---:|---:|---:|---:|
| epsilon 0 / 131,072 | 38.799 → 26.761 | 309.623 → 92.392 | 31,424,516 → 31,686,664 | 42,958,852 → 42,958,852 |
| epsilon 64 / 22 | 64.005 → 37.773 | 46.073 → 35.902 | 1,806,268 → 1,806,268 | 11,243,804 → 11,243,804 |
| default 2048 / 1 | 41.709 → 36.934 | 33.220 → 25.401 | 1,780,542 → 1,780,542 | 12,782,710 → 12,782,710 |
| tiny 32 keys / 1 | 0.014 → 0.010 | 18.742 → 17.880 | 16,985 → 16,985 | 18,153 → 18,153 |

The many-segment route improves 3.35x, adding 262,148 heap bytes (0.83%) for the
65,537-entry directory. Files remain 9,011,308 / 1,803,672 / 1,780,473 / 16,918
bytes. Small/default controls have fewer than 32 segments and no directory;
their timing differences cannot be attributed to radix routing and should not
be treated as evidence of a new routing algorithm in those cases.

For S segments the directory uses four to eight prefix buckets per segment,
capped at 65,536 buckets. Build/load preprocessing is O(S + P), directory memory
O(P). A prefix with one candidate routes directly; overlapping prefixes binary
search only their candidate range with an exact min/max check. Routing is
O(1 + log S_bucket), worst O(log S) under skew. It makes no uniform-distribution
assumption for correctness. The original per-segment storage and original input
positions are preserved; a new fixed-prefix partition/MPH wire format is not
needed. The directory is rebuilt on load and never serialized. Legacy Hybrid
fixtures preserve their old hash interpretation and byte-identical resaving.

## Reproduce

```text
python performance/compare_storage_hot_hybrid.py --phase all --revision d10dc11 --keys 1000000 --core 2
```

Choose an available core on another host. The script exports immutable baseline
source into ignored target/storage-hot-hybrid-comparison, builds isolated drivers
offline with the pinned registry versions, and runs ten scenarios once each.
It checks whole mmap file hashes and comparison metadata. Existing Cargo cache
dependencies are required; the working checkout is never reset. Exact values
and allocation counts are in storage_hot_hybrid_before.csv and
storage_hot_hybrid_after.csv; resolved versions are in
storage_hot_hybrid_dependencies.lock.
