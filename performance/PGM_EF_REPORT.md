# PGM and Elias-Fano precision/select comparison

The changes fix precision loss for large u64 coordinates, bound Elias-Fano
random access independently of zero gaps, and remove query allocations from
compressed PGM search. They trade additional select metadata and build work for
the bounded query path. Ordinary uncompressed and standalone EF queries are
not uniformly faster; those regressions are included below.

## Method

- Baseline: immutable revision `9f7bdb2`; candidate: the change containing this report.
- One pass per revision, eight datasets, 1,000,000 keys each. No repeated runs or
  median selection. These are local observations, not confidence intervals.
- Windows x86-64, Intel Family 6 Model 151 Stepping 2, Rust 1.100.0
  (`0ef1bc386`, LLVM 23.1.5), x86-64-v3 target, release opt-level 3, fat LTO,
  one codegen unit. The caller is pinned to logical core 2.
- Both drivers use default crate features but explicitly sequential PGM builds.
  Registry dependencies are identical; the resolved baseline lock is retained in
  `pgm_ef_dependencies.lock`.
- A System allocator wrapper counts live requested bytes, peak requested bytes
  and allocation calls in both versions. These are allocator payloads, not RSS,
  and exclude allocator metadata and unrelated input/oracle vectors. Timings
  include the same counter overhead. Input cloning is excluded from build time
  but the PGM-owned key array is included in retained/peak heap.
- PGM: epsilon 0 for dense inputs and 64 for random inputs; 100,000 point queries
  and 100,000 lower-bound queries per dataset. Keys arrive sorted, so build time
  does not measure sorting arbitrary input. Lower bounds use `key + 1`
  (saturating at MAX), with `slice::partition_point` as the independent oracle.
- EF: 100,000 random ranks for regular input; 16,384 accesses to rank 1 for the
  adversarial gap. Materialization decodes the first 64 keys 1,024 times into an
  already reserved vector; the reported time is per decoded key.
- All million known ranks are checked before and after serialization, outside
  timing. All eight rows pass for both revisions. PGM serialization stores full
  keys even when runtime storage is compressed. Ordinary EF wire sizes stay equal.

`dense` is `0..N`. `offset_dense` adds `2^52 + 2^27`. `max_dense` is the final
N consecutive u64 values. `random` is a sorted, deterministic SplitMix64
permutation of `0..N`. Suffix `_ef` enables actual EF compaction. EF `regular`
is `13*i + 7`; `long_gap` is zero followed by N-1 consecutive keys ending at
`next_power_of_two(N)^2`, which forces a long zero run inside the first rank block.

## Measurements

All times are before → after. Heap figures below are retained decimal MB.
Speedup is baseline time divided by candidate time; values below 1 mean slower.

| Engine / dataset | Build ms | Point/get ns | Speedup | Lower bound ns | Heap MB | Peak heap MB |
|---|---:|---:|---:|---:|---:|---:|
| PGM dense | 6.229 → 6.917 | 42.740 → 49.380 | 0.87x | 11.667 → 14.903 | 8.000 → 8.000 | 9.500 → 9.250 |
| PGM dense EF | 6.761 → 9.132 | 123.409 → 109.720 | 1.12x | 57.265 → 36.677 | 0.313 → 0.500 | 9.500 → 9.250 |
| PGM offset dense | 5.006 → 6.921 | 71.897 → 53.933 | 1.33x | 115.023 → 13.474 | 8.000 → 8.000 | 9.500 → 9.250 |
| PGM near u64::MAX | 31.359 → 6.883 | 275.707 → 54.628 | 5.05x | 129.757 → 16.100 | 40.988 → 8.000 | 88.990 → 9.250 |
| PGM random | 4.990 → 6.479 | 132.547 → 137.191 | 0.97x | 52.334 → 56.627 | 8.003 → 8.003 | 9.503 → 9.253 |
| PGM random EF | 9.109 → 12.015 | 719.592 → 388.692 | 1.85x | 648.872 → 229.931 | 5.822 → 6.009 | 13.822 → 14.009 |
| EF regular | 2.573 → 4.016 | 28.633 → 39.866 | 0.72x | — | 0.766 → 0.953 | 0.766 → 0.953 |
| EF long gap | 3.236 → 4.294 | 5620.978 → 2.716 | 2069.58x | — | 2.819 → 3.007 | 2.819 → 3.007 |

EF materialization: regular 4.004 → 4.077 ns/key; long gap
60.007 → 1.596 ns/key (37.60x faster).

Compressed PGM query allocation counts across 200,000 operations:
`dense_ef` 200,000 → 0; `random_ef` 199,987 → 0. The other rows allocate zero
in both versions. Exact byte counts, reported memory estimates and wire sizes
are retained in [`pgm_ef_before.csv`](pgm_ef_before.csv) and
[`pgm_ef_after.csv`](pgm_ef_after.csv).

## Precision, complexity and costs

The shifted dense PGM's measured error drops from 134,217,728 to 0. Near
u64::MAX the segment count drops from 999,512 to 1 and the error from 998,976
to 0. The new model keeps integer key/position origins, stores one relative f64
slope and measures the exact truncated prediction used by lookup. Numeric
storage remains eight bytes per segment. A bounded singleton fallback handles
rare rounding failures; the measured error is at most saturating `epsilon + 1`.
Random input retains 94 segments, with error 65 → 64. Model fitting and error
measurement are O(N); sorting arbitrary input remains O(N log N) worst case.

Old sampled EF select could scan O(N/64) high words through one zero gap, even
for an adjacent rank. New 64-rank groups either span fewer than 2048 bits or
store their positions directly. A get scans at most 33 high words and clears
at most 63 set bits, or reads a sparse position directly: O(1) for fixed word
width. Decoding k consecutive keys jumps across group boundaries and takes
O(k + 1). PGM searches a short compressed window in a 24-key stack buffer and
bisects a larger window directly: O(log S + log W) time and O(1) extra space,
where S is segment count and W is the error window width.

On this 64-bit target each derived directory entry takes 16 bytes instead of
the previous 4-byte sample, adding 187,500 bytes per million keys. Sparse groups
also retain up to 64 usize positions. The long-gap fixture has one sparse group,
adding another 512 bytes. This raises standalone EF retained memory by 24.49%
on regular keys and 6.67% on the long gap. Dense EF-compacted PGM grows 59.99%
from its much smaller baseline, while random EF PGM grows 3.22%. These are
deliberate costs of the bounded access scheme. Build work rises on most inputs;
standalone regular EF gets slow down 39.23%, plain dense PGM gets 15.54%, and
plain random PGM gets 3.50% in this single pass.

New PGM payloads use raw v6 under the existing standalone tag-5 envelope; loaded
legacy v2/v3/v4 absolute models retain their arithmetic and raw-v4 resave layout.
Section metadata distinguishes legacy f32 arrays from relative f64 slopes.
EF ordinary bytes stay compatible; a reserved zero-count plus `KIRAEF02` prefix
selects u64 samples only when a rank sample cannot fit in u32. The derived
directory is rebuilt on load. Frozen legacy fixtures cover public loading,
resaving and Hybrid embedding. Existing API signatures and deprecated options
remain; `PgmError` and the `MphConfig` alias are additive root exports.

## Reproduce

```text
python performance/compare_pgm_ef.py --phase all --revision 9f7bdb2 --keys 1000000 --core 2
```

Choose an available logical core on another host. The script exports baseline
source into ignored `target/pgm-ef-comparison`, builds isolated before/after
drivers offline using the pinned registry versions, and runs each dataset once.
It never checks out or resets the working tree. Offline dependencies must
already exist in the local Cargo cache. Different hardware, allocator, compiler,
query mix and key distribution can change the timing ratios.
