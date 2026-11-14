# Static Index optimization and audit items 13–15

The paired `million_build` run reduced default Index lookup time by **6.3% to
19.4%**, depending on workload, with unchanged reported index size. Default
build time fell **9.0%**. The first three lean baseline timings are outliers;
their large apparent improvements do not establish a fourfold algorithmic
speedup. Smaller single-thread cases do not all improve; their measured
regressions and construction-memory tradeoffs are included below.

## Workload and comparison controls

The baseline library is frozen at `824e0d1`; the candidate is the working tree
based on that revision. Both run the same `million_build` example and the same
added, untimed correctness validator. The mixed-key generator and timed lookup
functions are shared. Each process builds **40,000,000 keys**, then performs
**10,000,000 queries per workload and cache label**, for both default and lean
Index. Negative queries contain 70% known keys and 30% foreign keys.

There is **one pass per source state** (`KIRA_BENCH_RUNS=1`), not a repeated
statistical experiment. The example's displayed median and p95 are therefore
the same observation. Times below are aggregate elapsed time divided by query
count across **20 Rayon lookup workers**, not single-query latency. The workers
are not individually pinned: the printed lookup core list must not be read as
proof of per-worker affinity. The build pool has **8 workers**, explicitly
assigned cores `0,2,4,6,8,10,12,14`.

Both runs use Windows x86-64/MSVC, `rustc 1.100.0`, release optimization, fat LTO,
and one codegen unit. Saved metadata verifies identical settings, compiler,
normalized example and validator hashes, and resolved registry package versions.
The comparison wrapper's tuple-versus-JSON-list equality error was corrected
using the existing metadata; neither timed workload was rerun for that fix.

`cold` means the example does not perform its extra warm-up pass. It does not
flush CPU caches or the OS page cache. `warm` performs an untimed lookup pass
before the measured pass. Query-reference preparation and shuffling happen
outside the lookup timer; result allocation and consumption remain inside it.

## All twelve lookup rows

Time reduction is `(before - after) / before`. Values use the example's printed
two-decimal timings; small differences include that rounding.

| Mode | Workload | Cache label | Before, ns/query | After, ns/query | Time reduction |
|---|---|---|---:|---:|---:|
| Default | positive | cold | 19.21 | 15.95 | 17.0% |
| Default | positive | warm | 17.99 | 14.50 | 19.4% |
| Default | negative | cold | 15.54 | 13.45 | 13.4% |
| Default | negative | warm | 16.12 | 13.54 | 16.0% |
| Default | zipf | cold | 15.99 | 14.99 | 6.3% |
| Default | zipf | warm | 16.96 | 15.07 | 11.1% |
| Lean | positive | cold | 37.31 | 8.52 | 77.2% |
| Lean | positive | warm | 34.76 | 8.60 | 75.3% |
| Lean | negative | cold | 45.83 | 8.75 | 80.9% |
| Lean | negative | warm | 10.82 | 8.94 | 17.4% |
| Lean | zipf | cold | 10.00 | 8.49 | 15.1% |
| Lean | zipf | warm | 9.51 | 8.62 | 9.4% |

The lean baseline's first three measurements, **37.31, 34.76 and 45.83 ns/query**,
are inconsistent with its subsequent **9.51 to 10.82 ns/query** measurements and
prior observations around **8 to 9 ns/query**. The candidate consistently lands
at **8.49 to 8.94 ns/query**. This is encouraging, but the paired run alone
cannot separate optimization from scheduling, memory residency, or other host
noise in those outliers. They remain in the table rather than being discarded.
Even the smaller improvements need repeated, interleaved measurements before
being treated as stable speedup estimates.

## Build and memory

| Mode | Build before, ms | Build after, ms | Time reduction | Index before, B/key | Index after, B/key |
|---|---:|---:|---:|---:|---:|
| Default | 897.93 | 817.01 | 9.0% | 3.79 | 3.79 |
| Lean | 757.49 | 743.06 | 1.9% | 0.42 | 0.42 |

Final static-index table layouts and large-build geometry are unchanged. The
reported index footprint remains **3.79 B/key default** and **0.42 B/key lean**
(rounded by the example). These changes do not establish a memory reduction
for the 40M-key index. Build-time scratch and complete-process memory are
different quantities.

| Complete-process metric | Before | After | Change |
|---|---:|---:|---:|
| Peak working set | 12,170,104,832 B (11.3343 GiB) | 12,182,093,824 B (11.3455 GiB) | +0.099% |
| Highest observed private bytes | 12,518,735,872 B (11.6590 GiB) | 12,519,628,800 B (11.6598 GiB) | +0.007% |

The process peak includes generated key payloads, query copies, both index modes,
and the untimed correctness oracle. It cannot be attributed to index tables or
one build operation. Windows reports a working-set high-water mark; private
bytes are sampled every 250 ms and can miss a shorter peak. Neither metric shows
a meaningful reduction in this run. Whole-process elapsed time, including data
generation, queries and validation, was **84.06 s before** and **76.45 s after**;
that is not an isolated index-build measurement.

## Correctness and compatibility evidence

Each source state produced twelve timing rows and six successful full validation
receipts. Outside the timers, the validator binds every input's original bytes
to its scalar lookup slot and rejects missing, duplicate or out-of-range slots.
It then checks the actual byte-batch outputs for every workload: every known
query must return the slot bound to that exact original key, every output must
be in range, and query/output counts must agree. The timed XOR is not the
correctness oracle.

Foreign candidates are counted separately because default Index membership is
probabilistic and lean Index provides no membership check. Both source states
returned **28 candidates among 3,000,000 foreign queries** in default mode and
**3,000,000 among 3,000,000** in lean mode. The latter is expected closed-world
behavior, not evidence that all foreign keys are present. A default positive
answer also requires a full-key comparison if exact membership is required.

The new byte hashing path preserves the selected current or legacy canonical
hash version, including short-key encoding, and retains scalar fallback paths.
Existing `Index` entry points and existing serialized index/filter/MPH readers
remain available. New tests cover legacy hashes, map fallback, mapped formats,
membership widths, partial batches, empty input, caller-owned output, and
parallel byte batches. The complete default and no-default suites passed;
verification totals are recorded below.

## What changed, and its complexity

The byte pipeline hashes a block with one canonical-version/CPU dispatch, rather
than dispatching for each variable-length key. It prefetches the independent
payload addresses of full 64-query blocks. Lean MPH skips the fingerprint-stage
pass and slot scratch it does not need. The allocating wrapper writes each
output once instead of first filling the complete result vector with `None`.
For total input payload length `L` and `Q` queries, hashing remains `O(L + Q)`
and MPH lookup remains `O(Q)`; these are changes to constant factors and memory
traffic, not asymptotic lookup complexity.

The additive `Index::lookup_batch_pipelined_into(keys, out)` API reuses
caller-owned results. It overwrites the requested output prefix, preserves
query order and any extra output suffix, and panics when `out` is too short.
Its lookup path allocates nothing and uses bounded 64-query stack scratch.
The existing allocating API still returns its `Vec<Option<usize>>`, requiring
`O(Q)` output memory. `million_build` continues using that existing API, so the
new caller-owned variant is not responsible for its reported timings.

Audit item 13 changes split-block Bloom probes on AVX2 x86-64 to compute all
eight wrapped lane products in vector registers and test both halves of the
cache line directly. The scalar fallback and both serialized mask shifts
(current 26, legacy 27) retain their formulas. A probe remains `O(1)`. Tiny
filters now allocate complete cache lines matching their bit budget instead of
rounding every filter up to 256 lines (16 KiB); the minimum is 64 bytes. Large
filter geometry and the parallel build path remain unchanged, so the tiny-filter
saving does not reduce the 40M-key index footprint.

Audit item 14 keeps the inexpensive pairwise path for buckets of at most
32 keys. For a larger bucket of size `b`, duplicate detection sorts a reusable
copy in **`O(b log b)`** time and `O(b)` scratch, preserving source order and
deterministic eviction order. Pilot self-collision checking becomes **`O(b)`**
per attempted pilot, replacing an `O(b^2)` scan. A lazy per-worker bitset uses
`ceil(S / 64)` words for a part with `S` slots; only touched slots are cleared.
It introduces no full-table clear per candidate or generation-counter wrap.
This bounds those inner checks, not the complete randomized/evicting MPH build:
placement retries, salts and configured load still affect total construction
time. Ordinary small buckets do not allocate the large-candidate bitset.

Audit item 15 adds the shared benchmark oracle and controlled source comparison.
The validator costs `O(N)` slot-binding memory plus linear known-key/query
passes, all outside the timers. Its independent fault-injection tests reject
wrong slots, false negatives and length errors; foreign candidates remain
observable rather than being mislabeled as exact hits.

## External construction: additive bounded-memory option

`MphExternalBuilder` consumes a one-pass `u64` iterator with an expected count,
a scratch directory and an auxiliary-memory budget. Byte-key users must
canonicalize/prehash their keys themselves; this low-level option does not
replace `IndexBuilder`'s byte API. It spools records and recursively partitions
them with fanout at most 64, then builds one bounded part at a time. For `N`
keys and `P` output parts, partition I/O is
**`O(N * (1 + ceil(log_64 P)))` records**. It avoids an `O(N)` in-memory input
copy, at the cost of disk space, additional I/O and sequential part construction.

The budget includes auxiliary key arrays, sort copies, slot ownership,
candidate marks, work queues and I/O buffers, with conservative bookkeeping and
path reserves. It **excludes the final resident index arrays and storage already
owned by the caller/iterator**. The result still requires `O(N)` resident output;
this is not a disk-resident query engine or a total-process RAM cap. A skewed
partition that does not fit returns typed `BudgetExceeded` before loading its
key array. Other typed failures include wrong input count, configuration or
capacity errors, duplicate/build failure, and I/O failure.

The opaque immutable `MphIndex` preserves lookup invariants. `index_u64` is
closed-world; `lookup_u64` applies optional probabilistic inner fp8. Its
`to_bytes`/`from_bytes` use the existing **raw PtrHash25 payload**, with validated
structure and full input consumption, but no new outer checksum envelope.
Prerotation is fixed at zero, so mapping can differ from the in-memory builder.
Owned temporary files are removed on successful return; failure/unwind cleanup
is best effort through RAII, and pre-existing caller files are preserved.
This option adds `tempfile` as a normal dependency. Its measured time/memory
tradeoff is recorded below.

## Bloom and isolated BinaryFuse8 comparison

The paired filter driver completed once per source state, at **1, 257, 4,096 and
500,000 keys**, with **1,000,000 known-hit and 1,000,000 foreign-key probes per
row**. Distinct source ranges passed through the same SplitMix64 permutation
make the foreign set disjoint by construction. Every inserted key is checked
outside timing, BlockBloom additionally roundtrips its existing raw payload,
and every query loop reports zero allocations. All eight rows per state passed.

`BinaryFuse8` from **`xorf 0.13.0`** is an isolated comparison dependency; it was
not added as an engine backend or selected automatically. Its implementation is
unchanged between these source states. Each cell below shows **before -> after**;
FPR is the observed foreign-candidate percentage, not an exact-membership claim.

| Filter | Keys | Build, ms | Retained heap, B | Hit, ns/query | Miss, ns/query | Observed FPR |
|---|---:|---:|---:|---:|---:|---:|
| BlockBloom | 1 | 0.0081 -> 0.0058 | 16,384 -> 64 | 2.338 -> 2.151 | 2.241 -> 2.141 | 0 -> 0% |
| BinaryFuse8 | 1 | 0.0055 -> 0.0033 | 12 -> 12 | 2.251 -> 2.409 | 2.302 -> 2.457 | 0.3851 -> 0.3851% |
| BlockBloom | 257 | 0.0050 -> 0.0021 | 16,384 -> 384 | 2.298 -> 1.965 | 2.189 -> 1.983 | 0 -> 0.4841% |
| BinaryFuse8 | 257 | 0.0159 -> 0.0129 | 448 -> 448 | 2.336 -> 2.255 | 2.300 -> 2.289 | 0.3985 -> 0.3985% |
| BlockBloom | 4,096 | 0.0119 -> 0.0073 | 16,384 -> 5,632 | 2.321 -> 1.954 | 2.236 -> 1.963 | 0.0015 -> 0.6129% |
| BinaryFuse8 | 4,096 | 0.1123 -> 0.1060 | 5,632 -> 5,632 | 2.388 -> 2.247 | 2.290 -> 2.295 | 0.3860 -> 0.3860% |
| BlockBloom | 500,000 | 2.2833 -> 1.8554 | 834,280 -> 834,280 | 3.120 -> 2.157 | 2.685 -> 2.160 | 0.6585 -> 0.6585% |
| BinaryFuse8 | 500,000 | 14.9758 -> 13.3272 | 573,440 -> 573,440 | 2.799 -> 2.897 | 9.873 -> 2.723 | 0.3818 -> 0.3818% |

Tiny BlockBloom retained heap falls by **99.6%, 97.7% and 65.6%** at the first
three sizes. This removes substantial over-allocation, with a measurable
accuracy tradeoff: the old, oversized 257-key filter had zero observed false
positives and the 4,096-key filter had 15; their smaller replacements have
**4,841 (0.4841%)** and **6,129 (0.6129%)** respectively. Zero in the one-key sample
does not establish a zero false-positive probability. These filters and
BinaryFuse8 were not sized to identical measured FPR, so the table is not a
comparison at equal accuracy.

At 500,000 keys, BlockBloom's reported logical size is **688,184 B**, while its
retained allocator delta is **834,280 B** in both states. The latter includes
allocation capacity/reserves and retained parallel-build bookkeeping visible
to the global allocator, rather than only the logical filter words and inline
object size. Do not substitute one number for the other or infer a precise
overhead breakdown from their difference. Resident filter arrays stay below the
native-hugepage threshold in all four cases, so their storage is captured by the
heap counter.

Construction peak heap at 500,000 keys is **4,918,256 B (4.918 MB)** for BlockBloom
and **12,530,216 B (12.530 MB)** for BinaryFuse8, unchanged between source states.
Prepared input/query vectors and post-timing serialization/oracle allocations
are excluded. These are logical allocation payloads, not process RSS; native
mappings are excluded by the allocator counter in general.

The large BlockBloom probe reduction is observed with unchanged filter size and
FPR. Its exact repeatability still needs repeated measurements. BinaryFuse8's
unchanged implementation also changes timing, especially its 9.873 -> 2.723
ns/query miss result; that difference demonstrates measurement noise and must
not be credited to the engine optimization. Tiny build times are similarly
short enough that timer quantization and host state matter. The raw evidence is
`filter-before.csv` and `filter-after.csv` in the local evidence directory.

## MPH construction and bounded-memory tradeoff

Each case ran once per available source state, using the same driver and
inputs, an eight-worker pinned build pool and a main thread pinned to core 2.
Every successful index maps every known key bijectively and roundtrips the raw
payload with identical slots. The duplicate case must return `DuplicateKey`.
All three baseline and four candidate cases passed.

| Case | Keys | Build before → after, ms | Peak heap before → after, B | Reported index before → after, B |
|---|---:|---:|---:|---:|
| Ordinary, lambda 3 / alpha 0.98 | 1,000,000 | 16.0627 → 15.9527 | 19,988,360 → 20,910,508 | 416,416 → 416,416 |
| Oversized duplicate bucket | 16,385 | 30.8688 → 0.7513 | 723,909 → 854,989 | no output |
| Large buckets, lambda 32 / alpha 0.02 | 8,192 | 1.4800 → 1.3711 | 3,511,736 → 3,563,576 | 1,606,080 → 1,606,080 |
| External, 1 MiB auxiliary budget | 1,000,000 | unavailable → 602.2349 | unavailable → 659,425 | unavailable → 425,950 |

The adversarial duplicate case is **41.1 times faster**, with **131,080 B** of
additional peak heap for its sorted key copy. The large-bucket case takes 7.4%
less time and 1.5% more peak heap. Ordinary build time differs by only 0.7%,
while its observed peak increases **4.6% (922,148 B)**. Ordinary retained
allocator deltas are 416,264 → 418,568 B; the actual index size and wire size
(416,198 B) are unchanged. These process-global counters also include Rayon
task/scratch lifetime effects. Large-key sort copies and marks are lazy; this
single pass does not establish which allocation or scheduling change caused
the ordinary peak increase. It remains an observed cost, not a memory saving.

The streamed case generates all keys from its iterator without an input array.
Its retained allocator delta is **425,798 B**, and its peak is **659,425 B**,
well below the tested `1 MiB + reported output` bound. It is **37.8 times slower**
than the candidate ordinary build in this workload, with about **31.7 times
less peak heap**, excluding the ordinary builder's prepared input array.
Its different partition geometry produces a 424,832-byte wire payload. This
is a capacity option when input working memory matters, rather than a default
speed optimization. The driver also verifies cleanup of owned scratch files.

Native hugepages were unavailable in both MPH runs, so output arrays used the
ordinary allocator. Counters measure logical allocation payloads rather than
RSS; prepared input arrays and subsequent oracle/serialization allocations are
outside each build checkpoint. Ordinary and external construction have
different parallelism and partitioning, explicitly shown here.

## Hash, probe and public-pipeline diagnostics

The separate diagnostic completed twelve rows per source state with matching
input/hash digests and exact results. `compare_static_phases.py` adds methods
only to ignored copies, extracts the actual production hash prefix, and calls
the actual canonical probe function. It uses the frozen `824e0d1` million-key
generator, a scalar hash reference, and an exported-table MPH/Bloom/fingerprint
oracle. These cases use **1M keys, 1M queries, one thread pinned to core 2**.

| Mode | Workload | Hash before → after, ns | Probe before → after, ns | Public pipeline before → after, ns |
|---|---|---:|---:|---:|
| Default | positive | 17.582 → 16.588 | 15.292 → 14.409 | 32.725 → 32.939 |
| Default | 70% hits / 30% foreign | 16.718 → 14.182 | 20.011 → 22.547 | 38.145 → 38.983 |
| Lean | positive | 17.392 → 14.935 | 11.024 → 10.293 | 30.280 → 24.573 |
| Lean | 70% hits / 30% foreign | 16.870 → 14.934 | 10.864 → 9.933 | 29.189 → 24.354 |

The hash fragment improves in all four cases. Lean public pipelines take 18.8%
and 16.6% less time. The default public pipeline instead takes **0.7% and 2.2%
more time**, including a slower isolated mixed-query probe. This prevents a
claim that the candidate accelerates every index size/thread count. One pass
cannot distinguish such small pipeline differences from noise; the values are
retained without rerunning for a more favorable result.

Hash/probe phases allocate **zero bytes in zero allocations** in both states.
The existing allocating public pipeline uses **one 16,000,000-byte result
allocation** in each state. Reported index memory remains 3,792,744 B default
and 416,432 B lean. The new caller-owned API's zero-allocation behavior is
verified separately by public regression tests; this diagnostic deliberately
times the existing API used by `million_build`.

Hash timing includes an observable output copy; diagnostic boundaries, cache
state, query order and inlining differ from the production pipeline. Stage
times must not be summed and cannot replace the 40M-key/20-worker result.
The long-key scalar reference shares the legacy AES implementation; it is not
an independently reimplemented AES algorithm. Prepared data/index/oracles are
excluded from phase allocation checkpoints. Optional sampling with `samply`
could not start because its elevated helper required administrator privileges;
no sampled workload ran, and no sampled bottleneck claim is made.

## Verification summary

The final functional source passed **248 tests with default features** and
**246 with no default features**, once per full suite, with zero failures or
ignored tests. This adds **31 regressions** to the previous 217/215 totals.
The suites include frozen ordinary/mmap/legacy payloads, exact scalar/AES
parity, byte-batch tails and output reuse, concurrent batches, malformed
cursors, large collision checks, bounded external construction and cleanup.

`cargo fmt --all -- --check`, Clippy for all targets with all features and with
no default features (`-D warnings`), and rustdoc (`-D warnings`) passed.
`cargo bench --bench engines` also completed all seven engine rows at 32,768
keys with its strengthened known-key/rank/stable-ID and batch assertions.
That smaller execution validates the benchmark oracles; its timings are not
substituted for the historical README engine table.

Audit 13 is implemented as compact tiny filters and direct AVX2 masks, with
the isolated BinaryFuse comparison above. BinaryFuse and ZOR are not added as
new persisted backends: the measured build/memory/FPR tradeoff does not justify
changing the default or wire format in this task. Audit 14 adds bounded inner
checks and the external builder. Audit 15 adds explicit correctness oracles
and reproducible paired comparisons. Original remap compression research is
left as an alternative, since the existing remap bytes and lookup contract
are intentionally preserved.

## Evidence and reproduction

The saved, ignored evidence is under `target/static-index-optimization/`:
`million-before.log`, `million-after.log`, both `million-*-metadata.json` files,
and `million-validation.json`. These contain the twelve rows, six correctness
receipts per state, memory samples and comparison settings. Raw build/test logs
are kept locally because tool output can include machine-specific paths.

From the repository root, PowerShell 7, with dependencies available locally:

```powershell
python performance/compare_million.py --phase all --revision 824e0d1 --keys 40000000 --queries 10000000 --threads 20 --build-threads 8 --build-cores 0,2,4,6,8,10,12,14
```

This performs a new pair of full workloads; it is a reproduction command, not an
additional run behind the numbers in this report. The component drivers are
`filter_compare.rs` and `mph_build_compare.rs`; the latter enables its
`candidate` driver feature only when linked to the candidate library containing
the external-builder API. Component runs can be reproduced with:

```powershell
python performance/compare_components.py --kind filter --phase all
python performance/compare_components.py --kind mph --phase all
python performance/compare_static_phases.py --phase all --keys 1000000 --queries 1000000 --core 2
```

Component helpers preserve existing validated results instead of repeating
them. Dependency lock snapshots are committed beside the drivers. Diagnostic
CSV/metadata are under `target/static-phase-comparison/`; component CSV/metadata
are under `target/static-index-optimization/`. The initial diagnostic compile
failure and MPH dependency-seed validation failure were repaired before their
first workload execution; they did not consume an extra timing pass.
