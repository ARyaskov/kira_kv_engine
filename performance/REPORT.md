# Optimisation verification on 10 million keys

Date: 27 October 2026. Source: `02a44f4953a496da030651a640c52527b083df1b` (v0.7.0), compared with the working tree after the optimisations.

The primary comparison is [paired.csv](paired.csv): both libraries live in one process and use the same 10 million unique eight-byte keys, 1 million random queries, core 2, and the median of nine interleaved measurements. Release build, fat LTO, one codegen unit. Both implementations share their dependencies, pinned in [paired-dependencies.lock](paired-dependencies.lock). Windows x86_64, 20 logical CPUs, Intel Family 6 Model 151.

Writes in the default mode were measured in one full pass that alternates 65,536-key blocks between the implementations. The time includes automatic flush/compact. The background mode was measured in a separate full pass that waits for every started merge. This is not a median of nine builds.

## Before / after

| Metric | Before | After | Reduction |
|---|---:|---:|---:|
| Index: single hit | 80.13 ns/key | 74.84 ns/key | 6.6% |
| Index: single miss | 27.43 ns/key | 14.90 ns/key | 45.7% |
| Index: batch u64 | 61.94 ns/key | 24.18 ns/key | 61.0% |
| Index: batch bytes | 90.30 ns/key | 41.51 ns/key | 54.0% |
| mmap: open and release | 40.88 ms (open + drop) | 16.72 ms (open + drop) | 59.1% |
| DynamicIndex: insert, default mode | 4.38 µs/key | 2.79 µs/key | 36.4% |
| DynamicIndex: resident memory | 48.16 B/key | 28.16 B/key | 41.5% |
| DynamicIndex: lookup, default mode | 704.52 ns/key | 510.25 ns/key | 27.6% |
| DynamicIndex: insert, background merges | 4.38 µs/key | 1.15 µs/key | 73.7% |
| Background mode: lookup across tiers | 686.29 ns/key | 543.93 ns/key | 20.7% |
| Background mode: lookup after compact | 708.78 ns/key | 517.71 ns/key | 27.0% |

In the lookup rows the baseline index is re-measured next to each candidate, so absolute latencies from different rows must not be read as one fixed baseline. The machine is noisy; small single-lookup differences should not be extrapolated to other CPUs. The large batch and memory wins are also confirmed by the staged runs.

## Membership

The default remains Bloom + fp16. Without Bloom a miss pays for a full MPH and fingerprint probe. Every membership mode is probabilistic; fp32 is not an exact byte comparison either.

| Mode | Before, B/key | After, B/key | Memory change |
|---|---:|---:|---:|
| fp16, no Bloom | 3.793 | 2.416 | −36.3% |
| fp24, no Bloom | 3.793 | 3.416 | −9.9% |
| fp32, no Bloom | 3.793 | 4.416 | +16.4% |

In separate 1M-miss samples fp16 without Bloom produced 25 false hits; fp24/fp32 produced 0. This is an observation, not a guarantee of zero error probability.

## mmap memory

Extra peak allocation through the Rust allocator on open: **95,849,403 → 65,608 bytes** (about 99.93% lower). This is a separate metric from [staged_bench.rs](../examples/staged_bench.rs), not RSS. The tables still take about 38 MB, but they are read from the file mapping. Mapped data and the file cache are not counted by the Rust allocator counter. Checksum verification stays mandatory and reads the whole file.

On Windows the safe opener keeps a handle without write/delete sharing, which also rules out existing writable mappings. On Unix the safe opener uses an owned byte buffer; a real OS mmap is available through `unsafe open_mmap_unchecked` when the file is guaranteed to stay immutable. On big-endian targets or with insufficient alignment the arrays are decoded into owned memory. Old payload containers are still read through the copying decoder.

## DynamicIndex policy

The arena and slot-ordered metadata are always on. The default policy of full automatic compaction is preserved. Geometric background compaction is opt-in:

```rust
index.set_background_compaction(true)?;
// After writing, include the outstanding work in the measurement:
index.flush()?;
index.finish_background_compaction()?;
// For single-tier lookups:
index.compact()?;
```

Several tiers can make lookups slower than a freshly compacted index; an early A/B candidate showed a regression of up to 34% against the previous single-tier index. That is why background mode is not the default. Disabling the worker keeps size-tiered merges synchronous; `set_compaction_ratio` also selects size tiering. Deserialization restores the default policy.

## Staged verification

All runs below use `keys=10000000` and end with `correctness=pass`. The harness and thread-pinning settings changed during the work; these CSVs are a log of the stages, while the final summary table uses the single A/B harness.

| Stage | Log |
|---|---|
| Baseline | [baseline.csv](baseline.csv) |
| 1. Zero-copy mmap | [stage1_mmap.csv](stage1_mmap.csv) |
| 2. Three-stage batch | [stage2_pipeline.csv](stage2_pipeline.csv) |
| 3. Bloom and fp16/24/32 | [stage3_membership.csv](stage3_membership.csv) |
| 4. DynamicIndex arena | [stage4_arena.csv](stage4_arena.csv) |
| 5. Background geometric merges | [stage5_compaction.csv](stage5_compaction.csv) |

The harness checks every built MPH position and stable ID. The final A/B additionally compares all 10M keys in the regular and mmap indexes, checks for false negatives at every fingerprint width, and checks every ID in both dynamic modes. The staged harness also checks deletions, revival, and a full serialize/restore of 10M keys.

Final checks: **111 tests with default features and 110 without**, both Clippy builds with `-D warnings`, rustdoc with `-D warnings`, `cargo fmt --check`. The full logs of these runs contain local paths and are not part of the repository; the same Cargo commands reproduce them. The combined check also fixed a pre-existing bug in reading corrupted Elias–Fano select samples.

## Reproduction

PowerShell 7, from the repository root, with Cargo dependencies already fetched:

```powershell
.\scripts\staged_bench.ps1 -Core 2
.\performance\run_paired.ps1 -Core 2
```

`KIRA_STAGE_N` changes the size in the A/B harness; the default is 10M. `-StaticOnly` runs only the static part of the A/B. Source copies are created in ignored directories under `performance/`; the Git state is not changed. The main A/B uses the default system allocator. The staged harness uses an allocation counter; its allocation-heavy latencies include the cost of that counter and are not mixed into the main table.
