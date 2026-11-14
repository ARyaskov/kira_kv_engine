# Compaction and deletion collection comparison

Date: 14 November 2026. Baseline source: `b78b528`. Candidate: the point 4–6
maintenance changes accompanying this report. Raw results are in
`compaction_before.csv` and `compaction_after.csv`; the harness and runner are
`compaction_compare.rs` and `compare_compaction.py`.

## Workload and measurement

- Windows x86_64, Intel64 Family 6 Model 151 Stepping 2; Rust 1.100.0,
  `x86_64-pc-windows-msvc`, x86-64-v3, release optimization, fat LTO, one codegen unit.
- One pass per implementation/policy, one million unique eight-byte keys,
  flush threshold 4096, configured tier cap 8, fanout 4, synchronous compaction.
  The calling thread is pinned to logical core 2; library builds use the same
  persistent build-pool defaults. Third-party versions match across builds and
  are pinned in `compaction_dependencies.lock`.
- Append time includes automatic flush/merge work, a final `flush` and
  `finish_background_compaction`. Lookup time covers 100,000 random hits after
  maintenance. Then 80% of keys are deleted and maintenance is timed separately.
- Every inserted ID is checked against its input ordinal. After deletion and
  maintenance, all one million present/absent answers are checked, followed by
  another complete check after serialization/restoration. Every run reports
  `correctness=pass`.
- Memory is live/peak **Rust allocator bytes**, measured by a forwarding System
  allocator with atomic counters. Input keys and initial pool initialization are
  excluded by taking the baseline after allocating/warming them. This is not RSS;
  OS mappings, hugepages, native stacks and allocator overhead are not counted.
  Allocator instrumentation also affects the reported timings on both versions.
- These are single observations, not medians or universal speedup guarantees.
  No extra repetitions were run to smooth the machine's timing variation.

## Existing capped size-tiered mode: same API before and after

Both versions use `set_compaction_ratio(4)` and retain the hard global cap of 8.
The candidate merges the smallest adjacent pair when the cap forces a merge,
instead of repeatedly rewriting the oldest base.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Append, including automatic maintenance | 1.262 us/key | 0.842 us/key | time -33.3% |
| Random hit after append | 409 ns/key | 296 ns/key | time -27.6% |
| Records visited by automatic merges | 7,880,704 | 4,190,208 | -46.8% |
| Live heap after append | 26.81 MiB | 26.81 MiB | approximately unchanged |
| Peak heap during append | 81.37 MiB | 46.86 MiB | -42.4% |
| Retained tiers after append | 5 | 6 | +1 tier |
| Live heap after deleting 80% and maintenance | 57.92 MiB | 5.56 MiB | -90.4% |
| Remaining tombstones | 800,000 | 0 | reclaimed |
| Deletion-maintenance time | 0.002 ms | 116.346 ms | candidate performs collection |
| Peak heap during deletion/maintenance | 67.81 MiB | 73.38 MiB | +8.2% |

The old deletion-maintenance time is near zero because no garbage is collected.
The candidate pays for a full atomic rebuild, briefly retaining old data and the
replacement together, then releases dead keys and tombstone table capacity.
That transient peak is a real cost of the memory reduction, not a regression to
hide behind the final resident value.

## Explicit geometric mode versus the former default full-merge policy

This is an intentional policy comparison. The existing default remains Full;
the new policy must be selected explicitly:

```rust
use kira_kv_engine::CompactionPolicy;
index.set_compaction_policy(CompactionPolicy::Geometric)?;
```

| Metric | Old Full policy | New Geometric policy | Change |
|---|---:|---:|---:|
| Append, including automatic maintenance | 2.956 us/key | 0.612 us/key | time -79.3% |
| Records visited by automatic merges | 15,360,000 | 2,768,896 | -82.0% |
| Peak heap during append | 99.11 MiB | 41.40 MiB | -58.2% |
| Retained tiers after append | 5 | 8 | +3 tiers |
| Random hit after append | 218 ns/key | 289 ns/key | time +32.2% |
| Live heap after deleting 80% and maintenance | 57.92 MiB | 5.56 MiB | -90.4% |
| Deletion-maintenance time | 0.001 ms | 96.078 ms | candidate performs collection |
| Peak heap during deletion/maintenance | 67.81 MiB | 73.38 MiB | +8.2% |

Geometric mode trades a growing run budget for less rebuilding. With fanout r,
each merge promotes r same-level runs by one level; leaf weights are flush units
of at least one, so tiny manual flushes also promote. For F flush runs and n
inserted records, cumulative rebuilding is O(n log_r F), assuming expected
linear MPH building and bounded key width. With full B-key flushes, F is about
n/B. Up to O(r log_r F) runs can be probed per lookup. Key hashing/comparison
also depends on key bytes. Full and capped SizeTiered modes retain their old
hard bound and do not claim this geometric complexity.

## Compatibility and reproduction

`DynamicConfig` fields, existing method signatures, stable IDs and serialized
key/ID/config layout are unchanged. Policy, run weights and error diagnostics
are runtime state; restoration still produces a compacted Full-policy index.
The old default policy is covered by the existing merge-boundary regression.

From the repository root, with Python, the selected Rust toolchain and cached
Cargo dependencies available:

```powershell
python performance/compare_compaction.py --revision b78b528 --keys 1000000 --core 2
```

The runner writes disposable baseline/build copies under `target`, never changes
Git state, and performs exactly one measured pass for each of the four rows:
old Full, old SizeTiered, new SizeTiered, new Geometric. Baseline and candidate
are compiled from the same harness with the same dependency versions.
