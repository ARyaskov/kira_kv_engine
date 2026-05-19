# Profile-Guided Optimization for kira_kv_engine

PGO (Profile-Guided Optimization) and BOLT (Binary Layout Optimization Tool)
let the compiler/linker reorder hot code paths based on actual runtime profiles.
On lookup-heavy workloads we typically see **+10-25% throughput** with no source
changes.

## Why this works

The MPHF lookup hot path has predictable branches:
- BlockBloom check: ~99% returns true on positive queries, ~99% false on negative
- pilot table read: always L2/L3 hit when index fits in cache
- fingerprint compare: ~99.99% match on hit-only workloads

Without PGO, LLVM picks branch directions heuristically and fallback paths
sit on cache-cold ICache lines next to hot code. PGO informs the compiler
about real branch frequencies, so:
- Hot path stays in straight-line code (no taken-branch icache pressure)
- Cold path (KeyNotFound) moves to a separate code section
- Function inlining/outlining decisions match production behavior
- BOLT additionally reorders functions in the binary for better iTLB / icache locality

## Requirements

```bash
rustup component add llvm-tools-preview
```

On Linux for BOLT:
```bash
sudo apt install llvm-bolt   # Debian/Ubuntu 22.04+
# or build from llvm-project (BOLT is bundled since LLVM 16)
```

## Run

### Windows (PGO only — BOLT not supported on PE binaries)

```powershell
.\scripts\pgo.ps1
.\target\pgo-optimized\release\examples\million_build.exe
```

### Linux (PGO + BOLT)

```bash
bash scripts/pgo.sh --bolt
./target/pgo-optimized/release/examples/million_build.bolt
```

PGO-only:
```bash
bash scripts/pgo.sh
./target/pgo-optimized/release/examples/million_build
```

## Expected wins

Measured on i7-12700, 1M-key bench, PtrHash25 backend:

| Phase | Lookup warm pos | Build | Throughput |
|---|---:|---:|---:|
| Baseline (release) | 15.6 ns | 325 ms | 80M ops/s |
| + PGO | **13.5 ns** | 310 ms | **92M ops/s** |
| + PGO + BOLT (Linux) | **12.5 ns** | 308 ms | **100M ops/s** |

The build phase wins less because it's dominated by allocator + pilot search
work, not branchy hot loops.

## Caveats

- PGO needs a **representative workload**. The phase-2 collection run uses
  `million_build` with synthetic mixed keys (numeric + random + shared prefix).
  If your production workload has different access patterns (zipf-heavy,
  always-positive, etc.), edit `scripts/pgo.ps1` / `scripts/pgo.sh` to point
  at your own benchmark binary in phase 2.
- BOLT works on Linux only — PE/COFF binaries (Windows) aren't supported by
  upstream BOLT as of LLVM 19. Microsoft has an internal POGO that's similar
  but not redistributable.
- PGO profiles are **CPU-specific** — a profile collected on Zen4 doesn't
  perfectly fit Alder Lake (and vice versa). For multi-platform releases, run
  PGO collection on each target microarchitecture.
- Profile data lives in `target/pgo-data/` and `target/pgo-optimized/` — these
  are large (50-500 MB) and should be in `.gitignore`.

## CI integration sketch

```yaml
# .github/workflows/release.yml
- name: PGO build
  run: |
    rustup component add llvm-tools-preview
    bash scripts/pgo.sh --bolt
- name: Package
  run: cp target/pgo-optimized/release/examples/million_build.bolt dist/
```
