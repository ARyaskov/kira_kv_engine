#!/bin/bash
# PGO + optional BOLT pipeline for kira_kv_engine on Linux.
#
# Usage:   bash scripts/pgo.sh            # PGO only
#          bash scripts/pgo.sh --bolt     # PGO + BOLT (requires llvm-bolt)
# Output:  target/pgo-optimized/release/examples/million_build

set -euo pipefail

USE_BOLT=0
if [[ "${1:-}" == "--bolt" ]]; then
    USE_BOLT=1
fi

PGO_DATA="$PWD/target/pgo-data"
rm -rf "$PGO_DATA"
mkdir -p "$PGO_DATA"

# Phase 1: instrumented build.
echo "[PGO] Phase 1: building instrumented binary..."
RUSTFLAGS="-Cprofile-generate=$PGO_DATA" \
    cargo build --release --example million_build

# Phase 2: run workload.
echo "[PGO] Phase 2: running workload to collect profile..."
KIRA_BENCH_RUNS=3 ./target/release/examples/million_build || true

# Phase 3: merge profdata.
echo "[PGO] Phase 3: merging profile data..."
PROFDATA="$PGO_DATA/merged.profdata"
LLVM_PROFDATA=$(find "$(rustc --print sysroot)" -name 'llvm-profdata' -type f | head -1)
if [[ -z "$LLVM_PROFDATA" ]]; then
    echo "ERROR: llvm-profdata not found. Install: rustup component add llvm-tools-preview"
    exit 1
fi
"$LLVM_PROFDATA" merge -o "$PROFDATA" "$PGO_DATA"

# Phase 4: PGO-optimized rebuild.
echo "[PGO] Phase 4: rebuild with profile-guided optimization..."
RUSTFLAGS="-Cprofile-use=$PROFDATA -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release --example million_build --target-dir target/pgo-optimized

OPTIMIZED_BIN="target/pgo-optimized/release/examples/million_build"

if [[ $USE_BOLT -eq 1 ]]; then
    echo "[BOLT] Phase 5: BOLT layout optimization..."
    if ! command -v llvm-bolt &> /dev/null; then
        echo "ERROR: llvm-bolt not found. Install: apt install llvm-bolt (Debian) or build from llvm-project"
        exit 1
    fi
    # Run again under perf to collect branch trace.
    BOLT_PROFILE="$PGO_DATA/bolt.fdata"
    perf record -e cycles:u -j any,u -o "$PGO_DATA/perf.data" -- \
        "$OPTIMIZED_BIN"
    perf2bolt -p "$PGO_DATA/perf.data" -o "$BOLT_PROFILE" "$OPTIMIZED_BIN"
    BOLT_OUT="$OPTIMIZED_BIN.bolt"
    llvm-bolt "$OPTIMIZED_BIN" -o "$BOLT_OUT" -data="$BOLT_PROFILE" \
        -reorder-blocks=ext-tsp -reorder-functions=hfsort+ -split-functions \
        -split-all-cold -split-eh -dyno-stats
    echo "[BOLT] Done! Final binary: $BOLT_OUT"
else
    echo "[PGO] Done! Optimized binary: $OPTIMIZED_BIN"
fi
