# PGO (Profile-Guided Optimization) pipeline for kira_kv_engine on Windows.
#
# Usage:   .\scripts\pgo.ps1
# Output:  target/pgo-optimized/release/examples/million_build.exe
#
# Expected wins on Alder Lake / Zen4: +10-25% lookup, +5-15% build.

$ErrorActionPreference = 'Stop'

# Phase 1: build with instrumented profile.
$PgoData = "$PWD/target/pgo-data"
if (Test-Path $PgoData) { Remove-Item -Recurse -Force $PgoData }
New-Item -ItemType Directory -Force -Path $PgoData | Out-Null

Write-Host "[PGO] Phase 1: building instrumented binary..."
$env:RUSTFLAGS = "-Cprofile-generate=$PgoData"
cargo build --release --example million_build
if ($LASTEXITCODE -ne 0) { throw "Phase 1 build failed" }

Write-Host "[PGO] Phase 2: running workload to collect profile..."
# Use a representative workload — same as production lookup pattern.
$env:KIRA_BENCH_RUNS = "3"
& "$PWD/target/release/examples/million_build.exe"
if ($LASTEXITCODE -ne 0) { Write-Warning "Phase 2 run had errors (continuing)" }

Write-Host "[PGO] Phase 3: merging profile data..."
# llvm-profdata ships with rustup component `llvm-tools-preview`.
# rustup component add llvm-tools-preview
$ProfData = "$PgoData/merged.profdata"
$LlvmProfdata = (Get-ChildItem -Recurse -Path (rustc --print sysroot) -Filter 'llvm-profdata.exe' | Select-Object -First 1).FullName
if (-not $LlvmProfdata) {
    throw "llvm-profdata not found. Install via: rustup component add llvm-tools-preview"
}
& $LlvmProfdata merge -o $ProfData $PgoData
if ($LASTEXITCODE -ne 0) { throw "Profile merge failed" }

Write-Host "[PGO] Phase 4: rebuild with profile-guided optimization..."
$env:RUSTFLAGS = "-Cprofile-use=$ProfData -Cllvm-args=-pgo-warn-missing-function"
cargo build --release --example million_build --target-dir target/pgo-optimized
if ($LASTEXITCODE -ne 0) { throw "Phase 4 build failed" }

Write-Host ""
Write-Host "[PGO] Done!"
Write-Host "  Optimized binary: target/pgo-optimized/release/examples/million_build.exe"
Write-Host "  Run with: $env:KIRA_BENCH_RUNS = '3'; .\target\pgo-optimized\release\examples\million_build.exe"
