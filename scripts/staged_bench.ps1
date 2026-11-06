#requires -Version 7.0
param(
    [string]$BaselineRevision = '02a44f4953a496da030651a640c52527b083df1b',
    [int]$KeyCount = 10000000,
    [int]$Runs = 5,
    [int]$Core = 0,
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
$previousN = $env:KIRA_STAGE_N
$previousRuns = $env:KIRA_STAGE_RUNS
$previousCore = $env:KIRA_BENCH_CORE
try {
    $env:KIRA_STAGE_N = "$KeyCount"
    $env:KIRA_STAGE_RUNS = "$Runs"
    $env:KIRA_BENCH_CORE = "$Core"
    # Only a disposable source copy is written; the checkout and Git state stay intact.
    @'
from pathlib import Path
import re, subprocess, sys
revision=sys.argv[1]
root=Path('performance/baseline-source')
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'--','src','Cargo.toml','.cargo/config.toml','benches','examples/million_build.rs'],text=True).splitlines()
if not paths: raise RuntimeError('Baseline revision contains no source files')
for name in paths:
    target=root/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(subprocess.check_output(['git','show',revision+':'+name]))
source=Path('examples/staged_bench.rs').read_text(encoding='utf-8')
start=source.index('    for bits in [')
end=source.index('    let mut dynamic =',start)
source=source[:start]+source[end:]
source=re.sub(r'(?s)    // BENCH_NEW_BACKGROUND_BEGIN.*?    // BENCH_NEW_BACKGROUND_END\n','',source)
source=re.sub(r'(?m)^\s*dynamic\.finish_background_compaction\(\)\.unwrap\(\);\n','',source)
start=source.index('    let (merges, rebuilt) = dynamic.compaction_stats();')
end=source.index('    metric(&stage, "dynamic.tiers"',start)
source=source[:start]+source[end:]
(root/'examples').mkdir(exist_ok=True)
(root/'examples/staged_bench.rs').write_text(source,encoding='utf-8')
if Path('Cargo.lock').exists(): (root/'Cargo.lock').write_bytes(Path('Cargo.lock').read_bytes())
'@ | python - $BaselineRevision
    if ($LASTEXITCODE -ne 0) { throw 'Baseline source preparation failed' }
    if ($PrepareOnly) { return }
    cargo build --offline --profile bench --example staged_bench --manifest-path performance/baseline-source/Cargo.toml --target-dir target/baseline
    if ($LASTEXITCODE -ne 0) { throw 'Baseline build failed; run cargo fetch first if dependencies are missing' }
    cargo build --locked --offline --profile bench --example staged_bench
    if ($LASTEXITCODE -ne 0) { throw 'Current build failed' }
    & ./target/baseline/release/examples/staged_bench.exe baseline_pinned | Tee-Object -FilePath performance/baseline_pinned.csv
    if ($LASTEXITCODE -ne 0) { throw 'Baseline verification failed' }
    & ./target/release/examples/staged_bench.exe final_pinned | Tee-Object -FilePath performance/final_pinned.csv
    if ($LASTEXITCODE -ne 0) { throw 'Current verification failed' }
    & ./target/release/examples/staged_bench.exe final_background | Tee-Object -FilePath performance/final_background.csv
    if ($LASTEXITCODE -ne 0) { throw 'Background-mode verification failed' }
}
finally {
    $env:KIRA_STAGE_N = $previousN
    $env:KIRA_STAGE_RUNS = $previousRuns
    $env:KIRA_BENCH_CORE = $previousCore
    Pop-Location
}
