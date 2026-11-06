#requires -Version 7.0
param([int]$Core=2,[switch]$StaticOnly)
$ErrorActionPreference='Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
$previousCore=$env:KIRA_BENCH_CORE
$previousStatic=$env:KIRA_PAIRED_STATIC_ONLY
try {
    $env:KIRA_BENCH_CORE="$Core"
    $env:KIRA_PAIRED_STATIC_ONLY=if ($StaticOnly) { "1" } else { $null }
    & ./scripts/staged_bench.ps1 -PrepareOnly
    @'
from pathlib import Path
source=Path('performance/baseline-source'); target=Path('performance/paired-baseline')
def write(path,data):
    if not path.exists() or path.read_bytes()!=data: path.write_bytes(data)
for root in ['src','benches','examples']:
    for path in (source/root).rglob('*.rs'):
        dest=target/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);write(dest,path.read_bytes())
manifest=(source/'Cargo.toml').read_text(encoding='utf-8').replace('name = "kira_kv_engine"','name = "kira_kv_engine_baseline"',1)
write(target/'Cargo.toml',manifest.encode('utf-8'))
root=Path('performance/paired');root.mkdir(exist_ok=True)
write(root/'Cargo.toml','''[package]
name="kira_paired_bench"
version="0.0.0"
edition="2024"
[dependencies]
before={package="kira_kv_engine_baseline",path="../paired-baseline"}
after={package="kira_kv_engine",path="../.."}
core_affinity="0.8"
[[bin]]
name="paired"
path="../paired_bench.rs"
[profile.release]
opt-level=3
lto="fat"
codegen-units=1
[workspace]
'''.encode('utf-8'))
lock=Path('performance/paired-dependencies.lock')
if lock.exists(): write(root/'Cargo.lock',lock.read_bytes())
'@ | python -
    if ($LASTEXITCODE -ne 0) { throw 'Paired source preparation failed' }
    $outputFile=if ($StaticOnly) { "performance/paired_static.csv" } else { "performance/paired.csv" }
    cargo run --locked --offline --release --manifest-path performance/paired/Cargo.toml --target-dir target/paired --bin paired | Tee-Object -FilePath $outputFile
    if ($LASTEXITCODE -ne 0) { throw 'Paired benchmark failed' }
}
finally { $env:KIRA_BENCH_CORE=$previousCore;$env:KIRA_PAIRED_STATIC_ONLY=$previousStatic;Pop-Location }
