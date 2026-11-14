"""Reproduce one PGM/Elias-Fano comparison pass for each source revision."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--revision', default='9f7bdb2')
parser.add_argument('--keys', type=int, default=1_000_000)
parser.add_argument('--core', type=int, default=2)
parser.add_argument('--phase', choices=['before', 'after', 'all'], default='all')
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
work = root / 'target' / 'pgm-ef-comparison'
baseline = work / 'before'
work.mkdir(parents=True, exist_ok=True)
if args.phase in ('before', 'all'):
    names = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', args.revision, '--', 'src', 'Cargo.toml'],
        cwd=root, text=True,
    ).splitlines()
    if not names:
        raise RuntimeError('baseline source is unavailable')
    for name in names:
        target = baseline / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(['git', 'show', f'{args.revision}:{name}'], cwd=root))

header = ('label,engine,dataset,keys,queries,build_ms,lookup_ns,lower_bound_ns,'
          'materialize_ns_per_key,heap_bytes,peak_heap_bytes,reported_bytes,'
          'query_allocations,segments,max_error,wire_bytes,correctness\n')
for phase in (['before', 'after'] if args.phase == 'all' else [args.phase]):
    project = work / f'{phase}-driver'
    (project / 'src').mkdir(parents=True, exist_ok=True)
    engine = baseline if phase == 'before' else root
    manifest = f'''[package]
name="pgm_ef_{phase}_driver"
version="0.0.0"
edition="2024"
[dependencies]
engine={{package="kira_kv_engine",path="{engine.as_posix()}"}}
core_affinity="0.8"
[profile.release]
opt-level=3
lto="fat"
codegen-units=1
'''
    (project / 'Cargo.toml').write_text(manifest, encoding='utf-8')
    pinned = root / 'performance' / 'pgm_ef_dependencies.lock'
    shutil.copyfile(pinned if pinned.exists() else root / 'Cargo.lock', project / 'Cargo.lock')
    shutil.copyfile(root / 'performance' / 'pgm_ef_compare.rs', project / 'src' / 'main.rs')
    subprocess.run(['cargo', 'build', '--offline', '--release', '--manifest-path',
                    str(project / 'Cargo.toml'), '--target-dir', str(work / 'build')],
                   cwd=root, check=True)
    suffix = '.exe' if sys.platform == 'win32' else ''
    executable = work / 'build' / 'release' / f'pgm_ef_{phase}_driver{suffix}'
    environment = os.environ.copy()
    environment['KIRA_BENCH_CORE'] = str(args.core)
    print(f'Running {phase}: {args.keys} keys per dataset', flush=True)
    result = subprocess.run([str(executable), phase, str(args.keys)], cwd=root,
                            env=environment, stdout=subprocess.PIPE, text=True)
    print(result.stdout, end='', flush=True)
    destination = root / 'performance' / f'pgm_ef_{phase}.csv'
    destination.write_text(header + result.stdout, encoding='utf-8')
    result.check_returncode()
    (work / f'{phase}-metadata.json').write_text(json.dumps({
        'revision': args.revision if phase == 'before' else 'working-tree',
        'keys': args.keys, 'core': args.core,
        'lock': (project / 'Cargo.lock').read_text(),
        'rustc': subprocess.check_output(['rustc', '-vV'], text=True),
        'processor': environment.get('PROCESSOR_IDENTIFIER', 'unknown'),
    }, indent=2), encoding='utf-8')
