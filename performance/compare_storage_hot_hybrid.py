"""Run one storage, hot-tracker and Hybrid comparison per source revision."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

parser = argparse.ArgumentParser()
parser.add_argument('--revision', default='d10dc11')
parser.add_argument('--keys', type=int, default=1_000_000)
parser.add_argument('--core', type=int, default=2)
parser.add_argument('--phase', choices=['before', 'after', 'all'], default='all')
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
work = root / 'target' / 'storage-hot-hybrid-comparison'
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

header = ('label,component,dataset,keys,operations,build_ms,operation_ns,extract_ns,'
          'retained_bytes,peak_extra_bytes,allocations,segments,wire_bytes,correctness\n')
for phase in (['before', 'after'] if args.phase == 'all' else [args.phase]):
    project = work / f'{phase}-driver'
    (project / 'src').mkdir(parents=True, exist_ok=True)
    engine = baseline if phase == 'before' else root
    (project / 'Cargo.toml').write_text(f'''[package]
name="storage_hot_hybrid_{phase}_driver"
version="0.0.0"
edition="2024"
[dependencies]
engine={{package="kira_kv_engine",path="{engine.as_posix()}"}}
core_affinity="0.8"
[profile.release]
opt-level=3
lto="fat"
codegen-units=1
''', encoding='utf-8')
    pinned = root / 'performance' / 'storage_hot_hybrid_dependencies.lock'
    fallback = root / 'performance' / 'pgm_ef_dependencies.lock'
    shutil.copyfile(pinned if pinned.exists() else fallback, project / 'Cargo.lock')
    shutil.copyfile(root / 'performance' / 'storage_hot_hybrid_compare.rs', project / 'src' / 'main.rs')
    subprocess.run(['cargo', 'build', '--offline', '--release', '--manifest-path',
                    str(project / 'Cargo.toml'), '--target-dir', str(work / 'build')],
                   cwd=root, check=True)
    suffix = '.exe' if sys.platform == 'win32' else ''
    executable = work / 'build' / 'release' / f'storage_hot_hybrid_{phase}_driver{suffix}'
    environment = os.environ.copy()
    environment['KIRA_BENCH_CORE'] = str(args.core)
    print(f'Running {phase}: {args.keys} storage keys; bounded tracker/Hybrid workloads', flush=True)
    result = subprocess.run([str(executable), phase, str(args.keys)], cwd=root,
                            env=environment, stdout=subprocess.PIPE, text=True)
    print(result.stdout, end='', flush=True)
    destination = root / 'performance' / f'storage_hot_hybrid_{phase}.csv'
    destination.write_text(header + result.stdout, encoding='utf-8')
    result.check_returncode()
    rows = list(csv.DictReader(result.stdout.splitlines(), fieldnames=header.strip().split(',')))
    if len(rows) != 10 or not all(row['correctness'] == 'pass' for row in rows):
        raise RuntimeError('incomplete or failed benchmark oracle')
    files = {path.name.removeprefix(f'{phase}-'): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in work.glob(f'{phase}-*.mmap')}
    metadata = {
        'revision': args.revision if phase == 'before' else 'working-tree',
        'keys': args.keys, 'core': args.core, 'mmap_sha256': files,
        'lock': (project / 'Cargo.lock').read_text(),
        'rustc': subprocess.check_output(['rustc', '-vV'], text=True),
        'processor': environment.get('PROCESSOR_IDENTIFIER', 'unknown'),
    }
    (work / f'{phase}-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if phase == 'after':
        old = json.loads((work / 'before-metadata.json').read_text())
        def registry(lock):
            return {(p['name'], p['version'], p.get('checksum'))
                    for p in tomllib.loads(lock)['package'] if 'source' in p}
        if not (old['mmap_sha256'] == files and old['rustc'] == metadata['rustc']
                and old['processor'] == metadata['processor'] and old['keys'] == args.keys
                and old['core'] == args.core and registry(old['lock']) == registry(metadata['lock'])):
            raise RuntimeError('source comparison settings or serialized mmap bytes differ')
        print('All mmap bytes, registry versions, compiler and workload settings match.', flush=True)
