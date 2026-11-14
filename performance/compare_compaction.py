"""One measured pass per policy against a frozen baseline and current sources."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--revision', default='b78b528')
parser.add_argument('--keys', type=int, default=1_000_000)
parser.add_argument('--phase', choices=['before', 'after', 'all'], default='all')
parser.add_argument('--core', type=int, default=2)
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
work = root / 'target' / 'maintenance-comparison'
work.mkdir(parents=True, exist_ok=True)
baseline = work / 'before'


def run(command, **kwargs):
    print(' '.join(map(str, command)), flush=True)
    return subprocess.run(command, cwd=root, check=True, **kwargs)


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

header = ('label,policy,keys,append_ns_per_key,lookup_ns_per_key,append_heap_bytes,'
          'append_peak_heap_bytes,append_estimated_bytes,append_tiers,rebuilt_keys,'
          'before_gc_heap_bytes,gc_ms,after_gc_heap_bytes,gc_peak_heap_bytes,'
          'after_gc_estimated_bytes,tombstones,correctness\n')
phases = ['before', 'after'] if args.phase == 'all' else [args.phase]
for phase in phases:
    project = work / f'{phase}-driver'
    (project / 'src').mkdir(parents=True, exist_ok=True)
    engine = baseline if phase == 'before' else root
    manifest = f'''[package]
name = "compaction_{phase}_driver"
version = "0.0.0"
edition = "2024"
[features]
geometric = []
[dependencies]
engine = {{ package = "kira_kv_engine", path = "{engine.as_posix()}" }}
core_affinity = "0.8"
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
'''
    (project / 'Cargo.toml').write_text(manifest, encoding='utf-8')
    # The checked-in lock pins identical third-party versions for both builds.
    pinned = root / 'performance' / 'compaction_dependencies.lock'
    shutil.copyfile(pinned if pinned.exists() else root / 'Cargo.lock', project / 'Cargo.lock')
    shutil.copyfile(root / 'performance' / 'compaction_compare.rs', project / 'src' / 'main.rs')
    command = ['cargo', 'build', '--offline', '--release', '--manifest-path', str(project / 'Cargo.toml'),
               '--target-dir', str(work / 'build')]
    if phase == 'after':
        command += ['--features', 'geometric']
    run(command)
    suffix = '.exe' if sys.platform == 'win32' else ''
    executable = work / 'build' / 'release' / f'compaction_{phase}_driver{suffix}'
    policies = ['full', 'size'] if phase == 'before' else ['size', 'geometric']
    environment = os.environ.copy()
    environment['KIRA_BENCH_CORE'] = str(args.core)
    rows = []
    for policy in policies:
        result = run([str(executable), phase, policy, str(args.keys)], env=environment,
                     text=True, stdout=subprocess.PIPE)
        print(result.stdout, end='', flush=True)
        rows.append(result.stdout)
    destination = root / 'performance' / f'compaction_{phase}.csv'
    destination.write_text(header + ''.join(rows), encoding='utf-8')
    lock = (project / 'Cargo.lock').read_bytes()
    (work / f'{phase}-lock.json').write_text(json.dumps({
        'revision': args.revision if phase == 'before' else 'working-tree',
        'keys': args.keys, 'core': args.core,
        'lock': lock.decode(),
        'rustc': subprocess.check_output(['rustc', '-vV'], text=True),
        'processor': environment.get('PROCESSOR_IDENTIFIER', 'unknown'),
    }, indent=2), encoding='utf-8')
