"""Compare the actual million_build workload with shared, untimed validation."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import tomllib

parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['before', 'after', 'all'], default='all')
parser.add_argument('--revision', default='824e0d1')
parser.add_argument('--keys', type=int, default=40_000_000)
parser.add_argument('--queries', type=int, default=10_000_000)
parser.add_argument('--threads', type=int, default=20)
parser.add_argument('--build-threads', type=int, default=8)
parser.add_argument('--build-cores', default='0,2,4,6,8,10,12,14')
args = parser.parse_args()
if not 0 < args.queries <= args.keys < 2**32 or min(args.threads, args.build_threads) < 1:
    raise ValueError('require 0 < queries <= keys < 2^32 and positive thread counts')
root = Path(__file__).resolve().parent.parent
work = root / 'target' / 'static-index-optimization'
baseline = work / 'before'
work.mkdir(parents=True, exist_ok=True)
if args.phase in ('before', 'all'):
    names = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', args.revision, '--',
         'src', 'Cargo.toml', 'examples', 'benches'], cwd=root, text=True,
    ).splitlines()
    if not names:
        raise RuntimeError('baseline source is unavailable')
    for name in names:
        path = baseline / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(['git', 'show', f'{args.revision}:{name}'], cwd=root))
    # Both libraries run the same stronger harness. Validation is after all
    # timed queries of each build; generated data and timed functions are shared.
    shutil.copyfile(root / 'examples/million_build.rs', baseline / 'examples/million_build.rs')
    (baseline / 'examples/support').mkdir(exist_ok=True)
    shutil.copyfile(root / 'examples/support/validation.rs', baseline / 'examples/support/validation.rs')
    lock = root / 'performance/million_dependencies.lock'
    shutil.copyfile(lock if lock.exists() else root / 'Cargo.lock', baseline / 'Cargo.lock')


class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            'peak_working_set', 'working_set', 'paged_peak', 'paged',
            'nonpaged_peak', 'nonpaged', 'pagefile', 'peak_pagefile', 'private_bytes',
        )]


def run(executable, environment, output):
    child = subprocess.Popen([str(executable)], cwd=root, env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stop = threading.Event()
    memory = {'peak_working_set_bytes': None, 'peak_private_bytes_observed': None}

    def monitor():
        if os.name != 'nt':
            return
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
        query.restype = wintypes.BOOL
        while not stop.is_set():
            values = ProcessMemory()
            values.cb = ctypes.sizeof(values)
            if query(wintypes.HANDLE(int(child._handle)), ctypes.byref(values), values.cb):
                for field, value in [('peak_working_set_bytes', values.peak_working_set),
                                     ('peak_private_bytes_observed', values.private_bytes)]:
                    memory[field] = max(memory[field] or 0, value)
            stop.wait(0.25)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    with output.open('w', encoding='utf-8') as log:
        for line in child.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
    code = child.wait()
    stop.set()
    worker.join()
    if code:
        raise subprocess.CalledProcessError(code, str(executable))
    return memory


for phase in (['before', 'after'] if args.phase == 'all' else [args.phase]):
    source = baseline if phase == 'before' else root
    subprocess.run(['cargo', 'build', '--offline', '--locked', '--release', '-j', '1',
                    '--example', 'million_build', '--manifest-path', str(source / 'Cargo.toml'),
                    '--target-dir', str(work / 'million-build')], cwd=root, check=True)
    environment = os.environ.copy()
    environment.update({
        'KIRA_BENCH_N': str(args.keys), 'KIRA_BENCH_OPS': str(args.queries),
        'KIRA_BENCH_RUNS': '1', 'KIRA_BENCH_THREADS': str(args.threads),
        'KIRA_BENCH_CORE_IDS': ','.join(map(str, range(args.threads))),
        'KIRA_BUILD_THREADS': str(args.build_threads), 'KIRA_BUILD_CORE_IDS': args.build_cores,
        'RAYON_NUM_THREADS': str(args.threads),
    })
    executable = work / 'million-build/release/examples' / ('million_build.exe' if os.name == 'nt' else 'million_build')
    print(f'Running {phase}: {args.keys} keys, {args.queries} queries, one pass', flush=True)
    start = time.monotonic()
    memory = run(executable, environment, work / f'million-{phase}.log')
    text = (work / f'million-{phase}.log').read_text()
    rows = [line.split() for line in text.splitlines()
            if line.startswith(('PtrHash25 ', 'PtrHash25-lean '))]
    if len(rows) != 12 or text.count('correctness=pass') != 6:
        raise RuntimeError('missing timing rows or real correctness receipts')
    metadata = {
        'revision': args.revision if phase == 'before' else subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() + '+working-tree',
        'settings': {key: environment[key] for key in environment
                     if key.startswith(('KIRA_BENCH_', 'KIRA_BUILD_')) or key == 'RAYON_NUM_THREADS'},
        'compiler': subprocess.check_output(['rustc', '-vV'], text=True),
        'harness_sha256': hashlib.sha256((source / 'examples/million_build.rs').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'validation_sha256': hashlib.sha256((source / 'examples/support/validation.rs').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'registry_versions': sorted([p['name'], p['version']] for p in
                                    tomllib.loads((source / 'Cargo.lock').read_text())['package'] if 'source' in p),
        'rows': rows, 'memory': memory, 'elapsed_seconds': time.monotonic() - start,
    }
    (work / f'million-{phase}-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if phase == 'after':
        previous = json.loads((work / 'million-before-metadata.json').read_text())
        for field in ('settings', 'compiler', 'harness_sha256', 'validation_sha256', 'registry_versions'):
            if previous[field] != metadata[field]:
                raise RuntimeError(f'comparison metadata differs: {field}')
    print(f'{phase}: twelve rows and six full correctness receipts passed; memory={memory}', flush=True)
