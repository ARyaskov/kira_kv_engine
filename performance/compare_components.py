"""Run one validated filter/MPH comparison pass; preserve existing results."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import tomllib


ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'target/static-index-optimization'
BASELINE = WORK / 'before'
SETTINGS = {
    'KIRA_BENCH_CORE': '2',
    'KIRA_BUILD_THREADS': '8',
    'KIRA_BUILD_CORE_IDS': '0,2,4,6,8,10,12,14',
    'KIRA_BUILD_PIN': '1',
}
PROFILE = '[profile.release]\nopt-level=3\nlto="fat"\ncodegen-units=1\n'


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def command(arguments):
    return subprocess.check_output(arguments, cwd=ROOT, text=True).strip()


def engine_sources(source):
    paths = [source / 'Cargo.toml', *sorted((source / 'src').rglob('*.rs'))]
    return {path.relative_to(source).as_posix(): digest(path) for path in paths}


def ensure_baseline(revision):
    names = command(['git', 'ls-tree', '-r', '--name-only', revision, '--',
                     'src', 'Cargo.toml', 'examples', 'benches']).splitlines()
    if not names or 'Cargo.toml' not in names:
        raise RuntimeError('baseline revision/source is unavailable')
    for name in names:
        destination = BASELINE / name
        original = None
        if not destination.exists():
            original = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(original)
        # The million_build example has a deliberately shared validation harness.
        # Never replace it; only the engine sources must match the old revision.
        if name == 'Cargo.toml' or name.startswith('src/'):
            if original is None:
                original = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT)
            if destination.read_bytes().replace(b'\r\n', b'\n') != original.replace(b'\r\n', b'\n'):
                raise RuntimeError(f'exported baseline source differs from {revision}: {name}')
    expected = {name for name in names if name.startswith('src/') and name.endswith('.rs')}
    actual = {path.relative_to(BASELINE).as_posix() for path in (BASELINE / 'src').rglob('*.rs')}
    if actual != expected:
        raise RuntimeError('baseline source contains missing or stale Rust modules')


def registry_packages(lock):
    packages = tomllib.loads(lock.read_text(encoding='utf-8'))['package']
    registry = {}
    for package in packages:
        if 'source' in package:
            key = package['name'] + '@' + package['source']
            registry.setdefault(key, []).append([package['version'], package.get('checksum')])
    return {name: sorted(versions) for name, versions in registry.items()}


def check_common_packages(before, after):
    for name in before.keys() & after.keys():
        if before[name] != after[name]:
            raise RuntimeError(f'common dependency versions/checksums changed: {name}')


def check_resolved_packages(seed, resolved, allow_added):
    for name, versions in resolved.items():
        if name not in seed:
            if allow_added:
                continue
            raise RuntimeError(f'resolved dependency is absent from the pinned seed: {name}')
        if any(version not in seed[name] for version in versions):
            raise RuntimeError(f'resolved dependency version/checksum is not pinned: {name}')


def validate_csv(kind, phase, text):
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    expected = ({(filter_name, str(count)) for count in (1, 257, 4096, 500_000)
                 for filter_name in ('block_bloom', 'binary_fuse8')} if kind == 'filter'
                else {('ordinary', '1000000'), ('oversized_duplicate', '16385'),
                      ('large_buckets', '8192')})
    if kind == 'mph' and phase == 'after':
        expected.add(('external_1mib', '1000000'))
    case_field = 'filter' if kind == 'filter' else 'case'
    if len(rows) != len(expected) or {(row[case_field], row['keys']) for row in rows} != expected:
        raise RuntimeError('missing, duplicated or unexpected benchmark cases')
    for row in rows:
        if row.get('label') != phase or row.get('correctness') != 'pass' or None in row:
            raise RuntimeError('malformed row or missing correctness receipt')
        numbers = [value for key, value in row.items()
                   if key not in ('label', case_field, 'correctness')]
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in numbers):
            raise RuntimeError('invalid numeric measurement')
        if kind == 'filter':
            queries = int(row['queries'])
            false_positives = int(row['false_positives'])
            rate = float(row['false_positive_rate'])
            if queries != 1_000_000 or int(row['query_allocations']) != 0:
                raise RuntimeError('filter query count/allocation contract failed')
            if not 0 <= false_positives <= queries or abs(rate - false_positives / queries) > 5.1e-9:
                raise RuntimeError('inconsistent false-positive count/rate')
            # Both filters target about 1% or better. This is a generous quality
            # bound, not a performance threshold or an exact-membership claim.
            if rate > 0.02 or int(row['heap_bytes']) > int(row['peak_heap_bytes']):
                raise RuntimeError('filter FPR/heap contract failed')
        else:
            peak = int(row['heap_peak_bytes'])
            if int(row['heap_retained_bytes']) > peak:
                raise RuntimeError('MPH heap counters are inconsistent')
            if row['case'] == 'external_1mib' and peak > 1024 * 1024 + int(row['reported_index_bytes']):
                raise RuntimeError('external MPH exceeded its auxiliary-memory budget')
            if (int(row['wire_bytes']) == 0) != (row['case'] == 'oversized_duplicate'):
                raise RuntimeError('MPH serialization receipt failed')
    return rows


def environment_receipt(kind, revision, source, driver):
    build_environment = {key: value for key, value in os.environ.items()
                         if key in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC', 'RUSTUP_TOOLCHAIN',
                                    'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'CARGO_BUILD_TARGET',
                                    'CARGO_BUILD_RUSTFLAGS', 'CARGO_BUILD_RUSTC', 'CARGO_BUILD_RUSTC_WRAPPER')
                         or key.startswith(('CARGO_TARGET_', 'CARGO_PROFILE_RELEASE_'))}
    config = ROOT / '.cargo/config.toml'
    return {
        'kind': kind, 'revision': revision, 'engine_sources': engine_sources(source),
        'driver_sha256': digest(driver), 'compiler': command(['rustc', '-vV']),
        'isa_config_sha256': digest(config) if config.exists() else None,
        'build_environment': build_environment, 'settings': SETTINGS,
        'processor': os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
        'platform': platform.platform(), 'machine': platform.machine(),
        'keys': 500_000 if kind == 'filter' else 1_000_000,
        'runs_per_case': 1, 'release_profile': PROFILE,
        'memory_scope': 'allocator heap counters; native hugepage mappings excluded',
    }


def compare_environment(before, after):
    for field in ('kind', 'driver_sha256', 'compiler', 'isa_config_sha256', 'build_environment',
                  'settings', 'processor', 'platform', 'machine', 'keys', 'runs_per_case',
                  'release_profile', 'memory_scope'):
        if before[field] != after[field]:
            raise RuntimeError(f'comparison conditions changed: {field}')
    check_common_packages(before['registry_packages'], after['registry_packages'])


def prepare_project(kind, phase, source, driver, pinned):
    project = WORK / f'{kind}-{phase}'
    (project / 'src').mkdir(parents=True, exist_ok=True)
    name = f'{kind}_comparison_{phase}'
    manifest = (f'[package]\nname="{name}"\nversion="0.0.0"\nedition="2024"\n'
                f'[dependencies]\nengine={{package="kira_kv_engine",path="{source.as_posix()}"}}\n'
                'core_affinity="0.8"\n')
    if kind == 'filter':
        manifest += 'xorf={version="=0.13.0",default-features=false,features=["binary-fuse"]}\n'
    else:
        manifest += '[features]\ncandidate=[]\n'
    (project / 'Cargo.toml').write_text(manifest + PROFILE, encoding='utf-8')
    shutil.copyfile(driver, project / 'src/main.rs')
    shutil.copyfile(pinned, project / 'Cargo.lock')
    return project, name


def validate_project(project, kind, source):
    manifest = tomllib.loads((project / 'Cargo.toml').read_text(encoding='utf-8'))
    expected = {'engine', 'core_affinity', 'xorf'} if kind == 'filter' else {'engine', 'core_affinity'}
    dependencies = manifest['dependencies']
    engine = dependencies['engine']
    if (set(dependencies) != expected or engine.get('package') != 'kira_kv_engine'
            or set(engine) != {'package', 'path'} or Path(engine['path']).resolve() != source.resolve()
            or dependencies['core_affinity'] != '0.8'
            or manifest['profile']['release'] != {'opt-level': 3, 'lto': 'fat', 'codegen-units': 1}):
        raise RuntimeError('driver manifest has different source, features or release settings')
    if kind == 'filter' and dependencies['xorf'] != {
        'version': '=0.13.0', 'default-features': False, 'features': ['binary-fuse'],
    }:
        raise RuntimeError('filter comparison requires xorf 0.13.0 with binary-fuse only')


def run_phase(kind, phase, revision, driver, pinned):
    source = BASELINE if phase == 'before' else ROOT
    source_revision = revision if phase == 'before' else command(['git', 'rev-parse', 'HEAD']) + '+working-tree'
    metadata = environment_receipt(kind, source_revision, source, driver)
    result_path = WORK / f'{kind}-{phase}.csv'
    metadata_path = WORK / f'{kind}-{phase}-metadata.json'
    project = WORK / f'{kind}-{phase}'
    before_path = WORK / f'{kind}-before-metadata.json'
    if result_path.exists():
        rows = validate_csv(kind, phase, result_path.read_text(encoding='utf-8'))
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding='utf-8'))
            for field, value in metadata.items():
                if existing[field] != value:
                    raise RuntimeError(f'existing result belongs to different source/settings: {field}')
            metadata = existing
        elif phase == 'before' and (project / 'Cargo.lock').exists():
            validate_project(project, kind, source)
            if digest(project / 'src/main.rs') != digest(driver):
                raise RuntimeError('pre-existing baseline used a different benchmark driver')
            metadata['registry_packages'] = registry_packages(project / 'Cargo.lock')
            metadata['adoption'] = ('Pre-existing CSV preserved. Compiler/ISA/settings recorded at '
                                    'adoption; original timing was not repeated or retroactively measured.')
            metadata['rows'] = rows
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        else:
            raise RuntimeError('existing result has no source/environment receipt; refusing to replace it')
        if phase == 'after':
            compare_environment(json.loads(before_path.read_text(encoding='utf-8')), metadata)
        print(f'Preserved {result_path}: {len(rows)} validated data rows', flush=True)
        return
    project, name = prepare_project(kind, phase, source, driver, pinned)
    validate_project(project, kind, source)
    arguments = ['cargo', 'build', '--offline', '--release', '-j', '1', '--manifest-path',
                 str(project / 'Cargo.toml'), '--target-dir', str(WORK / f'{kind}-build')]
    if kind == 'mph' and phase == 'after':
        arguments += ['--features', 'candidate']
    environment = os.environ.copy()
    environment.update(SETTINGS)
    with (WORK / f'{kind}-build-{phase}.log').open('w', encoding='utf-8') as log:
        subprocess.run(arguments, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    metadata['registry_packages'] = registry_packages(project / 'Cargo.lock')
    # An isolated graph may prune unused versions from a larger lock seed. All
    # retained versions/checksums stay pinned; candidate-only dependencies may be
    # added. The actual before/after common-package comparison remains exact.
    check_resolved_packages(registry_packages(pinned), metadata['registry_packages'],
                            allow_added=phase == 'after')
    if phase == 'before':
        shutil.copyfile(project / 'Cargo.lock', pinned)
    else:
        compare_environment(json.loads(before_path.read_text(encoding='utf-8')), metadata)
    executable = WORK / f'{kind}-build/release' / (name + ('.exe' if os.name == 'nt' else ''))
    started = time.monotonic()
    result = subprocess.run([str(executable), phase, str(metadata['keys'])], cwd=ROOT,
                            env=environment, capture_output=True, text=True)
    (WORK / f'{kind}-{phase}.stderr.log').write_text(result.stderr, encoding='utf-8')
    result_path.write_text(result.stdout, encoding='utf-8')
    print(result.stdout, end='', flush=True)
    result.check_returncode()
    metadata['rows'] = validate_csv(kind, phase, result.stdout)
    metadata['elapsed_seconds'] = time.monotonic() - started
    metadata['command'] = [str(executable), phase, str(metadata['keys'])]
    metadata['features'] = ['candidate'] if kind == 'mph' and phase == 'after' else []
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'{kind}-{phase}: {len(metadata["rows"])} cases passed, one pass each', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('filter', 'mph'), required=True)
    parser.add_argument('--phase', choices=('before', 'after', 'all'), default='all')
    parser.add_argument('--revision', default='824e0d1')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        ensure_baseline(args.revision)
        driver = ROOT / f'performance/{"filter_compare" if args.kind == "filter" else "mph_build_compare"}.rs'
        pinned = ROOT / f'performance/{args.kind}_dependencies.lock'
        if not pinned.exists():
            existing = WORK / f'{args.kind}-before/Cargo.lock'
            shutil.copyfile(existing if existing.exists() else ROOT / 'Cargo.lock', pinned)
        phases = ('before', 'after') if args.phase == 'all' else (args.phase,)
        if 'after' in phases and not (WORK / f'{args.kind}-before-metadata.json').exists():
            # Adopt the already measured baseline without spending another run.
            if not (WORK / f'{args.kind}-before.csv').exists():
                if 'before' not in phases:
                    raise RuntimeError('run --phase before first; no baseline measurement exists')
            else:
                run_phase(args.kind, 'before', args.revision, driver, pinned)
        for phase in phases:
            run_phase(args.kind, phase, args.revision, driver, pinned)
    except Exception as error:
        (WORK / f'{args.kind}-{args.phase}-error.json').write_text(json.dumps({
            'kind': args.kind, 'phase': args.phase, 'error': str(error),
            'type': type(error).__name__,
        }, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
