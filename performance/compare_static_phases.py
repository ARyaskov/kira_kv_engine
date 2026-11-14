"""Prepare instrumented copies and run one diagnostic pass per source state.

Only ignored copies gain the temporary stage methods. The production checkout is
never rewritten. Phase timing changes inlining/cache behavior and adds a hash
output copy; phase times must not be summed into a production latency estimate.
"""
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


def function_body(source, name):
    start = source.index(f'fn {name}(')
    opening = source.index('{', start)
    depth = 1
    cursor = opening + 1
    state = 'code'
    block_depth = 0
    while cursor < len(source):
        char = source[cursor]
        pair = source[cursor:cursor + 2]
        if state == 'line':
            if char == '\n':
                state = 'code'
        elif state == 'string':
            if char == '\\':
                cursor += 1
            elif char == '"':
                state = 'code'
        elif state == 'block':
            if pair == '/*':
                block_depth += 1
                cursor += 1
            elif pair == '*/':
                block_depth -= 1
                cursor += 1
                if block_depth == 0:
                    state = 'code'
        elif pair == '//':
            state = 'line'
            cursor += 1
        elif pair == '/*':
            state = 'block'
            block_depth = 1
            cursor += 1
        elif char == '"':
            state = 'string'
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return source[opening + 1:cursor]
        cursor += 1
    raise ValueError(f'unclosed function body: {name}')


def instrument(source):
    if '__diag_hash_stage' in source or '__diag_probe_stage' in source:
        raise ValueError('production source already contains diagnostic wrappers')
    marker = 'Self::lookup_canonical_batch('
    if 'fn lookup_byte_chunk(' in source:
        body = function_body(source, 'lookup_byte_chunk')
        if body.count(marker) != 1:
            raise ValueError('unsupported byte-chunk stage shape')
        hash_code = body.split(marker, 1)[0]
        if 'let mut hashes = [0u64; 64];' not in hash_code:
            raise ValueError('unsupported hash scratch shape')
        declaration = ''
        flavor = 'production lookup_byte_chunk prefix, including prefetch'
    else:
        body = function_body(source, 'lookup_batch_pipelined')
        declaration = 'let mut hashes = [0u64; 64];'
        if declaration not in body or body.count(marker) != 1:
            raise ValueError('unsupported original byte pipeline')
        loop = 'for (input, result) in keys.chunks(64).zip(out.chunks_mut(64))'
        if loop not in body:
            raise ValueError('unsupported original chunk loop')
        inner = function_body('fn original() {' + body[body.index(loop):] + '}', 'original')
        opening = inner.index('{')
        hash_code = inner[opening + 1:].split(marker, 1)[0]
        flavor = 'production original per-key canonical hash loop'
    wrapper = f'''
// Diagnostic-only methods appended by compare_static_phases.py to an ignored copy.
impl Index {{
    #[doc(hidden)]
    #[inline(never)]
    pub fn __diag_hash_stage(&self, keys: &[&[u8]], out: &mut [u64]) {{
        assert_eq!(keys.len(), out.len());
        let engine = self.engine.as_ref().expect("diagnostic builds are nonempty");
        {declaration}
        for (input, output) in keys.chunks(64).zip(out.chunks_mut(64)) {{
            {hash_code}
            output.copy_from_slice(&hashes[..input.len()]);
        }}
    }}

    #[doc(hidden)]
    #[inline(never)]
    pub fn __diag_probe_stage(&self, hashes: &[u64], out: &mut [Option<usize>]) {{
        assert_eq!(hashes.len(), out.len());
        let engine = self.engine.as_ref().expect("diagnostic builds are nonempty");
        for (input, output) in hashes.chunks(64).zip(out.chunks_mut(64)) {{
            Self::lookup_canonical_batch(engine, input, output);
        }}
    }}
}}
'''
    return source + wrapper, {'flavor': flavor, 'hash_fragment_sha256': hashlib.sha256(hash_code.encode()).hexdigest()}


def expose_generator(source):
    old = 'struct QueryBytes {\n    key: Vec<u8>,\n    is_hit: bool,\n}'
    new = 'pub(crate) struct QueryBytes {\n    pub(crate) key: Vec<u8>,\n    pub(crate) is_hit: bool,\n}'
    if source.count(old) != 1:
        raise ValueError('unsupported frozen generator query type')
    source = source.replace(old, new)
    for name in ('gen_mixed_keys', 'gen_mixed_keys_missing', 'make_positive_queries_bytes', 'make_negative_queries_bytes'):
        pattern = f'fn {name}('
        if source.count(pattern) != 1:
            raise ValueError(f'unsupported frozen generator function: {name}')
        source = source.replace(pattern, 'pub(crate) ' + pattern)
    return source


def export(root, revision, destination):
    for path in ('src', 'examples', 'benches'):
        if revision is None:
            for source in (root / path).rglob('*'):
                if source.is_file():
                    target = destination / source.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        else:
            names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, '--', path], cwd=root, text=True).splitlines()
            for name in names:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=root))
    target = destination / 'Cargo.toml'
    if revision is None:
        shutil.copyfile(root / 'Cargo.toml', target)
    else:
        target.write_bytes(subprocess.check_output(['git', 'show', f'{revision}:Cargo.toml'], cwd=root))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', default='824e0d1')
    parser.add_argument('--keys', type=int, default=1_000_000)
    parser.add_argument('--queries', type=int, default=1_000_000)
    parser.add_argument('--core', type=int, default=2)
    parser.add_argument('--generation-threads', type=int, default=20)
    parser.add_argument('--phase', choices=['before', 'after', 'all'], default='all')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not (1000 <= args.queries <= args.keys <= 1_000_000 and args.generation_threads > 0):
        parser.error('require 1000 <= queries <= keys <= 1000000 and positive generation threads')
    root = Path(__file__).resolve().parent.parent
    work = root / 'target' / 'static-phase-comparison'
    work.mkdir(parents=True, exist_ok=True)
    original_generator = subprocess.check_output(['git', 'show', f'{args.revision}:examples/million_build.rs'], cwd=root).decode()
    generator = expose_generator(original_generator)
    states = ['before', 'after'] if args.phase == 'all' else [args.phase]
    for phase in states:
        library = work / f'{phase}-instrumented'
        export(root, args.revision if phase == 'before' else None, library)
        index = library / 'src/index.rs'
        original = index.read_text(encoding='utf-8')
        changed, instrumentation = instrument(original)
        index.write_text(changed, encoding='utf-8')
        project = work / f'{phase}-driver'
        (project / 'src').mkdir(parents=True, exist_ok=True)
        (project / 'Cargo.toml').write_text(f'''[package]
name="static_phases_{phase}_driver"
version="0.0.0"
edition="2024"
[dependencies]
kira_kv_engine={{path="{library.as_posix()}"}}
rand="0.8.5"
core_affinity="0.8"
tempfile="3"
rayon="1.11"
[profile.release]
opt-level=3
lto="fat"
codegen-units=1
''', encoding='utf-8')
        shutil.copyfile(root / 'Cargo.lock', project / 'Cargo.lock')
        shutil.copyfile(root / 'performance/static_phases.rs', project / 'src/main.rs')
        (project / 'src/million_data.rs').write_text(generator, encoding='utf-8')
        metadata = {
            'revision': args.revision if phase == 'before' else 'working-tree',
            'keys': args.keys, 'queries': args.queries, 'core': args.core,
            'generation_threads': args.generation_threads,
            'generator_sha256': hashlib.sha256(original_generator.encode()).hexdigest(),
            'uninstrumented_index_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'instrumentation': instrumentation,
            'limitations': 'Instrumented copies only; hash phase includes observable output copy, noinline diagnostic boundary; phases change inlining/cache/order and are not additive; one pass per case, not40M/20thread throughput.',
        }
        (work / f'{phase}-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        if args.prepare_only:
            print(f'Prepared {phase} diagnostic copy; no workload executed.', flush=True)
            continue
        subprocess.run(['cargo', 'build', '--offline', '--release', '-j', '1', '--manifest-path', str(project / 'Cargo.toml'), '--target-dir', str(work / 'build')], cwd=root, check=True)
        environment = os.environ.copy()
        environment['KIRA_BENCH_CORE'] = str(args.core)
        environment['KIRA_BENCH_THREADS'] = str(args.generation_threads)
        environment['RAYON_NUM_THREADS'] = '1'
        suffix = '.exe' if sys.platform == 'win32' else ''
        executable = work / 'build/release' / f'static_phases_{phase}_driver{suffix}'
        print(f'Running {phase}: single-thread diagnostic, {args.keys} keys / {args.queries} queries.', flush=True)
        result = subprocess.run([str(executable), phase, str(args.keys), str(args.queries)], cwd=root, env=environment, stdout=subprocess.PIPE, text=True)
        print(result.stdout, end='', flush=True)
        (work / f'{phase}.csv').write_text(result.stdout, encoding='utf-8')
        result.check_returncode()
        rows = list(csv.DictReader(result.stdout.splitlines()))
        if len(rows) != 12 or not all(row['correctness'] == 'pass' for row in rows):
            raise RuntimeError('incomplete diagnostic or failed oracle')
        metadata['lock'] = (project / 'Cargo.lock').read_text()
        metadata['rustc'] = subprocess.check_output(['rustc', '-vV'], text=True)
        metadata['processor'] = environment.get('PROCESSOR_IDENTIFIER', 'unknown')
        metadata['source_manifest'] = {path.relative_to(library).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((library / 'src').rglob('*.rs'))}
        (work / f'{phase}-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if 'after' in states and not args.prepare_only:
        before_meta = json.loads((work / 'before-metadata.json').read_text())
        after_meta = json.loads((work / 'after-metadata.json').read_text())
        def registry(lock):
            return {(p['name'], p['version'], p.get('checksum')) for p in tomllib.loads(lock)['package'] if 'source' in p}
        for name in ('keys', 'queries', 'core', 'generation_threads', 'generator_sha256', 'rustc', 'processor'):
            if before_meta[name] != after_meta[name]:
                raise RuntimeError(f'mismatched comparison setting: {name}')
        if registry(before_meta['lock']) != registry(after_meta['lock']):
            raise RuntimeError('mismatched resolved dependency versions')
        before = list(csv.DictReader((work / 'before.csv').open()))
        after = list(csv.DictReader((work / 'after.csv').open()))
        fields = ('mode', 'workload', 'phase', 'keys', 'queries', 'known_hits', 'known_misses', 'data_digest', 'hash_digest')
        if any([old[name] for name in fields] != [new[name] for name in fields] for old, new in zip(before, after)):
            raise RuntimeError('diagnostic inputs or exact reference hashes differ')
        print('All12 rows pass; input/hash identities and comparison settings match.', flush=True)


if __name__ == '__main__':
    main()
