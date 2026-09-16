#!/usr/bin/env python3
"""Eight-cell mixed-schedule experiment; each cell owns its reference.

Uses the reviewed follow-up process, cache and handoff helpers. GPU execution
is explicit; --plan writes the prospective matrix without importing torch.
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'next_campaign'))
import run_followup as gate


def matrix():
    # Baseline and the previously successful joint setting come first.
    combos = [(0, 1, 1), (1, 0, 0)]
    combos += [x for x in itertools.product((0, 1), repeat=3) if x not in combos]
    return [{'label': f'd{d}_c{c}_b{b}', 'deterministic': d,
             'combo_kernels': bool(c), 'benchmark_combo_kernel': bool(b)}
            for d, c, b in combos]


def selected_cells(labels):
    cells = matrix()
    if labels is None: return cells
    requested = labels.split(',')
    if len(set(requested)) != len(requested) or set(requested) - {c['label'] for c in cells}:
        raise ValueError('unknown or duplicate cell selection')
    return [c for c in cells if c['label'] in requested]


def record_args(cell):
    config = {key: cell[key] for key in ('combo_kernels', 'benchmark_combo_kernel')}
    return ['--model', 'Qwen/Qwen2.5-7B-Instruct', '--revision',
            'a09a35458c702b33eeacc393d103063234e8bc28', '--tp', '1',
            '--cudagraph', '1', '--prefix-caching', '0', '--mixed',
            '--inductor-config', json.dumps(config, separators=(',', ':')),
            '--tag', 'factorial_' + cell['label']]


def execute(args):
    out = args.out.resolve(); source = args.source.resolve()
    if out.exists():
        raise gate.GateError(f'refuse existing output: {out}')
    out.mkdir(parents=True)
    for name in ('e6', 'logs', 'archives', 'evidence', 'acks', 'cache'):
        (out / name).mkdir()
    gate.source_gate(source)
    import importlib.metadata as md
    reference = json.loads(args.reference_env.read_text())['installed_distributions']
    actual = {d.metadata['Name']: d.version for d in md.distributions() if d.metadata.get('Name')}
    if actual != reference: raise gate.GateError('installed distributions differ from reviewed runtime')
    shutil.copy2(args.reference_env, out / 'evidence' / 'reference_env.json')
    gate.write_json(out / 'plan.json', {'schema': 1, 'cells': selected_cells(args.cells), 'cold_repeats': args.cold_repeats})
    for path in (Path(__file__), Path(gate.__file__)):
        shutil.copy2(path, out / 'evidence' / path.name)
    library_digest = gate.sha256_file(Path(gate.__file__))
    base = os.environ.copy()
    base.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_V2_MODEL_RUNNER='0',
                VLLM_DISABLE_COMPILE_CACHE='1', HF_HOME='/root/hf', HF_HUB_OFFLINE='1')
    state = {'schema': 1, 'completed': [], 'cells': [], 'source_commit': gate.EXPECTED_SOURCE,
             'helper_sha256': library_digest}
    gate.write_json(out / 'state.json', state)
    for cell in selected_cells(args.cells):
        record = None; immutable = None; record_cache = None
        for phase in ['record', 'warm'] + [f'cold_{i}' for i in range(args.cold_repeats)]:
            label = cell['label'] + '_' + phase
            if (out / 'STOP').exists() or time.time() + args.timeout + 240 >= args.deadline:
                raise gate.GateError('STOP or insufficient deadline margin')
            gate.disk_gate(out, 25)
            assert gate.sha256_file(Path(gate.__file__)) == library_digest
            if record is not None:
                assert gate.tree_manifest(record) == immutable, 'reference changed'
            cache = record_cache if phase == 'warm' else out / 'cache' / label
            before = gate.cache_snapshot(cache) if phase == 'warm' else gate.prepare_cold_cache(cache)
            gate.write_json(out / 'logs' / (label + '.cache_before.json'), before)
            env = base | gate.cache_environment(cache) | {'TORCHINDUCTOR_DETERMINISTIC': str(cell['deterministic'])}
            log = out / 'logs' / (label + '.run.log')
            if phase == 'record':
                existing = set((out / 'e6').glob('*/record'))
                command = [sys.executable, 'run_e6.py', '--record', '--out', str(out / 'e6'), *record_args(cell)]
            else:
                replay = record.parent / ('replay_' + phase)
                command = [sys.executable, 'run_e6.py', '--replay', str(record), '--out', str(replay)]
            rc = gate.run_process(command, cwd=source / 'probes/shape', env=env, log=log, timeout=args.timeout)
            if rc:
                # An unsupported configuration is evidence, never an IDENTICAL result.
                # Stop this invocation; operator must inspect/archive the error before
                # continuing a separately specified configuration batch.
                gate.write_json(out / 'evidence' / (label + '.failure.json'),
                                {'cell': cell, 'phase': phase, 'returncode': rc})
                raise gate.GateError(f'{label} exited {rc}; preserve failure before continuation')
            extras = [out / 'evidence' / 'reference_env.json', out / 'plan.json', out / 'evidence' / 'factorial.py', out / 'evidence' / 'run_followup.py']
            if phase == 'record':
                record = gate.record_result_dir(out, existing); gate.ensure_record(record)
                record_cache = cache; immutable = gate.tree_manifest(record)
                gate.write_json(out / 'evidence' / (label + '.immutable.json'), immutable)
                extras.append(out / 'evidence' / (label + '.immutable.json'))
            else:
                rc = gate.run_process([sys.executable, 'run_e6.py', '--compare', str(record), str(replay)],
                                      cwd=source / 'probes/shape', env=env,
                                      log=out / 'logs' / (label + '.compare.log'), timeout=90)
                if rc:
                    raise gate.GateError(f'comparison failed: {label}')
                summary = gate.validate_replay_summary(out, record.parent.name, replay.name)
                extras.append(summary)
            run_dir = record if phase == 'record' else replay
            metadata = gate.load_json(run_dir / 'run.json')
            captured_env = gate.load_json(run_dir / 'env.json')
            if captured_env['env'].get('TORCHINDUCTOR_DETERMINISTIC') != str(cell['deterministic']):
                raise gate.GateError('captured deterministic setting differs from request')
            for key in ('combo_kernels', 'benchmark_combo_kernel'):
                if f"'{key}': {cell[key]}" not in metadata['resolved_config']:
                    raise gate.GateError(f'resolved {key} differs from request')
            gate.write_json(out / 'logs' / (label + '.cache_after.json'), gate.cache_snapshot(cache))
            gate.archive_cache(cache, out / 'archives' / label / 'cache')
            manifest = gate.handoff_manifest(out, label, record.parent, out / 'logs', out / 'archives', extras)
            gate.wait_for_ack(out / 'acks', label, manifest, 180, args.deadline)
            state['completed'].append(label); gate.write_json(out / 'state.json', state)
        state['cells'].append(cell); gate.write_json(out / 'state.json', state)
    gate.write_json(out / 'COMPLETE.json', state)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--cells', help='comma-separated matrix labels for an explicitly separated batch')
    ap.add_argument('--source', type=Path)
    ap.add_argument('--reference-env', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--deadline', type=int)
    ap.add_argument('--timeout', type=int, default=480)
    ap.add_argument('--cold-repeats', type=int, default=3)
    args = ap.parse_args()
    if args.plan:
        print(json.dumps({'cells': selected_cells(args.cells), 'cold_repeats': args.cold_repeats}, indent=2)); return
    if not all((args.source, args.out, args.deadline, args.reference_env)) or args.cold_repeats < 1:
        ap.error('execution requires --source, --out, --deadline and positive repeats')
    try:
        execute(args)
    except (gate.GateError, AssertionError) as error:
        if args.out.exists(): gate.stop(args.out, str(error))
        raise SystemExit(str(error))

if __name__ == '__main__': main()
