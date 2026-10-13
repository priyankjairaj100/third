#!/usr/bin/env python3
"""Execute one explicitly submitted CPU stage; never submit or retry jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import socket
import subprocess
import sys
import time

CORE = 'e5_sharded_cpu_v2_20261013'
CORE_SHA = '0eb7309ea3bc4e92d37ac155eef03ce2a59a5daffefc3e3725f511de557cfa81'
FALSE_FLAGS = {'original_route_acceptance': False, 'full_population_cache_acceptance': False,
               'confirmatory_study_ready': False, 'automatic_job_submission': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def valid_sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None, 'invalid_SHA256')
    return value


def verify_file(path, expected_sha):
    valid_sha(expected_sha)
    require(path.is_file() and not path.is_symlink(), 'regular_pinned_file_required')
    observed = binding(path)
    require(observed['sha256'] == expected_sha, 'pinned_file_SHA256_mismatch')
    return observed


def verify_package(folder, expected_sha):
    manifest = folder / 'package_manifest.json'
    receipt = verify_file(manifest, expected_sha)
    value = json.loads(manifest.read_text())
    require(isinstance(value.get('files'), dict) and value['files'], 'nonempty_package_inventory_required')
    for name, expected in value['files'].items():
        require(isinstance(name, str) and Path(name).name == name and name not in ('.', '..'), 'unsafe_package_member')
        path = folder / name
        require(path.is_file() and not path.is_symlink() and binding(path) == expected, 'package_member_mismatch')
    return receipt


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='mode', required=True)
    common = ('repo', 'workspace', 'view', 'replay_result', 'census_dir', 'budget_dir',
              'core_package_sha', 'qualification_result', 'qualification_sha', 'dispatch_root')
    for mode in ('plan', 'shard', 'assembly'):
        child = subparsers.add_parser(mode)
        for name in common:
            child.add_argument(name)
        if mode != 'plan':
            child.add_argument('full_plan'); child.add_argument('full_plan_sha')
        child.add_argument('dispatch_package_sha')
    return parser.parse_args(argv)


def task_index(environ):
    raw = environ.get('SLURM_ARRAY_TASK_ID', '')
    require(re.fullmatch(r'0|[1-9][0-9]*', raw) is not None and 0 <= int(raw) < 32,
            'array_task_must_be_integer_0_through_31')
    require(environ.get('SLURM_ARRAY_TASK_COUNT') == '32'
            and environ.get('SLURM_ARRAY_TASK_MIN') == '0'
            and environ.get('SLURM_ARRAY_TASK_MAX') == '31', 'exact_32_task_array_required')
    return int(raw)


def preflight(args):
    require(platform.system() == 'Linux' and os.environ.get('SLURM_JOB_ID'), 'Linux_Slurm_required')
    require(socket.gethostname().split('.')[:2] != ['csis', 'mn1'], 'login_node_prohibited')
    for name in ('core_package_sha', 'qualification_sha', 'dispatch_package_sha'):
        valid_sha(getattr(args, name))
    require(args.core_package_sha == CORE_SHA, 'wrong_prospective_core_package')
    for name in ('repo', 'workspace', 'view', 'replay_result', 'census_dir', 'budget_dir',
                 'qualification_result', 'dispatch_root'):
        value = Path(getattr(args, name))
        require(value.is_absolute(), 'absolute_paths_required')
        setattr(args, name, value.resolve())
    require(args.dispatch_root.is_relative_to(args.workspace / 'qualification')
            and args.dispatch_root.is_dir(), 'existing_private_qualification_root_required')
    require(args.qualification_result.is_relative_to(args.workspace), 'qualification_outside_workspace')
    require(args.repo.is_dir() and args.workspace.is_dir(), 'repository_and_workspace_required')
    expected_python = args.workspace / 'envs/encoder-cpu/bin/python'
    # Resolve symlinks only for interpreter identity; all launches reuse this exact venv path.
    require(Path(sys.executable).absolute() == expected_python.absolute(), 'pinned_CPU_interpreter_path_required')
    cap = resource.getrlimit(resource.RLIMIT_AS)[0]
    required_gib = 10 if args.mode == 'shard' else 40
    require(cap == resource.RLIM_INFINITY or cap >= required_gib * 1024**3, 'inherited_address_space_limit_too_small')
    index = task_index(os.environ) if args.mode == 'shard' else None
    package = verify_package(Path(__file__).resolve().parent, args.dispatch_package_sha)
    core = verify_package(args.repo / 'local_run' / CORE, CORE_SHA)
    qualification = verify_file(args.qualification_result, args.qualification_sha)
    value = json.loads(args.qualification_result.read_text())
    require(value.get('schema') == 'third-e5-sharded-cpu-qualification-v1' and value.get('passed') is True,
            'passed_reviewed_qualification_required')
    evidence = {'dispatch_package': package, 'core_package': core, 'qualification_result': qualification}
    if args.mode != 'plan':
        require(Path(args.full_plan).is_absolute(), 'absolute_plan_path_required')
        args.full_plan = Path(args.full_plan).resolve()
        require(args.full_plan.is_relative_to(args.workspace), 'plan_outside_workspace')
        evidence['full_plan'] = verify_file(args.full_plan, args.full_plan_sha)
        plan = json.loads(args.full_plan.read_text())
        require(plan.get('schema') == 'third-e5-cpu-shard-plan-v1' and plan.get('kind') == 'full'
                and len(plan.get('shards', [])) == 32 and plan.get('documents') == 140250
                and plan.get('qualification_result') == qualification, 'reviewed_full_plan_identity_mismatch')
    return index, evidence


def common_arguments(args):
    names = ('repo', 'workspace', 'view', 'replay_result', 'census_dir', 'budget_dir')
    result = [item for name in names for item in ('--' + name.replace('_', '-'), str(getattr(args, name)))]
    return result + ['--package-sha256', CORE_SHA, '--qualification-result', str(args.qualification_result),
                     '--qualification-sha256', args.qualification_sha]


def commands(args, stage, index):
    core = args.repo / 'local_run' / CORE
    python = str(args.workspace / 'envs/encoder-cpu/bin/python')
    common = common_arguments(args)
    if args.mode == 'plan':
        return [('plan', [python, str(core / 'driver.py'), 'plan-full', *common, '--out', str(stage / 'output')],
                 stage / 'output/result.json')]
    if args.mode == 'shard':
        base = [python, str(core / 'worker.py'), *common, '--plan', str(args.full_plan), '--shard-index', str(index)]
        return [('encode', [*base, '--mode', 'encode', '--out', str(stage / 'encode')], stage / 'encode/completion.json'),
                ('replay', [*base, '--mode', 'replay', '--out', str(stage / 'replay'), '--encoded-dir', str(stage / 'encode')],
                 stage / 'replay/completion.json')]
    base = [python, str(core / 'driver.py'), 'assemble', *common, '--plan', str(args.full_plan), '--out', str(stage / 'cache')]
    encoded = [str(args.dispatch_root / 'shards' / f'shard-{i:04d}' / 'encode') for i in range(32)]
    replayed = [str(args.dispatch_root / 'shards' / f'shard-{i:04d}' / 'replay') for i in range(32)]
    return [('assembly', [*base, '--encoded-dirs', *encoded, '--replay-dirs', *replayed], stage / 'cache/manifest.json')]


def run_child(command, stage, name):
    start = time.perf_counter()
    with (stage / (name + '.stdout.txt')).open('x') as stdout, (stage / (name + '.stderr.txt')).open('x') as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        code = process.wait()
        elapsed = time.perf_counter() - start
        for stream in (stdout, stderr):
            stream.flush(); os.fsync(stream.fileno())
    receipt = {'schema': 'third-e5-sharded-dispatch-child-v1', 'command': command, 'returncode': code,
               'child_pid': process.pid, 'launcher_pid': os.getpid(), 'child_wall_seconds_through_exit': elapsed,
               'timer_includes_child_artifact_publication_and_exit': True,
               'own_receipt_and_log_flush_excluded_from_timer': True, **FALSE_FLAGS}
    write_json(stage / (name + '.process.json'), receipt)
    require(code == 0, 'child_failed_no_retry_or_following_stage')
    return receipt


def run(args):
    index, evidence = preflight(args)
    stage = (args.dispatch_root / 'shards' / f'shard-{index:04d}' if args.mode == 'shard'
             else args.dispatch_root / ('plan-run' if args.mode == 'plan' else 'assembly-run'))
    require(not stage.exists(), 'new_stage_directory_required')
    os.umask(0o077)
    stage.mkdir(parents=True, exist_ok=False)
    write_json(stage / 'launch.json', {'mode': args.mode, 'shard_index': index, 'bindings': evidence,
                                    'hostname': socket.gethostname(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], **FALSE_FLAGS})
    report = {'schema': 'third-e5-sharded-dispatch-result-v1', 'passed': False, 'mode': args.mode,
              'shard_index': index, 'children': [], 'bindings': evidence, **FALSE_FLAGS}
    try:
        for name, command, expected in commands(args, stage, index):
            child = run_child(command, stage, name)
            report['children'].append({'name': name, 'process_receipt': binding(stage / (name + '.process.json')),
                                       'child_wall_seconds_through_exit': child['child_wall_seconds_through_exit']})
            _, observed = preflight(args)
            require(observed == evidence, 'launch_bindings_changed_during_child')
            require(expected.is_file() and json.loads(expected.read_text()).get('passed') is True,
                    'successful_child_artifact_missing')
            report['children'][-1]['artifact'] = binding(expected)
        report['passed'] = True
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    write_json(stage / 'dispatch_result.json', report)
    return report


def main():
    try:
        report = run(parse_args())
    except Exception as error:
        report = {'passed': False, 'error_type': type(error).__name__, 'error': str(error), **FALSE_FLAGS}
    print(json.dumps({key: report[key] for key in ('passed', 'mode', 'shard_index', 'error_type', 'error') if key in report}))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
