#!/usr/bin/env python3
"""Metadata-only launcher checks. No study datasets or empirical execution."""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import tempfile
from unittest import mock


def run_checks():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('launcher', root / 'run_local.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []

    def refused(name, function):
        try:
            function()
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            checks.append(name)
            return
        raise AssertionError('Expected refusal: ' + name)

    with tempfile.TemporaryDirectory(prefix='ccu_launcher_software_') as temporary:
        module.REPO = Path(temporary)
        module.WORKSPACE = module.REPO / 'local_workspace'
        module.init_workspace()
        for script, *_ in module.ROUTES.values():
            path = module.REPO / 'empirical_execution' / script
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# software-only stub\n')
        source = module.WORKSPACE / 'inputs/input.json'
        source.write_text('{}\n')
        launch = module.WORKSPACE / 'launch_specs/route.json'

        def put(route, arguments, role='preparation', dependencies=None, run_id='software_only'):
            value = {'schema': module.SCHEMA, 'run_id': run_id, 'evidence_role': role,
                     'route': route, 'arguments': arguments, 'dependencies': dependencies or []}
            launch.write_text(json.dumps(value))
            return value

        for route, (_, command, required, output, _) in module.ROUTES.items():
            arguments = {key: ('e5' if key == 'encoder' else 'civil_comments' if key == 'dataset' else str(source))
                         for key in required}
            if output:
                arguments[output] = str(module.WORKSPACE / 'runs/fresh')
            role = ('engineering' if route == 'engineering-dispatch' else
                    'development' if route in {'measure-resource', 'qualify-resource'} else
                    'confirmatory' if route == 'execute' else 'preparation')
            put(route, arguments, role)
            actual = module.plan(launch)
            assert actual['argv'][0] == module.sys.executable
            assert actual['argv'][1] == str(module.REPO / 'empirical_execution' / module.ROUTES[route][0])
            if command:
                assert actual['argv'][2] == command
            if route == 'engineering-dispatch':
                assert actual['argv'][-2:] == ['--mode', 'natural_text_engineering']
            checks.append('allowlisted_' + route)

        put('registry', {'out': str(module.WORKSPACE / 'runs/existing')})
        (module.WORKSPACE / 'runs/existing').mkdir()
        refused('existing_output_refused', lambda: module.plan(launch))
        put('registry', {'out': '../outside'})
        refused('escape_refused', lambda: module.plan(launch))
        value = put('registry', {'out': str(module.WORKSPACE / 'runs/fresh')})
        value['arguments']['shell'] = 'touch anything'
        launch.write_text(json.dumps(value))
        refused('unknown_argument_refused', lambda: module.plan(launch))
        put('engineering-dispatch', {key: str(source) for key in ('registry', 'jobs', 'bundles', 'policy')}
            | {'out': str(module.WORKSPACE / 'runs/fresh')}, 'confirmatory')
        refused('engineering_role_upgrade_refused', lambda: module.plan(launch))
        dependency = module.WORKSPACE / 'inputs/dependency.json'
        dependency.write_text('{"status":"ready","passed":true}')
        entry = {'path': str(dependency), 'sha256': module.digest(dependency), 'pointer': '/status', 'equals': 'ready'}
        put('registry', {'out': str(module.WORKSPACE / 'runs/fresh')}, dependencies=[entry])
        module.plan(launch)
        checks.append('real_dependency_bound')
        entry['equals'] = 'failed'
        put('registry', {'out': str(module.WORKSPACE / 'runs/fresh')}, dependencies=[entry])
        refused('dependency_status_mismatch_refused', lambda: module.plan(launch))
        entry['equals'] = 'ready'
        entry['sha256'] = '0' * 64
        put('registry', {'out': str(module.WORKSPACE / 'runs/fresh')}, dependencies=[entry])
        refused('dependency_hash_mismatch_refused', lambda: module.plan(launch))
        put('registry', {'out': str(module.WORKSPACE / 'runs/fresh')})
        (module.WORKSPACE / 'runner_logs/software_only').mkdir()
        refused('run_id_reuse_refused', lambda: module.plan(launch))
        refused('duplicate_json_refused', lambda: module.strict_json('{"a":1,"a":2}'))
        refused('nan_json_refused', lambda: module.strict_json('{"a":NaN}'))

        put('registry', {'out': str(module.WORKSPACE / 'runs/stub')}, run_id='blocked_receipt_stub')

        def fake_run(argv, **kwargs):
            assert isinstance(argv, list) and 'shell' not in kwargs
            output = Path(argv[-1])
            output.mkdir()
            (output / 'receipt.json').write_text('{"status":"blocked","primary_execution_allowed":false}')
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch.object(module, 'git_read', side_effect=lambda *args: 'stub_head' if args[0] == 'rev-parse' else ''), \
             mock.patch.object(module.subprocess, 'run', side_effect=fake_run):
            result = module.run(launch)
        assert result['status'] == 'process_completed'
        assert result['output']['reports']['receipt.json']['status'] == 'blocked'
        assert 'Only frozen executor' in result['scientific_acceptance']
        checks.append('zero_exit_does_not_claim_scientific_pass')
        assert (module.WORKSPACE / 'runner_logs/blocked_receipt_stub/result.json').is_file()
        checks.append('immutable_invocation_and_result_retained')

    return {'schema': 'ccu-local-launcher-software-check-1', 'status': 'passed', 'check_count': len(checks),
            'checks': checks, 'launcher_sha256': module.digest(root / 'run_local.py'),
            'checker_sha256': module.digest(__file__), 'evidence_role': 'software_only',
            'scope': 'Temporary metadata and stub subprocess routing; no empirical data or experiment execution.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Preserve existing report; choose a new --out')
    result = run_checks()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'check_count': result['check_count']}))


if __name__ == '__main__':
    main()
