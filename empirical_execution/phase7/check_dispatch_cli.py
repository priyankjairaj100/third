"""Focused CLI routing checks plus one reused natural Civil job; no primary study."""
from pathlib import Path
import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
import numpy as np
from phase7 import run_dispatch as cli
from phase6 import recipes
from phase6.check_dispatch import fixture


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reuse-natural-result', type=Path, help='Read a prior source-bound natural CLI result without launching another job; preserve its historical wrapper source scope.')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    paths = [Path(cli.__file__), Path(__file__), ROOT/'phase6/dispatch.py',
             ROOT/'phase6/check_dispatch.py', ROOT/'phase6/recipes.py', ROOT/'phase5/task_program.py']
    sources = {str(f.relative_to(ROOT)): cli.file_hash(f) for f in paths}
    policy = {'memory_bytes': 1536 * 1024**2, 'cpu_seconds': 75, 'wall_seconds': 80,
              'threads': 1, 'policy_role': 'phase7_explicit_Civil_engineering_smoke',
              'convex_verifier': 'exact_dyadic_integer_v1', 'convex_certificate_coordinates': 3_000_000,
              'state_audit_max_pair_coordinates': 150_000_000,
              'state_audit_max_coefficient_coordinates': 5_000_000}
    with tempfile.TemporaryDirectory(prefix='phase7-cli-contract-') as temp:
        base = Path(temp)
        registry, jobs = recipes.build_registry()
        write(base/'registry.json', registry)
        with gzip.open(base/'jobs.jsonl.gz', 'wt') as stream:
            for job in jobs:
                stream.write(json.dumps(job) + '\n')
        write(base/'bundles.json', {})
        write(base/'policy.json', policy)
        def arguments(output, explicit=True):
            argv = ['--registry', str(base/'registry.json'), '--jobs', str(base/'jobs.jsonl.gz'),
                    '--bundles', str(base/'bundles.json'), '--out', str(output), '--mode', cli.dispatch.PRIMARY_MODE]
            return argv + (['--policy', str(base/'policy.json')] if explicit else [])
        calls = []
        def routed(r, j, b, out, **kwargs):
            calls.append((r, j, b, kwargs))
            Path(out).mkdir()
            return {'software_routing_fixture': True, 'primary_outputs_accepted': 0}
        with patch.object(cli.dispatch, 'run_dispatch', side_effect=routed):
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(arguments(base/'explicit'))
                cli.main(arguments(base/'omitted', False))
            check('explicit_nondefault_policy_forwarded_exactly', calls[0][3]['policy'] == policy)
            check('omission_forwarded_as_None', calls[1][3]['policy'] is None)
            check('all_21332_registered_jobs_forwarded_without_execution_or_filtering', calls[0][0] == registry and calls[0][1] == jobs and len(jobs) == 21332)
            check('all_registered_method_cells_preserved', all(a['methods'] == b['methods'] for a, b in zip(calls[0][1], jobs)))
            check('primary_mode_and_bundle_base_preserved', calls[0][3]['mode'] == cli.dispatch.PRIMARY_MODE and calls[0][3]['base_dir'] == base)
            receipt = json.loads((base/'explicit/cli_invocation.json').read_text())
            check('receipt_binds_exact_policy_file_and_value', receipt['forwarded_policy'] == policy and receipt['input_file_sha256']['policy'] == cli.file_hash(base/'policy.json'))
            def refuses(name, text, output=None):
                (base/'policy.json').write_text(text)
                count = len(calls)
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        cli.main(arguments(output or base/name))
                except SystemExit as error:
                    check(name, error.code == 2 and len(calls) == count)
                else:
                    raise AssertionError(name)
            bad = [('missing_policy_field', {k:v for k,v in policy.items() if k != 'threads'}),
                   ('nonobject_policy', []), ('boolean_cpu_refused', {**policy, 'cpu_seconds': True}),
                   ('fractional_memory_refused', {**policy, 'memory_bytes': 123.5}),
                   ('too_small_memory_refused', {**policy, 'memory_bytes': 1024}),
                   ('zero_wall_refused', {**policy, 'wall_seconds': 0}),
                   ('unknown_field_refused', {**policy, 'wall_second': 9}),
                   ('negative_work_cap_refused', {**policy, 'convex_certificate_coordinates': -1}),
                   ('unknown_verifier_refused', {**policy, 'convex_verifier': 'approximate'}),
                   ('weaker_audit_tolerance_refused', {**policy, 'state_audit_relative_tolerance': 1e-6})]
            for name, value in bad:
                refuses(name, json.dumps(value))
            refuses('nonfinite_JSON_refused', json.dumps({**policy, 'wall_seconds': float('nan')}))
            refuses('duplicate_JSON_key_refused', '{"threads":1,"threads":2}')
            refuses('existing_output_refused_before_dispatch', json.dumps(policy), base/'explicit')
            check('only_two_contract_mock_calls_occurred', len(calls) == 2)
            write(base/'policy.json', policy)
            captured = {name: (base/filename).read_bytes() for name,filename in
                        [('registry','registry.json'),('jobs','jobs.jsonl.gz'),('bundles','bundles.json'),('policy','policy.json')]}
            original_load = cli.task_program.load_bound_value
            def mutate_after_capture(value, root):
                # Explicit temporary filesystem race fixture. All four paths
                # change after their bytes are captured, before dispatch.
                for filename in ('registry.json','jobs.jsonl.gz','bundles.json','policy.json'):
                    (base/filename).write_bytes(b'changed-after-snapshot')
                return original_load(value, root)
            with patch.object(cli.task_program, 'load_bound_value', side_effect=mutate_after_capture):
                with contextlib.redirect_stdout(io.StringIO()):
                    cli.main(arguments(base/'snapshot'))
            saved = json.loads((base/'snapshot/cli_invocation.json').read_text())
            check('all_top_level_receipt_hashes_bind_parsed_snapshot_during_file_mutation',
                  saved['input_file_sha256'] == {name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()})
            check('captured_policy_and_complete_jobs_survive_later_path_mutation', calls[-1][3]['policy'] == policy and calls[-1][0] == registry and calls[-1][1] == jobs)
    # One real, small method job; existing comments/labels and lexical arrays only.
    evidence_root = args.out if args.reuse_natural_result is None else args.reuse_natural_result
    natural = evidence_root/'natural_inputs'
    prior = None
    if args.reuse_natural_result is not None:
        prior = json.loads((evidence_root/'checks.json').read_text())
        check('prior_natural_result_passed_and_summary_bytes_unchanged', prior['status'] == 'passed' and
              prior['natural_summary_sha256'] == cli.file_hash(evidence_root/'natural_run/summary.json'))
        for name, value in prior['source_sha256'].items():
            path = ROOT/name
            if cli.file_hash(path) != value:
                historical = ROOT/'phase7/history/dispatch_cli_v1'/Path(name).name
                if not name.startswith('phase7/') or not historical.is_file() or cli.file_hash(historical) != value:
                    raise AssertionError('Unbound prior source: ' + name)
        check('historical_natural_wrapper_and_checker_bytes_preserved', True)
    def bind(value, stem='root'):
        if isinstance(value, np.ndarray):
            target = natural/(stem+'.npy')
            np.save(target, value, allow_pickle=False)
            return {'path': target.name, 'kind': 'npy', 'sha256': cli.file_hash(target)}
        if isinstance(value, dict):
            return {k:bind(v, stem+'_'+k) for k,v in value.items()}
        if isinstance(value, list):
            return [bind(v, stem+'_'+str(i)) for i,v in enumerate(value)]
        return value
    if args.reuse_natural_result is None:
        natural.mkdir()
        registry, jobs, bundles, _ = fixture()
        group = copy.deepcopy(next(g for g in registry['groups'] if g['block'] == 'B_relevance'))
        group['methods'] = ['O-T']
        job = copy.deepcopy(next(j for j in jobs if j['group_id'] == group['group_id'] and j['arm'] == 'R'))
        job['methods'] = ['O-T']
        registry = {'schema': 'ccu-phase7-natural-cli-smoke-1', 'groups': [group], 'jobs_content_sha256': cli.dispatch.digest([job])}
        # Bundle keys contain '/', so use one safe stem per request family.
        bound = {k:bind(v, 'bundle'+str(i)) for i,(k,v) in enumerate(bundles.items())}
        write(natural/'registry.json', registry)
        (natural/'jobs.jsonl').write_text(json.dumps(job)+'\n')
        write(natural/'bundles.json', bound)
        write(natural/'policy.json', policy)
        command = [sys.executable, str(Path(cli.__file__).resolve()), '--registry', str(natural/'registry.json'),
                   '--jobs', str(natural/'jobs.jsonl'), '--bundles', str(natural/'bundles.json'),
                   '--out', str(args.out/'natural_run'), '--policy', str(natural/'policy.json')]
        env = {**os.environ, 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
        result = subprocess.run(command, text=True, capture_output=True, env=env)
        (args.out/'natural_stdout.txt').write_text(result.stdout)
        (args.out/'natural_stderr.txt').write_text(result.stderr)
        check('one_real_Civil_job_CLI_exit_success', result.returncode == 0)
    summary = json.loads((evidence_root/'natural_run/summary.json').read_text())
    ledger = json.loads((evidence_root/'natural_run/jobs.json').read_text())
    lock = json.loads((evidence_root/'natural_run/lock.json').read_text())
    check('natural_job_completed_with_all_three_existing_checkpoints', summary['completed_jobs'] == 1 and ledger[0]['checkpoint_count'] == 3)
    check('natural_run_nondefault_policy_bound_exactly', lock['policy'] == policy)
    check('natural_run_not_promoted_to_primary', summary['primary_outputs_accepted'] == 0)
    check('natural_saved_head_checks_pass', all(r['passed'] for r in ledger[0]['methods'][0]['head_checks']))
    check('natural_input_and_engine_sources_unchanged', summary['inputs_unchanged'] and summary['source_hashes_unchanged'])
    check('wrapper_and_frozen_source_bindings_unchanged', all(cli.file_hash(ROOT/name) == value for name,value in sources.items()))
    report = {'schema':'ccu-phase7-dispatch-cli-checks-1', 'status':'passed', 'check_count':len(checks),
              'checks':checks, 'source_sha256':sources, 'registered_schedule_contract_jobs':21332,
              'registered_schedule_executed':False, 'natural_jobs_executed':int(prior is None),
              'natural_worker_services_newly_executed':int(prior is None), 'natural_worker_services_in_evidence':1,
              'natural_head_releases':3, 'primary_outputs_accepted':0,
              'scope':'Routing fixtures are transient software mocks. One reused Civil20 lexical job checks the real CLI; no source-rich, semantic, human, or primary evidence.',
              'natural_evidence_directory':str(evidence_root),
              'natural_wrapper_source_sha256':json.loads((evidence_root/'natural_run/cli_invocation.json').read_text())['wrapper_sha256'],
              'natural_evidence_is_current_wrapper_execution':prior is None,
              'prior_report_sha256':None if prior is None else cli.file_hash(evidence_root/'checks.json'),
              'natural_summary_sha256':cli.file_hash(evidence_root/'natural_run/summary.json')}
    write(args.out/'checks.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
