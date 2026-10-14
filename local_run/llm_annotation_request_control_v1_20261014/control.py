"""Single bounded tokenizer-only preparation submission after reviewed full census."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/nfs_home/users/poonam/ccu-acl2027/repo')
W = REPO.parent / 'local_workspace'
PRIOR = W / 'tools/llm-sampler-control-v1-20261014'
PRIOR_SHA = 'a1779a1ea98b0968ab46680be6a0249983381bcd25a02a2c405877a32b237cf9'
PACKAGE = W / 'tools/llm-annotation-requests-v1-20261014'
ROOT = W / 'qualification/llm-annotation-requests-v1-20261014-selection-a01'
CENSUS = W / 'qualification/llm-selection-sampler-v1-20261014-selection-a01'
LOCK = W / 'qualification/llm-only-v3-sampling-lock-a01/sampling_lock.json'
LOCK_SHA = '52823fc3cbfcc358f58dab29a21b5d01746bd0dbc8cd90c232113cae4c8d39b7'
RESERVE, ALLOWANCE = 8 * 1024**3, 128 * 1024**2
QWEN = Path('/nfs_home/users/poonam/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')
MISTRAL = W / 'models/mistral-nemo-hub-v2/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3'
QWEN_ASSETS = W / 'annotation_v5/qwen-pilot-v5-20261011-a01/preparation/weights_manifest.json'
MISTRAL_ASSETS = W / 'annotation_mistral_v1/mistral-pilot-v1-20261011-a01/preparation/weights_manifest.json'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def helpers():
    raw = (PRIOR / 'package_manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PRIOR_SHA, 'prior_control_manifest_changed')
    for name, wanted in json.loads(raw)['files'].items():
        require(Path(name).name == name and name != 'package_manifest.json', 'unsafe_prior_member')
        path = PRIOR / name
        require(path.is_file() and not path.is_symlink(), 'prior_member_not_regular')
        data = path.read_bytes()
        require({'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} == wanted, 'prior_member_changed')
    spec = importlib.util.spec_from_file_location('request_prior_control', PRIOR / 'control.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    return prior, prior.helpers()


def queue_clear(raw):
    for line in raw.splitlines():
        parts = line.split('|')
        require(len(parts) == 2, 'invalid_queue_row')
        job, name = parts
        require(job != '13571' and not name.startswith(('third-llm-sampl', 'third-llm-request', 'third-llm-infer')),
                'relevant_job_active_reconcile')


def check_result_review(result, review, result_binding):
    require(result.get('schema') == 'third-llm-selection-sample-v1' and result.get('mode') == 'selection'
            and result.get('passed') is True and result.get('stage') == 'complete'
            and result.get('production_sampling_performed') is True
            and result.get('selected_frame_rows') == 20996 and result.get('unordered_pair_count') == 220405510,
            'complete_full_census_required')
    require(review.get('schema') == 'third-llm-selection-metadata-identity-review-v1'
            and review.get('passed') is True and review.get('selection_result') == result_binding,
            'bound_independent_full_sample_review_required')
    require(result.get('inference_activation') is False and result.get('model_inference_calls') == 0,
            'unexpected_annotation_state')


def child_command(args):
    command = [str(W / 'envs/judge-qwen-v3-20261011/bin/python'), '-B', str(PACKAGE / 'prepare_requests.py')]
    fields = {'repo': REPO, 'workspace': W, 'sampler-package': W / 'tools/llm-selection-sampler-v1-20261014',
              'lock-package': W / 'tools/llm-sampling-lock-v1-20261014',
              'protocol-package': W / 'tools/model-annotation-v3-20261014',
              'protocol-evidence': W / 'tools/model-annotation-v3-evidence-v1-20261014',
              'frame-package': W / 'tools/e5-calibration-frame-v1-20261014',
              'frame-dir': W / 'qualification/e5-calibration-frame-v1-20261014-a01/run/frame',
              'selection-result': CENSUS / 'run/output/result.json', 'selection-review': CENSUS / 'independent_review.json',
              'sampling-lock': LOCK, 'qwen-model': QWEN, 'qwen-assets-manifest': QWEN_ASSETS,
              'mistral-model': MISTRAL, 'mistral-assets-manifest': MISTRAL_ASSETS,
              'out': ROOT / 'run/output', 'package-sha256': args.preparation_package_sha256,
              'selection-result-sha256': args.selection_result_sha256,
              'selection-review-sha256': args.selection_review_sha256, 'sampling-lock-sha256': LOCK_SHA}
    for name, value in fields.items():
        command.extend(['--' + name, str(value)])
    return command


def source_check(h, args):
    bindings = {'preparation': h.verify_package(PACKAGE, args.preparation_package_sha256),
                'selection_result': h.verify_file(CENSUS / 'run/output/result.json', args.selection_result_sha256),
                'selection_review': h.verify_file(CENSUS / 'independent_review.json', args.selection_review_sha256),
                'sampling_lock': h.verify_file(LOCK, LOCK_SHA),
                'qwen_assets': h.verify_file(QWEN_ASSETS, '5e89ce9479d130f3ee1d199720e42c56304494edee307bb3f618e19ca089ffc3'),
                'mistral_assets': h.verify_file(MISTRAL_ASSETS, '0fba5cf51d83b650e9c116ee490adf94d9f140cd3ad6f04d79b758f07d55bb29')}
    result, review = h.load(CENSUS / 'run/output/result.json'), h.load(CENSUS / 'independent_review.json')
    check_result_review(result, review, bindings['selection_result'])
    for name, wanted in result['files'].items():
        require(Path(name).name == name, 'unsafe_census_output_path')
        require(h.binding(CENSUS / 'run/output' / name) == wanted, 'census_metadata_changed')
    receipt = h.load(CENSUS / 'run/process_completion.json')
    tests = h.load(CENSUS / 'run/software_validation.json')
    require(receipt['returncode'] == 0 and receipt['console_cap_exceeded'] is False, 'census_process_failed')
    require(tests['passed'] is True and tests['tests'] == 22 and tests['skipped'] == 0, 'census_tests_failed')
    job = h.load(CENSUS / 'launch_receipt.json')['job_id']
    require(job == '13571', 'unexpected_census_job')
    accounting = h.command(['sacct', '-j', job, '-n', '-P', '-o', 'JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    h.terminal_parent(accounting, job)
    bindings['census_terminal_accounting'] = accounting
    return bindings


def parser():
    p = argparse.ArgumentParser()
    for name in ('control-package-sha256', 'preparation-package-sha256', 'selection-result-sha256', 'selection-review-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--expected-tests', type=int, required=True)
    return p


def main():
    args = parser().parse_args()
    prior, h = helpers()
    require(args.expected_tests > 0, 'positive_test_count_required')
    for name, value in vars(args).items():
        if name.endswith('sha256'):
            h.sha_argument(value)
    control = Path(__file__).resolve().parent
    code = h.verify_package(control, args.control_package_sha256)
    require(not os.path.lexists(ROOT), 'existing_preparation_stage_reconcile')
    require(not h.command(['git', 'status', '--porcelain'], cwd=REPO).strip(), 'executing_checkout_changed')
    initial = source_check(h, args)
    queue_clear(h.command(['squeue', '-h', '-u', 'poonam', '-o', '%i|%j']))
    qpath = REPO / 'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    h.verify_file(qpath, h.QUOTA_HELPER_SHA)
    spec = importlib.util.spec_from_file_location('request_fresh_quota', qpath)
    q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    mount = q.mount_identity(W); started = time.monotonic()
    raw = h.command(['quota', '-v', '-w'], timeout=30); quota = q.parse_login_quota(raw)
    prior.quota_gate(quota['available_soft_bytes'], time.monotonic() - started, mount)
    require(source_check(h, args) == initial, 'changed_preparation_inputs')
    require(h.verify_package(control, args.control_package_sha256) == code, 'changed_control')
    queue_clear(h.command(['squeue', '-h', '-u', 'poonam', '-o', '%i|%j']))
    prior.quota_gate(quota['available_soft_bytes'], time.monotonic() - started, mount)
    command = ['sbatch', '--parsable', '--nodelist=csis.cn1', '--job-name=third-llm-requests-selection',
               '--output', str(ROOT / 'slurm-%j.out'), '--error', str(ROOT / 'slurm-%j.err'),
               str(control / 'requests.sbatch'), str(control), *sys.argv[1:]]
    intent = {'schema': 'third-llm-request-preparation-launch-v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'root': str(ROOT), 'command': command, 'child_command': child_command(args), 'bindings': initial,
              'control_package': code, 'quota': quota, 'mount': mount, 'quota_raw': raw,
              'reserve_bytes': RESERVE, 'output_allowance_bytes': ALLOWANCE, 'quota_snapshot_not_reservation': True,
              'tokenizer_only': True, 'model_weight_reads_authorized': False, 'model_inference_calls': 0,
              'human_annotation_calls': 0, 'inference_activation': False, 'automatic_retry': False}
    os.umask(0o077); ROOT.mkdir(mode=0o700, exist_ok=False)
    print(json.dumps(prior.submit_once(h, ROOT, command, intent), sort_keys=True))


if __name__ == '__main__':
    main()
