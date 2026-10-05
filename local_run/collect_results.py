#!/usr/bin/env python3
"""Package an explicitly reviewed local report and redacted final-ledger accounting.

This does not authenticate human collection or validate reported estimands.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'empirical_execution/phase6/results/recipes_release/jobs.jsonl.gz'


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON number')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_:-]{0,95}', value):
        raise ValueError('Unsafe or malformed status token in dispatch')
    return value


def report_bytes(path):
    raw = path.read_bytes()
    value = strict_json(raw)
    template = strict_json((ROOT / 'local_run/templates/results_report.template.json').read_bytes())
    if not isinstance(value, dict) or set(value) != set(template):
        raise ValueError('Report must use the exact top-level template fields')
    if value['schema'] != 'ccu-local-results-report-1' or value['reviewed_for_sharing'] is not True:
        raise ValueError('Report must explicitly record completed sharing review')
    if b'REQUIRED_' in raw or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value['report_id']):
        raise ValueError('Replace required placeholders with actual observations')
    if value['stage'] not in {'setup','preparation','calibration','development','confirmatory','analysis'}:
        raise ValueError('Invalid report stage')
    if value['evidence_role'] not in {'preparation','engineering','development','confirmatory','analysis'}:
        raise ValueError('Invalid report evidence role')
    if value['status'] not in {'not_started','in_progress','blocked','failed','completed','completed_with_limitations'}:
        raise ValueError('Invalid declared report status')
    for key in ('inputs','completed_work','blocked_work','deviations','results','negative_findings','questions_for_research_assistant'):
        if not isinstance(value[key], list):
            raise ValueError('Expected report array: ' + key)
    for key in ('machine','runtime','human_collection'):
        if not isinstance(value[key], dict):
            raise ValueError('Expected report object: ' + key)
    if not isinstance(value['next_action'], str) or not value['next_action'].strip():
        raise ValueError('State the actual next action')
    return raw


def dispatch_accounting(directory):
    jobs_raw = (directory / 'jobs.json').read_bytes()
    summary_raw = (directory / 'summary.json').read_bytes()
    summary = strict_json(summary_raw)
    if summary.get('schema') != 'ccu-stable-root-study-1':
        raise ValueError('Not a Phase 9 final dispatch summary')
    if summary.get('final_jobs_sha256') != sha(jobs_raw):
        raise ValueError('Final ledger hash does not match summary')
    rows = strict_json(jobs_raw)
    if not isinstance(rows, list):
        raise ValueError('Final jobs.json must contain the complete job array')
    with gzip.open(REGISTRY, 'rt') as stream:
        expected = {j['job_id']: j for line in stream if line.strip() for j in [strict_json(line)]}
    ids = [row['job_id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Final ledger must preserve every canonical registered job exactly once')
    public_rows = []
    for row in rows:
        job = expected[row['job_id']]
        if row.get('group_id') != job.get('group_id'):
            raise ValueError('Group identity differs from canonical registry')
        accepted = row.get('primary_output_accepted')
        if type(accepted) is not bool:
            raise ValueError('Every final row needs its boolean primary acceptance flag')
        method_ids = [m['method'] for m in row.get('methods', [])]
        if len(method_ids) != len(set(method_ids)) or set(method_ids) != set(job['methods']):
            raise ValueError('Missing or duplicated planned methods in final ledger')
        methods = []
        for method in row.get('methods', []):
            if method['method'] not in job['methods']:
                raise ValueError('Unknown method in final ledger')
            methods.append({'method': method['method'], 'status': token(method['status'])})
        public_rows.append({'job_id': job['job_id'], 'group_id': job.get('group_id'),
                            'status': token(row['status']), 'primary_output_accepted': accepted,
                            'methods': methods})
    counts = dict(sorted(Counter(row['status'] for row in public_rows).items()))
    accepted_count = sum(row['primary_output_accepted'] for row in public_rows)
    if summary.get('status_counts') != counts or summary.get('primary_outputs_accepted') != accepted_count:
        raise ValueError('Summary counts disagree with bound final ledger')
    if summary.get('planned_jobs') != len(expected) or summary.get('observed_jobs') != len(rows) or summary.get('every_planned_job_retained') is not True:
        raise ValueError('Summary does not attest complete canonical job accounting')
    unchanged = summary.get('inputs_and_sources_unchanged')
    if unchanged is not None and type(unchanged) is not bool:
        raise ValueError('Malformed summary input/source stability flag')
    return {'schema': 'ccu-local-redacted-dispatch-1', 'final_jobs_sha256': sha(jobs_raw),
            'original_summary_sha256': sha(summary_raw), 'registry_jobs_sha256': sha(REGISTRY.read_bytes()),
            'status': token(summary['status']), 'inputs_and_sources_unchanged': unchanged,
            'planned_jobs': len(expected), 'observed_jobs': len(rows),
            'status_counts': counts, 'primary_outputs_accepted': accepted_count,
            'complete_identity_and_hash_accounting_checked': True,
            'scientific_acceptance_replayed_by_collector': False,
            'rows': sorted(public_rows, key=lambda row: row['job_id'])}


def build(report, dispatch, out):
    if out.exists():
        raise ValueError('Preserve existing report bundle; choose a new filename')
    payloads = {'report.json': report_bytes(report)}
    if dispatch is not None:
        payloads['dispatch_accounting.json'] = encode(dispatch_accounting(dispatch))
    bindings = ['local_run/EXPERIMENT_MATRIX.json', 'local_run/EMPIRICAL_PROGRAM.md',
                'empirical_execution/phase6/results/recipes_release/registry.json',
                'empirical_execution/phase6/results/recipes_release/jobs.jsonl.gz',
                'output/empirical_program/counterfactual_curation_empirical_protocol.tex',
                'empirical_execution/phase3/PREPARATION_AMENDMENT.txt', 'local_run/collect_results.py']
    dirty = git('status', '--porcelain', '--untracked-files=no')
    payloads['checkpoint.json'] = encode({'schema': 'ccu-local-report-checkpoint-1',
        'repository': 'https://github.com/priyankjairaj100/third',
        'git_commit': git('rev-parse','HEAD'), 'git_tree': git('rev-parse','HEAD^{tree}'),
        'tracked_checkout_clean': not bool(dirty),
        'tracked_status_entries': len(dirty.splitlines()) if dirty else 0,
        'source_sha256': {name: sha((ROOT / name).read_bytes()) for name in bindings},
        'user_report_sha256': sha(payloads['report.json']),
        'free_text_is_user_reviewed_not_automatically_sanitized': True,
        'contains_raw_data_models_or_human_forms_automatically': False,
        'reported_measurements_independently_verified': False})
    payloads['manifest.json'] = encode({'schema': 'ccu-local-return-manifest-1',
        'files': [{'name': name, 'bytes': len(raw), 'sha256': sha(raw)} for name, raw in sorted(payloads.items())],
        'manifest_self_hash': 'omitted; hash entire ZIP when reporting transport integrity'})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in sorted(payloads.items()):
                info = zipfile.ZipInfo(name, date_time=(2026,10,5,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100600 << 16
                archive.writestr(info, raw)
    return {'status': 'report_bundle_created', 'output': str(out), 'bytes': out.stat().st_size,
            'sha256': sha(out.read_bytes()), 'files': sorted(payloads),
            'scientific_measurements_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--dispatch', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.report, args.dispatch, args.out), indent=2))
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, 'BLOCKED: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
