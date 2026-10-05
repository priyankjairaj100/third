#!/usr/bin/env python3
"""Focused metadata-only collector checks. No corpus, model, human, or worker runs."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'empirical_execution'))
from phase9 import study_assembly

spec = importlib.util.spec_from_file_location('local_return_collector', ROOT / 'local_run/collect_results.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new immutable review report')
    checks = []
    report = {
        'schema': 'ccu-local-collector-review-1', 'status': 'failed',
        'evidence_role': 'metadata_only_software_controls',
        'corpus_records_created': 0, 'human_responses_created': 0,
        'workers_executed': 0, 'primary_study_started': False,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / 'local_run/collect_results.py', ROOT / 'empirical_execution/phase9/study_assembly.py', ROOT / 'local_run/templates/results_report.template.json')},
    }
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    def refuses(name, function):
        try:
            function()
        except (ValueError, KeyError, TypeError, OSError):
            checks.append(name)
            return
        raise AssertionError(name)
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    try:
        with tempfile.TemporaryDirectory(prefix='ccu_local_return_checks_') as temporary:
            base = Path(temporary)
            config = base / 'owners.json'
            write(config, {'owners': []})
            plan = study_assembly.assemble(config, base / 'plan')
            outcome = study_assembly.execute(base / 'plan/plan.json', base / 'dispatch')
            check('actual_empty_phase9_dispatch_complete_and_no_jobs_attempted',
                  outcome['observed_jobs'] == 21332 and outcome['every_planned_job_retained']
                  and outcome['attempted_jobs'] == outcome['primary_outputs_accepted'] == 0)
            directory = base / 'dispatch'
            rows = json.loads((directory / 'jobs.json').read_text())
            summary = json.loads((directory / 'summary.json').read_text())
            check('actual_phase9_complete_ledger_export', collector.dispatch_accounting(directory)['observed_jobs'] == 21332)
            # Successful extension executors omit group_id; blocked rows use null.
            changed = copy.deepcopy(rows)
            extension = next(row for row in changed if row['job_id'].startswith('H_recipes/'))
            extension.pop('group_id', None)
            extension['message'] = 'PRIVATE_SENTINEL_NEVER_RETURN'
            extension['artifact'] = '/private/account/PRIVATE_SENTINEL_NEVER_RETURN'
            write(directory / 'jobs.json', changed)
            altered = copy.deepcopy(summary)
            altered['final_jobs_sha256'] = collector.sha((directory / 'jobs.json').read_bytes())
            write(directory / 'summary.json', altered)
            public = collector.dispatch_accounting(directory)
            check('extension_missing_group_id_supported', public['observed_jobs'] == 21332)
            check('private_paths_and_errors_not_exported', 'PRIVATE_SENTINEL_NEVER_RETURN' not in json.dumps(public))
            write(directory / 'jobs.json', rows)
            write(directory / 'summary.json', summary)
            original_jobs = (directory / 'jobs.json').read_bytes()
            original_summary = (directory / 'summary.json').read_bytes()
            # Rebind once because formatting differs from the frozen writer.
            summary['final_jobs_sha256'] = collector.sha(original_jobs)
            write(directory / 'summary.json', summary)
            original_summary = (directory / 'summary.json').read_bytes()
            bad = copy.deepcopy(rows); bad[-1] = bad[0]
            write(directory / 'jobs.json', bad)
            new = copy.deepcopy(summary); new['final_jobs_sha256'] = collector.sha((directory / 'jobs.json').read_bytes())
            write(directory / 'summary.json', new)
            refuses('duplicate_or_missing_canonical_job_rejected', lambda: collector.dispatch_accounting(directory))
            (directory / 'jobs.json').write_bytes(original_jobs)
            (directory / 'summary.json').write_bytes(original_summary)
            (directory / 'jobs.json').write_bytes(original_jobs + b' ')
            refuses('changed_ledger_hash_rejected', lambda: collector.dispatch_accounting(directory))
            (directory / 'jobs.json').write_bytes(original_jobs)
            bad_summary = copy.deepcopy(summary); bad_summary['status_counts'] = {'completed': 21332}
            write(directory / 'summary.json', bad_summary)
            refuses('inconsistent_status_counts_rejected', lambda: collector.dispatch_accounting(directory))
            bad_summary = copy.deepcopy(summary); bad_summary['schema'] = 'unrelated-summary'
            write(directory / 'summary.json', bad_summary)
            refuses('wrong_dispatch_schema_rejected', lambda: collector.dispatch_accounting(directory))
            (directory / 'summary.json').write_bytes(original_summary)
            for kind in ('missing', 'duplicate'):
                altered_rows = copy.deepcopy(rows)
                methods = altered_rows[0]['methods']
                if kind == 'missing': methods.clear()
                else: methods.append(copy.deepcopy(methods[0]))
                write(directory / 'jobs.json', altered_rows)
                altered_summary = copy.deepcopy(summary)
                altered_summary['final_jobs_sha256'] = collector.sha((directory / 'jobs.json').read_bytes())
                write(directory / 'summary.json', altered_summary)
                refuses(kind + '_planned_method_rejected', lambda: collector.dispatch_accounting(directory))
            (directory / 'jobs.json').write_bytes(original_jobs)
            (directory / 'summary.json').write_bytes(original_summary)
            template = json.loads((ROOT / 'local_run/templates/results_report.template.json').read_text())
            template.update(report_id='metadata-only-check', next_action='No empirical action; software-control report.',
                            status='completed', reviewed_for_sharing=True,
                            completed_work=['Explicit metadata-only software control; no empirical measurements.'])
            report_path = base / 'report.json'; write(report_path, template)
            collector.build(report_path, None, base / 'partial.zip')
            check('partial_report_without_dispatch_supported', (base / 'partial.zip').is_file())
            collector.build(report_path, directory, base / 'full1.zip')
            collector.build(report_path, directory, base / 'full2.zip')
            check('deterministic_zip_for_unchanged_inputs', (base / 'full1.zip').read_bytes() == (base / 'full2.zip').read_bytes())
            with zipfile.ZipFile(base / 'full1.zip') as archive:
                check('only_four_allowlisted_members_in_dispatch_bundle', set(archive.namelist()) == {'report.json', 'checkpoint.json', 'manifest.json', 'dispatch_accounting.json'})
                check('collector_disclaims_measurement_verification', json.loads(archive.read('checkpoint.json'))['reported_measurements_independently_verified'] is False)
                manifest = json.loads(archive.read('manifest.json'))
                check('archive_members_match_manifest_hashes', all(collector.sha(archive.read(item['name'])) == item['sha256'] for item in manifest['files']))
            refuses('existing_zip_never_overwritten', lambda: collector.build(report_path, None, base / 'partial.zip'))
            template['reviewed_for_sharing'] = False; write(report_path, template)
            refuses('unreviewed_report_rejected', lambda: collector.build(report_path, None, base / 'unreviewed.zip'))
        report.update(status='passed', checks=checks, checks_passed=len(checks), primary_jobs_executed=0)
    except Exception as error:
        report.update(checks=checks, checks_passed=len(checks), exception_type=type(error).__name__, reason=str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status':report['status'], 'checks_passed':len(checks), 'out':str(args.out)}))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
