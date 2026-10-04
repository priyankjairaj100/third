#!/usr/bin/env python3
"""Software contract checks using existing natural records, never empirical evidence."""
import copy
import json
from pathlib import Path
from intake_validate import Validator, sha256, inventory, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'assets_intake_checks.json'
checks = []
skipped = []

def require(name, ok):
    if not ok:
        raise AssertionError(name)
    checks.append({'check': name, 'passed': True})

inv = inventory(ROOT)
require('natural_civil_fixture_is_100_rows', next(x for x in inv['natural_fixture_inventory'] if x['dataset_id'] == 'civil_comments')['rows'] == 100)
news = next((x for x in inv['natural_fixture_inventory'] if x['dataset_id'] == 'cc_news'), None)
if news is not None and 'rows' in news:
    require('natural_news_fixture_is_90_rows', news['rows'] == 90)
else:
    skipped.append({'check':'natural_news_fixture_is_90_rows',
                    'reason':'News text is omitted from the portable Civil-only package'})
require('inventory_blocks_confirmation', inv['confirmatory_ready'] is False)
fixture = ROOT / 'empirical_execution/data/civil_comments_engineering_preview.jsonl'
design = ROOT / 'output/empirical_program/study_design.json'
manifest = {'schema_version': SCHEMA_VERSION, 'synthetic_data_allowed': False,
            'study_design': {'path': str(design.relative_to(ROOT)), 'sha256': sha256(design)},
            'datasets': [{'dataset_id': 'civil_comments', 'outputs': 1,
                          'rows': {'path': str(fixture.relative_to(ROOT)), 'sha256': sha256(fixture)}}]}
report = Validator(ROOT).validate(manifest)
require('existing_preview_not_accepted_as_confirmatory_input', not report['mechanical_intake_passed'] and not report['confirmatory_ready'])
require('preview_missing_original_fields_detected', any(x.endswith('.original_fields_internal_consistency') for x in report['failed_checks']))
require('preview_missing_source_partitions_detected', any(x.endswith('.partitions_present') for x in report['failed_checks']))
require('preview_missing_semantic_cache_detected', any(x.endswith('.embeddings') for x in report['failed_checks']))
require('preview_missing_calibration_detected', any(x.endswith('.calibration') for x in report['failed_checks']))
forged = copy.deepcopy(manifest)
forged['human_approved'] = True
forged['reviewer'] = 'approved'
forged['quality_verified'] = 'yes'
forged_report = Validator(ROOT).validate(forged)
require('arbitrary_human_approval_fields_do_not_open_gate', not forged_report['confirmatory_ready'])
wrong = copy.deepcopy(manifest)
wrong['datasets'][0]['rows']['sha256'] = '0' * 64
require('wrong_digest_rejected', any(x.endswith('.rows') for x in Validator(ROOT).validate(wrong)['failed_checks']))
validator = Validator(ROOT)
require('outside_workspace_path_rejected', validator.artifact({'path': '../outside', 'sha256': '0'*64}, 'escape') is None)
# Changing source partition labels or matrix data would create manufactured
# empirical inputs. Missing-field/malformed-manifest tests suffice at this gate.
OUT.write_text(json.dumps({'scope': 'Software checks on already available natural fixtures and deliberately malformed metadata; not empirical results.',
                          'checks_passed': len(checks), 'checks': checks, 'skipped': skipped,
                          'fixture_hashes': {str(fixture.relative_to(ROOT)): sha256(fixture)},
                          'real_fixture_validation_failed_checks': report['failed_checks']}, indent=2) + '\n')
print(json.dumps({'checks_passed': len(checks), 'out': str(OUT)}))
