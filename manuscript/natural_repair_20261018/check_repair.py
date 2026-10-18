#!/usr/bin/env python3
"""Replay the Stage 2 manuscript evidence chain (Python 3.10+, stdlib only).

Uses only public aggregates and saved review receipts. Private observations,
features, states, model inference and universal proofs are not regenerated.
Historical checkers run only in temporary extractions of their pinned release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile

SOURCE_SHA = 'd42068febcb1b91ef9e35326cbc6bd2c19b3e6fbce8d0ee6091be53450b74703'
RELEASE_SHA = '71242f643f753b32ce9891ff8d971e21526155d89d53f7cda78ed0433fb77172'
BUNDLE_SHA = '611bb4baec6b62be1b7879f5d8f6afc3a41ffe99fc157101e252562c1ad9f475'
SUPPLEMENT_SHA = '99c9d2d4f962abc12480dae2cc89b3852a850bb825ed586182606232b9289a9c'
SOURCE_REL = 'manuscript/third_working_draft.tex'
HERE_REL = 'manuscript/natural_repair_20261018'
BUNDLE_REL = 'local_run/postnegative_curation_repair_results_v1_20261018'


def need(condition, code):
    if not condition:
        raise ValueError(code)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def binding(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': digest(raw)}


def bytes_binding(raw):
    return {'bytes': len(raw), 'sha256': digest(raw)}


def read(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, 'duplicate_json_key')
            out[key] = value
        return out
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)


def under(root, relative):
    name = PurePosixPath(relative)
    need(isinstance(relative, str) and not name.is_absolute() and
         '..' not in name.parts and '\\' not in relative, 'unsafe_relative_path')
    target = root.joinpath(*name.parts)
    need(target.resolve().is_relative_to(root.resolve()), 'escaped_input_root')
    return target


def extract(archive, destination):
    """Bounded extraction without traversal, duplicate names or symlinks."""
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        need(len(infos) <= 2000, 'archive_member_cap')
        need(len({i.filename for i in infos}) == len(infos), 'duplicate_zip_name')
        need(sum(i.file_size for i in infos) <= 128 * 1024**2, 'archive_expansion_cap')
        need(z.testzip() is None, 'archive_crc')
        for info in infos:
            mode = info.external_attr >> 16
            need(not stat.S_ISLNK(mode), 'archive_symlink')
            target = under(destination, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))
        return len(infos)


def run_json(script, arguments, cwd):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-B', str(script), *arguments],
                            cwd=cwd, env=env, text=True, capture_output=True,
                            timeout=180, check=False)
    need(result.returncode == 0, 'child_replay_failed:' + script.name)
    out = json.loads(result.stdout)
    need(out.get('passed') is True, 'child_replay_did_not_pass:' + script.name)
    return out


def check(root):
    need(sys.version_info >= (3, 10), 'Python_3_10_or_newer_required')
    here = under(root, HERE_REL)
    source = under(root, SOURCE_REL)
    source_bytes = source.read_bytes()
    need(digest(source_bytes) == SOURCE_SHA, 'current_source_pin')
    delta_path = here / 'delta_stage2.json'
    delta = read(delta_path)
    need(delta['schema'] == 'third-natural-repair-manuscript-delta-v1', 'delta_schema')
    need(delta['new_source_sha256'] == SOURCE_SHA, 'delta_target')
    need(delta['stage2_results_included'] is True and
         delta['original_failed_gates_changed'] is False, 'delta_scope')
    changes = delta['changes']
    need(len(changes) == 5, 'exact_five_changes')
    archive = under(root, 'manuscript/release_20261018_tmlr_v3.zip')
    need(binding(archive)['sha256'] == RELEASE_SHA, 'immutable_v3_release_pin')
    with tempfile.TemporaryDirectory(prefix='third-repair-history-') as tmp:
        historical_root = Path(tmp)
        members = extract(archive, historical_root)
        original = under(historical_root, SOURCE_REL).read_bytes()
        need(digest(original) == delta['prior_source_sha256'], 'delta_prior_source')
        reconstructed = original.decode('utf-8')
        for change in changes:
            need(set(change) == {'old', 'new'}, 'delta_change_fields')
            need(change['old'] and reconstructed.count(change['old']) == 1,
                 'delta_anchor_not_unique')
            reconstructed = reconstructed.replace(change['old'], change['new'])
        need(reconstructed.encode('utf-8') == source_bytes, 'exact_delta_reconstruction')
        prior_manifest = binding(under(historical_root,
            'local_run/postnegative_structure_results_v1_20261018/MANIFEST.json'))
        historical = run_json(under(historical_root,
            'manuscript/natural_development_20261018/check_natural.py'),
            ['--root', str(historical_root), '--output', str(historical_root/'replay_report.json')],
            historical_root)
        need(historical['source_sha256'] == digest(original) and
             historical['stage2_results_included'] is False and
             historical['private_observations_authenticated'] is False,
             'historical_replay_scope')
    bundle = under(root, BUNDLE_REL)
    manifest_path = bundle / 'MANIFEST.json'
    need(binding(manifest_path)['sha256'] == BUNDLE_SHA, 'stage2_bundle_pin')
    manifest = read(manifest_path)
    need(manifest['schema'] == 'third-stage2-public-bundle-manifest-v1', 'bundle_schema')
    for name, expected in manifest['files'].items():
        need(binding(under(bundle, name)) == expected, 'stage2_bundle_member_binding')
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    need(actual == set(manifest['files']) | {'MANIFEST.json'}, 'stage2_bundle_inventory')
    stage2 = run_json(bundle / 'replay_aggregates.py', [], root)
    summary = stage2['summary']
    need(summary['thresholds_completed'] == 51 and
         summary['record_arms_completed'] == summary['source_arms_completed'] == 51 and
         summary['singleton_membership_checks'] == 13056 and
         summary['checkpoint_count'] == 408 and summary['unavailable_arms'] == [],
         'stage2_counts')
    need(summary['no_selected_cutoff'] is True and
         summary['human_or_model_annotations'] == 0 and
         summary['task_learner_trained'] is False, 'stage2_claim_limits')
    need(binding(bundle/'independent_review.json')['sha256'] ==
         delta['stage2_independent_review_sha256'], 'delta_aggregate_review_binding')
    review_path = here / 'delta_review_stage2.json'
    review = read(review_path)
    need(review['schema'] == 'third-manuscript-stage2-independent-delta-review-v1' and
         review['passed'] is True and review['findings'] == [], 'delta_review_pass')
    rb = review['bindings']
    need(rb['current_manuscript'] == rb['source_after'] == bytes_binding(source_bytes) and
         rb['source_before'] == bytes_binding(original) and
         rb['delta'] == binding(delta_path), 'delta_review_source_bindings')
    for key, name in [('stage2_public_manifest','MANIFEST.json'),
                      ('stage2_independent_review','independent_review.json'),
                      ('stage2_field_interpretation','output_field_interpretation.json'),
                      ('stage2_resource_receipt','resource_receipt.json')]:
        need(rb[key] == binding(bundle/name), 'delta_review_evidence_binding')
    source_bindings = read(bundle/'SOURCE_BINDINGS.json')
    need(rb['stage2_executed_protocol'] == source_bindings['executed_protocol'] and
         rb['stage2_scientific_source'] == source_bindings['executed_package_members']['curation_repair.py'] and
         rb['preserved_stage1_public_manifest'] == prior_manifest,
         'delta_review_historical_source_bindings')
    reconstruction = review['exact_delta_reconstruction']
    need(reconstruction['replacements'] == 5 and all(reconstruction[k] is True for k in
         ('each_original_occurs_once','no_other_source_change',
          'prior_source_is_reviewed_stage1_source','reconstructed_after_equals_current')),
         'delta_review_reconstruction')
    claims = review['verified_claims']
    need(claims['singleton_membership_comparisons'] == 13056 and
         claims['checkpoint_count'] == 408 and
         claims['independent_audit_limited_to_bound_receipts'] is True,
         'delta_review_claim_scope')
    visual_path = here / 'visual_review.json'
    visual = read(visual_path)
    need(visual['schema'] == 'third-temporary-pdf-visual-review-v1' and
         visual['passed'] is True and visual['layout_passed'] is True,
         'current_visual_review_pass')
    need(visual['manuscript_source_sha256'] == SOURCE_SHA and
         visual['source_bytes'] == len(source_bytes) and visual['no_source_edits'] is True,
         'current_visual_source_binding')
    need(visual['findings'] == [] and visual['out_of_page_text'] == [] and
         visual['overfull_warnings'] == [] and visual['pages'] == 15 and
         visual['visually_inspected_full_pages'] == list(range(1,16)),
         'visual_review_findings')
    need(visual['native_preview_equivalence_claim'] is False and
         visual['previous_reviews_preserved'] is True, 'visual_review_scope')
    delivery_path = here / 'supplement_delivery.json'
    delivery = read(delivery_path)
    need(delivery['schema'] == 'anonymous-supplement-delivery-v4' and
         delivery['manuscript_source_sha256'] == SOURCE_SHA, 'supplement_delivery_source')
    need(delivery['stage2_bundle_bytes_preserved'] is True and
         delivery['previous_zip_bytes_preserved'] is True and
         delivery['new_independent_scientific_verification'] is False,
         'supplement_delivery_scope')
    privacy = delivery['privacy_screening']
    need(privacy['passed'] is True and privacy['findings'] == [] and
         privacy['private_data_or_operational_wrappers_included'] is False,
         'supplement_privacy_receipt')
    supplement = under(here, delivery['archive']['file'])
    expected = {k: delivery['archive'][k] for k in ('bytes','sha256')}
    need(binding(supplement) == expected and expected['sha256'] == SUPPLEMENT_SHA,
         'immutable_supplement_v4_pin')
    with tempfile.TemporaryDirectory(prefix='third-repair-supplement-') as tmp:
        supplement_root = Path(tmp)
        count = extract(supplement, supplement_root)
        need(count == delivery['archive_members'], 'supplement_member_count')
        anonymous = run_json(supplement_root/'check_all.py', [], supplement_root)
        need(anonymous == delivery['clean_extraction_replay'], 'supplement_replay_receipt_identity')
        need(anonymous['manuscript_source_sha256'] == SOURCE_SHA and
             anonymous['original_semantic_gate_passed'] is False and
             anonymous['generating_code_executed'] is False and
             anonymous['repair_stage2'] == stage2, 'anonymous_replay_scope')
    # Inputs are immutable; child scripts receive -B and do not write bundles.
    need(source.read_bytes() == source_bytes and binding(manifest_path)['sha256'] == BUNDLE_SHA,
         'source_or_bundle_manifest_changed_during_check')
    for name, expected in manifest['files'].items():
        need(binding(under(bundle, name)) == expected, 'stage2_bundle_mutated_during_check')
    return {
        'schema':'third-natural-repair-source-check-v1', 'passed':True,
        'source_sha256':SOURCE_SHA, 'exact_author_delta_reproduced':True, 'changes':5,
        'historical_v3_release_sha256':RELEASE_SHA, 'historical_release_members':members,
        'archived_stage1_checker_replay_passed':historical['passed'],
        'historical_reviews_rebound':False, 'stage2_public_manifest_sha256':BUNDLE_SHA,
        'stage2_aggregate_replay':stage2, 'delta_review_sha256':binding(review_path)['sha256'],
        'visual_review_sha256':binding(visual_path)['sha256'], 'source_bound_layout_receipt_passed':True,
        'anonymous_supplement_replay_passed':anonymous['passed'],
        'anonymous_supplement_sha256':SUPPLEMENT_SHA,
        'supplement_delivery_sha256':binding(delivery_path)['sha256'],
        'original_semantic_gate_passed':False, 'original_effect_study_complete':False,
        'private_observations_authenticated':False, 'private_graphs_or_states_recomputed':False,
        'current_source_compiled_by_this_checker':False, 'native_runtime_repaired_claim':False,
        'scientific_scope':'Exact source delta, public aggregate arithmetic and saved review receipts only; no new private-data, graph/state, inference, proof, utility or efficiency verification.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    need(not output.is_relative_to(under(root, 'local_run')), 'output_inside_immutable_bundle')
    need(not output.is_relative_to(under(root, 'manuscript/natural_development_20261018')),
         'output_inside_historical_evidence')
    forbidden = {under(root,SOURCE_REL), under(root,'manuscript/release_20261018_tmlr_v3.zip')}
    here = under(root,HERE_REL)
    forbidden.update(here/name for name in ('check_repair.py','delta_stage2.json','delta_review_stage2.json',
        'visual_review.json','supplement_delivery.json','tmlr_anonymous_supplement_v4.zip'))
    need(output not in forbidden, 'output_overwrites_input')
    result = check(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
