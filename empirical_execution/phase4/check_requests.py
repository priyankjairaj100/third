#!/usr/bin/env python3
"""Natural Civil-100 record checks plus explicitly algebra-only source fixtures."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from ccu.core import BlockerGraph
from ccu.data import read_natural_jsonl, lexical_engineering_features, sha256_file
from phase3.reference_graph import build_reference_graph
from phase4.requests import generate_manifest, verify_manifest, write_manifest, digest


def expect_error(function):
    try: function()
    except (ValueError, FileExistsError): return
    raise AssertionError('expected rejection')


def direct_selected(graph, deleted):
    # Independent scalar adjacency predicate; no production sampling helper.
    out = set()
    for i, record in enumerate(graph.record_ids):
        if record in deleted: continue
        alive_predecessor = False
        for j in range(int(graph.indptr[i]), int(graph.indptr[i+1])):
            if graph.record_ids[int(graph.indices[j])] not in deleted:
                alive_predecessor = True; break
        if not alive_predecessor: out.add(record)
    return out


def audit_manifest(manifest, graph):
    verify_manifest(manifest, graph)
    ids = set(graph.record_ids); owner = dict(zip(graph.record_ids, graph.source_ids))
    initial = direct_selected(graph, set()); observations = candidates = 0
    lookup = {p['trajectory_id']: p for p in manifest['trajectories']}
    for path in manifest['trajectories']:
        order = path['deletion_order']; seen = set(); prior_deleted = set()
        assert len(set(order)) == len(order) == path['initial_horizon']
        assert path['service_universe_size'] == (len(ids) if path['unit'] == 'record' else len(set(owner.values())))
        if path['arm'] == 'U': assert set(order) <= ids-initial
        if path['arm'] == 'S':
            assert set(order) <= set(manifest['genuine_source_sampling_frame'])
        for obs in path['observations']:
            seen.update(obs['newly_requested_units'])
            assert seen == set(order[:obs['checkpoint']])
            deleted = seen if path['unit'] == 'record' else {r for r in ids if owner[r] in seen}
            assert prior_deleted <= deleted
            selected = direct_selected(graph, deleted)
            assert obs['deleted_record_ids'] == sorted(deleted)
            assert obs['selected_record_ids'] == sorted(selected)
            assert obs['admitted_record_ids'] == sorted(selected-initial)
            assert obs['admissions'] == len(selected-initial)
            assert obs['zero_admission'] == (not (selected-initial))
            assert obs['remaining_horizon'] == len(order)-len(seen)
            assert obs['empty_retained_target'] == (not selected)
            assert obs['cumulative_request_set_sha256'] == digest(sorted(seen))
            prior_deleted = set(deleted); observations += 1
        if path['arm'] == 'R-volume-matched-to-S':
            source_path = lookup[path['matched_source_trajectory_id']]
            assert path['checkpoints'] == [o['cumulative_deleted_records'] for o in source_path['observations']]
            assert path['initial_horizon'] == max(path['checkpoints'], default=0)
            assert path['seed'] != source_path['seed']
        if path['arm'].startswith('A'):
            assert path['stress_candidates_scored'] == len(path['candidates']) <= 1024
            for entry in path['candidates']:
                deleted = set(entry['blocker_ids'])
                assert entry['unpadded_admissions'] == len(direct_selected(graph, deleted)-initial)
                if path['arm'] == 'A-excluded-blockers': assert deleted <= ids-initial
                candidates += 1
            if path['candidates']:
                best = min(path['candidates'], key=lambda c: (-c['unpadded_admissions'], c['candidate_id']))
                assert path['selected_candidate'] == best['candidate_id']
                assert path['deletion_order'][:best['blocker_count']] == sorted(best['blocker_ids'])
    assert manifest['confirmatory_study_ready'] is False
    assert manifest['source_provenance_verified_by_this_module'] is False
    assert not manifest['labels_used_for_generation']
    return observations, candidates


def main():
    out = ROOT/'phase4/results'; out.mkdir(parents=True, exist_ok=True)
    data = ROOT/'data/civil_comments_engineering_preview.jsonl'
    rows = read_natural_jsonl(data); ids = [r['record_id'] for r in rows]
    x, _ = lexical_engineering_features(rows, 128)
    graph = build_reference_graph(x, ids, .6, block_size=17)
    kinds = {s: 'unknown_singleton' for s in graph.source_ids}
    timings = []
    args = dict(dataset_id='civil_comments_engineering_preview', panel_id='all100_lexical128_software_checks',
                source_kinds=kinds)
    natural = generate_manifest(graph, **args, timing_sink=timings)
    obs, candidates = audit_manifest(natural, graph)
    replay = generate_manifest(graph, **args)
    assert natural == replay
    # Exercise natural all-record exhaustion; exclusions are not fabricated.
    r_paths = [p for p in natural['trajectories'] if p['arm'] == 'R']
    assert len(r_paths) == 256 and all(p['observations'][-1]['empty_retained_target'] for p in r_paths)
    assert all(p['initial_horizon'] == 100 and p['checkpoints'] == [1, 8, 32, 100] for p in r_paths)
    assert all(not p['checkpoints'] for p in natural['trajectories'] if p['arm'] in {'S', 'R-volume-matched-to-S'})
    assert natural['arm_summaries']['R']['per_checkpoint'][-1]['distinct_request_sets'] == 1
    # Same record/source request stream under a different threshold, fixed panel.
    sensitivity = build_reference_graph(x, ids, .8, block_size=17)
    alt = generate_manifest(sensitivity, **args, allocations=dict(R=8, S=0, U=0, A=0))
    assert [p['deletion_order'] for p in alt['trajectories']] == [p['deletion_order'] for p in r_paths[:8]]
    assert natural['graph_binding'] != alt['graph_binding']
    expect_error(lambda: verify_manifest(natural, sensitivity))
    bad = copy.deepcopy(natural); bad['trajectories'][0]['deletion_order'].reverse()
    expect_error(lambda: verify_manifest(bad))
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder)/'requests.json'
        write_manifest(natural, p); expect_error(lambda: write_manifest(natural, p))
        assert json.loads(p.read_text()) == natural

    # Four-vertex mathematical adjacency/source fixture, NOT a dataset or source evidence.
    # a->b and b->c; source P owns a,b, Q owns d; c is an unknown singleton.
    fixture = BlockerGraph(('a','b','c','d'), ('P','P','unknown:c','Q'), np.arange(4),
                           np.array([0,0,1,2,2]), np.array([0,1]), .5)
    fkinds = {'P':'genuine_native','Q':'genuine_native','unknown:c':'unknown_singleton'}
    sm = generate_manifest(fixture, dataset_id='algebra_fixture_not_corpus', panel_id='source_math_only',
                           source_kinds=fkinds, allocations=dict(R=4,S=8,U=4,A=4))
    sobs, scandidates = audit_manifest(sm, fixture)
    assert sm['source_service_universe'] == ['P','Q','unknown:c']
    assert sm['genuine_source_sampling_frame'] == ['P','Q']
    for p in sm['trajectories']:
        if p['arm'] == 'S':
            assert p['initial_horizon'] == 2 and p['checkpoints'] == [1,2]
            assert p['observations'][-1]['deleted_record_ids'] == ['a','b','d']
            assert p['observations'][-1]['selected_record_ids'] == ['c']
        if p['arm'] == 'R-volume-matched-to-S': assert p['initial_horizon'] == 3
    expect_error(lambda: generate_manifest(fixture, dataset_id='fixture', panel_id='x',
                  source_kinds={**fkinds,'P':'unknown_singleton'}))
    expect_error(lambda: generate_manifest(fixture, dataset_id='fixture', panel_id='x',
                  source_kinds={k:v for k,v in fkinds.items() if k != 'unknown:c'}))
    empty = BlockerGraph((), (), np.array([],dtype=int), np.array([0]), np.array([],dtype=int), .5)
    em = generate_manifest(empty, dataset_id='empty_algebra', panel_id='zero', source_kinds={},
                           allocations=dict(R=1,S=1,U=1,A=1))
    assert all(not p['deletion_order'] and not p['observations'] for p in em['trajectories'])
    noedge = BlockerGraph(('a','b'), ('a','b'), np.arange(2), np.array([0,0,0]), np.array([],dtype=int), .9)
    nm = generate_manifest(noedge, dataset_id='noedge_algebra', panel_id='zero_excluded',
                           source_kinds={'a':'unknown_singleton','b':'unknown_singleton'},
                           allocations=dict(R=0,S=0,U=2,A=2))
    assert all(p['status'] == 'structural_zero_empty_sampling_frame' for p in nm['trajectories'] if p['arm'] == 'U')
    assert all(p['status'] == 'no_stress_candidate_random_padding_only' for p in nm['trajectories'] if p['arm'].startswith('A'))
    for name, value in [('requests_natural_engineering.json', natural),
                        ('requests_search_timing_engineering.json', dict(evidence_role='engineering_nonconfirmatory',
                            manifest_sha256=natural['manifest_sha256'], timings=timings))]:
        (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    result = dict(status='passed', natural_records=len(rows), natural_dimension=128,
                  natural_curator='lexical_engineering_reference_FP64_tau_0.6',
                  natural_trajectories=len(natural['trajectories']), natural_observations_checked=obs,
                  natural_candidate_admission_counts_checked=candidates,
                  source_fixture_observations_checked=sobs, source_fixture_candidate_counts_checked=scandidates,
                  source_fixture_scope='invented IDs/ownership on mathematical adjacency only; no empirical dataset or provenance evidence',
                  actual_native_source_experiments_run=False, actual_semantic_experiments_run=False,
                  reproducible_manifest=True, paired_R_sensitivity_requests_preserved=True,
                  cumulative_prefixes_matched_source_volumes_and_remaining_horizons_checked=True,
                  empty_graph_empty_pool_failed_stress_and_full_deletion_retained=True,
                  seal_graph_mismatch_and_overwrite_rejected=True,
                  data_sha256=sha256_file(data), manifest_sha256=natural['manifest_sha256'],
                  code_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__), ROOT/'phase4/requests.py']})
    (out/'requests_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
