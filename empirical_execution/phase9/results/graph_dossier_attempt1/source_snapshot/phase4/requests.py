"""Frozen, label-blind deletion workloads for a fixed blocker graph.

No genuine-source provenance or semantic quality is inferred from metadata.
Graph-only diagnostic outcomes in a manifest belong to the isolated auditor,
never to the identifier-only repair worker. Wall-clock timings are separate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import platform
import random
import time

import numpy as np

SCHEMA = 'ccu-frozen-requests-1'
SOURCE_KINDS = {'genuine_native', 'unknown_singleton', 'engineering_proxy'}
DEFAULT_REQUESTS = dict(record_horizon=128, record_checkpoints=[1, 8, 32, 128],
                        source_horizon=8, source_checkpoints=[1, 2, 4, 8])
DEFAULT_ALLOCATIONS = dict(R=256, S=256, U=128, A=32)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _integer(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(f'{name} must be a {"positive" if positive else "nonnegative"} integer')
    return value


def derive_seed(master_seed, namespace, *parts):
    """Public SHA-256 stream key; length-prefixed UTF-8, first 128 digest bits."""
    _integer(master_seed, 'master_seed')
    h = hashlib.sha256()
    for item in (str(master_seed), namespace, *parts):
        if not isinstance(item, str):
            raise ValueError('seed namespace/parts must be strings')
        b = item.encode(); h.update(len(b).to_bytes(8, 'big')); h.update(b)
    return int.from_bytes(h.digest()[:16], 'big')


def _graph_state(graph):
    ids = tuple(graph.record_ids); sources = tuple(graph.source_ids)
    n = len(ids)
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != n:
        raise ValueError('unique nonempty stable record IDs required')
    if len(sources) != n or any(not isinstance(x, str) or not x for x in sources):
        raise ValueError('complete source ownership partition required')
    priority = tuple(graph.priority)
    if len(priority) != n or set(priority) != set(ids):
        raise ValueError('priority must be a complete record permutation')
    rank = {r: i for i, r in enumerate(priority)}
    ptr = np.asarray(graph.indptr); index = np.asarray(graph.indices)
    if (ptr.shape != (n+1,) or ptr.dtype.kind not in 'iu' or index.ndim != 1
            or index.dtype.kind not in 'iu' or ptr[0] != 0 or ptr[-1] != len(index)
            or np.any(ptr[1:] < ptr[:-1]) or np.any(index < 0) or np.any(index >= n)):
        raise ValueError('invalid blocker CSR')
    blockers = {}
    for i, rid in enumerate(ids):
        b = tuple(ids[int(j)] for j in index[int(ptr[i]):int(ptr[i+1])])
        if len(set(b)) != len(b) or any(rank[u] >= rank[rid] for u in b):
            raise ValueError('blockers must be distinct and precede owner in fixed priority')
        blockers[rid] = frozenset(b)
    threshold = float(graph.threshold)
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('finite graph threshold in [-1,1] required')
    owners = dict(zip(ids, sources))
    binding = dict(
        curator='global_earlier_raw_neighbor_suppression', records=n,
        edges=sum(map(len, blockers.values())), threshold_hex=threshold.hex(),
        record_ids_sha256=digest(sorted(ids)), priority_sha256=digest(priority),
        blocker_edges_sha256=digest([[r, sorted(blockers[r])] for r in sorted(ids)]),
        source_partition_sha256=digest([[r, owners[r]] for r in sorted(ids)]))
    binding['graph_sha256'] = digest(binding)
    return tuple(sorted(ids)), owners, blockers, binding


def graph_binding(graph):
    return _graph_state(graph)[3]


def _levels(requested, horizon):
    for x in requested:
        _integer(x, 'checkpoint', positive=True)
    return sorted({min(k, horizon) for k in requested}) if horizon else []


def _selected(ids, blockers, deleted):
    return {r for r in ids if r not in deleted and blockers[r] <= deleted}


def _quantiles(values):
    values = sorted(values)
    if not values:
        return {str(p): None for p in (0, .25, .5, .75, 1)}
    # Inverse empirical CDF; avoid a library-dependent interpolation convention.
    return {str(p): values[max(0, math.ceil(p*len(values))-1)] for p in (0, .25, .5, .75, 1)}


def _observations(ids, owners, blockers, path):
    initial = _selected(ids, blockers, set()); previous_selected = initial
    previous_units = set(); previous_deleted = set(); result = []
    for checkpoint in path['checkpoints']:
        units = set(path['deletion_order'][:checkpoint])
        deleted = units if path['unit'] == 'record' else {r for r in ids if owners[r] in units}
        current = _selected(ids, blockers, deleted); admitted = current - initial
        new_deleted = deleted - previous_deleted
        result.append(dict(
            checkpoint=checkpoint, cumulative_deleted_units=len(units),
            cumulative_deleted_records=len(deleted), remaining_horizon=path['initial_horizon']-len(units),
            newly_requested_units=[u for u in path['deletion_order'][:checkpoint] if u not in previous_units],
            deleted_record_ids=sorted(deleted), selected_record_ids=sorted(current),
            admitted_record_ids=sorted(admitted), admissions=len(admitted), has_admission=bool(admitted),
            originally_selected_deleted=len(initial & deleted), current_selected_records=len(current),
            previous_release_selected_deleted=len(previous_selected & new_deleted),
            retained_records=len(ids)-len(deleted), empty_retained_target=not current,
            zero_admission=not admitted, cumulative_request_set_sha256=digest(sorted(units))))
        previous_units, previous_deleted, previous_selected = units, deleted, current
    return result


def _path(arm, index, seed, unit, requested_horizon, frame, order, levels,
          ids, owners, blockers, metadata=None):
    horizon = len(order)
    if len(set(order)) != horizon or not set(order) <= set(frame):
        raise ValueError('request order must contain unique sampling-frame units')
    path = dict(trajectory_id=f'{arm}{index:03d}', arm=arm, unit=unit, seed=seed,
                requested_horizon=requested_horizon, initial_horizon=horizon,
                sampling_frame_size=len(frame), sampling_frame_sha256=digest(sorted(frame)),
                service_universe_size=len(ids) if unit == 'record' else len(set(owners.values())),
                deletion_order=list(order), checkpoints=list(levels),
                horizon_clipped=horizon < requested_horizon,
                clipping_reason='sampling_frame_smaller_than_requested_horizon' if horizon < requested_horizon else None,
                status='ready' if horizon else 'structural_zero_empty_sampling_frame',
                reset_between_checkpoints=False, budget='cumulative_unique_units_no_reset',
                exhaustion_policy='atomic_refusal_requires_separate_rebuild_to_reset')
    if metadata:
        path.update(metadata)
    path['observations'] = _observations(ids, owners, blockers, path)
    return path


def _candidate_search(ids, blockers, limit, rng, excluded_only):
    excluded = {r for r in ids if blockers[r]}
    pool = [r for r in ids if 0 < len(blockers[r]) <= limit
            and (not excluded_only or blockers[r] <= excluded)]
    sampled = sorted(rng.sample(pool, min(1024, len(pool))))
    inverse = defaultdict(list)
    for r in ids:
        for b in blockers[r]:
            inverse[b].append(r)
    scores = {}; candidates = []; best = None; best_score = -1
    for candidate in sampled:
        deleted = blockers[candidate]
        if deleted not in scores:
            touched = Counter(r for b in deleted for r in inverse[b])
            scores[deleted] = sum(r not in deleted and count == len(blockers[r])
                                  for r, count in touched.items())
        score = scores[deleted]
        candidates.append(dict(candidate_id=candidate, blocker_ids=sorted(deleted),
                               blocker_count=len(deleted), unpadded_admissions=score))
        if score > best_score:
            best, best_score = candidate, score
    prefix = sorted(blockers[best]) if best is not None else []
    prefix_set = set(prefix)
    remainder = [r for r in ids if r not in prefix_set]
    order = prefix + rng.sample(remainder, limit-len(prefix))
    return order, dict(
        stress_candidate_pool=len(pool), stress_candidates_scored=len(sampled),
        stress_candidate_cap=1024, candidate_sampling='uniform_without_replacement_or_census',
        candidates=candidates, distinct_candidate_blocker_sets_scored=len(scores),
        selected_candidate=best, stress_unpadded_admissions=best_score if best is not None else None,
        stress_unpadded_deleted_records=len(prefix),
        stress_ranking='admissions_after_exact_candidate_blocker_set_then_stable_candidate_ID',
        stress_prefix_order='sorted_stable_blocker_IDs', padding_frame='all_records',
        excluded_blockers_only=excluded_only,
        stress_search_status='candidate_found' if best is not None else 'no_candidate_random_padding_only')


def _code_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/'phase3/reference_graph.py', root/'phase3/panels.py']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def generate_manifest(graph, *, dataset_id, panel_id, source_kinds, design=None,
                      allocations=None, evidence_role='engineering_nonconfirmatory',
                      master_seed=20271003, record_horizon=None, source_horizon=None,
                      include_excluded_blocker_stress=True, timing_sink=None):
    """Freeze complete R/S/U/A plans before model outcomes.

    ``source_kinds`` is a provenance declaration, not verification of ownership.
    Labels and model outcomes are intentionally absent from this interface.
    A's graph outcomes are allowed only to instantiate the registered stress arm.
    ``timing_sink`` is an optional list; search timings are outside the seal.
    """
    for name, value in [('dataset_id', dataset_id), ('panel_id', panel_id)]:
        if not isinstance(value, str) or not value:
            raise ValueError(f'{name} must be a nonempty string')
    if evidence_role not in {'engineering_nonconfirmatory', 'prospective_confirmatory_preparation'}:
        raise ValueError('unsupported evidence role')
    _integer(master_seed, 'master_seed')
    ids, owners, blockers, binding = _graph_state(graph)
    source_ids = sorted(set(owners.values())); size = Counter(owners.values())
    if not isinstance(source_kinds, dict) or set(source_kinds) != set(source_ids):
        raise ValueError('declare the kind of every unit in the complete source partition')
    if any(kind not in SOURCE_KINDS for kind in source_kinds.values()):
        raise ValueError('source kind must be genuine_native, unknown_singleton, or engineering_proxy')
    if any(size[s] != 1 for s in source_ids if source_kinds[s] == 'unknown_singleton'):
        raise ValueError('unknown singleton sources must own exactly one record')
    genuine = [s for s in source_ids if source_kinds[s] == 'genuine_native']
    excluded = [r for r in ids if blockers[r]]
    design = {} if design is None else deepcopy(design)
    config = {**DEFAULT_REQUESTS, **design.get('requests', {})}
    if record_horizon is not None: config['record_horizon'] = record_horizon
    if source_horizon is not None: config['source_horizon'] = source_horizon
    kr = _integer(config['record_horizon'], 'record_horizon')
    ks = _integer(config['source_horizon'], 'source_horizon')
    allocations = dict(DEFAULT_ALLOCATIONS if allocations is None else allocations)
    if set(allocations) - {'R', 'S', 'U', 'A'}:
        raise ValueError('allocations only accept R/S/U/A; matched controls generated automatically')
    for arm in DEFAULT_ALLOCATIONS:
        allocations[arm] = _integer(allocations.get(arm, 0), f'{arm} allocation')
    if not isinstance(include_excluded_blocker_stress, bool):
        raise ValueError('include_excluded_blocker_stress must be Boolean')
    salt = design.get('seeds', {}).get('request_salt', 'ccu-v1-request')
    trajectories = []
    for arm in ['R', 'S', 'U', 'A', 'A-excluded-blockers']:
        if arm == 'A-excluded-blockers' and not include_excluded_blocker_stress: continue
        count = allocations['A' if arm.startswith('A') else arm]
        frame = genuine if arm == 'S' else excluded if arm == 'U' else ids
        requested = ks if arm == 'S' else kr; limit = min(requested, len(frame))
        levels = _levels(config['source_checkpoints' if arm == 'S' else 'record_checkpoints'], limit)
        for index in range(count):
            seed = derive_seed(master_seed, salt, dataset_id, panel_id, arm, f'{index:03d}')
            rng = random.Random(seed); extra = {}
            if arm.startswith('A'):
                started = time.perf_counter()
                order, extra = _candidate_search(ids, blockers, limit, rng, arm == 'A-excluded-blockers')
                if timing_sink is not None:
                    timing_sink.append(dict(trajectory_id=f'{arm}{index:03d}',
                                            graph_search_seconds=time.perf_counter()-started,
                                            candidates_scored=extra['stress_candidates_scored']))
                if extra['selected_candidate'] is None and limit:
                    extra['status'] = 'no_stress_candidate_random_padding_only'
            else:
                # A partial random permutation has exactly the required prefix law.
                order = rng.sample(list(frame), limit)
            path = _path(arm, index, seed, 'source' if arm == 'S' else 'record',
                         requested, frame, order, levels, ids, owners, blockers, extra)
            trajectories.append(path)
            if arm == 'S':
                counts = [o['cumulative_deleted_records'] for o in path['observations']]
                matched_limit = max(counts, default=0)
                mseed = derive_seed(master_seed, salt, dataset_id, panel_id,
                                    'R-volume-matched-to-S', f'{index:03d}')
                matched_order = random.Random(mseed).sample(list(ids), matched_limit)
                matched = _path('R-volume-matched-to-S', index, mseed, 'record', matched_limit,
                                ids, matched_order, sorted(set(counts)), ids, owners, blockers,
                                dict(matched_source_trajectory_id=path['trajectory_id'],
                                     matched_source_checkpoints=levels,
                                     source_raw_count_prefixes=counts,
                                     separate_record_horizon=True,
                                     status='ready' if counts else 'no_source_checkpoints'))
                trajectories.append(matched)
    summaries = {}
    for arm in sorted({p['arm'] for p in trajectories}):
        paths = [p for p in trajectories if p['arm'] == arm]
        levels = sorted({o['checkpoint'] for p in paths for o in p['observations']})
        summaries[arm] = dict(
            trajectories=len(paths), status_counts=dict(Counter(p['status'] for p in paths)),
            horizon_clipped_trajectories=sum(p['horizon_clipped'] for p in paths),
            distinct_ordered_trajectories=len({tuple(p['deletion_order']) for p in paths}),
            per_checkpoint=[dict(checkpoint=k, contributing_trajectories=sum(k in p['checkpoints'] for p in paths),
                                 distinct_request_sets=len({o['cumulative_request_set_sha256'] for p in paths
                                                           for o in p['observations'] if o['checkpoint'] == k}),
                                 zero_admission_trajectories=sum(o['zero_admission'] for p in paths
                                                               for o in p['observations'] if o['checkpoint'] == k),
                                 empty_retained_targets=sum(o['empty_retained_target'] for p in paths
                                                            for o in p['observations'] if o['checkpoint'] == k))
                            for k in levels])
    configuration = dict(requests=config, allocations=allocations, master_seed=master_seed,
                         request_salt=salt, include_excluded_blocker_stress=include_excluded_blocker_stress,
                         seed_derivation='SHA256 length-prefixed UTF8 master,salt,dataset,panel,arm,index; first128bits',
                         sampling_algorithm='Python random.Random.sample sorted frame, partial uniform permutation',
                         python_version=platform.python_version(),
                         prospective_amendments=['graph identity bound separately; shared R/S seeds retained across curator sensitivities',
                             'A blocker prefix sorted by ID; random all-record padding including A-excluded-blockers',
                             'U initial horizon clips to excluded frame; full record service universe retained',
                             'zero feasible horizons have explicit status and zero releases',
                             'wall-clock A search charged separately, outside deterministic manifest seal'])
    result = dict(schema=SCHEMA, evidence_role=evidence_role, dataset_id=dataset_id, panel_id=panel_id,
                  graph_binding=binding, configuration=configuration,
                  configuration_sha256=digest(configuration), input_design_sha256=digest(design),
                  code_sha256=_code_hashes(), source_kinds=dict(sorted(source_kinds.items())),
                  source_kind_declaration_sha256=digest(source_kinds),
                  source_provenance_verified_by_this_module=False,
                  source_service_universe=source_ids, genuine_source_sampling_frame=genuine,
                  source_size_quantiles=_quantiles(size.values()),
                  genuine_source_size_quantiles=_quantiles(size[s] for s in genuine),
                  source_unknown_records=sum(size[s] for s in source_ids if source_kinds[s] == 'unknown_singleton'),
                  engineering_proxy_records=sum(size[s] for s in source_ids if source_kinds[s] == 'engineering_proxy'),
                  initial_selected_records=len(ids)-len(excluded), initially_excluded_records=len(excluded),
                  trajectories=trajectories, arm_summaries=summaries,
                  confirmatory_study_ready=False, genuine_source_withdrawal_evidence=False,
                  model_results_used_for_generation=False, labels_used_for_generation=False,
                  graph_diagnostics_for_isolated_auditor_only=True,
                  inference_scope='simulated requests conditional on one fixed graph; A is graph stress, never prevalence',
                  empty_or_failed_initializations_preserved=True)
    result['manifest_sha256'] = digest(result)
    return result


def verify_manifest(manifest, graph=None):
    """Verify content seal and optionally exact bound graph identity.

    Integrity is not external provenance authentication or permission to run a
    confirmatory study. The declared code hashes bind historical generation.
    """
    if not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported request manifest schema')
    unsigned = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if digest(unsigned) != manifest.get('manifest_sha256'):
        raise ValueError('request manifest seal mismatch')
    if digest(manifest['configuration']) != manifest['configuration_sha256']:
        raise ValueError('request configuration seal mismatch')
    if graph is not None and graph_binding(graph) != manifest['graph_binding']:
        raise ValueError('request manifest belongs to a different graph/source partition')
    return True


def write_manifest(manifest, path):
    """Create once; never overwrite a frozen request manifest."""
    verify_manifest(manifest)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write('\n')
    return path
