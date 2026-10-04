"""Prespecified structure-only request extensions; frozen core stays unchanged."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import random
import time

from phase4.requests import (generate_manifest, verify_manifest, derive_seed, digest,
                             _graph_state, _path, _levels, _integer)

SCHEMA = 'ccu-structure-request-extensions-1'


def weighted_source_prefix(source_ids, source_sizes, horizon, seed):
    """Sequential PPS without replacement, exact integer cumulative mass draws.

    Conditional on previously selected sources, each remaining source has
    probability original_record_count / remaining_original_record_count.
    This is not uniform source sampling and is not PPS inclusion probability
    proportional to mass for every k; later inclusion probabilities are complex.
    """
    _integer(horizon, 'horizon'); _integer(seed, 'seed')
    sources = sorted(source_ids)
    if len(set(sources)) != len(sources) or set(source_sizes) != set(sources):
        raise ValueError('unique source IDs and exactly aligned mass weights required')
    if any(not isinstance(s, str) or not s for s in sources):
        raise ValueError('nonempty stable source IDs required')
    for size in source_sizes.values(): _integer(size, 'source record mass', positive=True)
    if horizon > len(sources): raise ValueError('source horizon exceeds sampling frame')
    rng = random.Random(seed); remaining = list(sources); order = []; draws = []
    total = sum(source_sizes.values())
    for _ in range(horizon):
        draw = rng.randrange(total); cumulative = 0
        for index, source in enumerate(remaining):
            cumulative += source_sizes[source]
            if draw < cumulative: break
        weight = source_sizes[source]
        draws.append(dict(source_id=source, original_record_mass=weight,
                          remaining_total_mass_before_draw=total, integer_draw=draw,
                          conditional_probability_numerator=weight,
                          conditional_probability_denominator=total))
        order.append(source); remaining.pop(index); total -= weight
    return order, draws


def generate_structure_extensions(graph, *, dataset_id, panel_id, source_kinds,
                                  design=None, master_seed=20271003,
                                  mass_source_paths=256, source_horizon=8,
                                  one_percent_allocations=None,
                                  evidence_role='engineering_nonconfirmatory',
                                  timing_sink=None):
    """Independent structure-only branches, with no semantic/provenance pass.

    The original source universe is untouched. Missing genuine sources remain
    explicit zero-release status paths. The one-percent branch is a separate
    service state/contract, not an extension after the core horizon is exhausted.
    """
    _integer(mass_source_paths, 'mass_source_paths'); _integer(source_horizon, 'source_horizon')
    _integer(master_seed, 'master_seed')
    original_design = {} if design is None else deepcopy(design)
    # Use the frozen core validation without generating any core trajectories.
    base = generate_manifest(graph, dataset_id=dataset_id, panel_id=panel_id,
                             source_kinds=source_kinds, design=original_design,
                             allocations=dict(R=0,S=0,U=0,A=0), master_seed=master_seed,
                             evidence_role=evidence_role, include_excluded_blocker_stress=False)
    ids, owners, blockers, binding = _graph_state(graph)
    sizes = Counter(owners.values()); genuine = base['genuine_source_sampling_frame']
    mass_weights = {s:sizes[s] for s in genuine}; limit = min(source_horizon, len(genuine))
    source_levels = _levels(original_design.get('requests', {}).get('source_checkpoints', [1,2,4,8]), limit)
    base_salt = original_design.get('seeds', {}).get('request_salt', 'ccu-v1-request')
    mass_salt = base_salt + '-mass-source-v1'; mass_paths = []
    for index in range(mass_source_paths):
        seed = derive_seed(master_seed, mass_salt, dataset_id, panel_id, 'S-mass', f'{index:03d}')
        started = time.perf_counter()
        order, draws = weighted_source_prefix(genuine, mass_weights, limit, seed)
        if timing_sink is not None:
            timing_sink.append(dict(branch='mass_source', trajectory_id=f'S-mass{index:03d}',
                                    request_sampling_seconds=time.perf_counter()-started))
        path = _path('S-mass', index, seed, 'source', source_horizon, genuine,
                     order, source_levels, ids, owners, blockers,
                     dict(branch='structure_only_mass_proportional_source',
                          sampling_law='sequential_PPS_without_replacement_by_initial_post_guard_record_count',
                          source_mass_draws=draws,
                          uniform_source_admission_formula_applicable=False,
                          pool_with_uniform_source_results=False))
        mass_paths.append(path)

    # ceil(N/100), exactly, including N=0; no floating-point rounding.
    percent_horizon = (len(ids)+99)//100
    percent_checkpoints = sorted(set([1,8,32,128] + ([percent_horizon] if percent_horizon else [])))
    one_design = deepcopy(original_design)
    one_design.setdefault('seeds', {})['request_salt'] = base_salt + '-one-percent-record-v1'
    one_design.setdefault('requests', {}).update(record_horizon=percent_horizon,
                                                record_checkpoints=percent_checkpoints)
    if one_percent_allocations is None: one_percent_allocations = dict(R=256,S=0,U=128,A=32)
    else: one_percent_allocations = dict(one_percent_allocations)
    if one_percent_allocations.get('S', 0) != 0:
        raise ValueError('one-percent record branch cannot contain source-service paths')
    one_timing = [] if timing_sink is not None else None
    one = generate_manifest(graph, dataset_id=dataset_id, panel_id=panel_id,
                            source_kinds=source_kinds, design=one_design,
                            allocations=one_percent_allocations, master_seed=master_seed,
                            evidence_role=evidence_role, include_excluded_blocker_stress=False,
                            timing_sink=one_timing)
    if timing_sink is not None:
        timing_sink.extend(dict(branch='one_percent_record', **t) for t in one_timing)
    summaries = []
    for checkpoint in source_levels:
        observations = [o for p in mass_paths for o in p['observations'] if o['checkpoint'] == checkpoint]
        summaries.append(dict(checkpoint=checkpoint, trajectories=len(observations),
                              distinct_request_sets=len({o['cumulative_request_set_sha256'] for o in observations}),
                              zero_admission_trajectories=sum(o['zero_admission'] for o in observations),
                              empty_retained_targets=sum(o['empty_retained_target'] for o in observations)))
    code_hashes = dict(base['code_sha256'])
    code_hashes['phase4/requests_extensions.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = dict(master_seed=master_seed, mass_source_paths=mass_source_paths,
                  mass_source_requested_horizon=source_horizon, mass_source_salt=mass_salt,
                  one_percent_allocations=one_percent_allocations,
                  one_percent_record_horizon=percent_horizon,
                  one_percent_requested_checkpoints=percent_checkpoints,
                  prospective_choices=[
                      'mass sampler sequential PPS without replacement by initial retained panel record count',
                      'one-percent branch independent RNG namespace and separate state; no reset after core exhaustion',
                      'one-percent checkpoints clip default 1/8/32/128 and append final ceil(N/100)',
                      'same-branch R/S-mass RNG keys omit edges to preserve paired curator sensitivities',
                      'no uniform-source analytic admission formula applied to mass-weighted samples'])
    result = dict(schema=SCHEMA, dataset_id=dataset_id, panel_id=panel_id,
                  evidence_role=evidence_role, graph_binding=binding,
                  original_design_sha256=digest(original_design), configuration=config,
                  configuration_sha256=digest(config), code_sha256=code_hashes,
                  source_kinds=base['source_kinds'],
                  source_service_universe=base['source_service_universe'],
                  genuine_source_sampling_frame=genuine,
                  source_provenance_verified_by_this_module=False,
                  source_mass_weights=mass_weights,
                  mass_source_trajectories=mass_paths,
                  mass_source_status_counts=dict(Counter(p['status'] for p in mass_paths)),
                  mass_source_distinct_ordered_paths=len({tuple(p['deletion_order']) for p in mass_paths}),
                  mass_source_checkpoint_summary=summaries,
                  one_percent_record_manifest=one,
                  one_percent_service_requires_separate_initial_state=True,
                  branches_have_separate_distributions_and_must_not_be_pooled=True,
                  source_mass_inference_scope='conditional simulated mass-biased source workload; not uniform-source prevalence',
                  graph_diagnostics_for_isolated_auditor_only=True,
                  confirmatory_study_ready=False, labels_used_for_generation=False,
                  model_results_used_for_generation=False)
    result['manifest_sha256'] = digest(result)
    return result


def verify_structure_extensions(manifest, graph=None):
    if not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported structure extension schema')
    unsigned = {k:v for k,v in manifest.items() if k != 'manifest_sha256'}
    if digest(unsigned) != manifest.get('manifest_sha256'):
        raise ValueError('structure extension seal mismatch')
    if digest(manifest['configuration']) != manifest['configuration_sha256']:
        raise ValueError('structure extension configuration mismatch')
    verify_manifest(manifest['one_percent_record_manifest'], graph)
    if graph is not None and _graph_state(graph)[3] != manifest['graph_binding']:
        raise ValueError('structure extension graph mismatch')
    return True
