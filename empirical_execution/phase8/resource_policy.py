"""Prospective audit-cap selection from accepted disjoint development receipts.

This workflow never runs a benchmark or creates an observation. Source/cache
acceptance may replay supplied real assets. No standalone receipt authorizes
primary execution. Observed count coverage is not a runtime or memory theorem.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from phase7 import resource_preflight as preflight

SCHEMA = 'ccu-observed-audit-resource-policy-1'
METHODS = {'O-G', 'O-T', 'B-E', 'B-A', 'P-I', 'P-S', 'P-R', 'B-F'}
SUMMARIES = {'P-I', 'P-S', 'P-R'}
SIGNATURE = ('method', 'unit', 'full_state', 'curator_dimension', 'learner_dimension',
             'outputs', 'solver', 'panel', 'lambda_reg', 'threshold', 'batch_rows',
             'compaction_fraction', 'encoder', 'encoder_id', 'encoder_revision', 'checkpoint_schedule')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be a positive finite observed number')
    return value


def source_bindings():
    files = ['phase8/resource_policy.py', 'phase7/resource_preflight.py', 'phase7/run_dispatch.py',
        'phase6/state_audit.py', 'phase6/acceptance.py', 'phase6/dispatch.py',
        'phase6/recipes.py', 'phase6/run_isolated.py', 'phase6/workers.py', 'phase6/methods.py']
    files += [str(path.relative_to(ROOT)) for folder in ('ccu','phase3','phase4','phase5','phase6')
              for path in sorted((ROOT/folder).glob('*.py'))]
    return {name: preflight.sha256((ROOT/name).read_bytes()) for name in sorted(set(files))}


def resource_input_fingerprint(bundle):
    """Exclude later attachments and testimony; keep the development replay ID.

    Exact source/cache replay, target profiles, observation plan/artifact hashes,
    and final selected policy have separate receipt bindings. No receipt hashes
    itself. Replacing preliminary testimony with the final review is allowed;
    changing its development replay ID or any other input is not.
    """
    from phase5 import task_program as task
    omitted = {'execution_policy', 'external_evidence_review',
               'resource_policy_qualification', 'resource_policy_observations'}
    value = {key: item for key,item in bundle.items() if key not in omitted}
    value['preliminary_development_source_replay_sha256'] = bundle['external_evidence_review']['development_source_replay_sha256']
    return digest(task.fingerprint(value))


def bound_json(reference, base):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('A path and exact SHA256 are required for each observation artifact')
    path = (Path(base)/reference['path']).resolve()
    if not path.is_relative_to(Path(base).resolve()) or not path.is_file():
        raise ValueError('Observation artifact is absent or outside the stable dossier root')
    raw = path.read_bytes()
    if preflight.sha256(raw) != reference['sha256']:
        raise ValueError('Observation artifact changed: ' + reference['path'])
    return preflight.loads(raw.decode()), path


def validate_profiles(profiles):
    if not isinstance(profiles, list) or not profiles:
        raise ValueError('The complete intended native audit-profile list is required')
    ids = []
    for profile in profiles:
        required = set(SIGNATURE) | {'profile_id', 'records', 'horizon', 'checkpoints'}
        if not isinstance(profile, dict) or set(profile) != required:
            raise ValueError('Audit profile must specify exactly the documented shape and execution fields')
        if not isinstance(profile['profile_id'], str) or not profile['profile_id']:
            raise ValueError('Nonempty profile ID required')
        ids.append(profile['profile_id'])
        if profile['method'] not in METHODS or profile['unit'] not in ('record', 'source'):
            raise ValueError('Unsupported native audit method or deletion unit')
        if type(profile['full_state']) is not bool:
            raise ValueError('Explicit boolean full_state required')
        if not profile['full_state'] and profile['method'] not in SUMMARIES | {'B-E'}:
            raise ValueError('This method has no mandatory light audit in the frozen dispatcher')
        if profile['encoder'] not in ('e5', 'mpnet'):
            raise ValueError('A native accepted encoder is required')
        expected_encoder = {'e5':'intfloat/multilingual-e5-base', 'mpnet':'sentence-transformers/all-mpnet-base-v2'}[profile['encoder']]
        if profile['encoder_id'] != expected_encoder or not isinstance(profile['encoder_revision'], str) or not profile['encoder_revision']:
            raise ValueError('Exact accepted native encoder identity and revision required')
        for name in ('records', 'horizon', 'checkpoints', 'curator_dimension', 'learner_dimension', 'outputs', 'batch_rows'):
            preflight.integer(profile[name], name, 1)
        if profile['horizon'] > profile['records'] or profile['checkpoints'] > profile['horizon']:
            raise ValueError('Invalid horizon/checkpoint count')
        schedule = profile['checkpoint_schedule']
        if not isinstance(schedule,list) or len(schedule)!=profile['checkpoints'] or any(type(v)is not int or v<1 for v in schedule) or schedule!=sorted(set(schedule)) or schedule[-1]!=profile['horizon']:
            raise ValueError('A complete strictly increasing checkpoint schedule ending at the horizon is required')
        if profile['solver'] not in ('cholesky', 'cg') or profile['panel'] not in ('cold_process', 'warm_service', 'every_release'):
            raise ValueError('Unsupported frozen solver or process panel')
        positive(profile['lambda_reg'], 'lambda_reg')
        for name, lower, upper in [('threshold', -1, 1), ('compaction_fraction', 0, 1)]:
            value = profile[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError('Invalid profile ' + name)
    if len(ids) != len(set(ids)):
        raise ValueError('Distinct profile IDs required')
    return profiles


def select_observed_envelope(profiles, observations, base_policy, *, defaults=None):
    """Pure count selection; caller must separately authenticate every observation.

    A SINGLE observation must jointly cover a profile's N, dimensions, method,
    horizon, and checkpoint count. We never combine maxima across observations
    to manufacture a larger joint workload. Unknown future graph state remains
    unqualified, even when these operational count thresholds are covered.
    """
    validate_profiles(profiles)
    preflight.validate_policy(base_policy)
    defaults = preflight.source_contract()['default_caps'] if defaults is None else defaults
    pair_cap = max(defaults[preflight.PAIR_CAP], max(p['records']*(p['records']-1)//2*p['curator_dimension'] for p in profiles))
    coverage = []
    selected_sets = []
    for target in profiles:
        witnesses = []
        for observed in observations:
            if any(observed.get(key) != target[key] for key in SIGNATURE):
                continue
            if observed['records'] < target['records'] or observed['horizon'] < target['horizon'] or observed['checkpoints'] < target['checkpoints']:
                continue
            if observed['pair_coordinates'] < pair_cap:
                continue
            if not observed['observed_worker_limits_fit']:
                continue
            witnesses.append(observed)
        if not witnesses:
            raise ValueError('No single accepted observation jointly covers profile ' + target['profile_id'])
        selected_sets.append((target, witnesses))
    full_sets = [(p, values) for p, values in selected_sets if p['full_state'] and p['method'] in SUMMARIES]
    if full_sets:
        coefficient_cap = min(max(o['maximum_fresh_symbol_coordinates'] for o in values) for _, values in full_sets)
        if coefficient_cap < defaults[preflight.COEFFICIENT_CAP]:
            raise ValueError('Observed full-state coverage does not reach the unchanged default coefficient cap')
    else:
        coefficient_cap = defaults[preflight.COEFFICIENT_CAP]
    for target, values in selected_sets:
        packed = target['learner_dimension']*(target['learner_dimension']+1)//2 + target['learner_dimension']*target['outputs'] + 1
        full = target['full_state'] and target['method'] in SUMMARIES
        qualifying = [o for o in values if not full or o['maximum_fresh_symbol_coordinates'] >= coefficient_cap]
        ceiling = coefficient_cap // packed if full else None
        coverage.append({'profile_id': target['profile_id'],
            'joint_observation_ids': sorted(o['observation_id'] for o in qualifying),
            'original_pair_demand': target['records']*(target['records']-1)//2*target['curator_dimension'],
            'future_exact_symbolic_keys': None,
            'symbolic_key_upper_bound': 2*target['records'] if full else None,
            'supported_symbolic_key_cap': ceiling,
            'coefficient_cap_sufficient_for_all_possible_keys': None if not full else 2*target['records']*packed <= coefficient_cap,
            'future_overflow_status': 'state_audit_failure_preserved_no_waiver',
            'runtime_or_RSS_bound': False})
    policy = {**base_policy, preflight.PAIR_CAP: pair_cap, preflight.COEFFICIENT_CAP: coefficient_cap,
              'policy_role': 'prospective_observed_audit_count_envelope_not_native_feasibility_guarantee'}
    preflight.validate_policy(policy)
    return {'execution_policy': policy, 'coverage': coverage,
        'selection_rule': 'Shared pair cap covers declared original shapes; shared coefficient cap is the minimum across full profiles of their maximum jointly covered observed coefficient demand.',
        'default_caps_not_lowered': True, 'convex_cap_changed': False,
        'extrapolation': 'none; dimensions, method and execution signature must match; declared row/horizon/checkpoint extents must be jointly observed',
        'unbounded_cost_factors': ['graph edges and source overlap', 'symbolic incidence and accumulation fan-in',
            'dictionary and rank-basis operations', 'snapshot sizes and serialization', 'conditioning and decoder iterations',
            'future host load, allocator behavior, and unobserved process peaks'],
        'qualification': 'Operational count-setting evidence only; no native runtime/RSS feasibility guarantee.'}


def _real_acceptance(group, candidate, base):
    from phase6.acceptance import accept_source_cache
    from phase6.dispatch import development_acceptance
    design = json.loads((ROOT.parent/'output/empirical_program/study_design.json').read_text())
    replay = accept_source_cache(group, candidate, design, base_dir=base)
    if replay.get('machine_acceptance_passed') is not True:
        raise ValueError('Actual source/cache acceptance failed: ' + '; '.join(replay.get('blockers', ['unaccepted inputs'])))
    development = development_acceptance(candidate, base, replay)
    return replay, development


def _input_shapes(bundle, base, development):
    import numpy as np
    directory = (Path(base)/bundle['development']['input_bundle_directory']).resolve()
    if not directory.is_relative_to(Path(base).resolve()):
        raise ValueError('Development input directory escaped the dossier root')
    required = ('metadata.json', 'graph.npz', 'curator.npy', 'learner.npy', 'targets.npy')
    hashes = {name: preflight.sha256((directory/name).read_bytes()) for name in required}
    if any(development['pilot_input_file_hashes'].get(name) != value for name, value in hashes.items()):
        raise ValueError('Development replay and observation input package differ')
    arrays = [np.load(directory/name, mmap_mode='r', allow_pickle=False) for name in ('curator.npy', 'learner.npy', 'targets.npy')]
    cx, x, y = arrays
    if any(a.ndim != 2 for a in arrays) or len({len(a) for a in arrays}) != 1:
        raise ValueError('Aligned accepted development arrays required')
    metadata = preflight.loads((directory/'metadata.json').read_text())
    with np.load(directory/'graph.npz', allow_pickle=False) as graph:
        edges = int(graph['indices'].size)
        max_degree = int(np.max(np.diff(graph['indptr']), initial=0))
    encoder = bundle['development'].get('encoder', 'e5')
    source_bundle = bundle['development'].get('source_bundle', bundle)
    entry = source_bundle['curators'][encoder]
    learner = source_bundle['learners'][encoder]
    if any(entry[key] != learner[key] for key in ('encoder_id','encoder_revision')):
        raise ValueError('Mixed curator/learner encoders are unsupported by this native pilot policy')
    return {'records': len(x), 'curator_dimension': cx.shape[1], 'learner_dimension': x.shape[1],
        'outputs': y.shape[1], 'input_hashes': hashes, 'threshold': metadata['threshold'],
        'encoder': encoder, 'encoder_id':entry['encoder_id'], 'encoder_revision':entry['encoder_revision'],
        'record_ids': metadata['record_ids'], 'source_ids': metadata['source_ids'],
        'graph_edges': edges, 'maximum_blocker_degree': max_degree,
        'feature_density': float(np.count_nonzero(x)/x.size) if x.size else 0,
        'input_array_bytes': sum(int(a.nbytes) for a in arrays)}


def _validate_target_bindings(profiles, bundle, replay, group):
    import numpy as np
    from phase5 import task_program as task
    from phase6 import recipes
    registry, _ = recipes.build_registry()
    matches = [g for g in registry['groups'] if g['block']=='C_full_methods' and g['variant']=='native'
               and g['corpus']==group['corpus'] and g['panel']==group['panel']]
    if len(matches) != 1:
        raise ValueError('V1 qualifies only a complete registered native C_full_methods panel, not an arbitrary group')
    canonical = matches[0]
    if group != canonical:
        raise ValueError('V1 requires the exact canonical native C group; other groups need separate qualification')
    cfg = canonical['configuration']
    if cfg['curator_encoder'] != cfg['learner_encoder'] or cfg['state_precision'] != 'FP64':
        raise ValueError('Mixed encoder or secondary-precision resource profiles need separate qualification')
    units = ['record'] + (['source'] if replay['validated_bindings']['genuine_source_ids'] else [])
    required = {(method,unit,True) for method in canonical['methods'] for unit in units}
    required |= {(method,unit,False) for method in SUMMARIES | {'B-E'} for unit in units}
    actual = [(p['method'],p['unit'],p['full_state']) for p in profiles]
    if len(actual) != len(set(actual)) or set(actual) != required:
        raise ValueError('Complete canonical full and light native audit-method/unit coverage is required')
    for profile in profiles:
        encoder = profile['encoder']
        if encoder != cfg['curator_encoder'] or profile['learner_dimension'] != cfg['learner_dimension'] or profile['solver'] != 'cholesky' or profile['panel'] != 'cold_process' or profile['batch_rows'] != 1024 or profile['compaction_fraction'] != .5:
            raise ValueError('Profile differs from canonical native-core execution settings; other settings need separate qualification')
        for entry in (bundle['curators'][encoder],bundle['learners'][encoder]):
            if profile['encoder_id'] != entry['encoder_id'] or profile['encoder_revision'] != entry['encoder_revision']:
                raise ValueError('Profile encoder identity/revision differs from accepted primary caches')
        curator = np.asarray(bundle['curators'][encoder]['train'])
        learner = np.asarray(bundle['learners'][encoder]['train'])
        targets = np.asarray(bundle['train_y'])
        outputs = 1 if targets.ndim == 1 else targets.shape[1]
        if profile['records'] != len(replay['validated_bindings']['train_ids']) or profile['curator_dimension'] != curator.shape[1] or profile['learner_dimension'] != learner.shape[1] or profile['outputs'] != outputs:
            raise ValueError('Target audit shape differs from the actually accepted primary bundle')
        if profile['unit'] == 'source' and not replay['validated_bindings']['genuine_source_ids']:
            raise ValueError('A genuine accepted source frame is required for source audit profiles')
        threshold, _ = task._parameter(bundle['curators'][encoder], 'threshold', encoder)
        regularizer, _ = task._parameter(bundle['learners'][encoder], 'lambda', encoder)
        if profile['threshold'] != threshold or profile['lambda_reg'] != regularizer:
            raise ValueError('Target profile differs from actual calibrated threshold or regularization')
        configuration = bundle['requests']['configuration']['requests']
        if profile['horizon'] != configuration[profile['unit']+'_horizon'] or profile['checkpoint_schedule'] != configuration[profile['unit']+'_checkpoints']:
            raise ValueError('Target profile does not cover its complete frozen deletion horizon/checkpoints')
    return {'canonical_group_id':canonical['group_id'], 'canonical_group_sha256':digest(canonical),
            'covered_profiles':len(profiles), 'unavailable_source_profiles':not bool(replay['validated_bindings']['genuine_source_ids']),
            'scope':'Native cold-process C core only; other registry groups or settings are not qualified.'}


def _read_observations(manifest, bundle, base, development, shapes, policy, codes, profiles):
    if not isinstance(manifest, dict) or set(manifest) != {'schema', 'plan', 'entries'} or manifest['schema'] != 'ccu-native-audit-observations-1':
        raise ValueError('A bound complete prospective audit-observation manifest is required')
    plan, _ = bound_json(manifest['plan'], base)
    if plan.get('evidence_role') != 'actual_disjoint_native_development':
        raise ValueError('Software or lexical observations cannot qualify native resource policy')
    if plan.get('development_source_replay_sha256') != development['development_source_replay_sha256']:
        raise ValueError('Observation plan belongs to different accepted development sources')
    if plan.get('target_profiles_sha256') != digest(profiles):
        raise ValueError('Prospective observation plan does not bind the complete intended profile list')
    if plan.get('resource_snapshot_sha256') != digest(bundle['development']['resource_snapshot']) or plan.get('development_resource_lock_sha256') != digest(bundle['development']['resource_lock']):
        raise ValueError('Prospective observation plan does not bind its development resource observations and lock')
    entries = manifest['entries']
    ids = [item['observation_id'] for item in entries]
    if not ids or ids != plan.get('observation_ids') or len(ids) != len(set(ids)):
        raise ValueError('Every planned observation, including failed attempts, must remain present in order')
    expected_services = {(item['path'], item['sha256']) for item in bundle['development']['service_reports']}
    observed = []
    current_worker_codes = {str(path.relative_to(ROOT)):preflight.sha256(path.read_bytes())
        for folder in ('ccu','phase3','phase4','phase5','phase6') for path in sorted((ROOT/folder).glob('*.py'))}
    for entry in entries:
        if set(entry) != {'observation_id', 'audit_report', 'audit_service_report', 'reference_service_report', 'audit_configuration'}:
            raise ValueError('Complete bound audit/reference/configuration references are required')
        audit, _ = bound_json(entry['audit_report'], base)
        service, _ = bound_json(entry['audit_service_report'], base)
        reference, _ = bound_json(entry['reference_service_report'], base)
        config, _ = bound_json(entry['audit_configuration'], base)
        if (entry['reference_service_report']['path'], entry['reference_service_report']['sha256']) not in expected_services:
            raise ValueError('Audit reference service is absent from accepted development observations')
        if audit.get('passed') is not True or audit.get('status') != 'passed' or not service.get('success') or not reference.get('success'):
            raise ValueError('Failed development observation retained; no policy exported from this plan')
        if audit.get('sha256') != digest({k:v for k,v in audit.items() if k != 'sha256'}):
            raise ValueError('Audit report digest differs')
        if audit.get('code_sha256') != codes['phase6/state_audit.py']:
            raise ValueError('Audit observations require the unchanged current state-audit source')
        if audit.get('audit_service_report_sha256') != entry['audit_service_report']['sha256'] or audit.get('measured_service_report_sha256') != entry['reference_service_report']['sha256']:
            raise ValueError('Audit report does not bind the supplied actual worker reports')
        if audit.get('method') != config.get('method') or type(audit.get('full_state_requested')) is not bool:
            raise ValueError('Audit method or full-state scope was retagged')
        relative_config = str(Path(entry['audit_configuration']['path']).relative_to(Path(entry['audit_service_report']['path']).parent))
        if service.get('artifact_hashes', {}).get(relative_config) != entry['audit_configuration']['sha256']:
            raise ValueError('Actual audit worker does not bind its supplied prospective configuration')
        for report in (service, reference):
            hashes = report.get('raw_input_files_sha256', {})
            if any(shapes['input_hashes'].get(name) != value for name,value in hashes.items()) or not {'metadata.json','graph.npz','learner.npy','targets.npy'} <= set(hashes):
                raise ValueError('Observed worker arrays differ from accepted development inputs')
            for name in ('phase6/methods.py','phase6/run_isolated.py','phase6/workers.py'):
                if report.get('project_python_source_hashes', {}).get(name) != codes[name]:
                    raise ValueError('Observed worker source differs: ' + name)
            if report['project_python_source_hashes'] != current_worker_codes:
                raise ValueError('The complete observed Python dependency map differs or is incomplete')
        if config.get('code_hashes') != current_worker_codes:
            raise ValueError('Prospective observation code map differs from actual worker/current sources')
        for key in ('method','unit','solver','panel'):
            if service.get(key) != config.get(key) or reference.get(key) != config.get(key):
                raise ValueError('Audit and original service execution signatures differ')
        if service.get('persistence') != 'every_release' or config.get('persistence') != 'every_release':
            raise ValueError('Every-release durable state observations required')
        if config.get('input_hashes') != service['raw_input_files_sha256']:
            raise ValueError('Prospective audit input hashes differ from actual worker inputs')
        if config.get('policy') != service.get('policy') or service.get('policy') != reference.get('policy'):
            raise ValueError('Audit and reference services used different prospective resource policies')
        if config['policy'].get('threads') != policy['threads']:
            raise ValueError('Observed and proposed worker thread counts differ')
        rows = audit.get('checkpoints', [])
        requests = config.get('requests', [])
        request_sha = hashlib.sha256(json.dumps(requests,separators=(',',':')).encode()).hexdigest()
        if len(rows) != len(requests) or not rows or request_sha != service.get('request_sha256') or request_sha != reference.get('request_sha256'):
            raise ValueError('Actual observation request/checkpoint coverage differs')
        cumulative=set(); schedule=[]
        for index,(batch,row) in enumerate(zip(requests,rows)):
            if not isinstance(batch,list) or any(not isinstance(unit,str) or not unit for unit in batch):
                raise ValueError('Observed deletion requests require actual unit IDs')
            cumulative.update(batch); schedule.append(len(cumulative))
            if row.get('index')!=index or row.get('deleted_unique_units')!=len(cumulative) or row.get('remaining_horizon')!=config['horizon']-len(cumulative):
                raise ValueError('Observed checkpoint does not bind actual distinct deletion counts and remaining horizon')
        if not schedule or schedule[-1]!=config['horizon'] or schedule!=sorted(set(schedule)):
            raise ValueError('Observation must execute the complete distinct-unit horizon, not only configure it')
        if config['unit'] == 'source':
            native_sources = {record['source_unit_id'] for record in bundle['development']['records']
                              if record.get('source_kind', '').startswith('native_')}
            if any(unit not in native_sources for batch in requests for unit in batch):
                raise ValueError('Development source requests contain nonnative or unknown source units')
        if any(row.get('passed') is not True or row.get('reconstructed_actual_snapshot_bound') is not True or row.get('replay_and_measured_heads_bitwise_equal') is not True for row in rows):
            raise ValueError('Every observed checkpoint must retain actual snapshot and head bindings')
        prospective = audit['prospective_audit_policy']
        expected_policy = {'absolute_tolerance':config['policy'].get('state_audit_absolute_tolerance',1e-11),
            'relative_tolerance':config['policy'].get('state_audit_relative_tolerance',1e-10),
            'max_pair_coordinates':config['policy'].get(preflight.PAIR_CAP,100_000_000),
            'max_coefficient_coordinates':config['policy'].get(preflight.COEFFICIENT_CAP,4_000_000)}
        if prospective != expected_policy:
            raise ValueError('Audit caps/tolerances differ from actual worker policy')
        for name, maximum in [('absolute_tolerance', 1e-11), ('relative_tolerance', 1e-10)]:
            value = prospective[name]
            if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError('Development audit weakened or invalidated a frozen numerical tolerance')
        packed = shapes['learner_dimension']*(shapes['learner_dimension']+1)//2 + shapes['learner_dimension']*shapes['outputs'] + 1
        fresh_counts = []
        for row in rows:
            if config['method'] in SUMMARIES and audit['full_state_requested']:
                coefficients = row['coefficients']
                if coefficients.get('passed') is not True or coefficients.get('complete_union_of_keys_compared') is not True:
                    raise ValueError('Complete designated coefficient checks are required')
                fresh_counts.append(sum(item['expected_designated_key'] is True for item in coefficients['rows']))
        total_time = positive(audit['charged_total_audit_seconds'], 'actual total audit seconds')
        parent_rss = preflight.integer(audit['auditor_process_lifetime_peak_rss_bytes'], 'auditor peak RSS', 1)
        stage_times, stage_rss, stage_cpu = [], [], []
        for stage in ('construction','repair'):
            measurement = service[stage]
            if measurement.get('exit_code') != 0 or measurement.get('timeout') is not False:
                raise ValueError('Incomplete observed audit worker stage')
            stage_times.append(positive(measurement['spawn_to_reap_seconds'], 'actual worker stage seconds'))
            cpu_values = [measurement['user_cpu_seconds'], measurement['system_cpu_seconds']]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in cpu_values):
                raise ValueError('Actual nonnegative worker CPU measurements required')
            stage_cpu.append(sum(cpu_values))
            stage_rss.append(max(preflight.integer(measurement['peak_rss_bytes'], 'wait4 RSS', 1),
                preflight.integer(measurement['sampled_process_tree_peak_rss_sum_bytes'], 'sampled worker RSS', 1)))
        indicator = parent_rss + max(stage_rss)
        values = {key: shapes[key] for key in ('records','curator_dimension','learner_dimension','outputs','threshold','encoder','encoder_id','encoder_revision')}
        values.update({key:config[key] for key in ('method','unit','solver','panel','horizon','lambda_reg','batch_rows','compaction_fraction')})
        values.update(observation_id=entry['observation_id'], full_state=audit['full_state_requested'], checkpoints=len(rows),
            checkpoint_schedule=schedule,
            pair_coordinates=shapes['records']*(shapes['records']-1)//2*shapes['curator_dimension'],
            maximum_fresh_symbolic_keys=max(fresh_counts, default=0),
            maximum_fresh_symbol_coordinates=max(fresh_counts, default=0)*packed,
            observed_worker_limits_fit=max(stage_times)<=policy['wall_seconds'] and max(stage_cpu)<=policy['cpu_seconds'] and indicator<=policy['memory_bytes'],
            total_audit_seconds=total_time, worker_stage_seconds=stage_times, worker_stage_cpu_seconds=stage_cpu,
            charged_worker_lifecycle_seconds=positive(audit['charged_worker_lifecycle_seconds'],'worker lifecycle seconds'),
            checkpoint_audit_seconds=[positive(row['audit_seconds'],'checkpoint audit seconds') for row in rows],
            parent_lifetime_peak_rss_bytes=parent_rss, observed_worker_rss_indicators=stage_rss,
            parent_plus_worker_rss_indicator=indicator, indicator_is_true_peak_bound=False,
            final_snapshot_bytes=preflight.integer(audit['audit_service_snapshot_bytes'],'final snapshot bytes'),
            graph_edges=shapes['graph_edges'], maximum_blocker_degree=shapes['maximum_blocker_degree'],
            feature_density=shapes['feature_density'], input_array_bytes=shapes['input_array_bytes'],
            bound_artifacts={key:entry[key] for key in entry if key!='observation_id'})
        observed.append(values)
    return observed, plan


def qualify_resource_policy(group, bundle, development_audits, target_profiles, *, base_dir='.'):
    """Recompute real acceptance and select a count policy, or return blocked.

    Preliminary external development replay binding must already be supplied.
    The final complete external review is deliberately deferred to the dossier
    assembler and unchanged dispatcher, after the candidate policy is known.
    """
    codes = source_bindings()
    result = {'schema': SCHEMA, 'status': 'blocked', 'execution_allowed': False,
        'execution_policy': None, 'source_sha256': codes, 'blockers': [],
        'actual_native_observations_accepted': 0, 'native_runtime_RSS_guarantee': False,
        'complete_primary_acceptance': False, 'final_external_review_still_required': True,
        'unknown_costs': ['graph rescore/setup breakdown', 'symbolic incidence work',
            'serialization time breakdown', 'true simultaneous process-tree peak', 'future native workload runtime and RSS']}
    stage = 'prospective_inputs'
    try:
        validate_profiles(target_profiles)
        if not isinstance(bundle.get('development'), dict):
            raise ValueError('Complete genuine disjoint-development dossier is absent')
        preliminary = bundle.get('external_evidence_review', {})
        if not isinstance(preliminary.get('development_source_replay_sha256'), str) or len(preliminary['development_source_replay_sha256']) != 64:
            raise ValueError('Externally supplied preliminary development replay binding is absent')
        from phase6 import recipes
        dossier = bundle['development']
        derived = recipes.derive_resource_policy(dossier['resource_snapshot'], dossier['stage_seconds'])
        if derived != dossier.get('resource_lock'):
            raise ValueError('Existing development resource lock does not recompute')
        base_policy = {'memory_bytes':derived['per_method_address_space_limit_bytes'],
            'cpu_seconds':derived['cpu_seconds_each_stage'], 'wall_seconds':derived['wall_seconds_each_stage'],
            'threads':derived['cpu_threads']}
        existing = bundle.get('execution_policy',{})
        for name,default in [('state_audit_absolute_tolerance',1e-11),('state_audit_relative_tolerance',1e-10)]:
            if name in existing:
                base_policy[name]=existing[name]
        if existing.get(preflight.CONVEX_CAP,2_000_000)!=2_000_000 or existing.get('convex_verifier','fraction_reference')!='fraction_reference':
            raise ValueError('Audit-only v1 cannot replace or qualify existing nondefault convex policy; use a separate bound convex workflow')
        preflight.validate_policy(base_policy)
        candidate = dict(bundle)
        candidate['execution_policy'] = base_policy
        stage = 'actual_source_and_disjoint_development_replay'
        replay, development = _real_acceptance(group, candidate, base_dir)
        result['source_replay_binding_sha256'] = replay['replay_binding_sha256']
        result['development_acceptance'] = development
        stage = 'native_target_and_observation_binding'
        target_scope = _validate_target_bindings(target_profiles, candidate, replay, group)
        shapes = _input_shapes(candidate, base_dir, development)
        observations, plan = _read_observations(development_audits, candidate, base_dir,
                                               development, shapes, base_policy, codes, target_profiles)
        stage = 'joint_observed_count_coverage'
        selected = select_observed_envelope(target_profiles, observations, base_policy)
        # Recheck the actual frozen development gate against the final candidate.
        from phase6.dispatch import development_acceptance
        candidate['execution_policy'] = selected['execution_policy']
        final_development = development_acceptance(candidate, base_dir, replay)
        if final_development != development:
            raise ValueError('Final policy altered the recomputed development acceptance')
        if source_bindings() != codes:
            raise ValueError('Qualification sources changed during actual input replay')
        result.update(status='qualified_observed_count_policy_pending_external_review',
            execution_policy=selected['execution_policy'], policy_recipe=selected,
            observations=observations, actual_native_observations_accepted=len(observations),
            target_profiles=target_profiles, target_profiles_sha256=digest(target_profiles),
            target_scope=target_scope,
            observation_manifest_sha256=digest(development_audits), observation_plan_sha256=digest(plan),
            development_resource_lock=derived, resource_input_fingerprint_sha256=resource_input_fingerprint(candidate),
            resource_fingerprint_scope='All bundle inputs except derived policy/qualification/observation containers and final review testimony; preliminary development replay ID remains bound.',
            preliminary_development_source_replay_sha256=preliminary['development_source_replay_sha256'],
            preliminary_review_scope='Caller-supplied development replay binding; does not replace the later complete external review.',
            final_phase6_development_acceptance_recomputed=True)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as error:
        result['blockers'].append({'stage':stage, 'exception_type':type(error).__name__, 'reason':str(error)})
    result['sha256'] = digest(result)
    return result


def verify_resource_receipt(receipt, group, bundle, development_audits, target_profiles, *, base_dir='.'):
    """A seal alone is insufficient: repeat current input acceptance and selection."""
    if receipt.get('execution_policy') is None or receipt.get('status') != 'qualified_observed_count_policy_pending_external_review':
        raise ValueError('A blocked resource receipt cannot supply an execution policy')
    actual = qualify_resource_policy(group, bundle, development_audits, target_profiles, base_dir=base_dir)
    if actual != receipt:
        raise ValueError('Resource receipt does not recompute from current accepted observations')
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New directory; blocked results contain no policy file.')
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.exit(2, 'Preserve existing result directory: ' + str(args.out) + '\n')
    try:
        raw = args.request.read_bytes()
        request = preflight.loads(raw.decode())
        if set(request) != {'group','bundle','development_audits','target_profiles'}:
            raise ValueError('Request requires group, bundle, development_audits, and target_profiles')
        from phase5.task_program import load_bound_value
        bundle = load_bound_value(request['bundle'], args.request.parent.resolve())
        result = qualify_resource_policy(request['group'], bundle, request['development_audits'],
            request['target_profiles'], base_dir=args.request.parent.resolve())
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out/'qualification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        invocation = {'request_sha256':preflight.sha256(raw), 'source_sha256':result['source_sha256'],
                      'request_base':str(args.request.parent.resolve()), 'qualification_sha256':result['sha256']}
        (args.out/'invocation.json').write_text(json.dumps(invocation, indent=2, allow_nan=False)+'\n')
        if result['execution_policy'] is not None:
            (args.out/'execution_policy.json').write_text(json.dumps(result['execution_policy'], indent=2, allow_nan=False)+'\n')
        print(json.dumps({'status':result['status'], 'policy_exported':result['execution_policy'] is not None,
                          'primary_execution_allowed':False, 'out':str(args.out)}))
        return result
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, 'BLOCKED: '+str(error)+'\n')


if __name__ == '__main__':
    main()
