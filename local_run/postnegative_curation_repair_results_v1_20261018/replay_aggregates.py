#!/usr/bin/env python3
"""Stage2 portable aggregate replay; Python 3.9+ standard library only.

No private input, source checkout, network, model or GPU is accessed.
Equality flags remain saved execution receipts, not local state/graph replay.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parent
GRID = [Fraction(i, 100) for i in range(50, 100)] + [Fraction(199, 200)]
METER_KEYS = {'initial_keys', 'initial_incidences', 'initial_quadratic_potential',
    'input_identifiers', 'fresh_deletions', 'key_shrink_events', 'tuple_cells_copied',
    'coefficient_collisions', 'scalar_additions', 'cancellations',
    'horizon_pruned_keys', 'inverse_incidence_removals',
    'full_scan_keys_for_same_atomic_sequence'}
FLAGS = ('all_selected_membership_coordinates_equal',
    'all_coefficient_and_index_state_equal', 'sequential_batch_reverse_equal',
    'fresh_retained_graph_equal', 'retry_state_unchanged')
ARM_SCOPE = ('Exact integer membership statistics and canonical logical state; '
             'no learner, task labels, physical erasure or comparative speed claim.')
COORDINATE = ('e_v is the deterministic original-panel record membership '
              'coordinate, not an annotation or supervised target.')

def need(value, code):
    if not value:
        raise AssertionError(code)

def integer(value, minimum=0, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)

def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0

def binding(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    def bad_constant(value):
        raise AssertionError('nonfinite_json_constant')
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=bad_constant)

def independent_seed(master, namespace, arm):
    # Independently reconstruct the documented length-delimited SHA-256 input.
    fields = [str(master).encode('utf-8'), namespace.encode('utf-8'), arm.encode('utf-8')]
    packed = b''.join(len(field).to_bytes(8, 'big') + field for field in fields)
    return int.from_bytes(hashlib.sha256(packed).digest()[:16], 'big')

def check_meter(arm, dimension):
    meter = arm['meter']; points = arm['checkpoints']; horizon = arm['effective_horizon']
    need(set(meter) == METER_KEYS, 'meter_fields')
    need(all(integer(value) for value in meter.values()), 'meter_nonnegative_integers')
    keys = meter['initial_keys']; incidences = meter['initial_incidences']
    potential = meter['initial_quadratic_potential']
    need(keys == arm['initial_keys'] and incidences == arm['initial_incidences'], 'meter_initial_bindings')
    need(0 <= keys <= 2 * dimension, 'membership_initial_key_bound')
    need(incidences <= horizon * keys, 'initial_incidence_bound')
    need(incidences <= potential and 2 * potential <= (horizon + 1) * incidences, 'initial_potential_bounds')
    need(meter['fresh_deletions'] == horizon == arm['singleton_checks'], 'singleton_fresh_deletion_count')
    need(meter['input_identifiers'] == horizon + len(points), 'retry_input_identifier_count')
    need(meter['key_shrink_events'] <= incidences, 'shrink_incidence_bound')
    need(meter['inverse_incidence_removals'] == incidences, 'terminal_inverse_incidence_conservation')
    need(meter['coefficient_collisions'] <= min(keys, meter['key_shrink_events']), 'collision_bounds')
    need(meter['scalar_additions'] == dimension * meter['coefficient_collisions'], 'scalar_addition_identity')
    need(meter['cancellations'] <= meter['coefficient_collisions'], 'cancellation_bound')
    need(meter['key_shrink_events'] <= meter['tuple_cells_copied'] <= potential, 'tuple_copy_bounds')
    final_keys = points[-1]['current_coefficient_keys']
    need(final_keys == keys - meter['coefficient_collisions'] - meter['cancellations'] - meter['horizon_pruned_keys'], 'key_conservation')
    need(final_keys == int(points[-1]['selected_records'] > 0), 'zero_horizon_constant_key_count')
    low = high = 0; previous_step = 0; previous_keys = keys
    for point in points:
        step = point['checkpoint']; current_keys = point['current_coefficient_keys']; width = step - previous_step
        need(width > 0 and 0 <= current_keys <= previous_keys, 'checkpoint_key_monotonicity')
        # For each checkpoint interval, the first pre-update count is known;
        # remaining pre-update counts lie between the two endpoint counts.
        low += previous_keys + (width - 1) * current_keys
        high += width * previous_keys
        previous_step, previous_keys = step, current_keys
    need(low <= meter['full_scan_keys_for_same_atomic_sequence'] <= high, 'full_scan_counter_bounds')
    return {'bounds_passed': True, 'full_scan_lower_bound': low, 'full_scan_upper_bound': high}

def main():
    need(sys.flags.optimize == 0, 'run_without_O')
    manifest=read(ROOT/'MANIFEST.json')
    need(manifest['schema']=='third-stage2-public-bundle-manifest-v1','manifest_schema')
    need(not any(p.is_symlink() for p in ROOT.rglob('*')),'no_symlinks')
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    need(actual==set(manifest['files'])|{'MANIFEST.json'},'exact_public_inventory')
    for name,pin in manifest['files'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts,'safe_manifest_path')
        need(binding(ROOT/name)==pin,'public_member_hash')
    run=ROOT/'run';done=read(run/'completion.json');intake=read(run/'intake.json');subset=read(run/'subset_oracle.json')
    plan=read(ROOT/'study_protocol.json');resources=read(ROOT/'resource_receipt.json');sources=read(ROOT/'SOURCE_BINDINGS.json')
    review=read(ROOT/'independent_review.json');reference=read(ROOT/'stage1_reference.json');note=read(ROOT/'output_field_interpretation.json')
    need(review['passed'] is True,'prior_independent_audit_passed')
    need(sources['executed_protocol']==review['bindings']['protocol']==resources['protocol_binding']==plan['public_transformation']['original_protocol_binding']==sources['executed_package_members']['PLAN.json'],'original_protocol_bindings')
    need(plan['public_transformation']['executed_document'] is False,'deidentified_protocol_view')
    need(sources['executed_package_manifest']==review['bindings']['package_manifest']==resources['package_manifest_binding'],'package_bindings')
    need(sources['frozen_research_sources']==plan['frozen_sources'] and sources['private_original_inputs']==plan['inputs'],'source_and_input_bindings')
    need(sources['helper_binding']==plan['helper_binding']==sources['executed_package_members']['stage1_common.py'],'helper_binding')
    need(note['source_binding']==review['bindings']['scientific_source']==sources['executed_package_members']['curation_repair.py'],'scientific_source_binding')
    need(note['frozen_protocol_binding']==sources['executed_protocol'],'field_note_protocol')
    need(binding(ROOT/'output_field_interpretation.json')==review['bindings']['output_field_interpretation'],'field_note_exact_bytes')
    need(sources['successful_schema_adapted_auditor']==review['bindings']['auditor'],'successful_auditor_binding')
    for key,name in (('completion','completion.json'),('intake','intake.json'),('subset_oracle','subset_oracle.json')):
        need(binding(run/name)==review['bindings'][key],'review_output_bindings')
    need({p.name for p in run.iterdir()}=={'completion.json','intake.json','subset_oracle.json'}|{'threshold_%02d.json'%i for i in range(51)},'complete_run_inventory')
    need(set(done) == {'status', 'threshold_indices', 'all51cuts_completed', 'model_or_human_annotation_calls',
         'task_learner_trained', 'v4_failure_retained', 'elapsed_seconds', 'peak_rss_bytes',
         'kernel_no_reaccess_isolation_claim', 'supervised_prediction_effect_claim', 'outputs'}, 'completion_fields')
    need(done['status'] == 'completed' and done['all51cuts_completed'] is True and done['threshold_indices'] == list(range(51)), 'all51_complete')
    need(done['model_or_human_annotation_calls'] == 0 and done['task_learner_trained'] is False, 'no_annotations_or_learner')
    need(done['v4_failure_retained'] is True and done['kernel_no_reaccess_isolation_claim'] is False and done['supervised_prediction_effect_claim'] is False, 'bounded_completion_claims')
    need(set(done['outputs']) == {'intake.json', 'subset_oracle.json'} | {'threshold_%02d.json' % i for i in range(51)}, 'completion_output_inventory')
    for name, value in done['outputs'].items():
        need(binding(run / name) == value, 'completion_output_binding')
    need(set(intake) == {'input_bindings', 'protocol_sha256', 'rows', 'labels_read', 'text_read', 'heldout_feature_slices_read', 'coordinate_definition', 'runtime'}, 'intake_fields')
    need(intake['protocol_sha256'] == sources['executed_protocol']['sha256'], 'intake_protocol_binding')
    need(intake['input_bindings'] == plan['inputs'], 'original_input_bindings')
    need(intake['runtime'] == plan['runtime'] == {'python':'3.13.7','numpy':'2.5.3','scipy':'1.17.1','platform':'Linux'}, 'runtime_pins')
    n = plan['input_rows']; need(n == intake['rows'] == 1001, 'panel_rows')
    need(intake['coordinate_definition'] == COORDINATE, 'membership_coordinate_definition')
    need(all(intake[k] is False for k in ('labels_read', 'text_read', 'heldout_feature_slices_read')), 'input_scope_flags')
    need(subset == reference['subset_oracle'], 'bounded_scalar_oracle_receipt')
    need(subset['passed'] is True and subset['blocker_set_comparisons'] == 816, 'scalar_oracle_pass')
    need(intake['input_bindings']==reference['original_inputs'] and intake['runtime']==reference['runtime'],'stage1_reference_shared_inputs')
    need(reference['stage1_public_manifest_binding']==review['bindings']['stage1_public_manifest'],'stage1_reference_binding')
    need(len(reference['thresholds'])==51 and [x['index'] for x in reference['thresholds']]==list(range(51)),'stage1_reference_complete')
    for i,cut in enumerate(GRID):
        need(reference['thresholds'][i]['threshold_fraction']==[cut.numerator,cut.denominator],'stage1_reference_cut')
    need(resources['process_returncode']==0 and resources['private_feature_or_label_output'] is False,'deidentified_process_scope')
    need(resources['runtime']==intake['runtime'],'resource_runtime')
    need(resources['stage_seconds']==done['elapsed_seconds'] and resources['stage_peak_rss_bytes']==done['peak_rss_bytes'],'resource_completion')
    need(0<resources['stage_seconds']<=resources['wrapper_seconds']<resources['wrapper_child_timeout_seconds'],'resource_elapsed')
    need(resources['stage_seconds']<resources['stage_wall_limit_seconds']==3600,'stage_wall_cap')
    need(0<resources['stage_peak_rss_bytes']<=resources['all_children_peak_rss_bytes']<resources['memory_requested_bytes']==8*1024**3,'resource_rss')
    need(resources['scratch_free_bytes_at_exit']>=resources['scratch_reserve_bytes']==8*1024**3,'scratch_reserve')
    need(resources['cpus_requested']==2 and resources['gpus']==0,'resource_allocation')
    rp = plan['request_protocol']
    need(rp['arms'] == ['R','S'] and rp['initial_horizon'] == 128 and rp['checkpoints'] == [1,8,32,128], 'fixed_request_plan')
    need(rp['master_seed'] == 20261018 and rp['namespace'] == 'third-postnegative-curation-repair-v1', 'seed_plan')
    need(rp['same_requests_across_all_cuts'] is True and rp['selection_or_stress_based_requests'] is False and rp['trajectories_per_arm'] == 1, 'no_outcome_request_selection')
    need(plan['thresholds'] == [[i,100] for i in range(50,100)] + [[199,200]], 'all51_fixed_threshold_plan')
    source_frames = {x['genuine_single_source_request_count'] for x in reference['thresholds']}
    need(len(source_frames) == 1, 'stage1_source_frame_consistency')
    frame_sizes = {'R': n, 'S': next(iter(source_frames))}
    digests = {}; profiles = {}; previous_selected = {}; source_universe = set(); summaries=[]
    unavailable=[]; arm_counts=Counter(); checkpoint_count=0; singleton_count=0; meter_checks=0; selected_monotonicity_checks=0
    timing_sum=0.0
    for i, cut in enumerate(GRID):
        result = read(run / ('threshold_%02d.json' % i))
        need(set(result) == {'index','threshold_fraction','threshold','arms','all_executed_checks_passed','semantic_validity_claim','primary_study_result'}, 'threshold_fields')
        need(result['index'] == i and result['threshold_fraction'] == [cut.numerator,cut.denominator] and result['threshold'] == float(cut), 'threshold_identity')
        need(result['all_executed_checks_passed'] is True and result['semantic_validity_claim'] is False and result['primary_study_result'] is False, 'threshold_scope_flags')
        need(set(result['arms']) == {'R','S'}, 'both_arms_present')
        first = reference['thresholds'][i]; initial_selected = first['initially_selected']
        cut_summary={'index':i,'threshold_fraction':[cut.numerator,cut.denominator],'arms':{}}
        for label in ('R','S'):
            arm=result['arms'][label]; need(arm['arm'] == label and arm['requested_horizon'] == 128, 'arm_identity')
            if arm['status'] == 'unavailable_empty_request_frame':
                need(set(arm) == {'arm','status','requested_horizon','effective_horizon','checkpoints'} and arm['effective_horizon'] == 0 and arm['checkpoints'] == [], 'explicit_unavailable_schema')
                unavailable.append({'index':i,'arm':label,'reason':'empty_request_frame'})
                need(frame_sizes[label] == 0, 'unavailable_contradicts_bound_nonempty_frame')
                cut_summary['arms'][label]={'status':arm['status']}; continue
            need(set(arm) == {'arm','unit','status','requested_horizon','effective_horizon','sampling_frame_size','request_seed','request_order_digest','singleton_checks','initial_keys','initial_incidences','construction_seconds','singleton_repair_seconds','meter','checkpoints','scope'}, 'completed_arm_fields')
            need(arm['status'] == 'completed' and arm['unit'] == ('record' if label=='R' else 'source'), 'arm_completed_unit')
            frame=frame_sizes[label]; horizon=min(128,frame); levels=sorted({min(k,horizon) for k in rp['checkpoints']})
            need(arm['sampling_frame_size'] == frame and arm['effective_horizon'] == horizon, 'frame_and_clipped_horizon')
            need(arm['request_seed'] == independent_seed(rp['master_seed'],rp['namespace'],label), 'independently_derived_seed')
            digest=arm['request_order_digest']; need(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest), 'request_digest_shape')
            need(label not in digests or digests[label] == digest, 'request_digest_changed_across_cuts');digests[label]=digest
            need(arm['scope'] == ARM_SCOPE and arm['singleton_checks'] == horizon, 'arm_scope_and_singletons')
            need(finite_positive(arm['construction_seconds']) and finite_positive(arm['singleton_repair_seconds']), 'arm_positive_timing')
            timing_sum += arm['construction_seconds'] + arm['singleton_repair_seconds']
            points=arm['checkpoints']; need([x['checkpoint'] for x in points] == levels, 'complete_checkpoints')
            profile=[]; previous_deleted=0
            for point in points:
                need(set(point) == {'checkpoint','remaining_horizon','deleted_records','selected_records','surviving_admissions','current_coefficient_keys','current_alive_units',*FLAGS,'fresh_oracle_seconds'}, 'checkpoint_fields')
                step=point['checkpoint'];deleted=point['deleted_records'];selected=point['selected_records'];admissions=point['surviving_admissions'];keys=point['current_coefficient_keys']
                need(point['remaining_horizon'] == horizon-step and integer(point['current_alive_units']), 'remaining_horizon_alive')
                need(all(point[key] is True for key in FLAGS), 'all_checkpoint_check_receipts')
                need(integer(deleted,step,n) and deleted > previous_deleted, 'deleted_record_count_bounds')
                if label=='R':
                    need(deleted == step and point['current_alive_units'] == n-step, 'record_deletion_counts')
                else:
                    universe=point['current_alive_units']+step;source_universe.add(universe)
                    need(frame <= universe <= n and deleted >= step, 'source_universe_bounds')
                    need(deleted <= n-point['current_alive_units'], 'source_survivor_record_bound')
                need(integer(selected,0,n-deleted) and integer(admissions,0,min(selected,n-initial_selected)), 'membership_count_bounds')
                retained_initial = selected-admissions
                need(0 <= retained_initial <= initial_selected and 0 <= initial_selected-retained_initial <= deleted, 'original_selection_survival_bounds')
                need(integer(keys,0,2*(n-deleted)) and (keys>0 or selected==0), 'checkpoint_key_count_bound')
                need(finite_positive(point['fresh_oracle_seconds']), 'oracle_timing');timing_sum+=point['fresh_oracle_seconds']
                profile.append({'checkpoint':step,'deleted_records':deleted,'current_alive_units':point['current_alive_units']})
                monotone_key=(label,step)
                if monotone_key in previous_selected:
                    need(selected >= previous_selected[monotone_key], 'fixed_retained_selection_threshold_monotonicity'); selected_monotonicity_checks+=1
                previous_selected[monotone_key]=selected
                previous_deleted=deleted;checkpoint_count+=1
            need(label not in profiles or profiles[label] == profile, 'deletion_volume_profile_changed_across_cuts');profiles[label]=profile
            checks=check_meter(arm,n);meter_checks+=1;arm_counts[label]+=1;singleton_count+=arm['singleton_checks']
            cut_summary['arms'][label]={'status':'completed','sampling_frame_size':frame,'effective_horizon':horizon,
                'singleton_checks':arm['singleton_checks'],'initial_keys':arm['initial_keys'],'initial_incidences':arm['initial_incidences'],
                'meter_bounds':checks,'meter':arm['meter'],'checkpoints':points,
                'construction_seconds':arm['construction_seconds'],'singleton_repair_seconds':arm['singleton_repair_seconds']}
        summaries.append(cut_summary)
    need(len(source_universe)<=1, 'stable_source_universe_across_all_cuts')
    need(timing_sum <= done['elapsed_seconds'], 'disjoint_reported_timing_bound')
    summary={'thresholds_completed':51,'record_arms_completed':arm_counts['R'],'source_arms_completed':arm_counts['S'],
        'unavailable_arms':unavailable,'checkpoint_count':checkpoint_count,'singleton_membership_checks':singleton_count,
        'request_frame_sizes':frame_sizes,'horizon':128,'checkpoints':[1,8,32,128],'no_selected_cutoff':True,
        'human_or_model_annotations':0,'task_learner_trained':False}
    need(summary==review['summary'],'independent_summary_identity')
    need(summaries==review['full51cut_summary'],'all51_review_summary_identity')
    need(digests==review['request_order_digests'] and profiles==review['request_profiles'],'request_receipt_identity')
    need(next(iter(source_universe),None)==review['source_universe_size_inferred_from_receipts'],'source_universe_receipt')
    need(timing_sum==resources['reported_disjoint_component_seconds'],'component_timing_identity')
    for key,value in review['runtime_summary'].items():need(resources[key]==value,'review_resource_identity')
    need(meter_checks==review['checks']['independent_meter_bound_and_conservation_checks']==102,'meter_check_count')
    need(selected_monotonicity_checks==review['checks']['fixed_retained_selection_monotonicity_checks']==400,'threshold_monotonicity_count')
    need(note['precise_interpretation']==review['fresh_retained_graph_equal_interpretation'],'field_interpretation')
    need(review['original_v4_failure_retained'] is True and review['original_semantic_effect_study_complete'] is False,'original_gates')
    allcuts=read(ROOT/'allcut_summary.json')
    need(allcuts['all_thresholds']==51 and allcuts['threshold_selected'] is False,'allcut_scope')
    need(allcuts['complete_correctness_check_summary']==summary,'allcut_check_summary')
    observed={}
    for label in ('R','S'):
        observed[label]={}
        for step in (1,8,32,128):
            points=[next(p for p in x['arms'][label]['checkpoints'] if p['checkpoint']==step) for x in summaries]
            observed[label][str(step)]={'cuts_with_positive_surviving_admissions':sum(p['surviving_admissions']>0 for p in points),
                'minimum_surviving_admissions':min(p['surviving_admissions'] for p in points),
                'maximum_surviving_admissions':max(p['surviving_admissions'] for p in points),
                'deleted_record_counts':sorted({p['deleted_records'] for p in points})}
    need(observed==allcuts['observed_surviving_admission_ranges_by_checkpoint'],'allcut_admission_summary')
    print(json.dumps({'passed':True,'public_files_verified':len(manifest['files']),
        'completion_output_hashes':len(done['outputs']),'meter_checks':meter_checks,
        'fixed_retained_selection_monotonicity_checks':selected_monotonicity_checks,'summary':summary,
        'scope':'Portable aggregate arithmetic/metadata replay; private states/graphs and original raw receipts are not recomputed.'},indent=2,sort_keys=True))

if __name__=='__main__':main()
