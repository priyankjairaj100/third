"""Prospective completion recipes. These recipes do not create scientific evidence."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse, gzip, hashlib, json, math, random, re

from phase5 import study_registry as legacy

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'ccu-phase6-recipes-1'
SEED = 20271003
CORE = ('civil_comments', 'askubuntu', 'cc_news')
LABELED = CORE[:2]
SIZES = (10000, 25000, 100000, 200000)
METHODS = list(legacy.METHODS)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seed(*parts):
    return int(digest([SEED, 'phase6-prospective-v1', *parts])[:32], 16)


def _recipe(name, executor, policy, *, new_choices=(), inputs=(), scope='required'):
    original = dict(legacy.UNRESOLVED)[name]
    return dict(id=name, prior_obligation=original, recipe_status='resolved_prospectively',
                executor=executor, scope=scope, policy=policy,
                new_prospective_choices=list(new_choices), actual_input_requirements=list(inputs),
                scientific_evidence_created=False)


def build_recipe_book():
    """Resolve decisions now. Keep observations and judgments as required inputs."""
    recipes = [
        _recipe('structure_mass_PPS', 'structure', {
            'panels': 'all12_core_structure_panels', 'trajectories_per_panel': 64,
            'law': 'successive_without_replacement_probability_proportional_to_original_source_record_count',
            'frame': 'genuine_native_sources_only', 'horizon': 8, 'checkpoints': [1, 2, 4, 8],
            'source_weight_after_other_source_deletion': 'unchanged_disjoint_original_mass',
            'matched_R': 'one_independent_uniform_record_permutation_per_path_at_its_raw_counts',
            'inference': 'separate_weighted_request_distribution_never_pool_uniform_S',
            'algorithm': 'pps_order_exact_integer_tickets'},
            new_choices=['64 weighted paths on each registered structure panel'],
            inputs=['authentic_panel_and_complete_source_partition']),
        _recipe('structure_one_percent_horizon', 'structure', {
            'panels': 'all12_core_structure_panels', 'allocations': {'R': 64, 'U': 32, 'A': 16},
            'initial_record_horizon': 'ceil(N/100)',
            'checkpoints': 'sorted_unique_positive_min(1,8,32,128,K) with each entry clipped at feasible K',
            'state': 'separate_initial_state_never_extend_128_record_state',
            'adversary': 'same_graph_only_candidate_rule_at_its_own_horizon'},
            new_choices=['64 R, 32 U, 16 A per structure panel', 'Include endpoint K even when K exceeds128'],
            inputs=['authentic_fixed_panel_graph']),
        _recipe('excluded_blocker_stress', 'structure', {
            'panels': 'all12_core_structure_panels', 'trajectories_per_panel': 32,
            'arm': 'A-excluded-blockers', 'horizon': 128, 'checkpoints': [1, 8, 32, 128],
            'candidate_rule': 'phase4.requests._candidate_search(excluded_only=True)',
            'candidate_cap': 1024, 'padding': 'fixed_all_record_order_after_actual_blocker_prefix',
            'no_candidate': 'retain_random_padding_only_status', 'search_cost': 'separate'},
            new_choices=['32 separate stress paths per structure panel'], inputs=['authentic_fixed_panel_graph']),
        _recipe('news_chronological', 'chronological', {
            'panel': 'cc_news/complete-calendar-window', 'streams': 1, 'resolution': 'day',
            'years': [2018, 2019], 'order': 'original_date_then_record_ID',
            'whole_group_rule': 'stop_before_next_whole_day_exceeds128_never_split_or_skip',
            'inference': 'descriptive_one_stream_no_independent_replicate_claim',
            'implementation': 'phase5.replication.news_chronological_requests'},
            new_choices=['One daily stream with cumulative record budget128'],
            inputs=['authentic_dates_and_complete_calendar_window']),
        _recipe('fresh_graph_audit', 'fresh_graph_audit', {
            'panels': 'each_core_primary10000', 'path_prefix_counts': {'R': 8, 'S': 4, 'U': 2, 'A': 2},
            'checkpoints': 'all_registered_on_each_selected_path',
            'oracle': 'original_retained_FP32_embeddings_fixed_priorities_pair_local_scorer',
            'all_other_paths': 'independently_built_exhaustive_cached_graph',
            'mismatch': 'fresh_similarity_adjudication_all_mismatches_failures_retained',
            'cost': 'audit_excluded_from_service_timing_and_reported_separately'},
            new_choices=['8 R, 4 S, 2 U, 2 A on each primary10k panel'], inputs=['authentic_embeddings']),
        _recipe('common_cached_graph_oracle', 'method_service', {
            'panels': 'both_labeled_primary10000', 'paths': 'first8R_first8S', 'repetitions': 5,
            'methods': ['O-G'], 'tier': 'cached_exhaustive_graph_reselection_plus_fresh_moments_decoder',
            'comparator': 'same_paths_existing_retained_similarity_O-G_tier',
            'graph_preparation': 'charged_once_shared_ledger_never_zero',
            'source_control': 'timing_panel_does_not_add_independent_request_trajectories'},
            new_choices=['Match existing 8 R/8 S timing panel'], inputs=['accepted_method_bundle']),
        _recipe('utility_reference', 'utility', {
            'panels': 'both_labeled_primary10000',
            'methods': ['curated_ridge', 'uncurated_ridge', 'same_size_hash_ridge', 'curated_logistic'],
            'hash_salt': 'ccu-v1-utility-hash', 'hash_serialization': 'UTF8(salt + NUL + record_ID)', 'lambda': 'same_calibration_only_representation_lock',
            'logistic': {'gradient_tolerance': 1e-11, 'max_iterations': 100,
                         'max_backtracks': 60, 'parameter_radius': 1e-8,
                         'certificate_max_coordinates': 2000000,
                         'over_budget': 'retained_infeasible_certificate_status'},
            'meaning': 'initial_task_utility_not_alternative_unlearning_targets'},
            new_choices=['Reuse frozen Phase5 logistic convergence policy without tolerance changes'],
            inputs=['authentic_train_test_and_calibration_task_bundle']),
        _recipe('negative_controls', 'negative_controls', {
            'curator_controls': ['exact_normalized_text_fixed_representative', 'fixed_order_greedy_independent_set'],
            'panels': 'each_core_primary10000', 'own_excluded_paths_per_curator': 32,
            'control_eligibility': 'own_initially_excluded_set_not_primary_curator_set',
            'checkpoints': [1, 8, 32, 128], 'invariant': 'surviving_selection_unchanged_for_own_excluded_deletion',
            'floor': 'both_labeled_primary10000_first64R_no_curation_fresh_ridge',
            'empty_pool': 'not_applicable_preserved_no_success_count'},
            new_choices=['32 own-excluded paths per curator per core10k', '64 R no-curation paths per labeled10k'],
            inputs=['authentic_records_graph_and_task_arrays']),
        _recipe('head_normalization_error_diagnostic', 'normalization_diagnostic', {
            'panel': 'civil_comments/primary-10000', 'trajectory_id': 'R000', 'checkpoint': 'final_feasible',
            'wrong_system': 'retained_Gram_plus_lambda_times_initial_selected_count',
            'correct_system': 'retained_Gram_plus_lambda_times_current_selected_count',
            'no_count_change': 'retain_zero_diagnostic_do_not_search_another_path',
            'role': 'wrong_implementation_diagnostic_never_method_competitor'},
            new_choices=['Fix R000 final feasible checkpoint before model outcomes'], inputs=['accepted_task_bundle']),
        _recipe('graph_envelope_conditional', 'graph_envelope', {
            'panels': 'each_core_primary10000', 'allocations': {'R': 32, 'S': 32},
            'bounds': 'phase5.boundary.score_intervals_and_graph_envelope',
            'uniformity': 'every_actual_scorer_pair_covered_not_sample_maximum_or_ANN_recall',
            'outputs': ['selection_containment', 'head_error_bound', 'observed_bound_tightness'],
            'gate': 'validated_runtime_arithmetic_assumptions_and_independent_interval_audit',
            'failure': 'retain_failed_or_uninstantiated_status_no_theorem_backed_claim'},
            new_choices=['Same 32 R/32 S boundary allocation on core10k panels'], inputs=['actual_bound_runtime_and_authentic_embeddings']),
        _recipe('every_release_persistence', 'method_service', {
            'panels': 'both_labeled_primary10000', 'paths': 'first8R_first8S', 'repetitions': 5,
            'methods': METHODS, 'persistence': 'every_release', 'primary_comparator': 'final_only',
            'charges': ['serialize', 'write', 'fsync', 'compaction', 'head_release'],
            'compaction_fraction': 0.5, 'batch_rows': 1024,
            'schedule': 'same_for_all_methods_rebuild_initial_state_each_repeat'},
            new_choices=['Use existing timing subset and fixed payload compaction fraction0.5'], inputs=['accepted_method_bundle']),
        _recipe('cold_warm_scope', 'method_service', {
            'panels': 'both_labeled_primary10000', 'paths': 'first8R_first8S', 'repetitions': 5,
            'methods': METHODS, 'primary_panel': 'fresh_process_load', 'additional_panel': 'warm_service',
            'warm_rule': 'warm_imports_and_BLAS_then_construct_fresh_state_before_any_request',
            'no_replay_on_mutated_state': True, 'cpu_threads': 1,
            'OS_page_cache': 'uncontrolled_reported_never_called_cold_disk',
            'method_order': 'seeded_shuffle_within_corpus_path_repeat', 'resource_policy': 'derive_resource_policy'},
            new_choices=['Match existing timing subset, fixed one CPU thread'], inputs=['actual_local_resource_snapshot', 'disjoint_development_stage_costs']),
        _recipe('official_Civil_splits_optional', 'disabled_optional', {
            'activated': False, 'reason': 'optional_comparability_scope_removed_before_confirmation_to_protect_required_primary_evidence',
            'primary_source_disjoint_protocol': 'unchanged',
            'activation': 'requires_separate_prospective_version_before_any_affected_outcomes'},
            new_choices=['Do not activate optional official-split comparison in this version'], scope='explicitly_deactivated_optional'),
        _recipe('human_threshold_calibration', 'human_threshold', {
            'corpus_encoder_pairs': [['civil_comments', 'e5'], ['civil_comments', 'mpnet'], ['askubuntu', 'e5'],
                                     ['cc_news', 'e5'], ['english_stackexchange', 'e5']],
            'selection_max_pairs_each': 600, 'selection_bins': 20, 'selection_per_bin': 30,
            'validation_max_pairs_each': 200, 'judgments_each': 3,
            'selection': 'least_strict_grid_threshold_weighted_positive_rate_at_least0.95_nonempty',
            'validation': 'independent_SRS_all_above_threshold_pairs_fresh_blinded_judgments',
            'gate': 'one_sided95pct_exact_finite_population_lower_bound_at_least0.90',
            'WCEP': 'inherits_same_encoder_CCNews_threshold_never_refits',
            'threshold_sensitivity_validation': 'two_additional_Civil_E5_200_pair_independent_audits',
            'response_authenticity': 'real_independent_collection_required_no_automatic_dispatch'},
            inputs=['authentic_pair_frames', 'actual_three_rater_responses', 'collection_process_record']),
        _recipe('human_admission_main', 'human_admission', {
            'maximum_admission_pairs': 400, 'maximum_control_pairs': 200,
            'quota_implementation': 'phase4.admission_audit.fixed_existing_quotas_and_strata',
            'sampling': 'unique_pairs_probability_sample_with_complete_frame_and_inclusion_weights',
            'former_blocker': 'fixed_uniform_ID_rule', 'raters': 3, 'shortfall': 'no_quota_transfer',
            'blinding': ['method', 'request_arm', 'labels', 'model_effects'],
            'analysis': 'phase5.human_analysis_with_missingness_bounds',
            'confidence_intervals': 'phase6.human_inference.analyze_intervals_alpha1over40_for_pair_and_context_families',
            'public_text': 'rights_limited_excerpts_only_no_News_bodies'},
            inputs=['authentic_admission_frames', 'actual_three_rater_responses', 'collection_process_record']),
        _recipe('human_admission_context', 'human_context', {
            'kinds': ['former_neighborhood', 'nearest_surviving_selected'],
            'maximum_records_each': 100, 'corpus_quotas_each': {'civil_comments': 34, 'askubuntu': 33, 'cc_news': 33},
            'selection': 'independent_fixed_SRS_from_complete_unique_admission_frame_per_kind',
            'raters': 3, 'target_character_limit': 8000, 'context_character_limit': 20000,
            'overflow': 'logged_truncation_and_missingness_bounds_never_claim_complete_context',
            'nearest_pool': 'checkpoint_surviving_selected_minus_query',
            'sampling_implementation': 'phase5.context_audit.prepare_context_audits'},
            inputs=['authentic_admission_frames', 'actual_three_rater_responses', 'collection_process_record']),
        _recipe('boundary_method_scope', 'method_service', {
            'all_eight_methods_retained': METHODS,
            'factors': ['projection32', 'projection64', 'projection128', 'projection256', 'lambda_x0.1', 'lambda_x10', 'FP32', 'zero_start_CG'],
            'FP32_role': 'separate_accuracy_memory_frontier',
            'FP32_required_variants': METHODS,
            'FP32_supported_variant': 'explicit_named_variant_and_actual_state_dtype_required',
            'unsupported_precision': 'retain_cell_status_no_silent_FP64_substitution',
            'primary_FP64_common_decoder': 'phase5.decoders.DEFAULT_POLICY',
            'CG': {'initialization': 'zero', 'maximum_iterations': '10*dimension', 'eta': 1e-10,
                   'preconditioner': None, 'warm_start': False},
            'no_silent_scope_narrowing': True}, inputs=['accepted_same_inputs_per_factor']),
        _recipe('matched_R_method_scope', 'all_registered_executors', {
            'applies': 'every_unique_registered_S_path_and_weighted_S_path_not_repeated_timing_or_audit_references',
            'methods': 'exactly_parent_method_list', 'initial_record_horizon': 'parent_final_raw_count',
            'checkpoints': 'deduplicated_parent_cumulative_raw_counts',
            'coupling': 'one_independent_record_permutation_prefix_path',
            'state': 'separately_sufficient_initial_record_state',
            'inference': 'diagnostic_not_extra_primary_R_replication',
            'source_missing': 'preserve_parent_and_control_unavailable_status'}, inputs=['actual_parent_source_path']),
        _recipe('test_source_bootstrap', 'statistics', {
            'replicates': 10000, 'seed': 20271004, 'primary_unit': 'whole_trajectory_with_all_checkpoints',
            'test_scope_primary': 'fixed_test_population', 'secondary': 'crossed_trajectory_and_whole_test_source',
            'pairing': 'same_draws_for_all_methods', 'nonlinear_metrics': 'recompute_macroF1_and_denominators_per_draw',
            'signed_zero_negative': 'retain_all', 'zero_denominator': 'undefined_no_floor',
            'missing_planned_cells': 'retain_status_unconditional_estimand_undefined',
            'loss_Holm_family': ['civil_R', 'civil_S', 'askubuntu_R', 'askubuntu_S'],
            'systems_Holm_family': ['civil_equal_R_S_vs_BE', 'askubuntu_equal_R_S_vs_BE'],
            'alpha': 0.05, 'directional_p_values': 'only_valid_test_outputs_never_derive_from_percentile_CI',
            'development_precision': 'precision_analysis_fixed_before_confirmation'}, inputs=['complete_run_registry', 'authentic_test_source_partition', 'disjoint_development_effects']),
        _recipe('primary_numerical_release_gate', 'method_service', {
            'eta_maximum': 1e-10, 'residual': 'coordinate_order_Neumaier_FP64',
            'condition': 'report_symmetric_eigenvalue_estimates_and_log_condition',
            'moment_symmetry': 'exact_stored_symmetry_required', 'empty_target': 'zero_moments_and_zero_head',
            'fallback': 'none_retain_failure_never_relax_tolerance',
            'certificate_scope': 'residual_verified_not_interval_certified',
            'strict_rational': 'separate_existing_exact_target_service_not_FP64_relabel',
            'audits': 'extended_precision_on_precommitted_paths_and_all_numerical_failures'},
            inputs=['actual_worker_output_and_independent_oracle']),
        _recipe('worker_dependency_and_IO_accounting', 'measurement', {
            'dependencies': 'hash_actual_loaded_ELF_modules_python_code_and_runtime_versions',
            'provenance_limit': 'loaded_dependency_inventory_not_complete_transitive_or_build_closure',
            'logical_reads': 'measured_file_read_bytes_and_instrumented_payload_metadata_accesses_separate_fields',
            'unmeasured_reads': None, 'physical_bus_or_cache_traffic': None,
            'memory': 'individual_wait4_RSS_and_sampled_simultaneous_process_tree_sum_separate_fields',
            'sampled_peak': 'descriptive_nonatomic_sweep_maximum_not_bound_on_true_concurrent_peak',
            'RSS_scope': 'shared_pages_may_double_count_and_short_lived_processes_may_be_missed',
            'never_sum_individual_peaks': True, 'output_bytes': 'explicit_writes_plus_snapshot_and_head_sizes',
            'access_gate': 'actual_kernel_denial_probe_after_method_only_state_load',
            'charges': 'construction_load_update_decode_release_persist_compaction_all_separate'},
            inputs=['actual_runtime_inventory', 'actual_instrumented_worker_logs']),
    ]
    assert {r['id'] for r in recipes} == {k for k, _ in legacy.UNRESOLVED}
    return dict(schema=SCHEMA, date='2026-10-04', recipe_count=len(recipes), unresolved_recipe_count=0,
                recipes=recipes, defaults_chosen_before_primary_outcomes=True,
                primary_semantic_study_started=False, scientific_acceptance_granted=False,
                actual_input_availability_not_inferred=True,
                supporting_policies={
                    'ANN': {'tables':8,'bits':10,'seed':202710041,'sample_size':200,'sample_seed':202710042,
                            'selection':'fixed_existing_candidate_scheme; poor_recall_is_retained'},
                    'CG': {'eta':1e-10,'initialization':'zero','max_iterations':'10*dimension'},
                    'refit_panel': {'salt':'ccu-v1-refit-panel','soft_target':5000,
                        'eligible_sources':'surviving_training_sources_minus_largest_primary_panel_sources_minus_second_replication_panel_sources',
                        'selection':'hash_order_whole_source_prefix','shortfall':'retain_no_source_reuse'},
                    'refit_configuration': {'clusters':71,'iterations':25,'seed':0,'alternative_seeds':[1,2,3],
                        'policy':'hard','spherical':True,'threads':1,'backend':'official_pinned_SemDeDup',
                        'epsilon':'1_minus_actual_calibration_threshold',
                        'new_prospective_default':'ceil(sqrt(5000)) clusters; qualification on disjoint development before confirmation',
                        'retained_below_cluster_count':'explicit_failure_no_adaptive_reduction'},
                    'convex_verification': {'default_max_coordinates':2000000,
                        'larger_limit_selector':'select_verification_budget',
                        'coordinate_definition':'n*d*q','parameter_tolerance':1e-8,
                        'observed_over_budget':'forecast_or_budget_refusal_not_observed_timeout'}},
                source_bindings={str(p.relative_to(legacy.REPO)): legacy.sha(p) for p in
                                 (legacy.PROTOCOL, legacy.DESIGN, Path(legacy.__file__))})


def pps_order_exact_integer_tickets(source_masses, *, draw_count, seed_value):
    """Sequential PPS without replacement. Integer tickets avoid rounded weights."""
    if type(draw_count) is not int or draw_count < 0 or type(seed_value) is not int:
        raise ValueError('Nonnegative integer draw count and integer seed required')
    items = sorted(source_masses.items())
    if any(not isinstance(k, str) or not k or type(v) is not int or v <= 0 for k, v in items):
        raise ValueError('Nonempty source IDs and positive integer masses required')
    rng = random.Random(seed_value); chosen = []
    for _ in range(min(draw_count, len(items))):
        ticket = rng.randrange(sum(m for _, m in items))
        for index, (sid, mass) in enumerate(items):
            if ticket < mass:
                chosen.append(sid); items.pop(index); break
            ticket -= mass
    return chosen


def one_percent_horizon(population_size, feasible_count=None):
    if type(population_size) is not int or population_size < 0:
        raise ValueError('Nonnegative integer population size required')
    if feasible_count is None: feasible_count = population_size
    if type(feasible_count) is not int or not 0 <= feasible_count <= population_size:
        raise ValueError('Feasible count must lie within the population')
    initial = (population_size + 99) // 100
    feasible = min(initial, feasible_count)
    levels = sorted({min(v, feasible) for v in (1, 8, 32, 128, initial) if min(v, feasible) > 0})
    return dict(initial_horizon=initial, feasible_horizon=feasible, checkpoints=levels,
                status='ready' if feasible else 'empty_request_frame')


def precision_analysis(effects, *, development_ids, confirmation_ids, simulations=10000,
                       candidate_counts=(64, 128, 256, 512, 1024), default_count=256):
    """Plan precision from disjoint observations. Empirical power is conditional.

    This uses paired trajectory effects, an empirical centered distribution, and
    a prespecified shift of0.01. It does not certify power under unseen tails.
    It does not change the registered allocation after observing confirmation.
    """
    import numpy as np
    from scipy.stats import beta, t
    x = np.asarray(effects, dtype=float)
    if x.ndim != 1 or len(x) < 16 or not np.isfinite(x).all():
        raise ValueError('At least16 finite development trajectory effects required')
    if any(not isinstance(v,str) or not v for v in list(development_ids)+list(confirmation_ids)):
        raise ValueError('Nonempty development and confirmation IDs required')
    if len(development_ids) != len(x) or len(set(development_ids)) != len(x):
        raise ValueError('Distinct development trajectory IDs required')
    if set(development_ids) & set(confirmation_ids):
        raise ValueError('Development and confirmation overlap')
    if type(simulations) is not int or simulations < 1000:
        raise ValueError('At least1000 simulation draws required')
    if not candidate_counts or any(type(n) is not int or n < 16 for n in candidate_counts):
        raise ValueError('Candidate counts must be integers at least16')
    if list(candidate_counts) != sorted(set(candidate_counts)) or default_count not in candidate_counts:
        raise ValueError('Ordered unique candidate counts including fixed default required')
    centered = x - x.mean(); sd = float(x.std(ddof=1)); rng = np.random.default_rng(20271004)
    rows = []; delta = .01; family_alpha = .05 / 4
    for n in candidate_counts:
        cutoff = float(t.ppf(1-family_alpha, n-1)); rejected = 0
        for begin in range(0, simulations, 128):
            b = min(128, simulations-begin)
            z = centered[rng.integers(0, len(x), size=(b, n))] + delta
            means = z.mean(axis=1); errors = z.std(axis=1, ddof=1) / math.sqrt(n)
            rejected += int(np.count_nonzero(means > cutoff * errors))
        lower = float(beta.ppf(.05, rejected, simulations-rejected+1)) if rejected else 0.
        half_width = float(t.ppf(.975, len(x)-1) * sd / math.sqrt(n))
        rows.append(dict(trajectories=n, projected_mean_half_width=half_width,
                         empirical_shifted_power=rejected/simulations,
                         Monte_Carlo_one_sided95_lower=lower,
                         meets_planning_targets=sd > 0 and half_width <= .01 and lower >= .8))
    adequate = [r['trajectories'] for r in rows if r['meets_planning_targets']]
    return dict(schema='ccu-development-precision-1', developmental_sample_size=len(x),
                development_IDs_sha256=digest(list(development_ids)), confirmation_IDs_sha256=digest(list(confirmation_ids)),
                effect_sha256=digest(x.tolist()), rows=rows, registered_default=default_count,
                smallest_candidate_meeting_targets=min(adequate) if adequate else None,
                allocation_actually_changed=False, simulations=simulations, shift=delta,
                target_half_width=.01, target_power=.8, family_alpha=family_alpha,
                status='development_degenerate_estimation_only' if sd == 0 else 'default_meets_conditional_planning_targets' if next(r for r in rows if r['trajectories']==default_count)['meets_planning_targets'] else 'default_estimation_only_or_prospective_amendment_required',
                limitations=['Empirical residual resampling does not model unseen tails',
                             'Projected half-width uses observed development variance, not a guaranteed bound',
                             'Monte Carlo interval covers simulation randomness only',
                             'No significance-based continuation or confirmation-data planning'])


def derive_resource_policy(snapshot, development_stage_seconds):
    """Lock common limits using actual local observations before confirmation."""
    limits = [snapshot.get(k) for k in ('physical_available_bytes', 'cgroup_remaining_bytes')]
    limits = [v for v in limits if v is not None]
    if not limits or any(type(v) is not int or v <= 0 for v in limits):
        raise ValueError('Actual positive local available-memory observations required')
    if type(snapshot.get('local_cpu_count')) is not int or snapshot['local_cpu_count'] < 1:
        raise ValueError('Actual local CPU count required')
    if not development_stage_seconds or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in development_stage_seconds):
        raise ValueError('Actual positive disjoint-development stage timings required')
    # Half of the observed remaining limit leaves the controller and shared state room.
    memory = min(limits) // 2
    timeout = math.ceil(4 * max(development_stage_seconds))
    return dict(schema='ccu-local-resource-policy-1', snapshot_sha256=digest(snapshot),
                development_times_sha256=digest(development_stage_seconds),
                per_method_address_space_limit_bytes=memory, cpu_threads=1,
                wall_seconds_each_stage=timeout, cpu_seconds_each_stage=timeout,
                formula='half_min_observed_available_memory;ceil(4*maximum_observed_development_stage_seconds)',
                methods_share_cap=True, oversize_status='infeasible_preserved_before_allocation',
                native_dimension_reduction=False, external_compute_authorized=False,
                amendment_after_confirmation=False)


def select_verification_budget(observations, *, rows, dimension, outputs, resource_policy):
    """Choose a measured development budget. The forecast is not a cost guarantee."""
    for value in (rows,dimension,outputs):
        if type(value) is not int or value < 1:raise ValueError('Positive target dimensions required')
    if not observations:raise ValueError('Actual dense development measurements required')
    rates=[];peaks=[]
    for entry in observations:
        required=('rows','dimension','outputs','feature_density','seconds','peak_rss_bytes','source_sha256','input_bytes')
        if any(k not in entry for k in required):raise ValueError('Complete measurement record required')
        if entry['dimension']!=dimension or entry['outputs']!=outputs or not .95<=entry['feature_density']<=1:
            raise ValueError('Dense measurements at the target dimension and output count required')
        if type(entry['rows']) is not int or entry['rows']<1 or not math.isfinite(entry['seconds']) or entry['seconds']<=0:
            raise ValueError('Actual positive development size and time required')
        if type(entry['peak_rss_bytes']) is not int or entry['peak_rss_bytes']<=0 or type(entry['input_bytes']) is not int or entry['input_bytes']<0:
            raise ValueError('Actual RSS and input byte counts required')
        if not isinstance(entry['source_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',entry['source_sha256']):
            raise ValueError('Development source hash required')
        rates.append(entry['seconds']/(entry['rows']*dimension*outputs))
        peaks.append(entry['peak_rss_bytes']+max(0,(4*dimension+8*outputs)*rows-entry['input_bytes']))
    demand=rows*dimension*outputs
    seconds=4*max(rates)*demand
    memory=max(peaks)
    time_cap=resource_policy['wall_seconds_each_stage'];memory_cap=resource_policy['per_method_address_space_limit_bytes']
    feasible=seconds<=time_cap and memory<=memory_cap
    return dict(schema='ccu-development-verification-budget-1',observations_sha256=digest(observations),
        requested_coordinates=demand,max_coordinates=demand if feasible else 2000000,
        forecast_seconds=seconds,forecast_peak_rss_bytes=memory,
        forecast_feasible=feasible,observed_timeout=False,observed_native_feasibility=False,
        formula='4*maximum_dense_seconds_per_coordinate*target_coordinates; maximum_observed_RSS_plus_extra_input_bytes',
        limitation='Finite development extrapolation; retain all actual timeout/OOM outcomes',
        parameter_tolerance=1e-8,decision_before_confirmation=True)


def build_registry():
    """Preserve Phase5 A–G jobs. Replace only the21 placeholder obligations."""
    old, old_jobs = legacy.build_registry()
    jobs = [deepcopy(j) for j in old_jobs if j.get('kind') != 'unresolved_protocol_obligation']
    recipe_book = build_recipe_book(); by_id = {r['id']: r for r in recipe_book['recipes']}
    extra = []
    def add(name, corpus, panel, suffix, *, methods=None, **fields):
        recipe = by_id[name]
        extra.append(dict(job_id=f'H_recipes/{name}/{corpus}/{panel}/{suffix}',
                          kind='resolved_recipe_job', recipe_id=name, corpus=corpus, panel=panel,
                          executor=recipe['executor'], methods=methods or ['independent_structure_oracle'],
                          required_artifact_keys=list(recipe['actual_input_requirements']),
                          execution_allowed=False, status='planned_requires_actual_inputs', **fields))
    for corpus in CORE:
        for size in SIZES:
            panel = f'primary-{size}'
            for i in range(64):
                parent = f'S-PPS{i:03d}'
                add('structure_mass_PPS', corpus, panel, parent, arm='S-PPS', trajectory_id=parent,
                    planned_horizon=8, planned_checkpoint_units=[1,2,4,8])
                add('structure_mass_PPS', corpus, panel, parent+'/volume-matched-R',
                    arm='R-volume-matched-to-S-PPS', trajectory_id='R-volume-matched-to-'+parent,
                    parent_source_trajectory=parent, planned_horizon='parent_final_raw_count',
                    planned_checkpoint_units='parent_cumulative_raw_counts')
            for arm, count in (('R',64),('U',32),('A',16)):
                for i in range(count):
                    add('structure_one_percent_horizon',corpus,panel,f'{arm}{i:03d}',arm=arm,
                        trajectory_id=f'{arm}{i:03d}', planned_horizon='ceil(N/100)',
                        planned_checkpoint_units='one_percent_horizon(N,feasible_frame).checkpoints')
            for i in range(32):
                add('excluded_blocker_stress',corpus,panel,f'A-excluded-blockers{i:03d}',
                    arm='A-excluded-blockers',trajectory_id=f'A-excluded-blockers{i:03d}',
                    planned_horizon=128,planned_checkpoint_units=[1,8,32,128])
        for arm,count in (('R',8),('S',4),('U',2),('A',2)):
            for i in range(count):
                add('fresh_graph_audit',corpus,'primary-10000',f'{arm}{i:03d}',arm=arm,
                    trajectory_id=f'{arm}{i:03d}', fresh_graph=True, parent_job_id=f'A_structure/{corpus}/primary-10000/native/{arm}{i:03d}')
        for curator in ('exact_text','fixed_order_greedy'):
            for i in range(32):
                add('negative_controls',corpus,'primary-10000',f'{curator}/U{i:03d}',
                    arm='own-excluded-U',trajectory_id=f'U{i:03d}',curator=curator,
                    planned_horizon=128,planned_checkpoint_units=[1,8,32,128])
        for arm in ('R','S'):
            for i in range(32):
                tid=f'{arm}{i:03d}'
                add('graph_envelope_conditional',corpus,'primary-10000',tid,arm=arm,trajectory_id=tid)
                if arm=='S':
                    add('graph_envelope_conditional',corpus,'primary-10000',tid+'/volume-matched-R',
                        arm='R-volume-matched-to-S',trajectory_id='R-volume-matched-to-'+tid,
                        parent_source_trajectory=tid,planned_horizon='parent_final_raw_count')
    add('news_chronological','cc_news','complete-calendar-window','daily-stream',arm='chronological',streams=1)
    for corpus in LABELED:
        for name in ('common_cached_graph_oracle','every_release_persistence','cold_warm_scope'):
            for arm in ('R','S'):
                for i in range(8):
                    for repeat in range(5):
                        add(name,corpus,'primary-10000',f'{arm}{i:03d}/repeat-{repeat}',
                            methods=['O-G'] if name=='common_cached_graph_oracle' else METHODS,
                            arm=arm,trajectory_id=f'{arm}{i:03d}',repeat=repeat,
                            parent_job_id=f'C_full_methods/{corpus}/primary-10000/native/{arm}{i:03d}')
        add('utility_reference',corpus,'primary-10000','initial-heads',
            methods=by_id['utility_reference']['policy']['methods'])
        for i in range(64):
            add('negative_controls',corpus,'primary-10000',f'no-curation/R{i:03d}',
                methods=['ordinary_ridge_deletion'],arm='R',trajectory_id=f'R{i:03d}',curator='none',
                parent_job_id=f'B_relevance/{corpus}/primary-10000/native/R{i:03d}')
    add('head_normalization_error_diagnostic','civil_comments','primary-10000','R000/final-feasible',
        methods=['correct_count_shift','wrong_frozen_count_shift'],trajectory_id='R000')
    for corpus, encoder in by_id['human_threshold_calibration']['policy']['corpus_encoder_pairs']:
        for stage in ('selection','validation'):
            add('human_threshold_calibration',corpus,'calibration',encoder+'/'+stage,
                methods=['blinded_human_analysis'],encoder=encoder,stage=stage)
    for sign in ('minus','plus'):
        add('human_threshold_calibration','civil_comments','calibration',f'e5/sensitivity-{sign}-validation',
            methods=['blinded_human_analysis'],encoder='e5',stage='sensitivity_validation',threshold_sign=sign)
    add('human_admission_main','all_core','primary-10000','balanced-unique-pair-audit',methods=['blinded_human_analysis'])
    for kind in ('former_neighborhood','nearest_surviving_selected'):
        add('human_admission_context','all_core','primary-10000',kind,methods=['blinded_human_analysis'])
    add('test_source_bootstrap','all_labeled','all_registered','joint-analysis',methods=['paired_and_crossed_bootstrap'])
    jobs += extra
    counts=dict(sorted(Counter(j['recipe_id'] for j in extra).items()))
    policy_only=[r['id'] for r in recipe_book['recipes'] if r['id'] not in counts]
    groups=deepcopy(old['groups'])
    for group in groups:
        conf=group['configuration']
        if group['block']=='E_full_refit':
            conf['cluster_policy']='phase6_recipe_book.supporting_policies.refit_configuration'
        if group['variant']=='zero-start-CG':
            conf['stopping_rule']='zero_start_no_preconditioner_max10d_eta1e-10'
        if group['variant']=='ANN':
            conf['ANN_policy']='phase6_recipe_book.supporting_policies.ANN'
    registry={**deepcopy(old), 'schema':'ccu-prospective-study-registry-2','groups':groups,
              'recipe_book_sha256':digest(recipe_book), 'recipe_code_sha256':legacy.sha(Path(__file__)),
              'legacy_registry_sha256':digest(old), 'legacy_preserved_job_count':len(jobs)-len(extra),
              'new_recipe_job_count':len(extra), 'recipe_job_counts':counts,
              'policy_only_recipes':policy_only, 'unresolved_obligations':[],
              'resolved_obligation_count':21, 'job_count':len(jobs),
              'jobs_content_sha256':digest(jobs), 'primary_semantic_study_started':False,
              'execution_allowed':False, 'actual_inputs_remain_required':True}
    return registry,jobs


def export(output_dir):
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Immutable export destination required')
    out.mkdir(parents=True,exist_ok=True)
    book=build_recipe_book(); registry,jobs=build_registry()
    for filename,value in [('recipes.json',book),('registry.json',registry)]:
        (out/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    raw=b''.join(json.dumps(j,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n' for j in jobs)
    (out/'jobs.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
    return registry


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('output_dir',type=Path); args=parser.parse_args()
    result=export(args.output_dir)
    print(json.dumps({k:result[k] for k in ('job_count','legacy_preserved_job_count','new_recipe_job_count','recipe_job_counts','policy_only_recipes')},indent=2))
