from pathlib import Path
import csv, json, shutil, hashlib, zipfile

root = Path('/workspace/scratch/35d4d4d8ba2c')
dest = root / 'output/empirical_program'
dest.mkdir(parents=True, exist_ok=True)

design = {
  'schema_version': '1.0',
  'status': 'design_reviewed_not_acquired_or_preregistered',
  'date': '2026-10-03',
  'paper': 'Unlearning What Was Never Trained',
  'venue_goal': 'ACL 2027',
  'experiment_results_present': False,
  'synthetic_datasets_allowed': False,
  'contracts': {
    'primary_curator': 'global_earlier_raw_neighbor_suppression',
    'primary_curator_fitted_to_deletable_data': False,
    'reference_graph': 'exhaustive_strict_cosine_threshold',
    'approximate_search_is_secondary_only': True,
    'service_output': 'head_only',
    'selected_ids_provided_by': 'isolated_independent_curator_oracle',
    'corpus_recovery_claim': False,
    'request_interface': 'identifiers_only',
    'retained_access': 'only_method_accounted_state',
    'history_claim': 'designated_abstract_state_only',
    'transcript_privacy_claim': False,
    'budget': 'cumulative_unique_units_no_reset',
    'exhaustion_policy': 'atomic_refusal',
    'source_service_universe': 'all_disjoint_partition_units_including_unknown_singletons',
    'source_random_sampling_frame': 'genuine_native_sources_only',
    'full_refit_comparison': 'separate_original_SemDeDup_branch_F_vs_R',
  },
  'seeds': {
    'master': 20271003,
    'source_split_salt': 'ccu-v1-source-split',
    'primary_panel_salt': 'ccu-v1-primary-panel',
    'replication_pool_salt': 'ccu-v1-replication-pool',
    'request_salt': 'ccu-v1-request',
    'priority_salt': 'ccu-v1-priority',
    'primary_priority_seed': 0,
    'sensitivity_priority_seeds': [1, 2],
    'bootstrap_seed': 20271004,
    'projection_seed': 20271005,
  },
  'datasets': {
    'civil_comments': {'role':'core_labeled','route':'TFDS civil_comments/CivilComments','version':'1.2.4','source':'publication_id+article_id','input':'text_only_no_parent_text','label':'original_toxicity_fraction','outputs':1,'author_withdrawal_claim':False},
    'askubuntu': {'role':'core_labeled','route':'corrected_company_April_2024_dump','date_filter':['2018-05-02','2023-12-31'],'records':'PostTypeId=1','source':'site+OwnerUserId_account_proxy','label':'top20_calibration_tags_multihot_keep_zero_targets','outputs':20,'benchmark':'new_question_level_task_not_duplicate_pair_retrieval'},
    'cc_news': {'role':'structure_and_human_audit','route':'vblagoje/cc_news_pinned_revision','calibration_year':2017,'confirmation_years':[2018,2019],'source':'registrable_domain_pinned_public_suffix_list','label':None,'untouched_raw_crawl_claim':False},
    'english_stackexchange': {'role':'within_platform_replication','route':'same_corrected_April_2024_dump_protocol','source':'site+OwnerUserId_account_proxy','outputs':20},
    'wcep100': {'role':'event_enriched_replication','route':'original_author_extracted_release','label':None,'event_ids_are_gold_duplicates':False,'source_arm':'only_if_original_urls_verified'},
  },
  'population': {
    'split_hash_quantiles': {'train':[0,0.70],'calibration':[0.70,0.85],'test':[0.85,1.0]},
    'split_unit':'genuine_source_or_missing_source_singleton',
    'primary_pool_replication_hash_quantile':[0,0.8],
    'second_panel_pool_replication_hash_quantile':[0.8,1.0],
    'guard':'train_side_exact_duplicates_crosssplit_duplicate_links_parent_crosssplit_and_primary_E5_similarity_gt_tau',
    'cross_encoder_population':'same_post_E5_guard_D_no_MPNet_refilter',
    'soft_primary_size_targets':[10000,25000,100000,200000],
    'group_boundary_policy':'include_whole_group_report_overshoot',
    'source_completeness_scope':'surviving_records_after_fixed_guards',
    'second_panel_size':10000,
    'full_refit_panel_size':5000,
    'wcep_article_soft_target':25000,
    'news_time_window':'first_complete_month_from_2018_01_with_at_least_10k_else_chronological_complete_month_prefix',
  },
  'features': {
    'primary_encoder':'intfloat/multilingual-e5-base',
    'secondary_encoder':'sentence-transformers/all-mpnet-base-v2',
    'curation_dimension':768,
    'native_learner_dimension':768,
    'projection_ablation_dimensions':[32,64,128,256],
    'projection':'fixed_data_independent_orthonormal_seeded',
    'chunk_content_tokens':256,
    'chunking':'nonoverlapping_all_chunks',
    'pooling':'content_token_weighted_mean_then_normalize',
    'e5_prefix':'query: ',
    'canonical_learner_z':'stored_FP32_document_vector_exactly_promoted_to_FP64_no_renormalization',
    'graph_e':'FP32_cache_promoted_and_reference_FP64_normalized',
  },
  'threshold_calibration': {
    'selection_pairs_max':600,
    'score_bins':20,
    'score_bin_range':[-1,1],
    'pairs_per_bin':30,
    'candidate_grid': [round(.50+.01*i,2) for i in range(50)]+[.995],
    'select':'least_strict_with_weighted_precision_at_least_0.95_and_nonzero_support',
    'support_is_reported_not_quality_guarantee':True,
    'validation_pairs':200,
    'validation_sampling':'independent_SRS_entire_above_threshold_population_possible_overlap_fresh_blinded_ratings_or_census',
    'quality_gate':'one_sided_95pct_lower_bound_at_least_0.90',
    'empty_pairs':'precision_undefined_quality_gate_not_passed',
    'sensitivity_delta':0.02,
    'post_confirmation_retuning':False,
  },
  'learner': {
    'type':'uniform_average_loss_matrix_ridge',
    'intercept':False,
    'normal_equations':'(M+lambda*n*I)Theta=H',
    'empty_target':'Theta=0',
    'regularization_grid':[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1],
    'selection':'five_source_group_folds_in_calibration_only_MSE_tie_larger_lambda',
    'moments_dtype':'FP64',
    'primary_decoder':'FP64_Cholesky_common_to_methods',
    'secondary_decoder':'zero_start_CG_common_stopping_rule',
    'normalized_residual_gate':1e-10,
    'rigorous_certificate_requires':'verified_moment_and_residual_error_allowance_not_merely_float_residual',
  },
  'requests': {
    'record_horizon':128,
    'record_checkpoints':[1,8,32,128],
    'source_horizon':8,
    'source_checkpoints':[1,2,4,8],
    'clipping':'clip_to_arm_feasibility_deduplicate_keep_empty_targets',
    'R':'uniform_permutation_of_all_records_prefixes',
    'S':'uniform_permutation_of_genuine_sources_prefixes',
    'U':'uniform_permutation_of_initially_excluded_records_prefixes',
    'A':'up_to_1024_real_candidates_per_fixed_seed_choose_largest_graph_only_admission_then_pad_with_seeded_ID_order',
    'matched_R':'one_independent_record_permutation_per_S_path_prefix_at_each_cumulative_raw_count_separate_horizon',
    'additional_structural_horizon':'ceil(0.01*N)_separate_initial_state',
  },
  'allocations_per_corpus_panel_primary_seed': {
    'structure': {'R':256,'S':256,'U':128,'A':32},
    'task_relevance': {'R':256,'S':256,'U':128,'A':32},
    'full_methods': {'R':64,'S':64,'U':32,'A':16},
    'replication': {'R':64,'S':64,'U':32,'A':0},
    'full_refit': {'R':32,'S':32,'U':0,'A':16},
    'convex_extension': {'R':32,'S':32,'U':0,'A':0},
    'each_boundary_configuration': {'R':32,'S':32,'U':0,'A':0},
  },
  'methods': {
    'O-G':'independent_fixed_curator_rebuild_and_common_decoder',
    'O-T':'correct_selected_set_fresh_moments_common_decoder',
    'B-E':'eligible_FP32_payload_antijoin_vectorized_signed_moments_same_horizon',
    'B-A':'all_current_payload_antijoin_greater_capability_disclosed',
    'P-I':'indexed_finite_horizon_statistic_summary',
    'P-S':'whole_summary_scan_ablation',
    'P-R':'rank_coordinate_scan_reconstruction',
    'B-F':'wrong_target_frozen_selection_diagnostic',
    'O-R':'separate_full_refit_branch',
  },
  'fairness': {
    'shared_solver':True,
    'same_canonical_z':True,
    'baseline_stores_outer_products_per_record':False,
    'baseline_compaction':'development_locked_masked_CSR_policy_same_rules_as_proposed_count_stale_bytes',
    'no_hidden_retained_lookup':True,
    'persistence':'final_checkpoint_primary_every_release_secondary',
    'native_failures_remain_in_results':True,
    'forbidden_claims':['value_coordinate_rank_implies_total_byte_optimality','speedup_over_retrain_implies_incremental_state_of_the_art','observed_ANN_recall_implies_uniform_graph_certificate'],
  },
  'audit': {
    'fresh_graph_paths_per_primary_corpus':16,
    'complete_state_rebuild_paths':{'R':16,'S':16},
    'full_refit_every_step_paths':{'R':2,'S':2},
    'timing_paths':{'R':8,'S':8},
    'fresh_process_timing_repetitions':5,
    'same_data_full_refit_repeats':5,
    'full_refit_seed_sensitivity_count':3,
    'model_workers_isolated_from_oracles':True,
  },
  'statistics': {
    'primary_unit':'independently_reset_trajectory_conditional_on_fixed_corpus',
    'bootstrap_replicates':10000,
    'test_uncertainty_secondary':'crossed_trajectory_and_whole_test_source_group_bootstrap',
    'primary_loss_effect':'(MSE_frozen-MSE_oracle)/MSE_predelete_report_signed',
    'prediction_effect':'RMS_frozen_vs_oracle_and_divide_by_predelete_score_SD_report_raw',
    'zero_normalizer':'undefined_no_floor',
    'exceedance_reporting_threshold':0.01,
    'threshold_is_universal_practical_margin':False,
    'loss_significance_family':'4_corpus_by_R_S_contrasts_Holm_0.05',
    'systems_significance_family':'2_corpus_equal_R_S_weighted_lifecycle_comparisons_Holm_0.05',
    'rare_failure_claim_from_64_runs':False,
    'confirmation_optional_stopping':False,
  },
  'human_audit': {
    'unique_admission_pairs_max':400,
    'excluded_control_pairs_max':200,
    'annotators_per_pair':3,
    'former_blocker_set_audit_max':100,
    'surviving_selected_comparison_audit_max':100,
    'sampling':'fixed_corpus_and_request_strata_record_inclusion_probabilities_census_scarce_strata_no_favorable_refill',
    'blinding':['method','request_family','model_effect','gold_label','desired_conclusion'],
    'separate_from_threshold_calibration':True,
  },
  'resources_proposed': {'host_GiB':256,'per_method_memory_cap_GiB':128,'accelerator_count':1,'wall_time_budget':'measured_development_projection_not_invented'},
  'pre_execution_lock_required': {
    'corpus_snapshots_and_checksums':None,
    'source_partition_hashes_and_counts':None,
    'post_guard_panel_manifest_hashes':None,
    'calibration_test_manifest_hashes':None,
    'parser_and_guard_commit':None,
    'encoder_and_tokenizer_revisions':None,
    'embedding_cache_hashes':None,
    'thresholds_and_independent_quality_audit':None,
    'tag_vocabularies_and_decision_thresholds':None,
    'selected_lambdas':None,
    'method_and_oracle_code_commits':None,
    'numerical_error_budget_and_reference_scoring_implementation':None,
    'source_service_universe_hash':None,
    'request_manifest_hashes':None,
    'hardware_and_thread_configuration':None,
    'memory_cap_and_common_timeout':None,
    'development_cost_and_precision_analysis':None,
    'full_refit_cluster_count_policy_and_seeds':None,
    'compaction_and_persistence_policies':None,
    'annotation_instructions_pay_and_content_protection':None,
  },
}
(dest/'study_design.json').write_text(json.dumps(design,indent=2)+'\n')

claims = [
 ['C1','Natural admissions','per-corpus R/S activation probability and magnitude including zeros','complete exact graph plus random requests','random/source natural population only','effect only U/A => stress result, no prevalence claim'],
 ['C2','NLP consequence','raw/normalized prediction RMS; signed relative MSE; weighted text audit','B-F versus O-G on identical fixed test data','same immutable learner and curated population','churn only or tiny effects => limited demonstrated NLP consequence'],
 ['C3','Exact scoped repair','oracle curator target; exact count/state keys; moment and numeric error bounds','independent fresh target and remaining-horizon builder','head and designated logical state, not corpus/IDs/transcript','unexplained mismatch => block exact implementation claim'],
 ['C4','Practical algorithm value','lifecycle time, physical persistent/peak memory, failure rate','optimized B-E same access/horizon/decoder','observed supported horizon and common cap','win only over rebuild => no incremental superiority; extra memory => tradeoff'],
 ['C5','Scope beyond fixed curator','F versus R selected-set/prediction gap and same-data refit floor','separate SemDeDup original branch','full-refit audit is a different target','large gap => clearly restrict applicability'],
]
with (dest/'claim_ledger.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['id','claim','primary_evidence','comparator','scope','decision_if_unfavorable']);w.writerows(claims)

columns = [
 'run_id','protocol_hash','corpus_id','corpus_snapshot_hash','panel_id','post_guard_manifest_hash','source_partition_hash','curator_contract','threshold','priority_seed','curation_encoder_sha','learner_encoder_sha','curation_dimension','learner_dimension','label_map_hash','lambda','method','method_commit','oracle_commit','request_family','trajectory_id','request_manifest_hash','checkpoint_index','initial_horizon','remaining_horizon','cumulative_units_deleted','cumulative_raw_deleted','cumulative_original_selected_deleted','raw_N','initial_selected_N','current_selected_N','admitted_N','eligible_N','coefficient_keys','structural_rank','decoder','precision','access_stratum','persistence_policy','hardware_block','timing_repetition','status','failure_reason','fallback_contract_change','setup_seconds','validate_seconds','repair_seconds','solve_seconds','release_seconds','persist_seconds','cumulative_seconds','logical_bytes','serialized_bytes','compressed_bytes','resident_bytes','peak_host_bytes','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','metadata_read_bytes','retained_payload_read_bytes','request_bytes','incidence_touches','key_merges','expired_entries','solver_iterations','oracle_curator_verified','moment_count_equal','state_audit_level','state_key_equal','moment_error_norm','normalized_solve_residual','rigorous_error_bound_available','parameter_error_bound','observed_parameter_discrepancy','test_mse','oracle_test_mse','frozen_test_mse','predelete_test_mse','raw_prediction_RMS','predelete_score_SD','relative_signed_MSE_change','normalized_prediction_RMS','head_hash','state_hash','notes'
]
with (dest/'run_registry_schema.csv').open('w',newline='') as f: csv.writer(f).writerow(columns)
with (dest/'annotation_schema.csv').open('w',newline='') as f:
    csv.writer(f).writerow(['item_id','corpus','sampling_frame_hash','hidden_request_stratum','inclusion_probability','record_id','comparison_record_id','annotator_id','semantic_category','entity_number_negation_time_distinction','task_relevant_distinction','uncertain','skipped','time_seconds','adjudication'])

readme = '''EMPIRICAL PROGRAM — ACL 2027 COUNTERFACTUAL CURATION
Version 1.0 | 3 October 2026

This is a reviewed proposed design, not measured results, a registered study,
or an implemented training/repair runner. No synthetic dataset is included.
The PDF is the narrative authority. The JSON is the operational study manifest.
The original theory specification remains separate and unchanged.

FILES
  study_design.json: decisions, request allocations, contracts and lock fields.
  claim_ledger.csv: what each claim needs and what unfavorable findings permit.
  run_registry_schema.csv: header-only schema; no fabricated example runs.
  annotation_schema.csv: header-only annotation schema; no fabricated judgments.
  memory_preflight.py: analytic byte estimates from measured structural counts.
  validate_design.py: draft checks, release allocation counts, and lock validation.
  counterfactual_curation_empirical_protocol.tex: complete editable PDF source.
  SHA256SUMS.txt: integrity hashes for package contents (excluding this hash file).

START
  python3 validate_design.py study_design.json
  python3 validate_design.py study_design.json --lock
The first checks internal design constraints and prints allocations. The second
MUST fail until acquisition/development outputs populate all lock fields. Fill
them with substantive resolved values, then perform a human lock review. A passed
validator checks manifest completeness/invariants, not scientific truth or that
an experiment was actually preregistered.

ANALYTIC PREFLIGHT (replace values with measured natural-graph counts)
  python3 memory_preflight.py --dimension 768 --outputs 20 \\
      --keys ACTUAL_KEYS --eligible ACTUAL_ELIGIBLE --rank ACTUAL_RANK \\
      --metadata-bytes ACTUAL_METADATA --workspace-bytes ACTUAL_WORKSPACE
Integer placeholders above are deliberately not executable data. Supply actual
counts from a natural corpus. The calculator includes packed dense FP64 statistics
and an FP32 eligible-row comparison; metadata/workspace inputs must be measured.
It is not an algorithm implementation or runtime predictor. Missing metadata
cannot establish feasibility or a physical-memory claim.

CORE EXECUTION ORDER
1. Acquire originals; pin checksums/schema/record identity; define the source
   partition and exact split/guard manifests. Keep all unfavorable outcomes.
2. Calibrate only on source-disjoint development data; validate semantic quality.
   Lock canonical features, scorer, threshold, lambda, labels and request seeds.
3. Build exhaustive natural graphs and forecast native-dimensional state bytes.
   Keep forecast-infeasible and OOM cells, separately labeled, in results.
4. Implement independent oracle and optimized eligible-payload B-E first; verify
   natural-data correctness, numerical bounds and worker access isolation.
5. Run locked confirmation and log every failure. No filtering after outcomes.
6. Run independent annotation, full-refit and convex panels; apply claim ledger.

KEY BOUNDARIES
  * Primary global ordered suppression is not complete standard SemDeDup.
  * Core release is head-only. An isolated oracle computes selected IDs; the
    proposed statistic summary is not assumed to recover IDs or document text.
  * Source service supports all partition units; random S samples genuine native
    sources only. Identical eligibility universe/horizon applies to every method.
  * Civil article groups are not authors. Stack owner IDs are account proxies.
  * Unknown sources are separate record singletons, never one pooled source.
  * Finite-horizon state is rebuilt only explicitly and with full cost/access.
  * Native d=768 results cannot disappear behind projected d=128 results.
  * Residual diagnostics are not rigorous interval certificates without verified
    accumulation and residual-evaluation error allowances.
  * 256 independent requests with zero events still allow a one-sided95% event
    probability upper bound about1.16%;64 allow about4.57%.
  * Hyperparameters/encoder pretraining remain outside deletion-sensitive D.
  * Raw text redistribution follows original source conditions. Release IDs,
    hashes and acquisition code where redistribution rights are not established.

All engineering changes after protocol lock require versioned amendments and
reruns of affected comparisons. This design cannot eliminate legitimate debugging
or guarantee reviewer acceptance; it prevents target drift and selective evidence.
'''
(dest/'README.txt').write_text(readme)
shutil.copy2(root/'output/pdf/counterfactual_curation_empirical_protocol.tex', dest/'counterfactual_curation_empirical_protocol.tex')

def finish():
    entries=[]
    for p in sorted(dest.iterdir()):
        if p.is_file() and p.name!='SHA256SUMS.txt':
            entries.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
    (dest/'SHA256SUMS.txt').write_text('\n'.join(entries)+'\n')
    out=root/'output/counterfactual_curation_empirical_program.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.iterdir()):
            if p.is_file(): z.write(p,arcname='empirical_program/'+p.name)
    print(f'Packaged {len(entries)+1} files: {out}')

if __name__=='__main__': finish()
