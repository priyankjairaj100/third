"""Check prospective recipes. Algebraic examples are software tests only."""
from pathlib import Path
import argparse, json, sys, tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase6 import recipes as r
from phase5 import study_registry as old

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def refuses(name,fn):
        try:fn()
        except (ValueError,FileExistsError):check(name,True)
        else:raise AssertionError(name)
    book=r.build_recipe_book();reg,jobs=r.build_registry();legacy,prior=old.build_registry()
    kept=[j for j in prior if j.get('kind')!='unresolved_protocol_obligation']
    check('all21_prior_obligations_resolved',set(x['id'] for x in book['recipes'])==set(x[0] for x in old.UNRESOLVED))
    check('all16912_legacy_jobs_unchanged',jobs[:len(kept)]==kept and len(kept)==16912)
    check('all21332_job_IDs_unique',len(jobs)==21332 and len({j['job_id'] for j in jobs})==len(jobs))
    check('4420_concrete_extension_jobs',reg['new_recipe_job_count']==4420)
    check('content_digest_recomputes',reg['jobs_content_sha256']==r.digest(jobs))
    check('no_remaining_placeholders',not any(j.get('kind')=='unresolved_protocol_obligation' for j in jobs))
    check('no_scientific_acceptance_fabricated',all(j['execution_allowed'] is False for j in jobs) and not book['scientific_acceptance_granted'])
    check('every_obligation_has_rule',all(x['policy'] and x['executor'] for x in book['recipes']))
    lookup={x['id']:x for x in book['recipes']}
    check('optional_Civil_explicitly_deactivated',lookup['official_Civil_splits_optional']['policy']['activated'] is False)
    check('all8_FP32_variants_required',lookup['boundary_method_scope']['policy']['FP32_required_variants']==r.METHODS)
    check('RSS_sweep_no_false_bound','not_bound' in lookup['worker_dependency_and_IO_accounting']['policy']['sampled_peak'])
    check('utility_hash_matches_frozen_executor',lookup['utility_reference']['policy']['hash_salt']=='ccu-v1-utility-hash')
    check('fresh_audit48',reg['recipe_job_counts']['fresh_graph_audit']==48)
    check('human_threshold12_includes_sensitivities',reg['recipe_job_counts']['human_threshold_calibration']==12)
    check('human_context_two_distinct',reg['recipe_job_counts']['human_admission_context']==2)
    check('weighted_source_matched_counts',sum(j.get('arm')=='S-PPS' for j in jobs)==768 and sum(j.get('arm')=='R-volume-matched-to-S-PPS' for j in jobs)==768)
    check('each_timing_policy160',all(reg['recipe_job_counts'][k]==160 for k in ('cold_warm_scope','common_cached_graph_oracle','every_release_persistence')))
    class Ticket:
        def __init__(self,ticket):self.ticket=ticket
        def randrange(self,total):assert 0<=self.ticket<total;return self.ticket
    first=[]
    for ticket in range(3):
        with patch.object(r.random,'Random',return_value=Ticket(ticket)):
            first.append(r.pps_order_exact_integer_tickets({'a':2,'b':1},draw_count=1,seed_value=0)[0])
    check('PPS_exact_integer_mass_law',first==['a','a','b'])
    a=r.pps_order_exact_integer_tickets({'a':3,'b':2,'c':1},draw_count=5,seed_value=51)
    b=r.pps_order_exact_integer_tickets({'c':1,'b':2,'a':3},draw_count=5,seed_value=51)
    check('PPS_order_invariant_without_replacement',a==b and len(a)==len(set(a))==3)
    check('PPS_empty_retained',r.pps_order_exact_integer_tickets({},draw_count=8,seed_value=1)==[])
    refuses('PPS_rejects_zero_mass',lambda:r.pps_order_exact_integer_tickets({'x':0},draw_count=1,seed_value=0))
    refuses('PPS_rejects_boolean_count',lambda:r.pps_order_exact_integer_tickets({'x':1},draw_count=True,seed_value=0))
    check('one_percent_independent_endpoint',r.one_percent_horizon(200000)=={'initial_horizon':2000,'feasible_horizon':2000,'checkpoints':[1,8,32,128,2000],'status':'ready'})
    check('one_percent_clip_deduplicates',r.one_percent_horizon(10000,3)['checkpoints']==[1,3])
    check('one_percent_empty_status',r.one_percent_horizon(0)['status']=='empty_request_frame')
    refuses('one_percent_invalid_frame',lambda:r.one_percent_horizon(10,11))
    snap={'physical_available_bytes':10000,'cgroup_remaining_bytes':6000,'local_cpu_count':9}
    policy=r.derive_resource_policy(snap,[2.,5.,3.])
    check('resource_actual_minimum_common',policy['per_method_address_space_limit_bytes']==3000 and policy['wall_seconds_each_stage']==20)
    check('resource_no_paid_or_native_shrink',not policy['external_compute_authorized'] and not policy['native_dimension_reduction'])
    refuses('resource_missing_costs',lambda:r.derive_resource_policy(snap,[]))
    refuses('resource_nonfinite',lambda:r.derive_resource_policy(snap,[float('nan')]))
    dev=[f'dev{i}' for i in range(16)];confirmation=['future']
    refuses('precision_rejects_overlap',lambda:r.precision_analysis([.1]*16,development_ids=dev,confirmation_ids=dev,simulations=1000))
    refuses('precision_no_dropped_nan',lambda:r.precision_analysis([.1]*15+[float('nan')],development_ids=dev,confirmation_ids=confirmation,simulations=1000))
    flat=r.precision_analysis([0.]*16,development_ids=dev,confirmation_ids=confirmation,simulations=1000,candidate_counts=(64,256))
    check('zero_variance_no_false_power_acceptance',flat['status']=='development_degenerate_estimation_only' and flat['smallest_candidate_meeting_targets'] is None)
    variable=r.precision_analysis([(-1)**i*.1 for i in range(16)],development_ids=dev,confirmation_ids=confirmation,simulations=1000,candidate_counts=(64,256))
    check('precision_no_auto_sample_amendment',variable['allocation_actually_changed'] is False and variable['registered_default']==256)
    check('precision_all_candidate_outcomes_saved',len(variable['rows'])==2 and all(0<=v['Monte_Carlo_one_sided95_lower']<=v['empirical_shifted_power']<=1 for v in variable['rows']))
    cost={'rows':100,'dimension':768,'outputs':1,'feature_density':1.,'seconds':.2,
          'peak_rss_bytes':50_000_000,'source_sha256':'0'*64,'input_bytes':100*(4*768+8)}
    # These arithmetic costs are schema fixtures, not development measurements.
    rp={'wall_seconds_each_stage':100,'per_method_address_space_limit_bytes':200_000_000}
    budget=r.select_verification_budget([cost],rows=10000,dimension=768,outputs=1,resource_policy=rp)
    check('larger_budget_requires_dense_matching_costs',budget['forecast_feasible'] and budget['max_coordinates']==7680000)
    check('forecast_not_observed_native_success',not budget['observed_native_feasibility'] and not budget['observed_timeout'])
    refuses('sparse_cost_cannot_qualify_dense_native',lambda:r.select_verification_budget([{**cost,'feature_density':.2}],rows=10000,dimension=768,outputs=1,resource_policy=rp))
    refuses('wrong_output_count_cannot_qualify_Stack',lambda:r.select_verification_budget([cost],rows=10000,dimension=768,outputs=20,resource_policy=rp))
    failed=r.select_verification_budget([cost],rows=10000,dimension=768,outputs=1,resource_policy={**rp,'wall_seconds_each_stage':60})
    check('native_forecast_refusal_is_not_timeout',not failed['forecast_feasible'] and not failed['observed_timeout'])
    check('base_checkpoint128_retained_in_large_horizon',r.one_percent_horizon(200000)['checkpoints']==[1,8,32,128,2000])
    check('CG_policy_has_no_remaining_recipe_placeholder',all(g['configuration'].get('stopping_rule')!='development_lock_required' for g in reg['groups']))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'export';r.export(path)
        refuses('immutable_export',lambda:r.export(path))
        check('saved_book_matches',json.loads((path/'recipes.json').read_text())==book)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'passed','check_count':len(checks),'checks':checks,'recipe_code_sha256':old.sha(r.__file__),
            'scope':'algebraic_and_schema_software_checks_not_empirical_data','actual_human_responses':0,
            'primary_study_started':False,'registry_jobs':len(jobs)}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
