"""Tiny algebraic producer checks; no dataset benchmark or resource qualification."""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from phase6.run_isolated import prepare_bundle
from phase9 import resource_measurement as rm
from phase3.reference_graph import build_reference_graph


def run(destination):
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False)
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append({'name':name,'passed':True})
    def rejected(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):check(name,True)
        else:check(name,False)
    x=np.asarray([[1,.1],[.9,.2],[-1,.1],[-.9,.2]],dtype=np.float32)
    y=np.full((4,1),.5,dtype=np.float64);ids=['fixture-a','fixture-b','fixture-c','fixture-d']
    package=prepare_bundle(x,x,y,ids,out/'inputs',threshold=.8,priority_seed=0)
    graph=build_reference_graph(x,ids,.8,ids,seed=0)
    removed=ids[int(graph.selected_indices()[0])]
    selected=sorted(map(int,graph.selected_indices([removed])))
    policy={'memory_bytes':2*1024**3,'cpu_seconds':30,'wall_seconds':30,'threads':1,
            'convex_certificate_coordinates':100,'convex_verifier':'exact_dyadic_integer_v1'}
    common={'kind':'graph','method':None,'curator':{'name':'software','id':'algebraic','revision':'1'},
        'learner':{'name':'software','id':'algebraic','revision':'1'},'curator_dimension':2,
        'learner_dimension':2,'outputs':1,'projection_sha256':None,'projection_seed':None,
        'threshold':.8,'lambda_reg':.1,'priority_seed':0,'unit':'record','horizon':1,
        'checkpoint_schedule':[1],'threads':1,'fresh_graph':False,'request_arm':'R'}
    request={'schema':rm.REQUEST_SCHEMA,'kind':'graph','profile_signature':common,
        'input_package':'inputs','policy':policy,'evidence_role':'software_only',
        'trajectory':{'trajectory_id':'software-one','unit':'record','arm':'R','deletion_order':[removed],'checkpoints':[1]},
        'source_kinds':{v:'unknown_singleton' for v in ids},'fresh_graph':False}
    rm.write(out/'graph_request.json',request)
    result=rm.collect(out/'graph_request.json',out/'graph')
    check('actual_algebraic_graph_route_completed',result['status']=='completed')
    check('software_role_never_claims_primary',result['evidence_role']=='software_only' and result['primary_acceptance'] is False)
    rows=json.loads((out/'graph/structure.json').read_text())
    check('graph_output_has_exact_request_and_rows',len(rows)==1 and rows[0]['selected_ids']==[ids[i] for i in selected] and rows[0]['checkpoint']==1)
    check('graph_worker_is_frozen_route',result['worker_result']['measured_operation']=='frozen_structural_job_with_input_graph_reconstruction')
    check('complete_measured_process_fields',all(result['process'][k]>=0 for k in ['spawn_to_reap_seconds','peak_rss_bytes','sampled_process_tree_peak_rss_sum_bytes','user_cpu_seconds','system_cpu_seconds']))
    check('source_and_input_frozen',result['source_and_input_unchanged'] and result['source_sha256']==rm.sources() and result['producer_source_sha256']==rm.producer_sources())
    check('invocation_and_configuration_bound',result['invocation_sha256']==rm.sha(out/'graph/invocation.json') and result['configuration_sha256']==rm.sha(out/'graph/prospective_configuration.json'))
    check('all_outputs_hash_bound',all(rm.sha(out/'graph'/p)==h for p,h in result['artifact_hashes'].items()))
    rejected('existing_output_preserved',lambda:rm.collect(out/'graph_request.json',out/'graph'))
    bad=copy.deepcopy(request);bad['trajectory']['deletion_order']*=2
    rejected('duplicate_actual_graph_ids_rejected',lambda:rm.validate_request(bad))
    bad=copy.deepcopy(request);bad['profile_signature']['checkpoint_schedule']=[2]
    rejected('incomplete_schedule_rejected',lambda:rm.validate_request(bad))
    bad=copy.deepcopy(request);bad['fresh_graph']=True
    rejected('fresh_route_retag_rejected',lambda:rm.validate_request(bad))
    bad=copy.deepcopy(request);bad['trajectory']['arm']='U'
    rejected('graph_arm_retag_rejected',lambda:rm.validate_request(bad))
    bad=copy.deepcopy(request);del bad['profile_signature']['fresh_graph']
    rejected('missing_fresh_route_binding_rejected',lambda:rm.validate_request(bad))
    bad=copy.deepcopy(request);bad['evidence_role']='primary_semantic'
    rejected('invalid_evidence_role_rejected',lambda:rm.validate_request(bad))
    np.savez(out/'candidate.npz',x=x[selected],y=y[selected],w=np.zeros((2,1),dtype=np.float64))
    signature={**common,'kind':'convex','unit':None,'horizon':None,'checkpoint_schedule':None,
        'convex_backend':'exact_dyadic_integer_v1','parameter_tolerance':'1/100000000',
        'sigmoid_bits':128,'sigmoid_max_terms':256,'sigmoid_max_squarings':64,
        'scope':'pointwise_dense_exact_verifier_only'}
    convex={'schema':rm.REQUEST_SCHEMA,'kind':'convex','profile_signature':signature,
        'input_package':'inputs','policy':policy,'evidence_role':'software_only',
        'candidate':{'path':'candidate.npz','sha256':rm.sha(out/'candidate.npz')},
        'selection':{'unit':'record','deleted_unit_ids':[removed],'selected_record_ids':[ids[i] for i in selected]}}
    rm.write(out/'convex_request.json',convex)
    verified=rm.collect(out/'convex_request.json',out/'convex')
    check('actual_pointwise_dyadic_route_completed',verified['status']=='completed' and verified['meets_parameter_tolerance'] is True)
    check('candidate_and_certificate_exact_bytes_bound',verified['candidate_sha256']==convex['candidate']['sha256'] and verified['certificate_sha256']==rm.sha(out/'convex/certificate.json'))
    check('pointwise_scope_without_optimization',verified['worker_result']['optimization_performed'] is False and verified['worker_result']['trajectory_execution_claim'] is False)
    check('positive_verifier_time_within_process',0<verified['certificate_seconds']<=verified['process']['spawn_to_reap_seconds'])
    bad=copy.deepcopy(convex);bad['profile_signature']['parameter_tolerance']='1/100'
    rejected('weakened_certificate_tolerance_rejected',lambda:rm.validate_request(bad))
    refused=copy.deepcopy(convex);refused['policy']['convex_certificate_coordinates']=0
    rm.write(out/'cap_request.json',refused)
    failed=rm.collect(out/'cap_request.json',out/'cap_failure')
    check('actual_cap_failure_retained',failed['status']=='observed_measurement_failure' and failed['process']['exit_code']!=0)
    check('failed_verifier_never_reports_certificate',not (out/'cap_failure/certificate.json').exists() and 'certificate_sha256' not in failed)
    check('failed_process_logs_bound','stderr.txt' in failed['artifact_hashes'] and bool((out/'cap_failure/stderr.txt').read_text()))
    rm._prepare_output(out/'private_guard_only','actual_disjoint_development')
    check('actual_output_privacy_guard_written_without_execution',(out/'private_guard_only/.gitignore').read_text().endswith('*\n!.gitignore\n'))
    from unittest.mock import patch
    # This explicit non-observation mock only exercises error preservation.
    sentinel={'exit_code':0,'timeout':False,'spawn_to_reap_seconds':0,
              'software_mock_not_observation':True}
    with patch.object(rm,'_isolated',return_value=sentinel), patch.object(rm,'package_hashes',side_effect=[rm.package_hashes(package),ValueError('mock post-run changed package')]):
        mutation=rm.collect(out/'graph_request.json',out/'post_mutation_mock')
    check('post_run_integrity_failure_retains_measurement',mutation['status']=='observed_measurement_failure' and mutation['source_and_input_unchanged'] is False and (out/'post_mutation_mock/measurement.json').exists())
    check('post_run_integrity_failure_has_reason',mutation['integrity_errors'][0]['stage']=='post_execution_binding' and mutation['process']['software_mock_not_observation'] is True)
    report={'schema':'ccu-resource-measurement-checks-1','evidence_role':'software_only',
        'passed':True,'checks':checks,'check_count':len(checks),'primary_resource_observations':0,
        'empirical_benchmark_performed':False,'scope':'four-row algebraic software fixtures, two positive subprocess routes, one required cap refusal, and one explicit non-observation integrity mock',
        'source_sha256':rm.sources(),'producer_source_sha256':rm.producer_sources(),
        'checker_sha256':rm.sha(__file__),'artifact_sha256':{name:rm.sha(out/name/'measurement.json') for name in ['graph','convex','cap_failure']}}
    rm.write(out/'checks.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();report=run(args.out);print(json.dumps({'passed':report['passed'],'checks':report['check_count']}))


if __name__=='__main__':main()
