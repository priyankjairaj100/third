#!/usr/bin/env python3
"""Natural-data convex integration; exact math fixtures are software checks only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,hashlib,math
from fractions import Fraction
from decimal import Decimal,localcontext
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from phase3.reference_graph import build_reference_graph
from phase4.requests import verify_manifest
from phase4.execution import write,append
from phase4.convex import (solve_fractional_logistic,certify_logistic,signed_warm_gradient,
 objective_gradient_hessian,sigmoid_interval,CertificateBudgetError)


def frac(value):return Fraction(int(value['numerator']),int(value['denominator']))
def without_weights(solved):return {k:v for k,v in solved.items() if k!='weights'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'phase4/results/convex_engineering');args=parser.parse_args()
    out=args.output_dir
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refusing to overwrite a frozen convex run')
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/civil_comments_engineering_preview.jsonl';rows=read_natural_jsonl(source,require_labels=True)
    ids=[r['record_id'] for r in rows];lookup={v:i for i,v in enumerate(ids)}
    x,feature_meta=lexical_engineering_features(rows,64);cx,curator_meta=lexical_engineering_features(rows,128)
    y=np.asarray([r['label'] for r in rows],dtype=np.float64)
    request_path=ROOT/'phase4/results/execution_engineering_final/d64/requests.json'
    manifest=json.loads(request_path.read_text());graph=build_reference_graph(cx,ids,.6);verify_manifest(manifest,graph)
    paths=[next(p for p in manifest['trajectories'] if p['arm']==arm) for arm in ['R','U','A']]
    locked={'scope':'reused Civil100 lexical engineering only; no synthetic empirical data',
      'semantic_encoder_used':False,'confirmatory_study_ready':False,'records':len(rows),'feature':feature_meta,'curator':curator_meta,
      'paths':[p['trajectory_id'] for p in paths],'checkpoints':[1,2,4,8],
      'learner':'single-output fractional toxicity logistic; full-coordinate regularization, no intercept',
      'lambda':.01,'numeric_gradient_tolerance':1e-11,'max_iterations':100,
      'certificate_parameter_error_tolerance':'1e-8','certificate_bits':128,
      'cold_start':'zero weights at every repaired selected set','warm_start':'previous certified head in same trajectory',
      'full_data_evaluations_required_by_both_solvers':True,'speedup_claim':False,
      'source_sha256':sha256_file(source),'request_manifest_sha256':manifest['manifest_sha256'],
      'request_file_sha256':sha256_file(request_path),
      'code_sha256':{name:sha256_file(ROOT/name) for name in ['phase4/convex.py','phase4/check_convex.py','phase4/requests.py','phase3/reference_graph.py','phase3/panels.py','ccu/certified_ridge.py']}}
    write(out/'design_lock.json',locked)
    for name in ['checkpoints.jsonl','heads.jsonl','certificates.jsonl']:(out/name).write_text('')
    initial=set(map(int,graph.selected_indices()));initial_ix=np.asarray(sorted(initial),dtype=int)
    build=solve_fractional_logistic(x[initial_ix],y[initial_ix],.01)
    initial_certificate=certify_logistic(x[initial_ix],y[initial_ix],.01,build['weights'])
    write(out/'initial_fit.json',{**without_weights(build),'certificate':initial_certificate,'selected_record_ids':[ids[i] for i in initial_ix]})
    if build['status']!='numerically_converged' or not initial_certificate['meets_parameter_tolerance']:
        raise RuntimeError('Initial candidate failed solver/certificate requirements; no reroll')
    results=[];certificates=[];failures=0
    for path in paths:
        old_selected=initial.copy();warm=build['weights'].copy();failed=False
        for k in path['checkpoints']:
            context={'trajectory_id':path['trajectory_id'],'arm':path['arm'],'checkpoint':k}
            if failed:
                row={**context,'status':'not_executed_after_path_failure'};append(out/'checkpoints.jsonl',row);results.append(row);continue
            try:
                dead={lookup[v] for v in path['deletion_order'][:k]}
                alive=np.asarray([i for i in range(len(ids)) if i not in dead],dtype=int)
                started=time.perf_counter();fresh=build_reference_graph(cx[alive],[ids[i] for i in alive],.6)
                selected={int(alive[j]) for j in fresh.selected_indices()};curation_seconds=time.perf_counter()-started
                expected=next(o['selected_record_ids'] for o in path['observations'] if o['checkpoint']==k)
                if {ids[i] for i in selected}!=set(expected):raise AssertionError('Fresh curation differs from frozen request audit')
                ix=np.asarray(sorted(selected),dtype=int);old_ix=np.asarray(sorted(old_selected),dtype=int)
                removed=np.asarray(sorted(old_selected-selected),dtype=int);added=np.asarray(sorted(selected-old_selected),dtype=int)
                _,old_g,_=objective_gradient_hessian(x[old_ix],y[old_ix],warm,.01,hessian=False)
                _,new_g,_=objective_gradient_hessian(x[ix],y[ix],warm,.01,hessian=False)
                signed=signed_warm_gradient(old_g,warm,.01,len(old_ix),len(ix),x[removed],y[removed],x[added],y[added])
                gradient_identity_error=float(np.max(np.abs(signed-new_g),initial=0))
                if gradient_identity_error>1e-12:raise AssertionError('Average-loss signed gradient identity failed')
                stale=signed-(1-len(old_ix)/len(ix))*.01*warm if len(ix) else signed
                stale_error=float(np.linalg.norm(stale-new_g))
                # Deterministic alternating ordering mitigates always-second warm runs;
                # this is still one-process diagnostic timing, not a systems study.
                methods={};certs={}
                order=['cold','warm'] if k in [1,4] else ['warm','cold']
                for method in order:
                    result=solve_fractional_logistic(x[ix],y[ix],.01,warm_start=warm if method=='warm' else None)
                    methods[method]=result
                    if result['status'] not in ['numerically_converged','empty_target_exact_zero']:
                        raise RuntimeError(f'{method} solver status {result["status"]}')
                    cert=certify_logistic(x[ix],y[ix],.01,result['weights']);certs[method]=cert
                    append(out/'certificates.jsonl',{**context,'method':method,'certificate':cert})
                    certificates.append(cert)
                    if not cert['meets_parameter_tolerance']:raise AssertionError(f'{method} failed exact residual certificate tolerance')
                difference=methods['warm']['weights']-methods['cold']['weights']
                exact_distance_sq=sum((Fraction.from_float(float(a))-Fraction.from_float(float(b)))**2
                  for a,b in zip(methods['warm']['weights'],methods['cold']['weights']))
                radius_sum=frac(certs['warm']['parameter_error_norm_upper'])+frac(certs['cold']['parameter_error_norm_upper'])
                if exact_distance_sq>radius_sum**2:raise AssertionError('Head discrepancy exceeds sum of certified radii')
                row={**context,'status':'completed_engineering','selected_count':len(ix),
                  'selected_record_ids':[ids[i] for i in ix],
                  'step_removed_selected_ids':[ids[i] for i in removed],'step_admitted_ids':[ids[i] for i in added],
                  'total_admissions_from_initial':len(selected-initial),'curation_seconds':curation_seconds,
                  'signed_gradient_identity_max_abs_error':gradient_identity_error,
                  'omitting_count_change_regularizer_gradient_l2_error':stale_error,
                  'signed_gradient_audit_rows':len(old_ix)+len(ix)+len(removed)+len(added),
                  'solver_order':order,'methods':{m:without_weights(v) for m,v in methods.items()},
                  'exact_cold_warm_distance_bounded_by_certificate_radii':True,
                  'cold_warm_head_l2_difference':float(np.linalg.norm(difference)),
                  'certificate_radii_display':{m:c['radius_float_for_display_only'] for m,c in certs.items()},
                  'certificate_seconds':{m:c['elapsed_seconds'] for m,c in certs.items()},
                  'certificate_work':{m:c['work'] for m,c in certs.items()}}
                append(out/'checkpoints.jsonl',row);results.append(row)
                append(out/'heads.jsonl',{**context,'weights':{m:v['weights'].tolist() for m,v in methods.items()}})
                warm=methods['warm']['weights'].copy();old_selected=selected
            except Exception as error:
                failures+=1;failed=True
                row={**context,'status':'failed','error_type':type(error).__name__,'error':str(error)}
                append(out/'checkpoints.jsonl',row);results.append(row)
    checks={}
    # Algebraic/numerical software controls, never empirical datasets.
    with localcontext() as ctx:
        ctx.prec=150
        for q in [Fraction(0),Fraction(1,4),Fraction(-1,4),Fraction(1),Fraction(-1),Fraction(8),Fraction(-8),Fraction(64),Fraction(-64)]:
            lo,hi,_=sigmoid_interval(q);neg_lo,neg_hi,_=sigmoid_interval(-q)
            if lo!=1-neg_hi or hi!=1-neg_lo:raise AssertionError('Sigmoid sign symmetry enclosure failed')
            value=Decimal(q.numerator)/Decimal(q.denominator);p=1/(1+(-value).exp())
            if not Decimal(lo.numerator)/Decimal(lo.denominator)<=p<=Decimal(hi.numerator)/Decimal(hi.denominator):
                raise AssertionError('High-precision numerical sigmoid crosscheck outside rational enclosure')
    checks['sigmoid_symmetry_and_150_digit_crosscheck']=9
    empty_x=np.empty((0,64),dtype=np.float32);empty_y=np.empty(0,dtype=np.float64)
    empty=solve_fractional_logistic(empty_x,empty_y,.01,warm_start=np.ones(64,dtype=np.float64))
    empty_cert=certify_logistic(empty_x,empty_y,.01,np.ones(64,dtype=np.float64))
    if np.any(empty['weights']) or frac(empty_cert['parameter_error_squared_upper'])!=64:
        raise AssertionError('Empty objective/certificate convention failed')
    checks['empty_target_zero_solver_and_exact_nonzero_candidate_radius']=True
    for name,call in [
      ('coordinate_cap',lambda:certify_logistic(x[:2],y[:2],.01,np.zeros(64),max_coordinates=1)),
      ('term_cap',lambda:sigmoid_interval(Fraction(1,2),max_terms=1)),
      ('squaring_cap',lambda:sigmoid_interval(Fraction(2),max_squarings=0))]:
        try:call()
        except CertificateBudgetError:checks[name+'_refused']=True
        else:raise AssertionError('Certificate budget failed to stop: '+name)
    successful=[r for r in results if r['status']=='completed_engineering']
    if not any(r['omitting_count_change_regularizer_gradient_l2_error']>1e-8 for r in successful):
        raise AssertionError('Natural fixture did not exercise changed-count normalization control')
    checks['natural_count_shift_negative_control_detected']=True
    summary={'status':'passed' if not failures else 'failed_with_preserved_rows','checkpoint_rows':len(results),
      'completed_checkpoint_rows':len(successful),'failure_events':failures,
      'certificates_including_initial':1+len(certificates),'rigorous_optimizer_error_certificates':True,
      'max_certified_parameter_radius_display':max(c['radius_float_for_display_only'] for c in certificates+[initial_certificate]),
      'max_cold_warm_l2_difference':max((r['cold_warm_head_l2_difference'] for r in successful),default=None),
      'zero_total_admission_checkpoint_rows':sum(r['total_admissions_from_initial']==0 for r in successful),
      'checks':checks,'statistical_unlearning_certificate':False,'semantic_study_result':False,'speedup_claim':False,
      'canonical_model_or_state_bytes_claim':False,'no_reaccess_claim':False,'source_service_tested':False,
      'scope':'single-output fractional toxicity; Stack multioutput extension not implemented',
      'design_lock_sha256':sha256_file(out/'design_lock.json')}
    write(out/'convex_checks.json',summary);print(json.dumps(summary,indent=2))
    if failures:raise RuntimeError('Convex candidate failures preserved; not all checks passed')

if __name__=='__main__':main()
