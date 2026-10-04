#!/usr/bin/env python3
"""Seven ORIGINAL Civil fractional outcomes. Binary20 cases are software tests only."""
from pathlib import Path
import argparse,copy,json,sys,time,tempfile
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from phase3.reference_graph import build_reference_graph
from phase4.requests import generate_manifest,digest
from phase4.convex import CertificateBudgetError
from phase5.convex_multioutput import *

TARGETS=['toxicity','severe_toxicity','obscene','threat','insult','identity_attack','sexual_explicit']

def without_heads(result):return {k:v for k,v in result.items() if k!='weights'}
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'phase5/results/convex_multioutput_engineering')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/civil_comments_engineering_preview.jsonl';rows=read_natural_jsonl(source)
    y=np.asarray([[r['fields'][t] for t in TARGETS] for r in rows],dtype=np.float64)
    x,_=lexical_engineering_features(rows,64);cx,_=lexical_engineering_features(rows,128)
    ids=[r['record_id'] for r in rows];lookup={r:i for i,r in enumerate(ids)};graph=build_reference_graph(cx,ids,.6)
    manifest=json.loads((ROOT/'phase4/results/execution_engineering_final/d64/requests.json').read_text())
    bindings=dict(source_sha256=sha256_file(source),request_manifest_sha256=manifest['manifest_sha256'],
      features_sha256=scalar._hash_array(x),targets_sha256=scalar._hash_array(y),lambda_hex=(.01).hex(),
      code_sha256=sha256_file(Path(__file__)),implementation_sha256=sha256_file(Path(__file__).with_name('convex_multioutput.py')))
    write(out/'design_lock.json',dict(input_bindings=bindings,evidence_role='engineering_nonconfirmatory',
      targets=TARGETS,records=len(rows),feature='lexical64',curator='lexical128',threshold=.6,
      checkpoints=[1,8],arms=['R','U','A'],source_arm_natural='unavailable; no native source IDs',
      normalization='sum over outputs of row-mean BCE + lambda/2 Frobenius squared',
      lambda_reg=.01,release_frobenius_radius=1e-8,semantic_encoder_used=False,
      Stack20_actual_data_used=False,source_claim=False,speedup_claim=False))
    initial=sorted(ids[i] for i in graph.selected_indices());initial_ix=np.array([lookup[r] for r in initial])
    fit=solve_multioutput(x[initial_ix],y[initial_ix],.01)
    cert=certify_multioutput(x[initial_ix],y[initial_ix],.01,fit['weights'])
    write(out/'initial_fit.json',dict(fit=without_heads(fit),weights=fit['weights'].tolist(),certificate=cert,selected_ids=initial))
    assert cert['meets_parameter_tolerance'];certs=[cert];results=[];saved=[];checks=[]
    for arm in ['R','U','A']:
        path=next(p for p in manifest['trajectories'] if p['arm']==arm);old=initial;warm=fit['weights'];previous_deleted=[]
        for k in [1,8]:
            selected,deleted=retained_selection(graph,manifest,path['trajectory_id'],k)
            ix=np.array([lookup[r] for r in selected],dtype=int);oi=np.array([lookup[r] for r in old],dtype=int)
            ri=np.array([lookup[r] for r in sorted(set(old)-set(selected))],dtype=int)
            ai=np.array([lookup[r] for r in sorted(set(selected)-set(old))],dtype=int)
            _,og=objective_gradient(x[oi],y[oi],warm,.01);_,ng=objective_gradient(x[ix],y[ix],warm,.01)
            sg=signed_gradient(og,warm,.01,len(oi),len(ix),x[ri],y[ri],x[ai],y[ai])
            error=float(np.max(np.abs(sg-ng),initial=0));assert error<1e-12
            cold=solve_multioutput(x[ix],y[ix],.01);repair=solve_multioutput(x[ix],y[ix],.01,warm_start=warm)
            cc=certify_multioutput(x[ix],y[ix],.01,cold['weights']);rc=certify_multioutput(x[ix],y[ix],.01,repair['weights'])
            assert cc['meets_parameter_tolerance'] and rc['meets_parameter_tolerance'];certs.extend([cc,rc])
            distance_sq=sum(((Fraction.from_float(float(a))-Fraction.from_float(float(b)))**2
              for a,b in zip(cold['weights'].flat,repair['weights'].flat)),Fraction(0))
            upper=fraction(cc['parameter_error_frobenius_upper'])+fraction(rc['parameter_error_frobenius_upper'])
            assert distance_sq<=upper**2
            resume=out/f'resume_{arm}_{k}.json'
            persisted=save_resume(resume,repair['weights'],selected_ids=selected,deleted_ids=deleted,input_bindings=bindings)
            restored,read_meter=load_resume(resume,input_bindings=bindings,expected_selected_ids=selected,expected_deleted_ids=deleted)
            assert np.array_equal(restored,repair['weights'])
            row=dict(arm=arm,trajectory_id=path['trajectory_id'],checkpoint=k,selected_ids=selected,deleted_ids=deleted,
              removed_selected=len(ri),admitted=len(ai),signed_gradient_max_error=error,
              cold=without_heads(cold),warm=without_heads(repair),certificates=dict(cold=cc,warm=rc),
              resume=dict(write=persisted,read=read_meter),certificate_distance_check_passed=True)
            results.append(row);saved.append(dict(arm=arm,checkpoint=k,cold=cold['weights'].tolist(),warm=repair['weights'].tolist()))
            old=selected;warm=restored;previous_deleted=deleted
    # Binary twenty-output fixtures verify algebra, NOT empirical Stack results.
    sx=np.asarray([[1.,0.],[0.,1.],[1.,1.]],dtype=np.float32)
    sy=np.asarray([[(i+c)%2 for c in range(20)] for i in range(3)],dtype=np.float64)
    solved=solve_multioutput(sx,sy,.25);sc=certify_multioutput(sx,sy,.25,solved['weights'])
    assert sc['meets_parameter_tolerance'];checks.append('binary20_exact_joint_certificate_software_only')
    objective,gradient=objective_gradient(sx,sy,solved['weights'],.25)
    assert abs(objective-solved['objective'])<1e-12;assert np.linalg.norm(gradient)<1e-9
    checks.append('sum_output_objective_and_gradient_match')
    empty=solve_multioutput(sx[:0],sy[:0],.25,warm_start=np.ones((2,20)))
    ec=certify_multioutput(sx[:0],sy[:0],.25,np.ones((2,20)))
    assert not empty['weights'].any() and fraction(ec['parameter_error_frobenius_squared_upper'])==40
    checks.append('empty_multilabel_target_exact_40_squared_radius')
    try:certify_multioutput(sx,sy,.25,solved['weights'],max_coordinates=119)
    except CertificateBudgetError:checks.append('whole_output_budget_refused_before_evaluation')
    else:raise AssertionError('total coordinate cap not enforced')
    # Tiny algebraic fixture ownership demonstrates source expansion only; no corpus provenance.
    source_ids=['fixture-source-a','fixture-source-a','fixture-source-b'];fixture_ids=['fixture-a','fixture-b','fixture-c']
    source_graph=build_reference_graph(sx,fixture_ids,.6,source_ids=source_ids)
    requests=generate_manifest(source_graph,dataset_id='software_fixture_only',panel_id='software_source_expansion',
      source_kinds={s:'genuine_native' for s in source_ids},
      design={'requests':{'record_horizon':8,'record_checkpoints':[1,8],'source_horizon':2,'source_checkpoints':[1,2]}},
      allocations={'R':0,'U':0,'A':0,'S':1},include_excluded_blocker_stress=False)
    # Manifest generator's genuine flag is purely software input here: never save/report native provenance.
    path=next(p for p in requests['trajectories'] if p['arm']=='S')
    selected,dead=retained_selection(source_graph,requests,path['trajectory_id'],2)
    expected={fixture_ids[i] for i,s in enumerate(source_ids) if s in path['deletion_order'][:2]}
    assert set(dead)==expected and len(dead)==3 and not selected
    checks.append('artificial_source_partition_software_expansion_only')
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp)/'state.json';save_resume(p,solved['weights'],selected_ids=['a'],deleted_ids=['b'],input_bindings={'x':'bound'})
        try:load_resume(p,input_bindings={'x':'different'},expected_selected_ids=['a'],expected_deleted_ids=['b'])
        except ValueError:checks.append('resume_foreign_input_rejected')
        else:raise AssertionError('resume input mismatch accepted')
    write(out/'checkpoints.json',results);write(out/'heads.json',saved)
    summary=dict(status='passed',natural_target_outputs=7,natural_checkpoint_rows=len(results),
      joint_certificates=len(certs),scalar_certificates=sum(c['outputs'] for c in certs),
      max_frobenius_certificate_radius_display=max(c['radius_float_for_display_only'] for c in certs),
      software_checks=checks,semantic_study_result=False,Stack20_empirical=False,source_service_empirical=False,
      speedup_claim=False,no_reaccess_claim=False,objective_normalization='sum_over_outputs_mean_over_rows')
    write(out/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
