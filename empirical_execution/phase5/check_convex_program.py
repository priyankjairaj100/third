#!/usr/bin/env python3
"""Three-arm natural engineering and algebraic source/eligibility/failure checks."""
from pathlib import Path
import argparse,copy,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from phase3.reference_graph import build_reference_graph
from phase4.requests import generate_manifest
from phase5.convex_program import LogisticPayload,run_convex_program,write,code_hashes

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'phase5/results/convex_program_engineering');args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    rows=read_natural_jsonl(ROOT/'data/civil_comments_engineering_preview.jsonl');ids=[r['record_id'] for r in rows]
    x,_=lexical_engineering_features(rows,64);cx,_=lexical_engineering_features(rows,128)
    y=np.asarray([[r['label']] for r in rows],dtype=np.float64)
    requests=json.loads((ROOT/'phase4/results/execution_engineering_final/d64/requests.json').read_text())
    result=run_convex_program(cx,x,y,ids,ids,{r:'unknown_singleton' for r in ids},requests,out/'natural',
      lambda_reg=.01,threshold=.6,eval_features=x,eval_targets=y,allocations={'R':2,'S':2},
      evaluation_scope='same reused original Civil100 pool; software task-output check only, not held-out performance')
    assert result['executed_trajectories']==2 and result['missing_trajectories']==2
    assert result['checkpoint_rows']==8 and result['method_releases']==24 and not result['failures']
    assert result['joint_certificates']==30 and result['scalar_certificates']==30
    checks=['natural_three_arms_eight_checkpoints_24releases_30certificates','missing_native_sources_preserved_not_replaced']
    graph=build_reference_graph(cx,ids,.6)
    state=LogisticPayload(graph,x,y,8,'record');initial_ids=set(state.counter.record_ids)
    assert initial_ids=={ids[i] for i,b in enumerate(graph.blockers) if len(b)<=8}
    x_mut=x.copy();y_mut=y.copy();state2=LogisticPayload(graph,x_mut,y_mut,8,'record')
    x_mut.fill(123);y_mut.fill(.75);actual,targets,_=state2.selected_payload()
    assert not np.any(actual==123) and np.any(targets!=.75)
    assert not any(hasattr(state2,k) for k in ('graph','curator','original_features','original_targets'))
    checks.append('horizon_payload_copies_no_constructor_array_reference')
    for path in requests['trajectories'][:2]:
        state=LogisticPayload(graph,x,y,8,'record');cumulative=[]
        for rid in path['deletion_order']:
            state.delete([rid]);cumulative.append(rid)
            expected={ids[i] for i in graph.selected_indices(cumulative)}
            assert set(state.counter.selected_ids())==expected
            assert all(s not in set(cumulative) for s in state.counter.selected_ids())
        assert state.counter.horizon==0
    checks.append('each_individual_delete_matches_fresh_membership_and_horizon')
    first=next(p for p in requests['trajectories'] if p['arm']=='R')
    state=LogisticPayload(graph,x,y,8,'record');state.delete(first['deletion_order'][:1])
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp)/'owned.npz';receipt=state.snapshot(p,np.zeros((64,1),dtype=np.float64))
        restored,_,_=LogisticPayload.load(p,expected_sha256=receipt['file_sha256'])
        for rid in first['deletion_order'][1:]:
            state.delete([rid]);restored.delete([rid])
            a,b,si1=state.selected_payload();c,d,si2=restored.selected_payload()
            assert si1==si2 and np.array_equal(a,c) and np.array_equal(b,d)
        try:LogisticPayload.load(p,expected_sha256='0'*64)
        except ValueError:checks.append('corrupted_resume_hash_refused')
        else:raise AssertionError('corrupted snapshot accepted')
    checks.append('midtrajectory_payload_resume_then_all_remaining_requests')
    sx=np.asarray([[1.,0.],[0.,1.],[1.,1.]],dtype=np.float32);sy=np.asarray([[0.,1.],[1.,0.],[1.,1.]],np.float64)
    si=['fixture-a','fixture-b','fixture-c'];ss=['fixture-one','fixture-one','fixture-two']
    sg=build_reference_graph(sx,si,.6,source_ids=ss)
    sm=generate_manifest(sg,dataset_id='software_fixture_only',panel_id='convex_source_contract',source_kinds={s:'genuine_native' for s in ss},
      design={'requests':{'record_horizon':3,'record_checkpoints':[1,3],'source_horizon':2,'source_checkpoints':[1,2]}},
      allocations={'R':0,'S':1,'U':0,'A':0},include_excluded_blocker_stress=False)
    source=run_convex_program(sx,sx,sy,si,ss,{s:'genuine_native' for s in ss},sm,out/'source_software_fixture',
      lambda_reg=.25,threshold=.6,eval_features=sx,eval_targets=sy,allocations={'R':0,'S':1},
      evaluation_scope='algebraic software fixture only; no corpus, native source or empirical evidence')
    assert source['checkpoint_rows']==2 and source['method_releases']==6 and not source['failures']
    last=json.loads(next((out/'source_software_fixture').glob('S*/checkpoint_2.json')).read_text())
    assert last['selected_ids']==[] and all(not np.asarray(v['weights']).any() for v in last['methods'].values())
    checks.append('two_source_prefixes_all_three_methods_full_deletion_zero_software_only')
    failed=run_convex_program(sx,sx,sy,si,ss,{s:'genuine_native' for s in ss},sm,out/'certificate_budget_failure',
      lambda_reg=.25,threshold=.6,eval_features=sx,eval_targets=sy,allocations={'R':0,'S':1},certificate_budget=0,
      evaluation_scope='algebraic refusal path only')
    assert len(failed['failures'])==3 and failed['method_releases']==0
    checks.append('certificate_budget_failure_preserves_failed_and_future_unexecuted_cells')
    for name,fn in [
      ('primary_activation',lambda:run_convex_program(sx,sx,sy,si,ss,{s:'genuine_native' for s in ss},sm,out/'bad_primary',lambda_reg=.25,threshold=.6,eval_features=sx,eval_targets=sy,evidence_role='confirmatory')),
      ('proxy_source_S',lambda:run_convex_program(sx,sx,sy,si,ss,{s:'engineering_proxy' for s in ss},sm,out/'bad_source',lambda_reg=.25,threshold=.6,eval_features=sx,eval_targets=sy,allocations={'R':0,'S':1}))]:
        try:fn()
        except ValueError:checks.append(name+'_refused')
        else:raise AssertionError(name+' was accepted')
    report=dict(status='passed',checks=checks,natural=result,software_source=source,
      failure_test_status=failed['status'],code_sha256=code_hashes(),check_sha256=sha256_file(Path(__file__)),
      no_synthetic_empirical_data=True,authentic_source_experiment=False,primary_study_started=False)
    write(out/'checks.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
