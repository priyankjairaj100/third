#!/usr/bin/env python3
"""Tiny algebra-only source/matrix regression checks; NO empirical dataset."""
from pathlib import Path
import json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase3.reference_graph import build_reference_graph
from phase4.requests import generate_manifest
from phase5.convex_multioutput import *

def main():
    x=np.asarray([[1.,0.],[0.,1.],[1.,1.],[-1.,1.]],dtype=np.float32)
    y=np.asarray([[(i+c)%2 for c in range(20)] for i in range(4)],dtype=np.float64)
    ids=['fixture-a','fixture-b','fixture-c','fixture-d'];owners=['fixture-source-a','fixture-source-a','fixture-source-b','fixture-source-c']
    graph=build_reference_graph(x,ids,.6,source_ids=owners)
    requests=generate_manifest(graph,dataset_id='software_fixture_only',panel_id='source_multioutput_contract',
      source_kinds={s:'genuine_native' for s in owners},
      design={'requests':{'record_horizon':4,'record_checkpoints':[1,4],
                         'source_horizon':3,'source_checkpoints':[1,2,3]}},
      allocations={'R':0,'U':0,'A':0,'S':1},include_excluded_blocker_stress=False)
    # 'genuine_native' here only exercises the API branch on explicitly artificial IDs;
    # no input or output is presented as a native-source empirical corpus.
    path=next(p for p in requests['trajectories'] if p['arm']=='S');lookup={r:i for i,r in enumerate(ids)}
    old={ids[i] for i in graph.selected_indices()};oi=np.array([lookup[r] for r in sorted(old)],dtype=int)
    warm=solve_multioutput(x[oi],y[oi],.25)['weights'];checks=[]
    for k in path['checkpoints']:
        current,deleted=retained_selection(graph,requests,path['trajectory_id'],k);current=set(current)
        assert set(deleted)=={r for r,s in zip(ids,owners) if s in path['deletion_order'][:k]}
        ix=np.array([lookup[r] for r in sorted(current)],dtype=int)
        removed=np.array([lookup[r] for r in sorted(old-current)],dtype=int)
        added=np.array([lookup[r] for r in sorted(current-old)],dtype=int)
        _,oldg=objective_gradient(x[oi],y[oi],warm,.25);_,newg=objective_gradient(x[ix],y[ix],warm,.25)
        updated=signed_gradient(oldg,warm,.25,len(oi),len(ix),x[removed],y[removed],x[added],y[added])
        assert np.max(np.abs(updated-newg),initial=0)<1e-13
        cold=solve_multioutput(x[ix],y[ix],.25);repair=solve_multioutput(x[ix],y[ix],.25,warm_start=warm)
        cold_cert=certify_multioutput(x[ix],y[ix],.25,cold['weights'])
        warm_cert=certify_multioutput(x[ix],y[ix],.25,repair['weights'])
        assert cold_cert['meets_parameter_tolerance'] and warm_cert['meets_parameter_tolerance']
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'resume.json';bindings=dict(fixture='algebra_only',checkpoint=k)
            wm=save_resume(target,repair['weights'],selected_ids=current,deleted_ids=deleted,input_bindings=bindings)
            warm,rm=load_resume(target,input_bindings=bindings,expected_selected_ids=current,expected_deleted_ids=deleted)
            assert wm['serialized_bytes']==rm['serialized_bytes'] and np.array_equal(warm,repair['weights'])
        checks.append(dict(source_checkpoint=k,deleted_records=len(deleted),selected_records=len(ix),
          changed_count_identity_max_error=float(np.max(np.abs(updated-newg),initial=0)),
          matrix_certificates_passed=2,output_dimension=20,resume_head_equal=True))
        old=current;oi=ix
    assert not old and not warm.any()
    result=dict(status='passed',scope='tiny algebraic software fixtures ONLY; no empirical source study',
      checkpoints=checks,empty_after_all_source_deletions=True,
      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      implementation_sha256=hashlib.sha256(Path(__file__).with_name('convex_multioutput.py').read_bytes()).hexdigest())
    out=ROOT/'phase5/results/convex_source_software_checks.json'
    with out.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
