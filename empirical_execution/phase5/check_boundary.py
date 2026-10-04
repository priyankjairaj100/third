"""Algebraic software checks plus new boundary integration on reused natural Civil100.

Mathematical arrays below are unit fixtures only, not synthetic empirical data.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,itertools,json,math
from fractions import Fraction
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from ccu.core import ridge_moments
from phase3.reference_graph import build_reference_graph
from phase3.panels import graph_normalize,reference_cosines
from phase4.execution import write,common_decode,array_hash
from phase5.boundary import (add,mul,divide_positive,score_intervals,graph_envelope,
 model_envelope_bound,lsh_candidates,missed_pair_sample,audit_ann,run_frontiers)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'phase5/results/boundary_engineering');args=p.parse_args();out=args.output_dir
    if out.exists() and any(out.iterdir()):raise FileExistsError('Frozen output exists; supply new directory')
    out.mkdir(parents=True,exist_ok=True);checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    # Independently compare endpoint arithmetic with exact binary rational values.
    scalars=[0.,np.nextafter(0.,1.),1.,-1.,2.**-500,2.**500,np.nextafter(1.,2.)]
    checked=0
    for a,b in itertools.product(scalars,repeat=2):
        a,b=float(a),float(b)
        for primitive,exact in [(add,Fraction(a)+Fraction(b)),(mul,Fraction(a)*Fraction(b))]:
            lo,hi=primitive((a,a),(b,b))
            if math.isfinite(float(lo)) and math.isfinite(float(hi)):
                check(f'primitive_exact_{checked}',Fraction(float(lo))<=exact<=Fraction(float(hi)));checked+=1
        if b>0 and math.isfinite(a/b):
            lo,hi=divide_positive((a,a),(b,b));check(f'division_exact_{checked}',Fraction(float(lo))<=Fraction(a)/Fraction(b)<=Fraction(float(hi)));checked+=1
    fixtures=[np.asarray([[1,0],[-1,0],[0,1],[1,1]],np.float32),
      np.asarray([[np.finfo(np.float32).smallest_subnormal,0],[0,np.finfo(np.float32).smallest_subnormal]],np.float32),
      np.asarray([[np.finfo(np.float32).max,1],[np.finfo(np.float32).max,-1]],np.float32),
      np.asarray([[1,1e-6,-1],[1,-1e-6,-1]],np.float32)]
    for k,x in enumerate(fixtures):
        z=graph_normalize(x);intervals=list(score_intervals(x))
        check(f'all_pairs_interval_frame_{k}',len(intervals)==len(x)*(len(x)-1)//2)
        for i,j,lo,hi in intervals:check(f'reference_interval_{k}_{i}_{j}',lo<=reference_cosines(z[i:i+1],z[j:j+1])[0,0]<=hi)
        tau=float(reference_cosines(z[:1],z[1:2])[0,0]);tau=min(1.,max(-1.,tau))
        graph=build_reference_graph(x,[str(i) for i in range(len(x))],tau)
        gl,gu,audit=graph_envelope(x,graph)
        for mask in range(1<<len(x)):
            deleted=[str(i) for i in range(len(x)) if mask>>i&1]
            check(f'sandwich_all_deletions_{k}_{mask}',set(gu.selected_indices(deleted))<=set(graph.selected_indices(deleted))<=set(gl.selected_indices(deleted)))
    x=fixtures[0];y=np.array([[0.],[1.],[.25],[.5]],dtype=np.float64);w=np.array([[.123],[-.234]],dtype=np.float64)
    for minimum,maximum in [(set(),set()),(set(),{0,1,2,3}),({1},{0,1,2,3}),({2},{2})]:
        cert=model_envelope_bound(x,y,w,minimum,maximum,.1);bound=Fraction(int(cert['radius_numerator']),int(cert['radius_denominator']))
        optional=list(maximum-minimum)
        for mask in range(1<<len(optional)):
            selected=sorted(minimum|{optional[i] for i in range(len(optional)) if mask>>i&1});ix=np.asarray(selected,dtype=int)
            target=common_decode(ridge_moments(x[ix],y[ix]),.1).weights
            check(f'all_subset_model_bound_{sorted(minimum)}_{mask}',np.linalg.norm(w-target)<=float(bound)+1e-14)
    for candidates in [set(),{(0,1)},set(itertools.combinations(range(4),2))]:
        sample,meta=missed_pair_sample(4,candidates,size=20)
        check(f'nonretrieved_census_{len(candidates)}',set(sample)==set(itertools.combinations(range(4),2))-candidates)
        check(f'nonretrieved_probability_{len(candidates)}',meta['sample_size']==meta['population_size'])
    seen=set()
    for seed in range(300):seen.update(missed_pair_sample(4,{(0,1)},size=1,seed=seed)[0])
    check('nonretrieved_no_excluded_stratum',seen==set(itertools.combinations(range(4),2))-{(0,1)})
    ca,ma=lsh_candidates(fixtures[0],tables=2,bits=3,seed=7);cb,mb=lsh_candidates(fixtures[0],tables=2,bits=3,seed=7)
    check('LSH_reproducible',ca==cb and ma==mb)
    rejected=0
    for invalid in [np.zeros((2,2),dtype=np.float32),np.ones((2,2),dtype=np.float64)]:
        try:lsh_candidates(invalid)
        except ValueError:rejected+=1
    check('invalid_feature_rejection',rejected==2)
    from phase4.requests import generate_manifest
    sg=build_reference_graph(x,[str(i) for i in range(len(x))],.6,['source-a','source-a','source-b','source-c'])
    sm=generate_manifest(sg,dataset_id='algebraic_software_fixture',panel_id='not_empirical',
      source_kinds={k:'engineering_proxy' for k in sg.source_ids},allocations={'R':1,'S':0,'U':0,'A':0},
      record_horizon=4,include_excluded_blocker_stress=False)
    # Source API semantics are checked without pretending proxies are native sources.
    from phase5.boundary import deleted_indices
    check('source_request_expands_whole_group',deleted_indices({'deletion_order':['source-a'],'unit':'source'},1,sg)=={0,1})
    check('record_request_does_not_expand_group',deleted_indices({'deletion_order':['0'],'unit':'record'},1,sg)=={0})
    check('FP32_numpy_solve_dtype',np.linalg.solve(np.eye(2,dtype=np.float32),np.ones((2,1),np.float32)).dtype==np.float32)
    source=ROOT/'data/civil_comments_engineering_preview.jsonl';records=read_natural_jsonl(source,require_labels=True)
    ids=[r['record_id'] for r in records];curator,meta=lexical_engineering_features(records,128)
    y=np.asarray([r['label'] for r in records],dtype=np.float64)[:,None];graph=build_reference_graph(curator,ids,.6)
    manifest_path=ROOT/'phase4/results/execution_engineering_final/d64/requests.json';requests=json.loads(manifest_path.read_text())
    # One data-independent projection sampled before output; SAME requests/factors.
    projection=np.random.Generator(np.random.PCG64(202710044)).standard_normal((128,32))/math.sqrt(32)
    projected=(curator.astype(np.float64)@projection).astype(np.float32)
    lock={'evidence_role':'reused_natural_Civil100_lexical_engineering_only','source_sha256':sha256_file(source),
      'records':len(records),'features':meta,'request_manifest_file_sha256':sha256_file(manifest_path),
      'projection_seed':202710044,'projection_generator':'PCG64.standard_normal divided by sqrt(32)',
      'projection_sha256':array_hash(projection),'projection_output_sha256':array_hash(projected),
      'projection_input':'same stored lexical d128 curator; no postprojection renormalization',
      'lambda_factors':[.1,1.,10.],'lambda_reference':.01,'no_parameter_tuning':True,
      'primary_semantic_study':False,'timing_and_memory_benchmark':False}
    write(out/'lock.json',lock);np.save(out/'fixed_projection.npy',projection,allow_pickle=False)
    result=run_frontiers(curator,projected,y,graph,requests,lambda_reg=.01,envelope=True)
    write(out/'boundary_audit.json',result)
    rows=result['rows'];check('three_locked_lambda_factors',len(rows)==3*sum(len(p['checkpoints']) for p in requests['trajectories']))
    check('natural_all_pair_envelope',result['graph_envelope']['all_pairs_certified']==4950)
    check('all_frontier_rows_retained',all(r['status']=='completed' for r in rows))
    summary={'status':'passed','software_checks':len(checks),'check_names':checks,
      'scope':'natural-preview lexical boundary integration and separately labeled algebraic software fixtures',
      'primary_study':False,'semantic_features':False,'new_independent_empirical_population':False,
      'checkpoint_factor_rows':len(rows),'distinct_checkpoint_rows':len(rows)//3,
      'ann_metrics':result['ann']['metrics'],'graph_envelope_uncertain_pairs':len(result['graph_envelope']['uncertain_edges']),
      'maximum_FP32_head_absolute_error':max(r.get('FP32_head_max_abs',0) for r in rows),
      'maximum_ANN_head_absolute_error':max(r['ann_head_max_abs'] for r in rows),
      'failed_rows':sum(r['status']!='completed' for r in rows),
      'source_hashes':{str(path.relative_to(ROOT)):sha256_file(path) for path in [Path(__file__),ROOT/'phase5/boundary.py',ROOT/'phase3/panels.py',ROOT/'phase3/reference_graph.py']}}
    write(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='check_names'},indent=2))

if __name__=='__main__':main()
