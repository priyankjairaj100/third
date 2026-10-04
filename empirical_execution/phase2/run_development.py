#!/usr/bin/env python3
"""Task-consequence diagnostics on existing natural text, never confirmation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fractions import Fraction
from collections import Counter
import hashlib,json,platform,time
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features,normalize_text,sha256_file
from ccu.core import build_blocker_graph,direct_oracle,ridge_moments,solve_ridge,normalized_graph_features
from ccu.structure import generate_request_trajectory,derive_seed,expected_admissions
from ccu.exact_canonical import ExactCanonicalSummary
from ccu.certified_ridge import factor_residual_certificate
from phase2.analysis import summarize_trajectories,prediction_metrics

OUT=ROOT/'phase2/results'
SALT='ccu-phase2-development-exacttext-split-v1'
ALLOCATIONS={'R':64,'U':32,'A':16}
DIMENSIONS=[64,768]
LEVELS=[1,2,4,8]
RUN_CONTEXT={'stage':'initialization'}

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def sha(s):return hashlib.sha256(s).hexdigest()
def head(x,y,indices):
    ix=np.asarray(indices,dtype=int)
    return solve_ridge(ridge_moments(x[ix],y[ix]),.01).weights
def mse(y,p):return float(np.mean((y-p)**2))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data/civil_comments_engineering_preview.jsonl'
    rows=read_natural_jsonl(path,require_labels=True)
    assignments=[]
    for row in rows:
        normalized=normalize_text(row['text'])
        group=sha(normalized.encode())
        digest=sha((SALT+'\0'+normalized).encode())
        train=int(digest,16)*5<4*(1<<256)
        assignments.append({'record_id':row['record_id'],'text_group_sha256':group,
          'partition':'development_train' if train else 'development_evaluation','partition_hash':digest})
    train=[i for i,a in enumerate(assignments) if a['partition']=='development_train']
    evaluation=[i for i,a in enumerate(assignments) if a['partition']=='development_evaluation']
    if not train or not evaluation:raise RuntimeError('fixed split empty; no automatic reroll')
    assert not ({assignments[i]['text_group_sha256'] for i in train}&{assignments[i]['text_group_sha256'] for i in evaluation})
    train_rows=[rows[i] for i in train];eval_rows=[rows[i] for i in evaluation]
    ids=[r['record_id'] for r in train_rows];lookup={v:i for i,v in enumerate(ids)}
    graph_x,graph_meta=lexical_engineering_features(train_rows,128)
    eval_graph_x,_=lexical_engineering_features(eval_rows,128)
    graph=build_blocker_graph(graph_x,ids,.6)
    initial=set(map(int,graph.selected_indices()))
    requests=[]
    for arm,allocation in ALLOCATIONS.items():
        for j in range(allocation):
            seed=derive_seed(20271003,'civil_phase2_development',arm,str(j))
            req=generate_request_trajectory(graph,arm,seed,8,LEVELS)
            r=req.to_dict();r['trajectory_id']=f'{arm}{j:03d}';requests.append(r)
    sources=['phase2/run_development.py','phase2/analysis.py','ccu/core.py','ccu/data.py',
      'ccu/structure.py','ccu/summary.py','ccu/exact_canonical.py','ccu/certified_ridge.py','ccu/native/exact_chart.cpp','ccu/native/exact_chart']
    design={'status':'development_lock_before_this_run_outcomes_not_preregistration',
      'scope':'reused Civil100 preview; lexical engineering; no source or semantic inference',
      'all_computation_local':True,'synthetic_empirical_data':False,'confirmatory':False,
      'split_salt':SALT,'grouping':'NFC plus collapsed whitespace; preserve case',
      'train_hash_fraction':'4/5','no_split_retry':True,'source_group_split':False,
      'previously_seen_in_engineering':True,'dimensions':DIMENSIONS,'lambda':.01,'threshold':.6,
      'curator_features':graph_meta,'requested_horizon':8,'requested_checkpoints':LEVELS,
      'allocations':ALLOCATIONS,'bootstrap_samples':10000,'bootstrap_seed':20271004,
      'baseline':'fresh refit on retained initial-selected rows with same lambda*n normalization',
      'oracle':'independently rescore retained training-only graph and rebuild moments',
      'strict_audit_paths':'first R,U,A path in stable ID order, d=768 only; all their checkpoints',
      'failure_policy':'record and stop; do not reroll or silently omit',
      'corpus_sha256':sha256_file(path),'source_hashes':{p:sha256_file(ROOT/p) for p in sources},
      'python_version':platform.python_version(),'numpy_version':np.__version__}
    write(OUT/'design_lock.json',design)
    write(OUT/'partition_manifest.json',assignments)
    write(OUT/'requests.json',requests)
    # Only fixed training text enters curation and request generation.
    score=normalized_graph_features(graph_x)@normalized_graph_features(eval_graph_x).T
    geometry={'training_records':len(train),'evaluation_records':len(evaluation),
      'initially_selected':len(initial),'initially_excluded':len(train)-len(initial),
      'train_graph_edges':graph.edge_count,'train_eval_exacttext_overlap':0,
      'crosssplit_lexical_pairs_above_threshold':int(np.count_nonzero(score>.6)),
      'crosssplit_max_lexical_score':float(np.max(score)),
      'semantic_or_source_leakage_guard_complete':False,
      'evaluation_never_in_graph_requests_or_learner':True,
      'not_applicable_arms':{'S':'native source fields unavailable; no singleton substitution'}}
    write(OUT/'geometry.json',geometry)
    registry=OUT/'checkpoints.jsonl';registry.write_text('')
    predpath=OUT/'predictions.jsonl';predpath.write_text('')
    y=np.asarray([r['label'] for r in train_rows],dtype=np.float64)[:,None]
    ey=np.asarray([r['label'] for r in eval_rows],dtype=np.float64)[:,None]
    write(OUT/'evaluation_targets.json',{'record_ids':[r['record_id'] for r in eval_rows],
      'original_labels':ey[:,0].tolist(),'source_file_sha256':sha256_file(path)})
    all_results=[];utility=[];cert_results=[];feature_metadata=[]
    for d in DIMENSIONS:
        RUN_CONTEXT.update(stage='dimension',dimension=d)
        print(f'Running development task consequence d={d}',flush=True)
        x,meta=lexical_engineering_features(train_rows,d)
        ex,emeta=lexical_engineering_features(eval_rows,d);ex=ex.astype(np.float64)
        feature_metadata.append({'dimension':d,'train':meta,'evaluation':emeta})
        pre=head(x,y,sorted(initial));prepred=ex@pre;preloss=mse(ey,prepred)
        centered=prepred-prepred[0:1]
        presd=float(np.sqrt(np.mean((centered-centered.mean(axis=0,keepdims=True))**2)))
        hash_order=sorted(range(len(train)),key=lambda i:sha(('ccu-phase2-size-control-v1\0'+ids[i]).encode()))
        utility.append({'dimension':d,'predelete_curated_mse':preloss,
          'predelete_score_sd':presd,'uncurated_mse':mse(ey,ex@head(x,y,range(len(train)))),
          'same_size_hash_mse':mse(ey,ex@head(x,y,hash_order[:len(initial)])),
          'training_mean_constant_mse':mse(ey,np.full_like(ey,float(y.mean()))),
          'training_mean':float(y.mean()),'evaluation_records':len(ey),
          'predelete_predictions':prepred[:,0].tolist(),
          'scope':'fixed reused development split; no calibrated utility claim'})
        cache={};scoped=[]
        first_by_arm={arm:next(r['trajectory_id'] for r in requests if r['arm']==arm) for arm in ALLOCATIONS}
        for req in requests:
            exact=None;strict_position=0
            if d==768 and req['trajectory_id']==first_by_arm[req['arm']]:
                exact=ExactCanonicalSummary.build(x,y,graph.blockers,req['initial_horizon'])
            for checkpoint in req['checkpoints']:
                RUN_CONTEXT.update(stage='checkpoint',trajectory_id=req['trajectory_id'],checkpoint=checkpoint)
                deleted=req['deletion_order'][:checkpoint];dead={lookup[n] for n in deleted}
                key=tuple(sorted(deleted));cache_hit=key in cache
                if not cache_hit:
                    oracle=direct_oracle(x,y,ids,.6,.01,deleted,graph.priority,graph_features=graph_x)
                    selected=set(map(int,oracle.selected_indices))
                    frozen_indices=sorted(initial-dead)
                    frozen=head(x,y,frozen_indices)
                    op=ex@oracle.solution.weights;fp=ex@frozen
                    additions=selected-initial
                    if not additions and not np.allclose(op,fp,atol=1e-12,rtol=1e-12):
                        raise AssertionError('zero-admission negative control changed predictions')
                    cache[key]=(selected,additions,op,fp)
                    del oracle
                selected,additions,op,fp=cache[key]
                metrics=prediction_metrics(ey,{'oracle':op,'frozen':fp},
                  predelete_predictions=prepred,predelete_mean=y.mean(axis=0))
                row={'dimension':d,'trajectory_id':req['trajectory_id'],'arm':req['arm'],
                  'checkpoint':checkpoint,'deleted_ids':deleted,'deleted_set_sha256':sha(json.dumps(key).encode()),
                  'selected_ids':[ids[i] for i in sorted(selected)],'added_ids':[ids[i] for i in sorted(additions)],
                  'addition_count':len(additions),'removed_selected_count':len(initial&dead),
                  'selected_count':len(selected),'frozen_selected_count':len(initial-dead),
                  **metrics,
                  'oracle_cache_hit':cache_hit,'status':'completed_development_diagnostic'}
                if exact is not None:
                    batch=[lookup[n] for n in deleted[strict_position:]];strict_position=checkpoint
                    exact.delete(batch)
                    w=exact.decode_candidate(.01)
                    cert=factor_residual_certificate(*exact.factor(),.01,w)
                    if not cert.meets_tolerance(Fraction(1,10**10)):raise AssertionError('strict audit bound fails')
                    retained=[i for i in range(len(train)) if i not in dead];ix=np.asarray(retained,dtype=int)
                    fresh_graph=build_blocker_graph(graph_x[ix],[ids[i] for i in retained],.6)
                    rebuilt=ExactCanonicalSummary.build(x[ix],y[ix],fresh_graph.blockers,exact.horizon,
                      owners=retained,universe_size=len(train),alive=retained)
                    if exact.canonical_bytes()!=rebuilt.canonical_bytes():raise AssertionError('strict retained state mismatch')
                    difference=float(np.max(np.abs(ex@w-op),initial=0))
                    if difference>1e-10:raise AssertionError('strict vs diagnostic oracle mismatch')
                    audit={'dimension':d,'trajectory_id':req['trajectory_id'],'checkpoint':checkpoint,
                      'state_sha256':sha(exact.canonical_bytes()),'fresh_bytes_equal':True,
                      'certificate':cert.to_dict(40),'max_diagnostic_prediction_difference':difference}
                    cert_results.append(audit)
                    row['strict_audit_index']=len(cert_results)-1
                with registry.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                prediction={'dimension':d,'trajectory_id':req['trajectory_id'],'arm':req['arm'],'checkpoint':checkpoint,
                  'oracle_predictions':op[:,0].tolist(),'frozen_predictions':fp[:,0].tolist()}
                with predpath.open('a') as f:f.write(json.dumps(prediction,allow_nan=False)+'\n')
                scoped.append(row);all_results.append(row)
        summary=summarize_trajectories(scoped,bootstrap_samples=10000,seed=20271004)
        write(OUT/f'analysis_d{d}.json',summary)
        print(f'Completed d={d}: {len(scoped)} checkpoints; {len(cache)} distinct deletion sets',flush=True)
    write(OUT/'utility_context.json',utility);write(OUT/'feature_metadata.json',feature_metadata)
    write(OUT/'strict_audits.json',cert_results)
    distinct=[]
    for arm in ALLOCATIONS:
        for k in LEVELS:
            part=[r for r in all_results if r['dimension']==DIMENSIONS[0] and r['arm']==arm and r['checkpoint']==k]
            distinct.append({'arm':arm,'checkpoint':k,'paths':len(part),
              'distinct_deleted_sets':len({r['deleted_set_sha256'] for r in part})})
    analytic=[]
    for arm in ['R','U']:
        frame=None if arm=='R' else [ids[i] for i in range(len(ids)) if i not in initial]
        for k in sorted({r['checkpoint'] for r in all_results if r['arm']==arm}):
            value=expected_admissions(graph,k,unit='record',sampling_frame=frame)
            part=[r for r in all_results if r['dimension']==DIMENSIONS[0] and r['arm']==arm and r['checkpoint']==k]
            analytic.append({'arm':arm,'checkpoint':k,'theoretical_mean_from_same_graph':value,
              'sample_mean_admissions':float(np.mean([r['addition_count'] for r in part])),
              'scope':'sampler check; shared graph is not independent theorem evidence'})
    write(OUT/'distinct_interventions.json',distinct);write(OUT/'analytic_mean_checks.json',analytic)
    summary={'status':'completed_development_not_confirmation','geometry':geometry,
      'requested_trajectories_per_dimension':sum(ALLOCATIONS.values()),'dimensions':DIMENSIONS,
      'checkpoint_rows':len(all_results),'strict_audited_checkpoints':len(cert_results),
      'zero_admission_checkpoint_rows':sum(r['addition_count']==0 for r in all_results),
      'failed_checkpoints':0,'scope':'reused preview conditional diagnostics only',
      'design_lock_sha256':sha256_file(OUT/'design_lock.json'),
      'request_manifest_sha256':sha256_file(OUT/'requests.json'),
      'partition_sha256':sha256_file(OUT/'partition_manifest.json'),
      'primary_study_started':False,'source_arm_executed':False,'semantic_encoder_used':False}
    write(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        write(OUT/'failure.json',{'type':type(error).__name__,'message':str(error),'status':'stopped_no_reroll','context':RUN_CONTEXT})
        raise
