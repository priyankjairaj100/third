#!/usr/bin/env python3
"""Run the new comparison plumbing on real reused Civil100 text; no new corpus."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,os,hashlib
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from phase3.reference_graph import build_reference_graph
from phase4.execution import run_comparison,write,common_decode,independent_edges
from ccu.core import RidgeMoments

def main():
    from phase4.requests import generate_manifest
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'phase4/results/execution_engineering');args=parser.parse_args()
    out=args.output_dir
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new audit output directory; prior artifacts immutable')
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/civil_comments_engineering_preview.jsonl'
    rows=read_natural_jsonl(source,require_labels=True);ids=[r['record_id'] for r in rows]
    curator,curator_meta=lexical_engineering_features(rows,128)
    graph=build_reference_graph(curator,ids,.6,block_size=32)
    design={'requests':{'record_horizon':8,'record_checkpoints':[1,2,4,8],
      'source_horizon':8,'source_checkpoints':[1,2,4,8]},'seeds':{'request_salt':'ccu-v1-request'}}
    manifest=generate_manifest(graph,dataset_id='civil_comments_reused_preview',panel_id='phase4_software_all100',
      source_kinds={s:'unknown_singleton' for s in graph.source_ids},design=design,
      allocations={'R':4,'S':0,'U':4,'A':2},evidence_role='engineering_nonconfirmatory',
      master_seed=20271003,record_horizon=8,include_excluded_blocker_stress=False)
    y=np.asarray([r['label'] for r in rows],dtype=np.float64)[:,None]
    manifest_info={'data_sha256':sha256_file(source),'natural_records':len(rows),'targets':'original toxicity fractions',
      'preview_reused':True,'evaluation_test_set_used':False,'primary_study':False,'features':curator_meta,
      'design':design,'allocation':{'R':4,'S':0,'U':4,'A':2},'dimensions':[64,768],
      'same_requests_for_both_dimensions':True,'no_source_evidence':'Native original source metadata unavailable',
      'blas_threads_requested':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}}
    write(out/'engineering_lock.json',manifest_info)
    summaries=[]
    for d in [64,768]:
        x,meta=lexical_engineering_features(rows,d);write(out/f'features_d{d}.json',meta)
        result=run_comparison(curator,x,y,ids,manifest,out/f'd{d}',threshold=.6,lambda_reg=.01)
        print(json.dumps({'dimension':d,**result}),flush=True);summaries.append(result)
    # Boundary controls use retained subsets of the same natural records.
    # These are software checks, not additional empirical paths/allocations.
    zeros=common_decode(RidgeMoments(np.zeros((64,64)),np.zeros((64,1)),0),.01)
    assert not zeros.weights.any()
    edges,selected=independent_edges(curator,ids,.6,deleted=ids)
    assert not edges and not selected
    primary_refused=False
    try:run_comparison(curator,x,y,ids,manifest,out/'illegal_primary',threshold=.6,lambda_reg=.01,evidence_role='primary')
    except RuntimeError as err:primary_refused='Primary execution blocked' in str(err)
    assert primary_refused and not (out/'illegal_primary').exists()
    rejected={}
    for case,extra in [('nan_tolerance',{'tolerance':float('nan')}),
                       ('boolean_tolerance',{'tolerance':True})]:
        try:run_comparison(curator,x,y,ids,manifest,out/case,threshold=.6,lambda_reg=.01,**extra)
        except ValueError:rejected[case]=not (out/case).exists()
        else:rejected[case]=False
    try:run_comparison(curator,x,y.astype(np.float32),ids,manifest,out/'downcast_target',threshold=.6,lambda_reg=.01)
    except ValueError:rejected['implicit_target_cast']=not (out/'downcast_target').exists()
    else:rejected['implicit_target_cast']=False
    altered=json.loads(json.dumps(manifest));altered['manifest_sha256']='0'*64
    try:run_comparison(curator,x,y,ids,altered,out/'tampered_manifest',threshold=.6,lambda_reg=.01)
    except ValueError:rejected['tampered_manifest']=not (out/'tampered_manifest').exists()
    else:rejected['tampered_manifest']=False
    if not all(rejected.values()):raise AssertionError('An invalid input bypassed a gate')
    audit={'status':'passed' if not any(s['failure_events'] for s in summaries) else 'failed',
      'scope':'natural Civil100 reused lexical software integration; no new paper results',
      'dimensions':[64,768],'trajectories_per_dimension':len(manifest['trajectories']),
      'checkpoint_rows':sum(s['checkpoint_rows'] for s in summaries),
      'failure_events':sum(s['failure_events'] for s in summaries),
      'empty_target_zero_head':True,'all_records_deleted_scalar_oracle_empty':True,
      'primary_without_full_protocol_refused':primary_refused,
      'invalid_inputs_rejected_before_output':rejected,
      'maximum_same_target_head_abs_difference':max(s['maximum_same_target_head_abs_difference'] for s in summaries),
      'maximum_same_target_moment_abs_difference':max(s['maximum_same_target_moment_abs_difference'] for s in summaries),
      'source_sha256':sha256_file(source),'check_code_sha256':sha256_file(Path(__file__))}
    write(out/'execution_checks.json',audit);print(json.dumps(audit,indent=2))
    if audit['status']!='passed':raise RuntimeError('One or more engineering paths failed; see raw preserved rows')

if __name__=='__main__':main()
