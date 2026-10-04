#!/usr/bin/env python3
"""Natural-preview integration plus explicitly algebraic software-only source checks."""
from pathlib import Path
import argparse, hashlib, json, sys, tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ccu.data import read_natural_jsonl, lexical_engineering_features, sha256_file
from phase5.refit import (Config,REFERENCE,OFFICIAL,fit_curator,frozen_fitted_selection,initial_graph,
    run_boundary,backend_status,vendor_integrity,ridge_head,write)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,
        default=ROOT/'phase5/results/refit_engineering');args=parser.parse_args();out=args.output_dir
    if out.exists() and any(out.iterdir()):raise FileExistsError('Frozen output exists')
    out.mkdir(parents=True,exist_ok=True);checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    manifest=vendor_integrity();check('ten_upstream_files_byte_exact',len(manifest['files'])==10)
    status=backend_status();write(out/'official_backend_status.json',status)
    config=Config(clusters=4,iterations=12,epsilon=.4,seed=1234,backend=REFERENCE,
        development_config_id='Civil100-lexical-integration-only-k4-it12; not primary selection')
    source=ROOT/'data/civil_comments_engineering_preview.jsonl'
    rows=read_natural_jsonl(source,require_labels=True);ids=[r['record_id'] for r in rows]
    x,meta=lexical_engineering_features(rows,128);z,learner_meta=lexical_engineering_features(rows,64)
    y=np.asarray([r['label'] for r in rows],dtype=np.float64)[:,None]
    design={'requests':{'record_horizon':8,'record_checkpoints':[1,2,4,8],
        'source_horizon':2,'source_checkpoints':[1,2]},'seeds':{'request_salt':'ccu-v1-request'}}
    write(out/'data_provenance.json',dict(source_sha256=sha256_file(source),feature=meta,
        learner=learner_meta,evidence_role='lexical_engineering_only',natural_rows=len(rows),
        genuine_source_groups=0,human_ratings=0,semantic_encoder=False,
        prediction_frame='same natural input pool; algebraic disagreement only, no held-out task claim'))
    summary=run_boundary(x,z,y,ids,ids,{i:'unknown_singleton' for i in ids},config,out/'natural',
        lambda_reg=.01,design=design,allocations={'R':2,'S':0,'A':2},seed_variants=[1235,1236,1237])
    check('natural_all_checkpoints_completed',summary['failures']==0)
    check('natural_source_arm_not_invented',summary['source_trajectories']==0)
    check('natural_first_two_record_paths_every_step',summary['checkpoint_rows']==24)
    initial=json.loads((out/'natural/original/fitted.json').read_text())
    initialhead=json.loads((out/'natural/original_head.json').read_text())
    check('branch_own_initial_selected_set',initialhead['selected_ids']==initial['selected_ids'])
    original_set=set(initial['selected_ids'])
    checkpoints=[json.loads(l) for l in (out/'natural/checkpoints.jsonl').read_text().splitlines()]
    repeat=json.loads((out/'natural/seed_audit.json').read_text())
    check('five_identical_seed_repeats_saved',len([r for r in repeat if r['same_declared_seed']])==5)
    check('reference_same_seed_selected_and_predictions_repeat',all(r['selected_symmetric_difference']==0 and r['prediction_rms']==0 for r in repeat[:5]))
    check('three_alternative_seed_results_saved',len(repeat[5:])==3)
    for row in checkpoints:
        for name,selected in row['selected_ids'].items():
            ix=np.asarray([i for i,r in enumerate(ids) if r in set(selected)],dtype=int)
            # Independent direct normal equations, no core moment/solver call.
            zd=z[ix].astype(np.float64)
            independent=np.linalg.solve(zd.T@zd+.01*len(ix)*np.eye(z.shape[1]),zd.T@y[ix]) if len(ix) else np.zeros((z.shape[1],1))
            check('saved_head_'+row['trajectory_id']+'_'+str(row['checkpoint'])+'_'+name,
                np.max(np.abs(independent-np.asarray(row['weights'][name])),initial=0)<1e-10)
    # A 6-point algebraic fixture tests source services and failure boundaries.
    # These vectors/labels/source names are mathematical software inputs only.
    # Their outcomes are NEVER counted as natural-data or semantic evidence.
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);v=np.asarray([[1,0],[1,0],[0,1],[0,1],[.8,.6],[.6,.8]],np.float32)
        labels=np.asarray([[0],[1],[.2],[.4],[.6],[.8]],np.float64)
        fids=list('abcdef');sources=['fixture-source-1']*2+['fixture-source-2']*2+['fixture-source-3']*2
        cfg=Config(1,2,.1,4,REFERENCE,development_config_id='algebraic software-only fixture')
        fitted=fit_curator(v,fids,cfg,p/'fit')
        graph=initial_graph(v,fids,sources,fitted,cfg)
        check('initial_blocker_graph_matches_branch_selection',
              [fids[i] for i in graph.selected_indices()]==fitted['selected_ids'])
        nochange,_=frozen_fitted_selection(v,fids,fitted,[],cfg)
        check('frozen_no_deletion_matches_branch_initial',nochange==fitted['selected_ids'])
        empty,ledger=frozen_fitted_selection(v,fids,fitted,fids,cfg)
        check('frozen_full_deletion_empty',empty==[] and ledger['similarity_matrix_elements_computed']==0)
        emptyfit=fit_curator(v[:0],[],cfg,p/'empty')
        check('fresh_full_deletion_empty',emptyfit['selected_ids']==[])
        sw=run_boundary(v,v,labels,fids,sources,{s:'genuine_native' for s in sources},cfg,p/'source_fixture',
            lambda_reg=.1,design={'requests':{'record_horizon':2,'record_checkpoints':[2],
                'source_horizon':2,'source_checkpoints':[2]}},allocations={'R':2,'S':2,'A':0},seed_variants=[5,6,7])
        srows=[json.loads(l) for l in (p/'source_fixture/checkpoints.jsonl').read_text().splitlines()]
        check('algebraic_source_and_record_every_step',sw['source_trajectories']==2 and len(srows)==10)
        check('algebraic_source_expands_whole_groups',all(r['deleted_records']==2*r['checkpoint'] for r in srows if r['arm']=='S'))
        check('algebraic_source_never_reaccess_claim',not sw['official_backend_executed'])
        for name,fn in [
            ('negative_threshold_refused',lambda:Config(1,2,1.1,4,REFERENCE,development_config_id='fixture').validate()),
            ('fixed_cluster_count_not_silently_reduced',lambda:fit_curator(v[:1],['a'],Config(2,2,.1,4,REFERENCE,development_config_id='fixture'),p/'too_few')),
            ('overwrite_refused',lambda:fit_curator(v,fids,cfg,p/'fit')),
            ('unknown_deleted_id_refused',lambda:frozen_fitted_selection(v,fids,fitted,['missing'],cfg)),
            ('changed_curator_features_refused',lambda:frozen_fitted_selection(v[::-1].copy(),fids,fitted,[],cfg)),
            ('primary_role_refused',lambda:run_boundary(v,v,labels,fids,sources,{s:'genuine_native' for s in sources},cfg,p/'primary',lambda_reg=.1,design=design,allocations={'R':1,'S':1,'A':0},seed_variants=[5,6,7],evidence_role='primary')),
        ]:
            try:fn()
            except (ValueError,RuntimeError,FileExistsError):check(name,True)
            else:check(name,False)
        if not status['official_backend_available']:
            try:fit_curator(v,fids,Config(1,2,.1,4,OFFICIAL,development_config_id='fixture'),p/'official')
            except RuntimeError:check('missing_official_dependencies_fail_closed',True)
            else:check('missing_official_dependencies_fail_closed',False)
    report=dict(status='passed',check_count=len(checks),checks=checks,natural_summary=summary,
        official_backend_status=status,software_fixture_used_only_for_source_api=True,
        official_output_equivalence_tested=False,primary_semantic_study=False,
        source_sha256=sha256_file(source),module_sha256=sha256_file(ROOT/'phase5/refit.py'),
        check_code_sha256=sha256_file(Path(__file__)))
    write(out/'refit_checks.json',report);print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
