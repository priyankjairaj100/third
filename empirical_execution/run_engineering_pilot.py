#!/usr/bin/env python3
"""Run local nonconfirmatory checks on acquired, natural Civil/CC-News previews.

This is not the full ACL study: preview provenance, unavailable source fields,
lexical features, uncalibrated thresholds, and small counts are recorded. No
synthetic documents, invented labels or held-out utility claims are produced.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,time,sys,subprocess,resource,platform
import numpy as np
from ccu.data import read_natural_jsonl,sha256_file,lexical_engineering_features
from ccu.core import build_blocker_graph,direct_oracle,EligiblePayloadState,solve_ridge
from ccu.summary import IndexedRidgeSummary,preflight,SummaryMemoryError

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def fixed_requests(graph,horizon,n_random=16,n_excluded=16):
    rng=np.random.default_rng(20271003)
    n=len(graph.record_ids);initial=set(map(int,graph.selected_indices()))
    excluded=np.array([i for i in range(n) if i not in initial],dtype=int)
    requests=[]
    for i in range(n_random):
        requests.append({'family':'R','id':f'R{i:03}','indices':rng.permutation(n)[:horizon].tolist()})
    for i in range(n_excluded):
        if len(excluded):requests.append({'family':'U','id':f'U{i:03}','indices':rng.permutation(excluded)[:horizon].tolist()})
    # Label-blind real blocker stress; no target is manufactured when none exists.
    candidates=[i for i in excluded if len(graph.blockers[int(i)])<=horizon]
    rng.shuffle(candidates)
    for j,i in enumerate(candidates[:8]):
        ids=list(map(int,graph.blockers[int(i)]))
        rest=[int(v) for v in rng.permutation(n) if int(v) not in ids and int(v)!=i]
        ids=(ids+rest)[:horizon]
        requests.append({'family':'A','id':f'A{j:03}','target_index':int(i),'indices':ids})
    return requests

def independent_coefficients(z,y,graph,deleted,horizon):
    """Fresh raw-row coefficient oracle; no proposed update/build functions."""
    d=z.shape[1];q=d*(d+1)//2;tri=np.triu_indices(d)
    coeff={};dead=set(deleted)
    for i in range(len(z)):
        if i in dead:continue
        b=tuple(sorted(int(v) for v in graph.blockers[i] if int(v) not in dead))
        if len(b)>horizon:continue
        v=z[i].astype(np.float64)
        stat=np.concatenate(((v[:,None]*v[None,:])[tri],v*float(y[i]),[1.0]))
        for key,sign in [(b,1),(tuple(sorted((*b,i))),-1)]:
            if len(key)>horizon:continue
            if key not in coeff:coeff[key]=np.zeros_like(stat)
            coeff[key]+=sign*stat
    return {k:v for k,v in coeff.items() if np.any(v!=0)}

def error(a,b):return float(np.max(np.abs(a-b),initial=0))

def run_numeric(rows,d,threshold,requests_limit=None):
    z,features_meta=lexical_engineering_features(rows,d)
    y=np.array([r['label'] for r in rows],dtype=np.float64)
    ids=[r['record_id'] for r in rows];k=min(8,len(rows))
    graph_z,graph_features_meta=lexical_engineering_features(rows,128)
    graph=build_blocker_graph(graph_z,ids,threshold)
    requests=fixed_requests(graph,k)
    if requests_limit is not None:requests=requests[:requests_limit]
    run_dir=OUT/f'civil_d{d}';run_dir.mkdir(parents=True,exist_ok=True)
    write_json(run_dir/'requests.json',requests) # fixed before any outcomes
    write_json(run_dir/'features.json',{'learner':features_meta,'curator':graph_features_meta,'graph_fixed_across_learner_dimensions':True})
    forecast=preflight(graph.blockers,d,1,k).to_dict()
    write_json(run_dir/'memory_preflight.json',forecast)
    record_file=run_dir/'checkpoint_registry.jsonl'
    failures=[];all_rows=[];initial_base_account=None;initial_summary_account=None;initial=set(map(int,graph.selected_indices()))
    started=time.perf_counter()
    for request in requests:
        try:
            start=time.perf_counter();base=EligiblePayloadState(graph,z,y,k)
            base_setup=time.perf_counter()-start
            start=time.perf_counter();summary=IndexedRidgeSummary.build(z,y,graph.blockers,k,max_coefficient_bytes=512*1024**2)
            summary_setup=time.perf_counter()-start
            initial_base_account=base.accounting();initial_summary_account=summary.accounting()
            write_json(run_dir/'initial_accounting.json',{'B-E':initial_base_account,'P-I':initial_summary_account})
            cumulative=[];position=0
            checkpoints=sorted(set(min(n,len(request['indices'])) for n in [1,2,4,k]))
            for checkpoint in checkpoints:
                batch=request['indices'][position:checkpoint];position=checkpoint;cumulative+=batch
                names=[ids[i] for i in batch]
                t=time.perf_counter();base.delete(names);base_repair=time.perf_counter()-t
                t=time.perf_counter();meter=summary.delete(batch);summary_repair=time.perf_counter()-t
                t=time.perf_counter();base_sol=base.decode(.01);base_solve=time.perf_counter()-t
                t=time.perf_counter();summary_sol=summary.decode(.01);summary_solve=time.perf_counter()-t
                t=time.perf_counter();oracle=direct_oracle(z,y,ids,threshold,.01,[ids[i] for i in cumulative],graph.priority,graph_features=graph_z)
                oracle_seconds=time.perf_counter()-t
                bm=base.moments();sm=summary.moments();expected=oracle.moments
                selected_equal=set(base.selected_ids())==set(oracle.selected_ids)
                count_equal=bm.count==sm.count==expected.count
                gram_err=max(error(bm.gram,expected.gram),error(sm.gram,expected.gram))
                cross_err=max(error(bm.cross,expected.cross),error(sm.cross,expected.cross))
                be_head=error(base_sol.weights,oracle.solution.weights)
                pi_head=error(summary_sol.weights,oracle.solution.weights)
                base.check_invariants();summary.check_invariants()
                state_audit=None
                if request is requests[0] and d<=128:
                    fresh=independent_coefficients(z,y,graph,cumulative,k-len(cumulative));current=summary.coefficients()
                    keys_equal=set(fresh)==set(current)
                    coef_err=max((error(fresh[key],current[key]) for key in set(fresh)&set(current)),default=0.0)
                    state_audit={'keys_equal':keys_equal,'max_abs_coefficient_error':coef_err,'remaining_horizon':summary.horizon}
                    if not keys_equal or coef_err>1e-9:raise AssertionError(f'coefficient audit mismatch {state_audit}')
                if not selected_equal or not count_equal or max(gram_err,cross_err)>1e-9 or max(be_head,pi_head)>1e-8:
                    raise AssertionError('independent oracle discrepancy beyond engineering diagnostic gate')
                admitted=len(set(map(int,oracle.selected_indices))-initial)
                row={'dimension':d,'curation':'fixed_128d_lexical_engineering_only','threshold':threshold,
                  'family':request['family'],'trajectory_id':request['id'],'cumulative_deleted':len(cumulative),
                  'remaining_horizon':summary.horizon,'initial_selected':len(initial),'selected_count':expected.count,
                  'admitted_count':admitted,'baseline_selected_ids_equal_oracle':selected_equal,'counts_equal':count_equal,
                  'max_abs_gram_error':gram_err,'max_abs_cross_error':cross_err,
                  'B-E_max_abs_head_error':be_head,'P-I_max_abs_head_error':pi_head,
                  'B-E_residual_fro':base_sol.normal_equation_residual_fro,
                  'P-I_residual_fro':summary_sol.normal_equation_residual_fro,
                  'B-E_setup_seconds':base_setup,'P-I_setup_seconds':summary_setup,
                  'B-E_repair_seconds':base_repair,'P-I_repair_seconds':summary_repair,
                  'B-E_solve_seconds':base_solve,'P-I_solve_seconds':summary_solve,
                  'oracle_seconds':oracle_seconds,'summary_update_counters':meter,
                  'state_audit':state_audit,'status':'passed_diagnostic_checks',
                  'timing_scope':'unreplicated_development_process_not_publishable_performance',
                  'numerical_scope':'FP64_diagnostic_fidelity_not_verified_interval_certificate'}
                all_rows.append(row)
                with record_file.open('a') as f:f.write(json.dumps(row)+'\n')
            del summary,base
        except Exception as exc:
            failed={'dimension':d,'trajectory_id':request['id'],'status':'failed','error':type(exc).__name__,'message':str(exc)}
            failures.append(failed)
            with record_file.open('a') as f:f.write(json.dumps(failed)+'\n')
    report={'dimension':d,'record_count':len(rows),'trajectories_planned':len(requests),
      'checkpoint_rows_passed':len(all_rows),'failed_trajectories':failures,
      'max_abs_head_error':max((max(r['P-I_max_abs_head_error'],r['B-E_max_abs_head_error']) for r in all_rows),default=0.0),
      'max_abs_moment_error':max((max(r['max_abs_gram_error'],r['max_abs_cross_error']) for r in all_rows),default=0.0),
      'checkpoint_rows_with_admissions':sum(r['admitted_count']>0 for r in all_rows),
      'maximum_admitted_count':max((r['admitted_count'] for r in all_rows),default=0),
      'initial_accounting':{'B-E':initial_base_account,'P-I':initial_summary_account},
      'elapsed_seconds':time.perf_counter()-started,'confirmatory':False}
    write_json(run_dir/'summary.json',report)
    return report

def isolation_check(rows):
    z,_=lexical_engineering_features(rows,128);y=np.array([r['label'] for r in rows])
    ids=[r['record_id'] for r in rows];g=build_blocker_graph(z,ids,.6)
    state=IndexedRidgeSummary.build(z,y,g.blockers,8)
    state_path=OUT/'isolation_initial_state.npz';state.save(state_path)
    batches=[[0],[1],[2,3],[4,5,6,7]]
    write_json(OUT/'isolation_requests.json',{'batches':batches,'lambda':.01})
    expected=[]
    for batch in batches:
        state.delete(batch);expected.append(state.solve_ridge(.01))
    result_path=OUT/'isolation_worker_results.json'
    proc=subprocess.run([sys.executable,str(ROOT/'repair_worker.py'),'--state',str(state_path),
       '--requests',str(OUT/'isolation_requests.json'),'--output',str(result_path)],capture_output=True,text=True)
    if proc.returncode:raise RuntimeError('isolated worker failed: '+proc.stderr[-2000:])
    got=json.loads(result_path.read_text())
    diffs=[error(a,np.array(b['weights'])) for a,b in zip(expected,got['outputs'])]
    report={'worker_returncode':proc.returncode,'checkpoint_count':len(diffs),'max_abs_head_error':max(diffs,default=0),
            'io_attempts_during_guarded_repair':got['io_attempts_during_repair'],
            'saved_current_state_bytes':state_path.stat().st_size,
            'scope':'fresh_process_state_only_replay_with_Python_IO_guard_not_OS_security_or_privacy_proof'}
    if len(diffs)!=len(batches) or max(diffs,default=0)>1e-10 or got['io_attempts_during_repair']:
        raise AssertionError('isolated replay mismatch')
    write_json(OUT/'isolation_summary.json',report)
    state_path.unlink() # restore point is reproducible; avoid retaining a redundant payload-sized file
    return report

def structure_panels(civil,news):
    from ccu.structure import structural_census
    outputs=[]
    for name,rows in [('civil_comments',civil),('cc_news',news)]:
        z,meta=lexical_engineering_features(rows,128)
        ids=[r['record_id'] for r in rows]
        sources=[r.get('source_id') or ('unknown-singleton:'+r['record_id']) for r in rows]
        for threshold in [.35,.60,.85]:
            graph=build_blocker_graph(z,ids,threshold,source_ids=sources)
            # structural_census API is integrated once the module audit is complete.
            for mode in (['record'] if name=='civil_comments' else ['record','source']):
                universe=len(set(ids if mode=='record' else sources))
                horizons=sorted(set(min(h,universe) for h in [0,1,2,4,8]))
                out=structural_census(graph,horizons,unit=mode,sampling_frame=sorted(set(sources)) if mode=='source' else None,dimensions=(128,768),outputs=(1,20))
                outputs.append({'corpus':name,'threshold':threshold,'mode':mode,
                    'curation_dimension':128,'curation':'lexical_unvalidated_preview_engineering_only',
                    'response_dimensions_for_forecast':'symbolic_C1_C20_byte_forecasts_only_no_news_labels_or_training',
                    'structure':out})
    write_json(OUT/'structure_census.json',outputs)
    return {'configurations':len(outputs),'confirmatory':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-native',action='store_true',help='Omit the 768-dimensional lexical stress check; this flag does not refer to E5.')
    parser.add_argument('--skip-structure',action='store_true',help='Run Civil numerical/replay checks only, without loading or hashing the optional News cache.')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    civil=read_natural_jsonl(ROOT/'data/civil_comments_engineering_preview.jsonl',require_labels=True)
    # Packaging-only amendment: the portable bundle omits News article text.
    # A Civil-only replay must not require or fingerprint an unused News cache.
    news=None if args.skip_structure else read_natural_jsonl(ROOT/'data/cc_news_engineering_preview.jsonl',require_sources=True)
    input_paths=[ROOT/'data/civil_comments_engineering_preview.jsonl']
    if news is not None:input_paths.append(ROOT/'data/cc_news_engineering_preview.jsonl')
    design={'date':'2026-10-04','scope':'nonconfirmatory_software_engineering_only','data_records':{'civil':len(civil),'news':None if news is None else len(news)},
      'input_sha256':{p.name:sha256_file(p) for p in input_paths},
      'features':'fixed_unfitted_word_bigram_hashing','dimensions':[64,128]+([] if args.skip_native else [768]),
      'numeric_threshold':.6,'structure_thresholds':[.35,.6,.85],
      'lambda':.01,'horizon':8,'checkpoints':[1,2,4,8],'R_trajectories':16,'U_trajectories':16,'A_candidates_max':8,
      'dimension_768_lexical_trajectories':1,'curation_dimension_fixed':128,'engineering_revision':'v1.1_fixed_graph_strict_requests','no_test_holdout_utility_claim':True,'no_human_validation_claim':True,'no_E5_claim':True,
      'no_synthetic_documents_or_labels':True,'summary_numeric_cap_bytes':512*1024**2}
    write_json(OUT/'engineering_design_lock.json',design)
    numeric=[]
    for d in design['dimensions']:
        print(f'Running real Civil engineering panel d={d}',flush=True)
        # Registry fresh per invocation; previous results live in versioned saved bundles.
        registry=OUT/f'civil_d{d}/checkpoint_registry.jsonl'
        if registry.exists():registry.unlink()
        numeric.append(run_numeric(civil,d,.6,requests_limit=1 if d==768 else None))
    isolation=isolation_check(civil)
    structure=None if args.skip_structure else structure_panels(civil,news)
    report={'status':'completed_engineering_pilot','scope':__doc__,'numeric':numeric,'isolation':isolation,
      'structure':structure,'process_peak_RSS_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'peak_scope':'whole_pilot_parent_process_not_attributable_to_one_method',
      'primary_study_blockers':['full_original_source_rich_datasets_not_locally_available',
        'E5_and_MPNet_weights_and_inference_runtime_not_available','human_semantic_quality_validation_not_completed',
        'native_dense_full_panel_memory_exceeds_current_host_if_forecast_requires_it'],
      'no_remote_compute_used':True}
    write_json(OUT/'execution_summary.json',report)
    print(json.dumps({'status':report['status'],'passed_checkpoint_rows':sum(x['checkpoint_rows_passed'] for x in numeric),
      'failed_trajectories':sum(len(x['failed_trajectories']) for x in numeric),'isolation':isolation,'structure':structure},indent=2))
    return int(any(x['failed_trajectories'] for x in numeric))

if __name__=='__main__':sys.exit(main())
