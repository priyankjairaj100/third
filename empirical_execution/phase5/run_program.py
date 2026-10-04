#!/usr/bin/env python3
"""Prospectively locked natural-text development matrix; never a primary study.

Every requested job survives in the ledger, including killed/failed jobs. The
large, reproducible service snapshots are temporary; exact reports, state byte
counts/hashes, stdout/stderr and released heads remain in the result archive.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,hashlib,importlib.util,json,os,platform,random,shutil,tempfile,time
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from phase4.execution import independent_edges
from phase5.run_isolated import prepare_bundle,run_service,DEFAULT_POLICY,METHODS,write

def dual_head(x,y,selected,lam):
    if not selected:return np.zeros((x.shape[1],y.shape[1]))
    z=x[selected].astype(np.float64);t=y[selected]
    return z.T@np.linalg.solve(z@z.T+lam*len(z)*np.eye(len(z)),t)

def dependency_hashes():
    paths=[ROOT/'phase5'/n for n in ('methods.py','payload.py','workers.py','run_isolated.py','run_program.py')]
    paths+=sorted((ROOT/'ccu').glob('*.py'))
    paths+= [ROOT/'phase3/reference_graph.py',ROOT/'phase4/execution.py']
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in paths}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/civil_comments_engineering_preview.jsonl'
    request_file=ROOT/'phase4/results/execution_engineering_final/d64/requests.json'
    manifest=json.loads(request_file.read_text());wanted=('R000','R001','U000','A000')
    paths=[next(t for t in manifest['trajectories'] if t['trajectory_id']==name) for name in wanted]
    rows=read_natural_jsonl(source,require_labels=True);ids=[r['record_id'] for r in rows]
    cx,feature_meta=lexical_engineering_features(rows,128)
    y=np.asarray([r['label'] for r in rows],dtype=np.float64)[:,None]
    jobs=[];rng=random.Random(2027100405)
    for dimension,repeats in ((64,5),(768,1)):
        for repeat in range(repeats):
            for path in paths:
                order=list(METHODS);rng.shuffle(order)
                for method in order:
                    jobs.append({'job_id':f'd{dimension}-r{repeat}-{path["trajectory_id"]}-{method}',
                                 'dimension':dimension,'repeat':repeat,'path':path['trajectory_id'],'method':method})
    locked=dependency_hashes()
    write(out/'prospective_lock.json',{
        'schema':'ccu-phase5-development-matrix-1','role':'reused_natural_lexical_engineering_only',
        'primary_study':False,'source_sha256':sha256_file(source),'request_manifest_sha256':sha256_file(request_file),
        'requests':[{'trajectory_id':t['trajectory_id'],'arm':t['arm'],'requests':[o['newly_requested_units'] for o in t['observations']]} for t in paths],
        'threshold':.6,'lambda_reg':.01,'horizon':8,'unit':'record','checkpoints':[1,2,4,8],
        'curator_features':feature_meta,'dimensions':[64,768],'repetitions':{'64':5,'768':1},
        'tolerance':1e-9,'policy':DEFAULT_POLICY,'persistence':'final','compaction_fraction':.5,
        'job_order':jobs,'method_order_seed':2027100405,'dependency_hashes':locked,
        'independent_oracle':'scalar ordered pair scorer plus dual ridge; outside method timer',
        'large_state_archive_policy':'temporary snapshots discarded after byte/hash reports retained; exact regeneration available',
        'genuine_source_arms':0,'reason_no_source_arms':'Civil preview has no native source metadata',
        'synthetic_empirical_data':False,'no_primary_runtime_or_speed_claim':True})
    write(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),
        'numpy':np.__version__,'optional_packages':{n:importlib.util.find_spec(n) is not None for n in ('torch','transformers','faiss','tqdm','datasets','pyarrow','safetensors')},
        'free_disk_bytes_at_start':shutil.disk_usage(out).free,'network_downloads':False})
    # Freeze oracle selections independently; never send these IDs into a repair worker.
    selections={};_,initial=independent_edges(cx,ids,.6)
    for t in paths:
        previous=set(initial)
        for i,o in enumerate(t['observations']):
            _,selected=independent_edges(cx,ids,.6,deleted=o['deleted_record_ids'])
            admissions=len(set(selected)-previous);previous=set(selected)
            selections[(t['trajectory_id'],i)]=(selected,admissions)
    feature_cache={};oracle_cache={};bundle_cache={};records=[]
    tmp_root=ROOT.parent/'tmp';tmp_root.mkdir(exist_ok=True)
    scratch=Path(tempfile.mkdtemp(prefix='phase5-program-',dir=tmp_root))
    began=time.perf_counter()
    for number,job in enumerate(jobs):
        if dependency_hashes()!=locked:raise RuntimeError('Locked execution source changed; preserve incomplete ledger')
        d=job['dimension'];path=next(t for t in paths if t['trajectory_id']==job['path'])
        if d not in feature_cache:
            x,meta=lexical_engineering_features(rows,d);feature_cache[d]=x
            bundle_cache[d]=prepare_bundle(cx,x,y,ids,scratch/f'bundle_d{d}',threshold=.6)
            write(out/f'learner_d{d}.json',meta)
            shutil.copyfile(bundle_cache[d]/'bundle.json',out/f'bundle_d{d}.json')
            for t in paths:
                for i,_ in enumerate(t['observations']):
                    oracle_cache[(d,t['trajectory_id'],i)]=dual_head(x,y,selections[(t['trajectory_id'],i)][0],.01)
        run=scratch/job['job_id'];archive=out/'jobs'/job['job_id'];archive.mkdir(parents=True)
        row={**job,'status':'started','head_checks':[]}
        try:
            report=run_service(bundle_cache[d],job['method'],[o['newly_requested_units'] for o in path['observations']],run,horizon=8)
            row['success']=report['success'];row['status']='passed' if report['success'] else 'worker_failure'
            row['construction']=report['construction'];row['repair']=report['repair']
            if report['success']:
                repaired=json.loads((run/'repair/repair_report.json').read_text())
                row['isolation']=repaired['isolation']
                for i,o in enumerate(path['observations']):
                    got=np.load(run/f'repair/head_{i:04d}.npy',allow_pickle=False)
                    expected=oracle_cache[(d,job['path'],i)];sel,admissions=selections[(job['path'],i)]
                    error=float(np.max(np.abs(got-expected),initial=0))
                    count=repaired['releases'][i]['count']
                    check={'checkpoint':o['checkpoint'],'maximum_absolute_head_error':error,
                           'count':count,'oracle_count':len(sel),'admissions':admissions,
                           'zero_admission':admissions==0,'passed':error<=1e-9 and count==len(sel)}
                    row['head_checks'].append(check)
                if not all(c['passed'] for c in row['head_checks']):row['status']='oracle_disagreement'
            row['lifecycle']=json.loads((run/'persistence_receipt.json').read_text())
        except Exception as exc:
            row['status']='orchestrator_failure';row['exception_type']=type(exc).__name__;row['message']=str(exc)
        finally:
            # Preserve all small exact outputs including errors. State bytes/hash
            # accounting stays in reports; state deletion is archive cleanup only.
            for f in sorted(run.rglob('*')) if run.exists() else []:
                if f.is_file() and not f.name.startswith('state.npz'):
                    target=archive/f.relative_to(run);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
            write(archive/'audit.json',row)
            if run.exists():shutil.rmtree(run)
        records.append(row)
        with (out/'job_ledger.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({'completed':number+1,'planned':len(jobs),'job_id':job['job_id'],'status':row['status']}),flush=True)
    shutil.rmtree(scratch)
    checks=[c for r in records for c in r['head_checks']]
    summary={'scope':'natural Civil100 lexical development; primary study unexecuted',
        'planned_jobs':len(jobs),'observed_jobs':len(records),'passed_jobs':sum(r['status']=='passed' for r in records),
        'failed_jobs':sum(r['status']!='passed' for r in records),'missing_jobs':len(jobs)-len(records),
        'head_checks':len(checks),'oracle_disagreements':sum(not c['passed'] for c in checks),
        'zero_admission_method_checkpoint_rows':sum(c['zero_admission'] for c in checks),
        'maximum_absolute_head_error':max((c['maximum_absolute_head_error'] for c in checks),default=None),
        'elapsed_including_oracle_and_archive_seconds':time.perf_counter()-began,
        'source_evidence':False,'semantic_evidence':False,'paper_speedup_claim':False,
        'dependency_hashes':locked,'all_requested_cells_preserved':True}
    write(out/'summary.json',summary);print(json.dumps(summary,indent=2))
    if summary['failed_jobs']:raise SystemExit(1)

if __name__=='__main__':main()
