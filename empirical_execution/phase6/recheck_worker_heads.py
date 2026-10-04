"""Recheck frozen saved heads after correcting the B-F-FP32 evaluator target."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import hashlib,json
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features
from phase4.execution import independent_edges
from phase6.check_workers import dual_head

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(directory,destination):
    directory=Path(directory);destination=Path(destination)
    if destination.exists():raise FileExistsError('Preserve prior evaluator evidence')
    old=json.loads((directory/'verification.json').read_text())
    rows=read_natural_jsonl(ROOT/'data/civil_comments_engineering_preview.jsonl',require_labels=True)[:20]
    ids=[r['record_id'] for r in rows];cx,_=lexical_engineering_features(rows,32)
    y=np.asarray([r['label'] for r in rows],np.float64)[:,None]
    arrays={d:lexical_engineering_features(rows,d)[0] for d in (16,768)}
    _,original=independent_edges(cx,ids,.6,set());cases=[];checks=[];head_count=0
    for oldcase in old['cases']:
        name=oldcase['case'];base=directory/name;m=oldcase['method'];d=oldcase['dimension'];x=arrays[d]
        config=json.loads((base/'prospective_configuration.json').read_text());report=json.loads((base/'repair/repair_report.json').read_text())
        initial=np.load(base/'construction/initial_head.npy');initial_error=float(np.linalg.norm(initial-dual_head(x,y,original)))
        errors=[];dead=set();target_changes=[]
        checks.append({'case':name,'check':'initial_shape_and_finite','passed':initial.shape==(d,1) and bool(np.isfinite(initial).all())})
        if not m.endswith('-FP32'):checks.append({'case':name,'check':'initial_FP64_dual_agreement','passed':initial_error<1e-7})
        head_count+=1
        for i,batch in enumerate(config['requests']):
            dead.update(batch);_,full=independent_edges(cx,ids,.6,dead)
            ix=[j for j in original if ids[j] not in dead] if m in ('B-F','B-F-FP32') else full
            head=np.load(base/f'repair/head_{i:04d}.npy');err=float(np.linalg.norm(head-dual_head(x,y,ix)));errors.append(err)
            target_changes.append(len(set(full)-set(ix)))
            row=report['releases'][i]
            flags={'count':row['count']==len(ix),'head_hash':sha(base/f'repair/head_{i:04d}.npy')==row['head_sha256'],
                   'gate':row['numerical_decoder']['release_allowed'] and row['numerical_decoder']['normalized_residual_eta']<=1e-10}
            if not m.endswith('-FP32'):flags['FP64_dual_agreement']=err<1e-7
            checks.extend({'case':name,'check':f'{key}_{i}','passed':bool(value)} for key,value in flags.items());head_count+=1
        cases.append({'case':name,'method':m,'initial_head_error_fro':initial_error,'checkpoint_head_error_fro':errors,
            'target':'retained_original_selection' if m in ('B-F','B-F-FP32') else 'full_counterfactual_selection',
            'counterfactual_additions_absent_from_wrong_target':target_changes})
    result={'schema':'ccu-worker-head-evaluator-correction-1','all_passed':all(c['passed'] for c in checks),
        'old_result_sha256':sha(directory/'verification.json'),'old_evaluator_sha256':old['source_sha256']['check_workers.py'],
        'old_evaluator_preserved_sha256':sha(Path(__file__).parent/'results/worker_evaluator_before_target_fix.txt'),
        'corrected_evaluator_sha256':sha(Path(__file__).parent/'check_workers.py'),'recheck_source_sha256':sha(__file__),
        'worker_production_sources_unchanged':all(sha(Path(__file__).parent/n)==h for n,h in old['source_sha256'].items() if n!='check_workers.py'),
        'original_archive_modified':False,'service_rerun':False,'head_count':head_count,'checkpoint_count':2*len(cases),
        'check_count':len(checks),'checks':checks,'cases':cases,
        'maximum_FP64_checkpoint_head_error_fro':max(v for c in cases if not c['method'].endswith('-FP32') for v in c['checkpoint_head_error_fro']),
        'maximum_FP32_checkpoint_head_error_fro':max(v for c in cases if c['method'].endswith('-FP32') for v in c['checkpoint_head_error_fro']),
        'scope':'same_Civil20_lexical_saved_heads;no_new_empirical_dataset_or_primary_semantic_result'}
    if not result['worker_production_sources_unchanged']:raise ValueError('Worker source changed; do not claim source-bound recheck')
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not result['all_passed']:raise AssertionError('Corrected target recheck failed')
    print(json.dumps({k:result[k] for k in ('all_passed','head_count','check_count','maximum_FP64_checkpoint_head_error_fro','maximum_FP32_checkpoint_head_error_fro')},indent=2))
    return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--destination',required=True);a=p.parse_args();run(a.directory,a.destination)
