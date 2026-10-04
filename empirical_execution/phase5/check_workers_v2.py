"""Nine methods x two solvers in real restricted workers; natural texts only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import hashlib,json
import numpy as np
from phase5.run_isolated_v2 import run_service,prepare_bundle
from phase5.methods import METHODS
from ccu.data import read_natural_jsonl,lexical_engineering_features
from ccu.core import ridge_moments
from phase4.execution import independent_edges
from phase5.decoders import decode

def run(destination):
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    rows=read_natural_jsonl(ROOT/'data/civil_comments_engineering_preview.jsonl',require_labels=True)[:20]
    ids=[r['record_id'] for r in rows];cx,_=lexical_engineering_features(rows,32);x,_=lexical_engineering_features(rows,16)
    y=np.asarray([r['label'] for r in rows],np.float64)[:,None]
    bundle=prepare_bundle(cx,x,y,ids,out/'input_bundle',threshold=.6)
    requests=[[ids[0]],[ids[1],ids[2]]];reports=[];checks=[]
    for solver in ('cholesky','cg'):
        for method in METHODS:
            target=out/f'{method}_{solver}';r=run_service(bundle,method,requests,target,horizon=3,solver=solver)
            if not r['success']:
                reports.append({'method':method,'solver':solver,'success':False,'service_report':str(target/'service_report.json')})
                continue
            rr=json.loads((target/'repair/repair_report.json').read_text());cr=json.loads((target/'construction/construction_report.json').read_text())
            dead=set();differences=[];flags={
                'kernel_TSYNC':rr['isolation']['seccomp_thread_synchronization']=='TSYNC',
                'existing_file_denied':rr['isolation']['denied_existing_file_probes']>0,
                'original_head_gate':cr['numerical_decoder']['release_allowed'],
                'native_hashes':bool(rr['loaded_native_dependencies']['files']) and all('sha256' in p for p in rr['loaded_native_dependencies']['files']),
                'input_read_meter_nonnegative':rr['snapshot_hash_and_load_read_meter']['read_call_bytes']>=rr['input_snapshot_logical_bytes'],
                'output_meter_positive':rr['output_write_call_bytes_before_report']>0}
            for i,batch in enumerate(requests):
                dead.update(batch);_,ix=independent_edges(cx,ids,.6,dead);fresh=ridge_moments(x[ix],y[ix]);want=decode(fresh,.01)
                got=np.load(target/f'repair/head_{i:04d}.npy');differences.append(float(np.linalg.norm(got-want.weights)))
                flags[f'head_{i}_oracle_agreement']=differences[-1]<1e-7
                flags[f'head_{i}_count']=rr['releases'][i]['count']==len(ix)
                flags[f'head_{i}_gate']=rr['releases'][i]['numerical_decoder']['release_allowed'] and rr['releases'][i]['numerical_decoder']['normalized_residual_eta']<=1e-10
            checks.extend({'case':f'{method}_{solver}','check':key,'passed':value} for key,value in flags.items())
            reports.append({'method':method,'solver':solver,'success':all(flags.values()),'oracle_head_error_fro':differences,'releases':rr['releases'],'isolation':rr['isolation']})
            print(method,solver,all(flags.values()),flush=True)
    report={'schema':'ccu-worker-v2-checks-1','evidence_role':'natural_Civil20_lexical_decoder_integration_only',
            'cases':reports,'all_passed':all(r['success'] for r in reports),'checks':checks,'check_count':len(checks),
            'release_count':sum(len(r.get('releases',[])) for r in reports),'semantic_study_started':False,
            'maximum_oracle_head_error_fro':max((v for r in reports for v in r.get('oracle_head_error_fro',[])),default=None),
            'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('decoders.py','workers_v2.py','run_isolated_v2.py','check_workers_v2.py')}}
    (out/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['all_passed']:raise AssertionError('Retained failed v2 cell; inspect verification.json')
    return report
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--destination',required=True);args=p.parse_args();run(args.destination)
