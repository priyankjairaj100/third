"""Fresh validation after successful v4 selection; full score stream replay."""
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import primitives as v

W=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace')
REPO=W.parent/'repo'
HERE=Path(__file__).resolve().parent
ROOT=W/'qualification/llm-only-v4-validation-sampling-a01'
V4=W/'tools/model-annotation-v4-20261014'
V4_SHA='959d32cfeb07598d7eb453f1564f2f4f1566227e62792bd9fab5e38a69dedcff'
INFERENCE=W/'tools/llm-v4-inference-v1-20261014'
INFERENCE_SHA='ef7449595cb5a90e8fc8a009103f958eb92a73abc7943d52c19a552e816c4207'
SELECTION=W/'qualification/llm-only-v4-selection-a01'
SELECTION_INFERENCE=W/'qualification/llm-only-v4-inference-a01'
STATISTICS=SELECTION_INFERENCE/'statistics/run/result.json'

def binding(path):
    raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bootstrap(folder,pin):
    v.require(binding(folder/'package_manifest.json')['sha256']==pin,'package_pin')
    for name,b in json.loads((folder/'package_manifest.json').read_bytes())['files'].items():
        v.require(Path(name).name==name and name!='package_manifest.json','safe_package_member')
        f=folder/name;v.require(f.is_file() and not f.is_symlink() and binding(f)==b,'package_member')
def module(name,path):
    v.require(name not in sys.modules,'fresh_module')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def setup(pin):
    bootstrap(HERE,pin);bootstrap(V4,V4_SHA);bootstrap(INFERENCE,INFERENCE_SHA)
    sys.path.insert(0,str(V4))
    import protocol as p
    import select_worker as original
    original.bootstrap(original.TAIL,original.TAIL_SHA);sys.path.insert(0,str(original.TAIL))
    import census as t
    t.verify(t.CONTROL,t.CONTROL_SHA);t.verify(t.SAMPLER,t.SAMPLER_SHA)
    c=t.module('v4_validation_legacy_control',t.CONTROL/'control.py')
    s=t.module('v4_validation_frame',t.SAMPLER/'sampler.py')
    plan=s.read_json(V4/'PLAN.json');p.validate(plan,V4)
    return p,original,t,c,s,plan

def recheck_selection(s,watch):
    report=s.read_json(STATISTICS);s.verify_seal(report,True)
    v.require(report['passed'] is True and report['package_sha256']==INFERENCE_SHA,'successful_selection_analysis')
    source=s.read_json(SELECTION/'run/result.json');s.verify_seal(source,True)
    watch.add(SELECTION/'run/result.json',report['sampling_result'])
    rows=[json.loads(x) for x in (SELECTION/'run/selection.private.jsonl').read_bytes().splitlines()]
    watch.add(SELECTION/'run/selection.private.jsonl',source['files']['selection.private.jsonl'])
    observations=[]
    for alias in ('qwen','mistral'):
        path=SELECTION_INFERENCE/('audit_'+alias)/'run/result.json'
        watch.add(path,report['audits'][alias]);audit=s.read_json(path);s.verify_seal(audit,True)
        v.require(audit['passed'] is True and audit['package_sha256']==INFERENCE_SHA,'selection_audit')
        observations.extend(audit['canonical_observations'])
    numeric=module('v4_validation_selection_numeric',INFERENCE/'statistics_core.py')
    calculated=numeric.selection(rows,source['populations'],observations)
    v.require(calculated==report['statistics'] and calculated['selection_passed'] is True,'reproduced_positive_selection')
    return report,rows

def independent_reference(vectors,panels,tau,rng):
    v.require(len(vectors)<=512,'bounded_full_matrix')
    n=len(vectors);z=panels.graph_normalize(vectors);matrix=panels.reference_cosines(z,z)
    samples=[];count=total=0;digest=hashlib.sha256()
    for left in range((n+255)//256):
        for right in range(left,(n+255)//256):
            for i in range(left*256,min((left+1)*256,n)):
                for j in range(max(i+1,right*256),min((right+1)*256,n)):
                    score=float(matrix[i,j]);total+=1
                    digest.update((str(i)+'\t'+str(j)+'\t'+float.hex(score)+'\n').encode('ascii'))
                    if score<=tau:continue
                    count+=1
                    if count<=200:samples.append((i,j,score))
                    else:
                        k=rng.randrange(count)
                        if k<200:samples[k]=(i,j,score)
    return {'population':count,'pair_count':total,'stream_sha256':digest.hexdigest(),
            'raw':sorted(samples,key=lambda x:(x[0],x[1]))}

def main():
    parser=argparse.ArgumentParser()
    for name in ('package-sha256','validation-lock-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();p,original,t,c,s,plan=setup(args.package_sha256)
    watch=s.Watch()
    for folder,pin in ((HERE,args.package_sha256),(V4,V4_SHA),(INFERENCE,INFERENCE_SHA),
                       (original.TAIL,original.TAIL_SHA),(t.CONTROL,t.CONTROL_SHA)):
        s.verify_package(folder,pin,watch)
    lb=watch.add(ROOT/'validation_lock.json')
    v.require(lb['sha256']==args.validation_lock_sha256,'validation_lock_pin')
    lock=s.read_json(ROOT/'validation_lock.json');s.verify_seal(lock,True)
    v.require(lock['root']==str(ROOT) and lock['package_sha256']==args.package_sha256,'validation_slot_package')
    watch.add(STATISTICS,lock['selection_statistics']);report,selection_rows=recheck_selection(s,watch)
    v.require(lock['locked_tau']==report['statistics']['selected_tau'],'locked_threshold')
    tau=Fraction(lock['locked_tau']['numerator'],lock['locked_tau']['denominator'])
    seeds_binding=watch.add(SELECTION/'sampling_lock.json',lock['sampling_lock'])
    seeds=s.read_json(SELECTION/'sampling_lock.json');s.verify_seal(seeds,True)
    plan_sha=p.binding(V4/'PLAN.json')['sha256']
    v.require(seeds['package_sha256']==V4_SHA and seeds['plan_sha256']==plan_sha and
              lock['plan_sha256']==plan_sha,'original_seed_lineage')
    tail=s.read_json(original.TAIL_RESULT);s.verify_seal(tail,True)
    watch.add(original.TAIL_RESULT,seeds['tail_result'])
    a=original.context(t,c);os.umask(0o077);out=ROOT/'run';out.mkdir(exist_ok=False)
    started=time.perf_counter();completion={'passed':False,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    try:
        pins,own,parent,legacy,cal,panels,runtime,frame,rows,vectors=s.load_context(a,watch)
        frame_sha=pins['frame_result']['sha256']
        make=lambda seed:v.generator(p,plan_sha,frame_sha,seed,'validation')
        diagnostic=hashlib.sha256(b'third-v4-validation-first512-parity-v1').hexdigest()
        bounded=v.collect(cal.iter_pair_scores(vectors[:512],256,{}),512,float(tau),make(diagnostic))
        reference=independent_reference(vectors[:512],panels,float(tau),make(diagnostic))
        v.require(bounded==reference,'independent_first512_score_and_sample_parity')
        s.publish(out/'bounded_parity.json',{'passed':True,'rows':512,'pair_count':bounded['pair_count'],
            'population':bounded['population'],'sample_size':len(bounded['raw']),
            'stream_sha256':bounded['stream_sha256'],'production_seed_used':False})
        audit={}
        def progress(count):print(json.dumps({'pairs':count,'elapsed_seconds':time.perf_counter()-started}),flush=True)
        result=v.collect(cal.iter_pair_scores(vectors,256,audit),len(vectors),float(tau),
                         make(seeds['seeds']['validation']),progress=progress)
        index=tail['threshold_grid'].index(float(tau))
        v.require(result['stream_sha256']==tail['observed']['score_stream_sha256'] and
            result['population']==tail['observed']['counts_strictly_above'][index],'full_tail_replay')
        sample=v.decorate(result['raw'],result['population'],rows,p,plan_sha,frame_sha,plan['inherits_unchanged']['models'])
        for i,j,score in result['raw']:
            actual=panels.reference_cosines(panels.graph_normalize(vectors[i:i+1]),panels.graph_normalize(vectors[j:j+1]))
            v.require(float(actual[0,0]).hex()==score.hex(),'every_sampled_score_replayed')
        swap_rng=v.generator(p,plan_sha,frame_sha,seeds['seeds']['validation_swap'],'validation_swap')
        swaps=s.swaps(sample,cal,swap_rng)
        overlap=sorted(set(r['opaque_pair_id'] for r in selection_rows)&set(r['opaque_pair_id'] for r in sample))
        s.closed_inventories(a,pins);watch.recheck()
        for name,items in [('validation.private.jsonl',sample),('validation_swap.private.jsonl',swaps)]:
            s.write_bytes(out/name,b''.join(s.canonical(x)+b'\n' for x in items),16<<20)
        s.publish(out/'input_bindings.json',{'files':watch.files})
        value=s.seal({'schema':'third-llm-v4-validation-sample-v1','passed':True,'stage':'complete',
            'created_utc':datetime.now(timezone.utc).isoformat(),'package_sha256':args.package_sha256,
            'plan_sha256':plan_sha,'frame_result':pins['frame_result'],'validation_lock':lb,
            'sampling_lock':seeds_binding,'selection_statistics':lock['selection_statistics'],
            'locked_tau':lock['locked_tau'],'population':result['population'],
            'selected_pairs':len(sample),'swap_pairs':len(swaps),'pair_count':result['pair_count'],
            'score_stream_sha256':result['stream_sha256'],'runtime':runtime,'numerical_audit':audit,
            'bounded_independent_parity':True,'all_selected_scores_pairwise_replayed':True,
            'full_above_threshold_frame':True,'label_filter':False,'new_entropy_draws':0,
            'selection_overlap_pairs':len(overlap),'overlap_requires_fresh_calls':True,
            'files':{name:s.binding(out/name) for name in ('validation.private.jsonl','validation_swap.private.jsonl',
                                                         'input_bindings.json','bounded_parity.json')},
            'model_inference_calls':0,'human_annotation_calls':0,'inference_activated':False,
            'semantic_validity_established':False,'confirmatory_study_ready':False,
            'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'elapsed_seconds':time.perf_counter()-started})
        s.publish(out/'result.json',value);completion['passed']=True
        print(json.dumps({'passed':True,'selected_pairs':len(sample),'population':result['population']}),flush=True)
    except BaseException as error:completion.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        completion['elapsed_seconds']=time.perf_counter()-started;s.publish(out/'process_completion.json',completion)

if __name__=='__main__':main()
