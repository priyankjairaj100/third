"""Slurm-only v4 sampler. Fresh locked sample; no model inference."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import protocol as p
import sampling as sampling

W = Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace')
ROOT = W/'qualification/llm-only-v4-selection-a01'
FOLDER = Path(__file__).resolve().parent
TAIL = W/'tools/llm-tail-census-v1-20261014'
TAIL_SHA = '8051be545c811c9d7c6636a3e6f0d6ce2ddb5b0bc79704a02cf0827ba389c887'
TAIL_RESULT = W/'qualification/llm-tail-census-v1-20261014-a01/run/result.json'

def bootstrap(folder, pin):
    raw=(folder/'package_manifest.json').read_bytes()
    p.require(hashlib.sha256(raw).hexdigest()==pin,'package_pin')
    for name,b in json.loads(raw)['files'].items():
        p.require(Path(name).name==name and name!='package_manifest.json','safe_member')
        f=folder/name
        p.require(f.is_file() and not f.is_symlink() and p.binding(f)==b,'package_member_binding')

def context(t, c):
    return SimpleNamespace(mode='diagnostic',repo=t.REPO,workspace=W,frame_dir=c.FRAME,
        frame_result_sha256=c.FRAME_SHA,protocol_package=c.PROTOCOL,frame_package=c.FRAME_PACKAGE,
        protocol_evidence=c.EVIDENCE,protocol_evidence_sha256=c.EVIDENCE_SHA,package_sha256=t.SAMPLER_SHA)

def expected_populations(tail, cuts):
    by_cut=dict(zip(tail['threshold_grid'],tail['observed']['counts_strictly_above']))
    counts=[tail['observed']['pair_count']-by_cut[cuts[0]]]
    counts += [by_cut[a]-by_cut[b] for a,b in zip(cuts,cuts[1:])]
    return counts+[by_cut[cuts[-1]]]

def independent_reference(vectors, panels, cuts, capacity, rngs):
    p.require(len(vectors)<=512,'bounded_reference')
    normalized=panels.graph_normalize(vectors)
    matrix=panels.reference_cosines(normalized,normalized)
    counts=[0]*(len(cuts)+1); reservoirs=[[] for _ in counts]; stream=hashlib.sha256();total=0
    for left in range((len(vectors)+255)//256):
        for right in range(left,(len(vectors)+255)//256):
            for i in range(left*256,min((left+1)*256,len(vectors))):
                for j in range(max(i+1,right*256),min((right+1)*256,len(vectors))):
                    score=float(matrix[i,j]);k=sum(score>threshold for threshold in cuts)
                    counts[k]+=1;total+=1
                    item=(i,j,score,k)
                    if counts[k]<=capacity:reservoirs[k].append(item)
                    else:
                        replace=rngs[k].randrange(counts[k])
                        if replace<capacity:reservoirs[k][replace]=item
                    stream.update((str(i)+'\t'+str(j)+'\t'+float.hex(score)+'\n').encode('ascii'))
    return {'populations':counts,'raw':sorted((x for r in reservoirs for x in r),key=lambda x:(x[0],x[1])),
            'stream_sha256':stream.hexdigest(),'pair_count':total}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package-sha256',required=True)
    parser.add_argument('--sampling-lock-sha256',required=True)
    args=parser.parse_args()
    bootstrap(FOLDER,args.package_sha256);bootstrap(TAIL,TAIL_SHA)
    sys.path.insert(0,str(TAIL));import census as t
    t.verify(t.CONTROL,t.CONTROL_SHA);t.verify(t.SAMPLER,t.SAMPLER_SHA)
    c=t.module('v4_legacy_control',t.CONTROL/'control.py')
    s=t.module('v4_frame_admission',t.SAMPLER/'sampler.py')
    a=context(t,c);watch=s.Watch();s.verify_package(FOLDER,args.package_sha256,watch)
    s.verify_package(TAIL,TAIL_SHA,watch);s.verify_package(t.CONTROL,t.CONTROL_SHA,watch)
    lock_binding=watch.add(ROOT/'sampling_lock.json')
    p.require(lock_binding['sha256']==args.sampling_lock_sha256,'external_sampling_lock_pin')
    lock=s.read_json(ROOT/'sampling_lock.json');s.verify_seal(lock,True)
    p.require(lock['package_sha256']==args.package_sha256 and lock['root']==str(ROOT),'lock_source_and_slot')
    plan=s.read_json(FOLDER/'PLAN.json');p.validate(plan,FOLDER)
    plan_sha=s.binding(FOLDER/'PLAN.json')['sha256']
    p.require(lock['plan_sha256']==plan_sha,'plan_lock')
    watch.add(TAIL_RESULT,lock['tail_result']);tail=s.read_json(TAIL_RESULT);s.verify_seal(tail,True)
    p.require(tail['passed'] is True and tail['all_220405510_pair_scores_replayed_exactly'] is True,'tail_replay')
    seeds=lock['seeds'];p.require(set(seeds)==set(plan['randomness']['seeds']),'all_four_seeds')
    p.require(len(set(seeds.values()))==4 and all(len(v)==64 and all(c in '0123456789abcdef' for c in v)
              for v in seeds.values()),'independent_seed_material')
    os.umask(0o077);out=s.safe(ROOT/'run');out.mkdir(exist_ok=False)
    started=time.perf_counter();completion={'passed':False,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    try:
        pins,own,parent,legacy_protocol,cal,panels,runtime,frame,rows,vectors=s.load_context(a,watch)
        cuts=plan['selection']['stratum_upper_cuts'];capacity=plan['selection']['capacity_per_stratum']
        frame_sha=pins['frame_result']['sha256']
        make=lambda seed:sampling.generators(plan_sha,frame_sha,seed,len(cuts)+1)
        diagnostic=hashlib.sha256(b'third-v4-first512-engineering-parity-v1').hexdigest()
        bounded=sampling.select(cal.iter_pair_scores(vectors[:512],256,{}),512,cuts,capacity,make(diagnostic))
        reference=independent_reference(vectors[:512],panels,cuts,capacity,make(diagnostic))
        p.require(bounded==reference,'first512_sampling_and_score_parity')
        s.publish(out/'bounded_parity.json',{'passed':True,'rows':512,'pair_count':bounded['pair_count'],
                   'populations':bounded['populations'],'sample_size':len(bounded['raw']),
                   'stream_sha256':bounded['stream_sha256'],'production_seed_used':False})
        audit={}
        def progress(n):print(json.dumps({'pairs':n,'elapsed_seconds':time.perf_counter()-started}),flush=True)
        result=sampling.select(cal.iter_pair_scores(vectors,256,audit),len(vectors),cuts,capacity,
                               make(seeds['selection']),progress)
        p.require(result['stream_sha256']==tail['observed']['score_stream_sha256'],'full_score_stream_replay')
        p.require(result['populations']==expected_populations(tail,cuts),'full_stratum_populations_replay')
        surrogate={'sha256':plan_sha,'models':plan['inherits_unchanged']['models']}
        sample=s.decorate(result['raw'],result['populations'],rows,p,surrogate,frame_sha)
        for item,(i,j,score,k) in zip(sample,result['raw']):
            actual=panels.reference_cosines(panels.graph_normalize(vectors[i:i+1]),
                                            panels.graph_normalize(vectors[j:j+1]))
            p.require(float(actual[0,0]).hex()==score.hex(),'every_selected_score_pairwise_replay')
            n=min(capacity,result['populations'][k]);w=Fraction(result['populations'][k],n)
            p.require(item['inclusion_weight']==s.fraction(w),'exact_inverse_weight')
        swap_rng=__import__('random').Random(int(p.domain_digest(p.PREFIX+'sampling',
            [plan_sha,frame_sha,'selection_swap',seeds['selection_swap'],None]),16))
        swaps=s.swaps(sample,cal,swap_rng)
        s.closed_inventories(a,pins);watch.recheck()
        for name,items in [('selection.private.jsonl',sample),('selection_swap.private.jsonl',swaps)]:
            raw=b''.join(s.canonical(x)+b'\n' for x in items)
            s.write_bytes(out/name,raw,16<<20)
        s.publish(out/'input_bindings.json',{'files':watch.files})
        report=s.seal({'schema':'third-llm-tail-selection-v4','passed':True,'stage':'complete',
            'created_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':plan_sha,
            'package_sha256':args.package_sha256,'sampling_lock':lock_binding,'frame_result':pins['frame_result'],
            'tail_result':lock['tail_result'],'runtime':runtime,'stratum_upper_cuts':cuts,
            'populations':result['populations'],'sample_counts':[min(capacity,n) for n in result['populations']],
            'selected_pairs':len(sample),'swap_pairs':len(swaps),'pair_count':result['pair_count'],
            'score_stream_sha256':result['stream_sha256'],'numerical_audit':audit,
            'bounded_independent_parity':True,'all_selected_scores_pairwise_replayed':True,
            'files':{name:s.binding(out/name) for name in ('selection.private.jsonl','selection_swap.private.jsonl',
                                                        'input_bindings.json','bounded_parity.json')},
            'model_inference_calls':0,'human_annotation_calls':0,'inference_activated':False,
            'threshold_selected':False,'semantic_validity_established':False,'confirmatory_study_ready':False,
            'sampling_performed':True,'sampling_scope':'post-v3 model-defined development follow-up',
            'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'elapsed_seconds':time.perf_counter()-started})
        s.publish(out/'result.json',report);completion['passed']=True
        print(json.dumps({'passed':True,'selected_pairs':len(sample),'swap_pairs':len(swaps)}),flush=True)
    except BaseException as error:
        completion.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        completion['elapsed_seconds']=time.perf_counter()-started
        s.publish(out/'process_completion.json',completion)

if __name__=='__main__':main()
