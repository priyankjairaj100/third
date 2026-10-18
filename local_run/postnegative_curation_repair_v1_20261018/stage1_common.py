#!/usr/bin/env python3
"""Postnegative natural training-only structure; no semantic qualification.

Calls frozen Phase 3 ordered FP64 scorer directly at every original grid cut.
No raw text, task label, held-out feature slice, model call, fit or repair effect.
Every output directory is exclusive. Partial and failed runs remain explicit.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import shutil
import sys
import time

THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
for key in THREADS:
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
GRID = [Fraction(i,100) for i in range(50,100)] + [Fraction(199,200)]
HORIZONS = (1,8,32,128)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    p=Path(path); before=p.stat(); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    after=p.stat()
    need((before.st_size,before.st_mtime_ns,before.st_ino)==
         (after.st_size,after.st_mtime_ns,after.st_ino),'input_changed_during_hash')
    return {'bytes':after.st_size,'sha256':h.hexdigest()}


def load(path):
    def unique(pairs):
        d={}
        for k,v in pairs:
            need(k not in d,'duplicate_json_key'); d[k]=v
        return d
    def invalid(v): raise ValueError('nonfinite_json')
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique,parse_constant=invalid)


def save(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    need(len(raw)<8*1024**2,'single_output_cap')
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def expected_record_admissions(hist,n,k):
    # P(all b blockers deleted, candidate alive) under uniform k-subset deletion.
    den=math.comb(n,k)
    value=sum((Fraction(count*math.comb(n-b-1,k-b),den)
        for b,count in hist.items() if 0<b<=k and k-b<=n-b-1),Fraction())
    return {'numerator':str(value.numerator),'denominator':str(value.denominator),
            'float_display':float(value),'deletion_budget':k}


def describe(graph,genuine):
    n=len(graph.record_ids); counts=[int(x) for x in graph.blocker_counts]
    hist=Counter(counts); blockers=graph.blockers
    one=[int(b[0]) for b in blockers if len(b)==1]
    excluded_singleton=sum(counts[u]>0 for u in one)
    sources=graph.source_ids; source_candidates=Counter()
    for i,b in enumerate(blockers):
        us={sources[int(j)] for j in b}
        if len(us)==1:
            unit=next(iter(us))
            if unit!=sources[i] and unit in genuine: source_candidates[unit]+=1
    return {'rows':n,'threshold':graph.threshold,'threshold_float_hex':float(graph.threshold).hex(),
        'edge_count':graph.edge_count,'initially_selected':hist.get(0,0),
        'initially_excluded':n-hist.get(0,0),
        'blocker_histogram':{str(b):c for b,c in sorted(hist.items())},
        'eligible_record_count_by_horizon':{str(k):sum(c for b,c in hist.items() if b<=k) for k in HORIZONS},
        'uniform_record_expected_admissions':{str(k):expected_record_admissions(hist,n,k) for k in HORIZONS if k<=n},
        'singleton_record_admission_total':len(one),
        'singleton_excluded_blocker_admission_total':excluded_singleton,
        'singleton_record_requests_with_admission':len(set(one)),
        'singleton_excluded_record_requests_with_admission':len({u for u in one if counts[u]>0}),
        'genuine_single_source_request_count':len(genuine),
        'genuine_single_source_admission_total':sum(source_candidates.values()),
        'genuine_single_source_requests_with_admission':len(source_candidates),
        'array_and_utf8_accounting':graph.accounting(),
        'source_interpretation':'Native accounts are contributor proxies; unknown singleton units excluded from source-request frame.'}


def independent_small_oracle(x,ids,sources,build):
    """Scalar implementation on first 16 original-order rows, all fixed cuts.

    This is a sampled implementation diagnostic, not a full graph replay.
    """
    import numpy as np
    n=min(16,len(ids)); x=x[:n]; ids=ids[:n]; sources=sources[:n]
    order=sorted(range(n),key=lambda i:(hashlib.sha256(f'priority-v1|0|{ids[i]}'.encode()).digest(),ids[i]))
    norm=[]
    for row in x:
        total=np.float64(0)
        for v in row:
            q=np.float64(v); total=np.float64(total+np.float64(q*q))
        length=np.sqrt(total)
        norm.append([np.float64(v)/length for v in row])
    scores={}
    for p,i in enumerate(order):
        for j in order[:p]:
            total=np.float64(0)
            for a,b in zip(norm[i],norm[j]): total=np.float64(total+np.float64(a*b))
            scores[i,j]=float(total)
    checks=0
    for cut in GRID:
        g=build(x,ids,float(cut),sources,seed=0,block_size=7)
        for i,b in enumerate(g.blockers):
            wanted={j for (late,j),s in scores.items() if late==i and s>float(cut)}
            need(set(map(int,b))==wanted,'scalar_subset_oracle_disagreement'); checks+=1
    return {'passed':True,'rows':n,'thresholds':len(GRID),'blocker_set_comparisons':checks,
        'scope':'First16 original-order panel rows, scalar IEEE FP64 normalization and dot; not full-population or full-graph replay.'}


def imports(repo,protocol):
    for rel,binding in protocol['frozen_sources'].items(): need(sha(repo/rel)==binding,'frozen_source_mismatch:'+rel)
    sys.path.insert(0,str(repo/'empirical_execution'))
    from phase3.reference_graph import build_reference_graph
    from phase3 import panels
    return build_reference_graph,panels


def select_rows(allocation,prefix,panels):
    need(len(allocation)==140250,'allocation_population')
    expected_roles={'development_train':10314,'evaluation_train':87873,'calibration':20996,'development_test':2138,'evaluation_test':18929}
    need(dict(Counter(r['role'] for r in allocation))==expected_roles,'role_counts')
    need(len({r['record_id'] for r in allocation})==len(allocation),'duplicate_ids')
    need([r['original_full_row_index'] for r in allocation]==list(range(140250)),'original_row_order')
    groups={}
    for r in allocation:
        source=r['source_unit_id']; part=panels.digest_partition(panels.hash_integer('ccu-v1-source-split',source))
        reserved=panels.hash_integer('third-llm-development-reserve-v1-20261016',source)*10<2**256
        role='calibration' if part=='calibration' else ('development_' if reserved else 'evaluation_')+part
        need(part==r['partition'] and role==r['role'],'allocation_role_mismatch')
        need(source not in groups or groups[source]==role,'source_role_overlap'); groups[source]=role
    p=prefix['1000']; wanted=set(p['record_ids'])
    need(p['actual']==1001 and p['target']==1000 and len(wanted)==1001,'fixed_prefix_size')
    chosen=[r for r in allocation if r['record_id'] in wanted]
    need(len(chosen)==1001 and all(r['role']=='development_train' for r in chosen),'training_only_prefix')
    sources={r['source_unit_id'] for r in chosen}
    need(sources==set(p['source_ids']),'prefix_source_set')
    need({r['record_id'] for r in allocation if r['source_unit_id'] in sources}==wanted,'whole_sources')
    devgroups={}
    for r in allocation:
        if r['role']=='development_train': devgroups.setdefault(r['source_unit_id'],[]).append(r['record_id'])
    expected=[]
    for s in sorted(devgroups,key=lambda s:(panels.hash_integer('third-llm-development-order-v1-20261016',s),s)):
        if len(expected)>=1000: break
        expected.extend(sorted(devgroups[s]))
    need(expected==p['record_ids'],'original_prefix_rule')
    return chosen


def run(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); complete=[]
    try:
        pp=Path(args.protocol).resolve(); need(sha(pp)['sha256']==args.protocol_sha256,'protocol_pin')
        protocol=load(pp)
        need(protocol['schema']=='third-postnegative-structure-stage1-v1','protocol_schema')
        need(protocol['status']=='frozen_for_stage1','protocol_not_frozen')
        import numpy as np
        need(platform.system()==protocol['runtime']['platform'],'runtime_platform')
        need(platform.python_version()==protocol['runtime']['python'],'runtime_python_version')
        need(np.__version__==protocol['runtime']['numpy'],'runtime_numpy_version')
        need(np.dtype('float64').itemsize==8 and np.finfo(np.float64).nmant==52,'IEEE_binary64')
        need(shutil.disk_usage(out).free>=protocol['resource_limits']['storage_reserve_bytes'],'free_space_reserve')
        def timeout(*unused): raise TimeoutError('stage_wall_cap')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(protocol['resource_limits']['wall_seconds'])
        build,panels=imports(Path(args.repo).resolve(),protocol)
        import scipy
        if 'scipy' in protocol['runtime']: need(scipy.__version__==protocol['runtime']['scipy'],'runtime_scipy_version')
        paths={k:Path(getattr(args,k)).resolve() for k in ('allocation','prefixes','vectors')}
        initial={k:sha(p) for k,p in paths.items()};need(initial==protocol['inputs'],'input_bindings')
        rows=select_rows(load(paths['allocation']),load(paths['prefixes']),panels)
        cache=np.load(paths['vectors'],mmap_mode='r',allow_pickle=False)
        need(cache.dtype==np.float32 and cache.shape==(140250,768),'cache_shape_dtype')
        x=np.array(cache[[r['original_full_row_index'] for r in rows]],dtype=np.float32,order='C',copy=True)
        need(np.isfinite(x).all(),'finite_training_features')
        ids=[r['record_id'] for r in rows];sources=[r['source_unit_id'] for r in rows]
        genuine={r['source_unit_id'] for r in rows if r['genuine_source_request_eligible'] and r['source_kind'].startswith('native_')}
        save(out/'intake.json',{'protocol_sha256':args.protocol_sha256,'local_inputs':initial,'original_inputs':protocol['inputs'],'rows':len(rows),
             'heldout_feature_slices_read':False,'labels_read':False,'text_read':False,
             'feature_slice_roles':['development_train'],
             'full_input_binding_scope':'Full input files rehashed as opaque bytes; only reserved development-training feature slices decoded.',
             'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.system()}})
        save(out/'subset_oracle.json',independent_small_oracle(x,ids,sources,build))
        for index,cut in enumerate(GRID):
            t=time.monotonic();graph=build(x,ids,float(cut),sources,seed=0,block_size=256)
            oracle=build(x,ids,float(cut),sources,seed=0,block_size=37)
            need(all(np.array_equal(getattr(graph,k),getattr(oracle,k)) for k in ('priority_indices','indptr','indices')),'full_graph_alternate_tiling_disagreement')
            del oracle
            result=describe(graph,genuine);result.update(index=index,threshold_fraction=[cut.numerator,cut.denominator],
                graph_and_replay_seconds=time.monotonic()-t,full_graph_alternate_tiling_equal=True,
                semantic_validity_claim=False,primary_study_result=False)
            save(out/f'threshold_{index:02d}.json',result);complete.append(index);del graph
            need(shutil.disk_usage(out).free>=protocol['resource_limits']['storage_reserve_bytes'],'free_space_reserve')
            need(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=protocol['resource_limits']['output_cap_bytes'],'output_cap')
        need({k:sha(p) for k,p in paths.items()}==initial,'input_changed_after_execution')
        for rel,binding in protocol['frozen_sources'].items():
            need(sha(Path(args.repo).resolve()/rel)==binding,'frozen_source_changed_after_execution:'+rel)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(out/'completion.json',{'status':'completed','all_thresholds_completed':complete==list(range(51)),
            'scope':'postnegative_training_only_natural_mechanism_characterization',
            'semantic_threshold_selected':False,'human_reviews':0,'model_calls':0,'repair_effects_measured':False,
            'primary_study_started':False,'original_v4_failure_retained':True,
            'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':rss if platform.system()=='Darwin' else rss*1024,
            'outputs':{p.name:sha(p) for p in sorted(out.glob('*.json'))}})
        print(json.dumps({'status':'completed','thresholds':len(complete),'rows':len(rows)}))
    except BaseException as e:
        save(out/'failure.json',{'status':'failed','exception_type':type(e).__name__,'reason':str(e),
            'completed_threshold_indices':complete,'partial_results_not_complete':True,'retry_automatic':False,
            'elapsed_seconds':time.monotonic()-started})
        raise
    finally:
        signal.alarm(0)


def main():
    ap=argparse.ArgumentParser()
    for k in ('repo','protocol','protocol-sha256','out','allocation','prefixes','vectors'): ap.add_argument('--'+k,required=True)
    run(ap.parse_args())

if __name__=='__main__': main()
