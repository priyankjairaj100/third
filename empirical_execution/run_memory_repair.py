#!/usr/bin/env python3
"""Natural-text engineering verification of exact-arithmetic memory repairs.

No synthetic dataset, labels, semantic-quality or speed-superiority claim.
All reports go to memory_results, leaving execution_01 evidence unchanged.
"""
from pathlib import Path
import argparse, gc, hashlib, json, time
import numpy as np
from ccu.data import read_natural_jsonl, lexical_engineering_features, sha256_file
from ccu.core import build_blocker_graph, direct_oracle, EligiblePayloadState
from ccu.summary import IndexedRidgeSummary
from ccu.factored_summary import FactoredRidgeSummary
from ccu.joint_span_summary import JointSpanRidgeSummary
from ccu.compact_payload import CompactEligiblePayloadState
from run_engineering_pilot import fixed_requests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'memory_results'
METHODS={'dense':IndexedRidgeSummary,'gram_factor':FactoredRidgeSummary,
         'joint_span':JointSpanRidgeSummary,'compact_payload':CompactEligiblePayloadState}

def err(a,b):return float(np.max(np.abs(a-b),initial=0))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def array_bytes(state):
    """Count actual owned numeric buffers reachable from state, once.

    Python containers/scalars and allocator slack are excluded explicitly.
    Traverses slots and __dict__; tracks roots to avoid charging views twice.
    """
    seen=set(); buffers={}
    def visit(v):
        if id(v) in seen:return
        seen.add(id(v))
        if isinstance(v,np.ndarray):
            base=v
            while isinstance(base.base,np.ndarray):base=base.base
            buffers[id(base)]=base.nbytes
        elif isinstance(v,dict):
            for k,x in v.items():visit(k);visit(x)
        elif isinstance(v,(set,list,tuple)):
            for x in v:visit(x)
        else:
            if hasattr(v,'__dict__'):visit(v.__dict__)
            for cls in type(v).__mro__:
                for name in getattr(cls,'__slots__',()):
                    if hasattr(v,name):visit(getattr(v,name))
    visit(state)
    return sum(buffers.values())

def coefficient_audit(dense, state):
    maximum=0.; count=0
    keys=set(dense._coeff)|set(state._coeff)
    zero=np.zeros(dense.packed_dimension)
    for key in sorted(keys):
        expected=dense._coeff[key].value if key in dense._coeff else zero
        actual=state.coefficient(key) if key in state._coeff else zero
        maximum=max(maximum,err(expected,actual));count+=1
    if maximum>1e-9:raise AssertionError(('coefficient mismatch',maximum))
    return {'keys_checked':count,'max_abs_coefficient_error':maximum}

def fingerprint(state):
    if hasattr(state,'state_digest'):return state.state_digest()
    h=hashlib.sha256(str((state.horizon,sorted(state.alive))).encode())
    for key in sorted(state._coeff):
        h.update(str(key).encode());h.update(state.coefficient(key).tobytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    rows=read_natural_jsonl(ROOT/'data/civil_comments_engineering_preview.jsonl',require_labels=True)
    ids=[r['record_id'] for r in rows]
    e,_=lexical_engineering_features(rows,128)
    graph=build_blocker_graph(e,ids,.6)
    requests=fixed_requests(graph,8)
    label_names=['toxicity','severe_toxicity','obscene','threat','insult','identity_attack','sexual_explicit']
    cases=[(64,1,40),(128,1,40),(768,1,40),(128,7,8),(768,7,8)]
    if args.quick:cases=[(64,1,2)]
    design={'scope':'natural_preview_lexical_engineering_only','confirmatory':False,
      'cases':cases,'curator_dimension':128,'threshold':.6,'horizon':8,
      'lambda':.01,'checkpoints':[1,2,4,8],'label_names':label_names,
      'input_sha256':sha256_file(ROOT/'data/civil_comments_engineering_preview.jsonl'),
      'no_synthetic_data':True,'no_tolerance_rank_truncation':True,
      'absolute_error_failure_threshold':1e-9,'memory_measure':'owned NumPy array bytes; not RSS or total live state',
      'no_remote_compute':True}
    write(OUT/'design.json',design)
    registry=OUT/'checkpoints.jsonl';registry.write_text('')
    summaries=[]; failures=[]
    for d,c,limit in cases:
        print(f'Memory repair d={d} outputs={c} paths={limit}',flush=True)
        z,_=lexical_engineering_features(rows,d)
        y=np.asarray([[r['fields'][name] for name in label_names[:c]] for r in rows],dtype=np.float64)
        result={'dimension':d,'outputs':c,'paths':limit,'checkpoints':0,'max_head_error':{},
                'max_moment_error':{},'max_live_array_bytes':{},'coefficient_audits':[],
                'max_compact_decode_error':None,'compact_decode_invocations':0}
        for path_number,request in enumerate(requests[:limit]):
            states={name:cls.build(z,y,graph.blockers,8) for name,cls in METHODS.items()}
            if path_number==0:
                old=EligiblePayloadState(graph,z,y,8)
                result['initial_array_bytes']={name:array_bytes(s) for name,s in states.items()}
                result['initial_array_bytes']['original_payload_with_gram']=array_bytes(old)
                result['initial_accounting']={name:s.accounting() for name,s in states.items()}
                del old
            cumulative=[];position=0
            for checkpoint in [1,2,4,8]:
                batch=request['indices'][position:checkpoint];position=checkpoint;cumulative+=batch
                oracle=direct_oracle(z,y,ids,.6,.01,[ids[i] for i in cumulative],graph.priority,graph_features=e)
                row={'dimension':d,'outputs':c,'path':request['id'],'family':request['family'],
                     'checkpoint':checkpoint,'deleted_indices':list(cumulative),'methods':{}}
                for name,s in states.items():
                    begin=time.perf_counter();s.delete(batch);repair_seconds=time.perf_counter()-begin
                    s.check_invariants()
                    m=s.moments();begin=time.perf_counter();w=s.solve_ridge(.01);decode_seconds=time.perf_counter()-begin
                    he=err(w,oracle.solution.weights)
                    me=max(err(m.gram,oracle.moments.gram),err(m.cross,oracle.moments.cross))
                    passed=m.count==oracle.moments.count and max(he,me)<1e-9
                    if name=='compact_payload':passed=passed and set(s.selected_ids())==set(map(int,oracle.selected_indices))
                    rec={'head_error':he,'moment_error':me,'count':m.count,'passed':bool(passed),
                         'array_bytes':array_bytes(s),'repair_seconds_unreplicated':repair_seconds,
                         'decode_seconds_unreplicated':decode_seconds}
                    if name=='joint_span' and hasattr(s,'decode_compact'):
                        cw=s.decode_compact(.01).weights
                        ce=err(cw,oracle.solution.weights);rec['compact_decode_head_error']=ce
                        result['max_compact_decode_error']=max(result['max_compact_decode_error'] or 0.,ce)
                        result['compact_decode_invocations']+=1
                        if ce>=1e-9:passed=False;rec['passed']=False
                    result['max_head_error'][name]=max(result['max_head_error'].get(name,0),he)
                    result['max_moment_error'][name]=max(result['max_moment_error'].get(name,0),me)
                    result['max_live_array_bytes'][name]=max(result['max_live_array_bytes'].get(name,0),rec['array_bytes'])
                    row['methods'][name]=rec
                    if not passed:failures.append({'case':[d,c],'path':request['id'],'checkpoint':checkpoint,'method':name,**rec})
                if path_number==0:
                    for name in ['gram_factor','joint_span']:
                        audit=coefficient_audit(states['dense'],states[name])
                        result['coefficient_audits'].append({'method':name,'checkpoint':checkpoint,**audit})
                result['checkpoints']+=1
                with registry.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            del states;gc.collect()
        summaries.append(result);write(OUT/f'd{d}_c{c}.json',result)
    boundaries=[]
    if not args.quick:
        z,_=lexical_engineering_features(rows,64);y=np.array([r['label'] for r in rows])
        for name in ['gram_factor','joint_span','compact_payload']:
            cls=METHODS[name];s=cls.build(z,y,graph.blockers,2)
            for request,label in [([0,0],'duplicate'),([len(rows)],'unknown'),([0,1,2],'budget')]:
                before=fingerprint(s)
                try:s.delete(request)
                except ValueError:pass
                else:raise AssertionError((name,label,'accepted invalid'))
                assert fingerprint(s)==before
            s.delete([0]);before=fingerprint(s)
            try:s.delete([0])
            except ValueError:pass
            else:raise AssertionError((name,'accepted retry'))
            assert fingerprint(s)==before
            states=[cls.build(z,y,graph.blockers,8) for _ in range(3)]
            states[0].delete(range(8))
            for j in range(8):states[1].delete([j])
            for j in reversed(range(8)):states[2].delete([j])
            ordererr=max(err(states[0].solve_ridge(.01),s.solve_ridge(.01)) for s in states[1:])
            assert ordererr<1e-9
            boundaries.append({'method':name,'atomic_rejection_checks':4,'order_head_error':ordererr})
        states={name:METHODS[name].build(z,y,graph.blockers,len(rows)) for name in ['gram_factor','joint_span','compact_payload']}
        fullmax={name:0. for name in states}
        for i in range(len(rows)):
            oracle=direct_oracle(z,y,ids,.6,.01,ids[:i+1],graph.priority,graph_features=e)
            for name,s in states.items():
                s.delete([i]);s.check_invariants()
                headerr=err(s.solve_ridge(.01),oracle.solution.weights)
                assert s.moments().count==oracle.moments.count and headerr<1e-9
                fullmax[name]=max(fullmax[name],headerr)
        for name,s in states.items():assert not np.any(s.solve_ridge(.01))
        boundaries.append({'full_deletion_releases':len(rows),'max_head_error':fullmax,'zero_final_heads':True})
    codefiles=[Path(__file__)]+[ROOT/'ccu'/n for n in ['factored_summary.py','joint_span_summary.py','compact_payload.py','summary.py','core.py']]
    report={'status':'passed' if not failures else 'failed','cases':summaries,'failures':failures,
      'release_checkpoints':sum(s['checkpoints'] for s in summaries),'boundary_checks':boundaries,
      'code_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in codefiles},
      'scope':__doc__,'confirmatory':False,'array_bytes_exclude_python_metadata_and_decoder_workspace':True,
      'max_live_array_bytes_scope':'maximum at observed post-batch checkpoints, excludes initial/intermediate/temporary allocations',
      'coefficient_audit_reference':'dense incremental implementation; released heads and moments separately use independently rescored retained-data oracle',
      'timing_scope':'unreplicated development observations; combined audit arrays are live, not a performance benchmark'}
    write(OUT/'summary.json',report)
    print(json.dumps({'status':report['status'],'checkpoints':report['release_checkpoints'],'cases':[{'d':s['dimension'],'c':s['outputs'],'bytes':s['initial_array_bytes'],'head_errors':s['max_head_error']} for s in summaries]},indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
