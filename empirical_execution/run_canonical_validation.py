#!/usr/bin/env python3
"""Strict canonical bytes and exact numerical certificates on unchanged text."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,time,gc
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from ccu.core import build_blocker_graph,direct_oracle
from ccu.exact_canonical import ExactCanonicalSummary
from ccu.certified_ridge import factor_residual_certificate
from ccu.strict_service import repair_and_release,CertificationFailure
from run_engineering_pilot import fixed_requests

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'canonical_results'
LABELS=['toxicity','severe_toxicity','obscene','threat','insult','identity_attack','sexual_explicit']

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def maxerr(a,b):return float(np.max(np.abs(a-b),initial=0))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rows=read_natural_jsonl(ROOT/'data/civil_comments_engineering_preview.jsonl',require_labels=True)
    ids=[r['record_id'] for r in rows];e,_=lexical_engineering_features(rows,128)
    graph=build_blocker_graph(e,ids,.6);paths=fixed_requests(graph,8)
    selected=[next(p for p in paths if p['family']==arm) for arm in ['R','U','A']]
    cases=[(16,1,selected),(64,1,selected),(768,1,selected),(768,7,selected[:1])]
    if args.quick:cases=[(16,1,selected[:1])]
    design={'scope':'100 unchanged real Civil preview records; lexical engineering only',
      'target':'exact dyadic FP32 features/FP64 labels; frozen graph; exact coefficient aggregation',
      'curator_dimension':128,'threshold':.6,'horizon':8,'lambda_exact_binary_input':repr(.01),
      'certificate_tolerance_exact':'1/10000000000','checkpoints':[1,2,4,8],
      'cases':[{'dimension':d,'outputs':c,'requests':ps} for d,c,ps in cases],
      'data_sha256':sha256_file(ROOT/'data/civil_comments_engineering_preview.jsonl'),
      'fresh_rebuild':'retained rows only; rescored retained graph; stable original unit IDs and remaining horizon',
      'no_synthetic_dataset_or_labels':True,'no_remote_compute':True,'confirmatory':False}
    write(OUT/'design.json',design)
    registry=OUT/'checkpoints.jsonl';registry.write_text('')
    summaries=[];history_checks=[];certs=[];count=0
    for d,c,requests in cases:
        print(f'Exact validation d={d} c={c} paths={len(requests)}',flush=True)
        x,_=lexical_engineering_features(rows,d)
        y=np.asarray([[r['fields'][name] for name in LABELS[:c]] for r in rows],dtype=np.float64)
        t=time.perf_counter();initial=ExactCanonicalSummary.build(x,y,graph.blockers,8)
        initial_seconds=time.perf_counter()-t;initial_bytes=initial.canonical_bytes()
        summary={'dimension':d,'outputs':c,'initial_build_seconds_unreplicated':initial_seconds,
                 'initial_accounting':initial.accounting(),'checkpoints':0,'max_legacy_head_difference':0.,
                 'max_certificate_upper_decimal':'0','certificate_upper_numeric_for_summary':0.,
                 'repair_seconds_total':0.,'fresh_seconds_total':0.,'certificate_seconds_total':0.}
        def fresh(deleted,horizon):
            kept=[i for i in range(len(rows)) if i not in deleted]
            ix=np.asarray(kept,dtype=int)
            retained_graph=build_blocker_graph(e[ix],[ids[i] for i in kept],.6)
            return ExactCanonicalSummary.build(x[ix],y[ix],retained_graph.blockers,horizon,
                owners=kept,universe_size=len(rows),alive=kept)
        for request in requests:
            state=ExactCanonicalSummary.from_bytes(initial_bytes,validate=False)
            deleted=[];position=0
            for checkpoint in [1,2,4,8]:
                batch=request['indices'][position:checkpoint];position=checkpoint;deleted+=batch
                t=time.perf_counter();meter=state.delete(batch);repair_seconds=time.perf_counter()-t
                state.check_invariants();encoded=state.canonical_bytes()
                t=time.perf_counter();rebuilt=fresh(set(deleted),8-len(deleted));fresh_seconds=time.perf_counter()-t
                same=encoded==rebuilt.canonical_bytes()
                if not same:raise AssertionError(('canonical fresh mismatch',d,c,request['id'],checkpoint))
                candidate=state.decode_candidate(.01)
                t=time.perf_counter();cert=factor_residual_certificate(*state.factor(),.01,candidate);certificate_seconds=time.perf_counter()-t
                if not cert.meets_tolerance(F(1,10**10)):
                    raise AssertionError(('certificate does not meet requested tolerance',d,c,request['id'],checkpoint,cert.to_dict()['error_upper_bound_decimal']))
                # Legacy finite-precision oracle is diagnostic; certified target
                # is the exact frozen-input objective, not its BLAS summation.
                oracle=direct_oracle(x,y,ids,.6,.01,[ids[i] for i in deleted],graph.priority,graph_features=e)
                assert state.factor()[-1]==oracle.moments.count
                difference=maxerr(candidate,oracle.solution.weights)
                upper=cert.to_dict(40)['error_upper_bound_decimal']
                record={'dimension':d,'outputs':c,'path':request['id'],'family':request['family'],
                  'checkpoint':checkpoint,'deleted_indices':deleted[:],
                  'state_bytes':len(encoded),'state_sha256':sha(encoded),'fresh_bytes_equal':same,
                  'legacy_float_head_difference':difference,'certificate':cert.to_dict(40),
                  'repair_seconds_unreplicated':repair_seconds,'fresh_build_seconds_unreplicated':fresh_seconds,
                  'certificate_seconds_unreplicated':certificate_seconds,'work':meter}
                with registry.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                summary['checkpoints']+=1;count+=1
                summary['max_legacy_head_difference']=max(summary['max_legacy_head_difference'],difference)
                if cert.upper_bound(40)>F(summary['max_certificate_upper_decimal']):summary['max_certificate_upper_decimal']=upper
                summary['repair_seconds_total']+=repair_seconds;summary['fresh_seconds_total']+=fresh_seconds;summary['certificate_seconds_total']+=certificate_seconds
                if d==768 and c==1 and request==requests[0] and checkpoint==8:
                    (OUT/'strict_768_after_eight.json').write_bytes(encoded)
                    write(OUT/'strict_768_release_certificate.json',cert.to_dict(40))
            # Same final set and horizon, three histories plus saved/reloaded state.
            variants=[]
            for mode in ['one_batch','singletons','reversed_singletons']:
                s=ExactCanonicalSummary.from_bytes(initial_bytes,validate=False)
                deletion=request['indices'][:8]
                if mode=='one_batch':s.delete(deletion)
                else:
                    for u in (deletion if mode=='singletons' else reversed(deletion)):s.delete([u])
                variants.append(s.canonical_bytes())
            assert all(v==encoded for v in variants)
            restored=ExactCanonicalSummary.from_bytes(encoded,validate=True)
            assert restored.canonical_bytes()==encoded
            history_checks.append({'dimension':d,'outputs':c,'path':request['id'],
                'one_batch_singletons_reverse_fresh_reload_equal':True,'sha256':sha(encoded),
                'candidate_head_bytes_equal_same_runtime':all(np.array_equal(ExactCanonicalSummary.from_bytes(v,validate=False).decode_candidate(.01),candidate) for v in variants)})
        summary.pop('certificate_upper_numeric_for_summary');summaries.append(summary)
        write(OUT/f'case_d{d}_c{c}.json',summary);gc.collect()
    boundaries=[]
    if not args.quick:
        x,_=lexical_engineering_features(rows,16);y=np.asarray([r['label'] for r in rows])
        base=ExactCanonicalSummary.build(x,y,graph.blockers,2)
        for request,label in [([0,0],'duplicate'),([len(rows)],'unknown'),([0,1,2],'budget')]:
            before=base.canonical_bytes()
            try:base.delete(request)
            except ValueError:pass
            else:raise AssertionError('invalid request accepted')
            assert base.canonical_bytes()==before;boundaries.append(label)
        base.delete([0]);before=base.canonical_bytes()
        try:base.delete([0])
        except ValueError:pass
        else:raise AssertionError('retry accepted')
        assert base.canonical_bytes()==before;boundaries.append('retry')
        # Rigorous release gate is exercised through the bytes-only interface.
        gate=ExactCanonicalSummary.build(x,y,graph.blockers,8).canonical_bytes()
        release=repair_and_release(gate,[0,1],.01,F(1,10**10))
        assert release.certificate.meets_tolerance(F(1,10**10))
        try:repair_and_release(gate,[0,1],.01,F(0))
        except CertificationFailure:boundaries.append('zero_tolerance_no_release')
        else:raise AssertionError('expected nonzero residual gate rejection')
        write(OUT/'service_release.json',{'state_sha256':release.state_sha256,'certificate':release.certificate.to_dict(40),'input_state_unchanged_sha256':sha(gate)})
        state=ExactCanonicalSummary.build(x,y,graph.blockers,len(rows));full=[]
        for i in range(len(rows)):
            state.delete([i]);state.check_invariants()
            kept=list(range(i+1,len(rows)));ix=np.asarray(kept,dtype=int)
            gg=build_blocker_graph(e[ix],[ids[j] for j in kept],.6)
            rebuilt=ExactCanonicalSummary.build(x[ix],y[ix],gg.blockers,len(kept),owners=kept,universe_size=len(rows),alive=kept)
            assert state.canonical_bytes()==rebuilt.canonical_bytes()
            w=state.decode_candidate(.01);cert=factor_residual_certificate(*state.factor(),.01,w)
            assert cert.meets_tolerance(F(1,10**10))
            full.append({'deleted_count':i+1,'fresh_bytes_equal':True,'state_sha256':sha(state.canonical_bytes()),'certificate':cert.to_dict(40)})
        assert not state.coefficient_keys() and not np.any(w) and cert.error_squared_upper_bound==0
        write(OUT/'full_deletion.json',full)
    sourcefiles=[Path(__file__)]+[ROOT/'ccu'/p for p in ['exact_canonical.py','certified_ridge.py','strict_service.py','native/exact_chart.cpp']]
    report={'status':'passed','regular_release_checkpoints':count,'cases':summaries,'history_checks':history_checks,
      'boundary_checks':boundaries,'full_deletion_releases':0 if args.quick else len(rows),
      'all_fresh_state_comparisons_byte_exact':True,'all_head_bounds_met_exact_tolerance':'1/10000000000',
      'code_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in sourcefiles},
      'target':'new exact-aggregation mode on frozen FP32 features and FP64 labels; not byte identity to legacy floating retraining',
      'no_synthetic_dataset_or_labels':True,'confirmatory':False}
    write(OUT/'summary.json',report)
    print(json.dumps({'status':'passed','regular_checkpoints':count,'full_deletion_releases':report['full_deletion_releases'],
      'cases':[{'dimension':r['dimension'],'outputs':r['outputs'],'bytes':r['initial_accounting']['canonical_serialized_bytes'],'upper':r['max_certificate_upper_decimal']} for r in summaries]},indent=2))

if __name__=='__main__':main()
