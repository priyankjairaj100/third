#!/usr/bin/env python3
"""Boundary checks on unchanged acquired Civil records; no artificial corpus."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from ccu.core import build_blocker_graph,EligiblePayloadState,direct_oracle
from ccu.summary import IndexedRidgeSummary,SummaryMemoryError

ROOT=Path(__file__).resolve().parent

def fingerprint(s):
    if isinstance(s,EligiblePayloadState):return s.state_digest()
    h=hashlib.sha256();h.update(str((s.horizon,sorted(s.alive))).encode())
    for key,val in sorted(s.coefficients().items()):h.update(str(key).encode());h.update(val.tobytes())
    return h.hexdigest()

def main():
    rows=read_natural_jsonl(ROOT/'data/civil_comments_engineering_preview.jsonl',require_labels=True)
    ids=[r['record_id'] for r in rows];z,_=lexical_engineering_features(rows,64)
    e,_=lexical_engineering_features(rows,128);y=np.array([r['label'] for r in rows])
    g=build_blocker_graph(e,ids,.6);events=[]
    def pair(k):return EligiblePayloadState(g,z,y,k),IndexedRidgeSummary.build(z,y,g.blockers,k)
    def reject(s,request,description):
        before=fingerprint(s)
        try:s.delete(request)
        except ValueError:pass
        else:raise AssertionError('invalid request was accepted: '+description)
        assert fingerprint(s)==before
        events.append({'check':description,'method':type(s).__name__,'status':'passed'})
    b,p=pair(2)
    reject(b,[ids[0],ids[0]],'duplicate_in_batch_atomic')
    reject(p,[0,0],'duplicate_in_batch_atomic')
    reject(b,['not-a-real-record-id'],'unknown_ID_atomic')
    reject(p,[len(ids)],'unknown_ID_atomic')
    reject(b,ids[:3],'over_budget_atomic');reject(p,[0,1,2],'over_budget_atomic')
    b.delete([ids[0]]);p.delete([0])
    reject(b,[ids[0]],'retry_atomic_no_budget_consumed');reject(p,[0],'retry_atomic_no_budget_consumed')
    b.delete([ids[1]]);p.delete([1])
    reject(b,[ids[2]],'exhaustion_atomic');reject(p,[2],'exhaustion_atomic')
    # Same final deletion set, different release/order choices, genuine IDs.
    b1,p1=pair(8);b2,p2=pair(8);b3,p3=pair(8)
    for i in range(8):b1.delete([ids[i]]);p1.delete([i])
    b2.delete(ids[:8]);p2.delete(range(8))
    for i in reversed(range(8)):b3.delete([ids[i]]);p3.delete([i])
    heads=[s.solve_ridge(.01) for s in [b1,p1,b2,p2,b3,p3]]
    discrepancy=max(float(np.max(np.abs(w-heads[0]))) for w in heads)
    assert discrepancy<1e-9
    events.append({'check':'batch_singleton_reversed_same_final_set','status':'passed','max_abs_head_difference':discrepancy})
    # Full natural corpus deletion: test the specified empty head and expose drift.
    b,p=pair(len(ids));deleted=[];max_head=0.0;max_moment=0.0
    for i in range(len(ids)):
        b.delete([ids[i]]);p.delete([i]);deleted.append(ids[i])
        oracle=direct_oracle(z,y,ids,.6,.01,deleted,g.priority,graph_features=e)
        bm,pm=b.moments(),p.moments()
        assert bm.count==pm.count==oracle.moments.count
        max_moment=max(max_moment,float(np.max(np.abs(pm.gram-oracle.moments.gram))))
        for state in [b,p]:max_head=max(max_head,float(np.max(np.abs(state.solve_ridge(.01)-oracle.solution.weights))))
    assert np.count_nonzero(p.solve_ridge(.01))==0 and np.count_nonzero(b.solve_ridge(.01))==0
    assert max_head<1e-9 and max_moment<1e-9
    drift=float(np.max(np.abs(p.statistics()[0])))
    events.append({'check':'all_100_natural_records_deleted','status':'passed','releases':len(ids),
      'max_abs_head_error':max_head,'max_abs_gram_error':max_moment,'empty_raw_moment_drift':drift,
      'empty_released_head_is_zero':True,'floating_key_canonicality_claim':False})
    try:IndexedRidgeSummary.build(z,y,g.blockers,8,max_coefficient_bytes=1)
    except SummaryMemoryError as ex:
        events.append({'check':'allocation_cap_preflight','status':'passed','forecast_bytes':ex.forecast.coefficient_bytes_upper_bound,'cap_bytes':1})
    else:raise AssertionError('memory preflight cap ignored')
    report={'status':'passed','data_sha256':sha256_file(ROOT/'data/civil_comments_engineering_preview.jsonl'),
      'scope':__doc__,'feature_scope':'fixed128d_lexical_curator_64d_lexical_learner_original_labels',
      'events':events,'event_count':len(events),'confirmatory':False}
    (ROOT/'results/boundary_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':len(events),'full_deletion_releases':len(ids),'maximum_head_error':max_head,'empty_moment_drift':drift},indent=2))

if __name__=='__main__':main()
