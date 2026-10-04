#!/usr/bin/env python3
"""Manifest consistency checks, not experimental verification or preregistration."""
import argparse,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('--lock',action='store_true')
    a=p.parse_args();x=json.loads(a.manifest.read_text());errors=[]
    def require(condition,message):
        if not condition: errors.append(message)
    require(x.get('schema_version')=='1.0','unknown schema version')
    require(x.get('synthetic_datasets_allowed') is False,'synthetic datasets forbidden in this protocol')
    c=x['contracts'];require(c['service_output']=='head_only','primary service must remain head-only')
    require(c['request_interface']=='identifiers_only','primary access interface changed')
    require(c['primary_curator_fitted_to_deletable_data'] is False,'primary fixed-curator restriction violated')
    require(c['exhaustion_policy']=='atomic_refusal','new exhaustion contract needs registered amendment')
    require(x['features']['native_learner_dimension']==768,'native main panel must not disappear')
    require(x['fairness']['same_canonical_z'] and x['fairness']['shared_solver'],'feature/decoder parity required')
    req=x['requests']
    for arm in ('record','source'):
        pts=req[arm+'_checkpoints']; k=req[arm+'_horizon']
        require(pts==sorted(set(pts)) and min(pts)>0 and max(pts)<=k,f'invalid {arm} checkpoint sequence')
    allocations=x['allocations_per_corpus_panel_primary_seed']
    for arm in ('R','S','U','A'):
        require(allocations['full_methods'][arm]<=allocations['task_relevance'][arm],f'full-method subset exceeds relevance pool for {arm}')
    salts=[v for k,v in x['seeds'].items() if k.endswith('_salt')]
    require(len(salts)==len(set(salts)),'hash salts must be distinct')
    cal=x['threshold_calibration']
    require(cal['validation_pairs']==200,'validation size amendment must be documented')
    require(cal['post_confirmation_retuning'] is False,'no post-confirmation threshold retuning')
    missing=[k for k,v in x['pre_execution_lock_required'].items() if v is None or v=='' or v=={} or v==[]]
    if a.lock:
        require(not missing,'unresolved pre-execution lock fields: '+', '.join(missing))
    counts={}
    for block,arms in allocations.items():
        counts[block]=sum(n*len(req['source_checkpoints' if arm=='S' else 'record_checkpoints']) for arm,n in arms.items())
    print(json.dumps({'valid_draft':not errors,'lock_requested':a.lock,'lock_complete':not missing,
        'unresolved_fields':missing,'checkpoint_releases_per_corpus_per_method_before_clipping':counts,
        'errors':errors,'scope':'checks design invariants and missing values, not claims or empirical implementation'},indent=2))
    return 1 if errors else 0

if __name__=='__main__':sys.exit(main())
