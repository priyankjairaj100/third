#!/usr/bin/env python3
"""Reconcile the append journal against immutable per-job reports, never rerun.

Preserve the original journal even when only a subset of lines survived. A
missing per-job artifact, duplicated identity or inconsistent summary is fatal.
"""
from pathlib import Path
import argparse,hashlib,json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reconcile(out):
    out=Path(out)
    lock=json.loads((out/'prospective_lock.json').read_text())
    planned=lock['job_order'];ids=[j['job_id'] for j in planned]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate planned identity')
    current=out/'job_ledger.jsonl';historical=out/'job_ledger_original.jsonl'
    if historical.exists():raise FileExistsError('Reconciliation already performed; use stored certificate')
    before=current.read_bytes();observed=[json.loads(l) for l in before.splitlines()]
    mapping={r['job_id']:r for r in observed}
    if len(mapping)!=len(observed) or set(mapping)-set(ids):raise ValueError('Journal duplicates or unexpected identities')
    rows=[];hashes={}
    for job in planned:
        path=out/'jobs'/job['job_id']/'audit.json';r=json.loads(path.read_text())
        if any(r[k]!=v for k,v in job.items()):raise ValueError('Per-job identity does not match plan')
        if job['job_id'] in mapping and mapping[job['job_id']]!=r:raise ValueError('Journal differs from saved audit')
        hashes[job['job_id']]=sha(path);rows.append(r)
    summary=json.loads((out/'summary.json').read_text())
    checks=[c for r in rows for c in r['head_checks']]
    expected={'planned_jobs':len(planned),'observed_jobs':len(rows),'passed_jobs':sum(r['status']=='passed' for r in rows),
              'failed_jobs':sum(r['status']!='passed' for r in rows),'head_checks':len(checks),
              'oracle_disagreements':sum(not c['passed'] for c in checks)}
    if any(summary[k]!=v for k,v in expected.items()):raise ValueError('Summary differs from recovered artifacts')
    historical.write_bytes(before)
    canonical=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows).encode()
    current.write_bytes(canonical)
    result={'schema':'ccu-journal-reconciliation-1','original_rows':len(observed),'canonical_rows':len(rows),
            'recovered_job_ids':[j for j in ids if j not in mapping],'original_sha256':sha(historical),
            'canonical_sha256':sha(current),'exact_per_job_audit_sha256':hashes,
            'all_original_rows_unchanged':True,'method_runs_repeated':False,
            'summary_matches_all_per_job_artifacts':True,
            'cause':'Append journal contained fewer rows than completed per-job artifacts; underlying cause unestablished',
            'reconciliation_code_sha256':sha(__file__)}
    (out/'journal_reconciliation.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='exact_per_job_audit_sha256'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');print(json.dumps(reconcile(p.parse_args().directory),indent=2))
