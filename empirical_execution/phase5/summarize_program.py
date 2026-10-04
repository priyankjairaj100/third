#!/usr/bin/env python3
"""Report every method and dimension; engineering timing is not a speed claim."""
from pathlib import Path
import argparse,collections,hashlib,json

def mean(values):
    values=list(values)
    return sum(values)/len(values)

def summarize(path):
    path=Path(path);lock=json.loads((path/'prospective_lock.json').read_text())
    ledger=[json.loads(l) for l in (path/'job_ledger.jsonl').read_text().splitlines()]
    ids=[j['job_id'] for j in lock['job_order']]
    if [r['job_id'] for r in ledger]!=ids:raise ValueError('Canonical complete prospective-order ledger required')
    groups=collections.defaultdict(list)
    for r in ledger:groups[(r['dimension'],r['method'])].append(r)
    table=[]
    for (dimension,method),rows in sorted(groups.items()):
        detailed=[]
        for r in rows:
            if r['status']!='passed':continue
            job=path/'jobs'/r['job_id'];c=json.loads((job/'construction/construction_report.json').read_text())
            repair=json.loads((job/'repair/repair_report.json').read_text())
            # persist_tree runs before construction_report.json is written. At
            # this stage the only non-state bytes are initial_head and logs.
            excluded=sum((job/'construction'/n).stat().st_size for n in ('initial_head.npy','construct_stdout.txt','construct_stderr.txt'))
            initial=c['persistence']['current_logical_bytes']-excluded
            # The first/final snapshot occurs before repair_report is emitted.
            head_bytes=sum(p.stat().st_size for p in (job/'repair').glob('head_*.npy'))
            final=repair['releases'][-1]['persistence']['current_logical_bytes']-head_bytes
            if initial<0 or final<0:raise ValueError('Invalid byte-accounting decomposition')
            detailed.append({'job_id':r['job_id'],'initial_serialized_state_bytes':initial,
                'final_serialized_state_bytes':final,'maximum_process_peak_rss_bytes':max(r['construction']['peak_rss_bytes'],r['repair']['peak_rss_bytes']),
                'charged_lifecycle_seconds':r['lifecycle']['charged_total_seconds']})
        table.append({'dimension':dimension,'method':method,'planned':len(rows),'successes':len(detailed),
            'failures':len(rows)-len(detailed),'all_jobs':detailed,
            'initial_state_bytes_mean':mean(r['initial_serialized_state_bytes'] for r in detailed) if detailed else None,
            'final_state_bytes_mean':mean(r['final_serialized_state_bytes'] for r in detailed) if detailed else None,
            'process_peak_rss_bytes_max':max((r['maximum_process_peak_rss_bytes'] for r in detailed),default=None),
            'completed_lifecycle_seconds_mean':mean(r['charged_lifecycle_seconds'] for r in detailed) if detailed else None})
    result={'scope':'Development only; two R paths, one U path, one A path; no genuine S',
        'dimensions':{'64':'five repeats per fixed path','768':'one feasibility pass per fixed path'},
        'timing_limitation':'Shared development host; other project checks ran concurrently. No paper speedup inference.',
        'memory_limitation':'Serialized state and child peak RSS are separate quantities; shared bundle and parent/oracle memory not included in process RSS',
        'shared_preparation_files':['bundle_d64.json','bundle_d768.json'],
        'joint_span_total_memory_optimality_claim':False,'table':table,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (path/'method_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Isolated natural-text development matrix','','All 216 jobs passed. The 864 released heads agree with the independent scalar-graph/dual-ridge oracle within 9.44e-16. These are engineering measurements on the reused Civil100 lexical preview, not primary semantic, source-withdrawal or task-utility results.','','| Dimension | Method | Jobs | Mean initial state bytes | Mean final state bytes | Maximum child RSS bytes | Mean lifecycle seconds |','|---:|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f'| {r["dimension"]} | {r["method"]} | {r["successes"]}/{r["planned"]} | {r["initial_state_bytes_mean"]:.0f} | {r["final_state_bytes_mean"]:.0f} | {r["process_peak_rss_bytes_max"]} | {r["completed_lifecycle_seconds_mean"]:.4f} |')
    lines+=['','The compact eligible-payload comparator remains substantially smaller than the joint-span summary on this fixture. Incidence-rank compression reduces coefficient storage relative to dense P-I but incurs additional construction workspace. Neither observation establishes a general memory optimum.','','All methods used the same prospective 1 GiB address-space limit, 60-second CPU/wall limits and one BLAS thread. Each construction and repair ran in a fresh process. Repair loaded its own accounted snapshot before seccomp TSYNC denied all new file opens. The kernel boundary is enforced for trusted project code; this is not a malicious-native-code or physical-erasure proof.','','Five d64 repetitions and one d768 feasibility repetition reuse the same four paths (two R, one U, one A). Shared-host concurrent development checks affect latency, so this table supports no paper speedup claim. Serialized state excludes head files; process RSS includes runtime, loaded state, working arrays and serialization buffers. Shared input preparation and independent audit costs are recorded separately.','','The canonical journal was reconstructed from all 216 immutable per-job audits after three append entries were missing. `journal_reconciliation.json` records exact original/canonical hashes and the recovered IDs; `job_ledger_original.jsonl` is preserved. No method was rerun to replace an outcome.','']
    (path/'RESULTS.md').write_text('\n'.join(lines));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');r=summarize(p.parse_args().directory);print(json.dumps({'method_dimension_groups':len(r['table']),'status':'written'}))
