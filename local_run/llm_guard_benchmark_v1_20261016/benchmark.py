"""Bounded real-cache throughput and exact parity qualification, no effects."""
import argparse
import datetime
import json
import os
import platform
import resource
import socket
import sys
import time
import numpy as np
from intake import HERE, REPO, W, FRAME, CACHE, binding, require, publish, verify_package, strict
from guard_core import scan

ROOT=W/'qualification/llm-guard-benchmark-v1-20261016-a01'
INTAKE=W/'qualification/llm-development-intake-v1-20261016-a01/run'


def main():
    p=argparse.ArgumentParser();p.add_argument('--package-sha256',required=True);args=p.parse_args()
    require(os.environ.get('SLURM_JOB_ID','').isdigit() and socket.gethostname() in ('cn1','csis.cn1'),'compute_only')
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')) and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_single_thread')
    verify_package(args.package_sha256);e=strict((HERE/'expectations.json').read_bytes());plan=strict((HERE/'PLAN.json').read_bytes())
    require(platform.python_version()==e['python'] and np.__version__==e['numpy'],'qualified_runtime')
    watched={INTAKE/'result.json':e['intake_result'],INTAKE/'independent_review.json':e['intake_review'],
             INTAKE/'allocation.private.json':e['allocation'],CACHE/'vectors.npy':e['assembly']['assembly_files']['vectors.npy']}
    watched.update({REPO/name:b for name,b in e['repository_files'].items()})
    for path,wanted in watched.items():require(binding(path)==wanted,'bound_input:'+str(path))
    intake=strict((INTAKE/'result.json').read_bytes());review=strict((INTAKE/'independent_review.json').read_bytes())
    require(intake['passed'] is True and review['passed'] is True and review['intake_result_sha256']==e['intake_result']['sha256'],'reviewed_reservation')
    rows=strict((INTAKE/'allocation.private.json').read_bytes());cache=np.load(CACHE/'vectors.npy',mmap_mode='r',allow_pickle=False)
    require(cache.shape==(140250,768) and cache.dtype==np.dtype('float32'),'full_FP32_cache')
    train_all=[r['original_full_row_index'] for r in rows if r['role']=='development_train']
    heldout_all=sorted((r['original_full_row_index'] for r in rows if r['partition']!='train'),key=lambda i:rows[i]['record_id'])
    require(len(train_all)==plan['full_development_reserve_rows'] and len(heldout_all)==plan['full_original_heldout_rows'],'full_scope_counts')
    train=train_all[:256];heldout=heldout_all[:1024];thresholds=[x['numerator']/x['denominator'] for x in plan['grid']]
    sys.path.insert(0,str(REPO));from empirical_execution.phase3 import panels
    out=ROOT/'run';out.mkdir(mode=0o700,exist_ok=False)
    start=time.perf_counter();got=scan(cache,train,heldout,[rows[i]['record_id'] for i in heldout],thresholds,panels,plan['block_size'])
    elapsed=time.perf_counter()-start;require(got['pairs']==256*1024,'complete_pilot_pairs')
    # Recheck three cuts with the original full guard implementation. Small gather
    # and reference memory/time are separately included in whole-child accounting.
    subset=train+heldout;fixture_rows=[rows[i] for i in subset];fixture_cache=cache[subset]
    parity=[];reference_start=time.perf_counter()
    for tau in plan['reference_thresholds']:
        k=thresholds.index(tau)
        _,kept,audit=panels.semantic_guard(fixture_rows,fixture_cache,tau,block_size=256)
        expected={r['record_id']:r for r in audit['removed']}
        actual={rows[i]['record_id']:{'record_id':rows[i]['record_id'],
                    'heldout_witness_id':rows[int(got['witness_indices'][j,k])]['record_id'],
                    'score':float(got['witness_scores'][j,k])}
                for j,i in enumerate(train) if got['witness_indices'][j,k]>=0}
        require(expected==actual,'original_guard_exact_witness_and_score_parity')
        require({rows[i]['record_id'] for i in train if rows[i]['record_id'] not in actual}==
                {fixture_rows[i]['record_id'] for i in kept if fixture_rows[i]['partition']=='train'},'exact_keep_parity')
        parity.append({'threshold':tau,'exact_witness_score_keep_parity':True})
    reference_seconds=time.perf_counter()-reference_start
    for path,wanted in watched.items():require(binding(path)==wanted,'input_changed')
    verify_package(args.package_sha256)
    report={'schema':'third-exact-guard-resource-benchmark-v1','passed':True,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],
        'package_sha256':args.package_sha256,'input_bindings':{str(k):v for k,v in watched.items()},
        'train_rows':len(train),'heldout_rows':len(heldout),'pairs':got['pairs'],'thresholds_prepared':len(thresholds),
        'exact_original_guard_parity':parity,'core_wall_seconds':elapsed,'reference_three_cut_wall_seconds':reference_seconds,
        'full_development_pairs':plan['full_pair_count'],'linear_pair_scaling_seconds':elapsed*plan['full_pair_count']/got['pairs'],
        'estimate_limit':'Small prefix benchmark only; excludes later full-input qualification, queue, full publication and scaling effects; not an ETA or runtime guarantee.',
        'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'full_development_guard_executed':False,'threshold_selected_by_benchmark':False,'semantic_validation':False,
        'test_labels_read':False,'utility_or_repair_outcomes_measured':False,'human_annotation_calls':0,
        'confirmatory_study_ready':False,'automatic_full_guard_launch':False}
    publish(out/'result.json',report);print(json.dumps({'passed':True,'pairs':got['pairs'],'core_wall_seconds':elapsed}),flush=True)


if __name__=='__main__':main()
