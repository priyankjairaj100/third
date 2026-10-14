#!/usr/bin/env python3
"""Offline independent replay of the bounded sampler qualifier, never production sampling."""
import argparse
import base64
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

PACKAGES = {
 'llm_selection_sampler_v1_20261014': 'c1da01e546c4ee0b5e84503b8513ec66f6ee5ae1ded16f46b2ce6b7801e18ce2',
 'llm_sampling_lock_v1_20261014': 'bcd8d75df5c8e973314ea664ed3acc249c94a75804c8965c9ba842af9524ef21',
 'llm_sampler_control_v1_20261014': 'a1779a1ea98b0968ab46680be6a0249983381bcd25a02a2c405877a32b237cf9',
 'model_annotation_v3_20261014': '56e97fe1b55f4047f4b8ddac1c7db00be22d69d45b270c8b31beab33d685aa01',
 'e5_calibration_frame_v1_20261014': '015ea8122ba5b7a366eba6f96fc3e4d64767bb7d8425c48b3b047ff5fad87fa8'}
W = '/nfs_home/users/poonam/ccu-acl2027/local_workspace'
R = '/nfs_home/users/poonam/ccu-acl2027/repo'
ROOT = W+'/qualification/llm-selection-sampler-v1-20261014-qualify-a01'
PREFIX = 'third-llm-only-v3/'
SEEDS = {k: hashlib.sha256((PREFIX+'engineering-first512-v1/'+k).encode()).hexdigest()
         for k in ('selection','selection_swap')}
STAGES = ['engineering_qualification_selection','engineering_qualification_selection_swap']


def require(condition, message):
    if not condition: raise ValueError(message)


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def strict(raw):
    def unique(pairs):
        result={}
        for k,v in pairs:
            require(k not in result,'duplicate_json_key'); result[k]=v
        return result
    def invalid(value): raise ValueError('nonfinite_json:'+value)
    return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid)


def binding(raw): return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def read(path, cap=16 << 20):
    path=Path(path).absolute()
    require('..' not in path.parts and all(not p.is_symlink() for p in (path,*path.parents)), 'unsafe_path')
    require(path.is_file() and path.stat().st_size <= cap,'file_cap_or_type')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')), 'changed_read')
    return raw


def domain(kind, values): return hashlib.sha256(canonical([PREFIX+kind,*values])).hexdigest()


def rng(plan, frame, stage, seed, bin_index):
    return random.Random(int(domain('sampling',[plan,frame,stage,seed,bin_index]),16))


def fraction(v): return {'numerator':v.numerator,'denominator':v.denominator}


def independent_replay(vectors, rows, plan_sha, frame_sha, models):
    """No production/frozen math, traversal, identity or reservoir functions imported."""
    import numpy as np
    require(vectors.shape == (512,768) and vectors.dtype == np.dtype('float32') and vectors.flags.c_contiguous,'diagnostic_vector_shape')
    require(np.isfinite(vectors).all() and np.any(vectors != 0,axis=1).all(),'invalid_vector')
    values=np.asarray(vectors,dtype=np.float64)
    squared=np.zeros(len(values),dtype=np.float64)
    for coordinate in range(values.shape[1]):
        squared += values[:,coordinate]*values[:,coordinate]
    normalized=values/np.sqrt(squared)[:,None]
    scores=np.zeros((len(values),len(values)),dtype=np.float64)
    for coordinate in range(values.shape[1]):
        scores += normalized[:,coordinate,None]*normalized[None,:,coordinate]
    # A separate scalar Python arithmetic path for the first19 complete pairs.
    scalar=[]
    for vector in vectors[:19]:
        terms=[float(x) for x in vector]; norm2=0.0
        for value in terms: norm2 += value*value
        norm=math.sqrt(norm2); scalar.append([value/norm for value in terms])
    pairwise=0
    for i in range(19):
        for j in range(i+1,19):
            total=0.0
            for a,b in zip(scalar[i],scalar[j]): total += a*b
            require(total.hex()==float(scores[i,j]).hex(),'scalar171_score_difference')
            pairwise += 1
    populations=[0]*20;reservoirs=[[] for _ in populations]
    generators=[rng(plan_sha,frame_sha,STAGES[0],SEEDS['selection'],b) for b in range(20)]
    edges=[float(Fraction(k,10)) for k in range(-10,11)]
    grid=[float(Fraction(k,100)) for k in range(50,100)]+[float(Fraction(199,200))]
    stream=hashlib.sha256();pairs=0
    audit={'band':1e-12,'recomputed_pairs':0,'maximum_absolute_score_difference':0.0,
      'threshold_decision_disagreements':0,'examples':[],
      'independent_accumulation':'math.fsum of FP64 component products after shared FP64 normalization',
      'out_of_range_raw_scores':0,'authority':'shared coordinate-ordered FP64 reference decides, including any disagreement',
      'real_arithmetic_certificate':False}
    for left in (0,256):
        for right in range(left,512,256):
            for i in range(left,left+256):
                for j in range(max(i+1,right),right+256):
                    score=float(scores[i,j]); require(math.isfinite(score),'nonfinite_score')
                    stream.update((str(i)+'\t'+str(j)+'\t'+score.hex()+'\n').encode('ascii'));pairs+=1
                    b=max(0,min(19,sum(score >= e for e in edges)-1)); populations[b]+=1
                    item=(i,j,score,b)
                    if populations[b] <= 30: reservoirs[b].append(item)
                    else:
                        position=generators[b].randrange(populations[b])
                        if position < 30: reservoirs[b][position]=item
                    audit['out_of_range_raw_scores']+=int(not -1<=score<=1)
                    if any(abs(score-t)<=1e-12 for t in grid):
                        check=math.fsum(float(a)*float(b) for a,b in zip(normalized[i],normalized[j]))
                        diff=abs(score-check);disagreements=sum((score>t)!=(check>t) for t in grid)
                        audit['recomputed_pairs']+=1
                        audit['maximum_absolute_score_difference']=max(audit['maximum_absolute_score_difference'],diff)
                        audit['threshold_decision_disagreements']+=disagreements
                        if disagreements and len(audit['examples'])<20:
                            audit['examples'].append({'i':i,'j':j,'reference_score_hex':score.hex(),'check_score_hex':check.hex()})
    sample=[]
    for i,j,score,b in sorted((item for reservoir in reservoirs for item in reservoir),key=lambda x:x[:2]):
        record_ids=[rows[i]['record_id'],rows[j]['record_id']]
        sorted_ids=sorted(record_ids,key=lambda value:value.encode('utf-8'))
        natural=domain('natural-pair',[frame_sha,sorted_ids])
        opaque=domain('opaque-pair',[plan_sha,frame_sha,natural])
        reverse=int(domain('orientation',[plan_sha,frame_sha,natural]),16)%2
        canonical_ids=sorted_ids[::-1] if reverse else sorted_ids
        probability=Fraction(min(30,populations[b]),populations[b])
        sample.append({'frame_row_indices':[i,j], 'original_full_row_indices':[rows[i]['original_full_row_index'],rows[j]['original_full_row_index']],
          'record_ids':record_ids,'source_unit_ids':[rows[i]['source_unit_id'],rows[j]['source_unit_id']],
          'same_source':rows[i]['source_unit_id']==rows[j]['source_unit_id'],
          'natural_pair_id':natural,'opaque_pair_id':opaque,'canonical_record_ids':canonical_ids,
          'canonical_frame_row_indices':[i,j] if record_ids==canonical_ids else [j,i],
          'score_hex':score.hex(),'bin':b,'bin_population':populations[b],'bin_sample_size':min(30,populations[b]),
          'inclusion_probability':fraction(probability),'inclusion_weight':fraction(1/probability),
          'model_seeds':{m['model_id']:int(domain('model-seed',[plan_sha,frame_sha,opaque,m['model_id'],m['revision']]),16)%(2**63) for m in models}})
    generator=rng(plan_sha,frame_sha,STAGES[1],SEEDS['selection_swap'],None)
    chosen=[];capacity=(len(sample)+9)//10
    for index in range(len(sample)):
        if index < capacity: chosen.append(index)
        else:
            position=generator.randrange(index+1)
            if position < capacity: chosen[position]=index
    swaps=[{'selection_index':i,'opaque_pair_id':sample[i]['opaque_pair_id'],
      'canonical_frame_row_indices':sample[i]['canonical_frame_row_indices'],
      'swapped_frame_row_indices':list(reversed(sample[i]['canonical_frame_row_indices'])),
      'model_seeds':sample[i]['model_seeds'],'diagnostic_only':True} for i in sorted(chosen)]
    require(pairs==130816 and pairwise==171,'replay_incomplete')
    return {'sample':sample,'swaps':swaps,'populations':populations,'stream_sha256':stream.hexdigest(),
            'pairwise_pairs':pairwise,'pair_count':pairs,'numerical_audit':audit}


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def validate(args):
    start=time.perf_counter();repo=Path(args.repo);root=Path(args.artifacts);watched={}
    def watch(path):
        raw=read(path);wanted=binding(raw);key=str(Path(path).absolute())
        require(key not in watched or watched[key]==wanted,'changed_watch');watched[key]=wanted;return raw
    package_bindings={};package_manifests={}
    for name,sha in PACKAGES.items():
        folder=repo/'local_run'/name;raw=watch(folder/'package_manifest.json')
        require(binding(raw)['sha256']==sha,'unreviewed_package:'+name)
        package_bindings[name]=binding(raw);package_manifests[name]=strict(raw)
        for member,wanted in strict(raw)['files'].items():
            require(Path(member).name==member and member not in ('.','..'),'unsafe_package_member')
            require(binding(watch(folder/member))==wanted,'changed_package_member:'+member)
    sampler=package_bindings['llm_selection_sampler_v1_20261014']
    pins=strict(watch(repo/'local_run/llm_selection_sampler_v1_20261014/expectations.json'))
    planraw=watch(repo/'local_run/model_annotation_v3_20261014/PLAN.json');plan=strict(planraw)
    require(binding(planraw)['sha256']==pins['plan_sha256'],'plan_pin')
    for name,wanted in pins['frozen_files'].items(): require(binding(watch(repo/name))==wanted,'frozen_source_changed')
    exported=watch(args.accounting_json);export=strict(exported)
    require(export['schema']=='third-llm-selection-qualification-terminal-export-v1' and export['root']==ROOT,'export_scope')
    job=export['job_id'];require(re.fullmatch('[1-9][0-9]*',job) is not None,'job_identity')
    inventory={'launch_intent.json','launch_receipt.json','run/process_completion.json','run/software_validation.json',
      'run/tests.txt','run/stdout.txt','run/stderr.txt','slurm-'+job+'.out','slurm-'+job+'.err',
      *['run/output/'+n for n in ('result.json','input_bindings.json','selection.private.jsonl','selection_swap.private.jsonl','diagnostic_frame_vectors.npy','diagnostic_frame_rows.private.jsonl')]}
    require(set(export['files'])==inventory,'export_inventory')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual==inventory|{'terminal_export.json'},'local_artifact_inventory')
    copied={}
    for name,entry in export['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe_export_path')
        raw=base64.b64decode(entry['base64'],validate=True)
        require(binding(raw)=={k:entry[k] for k in ('bytes','sha256')} and watch(root/name)==raw,'transport_binding:'+name)
        copied[name]=raw
    require(sum(map(len,copied.values()))==export['total_bytes']<=16<<20,'export_total')
    require(not export['queue'].strip(),'job_still_queued')
    accounting=[line.split('|') for line in export['accounting'].strip().split('\n')]
    parent=[line for line in accounting if line[0]==job]
    require(len(parent)==1 and len(parent[0])==6 and parent[0][1:3]==['COMPLETED','0:0'] and parent[0][5]=='csis.cn1' and int(parent[0][3])>0,'terminal_accounting')
    require(all(line[1:3]==['COMPLETED','0:0'] and line[5]=='csis.cn1' for line in accounting),'failed_job_step')
    obj=lambda name:strict(copied[name])
    launch=obj('launch_intent.json');receipt=obj('launch_receipt.json');process=obj('run/process_completion.json');software=obj('run/software_validation.json')
    require(receipt==dict(launch,job_id=job,returncode=0,stdout=job+'\n',stderr=''),'launch_receipt_mismatch')
    require(launch['automatic_retry'] is False and launch['mode']=='qualification' and launch['root']==ROOT,'launch_scope')
    require(launch['reserve_bytes']==8<<30 and launch['output_allowance_bytes']==128<<20 and launch['quota']['available_soft_bytes']>= (8<<30)+(128<<20),'launch_quota')
    control=load_module('independent_review_command_only',repo/'local_run/llm_sampler_control_v1_20261014/control.py')
    command=control.child_command(argparse.Namespace(mode='qualification',sampler_package_sha256=sampler['sha256']))
    require(command==launch['expected_child_command']==process['command'],'exact_child_command')
    expected_submit=['sbatch','--parsable','--nodelist=csis.cn1','--job-name=third-llm-sampler-qualification','--time=20',
      '--output',ROOT+'/slurm-%j.out','--error',ROOT+'/slurm-%j.err',W+'/tools/llm-sampler-control-v1-20261014/sampler.sbatch',
      W+'/tools/llm-sampler-control-v1-20261014','--mode','qualification','--control-package-sha256',PACKAGES['llm_sampler_control_v1_20261014'],
      '--sampler-package-sha256',sampler['sha256'],'--lock-package-sha256',PACKAGES['llm_sampling_lock_v1_20261014'],
      '--expected-tests','22','--time-limit-minutes','20']
    require(launch['command']==expected_submit,'submission_command')
    require(process['returncode']==0 and process['console_cap_exceeded'] is False and process['slurm_job_id']==job
      and process['sampler_package_sha256']==sampler['sha256'] and math.isfinite(process['child_wall_seconds_through_exit'])
      and process['child_wall_seconds_through_exit']>0,'process_failure')
    require(process['console_bytes']==len(copied['run/stdout.txt'])+len(copied['run/stderr.txt']),'console_count')
    require(software['passed'] is True and software['tests']==22 and all(software[k]==0 for k in ('skipped','errors','failures')),'software_tests')
    testtext=copied['run/tests.txt'].decode();require('Ran 22 tests in ' in testtext and testtext.rstrip().endswith('OK') and testtext.count(' ... ok\n')==22,'test_log')
    result=obj('run/output/result.json');qbinding=binding(copied['run/output/result.json'])
    require(result['sha256']==hashlib.sha256(canonical({k:v for k,v in result.items() if k!='sha256'})).hexdigest(),'result_seal')
    helper=load_module('independent_review_interface_check',repo/'local_run/llm_sampling_lock_v1_20261014/sampling_lock.py')
    helper.check_qualification(result,sampler)
    require(result['runtime']['slurm_job_id']==job and result['runtime']['interpreter']==W+'/envs/encoder-cpu/bin/python','runtime_job')
    require(result['diagnostic_frame_shape']==[512,768] and result['diagnostic_frame_dtype']=='float32','shape_attestation')
    require(result['score_stream_encoding']=='ASCII i TAB j TAB float.hex(score) LF','stream_encoding')
    for name,wanted in result['files'].items(): require(binding(copied['run/output/'+name])==wanted,'output_binding')
    inputs=obj('run/output/input_bindings.json');files=inputs['files']
    require(inputs['frame_payloads_rehashed'] is True and inputs['original_source_bytes_rehashed'] is True
      and inputs['encoder_forward_and_replay_validation_reexecuted'] is False and inputs['lock_evidence'] is None
      and inputs['source_frame_result']==pins['frame_result'] and inputs['sampler_package']==sampler,'input_scope')
    for name,wanted in pins['frame_files'].items(): require(files[W+'/'+pins['frame_relative']+'/'+name]==wanted,'frame_binding')
    require(files[W+'/'+pins['frame_relative']+'/result.json']==pins['frame_result'],'source_frame_pin')
    for name,key in [('records.jsonl','prepared_source_records_file'),('audit.json','prepared_source_audit_file')]:
        require(files[W+'/'+pins['source_relative']+'/'+name]==pins['frame_inherited'][key],'source_original_pin')
    for name,wanted in pins['frozen_files'].items():require(files[R+'/'+name]==wanted,'executed_source_pin')
    # Rehash historical evidence locally, keeping its role separate from actual executing code.
    evidence=repo/'local_workspace/reports/model-annotation-v3-evidence-v1-20261014'
    snapshotraw=watch(evidence/'snapshot_manifest.json');snapshot=strict(snapshotraw)
    require(binding(snapshotraw)['sha256']==pins['protocol_evidence_sha256'] and snapshot['files']==plan['bound_files'] and len(snapshot['files'])==39,'snapshot_pin')
    require(files[W+'/tools/model-annotation-v3-evidence-v1-20261014/snapshot_manifest.json']==binding(snapshotraw),'executed_snapshot_pin')
    for name,wanted in snapshot['files'].items():
        require(binding(watch(evidence/name))==wanted and files[W+'/tools/model-annotation-v3-evidence-v1-20261014/'+name]==wanted,'historical_member')
    def lines(name):
        raw=copied['run/output/'+name];require(raw.endswith(b'\n'),'missing_final_LF')
        return [strict(line) for line in raw.split(b'\n')[:-1]]
    rows=lines('diagnostic_frame_rows.private.jsonl'); require(len(rows)==512,'diagnostic_rows')
    previous=-1;ids=set()
    for i,row in enumerate(rows):
        require(set(row)=={'calibration_row_index','original_full_row_index','record_id','source_unit_id','source_kind','partition','text_sha256'},'row_schema')
        require(type(row['calibration_row_index']) is int and row['calibration_row_index']==i and type(row['original_full_row_index']) is int
          and previous<row['original_full_row_index']<140250,'row_order');previous=row['original_full_row_index']
        require(all(isinstance(row[k],str) and row[k] for k in ('record_id','source_unit_id','source_kind','text_sha256')),'row_strings')
        require(row['record_id'] not in ids and re.fullmatch('[0-9a-f]{64}',row['text_sha256']),'row_identity');ids.add(row['record_id'])
        d=int.from_bytes(hashlib.sha256(('ccu-v1-source-split\0'+row['source_unit_id']).encode()).digest(),'big')
        require(row['partition']=='calibration' and d*10 >= (2**256)*7 and d*20 < (2**256)*17,'source_partition')
    import numpy as np
    vectorstream=io.BytesIO(copied['run/output/diagnostic_frame_vectors.npy']);vectors=np.load(vectorstream,allow_pickle=False)
    require(vectorstream.tell()==len(vectorstream.getvalue()),'trailing_npy_bytes')
    replay_start=time.perf_counter();replay=independent_replay(vectors,rows,pins['plan_sha256'],pins['frame_result']['sha256'],plan['models'])
    require(replay['stream_sha256']==result['score_stream_sha256'],'full512_score_stream')
    require(replay['populations']==result['bin_populations'] and [min(30,x) for x in replay['populations']]==result['bin_sample_counts'],'full512_bins')
    require(replay['sample']==lines('selection.private.jsonl') and len(replay['sample'])==result['selected_pairs'],'reservoir_identity_weights')
    require(replay['swaps']==lines('selection_swap.private.jsonl') and len(replay['swaps'])==result['diagnostic_swap_pairs'],'swap_identity')
    require(replay['numerical_audit']==result['numerical_audit'],'numerical_audit')
    replay_seconds=time.perf_counter()-replay_start
    # Final byte equality includes every exported evidence file and all local code/evidence read.
    for path,wanted in watched.items():require(binding(read(path))==wanted,'changed_during_review:'+path)
    report={'schema':'third-llm-selection-qualification-independent-review-v1','passed':True,
      'qualification_result':qbinding,'sampler_package':sampler,'source_frame_result':pins['frame_result'],
      'checks':{k:True for k in ('full512_score_stream','full512_bins_and_reservoir','first19_pairwise171','identities_weights_swaps')},
      'production_sampling_performed':False,
      'scope':'Independently rehashed all exported receipts and exact five outputs, reviewed package members and historical evidence; independently replayed all 130816 coordinate-ordered FP64 scores from the copied first512 vectors, all bins and per-bin reservoirs, complete identities/orientations/rational weights/model seeds and swap reservoir. Separate scalar Python path matched all171 first19 scores. Full source/frame byte verification and exact first512 extraction remain the bound Slurm execution attestation; full original payloads were not independently copied or rehashed locally.',
      'audit_script':binding(read(__file__)),'terminal_export':binding(exported),'job_id':job,'slurm_parent_state':parent[0],
      'transport_files_verified':len(copied),'local_watched_files_rehashed':len(watched),'package_bindings':package_bindings,
      'diagnostic_vector_file':result['files']['diagnostic_frame_vectors.npy'],'diagnostic_row_file':result['files']['diagnostic_frame_rows.private.jsonl'],
      'score_stream_sha256':replay['stream_sha256'],'pair_count':replay['pair_count'],'bin_populations':replay['populations'],
      'selected_pairs':len(replay['sample']),'diagnostic_swap_pairs':len(replay['swaps']),'numerical_audit':replay['numerical_audit'],
      'remote_runtime':result['runtime'],'remote_score_collector_seconds':result['score_collector_seconds'],
      'remote_child_wall_seconds_through_exit':process['child_wall_seconds_through_exit'],'remote_software_tests':software,
      'local_runtime':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'interpreter':sys.executable},
      'local_replay_seconds':replay_seconds,'review_seconds':time.perf_counter()-start,
      'full_frame_or_original_source_payloads_independently_rehashed_locally':False,
      'all_first512_scores_independently_recomputed_locally':True,'historical_evidence_is_executing_checkout':False,
      'full220405510_census_replayed':False,'production_entropy_draws':0,'model_inference_calls':0,'human_annotation_calls':0,
      'source_authenticity_verified':False,'semantic_validity_established':False,'inference_activation':False,
      'confirmatory_study_ready':False,'human_annotation_required':False,
      'limitations':['Engineering subset has already been exposed; all its pairs remain eligible in production.',
        'Timing on512rows is conditional engineering evidence, not a full-run upper bound.',
        'PRNG SRS denotes the fixed uniform-draw algorithmic model, not cryptographic exact randomness.',
        'Helper functions were used only for receipt-contract checks; no sampler/scorer/protocol numerical or identity helper was used for local replay.'],
      'created_utc':datetime.now(timezone.utc).isoformat()}
    helper.check_review(report,qbinding,sampler)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('artifacts','accounting-json','repo','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    try:
        require(not args.output.exists(),'new_output_required')
        report=validate(args);raw=json.dumps(report,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
        with args.output.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        print(json.dumps({'passed':True,'output':str(args.output),'binding':binding(raw),'pairs':report['pair_count']}));return 0
    except Exception as exc:
        print(json.dumps({'passed':False,'error_type':type(exc).__name__,'error':str(exc)}),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
