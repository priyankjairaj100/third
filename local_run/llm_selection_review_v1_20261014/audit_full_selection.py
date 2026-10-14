#!/usr/bin/env python3
"""Local full-selection metadata/identity audit; no full census replay or inference."""
import argparse
import base64
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import re
import sys
from types import SimpleNamespace

W=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace')
ROOT=W/'qualification/llm-selection-sampler-v1-20261014-selection-a01'
SAMPLER_SHA='c1da01e546c4ee0b5e84503b8513ec66f6ee5ae1ded16f46b2ce6b7801e18ce2'
HELPER_SHA='bcd8d75df5c8e973314ea664ed3acc249c94a75804c8965c9ba842af9524ef21'
PROTOCOL_SHA='56e97fe1b55f4047f4b8ddac1c7db00be22d69d45b270c8b31beab33d685aa01'
FRAME_SHA='d4fc45ec0d38ddc94c2a9c82ee3948020d0acd1f47fd84582b87ad5e273aedb1'
EVIDENCE_SHA='25a0dae8bad0023ec1183b0ca4978ee1b2e64aedf20f7b0fe1cc3aad8810b754'
PLAN_SHA='7910272ad6d05e012b101c782ea2268b5585eb6ab0f9b49f38c1aac9b5e13419'
PREFIX='third-llm-only-v3/'
OUTPUT_NAMES={'input_bindings.json','selection.private.jsonl','selection_swap.private.jsonl','result.json'}
FALSE_FLAGS=('threshold_selected','model_calibration_complete','semantic_validity_established',
 'original_human_gate_satisfied','inference_activation','confirmatory_study_ready',
 'source_authenticity_verified','primary_input_acceptance','original_route_acceptance',
 'full_population_cache_acceptance','same_source_pairs_excluded','availability_filter_applied',
 'requires_human_annotations','engineering_subset_sampled')


def require(value,message):
    if not value:raise ValueError(message)

def canonical(v,ascii_only=False):
    return json.dumps(v,ensure_ascii=ascii_only,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def strict(raw):
    def pairs(items):
        v={}
        for k,x in items:require(k not in v,'duplicate_json_key');v[k]=x
        return v
    def bad(_):raise ValueError('nonfinite_json')
    def finite(text):
        value=float(text);require(math.isfinite(value),'nonfinite_json_float');return value
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=finite)

def raw_binding(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def file_binding(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'regular_file')
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    after=path.stat();require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                             (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'changed_during_read')
    return {'bytes':after.st_size,'sha256':h.hexdigest()}

def sealed(value,ascii_only=False):
    require(value.get('sha256')==hashlib.sha256(canonical({k:v for k,v in value.items() if k!='sha256'},ascii_only)).hexdigest(),'internal_seal')

def package(path,wanted=None):
    p=Path(path);b=file_binding(p/'package_manifest.json')
    if wanted:require(b['sha256']==wanted,'package_identity')
    m=strict((p/'package_manifest.json').read_bytes());require(m['files'],'empty_package')
    for name,value in m['files'].items():
        require(Path(name).name==name and name not in ('.','..','package_manifest.json'),'member_name')
        require(file_binding(p/name)==value,'package_member')
    return b

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def domain(kind,values):return hashlib.sha256(canonical([PREFIX+kind,*values])).hexdigest()

def parse_lines(raw):
    require(not raw or raw.endswith(b'\n'),'physical_LF_required')
    return [strict(line) for line in raw.split(b'\n')[:-1]]

def rational(v):
    require(isinstance(v,dict) and set(v)=={'numerator','denominator'} and all(type(v[k]) is int and v[k]>0 for k in v),'positive_rational')
    f=Fraction(v['numerator'],v['denominator']);require((f.numerator,f.denominator)==(v['numerator'],v['denominator']),'unreduced_rational');return f

def verify_sample(result,sample,swaps,lock,plan):
    counts=result['bin_populations']; sizes=result['bin_sample_counts']
    require(len(counts)==len(sizes)==20 and all(type(x) is int and x>=0 for x in counts+sizes),'bin_population_schema')
    require(sum(counts)==220405510 and sizes==[min(30,x) for x in counts],'bin_totals')
    require(len(sample)==sum(sizes)==result['selected_pairs']<=600,'sample_counts')
    seen=set();seen_ids=set();observed=[0]*20;weight_total=Fraction(0);previous=(-1,-1)
    index_mapping={};record_mapping={};full_mapping={}
    for row in sample:
        require(set(row)=={'frame_row_indices','original_full_row_indices','record_ids','source_unit_ids','same_source',
          'natural_pair_id','opaque_pair_id','canonical_record_ids','canonical_frame_row_indices','score_hex','bin',
          'bin_population','bin_sample_size','inclusion_probability','inclusion_weight','model_seeds'},'sample_row_inventory')
        ij=row['frame_row_indices'];require(len(ij)==2 and all(type(x) is int for x in ij) and 0<=ij[0]<ij[1]<20996,'pair_indices')
        pair=tuple(ij);require(pair>previous and pair not in seen,'sample_order_or_duplicate');previous=pair;seen.add(pair)
        ids=row['record_ids'];sources=row['source_unit_ids'];original=row['original_full_row_indices']
        require(len(ids)==len(sources)==len(original)==2 and all(isinstance(x,str) and x for x in ids+sources)
                and ids[0]!=ids[1] and all(type(x) is int for x in original) and 0<=original[0]<original[1]<140250,'sample_identity_schema')
        for k in (0,1):
            mapping=(original[k],ids[k],sources[k])
            require(ij[k] not in index_mapping or index_mapping[ij[k]]==mapping,'inconsistent_index_mapping')
            require(ids[k] not in record_mapping or record_mapping[ids[k]]==ij[k],'record_multiple_indices')
            require(original[k] not in full_mapping or full_mapping[original[k]]==ij[k],'fullrow_multiple_indices')
            index_mapping[ij[k]]=mapping;record_mapping[ids[k]]=ij[k];full_mapping[original[k]]=ij[k]
        require(row['same_source'] is (sources[0]==sources[1]),'same_source_claim')
        sorted_ids=sorted(ids,key=lambda x:x.encode('utf-8'))
        natural=domain('natural-pair',[FRAME_SHA,sorted_ids]);opaque=domain('opaque-pair',[PLAN_SHA,FRAME_SHA,natural])
        orientation=int(domain('orientation',[PLAN_SHA,FRAME_SHA,natural]),16)%2
        oriented=sorted_ids[::-1] if orientation else sorted_ids
        require(row['natural_pair_id']==natural and row['opaque_pair_id']==opaque and opaque not in seen_ids,'pair_identity')
        seen_ids.add(opaque)
        require(row['canonical_record_ids']==oriented and row['canonical_frame_row_indices']==[ij[ids.index(x)] for x in oriented],'canonical_orientation')
        score=float.fromhex(row['score_hex']);require(math.isfinite(score) and score.hex()==row['score_hex'],'score_hex')
        b=max(0,min(19,sum(score>=float(Fraction(k,10)) for k in range(-10,11))-1))
        require(all(type(row[k]) is int for k in ('bin','bin_population','bin_sample_size')) and row['bin']==b and row['bin_population']==counts[b] and row['bin_sample_size']==sizes[b],'sample_bin')
        observed[b]+=1;require(counts[b]>0 and sizes[b]>0,'empty_selected_bin')
        require(rational(row['inclusion_probability'])==Fraction(sizes[b],counts[b]) and
                rational(row['inclusion_weight'])==Fraction(counts[b],sizes[b]),'inclusion_weights')
        weight_total+=rational(row['inclusion_weight'])
        expected_seeds={m['model_id']:int(domain('model-seed',[PLAN_SHA,FRAME_SHA,opaque,m['model_id'],m['revision']]),16)%2**63 for m in plan['models']}
        require(row['model_seeds']==expected_seeds,'model_seeds')
    require(observed==sizes and weight_total==220405510,'sample_bin_counts_or_weighted_denominator')
    n=(len(sample)+9)//10;chosen=[]
    rng=random.Random(int(domain('sampling',[PLAN_SHA,FRAME_SHA,'selection_swap',lock['seed_hex']['selection_swap'],None]),16))
    for index in range(len(sample)):
        if index<n:chosen.append(index)
        else:
            position=rng.randrange(index+1)
            if position<n:chosen[position]=index
    expected=[]
    for i in sorted(chosen):
        row=sample[i];expected.append({'selection_index':i,'opaque_pair_id':row['opaque_pair_id'],
          'canonical_frame_row_indices':row['canonical_frame_row_indices'],
          'swapped_frame_row_indices':row['canonical_frame_row_indices'][::-1],
          'model_seeds':row['model_seeds'],'diagnostic_only':True})
    require(swaps==expected and len(swaps)==result['diagnostic_swap_pairs']<=60,'swap_SRS_replay')
    return {'selected_pairs':len(sample),'swap_pairs':len(swaps),'exact_weighted_denominator':220405510,
            'all_sample_identity_orientation_weight_checks_passed':True,'independent_swap_reservoir_replay_passed':True}

def read_export(path):
    raw=Path(path).read_bytes();require(len(raw)<=32<<20,'export_cap');e=strict(raw)
    require(e['root']==str(ROOT) and re.fullmatch('[1-9][0-9]*',str(e['job_id'])) and not e['queue'].strip(),'export_root_queue')
    job=str(e['job_id']);parents=[line.split('|') for line in e['accounting'].splitlines() if line.split('|')[0]==job]
    require(len(parents)==1 and len(parents[0])==6 and parents[0][1:3]==['COMPLETED','0:0'] and parents[0][5]=='csis.cn1','exact_terminal_parent')
    require(all(len(line.split('|'))==6 and line.split('|')[1:3]==['COMPLETED','0:0'] and line.split('|')[5]=='csis.cn1'
                for line in e['accounting'].splitlines() if line.strip()),'failed_accounting_step')
    files={};total=0
    for name,entry in e['files'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'export_relative_path')
        content=base64.b64decode(entry['base64'],validate=True);b=raw_binding(content)
        require(b=={k:entry[k] for k in ('bytes','sha256')},'export_file_binding');total+=len(content)
        require(len(content)<=16<<20 and total<=20<<20,'export_payload_cap');files[name]=content
    require(total==e['total_bytes'],'export_total_bytes')
    require({x.removeprefix('run/output/') for x in files if x.startswith('run/output/')}==OUTPUT_NAMES,'full_output_inventory')
    for name in ('launch_receipt.json','launch_intent.json','run/process_completion.json','run/software_validation.json','run/tests.txt','run/stdout.txt','run/stderr.txt'):
        require(name in files,'missing_process_evidence:'+name)
    expected={'launch_receipt.json','launch_intent.json','run/process_completion.json','run/software_validation.json',
              'run/tests.txt','run/stdout.txt','run/stderr.txt','slurm-'+job+'.out','slurm-'+job+'.err',
              *{'run/output/'+name for name in OUTPUT_NAMES}}
    require(set(files)==expected,'unexpected_export_inventory')
    return e,files,raw_binding(raw)

def verify_terminal(e,files,control,control_binding,lock_binding,q_binding,review_binding):
    get=lambda name:strict(files[name]);launch=get('launch_receipt.json');intent=get('launch_intent.json');proc=get('run/process_completion.json');tests=get('run/software_validation.json')
    require(launch['schema']=='third-llm-sampler-launch-v1' and launch['mode']=='selection' and launch['root']==str(ROOT)
            and launch['returncode']==0 and str(launch['job_id'])==str(e['job_id']) and launch['control_package']==control_binding,'launch_identity')
    require({k:v for k,v in launch.items() if k not in ('returncode','stdout','stderr','job_id')}==intent,'intent_receipt_parity')
    for key in ('automatic_retry','human_annotation_required','inference_activation','source_authenticity_verified'):
        require(launch[key] is False,'launch_scope')
    require(launch['human_reviews_planned']==launch['model_inference_calls']==0,'launch_no_annotations')
    arg=SimpleNamespace(mode='selection',sampler_package_sha256=SAMPLER_SHA,lock_package_sha256=HELPER_SHA,
         sampling_lock_sha256=lock_binding['sha256'],qualification_result_sha256=q_binding['sha256'],qualification_review_sha256=review_binding['sha256'])
    expected=control.child_command(arg)
    require(proc['command']==launch['expected_child_command']==expected,'exact_child_command')
    require(proc['returncode']==0 and proc['console_cap_exceeded'] is False and str(proc['slurm_job_id'])==str(e['job_id'])
            and proc['sampler_package_sha256']==SAMPLER_SHA and math.isfinite(proc['child_wall_seconds_through_exit'])
            and proc['child_wall_seconds_through_exit']>0,'child_completion')
    require(tests['passed'] is True and tests['tests']==22 and tests['failures']==tests['errors']==tests['skipped']==0,'software_tests')
    command=launch['command'];control_remote=W/'tools/llm-sampler-control-v1-20261014'
    prefix=['sbatch','--parsable','--nodelist=csis.cn1','--job-name=third-llm-sampler-selection','--time=60',
            '--output',str(ROOT/'slurm-%j.out'),'--error',str(ROOT/'slurm-%j.err'),str(control_remote/'sampler.sbatch'),str(control_remote)]
    require(command[:len(prefix)]==prefix and str(control.FULL_ROOT)==str(ROOT),'submission_scope')
    tail=command[len(prefix):];require(len(tail)%2==0 and len(set(tail[::2]))==len(tail[::2]),'submission_arg_shape')
    expected_args={'--mode':'selection','--control-package-sha256':control_binding['sha256'],
      '--sampler-package-sha256':SAMPLER_SHA,'--lock-package-sha256':HELPER_SHA,'--sampling-lock-sha256':lock_binding['sha256'],
      '--qualification-result-sha256':q_binding['sha256'],'--qualification-review-sha256':review_binding['sha256'],
      '--expected-tests':'22','--time-limit-minutes':'60'}
    require(dict(zip(tail[::2],tail[1::2]))==expected_args,'complete_submission_arguments')
    require(proc['console_bytes']==len(files['run/stdout.txt'])+len(files['run/stderr.txt']),'console_count')
    testlog=files['run/tests.txt'].decode();require('Ran 22 tests in ' in testlog and testlog.rstrip().endswith('OK')
              and testlog.count(' ... ok\n')==22,'raw_test_log')
    require(launch['reserve_bytes']==8<<30 and launch['output_allowance_bytes']==128<<20,'capacity_policy')
    return {'job_id':str(e['job_id']),'parent':'COMPLETED','exit_code':'0:0','node':'csis.cn1',
            'child_wall_seconds_through_exit':proc['child_wall_seconds_through_exit'],'tests':22,'skips':0}

def retained_scores(args,sample,frame,repo):
    if not args.vectors and not args.rowmap:
        return {'executed':False,'retained_scores_independently_verified':False,'record_to_frame_mapping_independently_verified':False,
                'reason':'Original complete frame vectors and rowmap were not supplied; exported hashes are execution evidence only.'}
    require(args.vectors is not None and args.rowmap is not None,'both_optional_frame_files_required')
    require(file_binding(args.vectors)==frame['files']['vectors.npy'] and file_binding(args.rowmap)==frame['files']['records.private.jsonl'],'optional_frame_payload_hash')
    import numpy as np
    sys.path.insert(0,str(repo/'empirical_execution'))
    from phase3 import panels
    require(Path(panels.__file__).resolve()==(repo/'empirical_execution/phase3/panels.py').resolve(),'scorer_origin')
    vectors=np.load(args.vectors,mmap_mode='r',allow_pickle=False);rows=parse_lines(Path(args.rowmap).read_bytes())
    require(vectors.shape==(20996,768) and vectors.dtype==np.dtype('float32') and len(rows)==20996,'optional_frame_shape')
    for row in sample:
        i,j=row['frame_row_indices'];actual=[rows[i],rows[j]]
        require([r['record_id'] for r in actual]==row['record_ids'] and [r['source_unit_id'] for r in actual]==row['source_unit_ids']
                and [r['original_full_row_index'] for r in actual]==row['original_full_row_indices'],'retained_mapping')
        score=panels.reference_cosines(panels.graph_normalize(vectors[i:i+1]),panels.graph_normalize(vectors[j:j+1]))
        require(float(score[0,0]).hex()==row['score_hex'],'retained_score')
    return {'executed':True,'retained_scores_independently_verified':True,'record_to_frame_mapping_independently_verified':True,
            'pairs_checked':len(sample),'numpy':np.__version__,'vectors':file_binding(args.vectors),'rowmap':file_binding(args.rowmap)}

def audit(args):
    repo=Path(args.repo).resolve();sampler=repo/'local_run/llm_selection_sampler_v1_20261014';helper=repo/'local_run/llm_sampling_lock_v1_20261014'
    sb=package(sampler,SAMPLER_SHA);hb=package(helper,HELPER_SHA);package(repo/'local_run/model_annotation_v3_20261014',PROTOCOL_SHA)
    control_dir=repo/'local_run/llm_sampler_control_v1_20261014';cb=package(control_dir,args.control_package_sha256)
    control=module('reviewed_sampler_control',control_dir/'control.py');h=module('reviewed_lock_helper',helper/'sampling_lock.py')
    pins=strict((sampler/'expectations.json').read_bytes())
    for path,wanted in pins['frozen_files'].items():require(file_binding(repo/path)==wanted,'frozen_code')
    fraw=Path(args.frame_result).read_bytes();frame=strict(fraw);require(raw_binding(fraw)==pins['frame_result'],'frame_result_raw');sealed(frame,True)
    plan=strict((repo/'local_run/model_annotation_v3_20261014/PLAN.json').read_bytes())
    qraw=Path(args.qualification_result).read_bytes();q=strict(qraw);qb=raw_binding(qraw);h.check_qualification(q,sb)
    rraw=Path(args.qualification_review).read_bytes();r=strict(rraw);rb=raw_binding(rraw);h.check_review(r,qb,sb)
    lraw=Path(args.sampling_lock).read_bytes();lock=strict(lraw);lb=raw_binding(lraw)
    intent=strict(Path(args.lock_intent).read_bytes())
    snapshot=lock['contract']['protocol_evidence_snapshot'];require(snapshot['sha256']==EVIDENCE_SHA,'historical_snapshot_pin')
    expected=h.contract(frame,sb,qb,rb,snapshot);h.validate_lock(lock,expected,hb)
    require(intent['schema']=='third-llm-only-sampling-lock-intent-v1' and intent['contract']==expected
            and intent['helper_package']==hb and intent['entropy_not_yet_requested'] is True and intent['no_retry_or_alternate_slot'] is True,'lock_intent')
    e,files,export_binding=read_export(args.terminal_export)
    terminal=verify_terminal(e,files,control,cb,lb,qb,rb)
    raw=files['run/output/result.json'];result=strict(raw);sealed(result)
    require(result['schema']=='third-llm-selection-sample-v1' and result['mode']=='selection' and result['passed'] is True
            and result['stage']=='complete' and result['sampler_package']==sb and result['source_frame_result']==pins['frame_result']
            and result['protocol_plan_sha256']==PLAN_SHA and result['frame_package_sha256']==pins['frame_package_sha256'],'selection_result')
    require(result['selected_frame_rows']==20996 and result['unordered_pair_count']==220405510 and result['scoring_block_size']==256
            and result['rng_stages']==['selection','selection_swap'] and result['sampling_performed'] is True
            and result['production_sampling_performed'] is True,'full_population_scope')
    expected_work={'rows':20996,'unordered_pairs':220405510,'blocks':83,'upper_tiles':3486,'computed_tile_entries':223102992,'coordinate_products':171343097856}
    require(result['workload']==result['full_frame_workload']==expected_work,'work_geometry')
    require(all(result[k] is False for k in FALSE_FLAGS) and result['model_inference_calls']==result['human_annotation_calls']==0,'result_claim_scope')
    require(result['runtime']['slurm_job_id']==str(e['job_id']) and result['runtime']['python']=='3.13.7' and
            result['runtime']['numpy']=='2.5.3' and result['runtime']['hostname']=='csis.cn1'
            and result['runtime']['interpreter']==str(W/'envs/encoder-cpu/bin/python')
            and type(result['runtime']['numeric_threads']) is int and result['runtime']['numeric_threads']==1
            and result['runtime']['gpu_used'] is False and result['runtime']['rng']=='Python random.Random MT19937'
            and result['runtime']['concurrent_node_load_not_controlled'] is True
            and isinstance(result['runtime']['cpu_affinity'],list) and result['runtime']['cpu_affinity']
            and all(type(x) is int and x>=0 for x in result['runtime']['cpu_affinity'])
            and len(set(result['runtime']['cpu_affinity']))==len(result['runtime']['cpu_affinity']),'execution_runtime')
    le={'sampling_lock':lb,'qualification_result':qb,'qualification_review':rb,'lock_helper_package':hb}
    require(result['lock_evidence']==le and result['parity'] is None and result['diagnostic_seeds'] is None
            and result['diagnostic_frame_artifacts_are_exact_first512_slice'] is False,'result_lock_bindings')
    require(set(result['files'])==OUTPUT_NAMES-{'result.json'},'selection_file_inventory')
    for name,wanted in result['files'].items():require(raw_binding(files['run/output/'+name])==wanted,'result_output_binding')
    inputs=strict(files['run/output/input_bindings.json'])
    require(inputs['sampler_package']==sb and inputs['source_frame_result']==pins['frame_result'] and inputs['lock_evidence']==le
            and inputs['frame_payloads_rehashed'] is True and inputs['original_source_bytes_rehashed'] is True
            and inputs['encoder_forward_and_replay_validation_reexecuted'] is False,'input_receipt')
    watched=inputs['files'];require(watched[str(W/pins['frame_relative']/'result.json')]==pins['frame_result'],'watched_frame_result')
    for name,wanted in frame['files'].items():require(watched[str(W/pins['frame_relative']/name)]==wanted,'watched_frame_payload')
    for name,key in [('records.jsonl','prepared_source_records_file'),('audit.json','prepared_source_audit_file')]:
        require(watched[str(W/pins['source_relative']/name)]==frame[key],'watched_source')
    require(watched[str(W/'qualification/llm-only-v3-sampling-lock-a01/sampling_lock.json')]==lb
            and watched[str(W/'qualification/llm-only-v3-sampling-lock-a01/intent.json')]==file_binding(args.lock_intent),'watched_lock')
    for name,wanted in pins['frozen_files'].items():require(watched[str(Path('/nfs_home/users/poonam/ccu-acl2027/repo')/name)]==wanted,'executed_scorer_pin')
    for folder,wanted in [('llm-selection-sampler-v1-20261014',sb),('llm-sampling-lock-v1-20261014',hb)]:
        require(watched[str(W/'tools'/folder/'package_manifest.json')]==wanted,'executed_package_pin')
    require(watched[str(W/'tools/model-annotation-v3-evidence-v1-20261014/snapshot_manifest.json')]==snapshot,'executed_snapshot_pin')
    require(re.fullmatch('[0-9a-f]{64}',result['score_stream_sha256']) and
            result['score_stream_encoding']=='ASCII i TAB j TAB float.hex(score) LF','score_stream_attestation')
    sample=parse_lines(files['run/output/selection.private.jsonl']);swaps=parse_lines(files['run/output/selection_swap.private.jsonl'])
    checked=verify_sample(result,sample,swaps,lock,plan);optional=retained_scores(args,sample,frame,repo)
    return {'schema':'third-llm-selection-metadata-identity-review-v1','passed':True,'state':'COMPLETED',
      'scope':'Independent local exported-artifact, identity, exact-weight and swap audit. Full census/bin totals/selection reservoir/stream SHA remain original execution attestations under the qualified pinned sampler.',
      'audit_source':file_binding(__file__),'terminal_export':export_binding,'terminal':terminal,
      'selection_result':raw_binding(raw),'selection_result_internal_seal':result['sha256'],'sampler_package':sb,
      'lock_helper_package':hb,'control_package':cb,'sampling_lock':lb,'lock_intent':file_binding(args.lock_intent),
      'source_frame_result':pins['frame_result'],'qualification_result':qb,'qualification_review':rb,
      'checked':checked,'optional_retained_replay':optional,
      'complete_census_independently_replayed':False,'selection_reservoir_independently_replayed':False,
      'bin_population_counts_independently_recomputed':False,'score_stream_independently_recomputed':False,
      'execution_attestations':{'workload':result['workload'],'bin_populations':result['bin_populations'],
        'score_stream_sha256':result['score_stream_sha256'],'numerical_audit':result['numerical_audit']},
      'next_inference_builder_bindings':{'selection_files':result['files'],'frame_files':frame['files'],
        'frame_raw_result_sha256':FRAME_SHA,'plan_raw_sha256':PLAN_SHA,'sampling_lock':lb,
        'ordered_ids_sha256':frame['calibration_ordered_ids_sha256'],
        'source_records':frame['prepared_source_records_file'],'source_audit':frame['prepared_source_audit_file'],
        'required_text_lookup':'Original full row index plus exact record_id and text_sha256 from bound rowmap; never vector position alone.',
        'inference_activation_still_required':True,'fresh_validation_calls_required':True},
      'local_review_runtime':{'python':platform.python_version()},'model_inference_calls':0,'human_annotation_calls':0,
      'inference_activation':False,'semantic_validity_established':False,'confirmatory_study_ready':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','terminal-export','sampling-lock','lock-intent','frame-result','qualification-result','qualification-review','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--control-package-sha256',required=True)
    p.add_argument('--vectors',type=Path);p.add_argument('--rowmap',type=Path);args=p.parse_args()
    require(not args.out.exists() and not args.out.is_symlink(),'new_review_output_required')
    try:report=audit(args)
    except Exception as exc:
        report={'schema':'third-llm-selection-metadata-identity-review-v1','passed':False,'state':'FAILED',
                'error_type':type(exc).__name__,'error':str(exc),'complete_census_independently_replayed':False,
                'inference_activation':False,'model_inference_calls':0,'human_annotation_calls':0}
    with args.out.open('x') as f:f.write(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':report['passed'],'review':str(args.out)}))
    return 0 if report['passed'] else 1

if __name__=='__main__':sys.exit(main())
