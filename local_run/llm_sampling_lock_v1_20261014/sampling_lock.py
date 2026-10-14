"""Bounded metadata-only sampling lock. No scoring, model imports or job launch."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys

FOLDER = Path(__file__).resolve().parent
SEED_NAMES = ('selection', 'validation', 'selection_swap', 'validation_swap')
PARITY_KEYS = ('full512_score_stream', 'full512_bins_and_reservoir', 'first19_pairwise171', 'identities_weights_swaps')
FALSE_FLAGS = ('sampling_performed', 'threshold_selected', 'model_calibration_complete',
 'inference_activation', 'source_authenticity_verified', 'original_route_acceptance',
 'full_population_cache_acceptance', 'human_completed', 'original_human_gate_satisfied',
 'semantic_validity_established', 'confirmatory_study_ready', 'requires_human_annotations')

class Refusal(ValueError): pass

def require(condition, message):
    if not condition: raise Refusal(message)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def digest(value): return hashlib.sha256(canonical(value)).hexdigest()

def seal(value):
    require('sha256' not in value, 'already_sealed')
    return dict(value, sha256=digest(value))

def verify_seal(value):
    require(isinstance(value, dict) and value.get('sha256') == digest({k:v for k,v in value.items() if k!='sha256'}), 'invalid_internal_seal')

def strict(raw):
    def unique(items):
        result={}
        for k,v in items:
            require(k not in result, 'duplicate_JSON_key');result[k]=v
        return result
    def invalid(value): raise Refusal('nonfinite_JSON_number')
    return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid)

def binding(raw): return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def check_binding(value):
    require(isinstance(value,dict) and set(value)=={'bytes','sha256'} and type(value['bytes']) is int and value['bytes']>0
      and isinstance(value['sha256'],str) and re.fullmatch('[0-9a-f]{64}',value['sha256']), 'invalid_raw_binding')
    return value

def pins(): return strict((FOLDER/'expectations.json').read_bytes())

def check_frame(frame):
    p=pins();verify_seal(frame)
    require(frame['sha256']==p['frame_internal_seal'] and frame.get('schema')=='third-e5-calibration-frame-v1'
      and frame.get('passed') is True and frame.get('stage')=='complete', 'frame_result_identity')
    require(frame.get('files')==p['frame_files'] and frame.get('frame_package_sha256')==p['frame_package']['sha256']
      and frame.get('fingerprint')==p['frame_fingerprint'], 'frame_package_or_payload_binding')
    for k,v in p['frame_metadata'].items(): require(type(frame.get(k)) is type(v) and frame.get(k)==v, 'frame_metadata_'+k)
    require(frame.get('within_calibration_same_source_exclusion') is False, 'same_source_pair_filter')
    for k in ('source_authenticity_verified','original_route_acceptance','full_population_cache_acceptance',
      'human_validation_claimed','requires_human_annotations','sampling_performed','threshold_selected',
      'semantic_validity_established','confirmatory_study_ready'):
        require(frame.get(k) is False, 'frame_scope_'+k)
    for k in ('actual_cache_payload_hashes_verified','all_32_encoded_and_fresh_replay_payloads_verified',
      'all_32_assembled_slices_exact','all_calibration_vectors_exact_FP32_gather','input_code_assets_initial_final_equal'):
        require(frame.get(k) is True, 'frame_attestation_'+k)

def check_qualification(result, sampler_binding):
    verify_seal(result);check_binding(sampler_binding);p=pins()
    require(result.get('schema')=='third-llm-selection-qualification-v1' and result.get('passed') is True
      and result.get('mode')=='qualification' and result.get('stage')=='complete', 'qualification_not_passed')
    require(result.get('source_frame_result')==p['frame_result'] and result.get('sampler_package')==sampler_binding,
      'qualification_input_or_package')
    for k,v in {'selected_frame_rows':512,'unordered_pair_count':130816,'scoring_block_size':256}.items():
        require(type(result.get(k)) is int and result[k]==v, 'qualification_count_'+k)
    require(result.get('production_sampling_performed') is False and all(result.get('parity',{}).get(k) is True for k in PARITY_KEYS),
      'qualification_parity_or_scope')
    require(isinstance(result.get('runtime'),dict) and result['runtime'] and isinstance(result.get('files'),dict) and result['files'],
      'qualification_runtime_and_artifacts_required')
    rt=result['runtime']
    for key,wanted in {'python':'3.13.7','numpy':'2.5.3','hostname':'csis.cn1','numeric_threads':1,
      'gpu_used':False,'rng':'Python random.Random MT19937','concurrent_node_load_not_controlled':True}.items():
        require(type(rt.get(key)) is type(wanted) and rt.get(key)==wanted, 'qualification_runtime_'+key)
    require(isinstance(rt.get('interpreter'),str) and rt['interpreter'].endswith('/envs/encoder-cpu/bin/python')
      and isinstance(rt.get('slurm_job_id'),str) and re.fullmatch('[1-9][0-9]*',rt['slurm_job_id'])
      and isinstance(rt.get('cpu_affinity'),list) and rt['cpu_affinity']
      and all(type(i) is int and i>=0 for i in rt['cpu_affinity']), 'qualification_runtime_execution_identity')
    require(set(result['files'])=={'input_bindings.json','selection.private.jsonl','selection_swap.private.jsonl',
      'diagnostic_frame_vectors.npy','diagnostic_frame_rows.private.jsonl'}, 'qualification_artifact_inventory')
    require(result.get('diagnostic_frame_artifacts_are_exact_first512_slice') is True
      and result['files']['diagnostic_frame_vectors.npy'].get('bytes')==512*768*4+128, 'qualification_copied_frame_binding')
    require(result.get('protocol_plan_sha256')==p['plan_file']['sha256'] and result.get('frame_package_sha256')==p['frame_package']['sha256'], 'qualification_protocol_pin')
    require(result.get('rng_stages')==p['sampling_algorithm']['diagnostic_stages']
      and result.get('diagnostic_seeds')=={k:hashlib.sha256(('third-llm-only-v3/engineering-first512-v1/'+k).encode()).hexdigest() for k in ('selection','selection_swap')}
      and result.get('lock_evidence') is None, 'qualification_diagnostic_randomness')
    populations=result.get('bin_populations');require(isinstance(populations,list) and len(populations)==20
      and all(type(n) is int and n>=0 for n in populations) and sum(populations)==130816, 'qualification_bin_population')
    counts=[min(30,n) for n in populations]
    require(result.get('bin_sample_counts')==counts and type(result.get('selected_pairs')) is int and result['selected_pairs']==sum(counts)
      and type(result.get('diagnostic_swap_pairs')) is int and result['diagnostic_swap_pairs']==(sum(counts)+9)//10, 'qualification_sample_counts')
    require(result.get('model_inference_calls')==0 and type(result.get('model_inference_calls')) is int
      and result.get('human_annotation_calls')==0 and type(result.get('human_annotation_calls')) is int, 'qualification_no_labels')
    for k in ('same_source_pairs_excluded','availability_filter_applied','requires_human_annotations',
      'threshold_selected','model_calibration_complete','semantic_validity_established','original_human_gate_satisfied',
      'inference_activation','confirmatory_study_ready','source_authenticity_verified','primary_input_acceptance',
      'original_route_acceptance','full_population_cache_acceptance'):
        require(result.get(k) is False, 'qualification_claim_'+k)
    for e in result['files'].values():check_binding(e)

def check_review(review, qualification_binding, sampler_binding):
    require(review.get('schema')=='third-llm-selection-qualification-independent-review-v1' and review.get('passed') is True,
      'independent_review_required')
    require(review.get('qualification_result')==qualification_binding and review.get('sampler_package')==sampler_binding
      and review.get('source_frame_result')==pins()['frame_result'], 'independent_review_binding')
    require(review.get('production_sampling_performed') is False and all(review.get('checks',{}).get(k) is True for k in PARITY_KEYS),
      'independent_review_checks')
    require(isinstance(review.get('scope'),str) and review['scope'].strip(), 'independent_review_scope')

def contract(frame_result, sampler_manifest_binding, qualification_binding, qualification_review_binding, protocol_evidence_binding):
    """Pure contract derivation from reviewed objects and this version's fixed pins."""
    check_frame(frame_result)
    for v in (sampler_manifest_binding,qualification_binding,qualification_review_binding,protocol_evidence_binding):check_binding(v)
    p=pins()
    return {'schema':'third-llm-only-sampling-contract-v1','protocol_package':p['protocol_package'],'plan_file':p['plan_file'],
      'protocol_evidence_snapshot':copy.deepcopy(protocol_evidence_binding),'evidence_snapshot_role':'historical_39_file_pin_evidence_only_not_executing_checkout',
      'frame_result':p['frame_result'],'frame_internal_seal':p['frame_internal_seal'],'frame_package':p['frame_package'],
      'frame_files':p['frame_files'],'frame_metadata':p['frame_metadata'],'frame_sha256':p['frame_result']['sha256'],
      'frame_fingerprint':p['frame_fingerprint'],'frame_role':p['frame_role'],'source_scope':p['source_scope'],
      'executing_repository_code':p['frozen_code'],'runtime':p['runtime'],'population':p['population'],
      'sampling_algorithm':p['sampling_algorithm'],'prior_exposure':p['prior_exposure'],
      'sampler_package':copy.deepcopy(sampler_manifest_binding),'bounded_qualification_result':copy.deepcopy(qualification_binding),
      'independent_qualification_review':copy.deepcopy(qualification_review_binding),
      'output_relative':p['output_relative'],'selection_root_relative':p['selection_root_relative']}

def build_lock(expected_contract, seed_hex, created_utc, helper_package_binding):
    """Pure constructor. Tests use explicit synthetic seeds; this function draws none."""
    check_binding(helper_package_binding)
    require(set(seed_hex)==set(SEED_NAMES) and all(isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) for x in seed_hex.values()),
      'four_exact_256bit_seed_values_required')
    date=datetime.fromisoformat(created_utc.replace('Z','+00:00'));require(date.tzinfo is not None,'timezone_required')
    return seal({'schema':'third-llm-only-sampling-lock-v1','created_utc':created_utc,'contract':copy.deepcopy(expected_contract),
      'helper_package':copy.deepcopy(helper_package_binding),'seed_hex':copy.deepcopy(seed_hex),
      'entropy':{'source':'os.urandom','separate_calls':4,'bytes_per_call':32,'generation_rounds':1,'regeneration_allowed':False,
        'scope':'Four independent OS draws after bounded engineering qualification; no seed search. Equality by chance does not trigger redraw.'},
      'role':'sampling_only_assessed_development','payload_hashes_verified_by_this_helper':False,
      'completed_census_required_before_lock':False,'completed_census_required_before_inference':True,
      'planned_human_reviews':0,**dict.fromkeys(FALSE_FLAGS,False)})

def validate_lock(lock, expected_contract, helper_package_binding):
    verify_seal(lock)
    expected=build_lock(expected_contract,lock.get('seed_hex',{}),lock.get('created_utc',''),helper_package_binding)
    require(lock==expected,'lock_contract_or_flags_changed')
    return {'passed':True,'scope':'Sampling commitment metadata only; no source authenticity, payload rehash, inference activation or scientific acceptance',
      'seed_count':4,'frame_sha256':expected_contract['frame_sha256'],'production_sampling_performed':False,'human_annotation_required':False}

def no_symlinks(path):
    path=Path(path).absolute()
    for part in (path,*path.parents):require(not part.is_symlink(),'symlink_refused')
    return path

def read_small(path, cap, watched=None):
    path=no_symlinks(path);st=path.stat();require(stat.S_ISREG(st.st_mode) and 0<st.st_size<=cap,'metadata_size_or_type')
    raw=path.read_bytes();require(len(raw)==st.st_size,'metadata_changed_during_read')
    if watched is not None:watched[path]=binding(raw)
    return raw

def read_bound(path, expected, watched=None):
    raw=read_small(path,pins()['limits']['metadata_file_bytes'],watched);require(binding(raw)==expected,'metadata_binding_mismatch')
    return strict(raw)

def verify_package(folder, expected_sha, watched):
    folder=no_symlinks(folder);p=pins();raw=read_small(folder/'package_manifest.json',p['limits']['metadata_file_bytes'],watched)
    require(binding(raw)['sha256']==expected_sha,'package_manifest_changed');manifest=strict(raw);files=manifest.get('files')
    require(isinstance(files,dict) and files,'empty_package');total=0
    for name,wanted in files.items():
        require(Path(name).name==name and name not in ('.','..','package_manifest.json'),'package_member_path')
        check_binding(wanted);require(total+wanted['bytes']<=p['limits']['package_total_bytes'],'package_total_cap')
        content=read_small(folder/name,p['limits']['package_file_bytes'],watched)
        require(binding(content)==wanted,'package_member_changed');total+=len(content)
    require(total<=p['limits']['package_total_bytes'],'package_total_cap');return binding(raw)

def verify_historical_snapshot(folder, expected_sha, plan, watched):
    folder=no_symlinks(folder);raw=read_small(folder/'snapshot_manifest.json',pins()['limits']['metadata_file_bytes'],watched)
    require(binding(raw)['sha256']==expected_sha,'historical_snapshot_manifest_changed')
    files=strict(raw).get('files');require(files==plan['bound_files'] and len(files)==39,'historical_snapshot_inventory')
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and str(p.relative_to(folder))!='snapshot_manifest.json'}
    require(actual==set(files),'historical_snapshot_extra_or_missing_files')
    total=0
    for name,wanted in files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'historical_snapshot_path')
        require(total+wanted['bytes']<=pins()['limits']['package_total_bytes'],'historical_snapshot_total_cap')
        content=read_small(folder/name,pins()['limits']['package_file_bytes'],watched);total+=len(content)
        require(binding(content)==wanted,'historical_snapshot_member_changed')
    require(total<=pins()['limits']['package_total_bytes'],'historical_snapshot_total_cap')
    return binding(raw)

def recheck(watched):
    for path,expected in watched.items():require(binding(read_small(path,pins()['limits']['package_file_bytes']))==expected,'initial_final_metadata_changed')

def durable_json(path,value):
    raw=canonical(value)+b'\n';require(len(raw)<=pins()['limits']['lock_file_bytes'],'lock_output_cap')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    sync_directory(Path(path).parent)
    return binding(raw)

def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def create_once(workspace, expected_contract, helper_binding, watched):
    workspace=no_symlinks(workspace).resolve(strict=True);out=workspace/pins()['output_relative']
    require(out.parent.is_dir(),'output_parent_required');no_symlinks(out)
    out.mkdir(mode=0o700,exist_ok=False);sync_directory(out.parent)
    try:
        durable_json(out/'intent.json',{'schema':'third-llm-only-sampling-lock-intent-v1','created_utc':datetime.now(timezone.utc).isoformat(),
          'contract':expected_contract,'helper_package':helper_binding,'hostname':socket.gethostname(),'pid':os.getpid(),
          'entropy_not_yet_requested':True,'no_retry_or_alternate_slot':True})
        # Exclusive durable intent precedes every production entropy call.
        raw_seeds={name:os.urandom(32) for name in SEED_NAMES}
        require(all(type(x) is bytes and len(x)==32 for x in raw_seeds.values()),'entropy_length_failure')
        lock=build_lock(expected_contract,{k:v.hex() for k,v in raw_seeds.items()},datetime.now(timezone.utc).isoformat(),helper_binding)
        recheck(watched);validate_lock(lock,expected_contract,helper_binding)
        result=durable_json(out/'sampling_lock.json',lock)
        return {'passed':True,'path':str(out/'sampling_lock.json'),'binding':result,'sampling_performed':False,'inference_activation':False}
    except Exception as error:
        # Do not retry entropy, remove intent, or permit another slot under this version.
        if not (out/'failure.json').exists():
            durable_json(out/'failure.json',{'passed':False,'error_type':type(error).__name__,'error':str(error),
              'production_entropy_may_have_been_consumed':True,'automatic_retry_allowed':False})
        raise

def collect(args):
    p=pins();watched={};repo=no_symlinks(args.repo).resolve(strict=True);workspace=no_symlinks(args.workspace).resolve(strict=True)
    helper=verify_package(FOLDER,args.helper_package_sha256,watched)
    protocol=verify_package(args.protocol_package,p['protocol_package']['sha256'],watched)
    require(protocol==p['protocol_package'],'protocol_package_binding')
    plan=read_bound(Path(args.protocol_package)/'PLAN.json',p['plan_file'],watched)
    snapshot=verify_historical_snapshot(args.protocol_evidence,args.protocol_evidence_sha256,plan,watched)
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('third_pinned_v3_protocol',Path(args.protocol_package)/'protocol.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    historical_validation=module.validate(plan,Path(args.protocol_evidence))
    require(historical_validation.get('passed') is True and historical_validation.get('bound_files_verified')==39, 'historical_protocol_validation_failed')
    # The snapshot is historical evidence, never a substitute executing checkout.
    for name,wanted in p['frozen_code'].items():read_bound_bytes=read_small(repo/name,p['limits']['package_file_bytes'],watched);require(binding(read_bound_bytes)==wanted,'executing_scorer_code_changed')
    frame_dir=no_symlinks(args.frame_dir).resolve(strict=True)
    require(frame_dir==workspace/p['frame_relative'],'fixed_frame_location_required')
    frame=read_bound(frame_dir/'result.json',p['frame_result'],watched);check_frame(frame)
    for name,wanted in p['frame_files'].items():
        path=no_symlinks(frame_dir/name);require(path.is_file() and path.stat().st_size==wanted['bytes'],'frame_payload_stat_mismatch')
    sampler=verify_package(args.sampler_package,args.sampler_package_sha256,watched)
    qualification_path=no_symlinks(args.qualification_result).resolve(strict=True)
    require(qualification_path.is_relative_to(workspace/p['qualification_root_relative']),'qualification_outside_fixed_root')
    qraw=read_small(qualification_path,p['limits']['metadata_file_bytes'],watched);qb=binding(qraw)
    require(qb['sha256']==args.qualification_result_sha256,'qualification_raw_hash');q=strict(qraw);check_qualification(q,sampler)
    require(q['runtime']['interpreter']==str(workspace/p['runtime']['interpreter_relative']), 'qualification_interpreter_wrong_workspace')
    review_raw=read_small(args.qualification_review,p['limits']['metadata_file_bytes'],watched);rb=binding(review_raw)
    require(rb['sha256']==args.qualification_review_sha256,'review_raw_hash');check_review(strict(review_raw),qb,sampler)
    return workspace,contract(frame,sampler,qb,rb,snapshot),helper,watched

def read_completed_lock(path, expected_contract, helper_package_binding):
    path=no_symlinks(path)
    require(path.name=='sampling_lock.json' and {p.name for p in path.parent.iterdir()}=={'intent.json','sampling_lock.json'}, 'partial_failed_or_extra_lock_inventory')
    intent=strict(read_small(path.parent/'intent.json',pins()['limits']['lock_file_bytes']))
    require(intent.get('schema')=='third-llm-only-sampling-lock-intent-v1' and intent.get('contract')==expected_contract
      and intent.get('helper_package')==helper_package_binding and intent.get('entropy_not_yet_requested') is True
      and intent.get('no_retry_or_alternate_slot') is True, 'intent_binding_changed')
    raw=read_small(path,pins()['limits']['lock_file_bytes']);lock=strict(raw)
    validate_lock(lock,expected_contract,helper_package_binding)
    require(datetime.fromisoformat(intent['created_utc']) <= datetime.fromisoformat(lock['created_utc']), 'intent_must_precede_draws')
    return lock,binding(raw)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('create','verify'))
    for key in ('repo','workspace','protocol-package','protocol-evidence','protocol-evidence-sha256','frame-dir','sampler-package',
      'sampler-package-sha256','qualification-result','qualification-result-sha256','qualification-review','qualification-review-sha256','helper-package-sha256'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    try:
        workspace,expected,helper,watched=collect(args)
        if args.mode=='create':result=create_once(workspace,expected,helper,watched)
        else:
            path=workspace/pins()['output_relative']/'sampling_lock.json'
            lock,raw_binding=read_completed_lock(path,expected,helper)
            result=validate_lock(lock,expected,helper);recheck(watched);result.update(path=str(path),binding=raw_binding)
        print(json.dumps(result,sort_keys=True));return 0
    except Exception as error:
        print(json.dumps({'passed':False,'error_type':type(error).__name__,'error':str(error),'automatic_retry_allowed':False},sort_keys=True),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
