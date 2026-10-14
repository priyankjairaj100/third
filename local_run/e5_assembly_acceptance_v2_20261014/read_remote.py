#!/usr/bin/env python3
"""Read-only prospective assembly status/metadata export; no payload reads/jobs."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import sys

GATE_SHA = '94147011564dffa8354680ca3e0a912d3a32667b8a9c791a083ad68858ebc616'
GATE_BYTES = 1303
GATE_RESULT_SHA = '8ceccfbda805bcf7a4e73b304ea88bdd2c218c3af5f1cb584240b9818b09c047'
GATE_BUNDLE_SHA = '393a90c6f4518f9e9292b8ebce07eebc09d4588697bd67c322c40ebabd24feb6'
LAUNCHER_BINDING = {'bytes':787,'sha256':'66511a7a14deaca6c73a7d9c04fc925b9942cbef72aa9f39def7c33e6036f092'}
STAGE_BUDGET_BINDING = {'bytes':6117,'sha256':'3068158ad2379cbfe251ca38ad313e17c6860488cf292f7b0cff93c71796275c'}
FIELDS = ('JobIDRaw','State','ExitCode','ElapsedRaw','MaxRSS','NodeList')
SCHEMA = 'third-e5-canonical-assembly-metadata-export-v2'


def load_gate(folder):
    folder=Path(folder);manifest=folder/'package_manifest.json'
    require= lambda yes,why: None if yes else (_ for _ in ()).throw(ValueError(why))
    require(manifest.is_file() and not manifest.is_symlink(),'regular_gate_manifest_required')
    raw=manifest.read_bytes();require(len(raw)==GATE_BYTES and hashlib.sha256(raw).hexdigest()==GATE_SHA,'sealed_gate_manifest_mismatch')
    for name,wanted in json.loads(raw)['files'].items():
        require(Path(name).name==name and name not in ('.','..'),'unsafe_gate_member')
        path=folder/name;require(path.is_file() and not path.is_symlink(),'regular_gate_member_required')
        raw=path.read_bytes();require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==wanted,'sealed_gate_member_changed')
    spec=importlib.util.spec_from_file_location('prior_assembly_metadata_gate',folder/'gate.py')
    gate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gate;spec.loader.exec_module(gate)
    return gate


def scheduler(job):
    queue=subprocess.check_output(['squeue','-u','poonam','-h','-o','%i|%T|%M|%R'],text=True,timeout=10)
    accounting=subprocess.check_output(['sacct','-j',job,'-P','-n','--format='+','.join(FIELDS)],text=True,timeout=10)
    return {'observed_utc':datetime.now(timezone.utc).isoformat(),'queue_stdout':queue,'accounting_stdout':accounting}


def parent_row(value,job,gate,require_success=True):
    active=[line for line in value['queue_stdout'].splitlines() if line.split('|',1)[0].strip()==job]
    parents=[dict(zip(FIELDS,line.split('|'))) for line in value['accounting_stdout'].splitlines() if line.split('|',1)[0].strip()==job]
    if require_success:
        gate.require(not active and len(parents)==1 and parents[0]['State']=='COMPLETED' and parents[0]['ExitCode']=='0:0' and parents[0]['NodeList']=='csis.cn1','exact_terminal_successful_assembly_on_cn1_required')
    return active,parents


def validate_launch(launch,wanted_sha,item,gate):
    gate.require(re.fullmatch('[0-9a-f]{64}',wanted_sha) is not None and item['sha256']==wanted_sha,'reviewed_assembly_launch_SHA_mismatch')
    gate.require(launch['schema']=='third-canonical-assembly-launch-v1' and launch['source_array_job_id']=='13498' and launch['returncode']==0,'assembly_launch_not_successful')
    job=str(launch['job_id']);gate.require(re.fullmatch('[0-9]+',job) is not None,'invalid_assembly_job')
    gate.require(launch['gate_package_manifest']=={'bytes':GATE_BYTES,'sha256':GATE_SHA},'launch_gate_package_changed')
    gate.require(launch.get('launcher_package_manifest')==LAUNCHER_BINDING and launch.get('stage_budget_report')==STAGE_BUDGET_BINDING and launch.get('budget_policy')=='third-assembly-only-capacity-policy-v1','reviewed_stage_launcher_or_budget_changed')
    quota=launch['quota_before_launch'];materialized=launch['materialized_output_budget']
    gate.require(quota['remaining_output_allowance_bytes']==832*1024**2 and quota['reserve_bytes']==8*1024**3 and type(quota['parsed']['available_soft_bytes']) is int and quota['parsed']['available_soft_bytes']>=832*1024**2+8*1024**3,'stage_capacity_or_unchanged_reserve_mismatch')
    gate.require(materialized['remaining_allowance_bytes']==832*1024**2 and materialized['original_output_allowance_bytes']==9*1024**3//4 and materialized['directory_and_stat_slack_bytes']==16*1024**2 and type(materialized['sum_max_logical_or_allocated_bytes']) is int and materialized['sum_max_logical_or_allocated_bytes']>=0 and materialized['sum_max_logical_or_allocated_bytes']+16*1024**2+832*1024**2<=9*1024**3//4,'original_total_output_envelope_changed')
    gate.require(launch['actual_payload_validation_deferred_to_sealed_Slurm_assembly_before_publication'] is True and all(launch[k] is False for k in ('full_population_cache_acceptance','confirmatory_study_ready','automatic_retry')),'assembly_launch_scope_changed')
    return job


def collect(gate,wanted_sha):
    pins=gate.expectations();root=pins['root'];stage=root+'/assembly-run';audit=root+'/terminal-metadata-audit-v1-a01'
    receipt_path=root+'/assembly_launch_receipt.json';receipt=gate.read_small(receipt_path);launch=gate.strict_json(base64.b64decode(receipt['base64']))
    job=validate_launch(launch,wanted_sha,receipt,gate);state=scheduler(job);parent_row(state,job,gate)
    names=[receipt_path,root+'/assembly_launch_intent.json',audit+'/bundle.json',audit+'/metadata_gate_result.json',
        stage+'/launch.json',stage+'/dispatch_result.json',stage+'/assembly.process.json',stage+'/cache/manifest.json']
    metadata={receipt_path:receipt}
    for path in names[1:]:metadata[path]=gate.read_small(path,cap=gate.CAP)
    value={'schema':SCHEMA,'root':root,'source_array_job_id':'13498','assembly_job_id':job,'assembly_launch_sha256':wanted_sha,
        'reader_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scheduler':state,'metadata':metadata,
        'payload_stats':{stage+'/cache/'+name:gate.file_stat(stage+'/cache/'+name) for name in ('vectors.npy','chunk_log.jsonl')},
        'log_stats':{stage+'/'+name:gate.file_stat(stage+'/'+name) for name in ('assembly.stdout.txt','assembly.stderr.txt')},
        'inventories':{path:gate.directory_inventory(path) for path in (stage,stage+'/cache')}}
    return value


def verify(value,gate,wanted_sha):
    pins=gate.expectations();root=pins['root'];stage=root+'/assembly-run';audit=root+'/terminal-metadata-audit-v1-a01'
    gate.require(value['schema']==SCHEMA and value['root']==root and value['source_array_job_id']=='13498' and value['assembly_launch_sha256']==wanted_sha,'wrong_assembly_export_identity')
    gate.require(value['reader_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reader_code_changed')
    expected={root+'/assembly_launch_receipt.json',root+'/assembly_launch_intent.json',audit+'/bundle.json',audit+'/metadata_gate_result.json',
        stage+'/launch.json',stage+'/dispatch_result.json',stage+'/assembly.process.json',stage+'/cache/manifest.json'}
    gate.require(set(value['metadata'])==expected,'exact_assembly_metadata_inventory_required')
    blobs={}
    for path,item in value['metadata'].items():
        raw=base64.b64decode(item['base64'],validate=True)
        gate.require(len(raw)<=gate.CAP and gate.binding(raw)=={k:item[k] for k in ('bytes','sha256')} and item['stat']['size']==len(raw),'metadata_raw_binding_mismatch')
        blobs[path]=raw
    gate.require(sum(map(len,blobs.values()))<=gate.CAP,'decoded_metadata_cap')
    get=lambda p:gate.strict_json(blobs[p]);bind=lambda p:gate.binding(blobs[p])
    launch=get(root+'/assembly_launch_receipt.json');job=validate_launch(launch,wanted_sha,bind(root+'/assembly_launch_receipt.json'),gate)
    gate.require(value['assembly_job_id']==job,'assembly_job_mismatch');_,parents=parent_row(value['scheduler'],job,gate)
    gate.require(get(root+'/assembly_launch_intent.json')=={k:v for k,v in launch.items() if k not in ('returncode','stdout','stderr','job_id')},'assembly_intent_receipt_mismatch')
    prior=get(audit+'/bundle.json');review=get(audit+'/metadata_gate_result.json');checked=gate.verify_bundle(prior)
    gate.require(bind(audit+'/metadata_gate_result.json')['sha256']==GATE_RESULT_SHA and bind(audit+'/bundle.json')['sha256']==GATE_BUNDLE_SHA,'independently_reviewed_prior_metadata_identity_changed')
    gate.require(all(review.get(k)==v for k,v in checked.items()),'prior_metadata_result_summary_mismatch')
    gate.require(review['bundle_file']=={'name':'bundle.json',**bind(audit+'/bundle.json')} and launch['reviewed_metadata_bundle']==bind(audit+'/bundle.json') and launch['reviewed_metadata_result']==bind(audit+'/metadata_gate_result.json'),'reviewed_preassembly_evidence_changed')
    priorraw=lambda p:base64.b64decode(prior['metadata'][p]['base64'],validate=True)
    priorget=lambda p:gate.strict_json(priorraw(p));priorbind=lambda p:gate.binding(priorraw(p))
    names=gate.paths(pins);plan=priorget(names['plan']);qualification=priorget(names['qualification']);budget=priorget(names['budget']);original=priorget(root+'/plan_launch_receipt.json')
    expected_bindings={'core_package':priorbind(names['core']),'dispatch_package':priorbind(names['dispatch']),'qualification_result':priorbind(names['qualification']),'full_plan':priorbind(names['plan'])}
    for key,wanted in expected_bindings.items():gate.require(launch[key]==wanted,'assembly_launch_dependency_changed')
    gate.require(launch['scipy_repair_result']==priorbind(names['repair']),'assembly_repair_binding_changed')
    common=original['common_arguments'];dispatch_folder=pins['repository']+'/local_run/e5_sharded_dispatch_v2_20261013'
    command=['sbatch','--parsable','--nodelist=csis.cn1','--job-name=third-e5-assembly-13498','--output',root+'/assembly-%j.out','--error',root+'/assembly-%j.err',dispatch_folder+'/assembly.sbatch',*common,names['plan'],priorbind(names['plan'])['sha256'],priorbind(names['dispatch'])['sha256']]
    gate.require(launch['command']==command,'exact_assembly_submission_command_required')
    execution=get(stage+'/launch.json');dispatch=get(stage+'/dispatch_result.json');process=get(stage+'/assembly.process.json');manifest=get(stage+'/cache/manifest.json')
    gate.require(execution['mode']==dispatch['mode']=='assembly' and execution['shard_index'] is None and dispatch['shard_index'] is None and execution['slurm_job_id']==job and execution['hostname']=='csis.cn1','assembly_execution_identity_mismatch')
    gate.require(execution['bindings']==dispatch['bindings']==expected_bindings and dispatch['passed'] is True,'assembly_dispatch_failed_or_changed_bindings')
    gate.require(all(d.get(key) is False for d in (execution,dispatch,process) for key in gate.DISPATCH_FALSE_FLAGS),'dispatch_claim_boundary_changed')
    children=dispatch['children'];gate.require(len(children)==1 and children[0]['name']=='assembly' and children[0]['artifact']==bind(stage+'/cache/manifest.json') and children[0]['process_receipt']==bind(stage+'/assembly.process.json'),'exact_successful_manifest_child_binding_required')
    arguments=[v for name,arg in zip(('repo','workspace','view','replay-result','census-dir','budget-dir'),common[:6]) for v in ('--'+name,arg)]
    arguments+=['--package-sha256',expected_bindings['core_package']['sha256'],'--qualification-result',names['qualification'],'--qualification-sha256',expected_bindings['qualification_result']['sha256']]
    cmd=[pins['workspace']+'/envs/encoder-cpu/bin/python',pins['repository']+'/local_run/e5_sharded_cpu_v2_20261013/driver.py','assemble',*arguments,'--plan',names['plan'],'--out',stage+'/cache','--encoded-dirs']
    cmd += [root+f'/shards/shard-{i:04d}/encode' for i in range(32)]
    cmd += ['--replay-dirs']+[root+f'/shards/shard-{i:04d}/replay' for i in range(32)]
    gate.require(process['schema']=='third-e5-sharded-dispatch-child-v1' and type(process['returncode']) is int and process['returncode']==0 and process['command']==cmd,'child_command_or_exit_mismatch')
    gate.require(all(type(process[k]) is int and process[k]>0 for k in ('child_pid','launcher_pid')),'child_process_identity_missing')
    seconds=process['child_wall_seconds_through_exit'];gate.require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds>0 and children[0]['child_wall_seconds_through_exit']==seconds and process['timer_includes_child_artifact_publication_and_exit'] is True,'child_wall_time_boundary_mismatch')
    gate.seal(manifest)
    identity={'schema':'third-e5-sharded-cpu-cache-v1','passed':True,'plan_sha256':plan['sha256'],'fingerprint':pins['fingerprint'],'rows':140250,'shape':[140250,768],'dtype':'float32','frame_ordered_ids_sha256':plan['frame_ordered_ids_sha256'],'full_source_frame_sha256':plan['full_source_frame_sha256']}
    gate.require(manifest['passed'] is True and all(manifest.get(k)==v for k,v in identity.items()),'cache_manifest_identity_mismatch')
    gate.require(gate.canonical(manifest['runtime'])==gate.canonical(qualification['runtime']) and manifest['all_rows_fresh_process_replayed_under_this_version'] is True,'cache_runtime_or_replay_claim_mismatch')
    gate.require(all(manifest[k] is False for k in ('original_monolithic_full_writer_executed','original_monolithic_full_replay_executed','source_authenticity_verified','primary_input_acceptance','full_population_cache_acceptance','confirmatory_study_ready')),'cache_claim_boundary_changed')
    shard_receipts=[{'index':i,'encoded_completion':priorbind(root+f'/shards/shard-{i:04d}/encode/completion.json'),'replay_completion':priorbind(root+f'/shards/shard-{i:04d}/replay/completion.json')} for i in range(32)]
    gate.require(manifest['shard_receipts']==shard_receipts,'all32_shard_receipt_bindings_required')
    files=manifest['files'];gate.require(set(files)=={'vectors.npy','chunk_log.jsonl'},'cache_file_reference_inventory_mismatch')
    expected_sizes={'vectors.npy':budget['exact_known_payload']['assembly_vectors_npy_bytes'],'chunk_log.jsonl':budget['exact_known_payload']['one_complete_chunk_log_bytes']}
    gate.require(files['chunk_log.jsonl']['sha256']==budget['exact_known_payload']['complete_chunk_log_sha256_if_passed'],'cache_log_reference_does_not_match_precounted_source_log')
    for name,size in expected_sizes.items():
        claim=files[name];gate.require(set(claim)=={'bytes','sha256'} and type(claim['bytes']) is int and claim['bytes']==size and re.fullmatch('[0-9a-f]{64}',claim['sha256']) is not None,'cache_size_or_hash_reference_mismatch')
        gate.require(value['payload_stats'][stage+'/cache/'+name]['size']==size,'actual_cache_file_size_mismatch')
    gate.require(set(value['payload_stats'])=={stage+'/cache/'+name for name in files} and set(value['log_stats'])=={stage+'/assembly.stdout.txt',stage+'/assembly.stderr.txt'},'cache_stat_inventory_mismatch')
    gate.require(value['inventories']=={stage:sorted(['launch.json','dispatch_result.json','assembly.process.json','assembly.stdout.txt','assembly.stderr.txt','cache']),stage+'/cache':['chunk_log.jsonl','manifest.json','vectors.npy']},'cache_or_stage_has_missing_extra_files')
    return {'schema':'third-e5-canonical-assembly-metadata-verification-v2','passed_metadata_checks':True,'assembly_job_id':job,'source_array_job_id':'13498','terminal_parent':parents[0],
        'launcher_package_manifest':LAUNCHER_BINDING,'stage_budget_report':STAGE_BUDGET_BINDING,
        'cache_manifest_file':bind(stage+'/cache/manifest.json'),'cache_manifest_internal_seal':manifest['sha256'],'cache_payload_hash_references':files,'rows':140250,'shape':[140250,768],'dtype':'float32','shard_pairs':32,
        'sealed_assembly_execution_reports_payload_checks_passed':True,'independent_cache_payload_hashes_verified':False,
        'independent_numerical_or_source_validation_performed':False,'original_route_acceptance':False,'full_population_cache_acceptance':False,'confirmatory_study_ready':False,
        'scope':'Successful sealed Slurm assembly and exact metadata bindings. Payload content validation is attested by the unchanged assembly execution; this reader independently checks metadata, stat identities and sizes only.'}


def recheck(value,gate):
    for path,item in value['metadata'].items():gate.require(gate.read_small(path,cap=gate.CAP)==item,'assembly_metadata_changed')
    for key in ('payload_stats','log_stats'):
        for path,observed in value[key].items():gate.require(gate.file_stat(path)==observed,'cache_or_log_stat_changed')
    for path,names in value['inventories'].items():gate.require(gate.directory_inventory(path)==names,'assembly_inventory_changed')
    root=gate.expectations()['root'];raw=base64.b64decode(value['metadata'][root+'/terminal-metadata-audit-v1-a01/bundle.json']['base64'])
    gate.recheck_observations(gate.strict_json(raw))


def status(gate,wanted_sha):
    root=gate.expectations()['root'];item=gate.read_small(root+'/assembly_launch_receipt.json');launch=gate.strict_json(base64.b64decode(item['base64']))
    job=validate_launch(launch,wanted_sha,item,gate);state=scheduler(job);active,parents=parent_row(state,job,gate,False)
    records={}
    for name in ('assembly-run/dispatch_result.json','assembly-run/assembly.process.json','assembly-run/cache/manifest.json'):
        path=Path(root)/name
        if path.exists():
            raw=gate.read_small(path);data=gate.strict_json(base64.b64decode(raw['base64']))
            records[name]={k:data.get(k) for k in ('passed','returncode','error_type','error','rows','shape') if k in data}
    diagnostics={}
    if not active and not(len(parents)==1 and parents[0]['State']=='COMPLETED' and parents[0]['ExitCode']=='0:0'):
        for name in (f'assembly-{job}.out',f'assembly-{job}.err','assembly-run/assembly.stderr.txt'):
            path=Path(root)/name
            if path.exists():
                before=gate.file_stat(path)
                with path.open('rb') as stream:stream.seek(max(0,before['size']-4000));diagnostics[name]=stream.read(4000).decode('utf-8','replace')
    return {'schema':'third-e5-canonical-assembly-status-v2','assembly_job_id':job,'scheduler':state,'queue':active,'parent_rows':parents,'observed_metadata':records,'failure_diagnostics':diagnostics,'metadata_verification_performed':False,'full_population_cache_acceptance':False,'confirmatory_study_ready':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gate-package',type=Path,required=True);parser.add_argument('--assembly-launch-sha256',required=True);parser.add_argument('--mode',choices=('status','export'),required=True);args=parser.parse_args()
    gate=load_gate(args.gate_package)
    def timeout(_signum,_frame):raise TimeoutError('assembly_metadata_reader_exceeded_180_seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    try:
        if args.mode=='status':value=status(gate,args.assembly_launch_sha256)
        else:
            value=collect(gate,args.assembly_launch_sha256);value['verification']=verify(value,gate,args.assembly_launch_sha256);recheck(value,gate)
            parent_row(scheduler(value['assembly_job_id']),value['assembly_job_id'],gate)
        raw=gate.canonical(value);gate.require(len(raw)+1<=gate.CAP,'export_exceeds_25MiB');print(raw.decode())
    finally:signal.alarm(0)


if __name__=='__main__':main()
