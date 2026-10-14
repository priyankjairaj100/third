#!/usr/bin/env python3
"""Bounded terminal benchmark export; no submits, model/source payload scans or writes."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
import launch_remote as gate

TOTAL_CAP=32*1024**2
ENCODED_CAP=48*1024**2
MAX_FILES=64


def inventory(root):
    gate.require(root.is_dir() and not root.is_symlink(),'regular_report_directory_required')
    files=[];total=0;entries=0
    for path in root.rglob('*'):
        entries+=1;gate.require(entries<=128,'bounded_export_directory_inventory_exceeded')
        info=path.lstat()
        gate.require(not stat.S_ISLNK(info.st_mode),'symlink_in_export')
        if stat.S_ISDIR(info.st_mode):continue
        gate.require(stat.S_ISREG(info.st_mode),'nonregular_export_file')
        total+=info.st_size
        gate.require(info.st_size<=gate.CAP and total<=TOTAL_CAP,'bounded_export_size_exceeded_preserve_originals')
        files.append(path.relative_to(root).as_posix())
        gate.require(len(files)<=MAX_FILES,'bounded_export_inventory_exceeded')
    return sorted(files),total


def check_success(decoded,job_id,source):
    required={'launch_intent.json','launch_receipt.json',f'slurm-{job_id}.out',f'slurm-{job_id}.err',
        'run/software_validation.json','run/process_completion.json','run/stdout.txt','run/stderr.txt',
        'run/benchmark/input_bindings.json','run/benchmark/cases.jsonl','run/benchmark/result.json'}
    gate.require(required<=set(decoded),'successful_run_missing_artifacts')
    def load(name):return gate.strict(decoded[name])
    tests=load('run/software_validation.json')
    gate.require(tests.get('passed') is True and tests.get('tests')==16
        and all(tests.get(k)==0 for k in ('failures','errors','skipped')),'successful_run_tests_incomplete')
    process=load('run/process_completion.json')
    gate.require(process.get('returncode')==0 and process.get('package_sha256')==gate.PACKAGE_SHA,'child_exit_or_package_mismatch')
    gate.require(process.get('command')==gate.benchmark_command(),'benchmark_child_command_changed')
    elapsed=process.get('child_wall_seconds_through_exit')
    gate.require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0,'invalid_child_wall_time')
    result=load('run/benchmark/result.json')
    gate.require(result.get('schema')=='third-frozen-pair-scorer-benchmark-v1' and result.get('passed') is True
        and result.get('stage')=='complete' and result.get('initial_final_bindings_equal') is True
        and result.get('package_sha256')==gate.PACKAGE_SHA,'successful_run_result_mismatch')
    for key in ('qualification_result','qualification_vectors','repair_result'):
        gate.require(result.get(key)==source[key],'result_source_binding_mismatch')
    for name,wanted in result.get('artifacts',{}).items():
        gate.require(Path(name).name==name,'unsafe_result_artifact_name')
        raw=decoded.get('run/benchmark/'+name)
        gate.require(raw is not None and {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==wanted,'result_artifact_binding_mismatch')
    gate.require(set(result.get('artifacts',{}))=={'cases.jsonl','input_bindings.json'},'result_artifact_inventory_mismatch')
    cases=[gate.strict(line) for line in decoded['run/benchmark/cases.jsonl'].split(b'\n') if line]
    gate.require(len(cases)==28 and all(c.get('completed') is True for c in cases),'successful_run_case_count_mismatch')
    gate.require(result['summary'].get('cases')==28 and result['summary'].get('all_cases_completed') is True
        and result['summary'].get('full_target_pair_count_executed')==0,'successful_run_summary_mismatch')
    for key in ('human_sample_drawn','calibration_sample_drawn','threshold_selected','full_population_scored',
                'full_population_cache_acceptance','semantic_validity_established','confirmatory_study_ready'):
        gate.require(result.get(key) is False,'scientific_claim_boundary_changed')
    gate.require(all(result.get(k)==0 for k in ('model_forwards','new_model_judgments','human_responses')),'unexpected_model_or_human_work')


def export(job_id,control_sha):
    gate.require(isinstance(job_id,str) and job_id.isdecimal(),'expected_job_id_required')
    folder=Path(__file__).resolve().parent
    control=gate.verify_package(folder,control_sha)
    gate.require('export_remote.py' in gate.load(folder/'package_manifest.json')['files'],'exporter_not_in_reviewed_package')
    source=gate.source_check()
    launch=gate.load(gate.ROOT/'launch_receipt.json')
    intent=gate.load(gate.ROOT/'launch_intent.json')
    gate.require(launch.get('job_id')==job_id and launch.get('returncode')==0
        and launch.get('root')==str(gate.ROOT) and launch.get('benchmark_package_sha256')==gate.PACKAGE_SHA
        and launch.get('control_package_manifest')==control,'wrong_launch_receipt')
    gate.require(all(launch.get(k)==v for k,v in intent.items()),'intent_receipt_mismatch')
    gate.require(launch.get('initial_input_bindings')==source,'launch_source_pins_changed')
    gate.require(launch.get('command')==gate.sbatch_command(),'unexpected_submitted_command')
    raw=gate.command(['sacct','-j',job_id,'-P','-n','--format=JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    terminal=gate.terminal_parent(raw,job_id,success=False)
    queue=gate.command(['squeue','-u',gate.USER,'-h','-o','%i|%j'])
    for line in queue.splitlines():
        fields=line.split('|',1);gate.require(len(fields)==2,'unexpected_queue_format')
        gate.require(fields[0]!=job_id,'benchmark_job_still_active')
    files,total=inventory(gate.ROOT)
    decoded={name:gate.read_small(gate.ROOT/name) for name in files}
    success=terminal['State']=='COMPLETED' and terminal['ExitCode']=='0:0'
    if success:
        gate.require(terminal['NodeList']==gate.NODE,'benchmark_not_on_qualified_node')
        check_success(decoded,job_id,source)
    gate.require(inventory(gate.ROOT)==(files,total),'export_inventory_changed')
    for name,value in decoded.items():gate.require(gate.read_small(gate.ROOT/name)==value,'export_file_changed')
    gate.require(gate.source_check()==source,'source_pins_changed_during_export')
    gate.verify_package(folder,control_sha)
    rows=[{'path':name,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest(),
           'base64':base64.b64encode(value).decode('ascii')} for name,value in decoded.items()]
    result={'schema':'third-pair-scorer-benchmark-terminal-export-v1','root':str(gate.ROOT),
        'job_id':job_id,'terminal_success':success,'terminal_parent':terminal,'raw_accounting':raw,
        'fresh_queue':queue,'control_package_manifest':control,'benchmark_package_sha256':gate.PACKAGE_SHA,
        'source_pins':source,'files':rows,'decoded_bytes':sum(r['bytes'] for r in rows),
        'all_small_files_retained':True,'truncated_files':0,'payloads_not_scanned':['full_source_text','full_census_ledger','model_weights','assembled_full_vectors'],
        'model_forwards':0,'human_responses':0,'confirmatory_study_ready':False}
    encoded=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)
    gate.require(len(encoded.encode())+1<=ENCODED_CAP,'encoded_export_cap_exceeded')
    return encoded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--control-package-sha256',required=True)
    args=parser.parse_args()
    print(export(args.job_id,args.control_package_sha256))


if __name__=='__main__':main()
