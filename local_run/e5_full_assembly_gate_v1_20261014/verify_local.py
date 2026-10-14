#!/usr/bin/env python3
"""Verify and retain a remote metadata export locally; no network or submission."""
import argparse
import base64
import json
import os
from pathlib import Path
import gate


def validate(value):
    pins=gate.expectations()
    gate.require(value['schema']=='third-e5-array-metadata-export-v1' and value['array_job_id']==pins['array_job_id'], 'wrong_export_identity')
    directory=Path(value['report_dir'])
    gate.require(str(directory)==os.path.normpath(str(directory)) and directory.parent==Path(pins['root']) and directory.name.startswith('terminal-metadata-audit-v1-a'), 'wrong_export_source')
    gate.require(set(value['files'])=={'bundle.json','metadata_gate_result.json'}, 'wrong_export_inventory')
    decoded={}
    for name,item in value['files'].items():
        raw=base64.b64decode(item['base64'],validate=True)
        gate.require(len(raw)<=gate.CAP and gate.binding(raw)=={k:item[k] for k in ('bytes','sha256')}, 'export_file_binding_mismatch')
        decoded[name]=raw
    gate.require(sum(map(len,decoded.values()))<=gate.CAP, 'export_decoded_cap')
    result=gate.strict_json(decoded['metadata_gate_result.json']);bundle=gate.strict_json(decoded['bundle.json'])
    gate.require(result['schema']==gate.SCHEMA and result['array_job_id']==pins['array_job_id'] and result['root']==pins['root'], 'wrong_result_identity')
    gate.require(result['bundle_file']=={'name':'bundle.json',**gate.binding(decoded['bundle.json'])}, 'result_bundle_binding_mismatch')
    gate.require(type(result['passed']) is bool and value['passed_metadata_gate'] is result['passed'], 'wrong_result_status')
    if result['passed']:
        checked=gate.verify_bundle(bundle)
        gate.require(all(result.get(k)==v for k,v in checked.items()), 'result_summary_mismatch')
    else:
        gate.require(result.get('error_type') and result.get('error'), 'failure_diagnostic_missing')
    return decoded,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    gate.require(args.transport.stat().st_size<=gate.CAP,'transport_exceeds_25MiB');raw=args.transport.read_bytes()
    decoded,result=validate(gate.strict_json(raw))
    out=gate.safe_path(args.out);gate.require(not out.exists(),'local_output_must_not_exist');os.umask(0o077);out.mkdir(parents=True,exist_ok=False)
    for name,contents in decoded.items():
        with (out/name).open('xb') as stream:stream.write(contents);stream.flush();os.fsync(stream.fileno())
    receipt={'schema':'third-e5-array-metadata-local-verification-v1','passed_metadata_gate':result['passed'],
        'array_job_id':result['array_job_id'],'transport':gate.binding(raw),
        'metadata_gate_result':gate.binding(decoded['metadata_gate_result.json']),
        'verified_export_files':{name:gate.binding(contents) for name,contents in decoded.items()},
        'remote_stat_and_scheduler_observations_are_attested':True,'heavy_vector_or_log_hashes_verified':False,
        'requires_unchanged_slurm_assembly_validation':True,'full_population_cache_acceptance':False,'confirmatory_study_ready':False}
    gate.write_json(out/'local_verification.json',receipt)
    for name,contents in decoded.items():gate.require((out/name).read_bytes()==contents,'local_copy_mismatch')
    print(json.dumps(receipt,sort_keys=True));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
