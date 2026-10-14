#!/usr/bin/env python3
"""Export existing gate evidence as bounded JSON; never submit or retry work."""
import argparse
import base64
import json
from pathlib import Path
import sys
import gate


def export(report_dir):
    pins = gate.expectations(); report_dir = gate.safe_path(report_dir)
    gate.require(report_dir.parent == Path(pins['root']) and report_dir.name.startswith('terminal-metadata-audit-v1-a'), 'wrong_report_directory')
    gate.require(gate.directory_inventory(report_dir) == ['bundle.json','metadata_gate_result.json'], 'exact_gate_output_required')
    rows = {name: gate.read_small(report_dir/name, cap=gate.CAP) for name in ('bundle.json','metadata_gate_result.json')}
    result = gate.strict_json(base64.b64decode(rows['metadata_gate_result.json']['base64']))
    bundle = gate.strict_json(base64.b64decode(rows['bundle.json']['base64']))
    gate.require(result['bundle_file'] == {'name':'bundle.json', **{k:rows['bundle.json'][k] for k in ('bytes','sha256')}}, 'bundle_result_binding_mismatch')
    if result['passed'] is True:
        summary = gate.verify_bundle(bundle)
        gate.require(all(result.get(k) == v for k,v in summary.items()), 'passed_result_summary_mismatch')
        gate.recheck_observations(bundle)
    for name, row in rows.items(): gate.require(gate.read_small(report_dir/name, cap=gate.CAP) == row, 'gate_output_changed_during_export')
    value = {'schema':'third-e5-array-metadata-export-v1','report_dir':str(report_dir),'array_job_id':pins['array_job_id'],
             'files':rows,'passed_metadata_gate':result['passed'], 'full_population_cache_acceptance':False}
    gate.require(len(gate.canonical(value))+1 <= gate.CAP, 'export_exceeds_25MiB')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report-dir',type=Path,required=True);args=parser.parse_args()
    print(gate.canonical(export(args.report_dir)).decode())


if __name__=='__main__': main()
