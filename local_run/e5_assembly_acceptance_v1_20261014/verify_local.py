#!/usr/bin/env python3
"""Recheck and retain a successful assembly metadata export; never read payloads."""
import argparse
import json
import os
from pathlib import Path
import read_remote


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gate-package','transport','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--assembly-launch-sha256',required=True);args=parser.parse_args()
    gate=read_remote.load_gate(args.gate_package);gate.require(args.transport.stat().st_size<=gate.CAP,'transport_exceeds_25MiB')
    raw=args.transport.read_bytes();value=gate.strict_json(raw);summary=read_remote.verify(value,gate,args.assembly_launch_sha256)
    gate.require(value['verification']==summary,'export_verification_summary_mismatch')
    out=gate.safe_path(args.out);gate.require(not out.exists(),'new_local_evidence_directory_required');os.umask(0o077);out.mkdir(parents=True,exist_ok=False)
    with (out/'terminal_export.json').open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    report={**summary,'transport':gate.binding(raw),'remote_scheduler_and_stat_observations_are_attested':True,
        'independent_payload_verification_route':'If needed, run a separately reviewed read-only compute-node verifier of actual cache hashes, frozen source/census bindings and exact shard slices; no model forward or duplicate assembly required.'}
    gate.write_json(out/'local_verification.json',report)
    gate.require((out/'terminal_export.json').read_bytes()==raw,'local_copy_mismatch');print(json.dumps(report,sort_keys=True))


if __name__=='__main__':main()
