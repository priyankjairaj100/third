#!/usr/bin/env python3
"""Metadata/software inspection only. Deliberately no inference/run subcommand."""
import argparse
import json
from pathlib import Path
from contract import checked_file,load_pins,strict_json
from kernel import source_path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--protocol-package',type=Path,required=True);a=p.parse_args()
    pins=load_pins()
    for name,b in pins['source_files'].items():checked_file(source_path(a.repo,a.protocol_package,name),b)
    checked_file(a.protocol_package/'package_manifest.json',pins['protocol_package'])
    for name,b in strict_json((a.protocol_package/'package_manifest.json').read_bytes())['files'].items():
        checked_file(a.protocol_package/name,b)
    print(json.dumps({'schema':'third-llm-inference-components-status-v1','source_pins_verified':True,
        'production_execution_enabled':False,'model_calls':0,'entropy_draws':0,'activation_created':False,
        'remaining':['independent request artifact audit','exclusive activation writer and reviewed receipt',
        'complete runtime controller and bounded Slurm resources','independent output/token/grammar auditor',
        'joint statistics producer-consumer tests','new actual-tokenizer and GPU integration qualification']}))
if __name__=='__main__':main()
