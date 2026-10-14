"""Fixed v4 paths and authenticated libraries; no submission or model loading."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import contract as C

W=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace')
REPO=W.parent/'repo'
HERE=Path(__file__).resolve().parent
ROOT=W/'qualification/llm-only-v4-inference-a01'
QUAL=W/'qualification/llm-v4-inference-qualification-a01'
PROTOCOL=W/'tools/model-annotation-v4-20261014'
REQUEST_PACKAGE=W/'tools/llm-v4-requests-v1-20261014'
REQUESTS=W/'qualification/llm-v4-requests-v1-20261014-a01/run'
OLD_REQUESTS=W/'tools/llm-annotation-requests-v2-20261014'
OLD_CONTROL=W/'tools/llm-annotation-request-control-v2-20261014'
CONTROL_SHA='d20c325301a2a0564f65f78c25fa3f9347bb560ec0e0ef5dd952bd5221a79704'
OLD_REQUEST_SHA='bf70b015f39ad0421d8bf66f6b357123f24aceaf9f99b5158fb9b51f490e92d7'
SELECTION=W/'qualification/llm-only-v4-selection-a01'

def bootstrap(folder,pin):
    C.require(C.binding(folder/'package_manifest.json')['sha256']==pin,'package_pin')
    manifest=C.strict_json((folder/'package_manifest.json').read_bytes())
    for name,b in manifest['files'].items():
        C.require(Path(name).name==name and name!='package_manifest.json','safe_package_member')
        C.checked_file(folder/name,b,1<<20)
    return manifest

def module(name,path):
    C.require(name not in sys.modules,'module_already_imported')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def setup(pin):
    bootstrap(HERE,pin);pins=C.load_pins()
    bootstrap(PROTOCOL,pins['protocol_package']['sha256'])
    bootstrap(REQUEST_PACKAGE,pins['request_preparation_package']['sha256'])
    bootstrap(OLD_CONTROL,CONTROL_SHA);bootstrap(OLD_REQUESTS,OLD_REQUEST_SHA)
    controls=module('v4_inference_path_helpers',OLD_CONTROL/'control.py')
    prior,h=controls.helpers()
    protocol=module('v4_inference_protocol',PROTOCOL/'protocol.py')
    protocol.validate(json.loads((PROTOCOL/'PLAN.json').read_bytes()),PROTOCOL)
    return pins,controls,prior,h,protocol

def request_contract(request_sha,pins):
    rb=C.binding(REQUESTS/'result.json');C.require(rb['sha256']==request_sha,'request_result_pin')
    result=C.strict_json((REQUESTS/'result.json').read_bytes());C.verify_seal(result)
    C.require(result['schema']=='third-llm-v4-requests-v1' and result['passed'] is True,'request_pass')
    C.require(result['plan_sha256']==pins['protocol_plan_sha256'] and
              result['package_sha256']==pins['request_preparation_package']['sha256'],'request_plan_package')
    C.checked_file(SELECTION/'run/result.json',result['selection_result'])
    C.checked_file(SELECTION/'sampling_lock.json',result['sampling_lock'])
    return {'request_manifest':rb,'request_preparation_package':pins['request_preparation_package'],
        'selection_result':result['selection_result'],'sampling_lock':result['sampling_lock'],
        'protocol_plan_sha256':pins['protocol_plan_sha256'],'frame_result_sha256':pins['frame_result_sha256'],
        'protocol_evidence_sha256':pins['protocol_evidence_sha256']},result

def publish(path,value):
    raw=C.canonical(value)+b'\n';C.require(len(raw)<=4<<20,'metadata_cap')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return C.binding(path)

def sealed(value):
    C.require('sha256' not in value,'already_sealed')
    return dict(value,sha256=C.sha(C.canonical(value)))
