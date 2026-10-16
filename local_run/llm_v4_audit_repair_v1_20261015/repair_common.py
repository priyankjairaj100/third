"""Exact-byte audit repair and explicit compatibility publication, no model calls."""
import argparse,hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
W=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace')
REPO=W.parent/'repo'
REPAIR_ROOT=W/'qualification/llm-v4-audit-repair-v1-20261015-a01'
EXECUTIONS={
 'selection':{'folder':'llm-v4-inference-v2-20261014','pin':'adcd4704a99bb6d3eabfbd8e5409453a96faa891b00aef93e399e2a23d1034fe','root':'llm-only-v4-inference-v2-a01'},
 'validation':{'folder':'llm-v4-validation-inference-v2-20261014','pin':'3b5d8a811909d33cefd9fd6b7c5dbb0e8d17fd1facca447fa0717f1f2ce438c7','root':'llm-only-v4-validation-inference-v2-a01'}}
SCOPE=PACKAGE_SHA=None

def require(ok,why):
 if not ok:raise ValueError(why)
def binding(path):
 raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def load(path):return json.loads(Path(path).read_bytes())
def verify_package(folder,pin):
 require(binding(folder/'package_manifest.json')['sha256']==pin,'package_pin')
 for name,wanted in load(folder/'package_manifest.json')['files'].items():
  p=folder/name;require(Path(name).name==name and name!='package_manifest.json' and p.is_file() and not p.is_symlink(),'package_path')
  require(binding(p)==wanted,'package_member:'+name)
def configure():
 global SCOPE,PACKAGE_SHA
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--scope',choices=tuple(EXECUTIONS),required=True);p.add_argument('--repair-package-sha256',required=True)
 a,rest=p.parse_known_args();SCOPE=a.scope;PACKAGE_SHA=a.repair_package_sha256
 verify_package(HERE,PACKAGE_SHA);e=EXECUTIONS[SCOPE];verify_package(W/'tools'/e['folder'],e['pin'])
 sys.path.insert(0,str(W/'tools'/e['folder']));sys.argv=[sys.argv[0],*rest]
def execution_root(scope):return W/'qualification'/EXECUTIONS[scope]['root']
def audit_root(scope,alias):
 require(scope in EXECUTIONS and alias in ('qwen','mistral'),'audit_scope')
 return REPAIR_ROOT/scope/('audit_'+alias)
def read_raw_utf8(path):return Path(path).read_bytes().decode('utf-8')
def provenance():
 return {'repair_package_sha256':PACKAGE_SHA,'scope':SCOPE,'auditor_source':binding(HERE/'audit.py'),
  'original_execution_package_sha256':EXECUTIONS[SCOPE]['pin'],'audit_job_id':os.environ['SLURM_JOB_ID'],
  'raw_read_policy':'Exact UTF-8 bytes, with no universal-newline conversion','model_calls_repeated':0}
def check_report(report,scope,pin,job):
 a=report.get('audit_repair',{});require(report.get('passed') is True and report['package_sha256']==EXECUTIONS[scope]['pin'],'execution_report_identity')
 require(a.get('repair_package_sha256')==pin and a.get('scope')==scope and a.get('audit_job_id')==job and a.get('auditor_source')==binding(HERE/'audit.py'),'repaired_audit_identity')
def verify_bridge(scope,alias):
 import contract as C
 root=audit_root(scope,alias);source=root/'run/result.json';receipt=load(root/'launch_receipt.json')
 report=load(source);C.verify_seal(report);check_report(report,scope,PACKAGE_SHA,receipt['job_id'])
 dest=execution_root(scope)/('audit_'+alias)/'run';bridge=load(dest/'ACCEPTED_AUDIT_REPAIR.json');C.verify_seal(bridge)
 require(bridge['repair_package_sha256']==PACKAGE_SHA and bridge['source_result']==binding(source) and bridge['source_result_path']==str(source),'bridge_source')
 require(bridge['source_launch_receipt']==binding(root/'launch_receipt.json') and bridge['source_process_completion']==binding(root/'process_completion.json'),'bridge_lifecycle')
 require(binding(dest/'result.json')==binding(source),'bridge_exact_copy')
 return {'bridge':binding(dest/'ACCEPTED_AUDIT_REPAIR.json'),'result':binding(source),'source_root':str(root)}
