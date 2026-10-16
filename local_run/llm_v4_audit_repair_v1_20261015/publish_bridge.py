"""Publish a source-bound exact copy for frozen downstream readers, without replacing files."""
import repair_common as R
R.configure()
import argparse,datetime,json,os
import common as x
import contract as C
import launch as original
p=argparse.ArgumentParser();p.add_argument('--model-alias',choices=('qwen','mistral'),required=True);a=p.parse_args()
pins,controls,prior,h,protocol=x.setup(R.EXECUTIONS[R.SCOPE]['pin']);root=R.audit_root(R.SCOPE,a.model_alias)
terminal=original.terminal(h,root,'run/result.json','csis.cn1');source=root/'run/result.json';report=R.load(source);C.verify_seal(report)
R.check_report(report,R.SCOPE,R.PACKAGE_SHA,terminal['job_id'])
raw=source.read_bytes();dest=x.ROOT/('audit_'+a.model_alias)/'run';C.require(not os.path.lexists(dest),'occupied_compatibility_view_reconcile')
original_stage=dest.parent;history={}
if original_stage.exists():
 for f in original_stage.iterdir():
  C.require(f.is_file() and not f.is_symlink(),'existing_stage_flat_regular_files');history[f.name]=C.binding(f)
else:original_stage.mkdir(mode=0o700,exist_ok=False)
os.umask(0o077);dest.mkdir(mode=0o700,exist_ok=False)
with (dest/'result.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
bridge=x.sealed({'schema':'third-accepted-audit-repair-bridge-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':R.SCOPE,'model_alias':a.model_alias,'repair_package_sha256':R.PACKAGE_SHA,'source_result':C.binding(source),
 'source_result_path':str(source),'source_launch_receipt':C.binding(root/'launch_receipt.json'),
 'source_process_completion':C.binding(root/'process_completion.json'),'source_accounting':terminal['accounting'],
 'preserved_original_stage_files':history,'original_failed_job_relabelled':False,'model_calls_repeated':0})
x.publish(dest/'ACCEPTED_AUDIT_REPAIR.json',bridge)
C.require(all(C.binding(original_stage/name)==b for name,b in history.items()),'original_stage_files_unchanged')
print(json.dumps({'published':True,**R.verify_bridge(R.SCOPE,a.model_alias)}))
