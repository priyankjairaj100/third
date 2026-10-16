"""Explicit, minimal source transformations from pinned v2 execution packages."""
def once(source,old,new):
 if source.count(old)!=1:raise ValueError('source_patch_not_unique:'+old[:100])
 return source.replace(old,new)
def preamble(source):
 end=source.index('\n')+1
 return source[:end]+'import repair_common as R\nR.configure()\n'+source[end:]
def audit(source):
 source=preamble(source)
 source=once(source,"raw=(callpath/'raw.private.txt').read_text()","raw=R.read_raw_utf8(callpath/'raw.private.txt')")
 source=once(source,"out=x.ROOT/('audit_'+args.model_alias)/'run'","out=R.audit_root(R.SCOPE,args.model_alias)/'run'")
 source=once(source,"'execution_result':C.binding(folder/'result.json')","'audit_repair':R.provenance(),'execution_result':C.binding(folder/'result.json')")
 return source
def launch(source):
 source=source.replace("third-v4-pool-val-","third-v4-byte-repair-").replace("third-v4-pool-","third-v4-byte-repair-")
 source=source.replace("third-v4-validation-stage-launch","third-v4-byte-repair-continuation-launch").replace("third-v4-stage-launch","third-v4-byte-repair-continuation-launch")
 source=preamble(source)
 source=once(source,"MODES=('qualification','qwen','audit_qwen','mistral','audit_mistral','statistics')","MODES=('mistral','statistics')")
 source=once(source,"root=root_for(args.mode);C.require", "C.require(args.package_sha256==R.EXECUTIONS[R.SCOPE]['pin'],'execution_package_pin')\n    accepted_audits={alias:R.verify_bridge(R.SCOPE,alias) for alias in (('qwen',) if args.mode=='mistral' else ('qwen','mistral'))}\n    root=root_for(args.mode);C.require")
 source=source.replace("('audit_qwen',x.ROOT/'audit_qwen','run/result.json','csis.cn1')","('audit_qwen',R.audit_root(R.SCOPE,'qwen'),'run/result.json','csis.cn1')")
 source=once(source,"('audit_mistral',x.ROOT/'audit_mistral','run/result.json','csis.cn1')","('audit_mistral',R.audit_root(R.SCOPE,'mistral'),'run/result.json','csis.cn1')")
 source=once(source,"'mode':args.mode,'root':str(root)","'repair_package_sha256':R.PACKAGE_SHA,'accepted_audit_repairs':accepted_audits,'mode':args.mode,'root':str(root)")
 return source
