"""Preserved family kernels with new v3 call identities. Library only; no production CLI.

No activation is created here. The future reviewed controller must admit the
request bundle, complete activation contract and stage budget before calling.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import resource
import sys
import os
import platform
import importlib.metadata
import time
from ledger import ArtifactIntegrityError
from contract import (Activation,binding,canonical,checked_file,digest,envelope,import_exact,
                      load_pins,require,sha,strict_json,validate_envelope,validate_planned_call)

def source_path(repo,protocol_package,logical):
    prefix='local_run/model_annotation_v4_20261014/'
    return Path(protocol_package)/logical[len(prefix):] if logical.startswith(prefix) else Path(repo)/logical

def runtime_preflight(pins):
    require(platform.system()=='Linux' and platform.python_version()==pins['runtime']['python'],'qualified_linux_python')
    job=os.environ.get('SLURM_JOB_ID','')
    require(job.isascii() and job.isdigit() and int(job)>0,'positive_slurm_job_required')
    for name,version in pins['runtime']['package_versions'].items():
        require(importlib.metadata.version(name)==version,'qualified_package_version:'+name)
    return True

class Watch:
    def __init__(self):self.files={};self.inventory={}
    def add(self,path,expected=None,allow_asset_symlink=False):
        p=Path(path).absolute();require(p.is_file(),'missing_bound_file')
        require(allow_asset_symlink or not p.is_symlink(),'metadata_symlink')
        b=binding(p);require(expected is None or b==expected,'initial_binding_mismatch:'+str(p))
        self.files[str(p)]={'binding':b,'resolved_path':str(p.resolve())};return b
    def snapshot(self,folder,names):
        p=Path(folder);require(not p.is_symlink(),'snapshot_root_symlink')
        actual={str(x.relative_to(p)) for x in p.rglob('*') if x.is_file()}
        require(actual==set(names),'snapshot_inventory');self.inventory[str(p)]=sorted(actual)
    def recheck(self):
        for name,v in self.files.items():
            p=Path(name);require(str(p.resolve())==v['resolved_path'] and binding(p)==v['binding'],'final_binding_changed:'+name)
        for folder,names in self.inventory.items():self.snapshot(folder,names)
        return True

class FamilyKernel:
    """One family per fresh process. Loading verifies every weight, not just metadata."""
    def __init__(self,repo,alias,protocol_package):
        self.repo=Path(repo).resolve();self.protocol_package=Path(protocol_package).resolve();self.pins=load_pins();self.alias=alias;self.model=self.pins['models'][alias]
        self.watch=Watch();self.watch.add(Path(__file__));self.watch.add(Path(__file__).with_name('contract.py'))
        self.watch.add(Path(__file__).with_name('expectations.json'))
        for name,pin in self.pins['source_files'].items():self.watch.add(source_path(self.repo,self.protocol_package,name),pin)
        self.watch.add(self.protocol_package/'package_manifest.json',self.pins['protocol_package'])
        protocol_manifest=strict_json((self.protocol_package/'package_manifest.json').read_bytes())
        for name,pin in protocol_manifest['files'].items():
            require(Path(name).name==name,'protocol_member_path')
            self.watch.add(self.protocol_package/name,pin)
        rel=self.model['pilot_code_relative'];self.pilot=import_exact('third_inference_qualified_'+alias,self.repo/rel,self.pins['source_files'][rel])
        protocol_rel='local_run/model_annotation_v4_20261014/protocol.py'
        self.protocol=import_exact('third_inference_protocol_'+alias,self.protocol_package/'protocol.py',self.pins['source_files'][protocol_rel])
        self.schema=strict_json((self.repo/'local_run/model_annotation_v2_20261011/model_judgment.schema.json').read_bytes())
        self.state=None;self.guard=None
    def bind_guard(self):
        relative=str(Path(self.model['pilot_code_relative']).parent/'bounded_reason.py');path=self.repo/relative
        if 'bounded_reason' in sys.modules:
            m=sys.modules['bounded_reason'];require(Path(m.__file__).resolve()==path.resolve(),'foreign_bounded_reason_module')
            require(binding(path)==self.pins['source_files'][relative],'guard_hash')
        else:m=import_exact('bounded_reason',path,self.pins['source_files'][relative])
        self.guard=m
    def load(self,activation,model_path,weights_manifest,out,runtime_dependency_source=None):
        require(isinstance(activation,Activation),'validated_activation_required')
        runtime_preflight(self.pins)
        package=Path(__file__).resolve().parent
        self.watch.add(package/'package_manifest.json',activation.contract['kernel_package'])
        for name,pin in strict_json((package/'package_manifest.json').read_bytes())['files'].items():
            require(Path(name).name==name,'kernel_member_path')
            self.watch.add(package/name,pin)
        require(self.state is None,'one_model_load_per_component')
        out=Path(out);require(out.is_dir() and not list(out.iterdir()),'new_empty_runtime_directory')
        model_path=Path(model_path).absolute();require(model_path.name==self.model['revision'],'model_revision_path')
        self.watch.add(weights_manifest,self.model['asset_manifest'])
        weights=strict_json(Path(weights_manifest).read_bytes())
        require(weights.get('model_id')==self.model['model_id'] and weights.get('revision',weights.get('model_revision'))==self.model['revision'],'asset_model_identity')
        declared={x['relative_path']:{'bytes':x['size_bytes'],'sha256':x['sha256']} for x in weights['files']}
        require(len(declared)==len(weights['files']) and declared==self.model['asset_files'],'asset_manifest_inventory')
        self.watch.snapshot(model_path,declared)
        # Complete initial raw hashing and resolved HF blob-target binding.
        for n,pin in declared.items():self.watch.add(model_path/n,pin,allow_asset_symlink=True)
        self.pilot.verify_weights(weights,model_path)
        self.bind_guard()
        runtime_rel=str(Path(self.model['pilot_code_relative']).parent/'runtime_plan.json')
        state={'runtime_plan':strict_json((self.repo/runtime_rel).read_bytes()),'manifest':{
            'model_path':str(model_path),'runtime_plan_sha256':self.pins['source_files'][runtime_rel]['sha256'],
            'runtime_dependency_source':runtime_dependency_source},'bound_files':{},'judgment_schema':self.schema,
            'system_prompt':(self.repo/'local_run/model_annotation_v2_20261011/judge_system_prompt.txt').read_text(),
            'task_definition':(self.repo/'local_run/model_annotation_v2_20261011/judge_task_definition.txt').read_text()}
        # The inherited loader receipt retains its original engineering schema;
        # it attests this component, never a legacy scientific admission.
        self.pilot.load_model(state,out)
        rt=state['runtime']
        for key,value in self.model['tokenizer_identity'].items():require(rt[key]==value,'qualified_tokenizer_identity:'+key)
        for path,h in state['bound_files'].items():self.watch.add(path,{'bytes':Path(path).stat().st_size,'sha256':h})
        self.state=state;self.activation=activation
        return rt
    def verify_tokenization(self,row,token):
        require(self.state is not None,'model_not_loaded')
        prepared=self.pilot.tokenize_call(self.state,row);encoded=prepared['encoded']
        require(set(encoded)=={'input_ids','attention_mask'},'encoded_fields')
        ids=encoded['input_ids'][0].tolist();mask=encoded['attention_mask'][0].tolist()
        require(len(encoded['input_ids'].shape)==2 and encoded['input_ids'].shape[0]==1,'batch_one')
        require(ids==token['input_token_ids'] and mask==[1]*len(ids),'prepared_tokens_changed')
        require(prepared['input_tokens']==token['input_tokens'] and prepared['rendered_prompt_sha256']==token['rendered_prompt_sha256'],'prepared_prompt_changed')
        require(self.pilot.digest_bytes(self.pilot.canonical(row))==token['input_row_sha256'],'prepared_text_changed')
        return prepared
    def recheck(self):return self.watch.recheck()


def execute_call(kernel,call,token,row,writer):
    """One bounded attempt; writer is a private no-clobber per-call artifact writer.

    Identity/token integrity errors propagate before generation. Model availability
    or strict-format failures become explicit outcomes, with original denominator.
    There is no retry, refusal classifier, score input, or other-model answer input.
    """
    validate_planned_call(call,kernel.protocol)
    state=kernel.state;pilot=kernel.pilot
    require(state is not None and kernel.guard is not None,'loaded_family_required')
    require(call['model_id']==kernel.model['model_id'] and call['model_revision']==kernel.model['revision'],'call_family')
    require(call['activation_lock_sha256']==kernel.activation.raw_binding['sha256'],'call_activation')
    require(call['request_row_sha256']==sha(canonical(token)) and call['request_index']==token['request_index'] and call['model_seed']==token['model_seed'],'call_request_binding')
    prepared=kernel.verify_tokenization(row,token)
    torch=state['torch'];input_ids=prepared['encoded']['input_ids'][0].tolist()
    metrics={'call_id':call['call_id'],'request_index':call['request_index'],'attempt':0,
        'model_seed':call['model_seed'],'input_tokens':len(input_ids),'output_tokens':0,
        'full_input_tokenization':True,'truncation_applied':False,'format_only_retry_limit':0,
        'effective_constraint_config_sha256':state['effective_constraint_config_sha256'],
        'token_eligibility_sha256':state['token_eligibility_sha256'],'rendered_prompt_sha256':prepared['rendered_prompt_sha256'],
        'logit_optimality_independently_replayed':False,'human_completed':False}
    status='context_overflow';judgment=raw_hash=error=None;guard=None;started=time.perf_counter();inference_start=None;gpu_started=False
    try:
        if len(input_ids)+512<=32768:
            status='runtime_error';torch.manual_seed(call['model_seed']);torch.cuda.manual_seed_all(call['model_seed'])
            torch.cuda.reset_peak_memory_stats(0);gpu_started=True
            encoded={k:v.to('cuda:0') for k,v in prepared['encoded'].items()}
            _,prefix=pilot.build_constraint(kernel.schema,state['tokenizer_data'])
            guard=kernel.guard.GreedyGuardProcessor(kernel.schema,state['tokenizer_data'],prefix.token_enforcer,input_ids,state['token_policy'])
            torch.cuda.synchronize();inference_start=time.perf_counter()
            with torch.inference_mode(),state['flash_context']():
                flags=state['flash_flags']();require(flags=={'flash':True,'math':False,'mem_efficient':False,'cudnn':False},'flash_only')
                metrics['sdpa_flags_during_generation']=flags
                generated=state['model'].generate(**encoded,generation_config=state['generation'],
                    prefix_allowed_tokens_fn=prefix,logits_processor=[guard],output_attentions=False)
            torch.cuda.synchronize();metrics['generation_seconds']=time.perf_counter()-inference_start
            ids=generated[0,len(input_ids):].detach().cpu().tolist();require(len(ids)<=512,'output_token_limit')
            raw=state['tokenizer'].decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            raw_hash=sha(raw.encode('utf8'));writer.raw('raw.private.txt',raw.encode('utf8'))
            writer.json('output_tokens.private.json',ids);metrics.update(output_tokens=len(ids),reached_output_token_cap=len(ids)==512)
            try:
                require(raw==guard.state.emitted_text,'full_output_prefix_decode_mismatch')
                metrics['full_output_prefix_decode_consistency_passed']=True
                pilot.replay_tokens(input_ids,ids,kernel.schema,state['tokenizer_data'],state['token_policy'])
                metrics['generated_token_filter_replay_passed']=True
                pilot.replay_constraint(raw,kernel.schema,state['constraint_alphabet'])
                metrics['decoded_character_grammar_replay_passed']=True
                judgment=pilot.parse_judgment(raw,kernel.schema);status='valid'
            except (ValueError,TypeError) as exc:
                status='invalid_json';error={'type':type(exc).__name__,'message':str(exc) or 'strict_format_failure'}
    except ArtifactIntegrityError:
        raise
    except Exception as exc:
        status='runtime_error';judgment=None;error={'type':type(exc).__name__,'message':str(exc) or 'generation_failure'}
    finally:
        if inference_start is not None and 'generation_seconds' not in metrics:
            metrics['generation_seconds']=time.perf_counter()-inference_start
        metrics.update(status=status,total_attempt_seconds=time.perf_counter()-started,
            process_peak_rss_kib_linux=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if gpu_started:
            metrics['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated(0)
            metrics['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved(0)
        if guard is not None:
            writer.json('guard_steps.private.json',guard.steps)
            metrics.update(guard_steps=len(guard.steps),guard_candidates_considered=sum(x['candidates_considered'] for x in guard.steps),
                           guard_selection_seconds=sum(x['selection_seconds'] for x in guard.steps))
        writer.json('metrics.json',metrics)
    response=envelope(call,status,judgment,raw_hash,state['runtime_manifest_sha256'],dict(writer.files),error)
    return validate_envelope(response,call,lambda x:pilot.validate_judgment(x,kernel.schema))
