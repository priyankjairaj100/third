"""Exclusive private artifact storage and complete ordinary-failure call ledger.

A killed process is NOT declared complete: an independent auditor must reconcile
missing completions against planned_calls.json. No attempted call is restarted.
"""
from __future__ import annotations
from collections import Counter
import os
from pathlib import Path
from contract import binding,canonical,envelope,require,validate_envelope

class ArtifactIntegrityError(RuntimeError):pass

def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

class PrivateOutput:
    def __init__(self,path,limit):
        require(type(limit) is int and limit>0,'positive_output_limit')
        self.root=Path(path);self.root.mkdir(mode=0o700,parents=False,exist_ok=False)
        self.root.chmod(0o700);fsync_dir(self.root.parent)
        self.limit=limit;self.used=0;self.files={}
    def write(self,path,raw):
        path=Path(path)
        try:
            require(path.parent.is_dir() and not path.parent.is_symlink(),'output_parent')
            require(path.is_relative_to(self.root) and path.resolve().is_relative_to(self.root.resolve()) and
                    '..' not in path.relative_to(self.root).parts and path.name not in ('','.','..'),'output_path')
            require(type(raw) is bytes and self.used+len(raw)<=self.limit,'output_budget_exceeded')
            # Exclusive creation preserves partial bytes if interrupted; never replace.
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            fsync_dir(path.parent);self.used+=len(raw);b=binding(path)
            self.files[str(path.relative_to(self.root))]=b;return b
        except Exception as exc:raise ArtifactIntegrityError(str(exc)) from exc
    def json(self,name,value):return self.write(self.root/name,canonical(value)+b'\n')
    def recheck(self):
        actual={str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file()}
        require(actual==set(self.files),'unexpected_or_missing_output_file')
        require(all(not p.is_symlink() for p in self.root.rglob('*')),'output_symlink')
        require(all(binding(self.root/n)==b for n,b in self.files.items()),'output_changed')
        require(sum(b['bytes'] for b in self.files.values())==self.used<=self.limit,'output_accounting')
        return True

class CallWriter:
    def __init__(self,store,call):
        self.store=store;self.call=call;self.path=store.root/'calls'/call['call_id'];self.files={}
        self.path.mkdir(mode=0o700,parents=False,exist_ok=False);fsync_dir(self.path.parent)
        store.write(self.path/'intent.json',canonical({'schema':'third-llm-call-intent-v4',
            'planned_call':call,'attempt':0,'retry_allowed':False})+b'\n')
    def raw(self,name,raw):
        require(Path(name).name==name and name not in ('intent.json','completion.json'),'artifact_name')
        b=self.store.write(self.path/name,raw);self.files[name]=b;return b
    def json(self,name,value):return self.raw(name,canonical(value)+b'\n')
    def complete(self,response):
        require(response['artifacts']==self.files,'completion_artifact_inventory')
        return self.store.write(self.path/'completion.json',canonical(response)+b'\n')

class Ledger:
    def __init__(self,path,limit,calls,judgment_validator):
        require(calls and len({c['call_id'] for c in calls})==len(calls),'unique_complete_planned_calls')
        require(len({c['request_index'] for c in calls})==len(calls),'unique_request_indices_per_process')
        require(len({c['model_id'] for c in calls})==1,'one_model_per_process')
        require({c['stage'] for c in calls} <= {'selection','selection_swap'},'one_selection_phase_per_process')
        require(all((c['stage'],c['orientation']) in {('selection','canonical'),('selection_swap','swapped_diagnostic')}
                    for c in calls),'stage_orientation')
        self.calls=list(calls);self.by_id={c['call_id']:c for c in calls};self.responses={};self.started=set()
        self.validate=judgment_validator;self.store=PrivateOutput(path,limit)
        self.store.json('planned_calls.json',calls);(self.store.root/'calls').mkdir(mode=0o700);fsync_dir(self.store.root)
    def begin(self,call):
        require(call==self.by_id.get(call['call_id']),'unplanned_call')
        require(call['call_id'] not in self.started and call['call_id'] not in self.responses,'at_most_one_attempt')
        writer=CallWriter(self.store,call);self.started.add(call['call_id']);return writer
    def record(self,writer,response):
        call=self.by_id[response['call_id']];validate_envelope(response,call,self.validate)
        require(writer.call==call and call['call_id'] in self.started and call['call_id'] not in self.responses,'completion_unique_intent')
        writer.complete(response);self.responses[call['call_id']]=response
    def finish(self,ordinary_fatal=None,runtime_hash=None):
        # A durable intent without completion is missing output, never not_attempted.
        for call in self.calls:
            cid=call['call_id']
            if cid not in self.responses:
                require(ordinary_fatal is not None,'incomplete_ledger_without_failure')
                status='missing_output' if cid in self.started else 'not_attempted'
                r=envelope(call,status,runtime_hash=runtime_hash,error={'type':'OrdinaryProcessFailure','message':str(ordinary_fatal)})
                validate_envelope(r,call,self.validate);self.responses[cid]=r
        ordered=[self.responses[c['call_id']] for c in self.calls]
        self.store.json('responses.json',ordered)
        summary={'schema':'third-llm-inference-ledger-v4','planned_calls':len(self.calls),'recorded_calls':len(ordered),
            'status_counts':dict(Counter(r['status'] for r in ordered)),'ordinary_fatal':ordinary_fatal,
            'complete_planned_denominator':True,'artifact_integrity_independently_verified':False,
            'human_annotation_calls':0,'retry_calls':0,'killed_process_completion_inferred':False}
        self.store.json('ledger.json',summary);self.store.recheck();return ordered,summary
