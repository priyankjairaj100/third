"""Finite conditional continuation of v4 selection through independent validation."""
from fractions import Fraction
import base64
import hashlib
import importlib.util
import json
from math import comb
import os
from pathlib import Path
import time

TASK=Path(os.environ['THIRD_TASK_DIRECTORY'])
REPO=Path(os.environ['THIRD_REPOSITORY'])
WORK=TASK/'work/resume-20261014'
PARENT=WORK/'v4_pipeline_a01'
HELPER=WORK/'v4_inference/continue_pipeline.py'
HELPER_SHA='8b18f4fe98bccf53ebca31c75a4b8c1daf8ab48e41c6b626b6c5af6c9f975272'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==HELPER_SHA
spec=importlib.util.spec_from_file_location('v4_validation_controller_helpers',HELPER)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.OUT=WORK/'v4_validation_pipeline_a01'
W=p.W
SAMPLE='110744b78528532f543701df77c4397484b188003aca3a3a47c39177c111d538'
REQUEST='321cc83af9e54f8dd454e6704afc77b93f4156fbadce1aa8baf2af625b10b906'
INFER='6c294e4ff3efd52f98552d8cbe41803eb44a2c5b61ba8500880a0358b93595b8'
ROOT=W+'/qualification/llm-only-v4-validation-inference-a01'
SAMPLING=W+'/qualification/llm-only-v4-validation-sampling-a01'
OLD=W+'/qualification/llm-only-v4-inference-a01'

def seal_check(value):
    raw=json.dumps({k:v for k,v in value.items() if k!='sha256'},ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    p.require(value['sha256']==hashlib.sha256(raw).hexdigest(),'result_canonical_seal')

def independent_review():
    paths={'sample':SAMPLING+'/run/validation.private.jsonl','sampling':SAMPLING+'/run/result.json',
        'statistics':ROOT+'/statistics/run/result.json',
        'selection_statistics':OLD+'/statistics/run/result.json'}
    for alias in ('qwen','mistral'):
        paths[alias]=ROOT+'/audit_'+alias+'/run/result.json'
        paths['selection_'+alias]=OLD+'/audit_'+alias+'/run/result.json'
    script='''import base64,hashlib,json
from pathlib import Path
files={}
for name,path in PATHS.items():
 p=Path(path);assert p.is_file() and not p.is_symlink() and p.stat().st_size<2*1024**2
 raw=p.read_bytes();files[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
print(json.dumps(files))
'''.replace('PATHS',repr(paths))
    exported=p.remote('independent_validation_export',script)
    private=REPO/'local_workspace/reports/llm-v4-independent-validation-review-a01';private.mkdir(exist_ok=False)
    data={};bindings={}
    for name,value in exported.items():
        raw=base64.b64decode(value['base64']);binding={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        p.require(binding=={k:v for k,v in value.items() if k!='base64'},'export_binding')
        bindings[name]=binding
        (private/(name+'.private.json' if name!='sample' else 'sample.private.jsonl')).write_bytes(raw)
        data[name]=[json.loads(line) for line in raw.splitlines()] if name=='sample' else json.loads(raw)
        if name!='sample':seal_check(data[name])
    source=data['sampling'];report=data['statistics'];stats=report['statistics'];rows=data['sample']
    p.require(report['passed'] is True and report['package_sha256']==INFER and
        report['validation_sample']==bindings['sampling'] and source['files']['validation.private.jsonl']==bindings['sample'],
        'validation_source_bindings')
    p.require(source['selection_statistics']==bindings['selection_statistics'] and
        source['locked_tau']==data['selection_statistics']['statistics']['selected_tau']==stats['locked_tau'],
        'selection_threshold_binding')
    N=source['population'];n=len(rows);p.require(N>0 and n==min(200,N),'uniform_validation_sample_size')
    tau=Fraction(stats['locked_tau']['numerator'],stats['locked_tau']['denominator'])
    ids=[row['opaque_pair_id'] for row in rows];p.require(len(set(ids))==n,'unique_pairs')
    for row in rows:
        p.require(float.fromhex(row['score_hex'])>float(tau) and
            Fraction(**row['inclusion_weight'])==Fraction(N,n),'strict_cut_exact_weight')
    labels={};earlier={}
    for alias in ('qwen','mistral'):
        p.require(report['audits'][alias]==bindings[alias] and data[alias]['passed'] is True and
            data[alias]['package_sha256']==INFER,'validated_audit_source')
        p.require(data['selection_statistics']['audits'][alias]==bindings['selection_'+alias],'selection_audit_binding')
        labels[alias]={r['opaque_pair_id']:r for r in data[alias]['canonical_observations']}
        earlier[alias]={r['opaque_pair_id']:r for r in data['selection_'+alias]['canonical_observations']}
        p.require(set(labels[alias])==set(ids) and len(data[alias]['canonical_observations'])==n,'complete_pair_denominator')
        p.require(data[alias]['activation']!=data['selection_'+alias]['activation'],'fresh_activation')
    x=sum(all(labels[a][identity]['status']=='valid' and labels[a][identity]['category'] in
              ('exact_copy','substantially_same_meaning') for a in labels) for identity in ids)
    bound=stats['finite_population_bound'];p.require((bound['N'],bound['n'],bound['x'])==(N,n,x) and bound['defined'] is True,'bound_denominator')
    L=bound['lower_success_count'];p.require(type(L) is int and 0<=L<=N,'lower_count')
    if n==N:p.require(L==x,'census_exactness')
    elif x==0:p.require(L==0,'zero_positive_lower_bound')
    else:
        def tail(K):
            return sum(comb(K,j)*comb(N-K,n-j) for j in range(max(x,n-(N-K)),min(n,K)+1))
        denominator=comb(N,n)
        p.require(20*tail(L)>denominator and (L==0 or 20*tail(L-1)<=denominator),'exact_bound_adjacent_tail_inversion')
    p.require(Fraction(**bound['lower_proportion'])==Fraction(L,N),'reported_lower_proportion')
    overlap=set(ids)&set(earlier['qwen'])&set(earlier['mistral']);changes=[]
    for identity in sorted(overlap):
        for alias in ('qwen','mistral'):
            before,after=earlier[alias][identity],labels[alias][identity]
            if (before['status'],before['category'])!=(after['status'],after['category']):
                changes.append((identity,before['model_id']))
    reported=[(r['opaque_pair_id'],r['model_id']) for r in stats['changed_overlap_observations']]
    p.require(set(changes)==set(reported) and len(changes)==len(reported) and
        stats['overlap_pairs']==len(overlap)==source['selection_overlap_pairs'],'independent_overlap_replay')
    passed=10*L>=9*N and not changes
    p.require(stats['numeric_lower_bound_passed']==(10*L>=9*N) and
        stats['model_validation_available']==(not changes) and stats['model_validation_passed']==passed,'independent_final_decision')
    result={'passed':True,'model_validation_passed':passed,'locked_tau':stats['locked_tau'],
        'population':N,'sample_size':n,'conservative_positives':x,'lower_success_count':L,
        'overlap_pairs':len(overlap),'changed_overlap_observations':len(changes),
        'exact_bound_and_decision_independently_recomputed':True,'bindings':bindings,
        'human_annotation_calls':0,'semantic_validity_established':False,'project_complete':False}
    p.write('independent_validation_review.json',result)
    (private/'review.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result

def parent_success():
    p.event('waiting_for_selection',parent=str(PARENT))
    while not (PARENT/'completion.json').exists():
        p.require(not (PARENT/'stopped.json').exists(),'Selection controller stopped; no retry or validation submission')
        process=json.loads((WORK/'v4_inference/controller_process.json').read_text())
        try:os.kill(process['pid'],0)
        except ProcessLookupError:raise RuntimeError('Selection controller absent without terminal receipt; reconcile before any submission')
        time.sleep(60)
    completion=json.loads((PARENT/'completion.json').read_text())
    review=completion['independent_statistics_review']
    p.require(completion['selection_pipeline_complete'] is True and review['passed'] is True and
        review['all51_exact_cuts_independently_recomputed'] is True,'completed_independent_selection_review')
    p.write('selection_completion.json',completion)
    return review

def main():
    os.umask(0o077);p.OUT.mkdir(exist_ok=False)
    p.write('intent.json',{'created_utc':p.now(),'pid':os.getpid(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parent_helper_sha256':HELPER_SHA,
        'packages':{'sampling':SAMPLE,'requests':REQUEST,'inference':INFER},
        'scope':'Conditional v4 independent validation only; no retuning or downstream empirical launches',
        'automatic_retry':False,'human_annotation_calls':0})
    try:
        review=parent_success()
        if not review['selection_passed']:
            p.write('completion.json',{'completed_utc':p.now(),'validation_run':False,'reason':'Selection failed',
                'no_retuning':True,'project_complete':False});p.event('selection_failed_validation_not_launched');return
        launch=p.run_cli('sampling_launch',W+'/tools/llm-v4-validation-sampler-v1-20261014',SAMPLE,'launch.py',
            ['--package-sha256',SAMPLE,'--selection-statistics-sha256',review['bindings']['statistics']['sha256']])
        sample=p.await_success('sampling',SAMPLING,launch['job_id'],'run/result.json','csis.cn1')
        launch=p.run_cli('requests_launch',W+'/tools/llm-v4-validation-requests-v1-20261014',REQUEST,'launch.py',
            ['--package-sha256',REQUEST,'--selection-result-sha256',sample['result_binding']['sha256']])
        requests=p.await_success('requests',W+'/qualification/llm-v4-validation-requests-v1-20261014-a01',
            launch['job_id'],'run/result.json','csis.cn1')
        p.require(0<requests['result']['planned_requests']<=440,'bounded_validation_calls')
        folder=W+'/tools/llm-v4-validation-inference-v1-20261014'
        args=['--package-sha256',INFER,'--request-result-sha256',requests['result_binding']['sha256']]
        launch=p.run_cli('qualification_launch',folder,INFER,'launch.py',['--mode','qualification',*args])
        qualified=p.await_success('qualification',W+'/qualification/llm-v4-validation-inference-qualification-a01',
            launch['job_id'],'run/result.json','csis.cn1')
        activation=p.run_cli('activation',folder,INFER,'activate.py',
            args+['--qualification-result-sha256',qualified['result_binding']['sha256']])
        args+=['--activation-sha256',activation['activation']['sha256'],'--resource-plan-sha256',activation['resource_plan']['sha256']]
        for mode in ('qwen','audit_qwen','mistral','audit_mistral','statistics'):
            launch=p.run_cli(mode+'_launch',folder,INFER,'launch.py',['--mode',mode,*args]);gpu=mode in ('qwen','mistral')
            p.await_success(mode,ROOT+'/'+mode,launch['job_id'],'output/result.json' if gpu else 'run/result.json',
                'csis.cn3' if gpu else 'csis.cn1')
        result=independent_review()
        p.write('completion.json',{'completed_utc':p.now(),'validation_run':True,'independent_review':result,
            'project_complete':False,'next_stage':'Versioned model-defined effect study if validation passed; preserve failure otherwise'})
        p.event('validation_complete',model_validation_passed=result['model_validation_passed'])
    except BaseException as error:
        p.write('stopped.json',{'utc':p.now(),'error_type':type(error).__name__,'error':str(error),
            'automatic_retry':False,'reconcile_before_resume':True});p.event('stopped',error=str(error));raise

if __name__=='__main__':main()
