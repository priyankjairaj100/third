"""One finite, fail-closed v4 selection pipeline; no recurring scheduler or retries."""
from datetime import datetime,timezone
from fractions import Fraction
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK=Path(os.environ['THIRD_TASK_DIRECTORY']).resolve()
REPO=Path(os.environ['THIRD_REPOSITORY']).resolve()
WORK=TASK/'work/resume-20261014'
OUT=WORK/'v4_pipeline_a01'
REMOTE=WORK/'remote_existing_session.py'
W='/nfs_home/users/poonam/ccu-acl2027/local_workspace'
INFER='ef7449595cb5a90e8fc8a009103f958eb92a73abc7943d52c19a552e816c4207'
REQUEST='2995c12d996b362e13eb273a54807a95c3516e8a223bc34efc77dc53220050db'
ROOT=W+'/qualification/llm-only-v4-inference-a01'

def now():return datetime.now(timezone.utc).isoformat()
def require(ok,why):
    if not ok:raise RuntimeError(why)
def write(name,value):
    p=OUT/name;p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
def event(kind,**value):
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'event':kind,**value},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def remote(label,script):
    # Exactly one bounded operation; any transport or program error ends the controller.
    r=subprocess.run([sys.executable,str(REMOTE)],input=script,capture_output=True,text=True,timeout=70)
    write(label+'_transport.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':now()})
    require(r.returncode==0,'Remote operation failed; stop without retry: '+label)
    return json.loads(r.stdout)

def status(label,root,job,result_relative,index):
    script='''import datetime,hashlib,json,subprocess
from pathlib import Path
r=Path(ROOT);job=JOB
raw=subprocess.check_output(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'],text=True,timeout=10)
rows=[x.split('|') for x in raw.splitlines() if x.split('|')[0]==job]
assert len(rows)==1,'missing_or_ambiguous_job'
p=r/RESULT
value=binding=None
if p.exists():
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<4*1024**2
 data=p.read_bytes();binding={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};obj=json.loads(data)
 keys=('schema','passed','package_sha256','plan_sha256','pair_count','populations','sample_counts','selected_pairs','swap_pairs','planned_requests','score_stream_sha256','statistics','models','ordinary_fatal','planned_calls','ledger')
 value={k:obj[k] for k in keys if k in obj}
console=r/'child.stdout.txt'
if not console.exists():console=r/('slurm-'+job+'.out')
tail=None
if console.exists():
 with console.open('rb') as f:f.seek(max(0,console.stat().st_size-1800));tail=f.read().decode('utf-8',errors='replace')
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accounting':raw,'state':rows[0][1],'exit':rows[0][2],'node':rows[0][5],'result':value,'result_binding':binding,'console_tail':tail}))
'''.replace('ROOT',repr(root)).replace('JOB',repr(job)).replace('RESULT',repr(result_relative))
    return remote(label+'_status_'+str(index).zfill(4),script)

def await_success(label,root,job,result_relative,node):
    index=0;last=None
    while True:
        r=status(label,root,job,result_relative,index);index+=1
        if r['state']!=last:event('stage_state',stage=label,job=job,state=r['state']);last=r['state']
        if r['state'] in ('PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'):
            time.sleep(60);continue
        require(r['state']=='COMPLETED' and r['exit']=='0:0' and r['node']==node,'Stage failed: '+label)
        require(r['result'] is not None and r['result']['passed'] is True,'No accepted terminal result: '+label)
        write(label+'_terminal.json',r);event('stage_complete',stage=label,job=job,result=r['result_binding'])
        return r

def run_cli(label,folder,pin,script,args):
    source='''import hashlib,json,subprocess,sys
from pathlib import Path
p=Path(FOLDER);raw=(p/'package_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==PIN
for name,b in json.loads(raw)['files'].items():
 assert Path(name).name==name and name!='package_manifest.json'
 raw=(p/name).read_bytes();assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==b
command=[PYTHON,'-B',str(p/SCRIPT),*ARGS]
r=subprocess.run(command,capture_output=True,text=True,timeout=45)
print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr);raise SystemExit(r.returncode)
'''.replace('FOLDER',repr(folder)).replace('PIN',repr(pin)).replace('PYTHON',repr(W+'/envs/encoder-cpu/bin/python')).replace('SCRIPT',repr(script)).replace('ARGS',repr(args))
    r=remote(label,source);write(label+'_receipt.json',r);return r

def independent_statistics_review():
    roots={'sample':W+'/qualification/llm-only-v4-selection-a01/run/selection.private.jsonl',
        'qwen':ROOT+'/audit_qwen/run/result.json','mistral':ROOT+'/audit_mistral/run/result.json',
        'statistics':ROOT+'/statistics/run/result.json'}
    source='''import base64,hashlib,json
from pathlib import Path
files={}
for name,path in PATHS.items():
 p=Path(path);assert p.is_file() and not p.is_symlink() and p.stat().st_size<2*1024**2
 raw=p.read_bytes();files[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
print(json.dumps(files))
'''.replace('PATHS',repr(roots))
    exported=remote('final_review_export',source);decoded={}
    private=REPO/'local_workspace/reports/llm-v4-independent-selection-review-a01';private.mkdir(exist_ok=False)
    for name,record in exported.items():
        raw=base64.b64decode(record['base64']);require(len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],'export_binding')
        (private/(name+'.private.json' if name!='sample' else 'sample.private.jsonl')).write_bytes(raw)
        decoded[name]=raw
    sample=[json.loads(line) for line in decoded['sample'].splitlines()]
    audits={alias:json.loads(decoded[alias]) for alias in ('qwen','mistral')}
    statistics=json.loads(decoded['statistics'])['statistics']
    labels={alias:{r['opaque_pair_id']:r for r in report['canonical_observations']} for alias,report in audits.items()}
    chosen=None
    for cut in statistics['grid_ledger']:
        tau=Fraction(cut['tau']['numerator'],cut['tau']['denominator']);num=den=Fraction()
        for row in sample:
            if float.fromhex(row['score_hex'])<=float(tau):continue
            weight=Fraction(row['inclusion_weight']['numerator'],row['inclusion_weight']['denominator']);den+=weight
            positive=all(labels[a][row['opaque_pair_id']]['status']=='valid' and
                labels[a][row['opaque_pair_id']]['category'] in ('exact_copy','substantially_same_meaning') for a in labels)
            num+=weight*positive
        frac=lambda value:None if value is None else {'numerator':value.numerator,'denominator':value.denominator}
        require(cut['weighted_denominator']==frac(den) and cut['weighted_positive_numerator']==frac(num)
                and cut['conservative_precision']==frac(num/den if den else None),'independent_weighted_cut_mismatch')
        passed=den>0 and num/den>=Fraction(19,20)
        require(cut['passes_selection_rule'] is passed,'selection_rule_mismatch')
        if passed and chosen is None:chosen=tau
    require(len(statistics['grid_ledger'])==51 and statistics['selected_tau']==frac(chosen),'chosen_threshold_mismatch')
    review={'passed':True,'all51_exact_cuts_independently_recomputed':True,'selected_tau':frac(chosen),
        'selection_passed':chosen is not None,'human_annotation_calls':0,
        'bindings':{k:{a:b for a,b in v.items() if a!='base64'} for k,v in exported.items()}}
    write('independent_statistics_review.json',review)
    (private/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    return review

def main():
    os.umask(0o077);OUT.mkdir(exist_ok=False)
    write('intent.json',{'created_utc':now(),'pid':os.getpid(),'pipeline_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inference_package_sha256':INFER,'request_package_sha256':REQUEST,'human_annotation_calls':0,'automatic_retry':False,
        'scope':'v4 selection only; fresh validation and full research completion remain separate'})
    try:
        sampled=await_success('sampling',W+'/qualification/llm-only-v4-selection-a01','13636','run/result.json','csis.cn1')
        require(sampled['result']['pair_count']==220405510 and sampled['result']['selected_pairs']==360 and
                sampled['result']['swap_pairs']==36 and sampled['result']['score_stream_sha256']==
                '899114828d4f9d130d57f900660efc5d07f6b90e4a76c76d3eaedc7233323ea2','full_v4_sampling_contract')
        launch=run_cli('requests_launch',W+'/tools/llm-v4-requests-v1-20261014',REQUEST,'launch.py',
            ['--package-sha256',REQUEST,'--selection-result-sha256',sampled['result_binding']['sha256']])
        requests=await_success('requests',W+'/qualification/llm-v4-requests-v1-20261014-a01',launch['job_id'],'run/result.json','csis.cn1')
        require(requests['result']['planned_requests']==792,'complete_v4_requests')
        request_sha=requests['result_binding']['sha256'];folder=W+'/tools/llm-v4-inference-v1-20261014'
        base=['--package-sha256',INFER,'--request-result-sha256',request_sha]
        launch=run_cli('qualification_launch',folder,INFER,'launch.py',['--mode','qualification',*base])
        qualify=await_success('qualification',W+'/qualification/llm-v4-inference-qualification-a01',launch['job_id'],'run/result.json','csis.cn1')
        activation=run_cli('activation',folder,INFER,'activate.py',base+['--qualification-result-sha256',qualify['result_binding']['sha256']])
        base+=['--activation-sha256',activation['activation']['sha256'],'--resource-plan-sha256',activation['resource_plan']['sha256']]
        for mode in ('qwen','audit_qwen','mistral','audit_mistral','statistics'):
            launch=run_cli(mode+'_launch',folder,INFER,'launch.py',['--mode',mode,*base])
            gpu=mode in ('qwen','mistral')
            await_success(mode,ROOT+'/'+mode,launch['job_id'],'output/result.json' if gpu else 'run/result.json',
                          'csis.cn3' if gpu else 'csis.cn1')
        review=independent_statistics_review()
        write('completion.json',{'completed_utc':now(),'selection_pipeline_complete':True,'project_complete':False,
            'independent_statistics_review':review,'next_stage':'Fresh validation if selection passed; preserve negative result otherwise'})
        event('pipeline_complete',selection_passed=review['selection_passed'],selected_tau=review['selected_tau'])
    except BaseException as error:
        write('stopped.json',{'utc':now(),'error_type':type(error).__name__,'error':str(error),
            'automatic_retry':False,'reconcile_existing_receipts_before_any_resume':True})
        event('stopped',error_type=type(error).__name__,error=str(error));raise

if __name__=='__main__':main()
