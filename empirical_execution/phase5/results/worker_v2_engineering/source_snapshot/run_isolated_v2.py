"""Version 2: residual-gated separately constructed seccomp FD-only services.

This is a development integration. Shared graph/input preparation remains a
separate charged stage. It does not waive the semantic/human study gates.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hashlib,json,os,resource,shutil,signal,subprocess,time
import numpy as np
from phase3.reference_graph import build_reference_graph
from phase5.methods import METHODS

DEFAULT_POLICY={'memory_bytes':1024**3,'cpu_seconds':60,'wall_seconds':60,'threads':1,
                'policy_role':'prospective_development_envelope_not_primary_resource_lock'}

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,obj):Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')

def prepare_bundle(curator,learner,targets,record_ids,destination,*,threshold,source_ids=None,priority_seed=0):
    dst=Path(destination).resolve();dst.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    cx=np.asarray(curator);x=np.asarray(learner);y=np.asarray(targets)
    if y.ndim==1:y=y[:,None]
    ids=list(record_ids);sources=ids if source_ids is None else list(source_ids)
    if cx.dtype!=np.float32 or x.dtype!=np.float32 or y.dtype!=np.float64 or len(x)!=len(ids) or len(y)!=len(ids):raise ValueError('Aligned FP32 features and FP64 targets required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite arrays required')
    graph=build_reference_graph(cx,ids,threshold,sources,seed=priority_seed)
    graph_seconds=time.perf_counter()-start
    np.save(dst/'curator.npy',cx,allow_pickle=False);np.save(dst/'learner.npy',x,allow_pickle=False);np.save(dst/'targets.npy',y,allow_pickle=False)
    np.savez(dst/'graph.npz',priority=graph.priority_indices,indptr=graph.indptr,indices=graph.indices)
    write(dst/'metadata.json',{'record_ids':ids,'source_ids':sources,'threshold':threshold,'priority_seed':priority_seed,
                             'evidence_role':'development_engineering_only','graph_seconds':graph_seconds})
    files={p.name:{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in dst.iterdir() if p.is_file()}
    write(dst/'bundle.json',{'files':files,'shared_graph_build_seconds':graph_seconds,'total_prepare_seconds':time.perf_counter()-start,
                           'shared_input_bytes':sum(p['bytes'] for p in files.values())})
    return dst

def child(stage,root,policy,config=None):
    worker=Path(__file__).with_name('workers_v2.py').resolve();argv=[sys.executable,str(worker),stage,'--directory',str(root)]
    if config is not None:argv.extend(['--config',str(config)])
    env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1',
         'OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE'}
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):env[k]=str(policy['threads'])
    started=time.perf_counter();timed_out=False
    with (root/f'{stage}_stdout.txt').open('wb') as out,(root/f'{stage}_stderr.txt').open('wb') as err:
        proc=subprocess.Popen(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True,start_new_session=True)
        while True:
            pid,status,usage=os.wait4(proc.pid,os.WNOHANG)
            if pid:break
            if time.perf_counter()-started>policy['wall_seconds']:
                timed_out=True;os.killpg(proc.pid,signal.SIGKILL);pid,status,usage=os.wait4(proc.pid,0);break
            time.sleep(.02)
        proc.returncode=os.waitstatus_to_exitcode(status)
    return {'exit_code':proc.returncode,'timeout':timed_out,'spawn_to_reap_seconds':time.perf_counter()-started,
            'peak_rss_bytes':usage.ru_maxrss*1024,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,
            'read_block_operations':usage.ru_inblock,'write_block_operations':usage.ru_oublock,
            'block_operations_are_not_logical_byte_counts':True,'process_only_rss_not_deployment_memory':True}

def run_service(bundle,method,requests,destination,*,horizon,unit='record',lambda_reg=.01,policy=None,
                persistence='final',compaction_fraction=.5,batch_rows=1024,solver='cholesky'):
    """Construct then exec repair. Requests are delta string-ID lists per release."""
    started=time.perf_counter();bundle=Path(bundle).resolve();dest=Path(destination).resolve()
    if solver not in ('cholesky','cg'):raise ValueError('Unknown locked common decoder')
    if method not in METHODS or unit not in ('record','source') or persistence not in ('final','every_release'):raise ValueError('Invalid locked service configuration')
    policy=dict(DEFAULT_POLICY if policy is None else policy)
    if type(policy.get('threads'))is not int or policy['threads']<1 or policy.get('wall_seconds',0)<=0:raise ValueError('Invalid common resource policy')
    if not isinstance(requests,list) or any(not isinstance(b,list) or any(not isinstance(i,str) for i in b) for b in requests):raise ValueError('Delta string-ID request batches required')
    dest.mkdir(parents=True,exist_ok=False);construction=dest/'construction';private=dest/'repair';construction.mkdir();private.mkdir()
    expected=json.loads((bundle/'bundle.json').read_text())['files'];used=('metadata.json','graph.npz','learner.npy','targets.npy')+ (('curator.npy',) if method=='O-G' else ())
    conf={'solver':solver,'method':method,'unit':unit,'horizon':horizon,'lambda_reg':lambda_reg,'input_bundle':str(bundle),
          'input_hashes':{n:expected[n]['sha256'] for n in used},'policy':policy,'compaction_fraction':compaction_fraction,'batch_rows':batch_rows}
    config=dest/'construction.json';write(config,conf)
    code_hashes={str(p.relative_to(ROOT)):file_hash(p) for folder in ('ccu','phase3','phase4','phase5') for p in sorted((ROOT/folder).glob('*.py'))}
    report={'schema':'ccu-isolated-service-2','worker_version':2,'solver':solver,'decoder_schema':'ccu-common-ridge-decoder-1','method':method,'unit':unit,'policy':policy,'persistence':persistence,'construction':None,'repair':None,
            'project_python_source_hashes':code_hashes,'dependency_binary_complete_pinning':False,'paper_systems_result':False,
            'raw_input_files_sha256':conf['input_hashes'],'request_sha256':hashlib.sha256(json.dumps(requests,separators=(',',':')).encode()).hexdigest()}
    write(dest/'prospective_configuration.json',{**conf,'requests':requests,'persistence':persistence,'code_hashes':code_hashes})
    report['construction']=child('construct',construction,policy,config)
    if report['construction']['exit_code']==0:
        stage=time.perf_counter();states=list(construction.glob('state.npz*'))
        for p in states:shutil.copyfile(p,private/p.name)
        # An existing plain sentinel proves forbidden reads; it contains no corpus
        # data and its path is the sole denied probe passed into repair.
        probe=dest/'forbidden_probe.txt';probe.write_text('no-reaccess-test-sentinel\n')
        repair_conf={'solver':solver,'method':method,'unit':unit,'lambda_reg':lambda_reg,'requests':requests,'policy':policy,'persistence':persistence,
                     'state_hashes':{p.name:file_hash(p) for p in states},'denied_probe_paths':[str(probe)]}
        write(private/'repair.json',repair_conf);report['state_transfer_seconds']=time.perf_counter()-stage
        report['repair']=child('repair',private,policy)
    report['total_lifecycle_seconds_before_final_report']=time.perf_counter()-started
    report['success']=report['construction']['exit_code']==0 and report['repair'] is not None and report['repair']['exit_code']==0
    report['artifact_bytes']=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
    report['artifact_hashes']={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()}
    write(dest/'service_report.json',report)
    # Root measurement bookkeeping and final report are included in lifecycle.
    for p in dest.rglob('*'):
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    receipt={'charged_total_seconds':time.perf_counter()-started,'service_report_sha256':file_hash(dest/'service_report.json'),
             'receipt_self_write_excluded':True,'OS_page_cache_flush_performed':False}
    write(dest/'persistence_receipt.json',receipt)
    return report
