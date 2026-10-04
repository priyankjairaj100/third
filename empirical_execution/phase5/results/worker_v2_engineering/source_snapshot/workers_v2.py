"""Version 2: residual-gated common decoding and kernel FD-only repair.

Actual service uses seccomp TSYNC denial of new file opens after own-state load.
Landlock probe remains separately callable; this runtime returned ENOSYS. This
is not memory erasure or a security proof against malicious native code.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,ctypes,errno,hashlib,io,json,math,os,platform,resource,time,zipfile
import numpy as np
from ccu.core import BlockerGraph,RidgeMoments,solve_ridge
from ccu.joint_span_summary import JointSpanRidgeSummary
from phase5.methods import METHODS,build_method,load_method
from phase5.payload import PayloadState
from phase5.decoders import decode as numerical_decode, SCHEMA as DECODER_SCHEMA
from ccu.core import RidgeSolution

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def persist_tree(root):
    """Charge complete requested fsync, returning exact logical file bytes."""
    start=time.perf_counter();n=0
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('No symlinks in service outputs')
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
            n+=p.stat().st_size
    for p in sorted([root]+[p for p in root.rglob('*') if p.is_dir()],reverse=True):
        fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    return {'fsync_seconds':time.perf_counter()-start,'current_logical_bytes':n}

def confine_files(root):
    if sys.platform!='linux' or platform.machine() not in ('x86_64','aarch64'):raise RuntimeError('Supported Linux Landlock required')
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    abi=libc.syscall(444,0,0,1)
    if abi<3:raise RuntimeError(f'Landlock ABI >=3 required; got {abi}, errno={ctypes.get_errno()}')
    # ABI3 filesystem operations through TRUNCATE; later filesystem operations
    # do not permit opening an otherwise forbidden raw-data file for reading.
    handled=(1<<15)-1
    class RuleSet(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_=1
        _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
    rules=RuleSet(handled);fd=libc.syscall(444,ctypes.byref(rules),ctypes.sizeof(rules),0)
    if fd<0:raise OSError(ctypes.get_errno(),'landlock_create_ruleset')
    parent=os.open(root,os.O_PATH|os.O_CLOEXEC)
    try:
        # No execution or special-file creation. Ordinary state reads/writes,
        # directories, rename, links and truncation only inside this private root.
        allow=handled & ~((1<<0)|(1<<6)|(1<<8)|(1<<9)|(1<<10))
        rule=PathRule(allow,parent)
        if libc.syscall(445,fd,1,ctypes.byref(rule),0)!=0:raise OSError(ctypes.get_errno(),'landlock_add_rule')
        if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_NO_NEW_PRIVS')
        if libc.syscall(446,fd,0)!=0:raise OSError(ctypes.get_errno(),'landlock_restrict_self')
    finally:os.close(parent);os.close(fd)
    return {'landlock_abi':int(abi),'filesystem_read_write_allowlist':'private_service_directory_only',
            'imports_completed_before_restriction':True,'network_or_IPC_isolation_claim':False,
            'physical_erasure_claim':False}

def limits(policy):
    memory=policy['memory_bytes'];cpu=policy['cpu_seconds']
    if type(memory)is not int or memory<64*1024**2 or type(cpu)is not int or cpu<1:raise ValueError('Invalid fixed resource policy')
    resource.setrlimit(resource.RLIMIT_AS,(memory,memory));resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

def confine_open_syscalls():
    """Kernel seccomp denies new opens, process access and communication.

    Runs only after own-state load and closure of all input descriptors. Output
    fds were opened O_WRONLY. x86-64-only ABI and x32 rejection fail closed.
    """
    if sys.platform!='linux' or platform.machine()!='x86_64':raise RuntimeError('FD-only seccomp backend requires Linux x86-64')
    class Filter(ctypes.Structure):_fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint)]
    class Program(ctypes.Structure):_fields_=[('length',ctypes.c_ushort),('filters',ctypes.POINTER(Filter))]
    # open/creat/openat/openat2/open_by_handle_at, exec, process inspection,
    # sockets, SysV IPC, fork/vfork/clone/clone3 and io_uring setup/enter/register.
    denied=[2,85,257,437,304,59,322,101,310,311,438,41,42,43,44,45,46,47,49,50,53,288,
            29,30,31,64,65,66,67,68,69,70,71,56,57,58,435,425,426,427]
    rules=[Filter(0x20,0,0,4),Filter(0x15,1,0,0xc000003e),Filter(0x06,0,0,0x80000000),
           Filter(0x20,0,0,0),Filter(0x35,0,1,0x40000000),Filter(0x06,0,0,0x00050000|errno.EACCES)]
    for nr in denied:rules.extend([Filter(0x15,0,1,nr),Filter(0x06,0,0,0x00050000|errno.EACCES)])
    rules.append(Filter(0x06,0,0,0x7fff0000));arr=(Filter*len(rules))(*rules);program=Program(len(arr),arr)
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    thread_count=len(os.listdir('/proc/self/task'))
    # TSYNC atomically applies the restriction to every pre-existing BLAS/native
    # thread. Nonzero (including a positive unsynchronizable TID) fails closed.
    if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'no_new_privs failed')
    installed=libc.syscall(317,1,1,ctypes.byref(program))
    if installed!=0:raise RuntimeError(f'seccomp TSYNC installation failed return={installed} errno={ctypes.get_errno()}')
    return {'backend':'seccomp_fd_only','kernel_denied_syscalls':denied,'all_new_file_opens_denied':True,
            'boundary':'after imports and own-method snapshot load; before first deletion',
            'only_output_file_descriptors_opened_before_restriction':True,
            'seccomp_thread_synchronization':'TSYNC','native_threads_at_restriction':thread_count,
            'physical_erasure_claim':False,'malicious_native_code_security_proof':False}

def write(path,obj):path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')

class NumericalGateError(RuntimeError):
    def __init__(self,diagnostics):
        self.diagnostics=diagnostics
        super().__init__('Numerical release refused: '+json.dumps(diagnostics,sort_keys=True,allow_nan=False))

def decode(state,lam,solver):
    result=numerical_decode(state.moments(),lam,solver=solver)
    d=result.diagnostics
    if not d['release_allowed']:raise NumericalGateError(d)
    sol=RidgeSolution(result.weights,d['absolute_residual_fro'],
                      d.get('parameter_error_from_stored_moments_diagnostic') or 0.,
                      d['absolute_residual_fro']/d['count'] if d['count'] else 0.,
                      float(lam),d['count'],solver)
    return sol,d

def construct(config_path,root):
    conf=json.loads(config_path.read_text());limits(conf['policy']);start=time.perf_counter()
    inp=Path(conf['input_bundle']);meta=json.loads((inp/'metadata.json').read_text())
    for name,expected in conf['input_hashes'].items():
        if file_hash(inp/name)!=expected:raise ValueError('Construction input hash changed')
    x=np.load(inp/'learner.npy',allow_pickle=False);y=np.load(inp/'targets.npy',allow_pickle=False)
    with np.load(inp/'graph.npz',allow_pickle=False) as z:
        graph=BlockerGraph(tuple(meta['record_ids']),tuple(meta['source_ids']),z['priority'].copy(),z['indptr'].copy(),z['indices'].copy(),meta['threshold'])
    curator=np.load(inp/'curator.npy',allow_pickle=False) if conf['method']=='O-G' else None
    loaded=time.perf_counter();state=build_method(conf['method'],graph,x,y,conf['horizon'],unit=conf['unit'],compaction_fraction=conf['compaction_fraction'],batch_rows=conf['batch_rows'],curator=curator)
    built=time.perf_counter()
    try:sol,drift=decode(state,conf['lambda_reg'],conf['solver'])
    except NumericalGateError as exc:
        write(root/'numerical_failure.json',exc.diagnostics);persist_tree(root);raise
    decoded=time.perf_counter()
    np.save(root/'initial_head.npy',sol.weights,allow_pickle=False)
    release=time.perf_counter();state.snapshot(root/'state.npz');saved=time.perf_counter()
    report={'worker_version':2,'decoder_schema':DECODER_SCHEMA,'stage':'construction','method':conf['method'],'input_load_seconds':loaded-start,'state_build_seconds':built-loaded,
            'initial_decode_seconds':decoded-built,'head_release_seconds':release-decoded,'snapshot_seconds':saved-release,
            'state_accounting':state.accounting(),'selected_count':sol.count,'ridge_residual':sol.normal_equation_residual_fro,
            'head_sha256':file_hash(root/'initial_head.npy'),'numerical_decoder':drift,'state_hashes':{p.name:file_hash(p) for p in root.glob('state.npz*')}}
    report['persistence']=persist_tree(root);write(root/'construction_report.json',report);persist_tree(root)

def repair(root):
    # Everything needed is in the private directory, including an identifiers-only
    # request list. No raw input path or original feature matrix is passed here.
    conf=json.loads((root/'repair.json').read_text());limits(conf['policy'])
    if any(not Path(p).is_file() for p in conf['denied_probe_paths']):raise ValueError('Probes must be existing real files')
    for name,h in conf['state_hashes'].items():
        if file_hash(root/name)!=h:raise ValueError('Snapshot hash changed')
    start=time.perf_counter();state=load_method(conf['method'],root/'state.npz');load_seconds=time.perf_counter()-start
    for p in root.glob('state.npz*'):p.unlink()
    for name in os.listdir('/proc/self/fd'):
        fd=int(name)
        if fd>=3:
            try:os.close(fd)
            except OSError:pass
    # No input fd survives. All further persistence uses preopened WRITE-ONLY
    # descriptors, never reopens a path. Full bytes serialization is charged.
    names=[f'head_{i:04d}.npy' for i in range(len(conf['requests']))]+['state.npz','repair_report.json','numerical_failure.json']
    if conf['method']=='P-I-jointspan':names.append('state.npz.units.json')
    handles={name:os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_CLOEXEC,0o600) for name in names}
    directory_fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
    isolation=confine_open_syscalls()
    for name in conf['denied_probe_paths']:
        try:fd=os.open(name,os.O_RDONLY)
        except PermissionError:continue
        else:os.close(fd);raise RuntimeError('Kernel no-reaccess probe unexpectedly succeeded')
    isolation['denied_existing_file_probes']=len(conf['denied_probe_paths'])
    isolation['input_file_descriptors_closed_before_repair']=True
    def emit(name,raw):
        fd=handles[name];os.lseek(fd,0,os.SEEK_SET);os.ftruncate(fd,0)
        view=memoryview(raw)
        while view:
            n=os.write(fd,view);view=view[n:]
        return hashlib.sha256(raw).hexdigest()
    def persist():
        began=time.perf_counter()
        for fd in handles.values():os.fsync(fd)
        os.fsync(directory_fd)
        return {'fsync_seconds':time.perf_counter()-began,'current_logical_bytes':sum(os.fstat(fd).st_size for fd in handles.values())}
    state_hashes={}
    def snapshot():
        began=time.perf_counter();raw=state.snapshot_bytes();parts={'state.npz':raw} if isinstance(raw,bytes) else raw
        state_hashes.clear();state_hashes.update({name:emit(name,value) for name,value in parts.items()})
        return {'snapshot_seconds':time.perf_counter()-began,'full_serialization_buffer_bytes':sum(map(len,parts.values())),'persistence':persist()}
    releases=[]
    for index,batch in enumerate(conf['requests']):
        began=time.perf_counter();op=state.delete(batch);updated=time.perf_counter()
        try:sol,drift=decode(state,conf['lambda_reg'],conf['solver'])
        except NumericalGateError as exc:
            emit('numerical_failure.json',(json.dumps(exc.diagnostics,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());persist();raise
        solved=time.perf_counter()
        buf=io.BytesIO();np.save(buf,sol.weights,allow_pickle=False);head_hash=emit(f'head_{index:04d}.npy',buf.getvalue());released=time.perf_counter()
        row={'index':index,'request_bytes':len(json.dumps(batch,separators=(',',':')).encode()),
             'repair_seconds':updated-began,'moments_and_solve_seconds':solved-updated,'head_release_seconds':released-solved,
             'count':sol.count,'ridge_residual':sol.normal_equation_residual_fro,'head_sha256':head_hash,'numerical_decoder':drift,
             'operation':{k:v for k,v in op.items() if k not in ('removed_selected_ids','admitted_ids')},
             'state_accounting':state.accounting()}
        if conf['persistence']=='every_release' or index==len(conf['requests'])-1:
            row.update(snapshot())
        releases.append(row)
    if not conf['requests']:
        empty_persist=snapshot()
    else:empty_persist=None
    report={'worker_version':2,'decoder_schema':DECODER_SCHEMA,'stage':'repair','method':conf['method'],'unit':conf['unit'],'isolation':isolation,'load_seconds':load_seconds,
            'releases':releases,'empty_request_persistence':empty_persist,'persistent_head_only_output':True,
            'selected_ID_exports':False,'state_hashes':dict(state_hashes),
            'evidence_role':'development_engineering_only','confirmatory_study_ready':False,
            'OS_page_cache_cold_claim':False,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    emit('repair_report.json',(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());persist()
    for fd in handles.values():os.close(fd)
    os.close(directory_fd)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('construct','repair'));p.add_argument('--directory',required=True);p.add_argument('--config')
    a=p.parse_args();root=Path(a.directory).resolve();os.chdir(root)
    if a.stage=='construct':construct(Path(a.config).resolve(),root)
    else:repair(root)
if __name__=='__main__':main()
