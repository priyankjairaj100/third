"""Source-bound isolated development measurements for graph and exact verification.

This producer runs no optimizer. Reports do not authenticate provenance or grant
primary-study acceptance. A software-only fixture can never qualify as actual
disjoint development. Frozen phase modules are imported without modification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase7 import resource_preflight as pf

REQUEST_SCHEMA = 'ccu-resource-measurement-request-1'
CONFIG_SCHEMA = 'ccu-resource-measurement-configuration-1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
        allow_nan=False) + '\n')


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for folder in ('ccu','phase3','phase4','phase5','phase6')
            for p in sorted((ROOT / folder).glob('*.py'))}


def producer_sources():
    return {name: sha(ROOT / name) for name in ('phase9/resource_measurement.py',
                                               'phase7/resource_preflight.py')}


def contained(path, base):
    resolved = (Path(base) / path).resolve()
    if not resolved.is_relative_to(Path(base).resolve()):
        raise ValueError('Input path escapes request directory')
    return resolved


def package_hashes(package):
    manifest = pf.loads((package / 'bundle.json').read_text())
    if not isinstance(manifest.get('files'), dict):
        raise ValueError('Complete derived input package manifest required')
    required = {'metadata.json','graph.npz','curator.npy'}
    if not required <= set(manifest['files']):
        raise ValueError('Derived package omits graph inputs')
    result = {}
    for name, row in manifest['files'].items():
        if Path(name).name != name or name == 'bundle.json':
            raise ValueError('Only flat named package inputs are supported')
        path = package / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Input must be a regular nonsymlink file')
        actual = sha(path)
        if actual != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Derived package bytes changed: ' + name)
        result[name] = actual
    return result


def validate_request(request):
    common = {'schema','kind','profile_signature','input_package','policy','evidence_role'}
    kind = request.get('kind')
    extra = {'trajectory','source_kinds','fresh_graph'} if kind == 'graph' else {'candidate','selection'}
    if kind not in {'graph','convex'} or set(request) != common | extra or request['schema'] != REQUEST_SCHEMA:
        raise ValueError('Exact versioned graph or convex request fields required')
    if request['evidence_role'] not in {'actual_disjoint_development','software_only'}:
        raise ValueError('Explicit actual-development or software-only role required')
    pf.validate_policy(request['policy'])
    s = request['profile_signature']
    if s.get('kind') != kind or s.get('threads') != request['policy']['threads']:
        raise ValueError('Signature kind/threads differ from the measured execution')
    if kind == 'graph':
        p = request['trajectory']
        if p.get('unit') not in {'record','source'} or not isinstance(p.get('arm'), str):
            raise ValueError('Actual graph trajectory unit/arm required')
        order, checkpoints = p['deletion_order'], p['checkpoints']
        if not isinstance(order,list) or any(not isinstance(v,str) or not v for v in order) or len(order) != len(set(order)):
            raise ValueError('Unique nonempty deletion IDs required')
        if not checkpoints or any(type(k) is not int or k < 1 or k > len(order) for k in checkpoints) or checkpoints != sorted(set(checkpoints)):
            raise ValueError('Nonempty valid checkpoint schedule required')
        if (p['unit'],len(order),checkpoints) != (s['unit'],s['horizon'],s['checkpoint_schedule']):
            raise ValueError('Actual graph request does not match complete profile schedule')
        if s.get('request_arm') != p['arm']:
            raise ValueError('Actual graph arm differs from the measured profile')
        if type(request['fresh_graph']) is not bool or type(s.get('fresh_graph')) is not bool or request['fresh_graph'] != s['fresh_graph']:
            raise ValueError('Fresh-graph setting differs from the measured profile')
    else:
        if s.get('convex_backend') not in {'fraction_reference','exact_dyadic_integer_v1'}:
            raise ValueError('Unknown exact verifier backend')
        if (s.get('parameter_tolerance'),s.get('sigmoid_bits'),s.get('sigmoid_max_terms'),s.get('sigmoid_max_squarings')) != ('1/100000000',128,256,64):
            raise ValueError('Frozen certificate settings cannot change')
        if set(request['candidate']) != {'path','sha256'} or set(request['selection']) != {'unit','deleted_unit_ids','selected_record_ids'}:
            raise ValueError('Bound candidate and exact pointwise selection required')
        if request['selection']['unit'] not in {'record','source'}:
            raise ValueError('Pointwise selection unit required')
    return request


def _read_cell(conf):
    import numpy as np
    from phase3.reference_graph import build_reference_graph
    package = Path(conf['input_package'])
    if package_hashes(package) != conf['input_hashes'] or sha(package/'bundle.json') != conf['input_manifest_sha256']:
        raise ValueError('Input package changed before worker load')
    meta = pf.loads((package/'metadata.json').read_text())
    ids, owners = meta['record_ids'], meta['source_ids']
    cx = np.load(package/'curator.npy',allow_pickle=False)
    x = np.load(package/'learner.npy',allow_pickle=False) if 'learner.npy' in conf['input_hashes'] else None
    y = np.load(package/'targets.npy',allow_pickle=False) if 'targets.npy' in conf['input_hashes'] else None
    s = conf['profile_signature']
    if cx.shape != (len(ids),s['curator_dimension']) or meta['threshold'] != s['threshold'] or meta['priority_seed'] != s['priority_seed']:
        raise ValueError('Resolved curator dimensions/settings differ')
    if s['learner_dimension'] is not None and (x is None or y is None or x.shape != (len(ids),s['learner_dimension']) or y.shape != (len(ids),s['outputs'])):
        raise ValueError('Resolved learner dimensions differ')
    graph = build_reference_graph(cx,ids,meta['threshold'],owners,seed=meta['priority_seed'])
    with np.load(package/'graph.npz',allow_pickle=False) as stored:
        for key,actual in [('priority',graph.priority_indices),('indptr',graph.indptr),('indices',graph.indices)]:
            if not np.array_equal(stored[key],actual):
                raise ValueError('Stored graph does not match actual vectors')
    return {'graph':graph,'cx':cx,'x':x,'y':y,'lambda':s['lambda_reg'],
            'priority_seed':s['priority_seed'],'source_kinds':conf.get('source_kinds',{})}


def worker(config_path):
    conf = pf.loads(Path(config_path).read_text())
    policy = pf.validate_policy(conf['policy'])
    resource.setrlimit(resource.RLIMIT_AS,(policy['memory_bytes'],policy['memory_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(policy['cpu_seconds'],policy['cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if sources() != conf['source_sha256'] or producer_sources() != conf['producer_source_sha256']:
        raise ValueError('Worker source changed before execution')
    import numpy as np
    out = Path(config_path).parent
    cell = _read_cell(conf)
    if conf['kind'] == 'graph':
        from phase6 import dispatch
        path = conf['trajectory'];graph = cell['graph']
        allowed = set(graph.record_ids) if path['unit']=='record' else set(graph.source_ids) & {s for s,k in conf['source_kinds'].items() if k=='genuine_native'}
        if not set(path['deletion_order']) <= allowed:
            raise ValueError('Unknown or nongenuine graph deletion IDs')
        job = {'arm':path['arm'],'fresh_graph':conf['fresh_graph']}
        result = dispatch.structural_job(job,path,cell,out)
        result['measured_operation'] = 'frozen_structural_job_with_input_graph_reconstruction'
    else:
        from fractions import Fraction
        from phase5 import convex_multioutput
        from phase6 import dyadic_convex
        selection = conf['selection'];graph = cell['graph'];s = conf['profile_signature']
        dead = selection['deleted_unit_ids'];unit = selection['unit']
        allowed = set(graph.record_ids if unit=='record' else graph.source_ids)
        if len(dead) != len(set(dead)) or not set(dead) <= allowed:
            raise ValueError('Unknown or repeated pointwise deletion IDs')
        deleted = [rid for rid,owner in zip(graph.record_ids,graph.source_ids) if (rid if unit=='record' else owner) in set(dead)]
        selected = sorted(map(int,graph.selected_indices(deleted)))
        if selection['selected_record_ids'] != [graph.record_ids[i] for i in selected]:
            raise ValueError('Pointwise selected rows differ from graph')
        candidate = out/'candidate.npz'
        if sha(candidate) != conf['candidate']['sha256']:
            raise ValueError('Candidate bytes changed before verifier')
        with np.load(candidate,allow_pickle=False) as a:
            if set(a.files) != {'x','y','w'}:
                raise ValueError('Exact x/y/w candidate arrays required')
            x,y,w = a['x'],a['y'],a['w']
        if x.dtype != np.float32 or y.dtype != np.float64 or w.dtype != np.float64 or w.shape != (s['learner_dimension'],s['outputs']):
            raise ValueError('Frozen candidate dtypes/shape differ')
        if not np.array_equal(x,cell['x'][selected]) or not np.array_equal(y,cell['y'][selected]):
            raise ValueError('Candidate arrays differ from retained original rows')
        verifier = dyadic_convex.certify_multioutput if s['convex_backend']=='exact_dyadic_integer_v1' else convex_multioutput.certify_multioutput
        started = time.perf_counter()
        certificate = verifier(x,y,s['lambda_reg'],w,max_coordinates=policy.get(pf.CONVEX_CAP,2_000_000),
            parameter_tolerance=Fraction(1,10**8),bits=128,max_terms=256,max_squarings=64)
        seconds = time.perf_counter()-started
        write(out/'certificate.json',certificate)
        result = {'status':'completed','measured_operation':'pointwise_exact_convex_verification_only',
            'certificate_seconds':seconds,'certificate_sha256':sha(out/'certificate.json'),
            'candidate_sha256':sha(candidate),'meets_parameter_tolerance':certificate['meets_parameter_tolerance'],
            'optimization_performed':False,'trajectory_execution_claim':False}
    if sources() != conf['source_sha256'] or producer_sources() != conf['producer_source_sha256']:
        raise ValueError('Worker source changed during execution')
    if package_hashes(Path(conf['input_package'])) != conf['input_hashes'] or sha(Path(conf['input_package'])/'bundle.json') != conf['input_manifest_sha256']:
        raise ValueError('Input changed during worker execution')
    write(out/'worker_result.json',result)
    return result


def _isolated(argv, out, policy, env):
    from phase6.measurement import process_tree_sample
    started=time.perf_counter();timed_out=False;samples=[];peak=0;seen=set()
    with (out/'stdout.txt').open('wb') as stdout,(out/'stderr.txt').open('wb') as stderr:
        proc=subprocess.Popen(argv,cwd=out,env=env,stdin=subprocess.DEVNULL,stdout=stdout,
            stderr=stderr,close_fds=True,start_new_session=True)
        while True:
            sample=process_tree_sample(proc.pid);peak=max(peak,sample['rss_sum_bytes']);seen.update(sample['observed_pids'])
            samples.append({'elapsed_seconds':time.perf_counter()-started,**sample})
            pid,status,usage=os.wait4(proc.pid,os.WNOHANG)
            if pid:break
            if time.perf_counter()-started > policy['wall_seconds']:
                timed_out=True;os.killpg(proc.pid,signal.SIGKILL);pid,status,usage=os.wait4(proc.pid,0);break
            time.sleep(.02)
        proc.returncode=os.waitstatus_to_exitcode(status)
    return {'exit_code':proc.returncode,'timeout':timed_out,'spawn_to_reap_seconds':time.perf_counter()-started,
        'peak_rss_bytes':usage.ru_maxrss*1024,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,
        'sampled_process_tree_peak_rss_sum_bytes':peak,'process_tree_observed_pids':sorted(seen),'process_tree_samples':samples,
        'process_tree_sampling_scope':'non_atomic_proc_sweep_every_at_least_20ms_shared_pages_counted_per_process_short_lived_children_may_be_missed',
        'process_tree_exact_peak_claim':False,'process_only_rss_not_deployment_memory':True,
        'read_block_operations':usage.ru_inblock,'write_block_operations':usage.ru_oublock,
        'block_operations_are_not_logical_byte_counts':True}


def _prepare_output(out, evidence_role):
    if out.exists():raise ValueError('Preserve existing measurement output')
    out.mkdir(parents=True,exist_ok=False)
    if evidence_role == 'actual_disjoint_development':
        (out/'.gitignore').write_text('# Actual development arrays and observations remain private by default.\n*\n!.gitignore\n')


def collect(request_path, out_dir):
    """Execute exactly one supplied request; preserve failures and all receipts."""
    began=time.perf_counter();request_path=Path(request_path).resolve();out=Path(out_dir).resolve()
    raw=request_path.read_bytes();request=validate_request(pf.loads(raw.decode()))
    package=contained(request['input_package'],request_path.parent)
    hashes=package_hashes(package);code=sources();producer=producer_sources()
    candidate=None
    if request['kind']=='convex':
        candidate=contained(request['candidate']['path'],request_path.parent)
        if sha(candidate)!=request['candidate']['sha256']:raise ValueError('Bound candidate does not match')
    _prepare_output(out, request['evidence_role']);(out/'request.json').write_bytes(raw)
    if candidate is not None:
        shutil.copyfile(candidate,out/'candidate.npz');write(out/'selection.json',request['selection'])
    conf={**request,'schema':CONFIG_SCHEMA,'input_package':str(package),'input_hashes':hashes,
        'input_manifest_sha256':sha(package/'bundle.json'),'request_sha256':hashlib.sha256(raw).hexdigest(),
        'source_sha256':code,'producer_source_sha256':producer}
    write(out/'prospective_configuration.json',conf)
    argv=[sys.executable,str(Path(__file__).resolve()),'--worker','--config',str(out/'prospective_configuration.json')]
    env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1',
         'OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE'}
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        env[key]=str(request['policy']['threads'])
    invocation={'argv':argv,'cwd':str(out),'environment':env,'python_executable_sha256':sha(sys.executable),
        'configuration_sha256':sha(out/'prospective_configuration.json'),'request_sha256':conf['request_sha256'],
        'source_sha256':code,'producer_source_sha256':producer,'dependency_binary_complete_pinning':False}
    write(out/'invocation.json',invocation)
    process=_isolated(argv,out,request['policy'],env);write(out/'process.json',process)
    integrity_errors=[];worker_result=None
    try:
        if (out/'worker_result.json').exists():worker_result=pf.loads((out/'worker_result.json').read_text())
    except (ValueError,TypeError,OSError) as error:
        integrity_errors.append({'stage':'worker_result_read','exception_type':type(error).__name__,'reason':str(error)})
    try:
        unchanged=(sources()==code and producer_sources()==producer and package_hashes(package)==hashes
                   and sha(package/'bundle.json')==conf['input_manifest_sha256'])
        if not unchanged:integrity_errors.append({'stage':'post_execution_binding','reason':'Source or input hashes differ'})
    except (ValueError,TypeError,KeyError,OSError) as error:
        unchanged=False
        integrity_errors.append({'stage':'post_execution_binding','exception_type':type(error).__name__,'reason':str(error)})
    success=process['exit_code']==0 and process['timeout'] is False and unchanged and worker_result is not None and worker_result['status']=='completed'
    report={'schema':'ccu-'+request['kind']+'-development-measurement-1','status':'completed' if success else 'observed_measurement_failure',
        'kind':request['kind'],'evidence_role':request['evidence_role'],'process':process,'input_hashes':hashes,
        'input_manifest_sha256':conf['input_manifest_sha256'],'source_sha256':code,'producer_source_sha256':producer,
        'profile_signature_sha256':digest(request['profile_signature']),'threads':request['policy']['threads'],
        'request_sha256':conf['request_sha256'],'configuration_sha256':sha(out/'prospective_configuration.json'),
        'invocation_sha256':sha(out/'invocation.json'),'source_and_input_unchanged':unchanged,'worker_result':worker_result,
        'integrity_errors':integrity_errors,
        'primary_acceptance':False,'execution_authenticity_requires_external_review':True,
        'measurement_scope':'fresh process includes imports, source/input checks, graph reconstruction, frozen route, and worker output writes; excludes parent intake/candidate copy and parent report writes',
        'parent_collection_seconds_before_report':time.perf_counter()-began}
    if request['kind']=='convex':
        report['candidate_sha256']=sha(out/'candidate.npz')
        if worker_result is not None:
            for key in ('certificate_sha256','certificate_seconds','meets_parameter_tolerance'):
                if key in worker_result:report[key]=worker_result[key]
    report['artifact_hashes']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'measurement.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--config',type=Path)
    parser.add_argument('--request',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    if args.worker:
        if args.config is None or args.request is not None or args.out is not None:parser.error('Worker requires only --config')
        worker(args.config);return
    if args.request is None or args.out is None or args.config is not None:parser.error('Require --request and fresh --out')
    report=collect(args.request,args.out)
    print(json.dumps({'status':report['status'],'evidence_role':report['evidence_role'],'primary_acceptance':False}))


if __name__=='__main__':main()
