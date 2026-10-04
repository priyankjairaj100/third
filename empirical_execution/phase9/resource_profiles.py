"""Canonical-job resource adapters; actual evidence, never metadata admission.

Frozen phases are imported unchanged. This module derives profiles from actual
resolved arrays and requests, replays derived development packages, and checks
saved observations. It does not start experiments, fit models, or promise cost
bounds. Exact convex certificate replay is charged input verification.
"""
from pathlib import Path
from fractions import Fraction
from copy import deepcopy
import argparse
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
import numpy as np
from phase7 import resource_preflight as pf
from phase8 import resource_policy as prior
from phase6 import recipes,dispatch,extensions
from phase5 import task_program as task
from phase4 import requests as rq

SCHEMA='ccu-canonical-resource-profiles-1'
SUMMARIES={'P-I','P-S','P-R'}
WORKERS={'O-G','O-T','B-E','B-A','P-I','P-S','P-R','B-F','O-G-cached'}
METHOD_RECIPES={'common_cached_graph_oracle','every_release_persistence','cold_warm_scope'}
GRAPH_RECIPES={'structure_mass_PPS','structure_one_percent_horizon','excluded_blocker_stress','news_chronological','fresh_graph_audit'}


def digest(value):return prior.digest(value)
def sha(path):return pf.sha256(Path(path).read_bytes())


def codes():
    result=prior.source_bindings()
    result['phase9/resource_profiles.py']=sha(__file__)
    result['phase9/resource_measurement.py']=sha(ROOT/'phase9/resource_measurement.py')
    return result


def projection(bundle):
    omitted={'execution_policy','external_evidence_review','resource_policy_qualification','resource_policy_observations'}
    value={k:v for k,v in bundle.items() if k not in omitted}
    value['immutable_development_replay_binding']=bundle.get('external_evidence_review',{}).get('development_source_replay_sha256')
    return task.fingerprint(value)


def canonical_owner_jobs(owner_key,bundle):
    """Complete canonical groups/recipe scopes; never caller-selected profiles."""
    registry,jobs=recipes.build_registry();lookup={g['group_id']:g for g in registry['groups']}
    reviewed=bundle.get('_dossier_group_ids')
    scope=bundle.get('phase9_scope',{})
    if set(scope)!={'key','group_ids','extension_recipes'} or scope['key']!=owner_key:
        raise ValueError('Bound Phase9 owner scope is required before qualification')
    requested=scope['group_ids'];names=scope['extension_recipes']
    if not isinstance(reviewed,list) or not isinstance(requested,list) or not isinstance(names,list) or len(requested)!=len(set(requested)) or len(names)!=len(set(names)) or not set(requested)<=set(reviewed):
        raise ValueError('Reviewed canonical group IDs are required')
    groups=[lookup[k] for k in requested]
    if any(g['request_family']!=owner_key for g in groups):
        raise ValueError('Every reviewed group must belong to this immutable owner family')
    available={j['recipe_id'] for j in jobs if j.get('recipe_id') and not j['recipe_id'].startswith('human_') and j['recipe_id']!='test_source_bootstrap' and j.get('corpus','')+'/'+j.get('panel','')==owner_key}
    if not set(names)<=available:raise ValueError('Unregistered or wrong-owner extension scope')
    selected=[]
    for job in jobs:
        if job.get('group_id') in requested:
            selected.append(job)
        elif job.get('recipe_id') in names and job.get('corpus','')+'/'+job.get('panel','')==owner_key:
            selected.append(job)
    if not selected:raise ValueError('Nonempty canonical task scope required; global human/statistical jobs use their own gates')
    return registry,groups,selected


def _base_policy(bundle):
    dossier=bundle['development']
    lock=recipes.derive_resource_policy(dossier['resource_snapshot'],dossier['stage_seconds'])
    if lock!=dossier['resource_lock']:raise ValueError('Frozen base resource derivation does not reproduce')
    value={'memory_bytes':lock['per_method_address_space_limit_bytes'],'cpu_seconds':lock['cpu_seconds_each_stage'],
           'wall_seconds':lock['wall_seconds_each_stage'],'threads':lock['cpu_threads']}
    existing=bundle.get('execution_policy',{})
    for key in ('state_audit_absolute_tolerance','state_audit_relative_tolerance'):
        if key in existing:value[key]=existing[key]
    value['convex_verifier']=existing.get('convex_verifier','fraction_reference')
    pf.validate_policy(value)
    return value,lock


def _accept_group(group,bundle,base,policy):
    candidate={**bundle,'execution_policy':policy}
    replay,development=prior._real_acceptance(group,candidate,base)
    return replay,development


def _encoder(bundle,name,role):
    entry=bundle[role][name]
    return {'name':name,'id':entry['encoder_id'],'revision':entry['encoder_revision']}


def _signature(group,bundle,cell,*,kind,method=None,path=None,panel='cold_process',persistence='final',full=False,backend=None):
    conf=group['configuration'];x,y=cell['x'],cell['y']
    result={'kind':kind,'method':method,'curator':_encoder(bundle,conf['curator_encoder'],'curators'),
        'learner':None if x is None else _encoder(bundle,conf['learner_encoder'],'learners'),
        'curator_dimension':int(cell['cx'].shape[1]),'learner_dimension':None if x is None else int(x.shape[1]),
        'outputs':None if y is None else int(y.shape[1]),'projection_sha256':cell['geometry'].get('projection_sha256'),
        'projection_seed':conf.get('projection_seed'),'threshold':cell['graph'].threshold,'lambda_reg':cell['lambda'],
        'priority_seed':cell['priority_seed'],'unit':None if path is None else path['unit'],
        'horizon':0 if path is None else len(path['deletion_order']),
        'checkpoint_schedule':[] if path is None else list(path['checkpoints']),
        'solver':'cg' if conf.get('decoder') in {'cg','zero_start_CG'} else 'cholesky',
        'state_precision':conf.get('state_precision','FP64'),'panel':panel,'persistence':persistence,
        'full_state':full,'audit_requested':kind=='worker' and conf.get('state_precision','FP64')=='FP64' and (full or method in SUMMARIES|{'B-E'}),
        'batch_rows':1024,'compaction_fraction':.5,'threads':1,'convex_backend':backend,
        'parameter_tolerance':'1/100000000' if kind=='convex' else None,
        'sigmoid_bits':128 if kind=='convex' else None,'sigmoid_max_terms':256 if kind=='convex' else None,
        'sigmoid_max_squarings':64 if kind=='convex' else None}
    if kind=='convex':
        # A saved certificate qualifies a pointwise verifier call only. It is
        # not evidence that a trajectory, optimizer, or Newton warm start ran.
        for name in ('unit','horizon','checkpoint_schedule','solver','state_precision','panel','persistence','full_state','batch_rows','compaction_fraction'):
            result[name]=None
        result['scope']='pointwise_dense_exact_verifier_only'
    return result


def compile_profiles(owner_key,bundle,*,base_dir='.',require_acceptance=True):
    """Canonical scope discovery; require_acceptance=False is software inspection only.

    The public qualifier always requires real acceptance. No unchecked compiler
    result can supply a policy or pass verify_owner_resources.
    """
    if bundle.get('phase9_stable_root')!=str(Path(base_dir).resolve()):
        raise ValueError('Phase9 stable root must match the actual immutable owner root')
    registry,groups,jobs=canonical_owner_jobs(owner_key,bundle)
    policy,lock=_base_policy(bundle);group_lookup={g['group_id']:g for g in groups}
    cells={};manifests={};acceptance={};profiles={};job_rows=[];group_objects={}
    for job in jobs:
        group=group_lookup.get(job.get('group_id'))
        recipe=job.get('recipe_id')
        if group is None:group=extensions._group(job,bundle,dispatch.PRIMARY_MODE)
        gid=group['group_id'];group_objects[gid]=group
        if gid not in cells:
            if require_acceptance:
                replay,development=_accept_group(group,bundle,base_dir,policy)
                acceptance[gid]={'source_replay':replay,'development':development}
            cells[gid]=dispatch.resolve(group,bundle)
            if recipe is None:manifests[gid]=dispatch.request_for_group(group,bundle,cells[gid])
        cell=cells[gid];path=None
        row={'job_id':job['job_id'],'group_id':gid,'profile_ids':[]}
        active_recipe=recipe in METHOD_RECIPES|GRAPH_RECIPES|{'utility_reference'}
        active_core=recipe is None and (group['block'] in {'F_convex'} or group['methods']==['independent_structure_oracle'] or (set(group['methods'])<=dispatch.KNOWN_METHODS and cell['x'] is not None))
        if not (active_recipe or active_core):
            row.update(disposition='outside_added_audit_and_verifier_caps',reason='Own frozen branch gates and common base limits remain mandatory; no runtime/RSS qualification here.')
            job_rows.append(row);continue
        if recipe is None:
            if group['block']!='E_full_refit':
                paths={p['trajectory_id']:p for p in manifests[gid]['trajectories']}
                path=paths[job['trajectory_id']];dispatch._check_path(job,path,bundle)
        elif recipe!='utility_reference':
            path=extensions.extension_path(job,bundle,cell)
        if path is not None and not path['checkpoints']:
            row.update(disposition='unavailable_or_structural_zero',reason=path.get('status'),actual_horizon=len(path['deletion_order']))
            job_rows.append(row);continue
        kinds=[]
        if recipe in METHOD_RECIPES or (recipe is None and set(group['methods'])<=dispatch.KNOWN_METHODS and cell['x'] is not None):
            methods=['O-G-cached'] if recipe=='common_cached_graph_oracle' else job.get('methods',group['methods'])
            panel='warm_service' if recipe=='cold_warm_scope' else 'every_release' if recipe=='every_release_persistence' else job.get('panel_mode','cold_process')
            persistence='every_release' if panel=='every_release' else job.get('persistence','final')
            for method in methods:
                kinds.append(_signature(group,bundle,cell,kind='worker',method=method,path=path,panel=panel,persistence=persistence,full=bool(job.get('complete_state_rebuild_each_checkpoint'))))
        elif group['block']=='F_convex' or recipe=='utility_reference':
            backend='exact_dyadic_integer_v1' if recipe=='utility_reference' else policy['convex_verifier']
            kinds.append(_signature(group,bundle,cell,kind='convex',path=path,backend=backend))
        elif group['methods']==['independent_structure_oracle'] or recipe in GRAPH_RECIPES:
            signature=_signature(group,bundle,cell,kind='graph',path=path)
            signature.update(request_arm=path['arm'],fresh_graph=bool(job.get('fresh_graph',False) or recipe=='fresh_graph_audit'))
            kinds.append(signature)
        else:
            row.update(disposition='outside_added_audit_and_verifier_caps',
                reason='Own frozen branch gates and common base limits remain mandatory; no runtime/RSS qualification here.')
        for signature in kinds:
            pid=digest(signature)
            selected=[len(cell['graph'].selected_indices())]
            if signature['kind']=='convex' and path is not None:
                for checkpoint in path['checkpoints']:
                    dead=dispatch.boundary.deleted_indices(path,checkpoint,cell['graph'])
                    selected.append(len(cell['graph'].selected_indices([cell['graph'].record_ids[i] for i in dead])))
            if pid not in profiles:
                n=len(cell['graph'].record_ids)
                profiles[pid]={'profile_id':pid,'signature':signature,'original_records':n,
                    'selected_counts':selected if signature['kind']=='convex' else None,
                    'maximum_selected_records':max(selected) if signature['kind']=='convex' else None,
                    'future_symbolic_keys':None,'symbolic_key_upper_bound':2*n,
                    'job_ids':[],'group_id':gid,'selected_counts_by_job':{},'utility_budget_required':False,
                    'resolved_array_bindings':{k:task.fingerprint(cell[k]) for k in ('cx','x','y')},
                    'geometry_binding':cell['geometry']}
            elif profiles[pid]['original_records']!=len(cell['graph'].record_ids):
                raise ValueError('A profile signature unexpectedly spans different panel sizes')
            if signature['kind']=='convex':
                profiles[pid]['selected_counts_by_job'][job['job_id']]=selected
                profiles[pid]['selected_counts']=sorted(set(profiles[pid]['selected_counts']+selected))
                profiles[pid]['maximum_selected_records']=max(profiles[pid]['selected_counts'])
                profiles[pid]['utility_budget_required'] |= recipe=='utility_reference'
            profiles[pid]['job_ids'].append(job['job_id']);row['profile_ids'].append(pid)
        if kinds:row['disposition']='matching_observations_required'
        job_rows.append(row)
    public={'schema':SCHEMA,'owner_key':owner_key,'registry_sha256':digest(registry),'canonical_jobs_sha256':digest(jobs),
        'reviewed_group_ids':[g['group_id'] for g in groups],'covered_job_ids':[j['job_id'] for j in jobs],
        'jobs':job_rows,'profiles':list(profiles.values()),'base_policy':policy,'resource_lock':lock,
        'source_acceptance_required':require_acceptance,'cost_guarantee':False}
    public['sha256']=digest(public)
    return public,{'cells':cells,'acceptance':acceptance,'groups':group_objects,'jobs':jobs}


def _contained(path,base):
    resolved=(Path(base)/path).resolve()
    if not resolved.is_relative_to(Path(base).resolve()):raise ValueError('Artifact path escaped immutable owner root')
    return resolved


def _bound(reference,base):return prior.bound_json(reference,base)


def replay_derived_package(profile,entry,bundle,context,*,base_dir):
    """Reconstruct mixed/projected development arrays from accepted real caches."""
    signature=profile['signature'];dossier=bundle['development'];source=dossier.get('source_bundle',bundle)
    accepted=context['acceptance'][profile['group_id']]['development']
    package=_contained(entry['input_package'],base_dir)
    manifest=json.loads((package/'bundle.json').read_text());metadata=json.loads((package/'metadata.json').read_text())
    rows=dossier['records'];ids=[r['record_id'] for r in rows];sources=[r['source_unit_id'] for r in rows]
    if metadata['record_ids']!=ids or metadata['source_ids']!=sources:raise ValueError('Derived pilot IDs/owners differ from accepted disjoint development')
    from phase3.run_preparation import read_rows
    prepared=read_rows((Path(base_dir)/source['source_acceptance']['prepared']['records']).resolve())
    lookup={r['record_id']:i for i,r in enumerate(prepared)};indices=np.asarray([lookup[r] for r in ids],dtype=int)
    def cache(identity):
        name=identity['name'];actual=source['curators'].get(name,source.get('learners',{}).get(name))
        if actual['encoder_id']!=identity['id'] or actual['encoder_revision']!=identity['revision']:
            raise ValueError('Derived pilot encoder identity/revision differs from accepted caches')
        directory=(Path(base_dir)/source['source_acceptance']['caches'][name]['directory']).resolve()
        return np.load(directory/'vectors.npy',mmap_mode='r',allow_pickle=False)[indices]
    cx=cache(signature['curator']);x=None;y=None
    if signature['learner'] is not None:
        x=cache(signature['learner'])
        if signature['projection_sha256'] is not None:
            p=np.asarray(bundle['projections'][str(signature['learner_dimension'])]['matrix'])
            if task.array_hash(p)!=signature['projection_sha256'] or not np.array_equal(p,task.seeded_projection(x.shape[1],signature['learner_dimension'],signature['projection_seed'])):
                raise ValueError('Derived learner projection differs from the registered fixed projection')
            x=(x.astype(np.float64)@p).astype(np.float32)
        if bundle['task']=='civil':y=np.asarray([[r['original_fields']['toxicity']] for r in rows],np.float64)
        elif bundle['task']=='multilabel':
            learner=bundle['learners'][signature['learner']['name']]
            y=task.model_selection.encode_tag_targets(rows,learner['calibration']['vocabulary_lock'])
        else:raise ValueError('Do not fabricate targets for an unlabeled pilot')
    expected={'curator.npy':cx}
    if x is not None:expected.update({'learner.npy':x,'targets.npy':y})
    for name,value in expected.items():
        if not np.array_equal(np.load(package/name,allow_pickle=False),value):raise ValueError('Derived pilot array does not replay: '+name)
    if metadata['threshold']!=signature['threshold'] or metadata['priority_seed']!=signature['priority_seed']:
        raise ValueError('Derived graph settings do not match canonical resolved configuration')
    from phase3.reference_graph import build_reference_graph
    graph=build_reference_graph(cx,ids,signature['threshold'],sources,seed=signature['priority_seed'])
    with np.load(package/'graph.npz',allow_pickle=False) as stored:
        for key,value in [('priority',graph.priority_indices),('indptr',graph.indptr),('indices',graph.indices)]:
            if not np.array_equal(stored[key],value):raise ValueError('Derived graph does not replay')
    hashes={name:sha(package/name) for name in manifest['files']}
    if any(manifest['files'][name]['sha256']!=value for name,value in hashes.items()):raise ValueError('Derived package hash changed')
    if not (set(expected)|{'metadata.json','graph.npz'})<=set(hashes):raise ValueError('Derived package omits required files')
    return {'directory':package,'arrays':{'cx':cx,'x':x,'y':y},'graph':graph,'input_hashes':hashes,
        'records':len(ids),'record_ids':ids,'source_ids':sources,
        'native_source_ids':sorted({r['source_unit_id'] for r in rows if r.get('source_kind','').startswith('native_')}),
        'accepted_development':accepted,
        'derivation_binding':{'input_hashes':hashes,'accepted_development_replay':accepted['development_source_replay_sha256'],
            'signature_sha256':digest(signature),'source_cache_replay_required':True}}


def _worker_sources():
    return {str(p.relative_to(ROOT)):sha(p) for folder in ('ccu','phase3','phase4','phase5','phase6') for p in sorted((ROOT/folder).glob('*.py'))}


def _measurements(measurement,policy):
    if measurement.get('exit_code')!=0 or measurement.get('timeout') is not False:raise ValueError('Noncompletion is retained; no qualifying observation')
    wall=prior.positive(measurement['spawn_to_reap_seconds'],'observed wall seconds')
    cpu=sum(measurement[k] for k in ('user_cpu_seconds','system_cpu_seconds'))
    if any(type(measurement[k]) not in (int,float) or not math.isfinite(measurement[k]) or measurement[k]<0 for k in ('user_cpu_seconds','system_cpu_seconds')):raise ValueError('Actual nonnegative CPU times required')
    rss=max(pf.integer(measurement['peak_rss_bytes'],'wait4 RSS',1),pf.integer(measurement['sampled_process_tree_peak_rss_sum_bytes'],'sampled RSS',1))
    return {'wall_seconds':wall,'cpu_seconds':cpu,'reported_rss_indicator':rss,
            'fits_reported_limits':wall<=policy['wall_seconds'] and cpu<=policy['cpu_seconds'] and rss<=policy['memory_bytes'],
            'true_peak_or_address_space_bound':False}


def worker_observation(profile,entry,derived,bundle,base,policy):
    signature=profile['signature'];report,report_path=_bound(entry['service_report'],base);config,_=_bound(entry['configuration'],base)
    if report.get('success') is not True:raise ValueError('Failed worker observation blocks this prospective plan')
    method=signature['method']+('-FP32' if signature['state_precision']=='FP32' else '')
    expected={'method':method,'unit':signature['unit'],'solver':signature['solver'],'panel':signature['panel'],
              'horizon':signature['horizon'],'lambda_reg':signature['lambda_reg'],'persistence':signature['persistence'],
              'batch_rows':signature['batch_rows'],'compaction_fraction':signature['compaction_fraction']}
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Measured worker configuration differs from canonical profile')
    for k in ('method','unit','solver','panel','persistence'):
        if report.get(k)!=expected[k]:raise ValueError('Retagged worker execution signature')
    if config.get('policy')!=report.get('policy') or any(config['policy'].get(k)!=policy[k] for k in ('memory_bytes','cpu_seconds','wall_seconds','threads')):raise ValueError('Worker base resource/process policy differs')
    if config.get('code_hashes')!=_worker_sources() or report.get('project_python_source_hashes')!=_worker_sources():raise ValueError('Complete observed worker dependencies differ')
    rel=str(Path(entry['configuration']['path']).relative_to(Path(entry['service_report']['path']).parent))
    if report.get('artifact_hashes',{}).get(rel)!=entry['configuration']['sha256']:raise ValueError('Service does not bind its actual configuration')
    used={'metadata.json','graph.npz','learner.npy','targets.npy'}|({'curator.npy'} if method in {'O-G','O-G-FP32'} else set())
    if report.get('raw_input_files_sha256')!={name:derived['input_hashes'][name] for name in used} or config.get('input_hashes')!=report['raw_input_files_sha256']:raise ValueError('Service arrays differ from replayed transformed pilot')
    requests=config['requests'];cumulative=set();schedule=[]
    allowed=set(derived['record_ids'] if signature['unit']=='record' else [r['source_unit_id'] for r in bundle['development']['records'] if r.get('source_kind','').startswith('native_')])
    for batch in requests:
        if any(identifier not in allowed for identifier in batch):raise ValueError('Unknown or nongenuine development deletion unit')
        cumulative.update(batch);schedule.append(len(cumulative))
    if schedule!=signature['checkpoint_schedule'] or (schedule and schedule[-1]!=signature['horizon']):raise ValueError('Complete actual distinct-unit schedule differs')
    if report.get('request_sha256')!=hashlib.sha256(json.dumps(requests,separators=(',',':')).encode()).hexdigest():raise ValueError('Measured request hash differs')
    stages=[_measurements(report[name],policy) for name in ('construction','repair')]
    result={'profile_id':profile['profile_id'],'records':derived['records'],'stages':stages,'observed_limits_fit':all(s['fits_reported_limits'] for s in stages),
        'symbol_coordinates':None,'pair_coordinates':None,'worker_report_sha256':entry['service_report']['sha256'],'derivation':derived['derivation_binding']}
    if signature['audit_requested']:
        audit,_=_bound(entry['audit_report'],base)
        if audit.get('passed') is not True or audit.get('sha256')!=digest({k:v for k,v in audit.items() if k!='sha256'}) or audit.get('code_sha256')!=sha(ROOT/'phase6/state_audit.py'):raise ValueError('Unverified or changed actual state audit')
        if audit['method']!=signature['method'] or audit['full_state_requested'] is not signature['full_state'] or audit['measured_service_report_sha256']!=entry['service_report']['sha256']:raise ValueError('State audit scope/reference differs')
        auxiliary,auxpath=_bound(entry['audit_service_report'],base)
        if audit['audit_service_report_sha256']!=entry['audit_service_report']['sha256'] or auxiliary.get('success') is not True or auxiliary.get('project_python_source_hashes')!=_worker_sources():raise ValueError('Auxiliary audit worker binding differs')
        if auxiliary.get('raw_input_files_sha256')!=report['raw_input_files_sha256'] or auxiliary.get('request_sha256')!=report['request_sha256'] or auxiliary.get('policy')!=report['policy']:raise ValueError('Auxiliary audit worker inputs differ')
        auxconf,_=_bound(entry['audit_configuration'],base)
        relative=str(Path(entry['audit_configuration']['path']).relative_to(Path(entry['audit_service_report']['path']).parent))
        if auxiliary.get('artifact_hashes',{}).get(relative)!=entry['audit_configuration']['sha256']:
            raise ValueError('Auxiliary service does not bind its actual prospective configuration')
        for key,value in expected.items():
            wanted='every_release' if key=='persistence' else value
            if auxconf.get(key)!=wanted:raise ValueError('Auxiliary configuration differs: '+key)
        for key in ('method','unit','solver','panel'):
            if auxiliary.get(key)!=expected[key]:raise ValueError('Auxiliary execution signature differs: '+key)
        if auxiliary.get('persistence')!='every_release' or auxconf.get('requests')!=requests or auxconf.get('policy')!=report['policy'] or auxconf.get('input_hashes')!=report['raw_input_files_sha256'] or auxconf.get('code_hashes')!=_worker_sources():
            raise ValueError('Auxiliary complete execution/configuration binding differs')
        prospective=audit['prospective_audit_policy']
        for name,maximum in [('absolute_tolerance',1e-11),('relative_tolerance',1e-10)]:
            value=prospective[name]
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=maximum:raise ValueError('Audit numerical gate weakened')
        if prospective!={'absolute_tolerance':report['policy'].get('state_audit_absolute_tolerance',1e-11),'relative_tolerance':report['policy'].get('state_audit_relative_tolerance',1e-10),'max_pair_coordinates':report['policy'].get(pf.PAIR_CAP,100_000_000),'max_coefficient_coordinates':report['policy'].get(pf.COEFFICIENT_CAP,4_000_000)}:raise ValueError('Actual audit caps differ from worker policy')
        rows=audit['checkpoints']
        if len(rows)!=len(schedule):raise ValueError('Missing audit checkpoint')
        maxima=[];d=signature['learner_dimension'];q=signature['outputs'];packed=d*(d+1)//2+d*q+1
        for i,(row,count) in enumerate(zip(rows,schedule)):
            if row.get('passed') is not True or row.get('index')!=i or row.get('deleted_unique_units')!=count or row.get('remaining_horizon')!=signature['horizon']-count or row.get('reconstructed_actual_snapshot_bound') is not True or row.get('replay_and_measured_heads_bitwise_equal') is not True:raise ValueError('Incomplete actual audit checkpoint binding')
            if signature['full_state'] and signature['method'] in SUMMARIES:
                coefficients=row['coefficients']
                if coefficients.get('passed') is not True or coefficients.get('complete_union_of_keys_compared') is not True:raise ValueError('Full coefficient comparison missing')
                maxima.append(sum(v['expected_designated_key'] is True for v in coefficients['rows'])*packed)
        auxiliary_stages=[_measurements(auxiliary[s],policy) for s in ('construction','repair')]
        parent=pf.integer(audit['auditor_process_lifetime_peak_rss_bytes'],'auditor lifetime RSS',1)
        indicator=parent+max(s['reported_rss_indicator'] for s in auxiliary_stages)
        result.update(symbol_coordinates=max(maxima,default=0),pair_coordinates=derived['records']*(derived['records']-1)//2*signature['curator_dimension'],
            auxiliary_stages=auxiliary_stages,total_audit_seconds=prior.positive(audit['charged_total_audit_seconds'],'actual audit seconds'),
            reported_parent_plus_worker_RSS_indicator=indicator,RSS_indicator_is_bound=False,
            observed_limits_fit=result['observed_limits_fit'] and all(s['fits_reported_limits'] for s in auxiliary_stages) and indicator<=policy['memory_bytes'])
    return result


def _producer_evidence(profile,entry,derived,base,policy):
    """Read the retained executable producer record; schema flags are insufficient."""
    from phase9 import resource_measurement as producer
    report,path=_bound(entry['measurement'],base);out=path.parent
    if report.get('evidence_role')!='actual_disjoint_development' or report.get('status')!='completed' or report.get('primary_acceptance') is not False or report.get('source_and_input_unchanged') is not True:
        raise ValueError('Actual completed development producer record required; software or failure is not evidence')
    kind=profile['signature']['kind']
    if report.get('schema')!='ccu-'+kind+'-development-measurement-1' or report.get('kind')!=kind:
        raise ValueError('Wrong measured route')
    def artifact(name):
        expected=report['artifact_hashes'][name];actual=_contained(str((out/name).relative_to(Path(base).resolve())),base)
        if not actual.is_file() or sha(actual)!=expected:raise ValueError('Measured producer artifact changed: '+name)
        return pf.loads(actual.read_text())
    for name,value in report.get('artifact_hashes',{}).items():
        location=(out/name).resolve()
        if not location.is_relative_to(out.resolve()) or not location.is_file() or sha(location)!=value:
            raise ValueError('Incomplete or escaped producer output binding')
    configuration=artifact('prospective_configuration.json');request=artifact('request.json');invocation=artifact('invocation.json')
    producer.validate_request(request)
    if request['evidence_role']!='actual_disjoint_development' or request['profile_signature']!=profile['signature'] or configuration['profile_signature']!=profile['signature']:
        raise ValueError('Measured signature or development role differs from canonical profile')
    for name,key in [('request.json','request_sha256'),('prospective_configuration.json','configuration_sha256'),('invocation.json','invocation_sha256')]:
        if report[key]!=report['artifact_hashes'][name]:raise ValueError('Producer self-binding differs')
    expected_configuration={**request,'schema':producer.CONFIG_SCHEMA,'input_package':str(derived['directory'].resolve()),
        'input_hashes':derived['input_hashes'],'input_manifest_sha256':sha(derived['directory']/'bundle.json'),
        'request_sha256':report['request_sha256'],'source_sha256':_worker_sources(),'producer_source_sha256':producer.producer_sources()}
    if configuration!=expected_configuration:raise ValueError('Actual producer configuration does not replay')
    if report['source_sha256']!=_worker_sources() or report['producer_source_sha256']!=producer.producer_sources() or report['input_hashes']!=derived['input_hashes'] or report['input_manifest_sha256']!=expected_configuration['input_manifest_sha256']:
        raise ValueError('Producer source and actual input lineage differs')
    for key in ('configuration_sha256','request_sha256','source_sha256','producer_source_sha256'):
        if invocation.get(key)!=report[key]:raise ValueError('Actual invocation binding differs: '+key)
    if invocation.get('argv')!=[sys.executable,str(Path(producer.__file__).resolve()),'--worker','--config',str(out/'prospective_configuration.json')] or invocation.get('cwd')!=str(out) or invocation.get('python_executable_sha256')!=sha(sys.executable):
        raise ValueError('Actual retained executable/invocation differs')
    thread_names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
    expected_environment={'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE',**{k:str(policy['threads']) for k in thread_names}}
    if set(invocation['environment'])!=set(expected_environment)|{'PATH'} or any(invocation['environment'].get(k)!=v for k,v in expected_environment.items()):
        raise ValueError('Producer process/thread environment differs')
    if any(request['policy'].get(k)!=policy[k] for k in ('memory_bytes','cpu_seconds','wall_seconds','threads')) or report['threads']!=policy['threads'] or report['profile_signature_sha256']!=digest(profile['signature']):
        raise ValueError('Producer thread/settings receipt differs')
    if artifact('process.json')!=report['process'] or artifact('worker_result.json')!=report['worker_result'] or report['worker_result'].get('status')!='completed':
        raise ValueError('Process/result receipt is internally inconsistent')
    return report,configuration,out


def graph_observation(profile,entry,derived,base,policy):
    """Bind an actual externally reviewed graph measurement and replay its bytes.

    A declared producer hash does not establish historical authenticity. The
    later complete external review must cover measurement execution and host.
    """
    report,conf,out=_producer_evidence(profile,entry,derived,base,policy)
    if not {'structure.json','structure_forecast.json'}<=set(report['artifact_hashes']):raise ValueError('Complete graph producer outputs must be byte-bound')
    trajectory=conf['trajectory'];allowed=set(derived['record_ids'] if trajectory['unit']=='record' else derived['native_source_ids'])
    if not set(trajectory['deletion_order'])<=allowed:raise ValueError('Graph trajectory uses unknown or nongenuine development units')
    if set(conf['source_kinds'])!=set(derived['source_ids']) or {s for s,k in conf['source_kinds'].items() if k=='genuine_native'}!=set(derived['native_source_ids']):
        raise ValueError('Graph source frame differs from actual accepted development')
    rows=pf.loads((out/'structure.json').read_text())
    if len(rows)!=len(trajectory['checkpoints']):raise ValueError('Graph observation omitted checkpoint output')
    for row,checkpoint in zip(rows,trajectory['checkpoints']):
        dead=set(trajectory['deletion_order'][:checkpoint]);deleted=[rid for rid,source in zip(derived['record_ids'],derived['source_ids']) if (rid if trajectory['unit']=='record' else source) in dead]
        selected=[derived['record_ids'][i] for i in sorted(map(int,derived['graph'].selected_indices(deleted)))]
        if row['status']!='completed' or row['checkpoint']!=checkpoint or row['selected_ids']!=selected or row['selected_count']!=len(selected) or row['independent_graph_checked'] is not True:
            raise ValueError('Actual graph output does not replay complete retained membership')
    measured=_measurements(report['process'],policy)
    return {'profile_id':profile['profile_id'],'records':derived['records'],'measurement':measured,
        'observed_limits_fit':measured['fits_reported_limits'],'graph_pair_count_proxy':derived['records']*(derived['records']-1)//2*profile['signature']['curator_dimension'],
        'pair_coordinates':None,'symbol_coordinates':None,'derivation':derived['derivation_binding'],
        'measurement_authenticity_requires_external_review':True}


def _certificate_core(value):
    if isinstance(value,dict):return {k:_certificate_core(v) for k,v in value.items() if k not in {'elapsed_seconds','sha256','dyadic_work'}}
    if isinstance(value,list):return [_certificate_core(v) for v in value]
    return value


def convex_observation(profile,entry,derived,base,policy):
    from phase4 import convex as scalar
    from phase5 import convex_multioutput as multi
    from phase6 import dyadic_convex
    signature=profile['signature'];measurement,configuration,out=_producer_evidence(profile,entry,derived,base,policy)
    report,_=_bound(entry['certificate'],base);selection,_=_bound(entry['selection'],base)
    candidate=_contained(entry['candidate']['path'],base)
    if sha(candidate)!=entry['candidate']['sha256']:raise ValueError('Convex candidate bytes changed')
    if configuration['selection']!=selection or configuration['candidate']['sha256']!=entry['candidate']['sha256'] or measurement['artifact_hashes'].get('candidate.npz')!=entry['candidate']['sha256'] or measurement['artifact_hashes'].get('selection.json')!=entry['selection']['sha256'] or measurement['artifact_hashes'].get('certificate.json')!=entry['certificate']['sha256']:
        raise ValueError('Convex candidate/selection/certificate differ from actual producer inputs and outputs')
    dead=selection['deleted_unit_ids'];unit=selection['unit']
    if unit not in {'record','source'}:raise ValueError('Actual pointwise selection unit required')
    allowed=set(derived['record_ids'] if unit=='record' else derived['native_source_ids'])
    if len(dead)!=len(set(dead)) or not set(dead)<=allowed:raise ValueError('Invalid actual convex deletion set')
    deleted_records=[rid for rid,source in zip(derived['record_ids'],derived['source_ids']) if (rid if unit=='record' else source) in set(dead)]
    selected=sorted(map(int,derived['graph'].selected_indices(deleted_records)))
    selected_ids=[derived['record_ids'][i] for i in selected]
    if selection.get('selected_record_ids')!=selected_ids:raise ValueError('Convex rows do not derive from retained curation')
    with np.load(candidate,allow_pickle=False) as arrays:
        x,y,w=arrays['x'],arrays['y'],arrays['w']
        if not np.array_equal(x,derived['arrays']['x'][selected]) or not np.array_equal(y,derived['arrays']['y'][selected]):raise ValueError('Convex features or original targets differ from accepted transformed development')
        if x.dtype!=np.float32 or y.dtype!=np.float64 or w.dtype!=np.float64 or w.shape!=(signature['learner_dimension'],signature['outputs']):raise ValueError('Convex candidate dtype/shape differs')
        density=float(np.count_nonzero(x)/x.size) if x.size else None
        if x.size and density<.95:raise ValueError('Dense convex verification requires actual density at least .95')
        if measurement.get('schema')!='ccu-convex-development-measurement-1' or measurement.get('status')!='completed':raise ValueError('Actual completed convex measurement required')
        if measurement.get('candidate_sha256')!=entry['candidate']['sha256'] or measurement.get('certificate_sha256')!=entry['certificate']['sha256'] or measurement.get('source_sha256')!=_worker_sources():raise ValueError('Convex observation source/candidate/certificate bindings differ')
        if measurement.get('profile_signature_sha256')!=digest(signature) or measurement.get('threads')!=policy['threads']:raise ValueError('Convex backend/settings/thread receipt differs')
        if report.get('meets_parameter_tolerance') is not True:raise ValueError('Observed exact certificate failed the unchanged release tolerance')
        cap=report['total_coordinate_budget'];pf.integer(cap,'actual convex count cap')
        if cap!=configuration['policy'].get(pf.CONVEX_CAP,2_000_000):raise ValueError('Actual convex producer cap differs from certificate')
        verifier=dyadic_convex.certify_multioutput if signature['convex_backend']=='exact_dyadic_integer_v1' else multi.certify_multioutput
        rebuilt=verifier(x,y,signature['lambda_reg'],w,max_coordinates=cap,parameter_tolerance=Fraction(1,10**8),bits=128,max_terms=256,max_squarings=64)
        if _certificate_core(rebuilt)!=_certificate_core(report):raise ValueError('Exact stored-value certificate does not replay')
        coordinates=len(x)*x.shape[1]*y.shape[1]
        seconds=prior.positive(measurement['certificate_seconds'],'actual certificate seconds')
        process=_measurements(measurement['process'],policy)
        if seconds>process['wall_seconds']:raise ValueError('Certificate time exceeds measured process wall time')
        return {'profile_id':profile['profile_id'],'records':derived['records'],'selected_records':len(x),'coordinate_demand':coordinates,
            'dimension':x.shape[1],'outputs':y.shape[1],'feature_density':density,'certificate_seconds':seconds,
            'process':process,'observed_limits_fit':process['fits_reported_limits'],
            'input_bytes':int(x.nbytes+y.nbytes+w.nbytes),'derivation':derived['derivation_binding'],
            'measurement_sha256':entry['measurement']['sha256'],
            'certificate_exact_replay':True,'native_runtime_guarantee':False,'pair_coordinates':None,'symbol_coordinates':None}


def select_policy(manifest,observations):
    policy=dict(manifest['base_policy']);defaults=pf.source_contract()['default_caps'];coverage=[]
    required_pair=defaults[pf.PAIR_CAP];full_limits=[];convex_demand=defaults[pf.CONVEX_CAP]
    for profile in manifest['profiles']:
        signature=profile['signature'];matches=[o for o in observations if o['profile_id']==profile['profile_id'] and o['records']>=profile['original_records'] and o['observed_limits_fit']]
        if not matches:raise ValueError('No jointly matching accepted observation for profile '+profile['profile_id'])
        if signature['audit_requested']:
            demand=profile['original_records']*(profile['original_records']-1)//2*signature['curator_dimension']
            required_pair=max(required_pair,demand)
            matches=[o for o in matches if o['pair_coordinates']>=demand]
            if not matches:raise ValueError('No actual audit observation covers original pair predicate demand')
            if signature['full_state'] and signature['method'] in SUMMARIES:
                full_limits.append(max(o['symbol_coordinates'] for o in matches))
        if signature['kind']=='convex':
            demand=profile['maximum_selected_records']*signature['learner_dimension']*signature['outputs']
            matches=[o for o in matches if o['selected_records']>=profile['maximum_selected_records'] and o['coordinate_demand']>=demand]
            convex_demand=max(convex_demand,demand)
        if not matches:raise ValueError('Observed count extent does not cover actual resolved demand')
        coverage.append({'profile_id':profile['profile_id'],'observation_ids':[o['observation_id'] for o in matches],
            'future_symbolic_keys':None,'actual_primary_selected_counts':profile['selected_counts'],
            'runtime_RSS_guarantee':False})
    coefficient_cap=min(full_limits) if full_limits else defaults[pf.COEFFICIENT_CAP]
    if coefficient_cap<defaults[pf.COEFFICIENT_CAP]:raise ValueError('Observed full coefficient extent does not cover the unchanged default cap')
    policy.update({pf.PAIR_CAP:required_pair,pf.COEFFICIENT_CAP:coefficient_cap,pf.CONVEX_CAP:convex_demand,
        'policy_role':'canonical_registered_observed_count_settings_not_native_cost_guarantee'})
    pf.validate_policy(policy)
    return {'execution_policy':policy,'coverage':coverage,'runtime_forecast':None,'RSS_forecast':None,
        'unknown_graph_dependent_symbolic_counts':True,'overflow':'retain original refusal/failure; no waiver',
        'convex_scope':'Exact replay and joint observed count extent; no optimization-time qualification',
        'full_state_cap_scope':'Minimum observed full-profile maximum; does not guarantee all future key counts fit.'}


def utility_budget_bindings(manifest,observations,bundle):
    """Reproduce the separate frozen utility cap, which ignores global policy.

    Its rate extrapolation remains a forecast. The new qualifier also requires
    a joint observed count envelope through select_policy; neither is a bound.
    """
    results=[]
    for profile in manifest['profiles']:
        if not profile['utility_budget_required']:continue
        sig=profile['signature'];d,q=sig['learner_dimension'],sig['outputs']
        demand=profile['maximum_selected_records']*d*q
        planned=bundle.get('convex_verification_development')
        if planned is None:
            if demand>2_000_000:raise ValueError('Utility needs its own reviewed convex_verification_development lock; global cap is ignored')
            results.append({'profile_id':profile['profile_id'],'executed_cap':2_000_000,'lock':None,'forecast_is_guarantee':False});continue
        matching=[o for o in observations if o['profile_id']==profile['profile_id'] and o['selected_records']>0]
        expected=[{'rows':o['selected_records'],'dimension':o['dimension'],'outputs':o['outputs'],
            'feature_density':o['feature_density'],'seconds':o['certificate_seconds'],
            'peak_rss_bytes':o['process']['reported_rss_indicator'],'source_sha256':o['measurement_sha256'],
            'input_bytes':o['input_bytes']} for o in matching]
        if not expected or planned['observations']!=expected or planned['resource_policy']!=manifest['resource_lock']:
            raise ValueError('Utility forecast inputs must exactly project accepted dense producer observations and common resource lock')
        rebuilt=recipes.select_verification_budget(expected,rows=profile['original_records'],dimension=d,outputs=q,resource_policy=manifest['resource_lock'])
        if rebuilt!=planned['locked_budget']:raise ValueError('Separate utility budget does not reproduce')
        if not rebuilt['forecast_feasible'] or rebuilt['max_coordinates']<demand:raise ValueError('Frozen utility budget preserves its forecast/cap refusal')
        results.append({'profile_id':profile['profile_id'],'executed_cap':rebuilt['max_coordinates'],'lock':rebuilt,'forecast_is_guarantee':False})
    return results


def qualify_owner_resources(owner_key,bundle,evidence,*,base_dir='.'):
    start=codes();initial_projection=digest(projection(bundle));result={'schema':SCHEMA,'owner_key':owner_key,'status':'blocked','execution_allowed':False,
        'execution_policy':None,'covered_job_ids':[],'actual_observations_accepted':0,'source_sha256':start,
        'native_cost_guarantee':False,'complete_primary_acceptance':False,'blockers':[]}
    stage='canonical_scope_and_actual_acceptance'
    try:
        manifest,context=compile_profiles(owner_key,bundle,base_dir=base_dir,require_acceptance=True)
        result['profile_manifest']=manifest
        stage='prospective_observation_plan'
        if set(evidence)!={'plan','observations'}:raise ValueError('Complete bound observation plan and observations required')
        plan,_=_bound(evidence['plan'],base_dir)
        if plan.get('evidence_role')!='actual_disjoint_development' or plan.get('profile_manifest_sha256')!=manifest['sha256']:
            raise ValueError('Prospective plan does not bind actual canonical profiles')
        entries=evidence['observations'];ids=[e['observation_id'] for e in entries]
        if ids!=plan.get('observation_ids') or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicate planned observation, including failures')
        if plan.get('development_resource_lock_sha256')!=digest(manifest['resource_lock']):raise ValueError('Plan development resource lock differs')
        lookup={p['profile_id']:p for p in manifest['profiles']};observed=[];packages=[]
        for entry in entries:
            stage='replay_derived_development_'+entry['observation_id'];profile=lookup[entry['profile_id']]
            derived=replay_derived_package(profile,entry,bundle,context,base_dir=base_dir)
            packages.append((derived['directory'],derived['input_hashes']))
            kind=profile['signature']['kind']
            if kind=='worker':value=worker_observation(profile,entry,derived,bundle,base_dir,manifest['base_policy'])
            elif kind=='graph':value=graph_observation(profile,entry,derived,base_dir,manifest['base_policy'])
            else:value=convex_observation(profile,entry,derived,base_dir,manifest['base_policy'])
            value['observation_id']=entry['observation_id'];observed.append(value)
        stage='common_policy_selection';selection=select_policy(manifest,observed)
        selection['utility_budget_bindings']=utility_budget_bindings(manifest,observed,bundle)
        candidate={**bundle,'execution_policy':selection['execution_policy']}
        for gid,accepted in context['acceptance'].items():
            repeated=dispatch.development_acceptance(candidate,base_dir,accepted['source_replay'])
            if repeated!=accepted['development']:raise ValueError('Final common policy changed frozen development acceptance')
        if codes()!=start:raise ValueError('Bound code changed during qualification')
        if digest(projection(bundle))!=initial_projection:raise ValueError('Bound input configuration changed during qualification')
        for directory,hashes in packages:
            if any(sha(directory/name)!=value for name,value in hashes.items()):raise ValueError('Derived input bytes changed during qualification')
        for entry in entries:
            for ref in entry.values():
                if isinstance(ref,dict) and set(ref)=={'path','sha256'} and sha(_contained(ref['path'],base_dir))!=ref['sha256']:
                    raise ValueError('Bound observation changed during qualification')
        result.update(status='qualified_registered_count_policy_pending_full_review',execution_policy=selection['execution_policy'],
            covered_job_ids=manifest['covered_job_ids'],selection=selection,observations=observed,
            actual_observations_accepted=len(observed),evidence_sha256=digest(evidence),bundle_input_sha256=digest(projection(bundle)),
            source_replay_bindings={gid:v['source_replay']['replay_binding_sha256'] for gid,v in context['acceptance'].items()},
            external_measurement_authenticity_review_required=True,
            qualification_limits=['Source/development replay and exact file consistency do not authenticate historical execution.',
                'Reported RSS is neither total address space nor a simultaneous-process peak guarantee.',
                'Graph density, source overlap, incidence fan-in, snapshots and conditioning remain future cost factors.',
                'Outside-added-cap branches retain their own acceptance and resource limitations.'])
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,ImportError) as error:
        result['blockers'].append({'stage':stage,'exception_type':type(error).__name__,'reason':str(error)})
    result['sha256']=digest(result);return result


def verify_owner_resources(receipt,owner_key,bundle,evidence,*,base_dir='.'):
    if receipt.get('execution_policy') is None:raise ValueError('Blocked resource receipt cannot qualify execution')
    actual=qualify_owner_resources(owner_key,bundle,evidence,base_dir=base_dir)
    if actual!=receipt:raise ValueError('Canonical resource receipt does not reproduce from current actual inputs')
    return actual


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():parser.exit(2,'Preserve existing output directory\n')
    try:
        raw=args.request.read_bytes();request=pf.loads(raw.decode());base=args.request.parent.resolve()
        if set(request)!={'owner_key','bundle','evidence'}:raise ValueError('owner_key, bundle, and evidence required')
        bundle=task.load_bound_value(request['bundle'],base)
        report=qualify_owner_resources(request['owner_key'],bundle,request['evidence'],base_dir=base)
        if args.request.read_bytes()!=raw:raise ValueError('Captured qualification request changed')
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (args.out/'invocation.json').write_text(json.dumps({'request_sha256':pf.sha256(raw),'source_sha256':report['source_sha256']},indent=2)+'\n')
        if report['execution_policy'] is not None:(args.out/'execution_policy.json').write_text(json.dumps(report['execution_policy'],indent=2)+'\n')
        print(json.dumps({'status':report['status'],'policy_exported':report['execution_policy'] is not None,'execution_allowed':False}));return report
    except (ValueError,TypeError,OSError,KeyError) as error:parser.exit(2,'BLOCKED: '+str(error)+'\n')


if __name__=='__main__':main()
