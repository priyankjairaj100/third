"""One complete study ledger over immutable, separately reviewed input roots.

Only routing and accounting are new. Phase 6 acceptance, requests, numerical
gates, state audits, and branch executors retain their frozen implementations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
from phase5 import task_program as task
from phase6 import dispatch, extensions, recipes
from phase7.run_dispatch import validate_policy
from phase7.prepare_calibration import strict_json, file_hash
from phase8 import dossiers

SCHEMA = 'ccu-stable-root-study-1'
DESIGN = ROOT.parent/'output/empirical_program/study_design.json'


def write(path, value, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if replace:
        temp = path.with_name(path.name+'.next'); temp.write_text(raw); temp.replace(path)
    else:
        with path.open('x') as stream: stream.write(raw)


def codes():
    result = dispatch.codes()
    for folder in ('phase7','phase8'):
        for path in sorted((ROOT/folder).glob('*.py')):
            result[str(path.relative_to(ROOT))] = file_hash(path)
    for name in ('study_assembly.py','resource_profiles.py','resource_measurement.py','graph_dossier.py'):
        path = ROOT/'phase9'/name
        if path.exists(): result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def canonical_owners(registry, jobs):
    """Derive one input key per canonical job; no caller-supplied assignment."""
    expected_registry, expected_jobs = recipes.build_registry()
    if registry != expected_registry or jobs != expected_jobs:
        raise ValueError('The complete unchanged canonical registry and jobs are required')
    groups = {g['group_id']:g for g in registry['groups']}; result = {}
    for job in jobs:
        if job.get('group_id') in groups:
            gid = job['group_id']; key = groups[gid]['request_family']
        else:
            gid = None; name = job['recipe_id']
            if name == 'human_threshold_calibration': key = 'human_threshold/'+job['corpus']+'/'+job['encoder']
            elif name == 'human_admission_main': key = 'human_admission'
            elif name == 'human_admission_context': key = 'human_context'
            elif name == 'test_source_bootstrap': key = 'statistics'
            else: key = job['corpus']+'/'+job['panel']
        result[job['job_id']] = {'key':key, 'group_id':gid}
    if len(result) != len(jobs): raise ValueError('Duplicate canonical jobs')
    return result


def _bound_json(descriptor, root, *, loaded=True):
    if not isinstance(descriptor,dict) or set(descriptor)!={'path','sha256'}:
        raise ValueError('File binding requires exactly path and sha256')
    root = Path(root).resolve(); path = (root/descriptor['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Input path escapes its stable root or is absent')
    if file_hash(path) != descriptor['sha256']: raise ValueError('Bound input changed')
    value = dossiers.load_value(path,base=root) if loaded else strict_json(path.read_text())
    if loaded:
        from phase9.graph_dossier import hydrate_human_graphs
        value=hydrate_human_graphs(value)
    if file_hash(path) != descriptor['sha256']: raise ValueError('Bound input changed during loading')
    return path,value


def _selected(spec, ownership, jobs):
    groups = spec.get('group_ids',[]); names = spec.get('extension_recipes',[])
    if not isinstance(groups,list) or not isinstance(names,list) or len(set(groups))!=len(groups) or len(set(names))!=len(names):
        raise ValueError('Unique explicit group and extension-recipe lists required')
    available_groups = {o['group_id'] for o in ownership.values() if o['key']==spec['key'] and o['group_id'] is not None}
    available_names = {j['recipe_id'] for j in jobs if ownership[j['job_id']]['key']==spec['key'] and ownership[j['job_id']]['group_id'] is None}
    if set(groups)-available_groups or set(names)-available_names: raise ValueError('Scope is not owned by this canonical input key')
    return [j['job_id'] for j in jobs if ownership[j['job_id']]['key']==spec['key'] and
            (j.get('group_id') in groups or (ownership[j['job_id']]['group_id'] is None and j['recipe_id'] in names))]


def assemble(config_path, out):
    """Create a complete prospective routing plan; never approve its inputs."""
    config_path=Path(config_path).resolve(); out=Path(out)
    out.mkdir(parents=True,exist_ok=False); (out/'.gitignore').write_text('*\n')
    source=codes(); captured=config_path.read_bytes()
    report={'schema':SCHEMA,'status':'blocked','primary_execution_allowed':False,'source_sha256':source}
    try:
        config=strict_json(captured.decode('utf-8'))
        if set(config)!={'owners'} or not isinstance(config['owners'],list): raise ValueError('Configuration requires owners list only')
        registry,jobs=recipes.build_registry(); ownership=canonical_owners(registry,jobs)
        owners={}; selected=set()
        for entry in config['owners']:
            if set(entry)-{'key','root','bundle','group_ids','extension_recipes','resource'}: raise ValueError('Unknown owner fields')
            key=entry['key']
            if key in owners: raise ValueError('Two owners claim the same input key')
            if key not in {o['key'] for o in ownership.values()}: raise ValueError('Unregistered input key')
            spec=copy.deepcopy(entry); spec['root']=str((config_path.parent/entry['root']).resolve())
            ids=_selected(spec,ownership,jobs)
            if not ids: raise ValueError('Owner must select an explicit nonempty registered scope')
            spec['owned_job_ids']=ids
            # Missing bindings stay visible in the plan; execution replays them.
            try:
                _,bundle=_bound_json(spec['bundle'],spec['root'])
                if spec.get('group_ids') and not set(spec['group_ids'])<=set(bundle.get('_dossier_group_ids',[])):
                    raise ValueError('Selected group scope is absent from the reviewed bundle')
                spec['loaded_bundle_fingerprint']=recipes.digest(task.fingerprint(bundle))
                spec['input_status']='loaded_pending_actual_acceptance'
            except (ValueError,KeyError,OSError,TypeError) as error:
                spec['input_status']='blocked_input'; spec['input_error']=str(error)
            owners[key]=spec; selected.update(ids)
        if source!=codes() or captured!=config_path.read_bytes(): raise ValueError('Bound sources or configuration changed')
        plan={'schema':SCHEMA,'registry_sha256':recipes.digest(registry),'jobs_sha256':recipes.digest(jobs),
              'design_sha256':file_hash(DESIGN),'source_sha256':source,'owners':owners,
              'ownership':ownership,'planned_jobs':len(jobs),'selected_jobs':len(selected),
              'input_approval':False,'path_rule':'Each signed object retains its original literal strings and stable root.'}
        write(out/'plan.json',plan)
        report.update(status='routing_plan_ready',planned_jobs=len(jobs),selected_jobs=len(selected),
                      missing_scope_jobs=len(jobs)-len(selected),plan_sha256=file_hash(out/'plan.json'))
    except (ValueError,KeyError,TypeError,OSError) as error:
        report.update(exception_type=type(error).__name__,reason=str(error))
    write(out/'receipt.json',report); return report


def verify_owner_resources(spec, bundle, registry, jobs):
    """Recompute a separately supplied Phase 9 qualification; never trust a flag."""
    from phase9.resource_profiles import verify_owner_resources as verify
    _,request=_bound_json(spec['resource']['request'],spec['root'])
    _,receipt=_bound_json(spec['resource']['receipt'],spec['root'])
    if bundle.get('resource_policy_qualification',{}).get(spec['key'])!=receipt or bundle.get('resource_policy_observations',{}).get(spec['key'])!=request:
        raise ValueError('Resource receipt and evidence must be attached inside the finally reviewed bundle')
    return verify(receipt,spec['key'],bundle,request,base_dir=spec['root'])


def _review_stage(base,stage,action):
    base=Path(base).resolve()
    if not (base/'.gitignore').is_file() or (base/'.gitignore').read_text()!='*\n':
        raise ValueError('Stable candidate root requires its private output guard')
    out=base/stage;out.mkdir(exist_ok=False);(out/'.gitignore').write_text('*\n')
    source=codes();report={'schema':SCHEMA,'stage':stage,'status':'blocked','primary_execution_allowed':False,'source_sha256':source}
    try:
        result=action(out)
        if source!=codes(): raise ValueError('Review preparation sources changed')
        publish=result.pop('_commit',None)
        if publish is not None: result.update(publish())
        report.update(result)
    except Exception as error: report.update(exception_type=type(error).__name__,reason=str(error),partial_outputs_accepted=False)
    write(out/'receipt.json',report);return report


def _stage_candidate(path,stage,expected_status):
    path=Path(path).resolve();base=path.parent
    receipt=strict_json((base/stage/'receipt.json').read_text())
    if receipt.get('status')!=expected_status or receipt.get('candidate',{}).get('sha256')!=file_hash(path):
        raise ValueError('Candidate is not bound by a successful preparation receipt')
    return base,dossiers.load_value(path,base=base)


def prepare_resource_candidate(candidate_path,attachment_path):
    """Attach supplied development and explicit scope before resource selection."""
    candidate_path,base=dossiers.candidate_root(candidate_path)
    candidate_sha=file_hash(candidate_path);attachment_sha=file_hash(attachment_path)
    def action(out):
        bundle=dossiers.load_value(candidate_path,base=base)
        attachment=dossiers.load_value(attachment_path,base=base)
        if set(attachment)!={'scope','development','external_evidence_review'}:
            raise ValueError('Resource preparation needs scope and supplied development/preliminary review')
        scope=attachment['scope']
        if set(scope)!={'key','group_ids','extension_recipes'}: raise ValueError('Explicit owner scope required')
        registry,jobs=recipes.build_registry();ownership=canonical_owners(registry,jobs)
        selected=_selected(scope,ownership,jobs)
        if not selected or not set(scope['group_ids'])<=set(bundle.get('_dossier_group_ids',[])):
            raise ValueError('Scope exceeds existing candidate groups')
        bundle['phase9_scope']=scope;bundle['development']=attachment['development']
        bundle['phase9_stable_root']=str(base)
        bundle['external_evidence_review']=attachment['external_evidence_review']
        if file_hash(candidate_path)!=candidate_sha or file_hash(attachment_path)!=attachment_sha: raise ValueError('Preparation inputs changed')
        return {'status':'resource_candidate_ready','selected_job_ids':selected,'input_approved':False,
                '_commit':lambda:{'candidate':dossiers.dump_value(bundle,base/'resource_candidate.phase9.json')}}
    return _review_stage(base,'phase9_resource_stage',action)


def prepare_review(candidate_path,attachment_path):
    """Attach actually replayed qualification, then ask for full external review."""
    base,bundle=_stage_candidate(candidate_path,'phase9_resource_stage','resource_candidate_ready')
    candidate_sha=file_hash(candidate_path);attachment_sha=file_hash(attachment_path)
    def action(out):
        from phase9.resource_profiles import verify_owner_resources as verify
        supplied=dossiers.load_value(attachment_path,base=base)
        if set(supplied)!={'qualification','evidence'}: raise ValueError('Qualification and actual evidence required')
        scope=bundle['phase9_scope'];q=supplied['qualification'];evidence=supplied['evidence']
        bundle['execution_policy']=validate_policy(q['execution_policy'])
        verified=verify(q,scope['key'],bundle,evidence,base_dir=base)
        registry,jobs=recipes.build_registry();selected=_selected(scope,canonical_owners(registry,jobs),jobs)
        if verified!=q or not set(selected)<=set(q['covered_job_ids']): raise ValueError('Qualification does not reproduce selected scope')
        bundle['resource_policy_qualification']={scope['key']:q}
        bundle['resource_policy_observations']={scope['key']:evidence}
        first=next(g for g in registry['groups'] if g['group_id'] in scope['group_ids'])
        from phase6.acceptance import accept_source_cache
        replay=accept_source_cache(first,bundle,strict_json(DESIGN.read_text()),base_dir=base)
        write(out/'source_cache_replay.json',replay)
        if replay.get('machine_acceptance_passed') is not True: raise ValueError('Source/cache replay failed')
        archive=dossiers.archive_replay(bundle['source_acceptance'],out/'archive_replay')
        groups=[g for g in registry['groups'] if g['group_id'] in scope['group_ids']]
        request=dossiers._review_request(bundle,groups,replay,archive)
        request['phase9_selected_job_ids']=selected
        if file_hash(candidate_path)!=candidate_sha or file_hash(attachment_path)!=attachment_sha: raise ValueError('Review preparation inputs changed')
        def publish():
            descriptor=dossiers.dump_value(bundle,base/'review_candidate.phase9.json')
            write(base/'final_review_request.phase9.json',request);return {'candidate':descriptor}
        return {'status':'review_candidate_ready','input_approved':False,'_commit':publish}
    return _review_stage(base,'phase9_review_stage',action)


def finalize_review(candidate_path,review_path):
    """Apply only supplied testimony and rerun resource plus frozen input gates."""
    base,bundle=_stage_candidate(candidate_path,'phase9_review_stage','review_candidate_ready')
    candidate_sha=file_hash(candidate_path);review_sha=file_hash(review_path)
    def action(out):
        from phase9.resource_profiles import verify_owner_resources as verify
        scope=bundle['phase9_scope'];bundle['external_evidence_review']=dossiers.load_value(review_path,base=base)
        q=bundle['resource_policy_qualification'][scope['key']];evidence=bundle['resource_policy_observations'][scope['key']]
        if verify(q,scope['key'],bundle,evidence,base_dir=base)!=q: raise ValueError('Resource qualification changed')
        archive=dossiers.archive_replay(bundle['source_acceptance'],out/'archive_replay')
        registry,jobs=recipes.build_registry();groups=[g for g in registry['groups'] if g['group_id'] in scope['group_ids']]
        if not groups: raise ValueError('At least one explicitly selected registered task group required')
        for index,group in enumerate(groups):
            accepted=dispatch.acceptance(group,bundle,strict_json(DESIGN.read_text()),mode=dispatch.PRIMARY_MODE,base_dir=base)
            write(out/f'acceptance_{index:03d}.json',accepted)
            if accepted.get('primary_inputs_accepted') is not True: raise ValueError('Frozen final group input acceptance failed')
        if file_hash(candidate_path)!=candidate_sha or file_hash(review_path)!=review_sha: raise ValueError('Final review inputs changed')
        return {'status':'selected_group_inputs_accepted_execution_still_rechecks',
                '_commit':lambda:{'candidate':dossiers.dump_value(bundle,base/'bundle.phase9.final.json')},
                'scope':scope,'archive_replay':archive,'input_approval_scope':'Declared task groups only; every extension rechecks its own frozen gate.'}
    return _review_stage(base,'phase9_final_stage',action)


def _row(job, status, **extra):
    return {'job_id':job['job_id'],'group_id':job.get('group_id'),'status':status,
            'methods':[{'method':m,'status':status} for m in job.get('methods',[])],
            'primary_output_accepted':False,**extra}


def _finish(row, job=None):
    if job is not None and not row.get('methods'):
        row['methods']=[{'method':m,'status':row['status']} for m in job.get('methods',[])]
    for method in row.get('methods',[]):
        if method.get('status') in {'planned','not_started'}: method['status']=row['status']
    return row


def execute_group(group,jobs,bundle,out,*,base_dir,policy,design,record):
    """Frozen group gates and branch functions, under this group's stable root."""
    out=Path(out); out.mkdir(parents=True,exist_ok=False); scratch=None
    completed=set()
    try:
        if bundle.get('execution_policy')!=policy: raise ValueError('Policy differs from the reviewed bundle')
        accepted=dispatch.acceptance(group,bundle,design,mode=dispatch.PRIMARY_MODE,base_dir=base_dir)
        write(out/'acceptance.json',accepted)
        if not accepted.get('execution_allowed') or not accepted.get('primary_inputs_accepted'):
            raise ValueError('Frozen primary input acceptance failed')
        cell=dispatch.resolve(group,bundle); manifest=dispatch.request_for_group(group,bundle,cell)
        cell['_validated_source_bindings']=accepted.get('validated_source_bindings')
        if group['configuration'].get('threshold','calibrated')!='calibrated' and not cell['geometry'].get('threshold_quality',{}).get('quality_pass'):
            raise ValueError('Separate threshold quality gate failed')
        write(out/'requests.json',manifest); write(out/'geometry.json',cell['geometry'])
        paths={p['trajectory_id']:p for p in manifest['trajectories']}
        if group['block']=='E_full_refit':
            rows,summary=dispatch.full_refit_group(group,jobs,bundle,cell,out,primary=True)
            if len(rows)!=len(jobs) or {r['job_id'] for r in rows}!={j['job_id'] for j in jobs}:
                raise ValueError('Full-refit branch changed the complete owned job set')
            write(out/'branch_summary.json',summary)
            for row in rows: record(_finish(row)); completed.add(row['job_id'])
            return
        if set(group['methods'])<=dispatch.KNOWN_METHODS and cell['x'] is not None:
            from phase6 import run_isolated
            (ROOT/'tmp').mkdir(exist_ok=True)
            scratch=tempfile.TemporaryDirectory(prefix='phase9_group_',dir=ROOT/'tmp')
            package=run_isolated.prepare_bundle(cell['cx'],cell['x'],cell['y'],bundle['train_ids'],Path(scratch.name)/'inputs',
                threshold=cell['graph'].threshold,source_ids=bundle['source_ids'],priority_seed=cell['priority_seed'])
            cell['_prepared_bundle_path']=str(package)
            write(out/'shared_bundle.json',strict_json((package/'bundle.json').read_text()))
            write(out/'shared_preparation.json',{'scope':'one input package per group; time and bytes in shared_bundle.json',
                'charged_once':True,'repair_service_cost':False})
        for index,job in enumerate(jobs):
            target=out/f'job_{index:04d}'; target.mkdir(); row=_row(job,'planned')
            try:
                path=paths[job['trajectory_id']]; dispatch._check_path(job,path,bundle)
                row.update(actual_horizon=len(path['deletion_order']),actual_checkpoint_units=path['checkpoints'])
                if not path['checkpoints']:
                    row.update(status='structural_zero_or_unavailable_arm',reason=path['status'],checkpoint_count=0)
                elif group['methods']==['independent_structure_oracle']:
                    row.update(dispatch.structural_job(job,path,cell,target))
                elif group['block']=='F_convex': row.update(dispatch.convex_job(job,path,group,bundle,cell,target,policy))
                elif group['variant']=='ANN':
                    approximate,audit=dispatch.boundary.audit_ann(cell['cx'],cell['graph'],**bundle['ANN_policy']); comparisons=[]
                    for k in path['checkpoints']:
                        dead=dispatch.boundary.deleted_indices(path,k,cell['graph']); deleted=[cell['graph'].record_ids[i] for i in dead]
                        exact=set(map(int,cell['graph'].selected_indices(deleted))); approx=set(map(int,approximate.selected_indices(deleted)))
                        comparisons.append({'checkpoint':k,'selected_count':len(exact),'approximate_selected_count':len(approx),
                                            'selected_symmetric_difference':len(exact^approx)})
                    audit['checkpoints']=comparisons; write(target/'ANN.json',audit)
                    row.update(status='completed',artifact='ANN.json',checkpoint_count=len(comparisons))
                elif set(group['methods'])<=dispatch.KNOWN_METHODS and cell['x'] is not None:
                    row.update(dispatch.method_job(job,path,group,bundle,cell,target,policy))
                else: raise NotImplementedError('No frozen executor for this explicit group')
                row['primary_output_accepted']=row['status']=='completed'
            except Exception as error:
                row.update(status='blocked_missing_input' if isinstance(error,(KeyError,FileNotFoundError)) else
                           'unsupported_executor' if isinstance(error,NotImplementedError) else 'failed',
                           exception_type=type(error).__name__,message=str(error),primary_output_accepted=False)
            row=_finish(row); write(target/'outcome.json',row); record(row); completed.add(job['job_id'])
    except Exception as error:
        for job in jobs:
            if job['job_id'] not in completed:
                record(_row(job,'blocked_missing_input' if isinstance(error,(KeyError,FileNotFoundError)) else 'input_acceptance_failed',
                            exception_type=type(error).__name__,message=str(error)))
    finally:
        if scratch is not None: scratch.cleanup()


class RootedDispatch:
    """Read-only facade; dependency acceptance keeps each source's own root."""
    def __init__(self,loaded): self.loaded=loaded
    def __getattr__(self,name): return getattr(dispatch,name)
    def acceptance(self,group,bundle,design,*,mode,base_dir):
        key=group['request_family']
        if key not in self.loaded or self.loaded[key]['bundle'] is not bundle:
            raise ValueError('Missing or substituted dependency owner')
        owner=self.loaded[key]
        if not owner['resource_verified']: raise ValueError('Dependency resource qualification failed')
        return dispatch.acceptance(group,bundle,design,mode=mode,base_dir=owner['spec']['root'])


def validate_statistics_dispatch(dossier,base_dir):
    """Statistics consume only a final complete ledger, never provisional rows."""
    path,summary=_bound_json(dossier['phase9_dispatch_evidence'],base_dir,loaded=False)
    if path.name!='summary.json' or summary.get('schema')!=SCHEMA or summary.get('status')!='completed_dispatch' or summary.get('inputs_and_sources_unchanged') is not True:
        raise ValueError('Statistics require an unchanged successful Phase 9 dispatch summary')
    ledger_path=path.parent/'jobs.json'
    if file_hash(ledger_path)!=summary.get('final_jobs_sha256'): raise ValueError('Final complete ledger differs from its summary')
    rows=strict_json(ledger_path.read_text());_,jobs=recipes.build_registry()
    if len(rows)!=len(jobs) or [r['job_id'] for r in rows]!=[j['job_id'] for j in jobs]:
        raise ValueError('Statistics require the entire canonical final ledger in order')
    for entry in dossier['job_outcomes']:
        descriptor=entry['outcome'];bound=(Path(base_dir)/descriptor['path']).resolve()
        if bound!=ledger_path or descriptor['sha256']!=summary['final_jobs_sha256']:
            raise ValueError('All analysis outcomes must bind the final complete ledger, not provisional per-job files')
    return {'summary_sha256':file_hash(path),'final_jobs_sha256':summary['final_jobs_sha256'],'planned_jobs':len(jobs)}


def execute(plan_path,out):
    """Run only explicitly owned canonical cells; retain every other planned row."""
    began=time.perf_counter();plan_path=Path(plan_path).resolve(); out=Path(out)
    out.mkdir(parents=True,exist_ok=False); (out/'.gitignore').write_text('*\n')
    registry,jobs=recipes.build_registry(); canonical=canonical_owners(registry,jobs)
    ledger={j['job_id']:_row(j,'not_started') for j in jobs}; order=[j['job_id'] for j in jobs]
    sources=codes(); attempted=set(); finalized=set(); loaded={}; caches={}; summary={'schema':SCHEMA,'status':'blocked'}
    write(out/'jobs.json',[ledger[k] for k in order])
    def record(row):
        jid=row['job_id']
        if jid not in ledger or jid in finalized: raise ValueError('Unplanned or duplicate job result')
        ledger[jid]=row; finalized.add(jid)
        with (out/'journal.jsonl').open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
    try:
        captured=plan_path.read_bytes(); plan=strict_json(captured.decode('utf-8'))
        import hashlib
        plan_hash=hashlib.sha256(captured).hexdigest()
        if plan.get('schema')!=SCHEMA or plan.get('ownership')!=canonical or plan.get('registry_sha256')!=recipes.digest(registry) or plan.get('jobs_sha256')!=recipes.digest(jobs):
            raise ValueError('Plan is not bound to the complete canonical schedule')
        if plan.get('source_sha256')!=sources or plan.get('design_sha256')!=file_hash(DESIGN): raise ValueError('Plan sources or design changed')
        design=strict_json(DESIGN.read_text()); write(out/'plan.json',plan)
        for key,spec in plan['owners'].items():
            owner_began=time.perf_counter()
            owner={'spec':spec,'resource_verified':False}; loaded[key]=owner
            try:
                if key!=spec['key'] or spec['owned_job_ids']!=_selected(spec,canonical,jobs): raise ValueError('Owner scope changed')
                path,bundle=_bound_json(spec['bundle'],spec['root']); owner.update(bundle=bundle,path=path,before=recipes.digest(task.fingerprint(bundle)))
                if owner['before']!=spec.get('loaded_bundle_fingerprint'): raise ValueError('Loaded reviewed object differs from plan')
                if bundle.get('phase9_stable_root')!=spec['root']: raise ValueError('Plan root differs from the root inside the reviewed object')
                if not set(spec.get('group_ids',[]))<=set(bundle.get('_dossier_group_ids',[])): raise ValueError('Reviewed group scope differs')
                if spec.get('group_ids') or not key.startswith(('human_','statistics')):
                    scope=bundle.get('phase9_scope',{})
                    if scope.get('key')!=key or not set(spec.get('group_ids',[]))<=set(scope.get('group_ids',[])) or not set(spec.get('extension_recipes',[]))<=set(scope.get('extension_recipes',[])):
                        raise ValueError('Selected ownership exceeds the finally reviewed Phase 9 scope')
                    policy=validate_policy(bundle['execution_policy']); verified=verify_owner_resources(spec,bundle,registry,jobs)
                    if verified.get('execution_policy')!=policy or not set(spec['owned_job_ids'])<=set(verified.get('covered_job_ids',[])):
                        raise ValueError('Qualification does not cover exact owned jobs and policy')
                    owner.update(policy=policy,resource_verified=True,resource_receipt=verified)
                else:
                    owner.update(policy=None,resource_verified=True)
                owner['status']='loaded_pending_frozen_acceptance'
            except Exception as error: owner.update(status='blocked_owner_input',reason=str(error),exception_type=type(error).__name__)
            owner['input_loading_and_resource_replay_seconds']=time.perf_counter()-owner_began
        write(out/'owners.json',{k:{a:v for a,v in o.items() if a not in {'bundle','before','path'}} for k,o in loaded.items()})
        for gi,group in enumerate(registry['groups']):
            gjobs=[j for j in jobs if j.get('group_id')==group['group_id']]; key=group['request_family']; owner=loaded.get(key)
            owned=owner is not None and group['group_id'] in owner['spec'].get('group_ids',[])
            if not owned or owner['status']!='loaded_pending_frozen_acceptance':
                for job in gjobs: record(_row(job,'blocked_unowned_scope' if not owned else 'blocked_owner_input',owner_key=key))
                continue
            for job in gjobs:
                if job['job_id'] in attempted: raise ValueError('Duplicate execution attempt')
                attempted.add(job['job_id'])
            execute_group(group,gjobs,owner['bundle'],out/f'group_{gi:03d}',base_dir=owner['spec']['root'],
                          policy=owner['policy'],design=design,record=record)
            write(out/'jobs.json',[ledger[k] for k in order],replace=True)
        facade=RootedDispatch(loaded)
        bundles={k:o['bundle'] for k,o in loaded.items() if o['status']=='loaded_pending_frozen_acceptance'}
        for job in jobs:
            if job['job_id'] in finalized: continue
            key=canonical[job['job_id']]['key']; owner=loaded.get(key)
            if owner is None or job['job_id'] not in owner['spec']['owned_job_ids']:
                record(_row(job,'blocked_unowned_scope',owner_key=key)); continue
            if owner['status']!='loaded_pending_frozen_acceptance':
                record(_row(job,'blocked_owner_input',owner_key=key)); continue
            if job['job_id'] in attempted: raise ValueError('Duplicate execution attempt')
            attempted.add(job['job_id']); target=out/'extensions'/recipes.digest(job['job_id'])[:20]; target.mkdir(parents=True)
            try:
                cache=caches.setdefault(key,{})
                if key=='statistics':
                    write(target/'final_dispatch_binding.json',validate_statistics_dispatch(owner['bundle'],owner['spec']['root']))
                row=extensions.run_extension(job,bundles,target,{'dispatch_module':facade,'policy':owner['policy'],
                    'mode':dispatch.PRIMARY_MODE,'design':design,'base_dir':owner['spec']['root'],'extension_cache':cache})
                if row.get('job_id')!=job['job_id']: raise ValueError('Extension job identity changed')
            except Exception as error:
                row=_row(job,'blocked_recipe_executor_input' if isinstance(error,(KeyError,FileNotFoundError,ImportError)) else 'extension_failure',
                         exception_type=type(error).__name__,message=str(error))
            row=_finish(row,job); write(target/'outcome.json',row); record(row)
        unchanged=sources==codes() and file_hash(plan_path)==plan_hash
        for owner in loaded.values():
            if 'bundle' in owner:
                unchanged &= recipes.digest(task.fingerprint(owner['bundle']))==owner['before'] and file_hash(owner['path'])==owner['spec']['bundle']['sha256']
        if not unchanged:
            for row in ledger.values():
                if row.get('primary_output_accepted'): row.update(primary_output_accepted=False,acceptance_revoked='Bound inputs or source changed')
        summary.update(status='completed_dispatch' if unchanged else 'completed_acceptance_revoked',inputs_and_sources_unchanged=unchanged)
    except Exception as error:
        summary.update(exception_type=type(error).__name__,reason=str(error))
        for job in jobs:
            if job['job_id'] not in finalized: record(_row(job,'blocked_coordinator_failure',message=str(error)))
        for row in ledger.values(): row['primary_output_accepted']=False
    finally:
        for cache in caches.values():
            for temp in cache.get('_temporary_directories',[]): temp.cleanup()
    summary.update(planned_jobs=len(jobs),observed_jobs=len(ledger),every_planned_job_retained=len(finalized)==len(jobs),
        attempted_jobs=len(attempted),status_counts=dict(Counter(r['status'] for r in ledger.values())),
        primary_outputs_accepted=sum(bool(r.get('primary_output_accepted')) for r in ledger.values()),
        source_sha256=sources,source_sha256_at_end=codes(),external_testimony_authenticated_by_code=False,
        numerical_and_branch_executors='unchanged Phase 6 functions',duplicate_execution_attempts=0)
    summary['coordinator_total_wall_seconds']=time.perf_counter()-began
    summary['input_replay_cost_scope']='Included in coordinator total; per-owner loading/qualification times in owners.json; no repair-time or peak-memory claim.'
    write(out/'jobs.json',[ledger[k] for k in order],replace=True)
    summary['final_jobs_sha256']=file_hash(out/'jobs.json')
    summary['raw_job_artifacts_are_provisional']=True
    write(out/'summary.json',summary); return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('assemble'); a.add_argument('--config',type=Path,required=True); a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('execute'); a.add_argument('--plan',type=Path,required=True); a.add_argument('--out',type=Path,required=True)
    for name in ('prepare-resource','prepare-review','finalize-review'):
        a=sub.add_parser(name);a.add_argument('--candidate',type=Path,required=True)
        a.add_argument('--review' if name=='finalize-review' else '--attachment',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='assemble': result=assemble(args.config,args.out)
    elif args.command=='execute': result=execute(args.plan,args.out)
    elif args.command=='prepare-resource': result=prepare_resource_candidate(args.candidate,args.attachment)
    elif args.command=='prepare-review': result=prepare_review(args.candidate,args.attachment)
    else: result=finalize_review(args.candidate,args.review)
    print(json.dumps(result,allow_nan=False)); return result


if __name__=='__main__': main()
