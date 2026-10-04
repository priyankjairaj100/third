"""Versioned study dispatcher with explicit per-job and per-method outcomes.

The dispatcher executes algorithms. External provenance remains an explicit
input to source acceptance. It never infers human collection from file presence.
"""
from __future__ import annotations
from pathlib import Path
import argparse, copy, gzip, hashlib, json, math, random, shutil, sys, tempfile, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from phase3 import calibration as cal
from phase3.reference_graph import build_reference_graph
from phase4 import requests as requests_module
from phase4.execution import array_hash, independent_edges
from phase5 import task_program as task, boundary, decoders, refit, convex_program
from phase5.study_registry import digest, build_registry
from ccu.core import ridge_moments

SCHEMA='ccu-study-dispatch-1'
PRIMARY_MODE='accepted_primary_inputs'
ENGINEERING_MODE='natural_text_engineering'
KNOWN_METHODS={'O-G','O-G-cached','O-T','B-E','B-A','P-I','P-S','P-R','B-F','P-I-jointspan','B-E-compact'}

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def codes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('ccu','phase3','phase4','phase5','phase6')
            for p in sorted((ROOT/folder).glob('*.py'))}

def dual_head(x,y,selected,lam):
    """Independent dimension-adaptive solve. No production decoder or moments."""
    i=np.asarray(sorted(selected),dtype=int)
    if not len(i):return np.zeros((x.shape[1],y.shape[1]),np.float64)
    z=x[i].astype(np.float64);target=y[i]
    if len(i)<=x.shape[1]:
        return z.T@np.linalg.solve(z@z.T+float(lam)*len(i)*np.eye(len(i)),target)
    return np.linalg.solve(z.T@z+float(lam)*len(i)*np.eye(x.shape[1]),z.T@target)

def saved_head_gate(x,y,selected,lam,actual):
    """Evaluate the saved head itself. Never replace it with a new solver output."""
    actual=np.asarray(actual,np.float64);ix=np.asarray(sorted(selected),dtype=int)
    if actual.shape!=(x.shape[1],y.shape[1]) or not np.isfinite(actual).all():
        return {'passed':False,'reason':'invalid saved head'}
    expected=dual_head(x,y,selected,lam);err=float(np.linalg.norm(actual-expected))
    if not len(ix):return {'passed':bool(np.count_nonzero(actual)==0),'head_frobenius_error':err,'normalized_residual_eta':0. if np.count_nonzero(actual)==0 else None,'empty_target':True}
    z=x[ix].astype(np.float64);gram=z.T@z;cross=z.T@y[ix]
    matrix=gram+float(lam)*len(ix)*np.eye(x.shape[1]);opnorm=float(np.linalg.eigvalsh(matrix)[-1])
    residual=decoders._residual(matrix,actual,cross,opnorm)
    return {**residual,'head_frobenius_error':err,'passed':bool(residual['normalized_residual_eta']<=1e-10 and err<=1e-9*max(1.,float(np.linalg.norm(expected)))),
        'checked_actual_saved_head':True,'interval_certificate':False}

def independent_selection(cell,dead,*,fresh=False):
    """Build one independent exhaustive adjacency. Reuse it on retained IDs."""
    ids=list(cell['graph'].record_ids);seed=cell['priority_seed'];tau=cell['graph'].threshold
    if fresh:
        _,selected=independent_edges(cell['cx'],ids,tau,[ids[i] for i in dead],seed=seed)
        return set(selected)
    if '_independent_blockers' not in cell:
        started=time.perf_counter();x=np.asarray(cell['cx']);n,d=x.shape
        norms=np.zeros(n,np.float64)
        for k in range(d):
            coordinate=x[:,k].astype(np.float64);norms+=coordinate*coordinate
        norms=np.sqrt(norms)
        priority=sorted(range(n),key=lambda i:(hashlib.sha256(f'priority-v1|{seed}|{ids[i]}'.encode()).digest(),ids[i]))
        rank=np.empty(n,np.int64);rank[priority]=np.arange(n)
        blockers=[[] for _ in range(n)];block=64
        for begin in range(0,n,block):
            end=min(n,begin+block);left=x[begin:end].astype(np.float64)/norms[begin:end,None]
            for other in range(begin,n,block):
                stop=min(n,other+block);right=x[other:stop].astype(np.float64)/norms[other:stop,None]
                score=np.zeros((end-begin,stop-other),np.float64)
                for k in range(d):score+=left[:,k,None]*right[None,:,k]
                for a,b in np.argwhere(score>tau):
                    i,j=begin+int(a),other+int(b)
                    if i>=j:continue
                    child,parent=(i,j) if rank[i]>rank[j] else (j,i)
                    blockers[child].append(parent)
        pointers=np.cumsum([0]+[len(v) for v in blockers],dtype=np.int64)
        indices=np.asarray([i for row in blockers for i in sorted(row)],np.int64)
        cell['_independent_blockers']=(pointers,indices)
        cell['_independent_graph_seconds']=time.perf_counter()-started
        cell['_independent_graph_bytes']=pointers.nbytes+indices.nbytes
    pointers,indices=cell['_independent_blockers'];gone=np.zeros(len(ids),bool)
    if dead:gone[np.asarray(sorted(dead),dtype=int)]=True
    selected={i for i in range(len(ids)) if not gone[i] and bool(np.all(gone[indices[pointers[i]:pointers[i+1]]]))}
    return selected

def recompute_calibration(entry,representation):
    """Reconstruct both sampling manifests, then recompute their response gates."""
    lock=entry['threshold_lock']
    if lock.get('schema')==task.ENGINEERING_LOCK:raise ValueError('Engineering threshold cannot pass primary acceptance')
    d=entry['calibration'];selection=d['selection_manifest'];validation=d['validation_manifest']
    expected=cal.prepare_selection(d['features'],d['records'],d['provenance'],
        seed=selection['seed'],block_size=selection['frame']['scorer']['block_size'])
    if expected!=selection:raise ValueError('Selection pairs do not reconstruct from calibration features')
    selected=cal.select_threshold(expected,d['selection_responses'])
    if selected!=lock:raise ValueError('Threshold lock differs from recomputed human response selection')
    expected_validation=cal.prepare_validation(d['features'],d['records'],selected,
        seed=validation['seed'],block_size=validation['frame']['scorer']['block_size'])
    if expected_validation!=validation:raise ValueError('Fresh validation sample does not reconstruct')
    gate=cal.quality_gate(expected_validation,d['validation_responses'],selected)
    if gate!=d['quality_report'] or not gate['statistical_and_declared_scope_eligible']:
        raise ValueError('Independent quality response gate failed')
    actual,_=task._parameter(entry,'threshold',representation)
    return {'threshold':actual,'selection_sha256':selection['sha256'],
            'validation_sha256':validation['sha256'],'quality_sha256':gate['sha256'],
            'genuine_human_collection_authenticated_by_software':False}

def recompute_requests(bundle,graph,*,primary=False):
    """Regenerate complete orders, checkpoints, matched controls and observations."""
    supplied=bundle['requests'];requests_module.verify_manifest(supplied,graph)
    cfg=supplied['configuration'];design=bundle.get('request_design')
    if design is None:raise KeyError('request_design required for deterministic request reconstruction')
    if digest(design)!=supplied['input_design_sha256']:raise ValueError('Request design does not match its manifest')
    expected=requests_module.generate_manifest(graph,dataset_id=supplied['dataset_id'],
        panel_id=supplied['panel_id'],source_kinds=bundle['source_kinds'],design=design,
        allocations=cfg['allocations'],evidence_role=supplied['evidence_role'],
        master_seed=cfg['master_seed'],record_horizon=cfg['requests']['record_horizon'],
        source_horizon=cfg['requests']['source_horizon'],
        include_excluded_blocker_stress=cfg['include_excluded_blocker_stress'])
    if expected!=supplied:raise ValueError('Frozen requests do not regenerate exactly')
    if primary and supplied['evidence_role']!='prospective_confirmatory_preparation':
        raise ValueError('Primary requests require prospective preparation role')
    return {'manifest_sha256':supplied['manifest_sha256'],'paths':len(supplied['trajectories']),
            'recomputed':True}

def acceptance(group,bundle,design,*,mode,base_dir):
    """Return a real gate. Successful source acceptance can activate supplied inputs."""
    checks=[];replay=None
    def run(name,fn):
        try:
            value=fn();checks.append({'name':name,'passed':True,'detail':value});return value
        except Exception as e:
            checks.append({'name':name,'passed':False,'exception_type':type(e).__name__,'message':str(e)});return None
    if mode not in {ENGINEERING_MODE,PRIMARY_MODE}:raise ValueError('Unknown dispatch mode')
    if bundle.get('dataset_id')!=group['corpus']:
        checks.append({'name':'corpus_binding','passed':False,'message':'Corpus and bundle differ'})
    if mode==PRIMARY_MODE:
        # Import the versioned real-asset replayer, never the always-false Phase4 inspector.
        def source_accept():
            from phase6.acceptance import accept_source_cache
            report=accept_source_cache(group,bundle,design,base_dir=base_dir)
            if not report.get('machine_acceptance_passed'):raise ValueError(json.dumps(report,sort_keys=True))
            return report
        replay=run('replayed_source_and_cache_acceptance',source_accept)
        run('external_evidence_trust_record',lambda:external_review(bundle,replay,base_dir))
        run('development_resource_and_precision',lambda:development_acceptance(bundle,base_dir,replay))
        for name,entry in bundle.get('curators',{}).items():
            run('calibration.'+name,lambda e=entry,n=name:recompute_calibration(e,n))
        for name,entry in bundle.get('learners',{}).items():
            def regularization(e=entry,n=name):
                if e['lambda_lock'].get('schema')==task.ENGINEERING_LOCK:
                    raise ValueError('Engineering lambda cannot pass primary acceptance')
                value,seal=task._parameter(e,'lambda',n)
                return {'lambda':value,'lock_sha256':seal}
            run('calibration_lambda.'+name,regularization)
        def primary_request_contract():
            supplied=bundle['requests'];cfg=supplied['configuration']
            if bundle['request_design']!=design:raise ValueError('Primary request design differs from the authoritative study design')
            if supplied['dataset_id']!=group['corpus'] or supplied['panel_id']!=group['panel']:
                raise ValueError('Primary request population identity differs from the registered group')
            if cfg['requests']!=design['requests'] or cfg['master_seed']!=design['seeds']['master']:
                raise ValueError('Primary horizons, checkpoints, or seed differ from the authoritative schedule')
            for arm,count in group.get('allocations',{}).items():
                if cfg['allocations'].get(arm,0)<count:raise ValueError('Frozen request allocation is smaller than the registered group')
            return {'request_design_sha256':digest(design),'panel_id':supplied['panel_id'],'schedule_recomputed':True}
        run('authoritative_primary_request_schedule',primary_request_contract)
        def configuration_contract():
            from phase6.recipes import build_recipe_book
            policies=build_recipe_book()['supporting_policies']
            if group['variant']=='ANN':
                expected={k:policies['ANN'][k] for k in ('tables','bits','seed','sample_size','sample_seed')}
                if bundle.get('ANN_policy')!=expected:raise ValueError('ANN configuration differs from the fixed recipe')
            if group['block']=='E_full_refit':
                expected=policies['refit_configuration'];actual=bundle['refit_configuration']
                for key in ('clusters','iterations','seed','policy','spherical','threads'):
                    if actual.get(key)!=expected[key]:raise ValueError('Refit configuration differs: '+key)
                tau,_=task._parameter(bundle['curators']['e5'],'threshold','e5')
                if actual.get('backend')!=refit.OFFICIAL or actual.get('epsilon')!=1-float(tau):
                    raise ValueError('Refit backend or epsilon differs from the fixed recipe')
                if bundle.get('refit_seed_variants')!=expected['alternative_seeds']:raise ValueError('Refit seed variants differ')
            return {'recipe_policies_sha256':digest(policies),'checked_group_variant':group['variant']}
        run('prospective_algorithm_configuration',configuration_contract)
        if group['block']=='F_convex' and bundle.get('task')=='multilabel':
            run('separate_logistic_tag_decision_rule',lambda:logistic_rule(bundle,group['configuration']['learner_encoder']))
    else:
        run('explicit_engineering_scope',lambda: _engineering_scope(bundle))
    labeled=bundle.get('task') in {'civil','multilabel'}
    if labeled:
        validated=run('task_populations_and_parameters',lambda: _validated_task(bundle))
        graph=None if validated is None else _base_graph(bundle)
    else:graph=run('structure_population',lambda:_base_graph(bundle))
    # Graph arrays are held only in this process; do not serialize them in reports.
    if checks and isinstance(checks[-1].get('detail'),type(graph)) and graph is not None and not labeled:
        checks[-1]['detail']={'records':len(graph.record_ids),'graph_binding':requests_module.graph_binding(graph)}
    if graph is not None:run('recomputed_requests',lambda:recompute_requests(bundle,graph,primary=mode==PRIMARY_MODE))
    allowed=bool(checks) and all(c['passed'] for c in checks)
    return {'schema':'ccu-dispatch-acceptance-1','group_id':group['group_id'],'mode':mode,
        'execution_allowed':allowed,'primary_inputs_accepted':allowed and mode==PRIMARY_MODE,
        'external_provenance_authenticated_by_software':False,'checks':checks,
        'validated_source_bindings':None if replay is None else replay.get('validated_bindings'),
        'input_bindings':task.fingerprint(bundle)}

def external_review(bundle,replay,base_dir):
    """Validate a supplied trust record. Software does not prove its testimony."""
    from datetime import datetime
    if replay is None:raise ValueError('Machine replay must pass before external review binding')
    review=bundle['external_evidence_review'];cal._verify(review)
    if review.get('schema')!='ccu-external-evidence-review-1':raise ValueError('Versioned external review required')
    if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():raise ValueError('Named accountable reviewer required')
    stamp=datetime.fromisoformat(review['completed_at'].replace('Z','+00:00'))
    if stamp.tzinfo is None:raise ValueError('Review time needs a timezone')
    inputs={k:v for k,v in bundle.items() if k!='external_evidence_review'}
    if review.get('reviewed_bundle_sha256')!=digest(task.fingerprint(inputs)):
        raise ValueError('Review is not bound to these exact input arrays and records')
    if review.get('replay_binding_sha256')!=replay.get('replay_binding_sha256'):
        raise ValueError('External review does not bind the exact source/cache replay')
    required={'original_archive_identity','native_source_meaning_and_coverage','permissions_and_content_use',
              'encoder_revision_origin','genuine_independent_human_collection','development_only_configuration'}
    if set(review.get('evidence',{}))!=required:raise ValueError('Every external trust question needs a referenced evidence file')
    base=Path(base_dir).resolve();receipts={}
    for name,item in review['evidence'].items():
        if item.get('decision')!='accepted' or not isinstance(item.get('rationale'),str) or not item['rationale'].strip():
            raise ValueError('External reviewer has not accepted '+name)
        p=(base/item['path']).resolve()
        if not p.is_relative_to(base) or not p.is_file() or p.stat().st_size==0 or task.file_hash(p)!=item['sha256']:
            raise ValueError('External evidence file missing or changed: '+name)
        receipts[name]=item['sha256']
    return {'review_sha256':review['sha256'],'reviewer':review['reviewer'],'evidence_sha256':receipts,
        'decision_source':'accountable supplied external review; not machine authentication of testimony'}

def development_acceptance(bundle,base_dir,replay):
    """Recompute the resource formula and the declared development precision analysis."""
    from phase6 import recipes
    if replay is None:raise ValueError('Development must bind to the actual source replay')
    dossier=bundle['development'];base=Path(base_dir).resolve();observed=[]
    from phase6.acceptance import _path,accept_source_cache
    from phase3.run_preparation import read_rows
    # The preparation file has already been reconstructed from the original
    # parser. Development records must be exact members of that accepted file.
    source_bundle=dossier.get('source_bundle',bundle)
    if source_bundle is not bundle:
        pilot_replay=accept_source_cache(dossier['source_group'],source_bundle,
            json.loads((ROOT.parent/'output/empirical_program/study_design.json').read_text()),base_dir=base)
        if not pilot_replay['machine_acceptance_passed']:raise ValueError('Development source/cache replay failed')
    else:pilot_replay=replay
    if bundle['external_evidence_review'].get('development_source_replay_sha256')!=pilot_replay['replay_binding_sha256']:
        raise ValueError('External development review does not bind the actual pilot source/cache replay')
    if source_bundle.get('task') not in {'civil','multilabel'}:
        raise ValueError('Resource pilots require an accepted labeled source bundle; unlabeled corpora must supply that separate bundle')
    prepared=read_rows(_path(source_bundle['source_acceptance']['prepared']['records'],base))
    lookup={r['record_id']:r for r in prepared}
    development_rows=dossier['records']
    if not development_rows or len({r['record_id'] for r in development_rows})!=len(development_rows):
        raise ValueError('Distinct actual development records are required')
    if any(lookup.get(r['record_id'])!=r for r in development_rows):
        raise ValueError('Development record differs from the replayed original preparation')
    development_sources=sorted({r['source_unit_id'] for r in development_rows})
    if development_sources!=sorted(dossier['development_source_ids']):
        raise ValueError('Development source IDs do not derive from accepted original records')
    tested=replay['validated_bindings']
    confirmation_sources=set(tested['source_ids']+tested['evaluation_source_ids'])
    confirmation_records=set(tested['train_ids']+tested['evaluation_ids']+tested['calibration_ids'])
    pilot_tested=pilot_replay['validated_bindings']
    confirmation_sources.update(pilot_tested['source_ids']+pilot_tested['evaluation_source_ids'])
    confirmation_records.update(pilot_tested['train_ids']+pilot_tested['evaluation_ids']+pilot_tested['calibration_ids'])
    if confirmation_sources & set(development_sources) or confirmation_records & {r['record_id'] for r in development_rows}:
        raise ValueError('Replayed development records or sources overlap confirmation or calibration')
    if bundle.get('dataset_id')=='wcep100':
        news=bundle['inherited_news_bundle'];news_ids=set(news['train_ids']+news.get('evaluation_ids',[]))
        for entry in news.get('curators',{}).values():news_ids.update(r['record_id'] for r in entry['calibration']['records'])
        if news_ids & {r['record_id'] for r in development_rows} or set(news['source_ids']+news.get('evaluation_source_ids',[])) & set(development_sources):
            raise ValueError('Development overlaps the inherited News study')
    # Link actual worker input files to exact accepted semantic cache rows.
    input_dir=_path(dossier['input_bundle_directory'],base)
    input_manifest=json.loads((input_dir/'bundle.json').read_text());metadata=json.loads((input_dir/'metadata.json').read_text())
    pilot_ids=[r['record_id'] for r in development_rows]
    if metadata['record_ids']!=pilot_ids or metadata['source_ids']!=[r['source_unit_id'] for r in development_rows]:
        raise ValueError('Pilot worker row/source IDs differ from accepted development records')
    raw_lookup={r['record_id']:i for i,r in enumerate(prepared)};ix=np.asarray([raw_lookup[r] for r in pilot_ids],dtype=int)
    encoder=dossier.get('encoder','e5')
    cache_dir=_path(source_bundle['source_acceptance']['caches'][encoder]['directory'],base)
    accepted_cache=np.load(cache_dir/'vectors.npy',mmap_mode='r',allow_pickle=False)
    for name in ('curator.npy','learner.npy'):
        if not np.array_equal(np.load(input_dir/name,allow_pickle=False),accepted_cache[ix]):
            raise ValueError('Pilot '+name+' differs from replayed native semantic features')
    selected_threshold,_=task._parameter(source_bundle['curators'][encoder],'threshold',encoder)
    if metadata['threshold']!=selected_threshold or metadata['priority_seed']!=0:
        raise ValueError('Pilot graph differs from the calibrated curator and fixed primary priority')
    expected_graph=build_reference_graph(accepted_cache[ix],pilot_ids,selected_threshold,
        source_ids=[r['source_unit_id'] for r in development_rows],seed=0)
    with np.load(input_dir/'graph.npz',allow_pickle=False) as stored:
        for key,wanted in [('priority',expected_graph.priority_indices),('indptr',expected_graph.indptr),('indices',expected_graph.indices)]:
            if not np.array_equal(stored[key],wanted):raise ValueError('Pilot graph does not recompute: '+key)
    if source_bundle['task']=='civil':targets=np.asarray([[r['original_fields']['toxicity']] for r in development_rows],np.float64)
    else:
        vocabulary=source_bundle['learners'][encoder]['calibration']['vocabulary_lock']
        targets=task.model_selection.encode_tag_targets(development_rows,vocabulary)
    if not np.array_equal(np.load(input_dir/'targets.npy',allow_pickle=False),targets):
        raise ValueError('Pilot targets differ from original accepted labels')
    file_hashes={name:task.file_hash(input_dir/name) for name in input_manifest['files']}
    if any(input_manifest['files'][name]['sha256']!=value for name,value in file_hashes.items()):
        raise ValueError('Development input package changed')
    for item in dossier['service_reports']:
        path=(base/item['path']).resolve()
        if not path.is_relative_to(base) or not path.is_file() or task.file_hash(path)!=item['sha256']:
            raise ValueError('Development service report missing or changed')
        report=json.loads(path.read_text())
        if report.get('success') is not True:raise ValueError('Resource pilot must retain failed services separately')
        required={'metadata.json','graph.npz','learner.npy','targets.npy'}
        if report.get('method') in {'O-G','O-G-FP32'}:required.add('curator.npy')
        if set(report['raw_input_files_sha256'])!=required:
            raise ValueError('Development report omits or adds method input files')
        if any(file_hashes.get(name)!=value for name,value in report['raw_input_files_sha256'].items()):
            raise ValueError('Resource timings came from different worker input arrays')
        for stage in ('construction','repair'):
            observed.append(report[stage]['spawn_to_reap_seconds'])
    if observed!=dossier['stage_seconds']:raise ValueError('Resource timings differ from actual service reports')
    derived=recipes.derive_resource_policy(dossier['resource_snapshot'],observed)
    if derived!=dossier['resource_lock']:raise ValueError('Resource policy does not recompute')
    expected={'memory_bytes':derived['per_method_address_space_limit_bytes'],'cpu_seconds':derived['cpu_seconds_each_stage'],
        'wall_seconds':derived['wall_seconds_each_stage'],'threads':derived['cpu_threads']}
    if any(bundle['execution_policy'].get(k)!=v for k,v in expected.items()):raise ValueError('Worker policy differs from derived development limits')
    precision=dossier['precision_inputs']
    planned=recipes.precision_analysis(precision['effects'],development_ids=precision['development_ids'],
        confirmation_ids=precision['confirmation_ids'])
    if planned!=dossier['precision_report']:raise ValueError('Development precision analysis does not recompute')
    return {'resource_lock_sha256':digest(derived),'precision_report_sha256':digest(planned),
        'development_source_ids_sha256':digest(dossier['development_source_ids']),
        'development_source_replay_sha256':pilot_replay['replay_binding_sha256'],
        'pilot_input_file_hashes':file_hashes,'actual_worker_array_lineage_checked':True,
        'historical_collection_and_precision_effect_derivation_require_bound_external_review':True}

def logistic_rule(bundle,encoder):
    from phase6.logistic_selection import select_logistic_decision_thresholds
    entry=bundle['learners'][encoder];dossier=entry['calibration']
    expected=select_logistic_decision_thresholds(dossier['features'],dossier['records'],
        entry['lambda_lock']['provenance'],entry['lambda_lock'],dossier['vocabulary_lock'])
    if entry.get('logistic_threshold_lock')!=expected:raise ValueError('Logistic decision rule does not recompute from calibration-only probabilities')
    return {'lock_sha256':expected['sha256'],'distinct_from_ridge_decisions':True}

def convex_fit(x,y,lam,warm,ex,ey,tolerance,budget,backend):
    if backend=='fraction_reference':return convex_program._fit(x,y,lam,warm,ex,ey,tolerance,budget)
    if backend!='exact_dyadic_integer_v1':raise ValueError('Unknown prospectively locked convex verifier')
    from phase6.dyadic_convex import certify_multioutput
    solved=convex_program.convex.solve_multioutput(x,y,lam,warm_start=warm)
    cert=certify_multioutput(x,y,lam,solved['weights'],parameter_tolerance=tolerance,max_coordinates=budget)
    logits=ex.astype(np.float64)@solved['weights'];probs=convex_program.convex.scalar.expit(logits)
    bce=np.where(logits>=0,np.logaddexp(0,-logits)+(1-ey)*logits,np.logaddexp(0,logits)-ey*logits)
    if not np.isfinite(logits).all() or not np.isfinite(bce).all():raise FloatingPointError('Nonfinite convex evaluation')
    return {'solver':{k:v for k,v in solved.items() if k!='weights'},'weights':solved['weights'].tolist(),
        'certificate':cert,'released':bool(cert['meets_parameter_tolerance']),
        'verifier_backend':backend,'evaluation':{'fractional_bce_mean_over_rows_and_outputs':float(np.mean(bce)),
            'fractional_target_probability_mse':float(np.mean((probs-ey)**2)),
            'score_logits':logits.tolist(),'probabilities':probs.tolist(),'rows':len(ex)}}

def _engineering_scope(bundle):
    if bundle.get('evidence_role')!=ENGINEERING_MODE or bundle.get('natural_input_sha256') is None:
        raise ValueError('Engineering bundle must bind an existing natural input and its explicit role')
    path=Path(bundle['natural_input_path'])
    if not path.is_file() or task.file_hash(path)!=bundle['natural_input_sha256']:
        raise ValueError('Natural input file is absent or changed')
    return {'source_sha256':bundle['natural_input_sha256'],'semantic_claim':False}

def _validated_task(bundle):
    y,ey,graph=task.validate_bundle(bundle)
    return {'train_rows':len(y),'evaluation_rows':len(ey),'outputs':y.shape[1],
            'graph_binding':requests_module.graph_binding(graph)}

def _base_graph(bundle):
    ids=bundle['train_ids'];sources=bundle['source_ids']
    if len(set(ids))!=len(ids) or len(sources)!=len(ids):raise ValueError('Complete unique record/source identity required')
    if set(bundle['source_kinds'])!=set(sources):raise ValueError('Source-kind partition is incomplete')
    name=bundle['base_curator'];entry=bundle['curators'][name]
    if entry['train_ids']!=ids:raise ValueError('Curator row order differs')
    x=task.bound_array(entry['train'],len(ids),'curator');tau,_=task._parameter(entry,'threshold',name)
    return build_reference_graph(x,ids,tau,sources,seed=bundle.get('base_priority_seed',0))

def resolve(group,bundle):
    c=group['configuration'];g=copy.deepcopy(group)
    # This obtains fixed geometry only. Actual state precision and solver remain
    # explicit in each worker call and are never replaced by this geometry step.
    g['configuration']['state_precision']='FP64';g['configuration']['decoder']='Cholesky'
    if bundle.get('task') in {'civil','multilabel'}:
        x,ex,graph,lam,thresholds,geometry=task.resolve_cell(g,bundle)
        y=np.asarray(bundle['train_y']);ey=np.asarray(bundle['evaluation_y'])
        if y.ndim==1:y=y[:,None]
        if ey.ndim==1:ey=ey[:,None]
    else:
        name=c['curator_encoder'];entry=bundle['curators'][name];tau,seal=task._parameter(entry,'threshold',name)
        tau=task._threshold(tau,c.get('threshold','calibrated'))
        cx=task.bound_array(entry['train'],len(bundle['train_ids']),'curator')
        graph=build_reference_graph(cx,bundle['train_ids'],tau,bundle['source_ids'],seed=c.get('priority_seed',0))
        x=ex=y=ey=thresholds=None;lam=None;geometry={'threshold':tau,'threshold_lock_sha256':seal}
    return {'x':x,'ex':ex,'y':y,'ey':ey,'graph':graph,'lambda':lam,'thresholds':thresholds,'priority_seed':c.get('priority_seed',0),
        'geometry':geometry,'cx':np.asarray(bundle['curators'][c['curator_encoder']]['train']),
        'source_kinds':bundle['source_kinds']}

def request_for_group(group,bundle,cell):
    """Every curator has its own graph-dependent U/A requests."""
    base=bundle['requests'];graph=cell['graph']
    if requests_module.graph_binding(graph)==base['graph_binding']:return base
    alt=bundle.get('sensitivity_requests',{}).get(group['group_id'])
    if alt is None:
        cfg=base['configuration']
        alt=requests_module.generate_manifest(graph,dataset_id=base['dataset_id'],panel_id=base['panel_id'],
            source_kinds=bundle['source_kinds'],design=bundle['request_design'],allocations=cfg['allocations'],
            evidence_role=base['evidence_role'],master_seed=cfg['master_seed'],
            record_horizon=cfg['requests']['record_horizon'],source_horizon=cfg['requests']['source_horizon'],
            include_excluded_blocker_stress=cfg['include_excluded_blocker_stress'])
    shadow={**bundle,'requests':alt};recompute_requests(shadow,graph)
    # R/S orders use the same corpus/panel seed namespace. U/A use the current graph.
    return alt

def _check_path(job,path,bundle):
    if job.get('arm')!=path['arm']:raise ValueError('Registry and request arms differ')
    if path['unit']=='source' and path['arm']=='S':
        if any(bundle['source_kinds'].get(s)!='genuine_native' for s in path['deletion_order']):
            raise ValueError('S requests contain unknown or proxy sources')
    if len(set(path['deletion_order']))!=len(path['deletion_order']):raise ValueError('Duplicate request IDs')
    if sorted(set(path['checkpoints']))!=path['checkpoints'] or any(k<1 or k>len(path['deletion_order']) for k in path['checkpoints']):
        raise ValueError('Invalid request checkpoints')
    planned=job.get('planned_checkpoint_units')
    if isinstance(planned,list):
        expected=sorted({min(k,len(path['deletion_order'])) for k in planned if len(path['deletion_order'])>0})
        if path['checkpoints']!=expected:raise ValueError('Actual checkpoints differ from the registered clipped schedule')

def structural_job(job,path,cell,out):
    graph=cell['graph'];ids=list(graph.record_ids);initial=set(map(int,graph.selected_indices()));rows=[]
    from ccu.structure import structural_census
    horizon=len(path['deletion_order']);arm=path['arm']
    if path['unit']=='source':frame=sorted(s for s,kind in cell['source_kinds'].items() if kind=='genuine_native')
    elif arm=='U':frame=sorted(ids[i] for i in range(len(ids)) if i not in initial)
    else:frame=sorted(ids)
    dimensions=(cell['x'].shape[1],) if cell['x'] is not None else (768,)
    outputs=(cell['y'].shape[1],) if cell['y'] is not None else (1,20)
    key=digest([horizon,path['unit'],frame,dimensions,outputs,arm])
    cache=cell.setdefault('_structural_forecasts',{})
    if key not in cache:
        census=structural_census(graph,[horizon],unit=path['unit'],sampling_frame=frame,dimensions=dimensions,outputs=outputs)
        if arm not in {'R','S','U','R-volume-matched-to-S'}:
            for item in census['horizons']:item['workload_expectation']=None
        census['sampling_law']=arm
        census['byte_forecasts_are_not_measured_total_memory']=True
        census['unlabeled_output_counts_are_explicit_hypothetical_forecasts']=cell['y'] is None
        cache[key]=census
    write(out/'structure_forecast.json',cache[key])
    for checkpoint in path['checkpoints']:
        dead=boundary.deleted_indices(path,checkpoint,graph)
        deleted=[ids[i] for i in sorted(dead)]
        target=independent_selection(cell,dead,fresh=job.get('fresh_graph',False));production=set(map(int,graph.selected_indices(deleted)))
        rows.append({'checkpoint':checkpoint,'status':'completed' if target==production else 'graph_disagreement',
          'selected_count':len(target),'admissions':len(target-initial),'zero_admission':not bool(target-initial),
          'deleted_records':len(dead),'deleted_units':checkpoint,'unit':path['unit'],
          'selected_ids':[ids[i] for i in sorted(target)],'independent_graph_checked':True})
    write(out/'structure.json',rows)
    return {'status':'completed' if all(r['status']=='completed' for r in rows) else 'failed',
            'checkpoint_count':len(rows),'artifact':'structure.json','zero_admission_checkpoints':sum(r['zero_admission'] for r in rows),
            'structural_forecast':'structure_forecast.json','forecast_initial_horizon':horizon,
            'independent_graph_seconds':cell.get('_independent_graph_seconds'),'independent_graph_bytes':cell.get('_independent_graph_bytes'),
            'fresh_similarity_each_checkpoint':job.get('fresh_graph',False)}

def _copy_outputs(source,target):
    """Retain exact reports and heads. Record state hashes before scratch cleanup."""
    target.mkdir(parents=True,exist_ok=False)
    for p in sorted(source.rglob('*')):
        if p.is_file() and not p.name.startswith('state.npz'):
            q=target/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)

def analysis_binding(job,path,group,bundle,cell):
    accepted=cell.get('_validated_source_bindings') or {}
    kinds=accepted.get('evaluation_source_kinds',bundle.get('evaluation_source_kinds'))
    sources=accepted.get('evaluation_source_ids',bundle.get('evaluation_source_ids'))
    thresholds=cell.get('thresholds')
    encoded=None if thresholds is None else ['Infinity' if math.isinf(float(v)) and v>0 else float(v) for v in thresholds]
    return {'group_id':group['group_id'],'dataset_id':group['corpus'],'panel_id':group['panel'],
        'configuration_id':digest(group['configuration']),'configuration':group['configuration'],
        'arm':path['arm'],'trajectory_id':path['trajectory_id'],'evaluation_ids':bundle.get('evaluation_ids',[]),
        'evaluation_source_ids':sources,'evaluation_source_kinds':kinds,'thresholds':encoded,
        'evaluation_targets_sha256':None if cell['ey'] is None else array_hash(cell['ey']),
        'accepted_source_replay':bool(accepted),'role':bundle.get('evidence_role','primary_input_candidate')}

def method_job(job,path,group,bundle,cell,out,policy):
    from phase6 import run_isolated as isolated
    graph=cell['graph'];ids=list(graph.record_ids);methods=[];initial=set(map(int,graph.selected_indices()))
    x,y,ex,lam=cell['x'],cell['y'],cell['ex'],cell['lambda']
    config=group['configuration'];fp32=config.get('state_precision')=='FP32'
    solver='cg' if config.get('decoder') in {'zero_start_CG','cg'} else 'cholesky'
    before=ex.astype(np.float64)@dual_head(x,y,initial,lam)
    oracle={}
    for checkpoint in path['checkpoints']:
        dead=boundary.deleted_indices(path,checkpoint,graph);deleted=[ids[i] for i in dead]
        selected=independent_selection(cell,dead,fresh=job.get('fresh_graph',False))
        if set(selected)!=set(map(int,graph.selected_indices(deleted))):raise AssertionError('Independent graph disagreement')
        oracle[checkpoint]=(set(selected),initial-dead,dead)
    batches=[];last=0
    for k in path['checkpoints']:batches.append(path['deletion_order'][last:k]);last=k
    with tempfile.TemporaryDirectory(prefix='dispatch_',dir=ROOT/'tmp') as scratch_name:
        scratch=Path(scratch_name)
        package=Path(cell['_prepared_bundle_path']) if '_prepared_bundle_path' in cell else isolated.prepare_bundle(
            cell['cx'],x,y,ids,scratch/'inputs',threshold=graph.threshold,
            source_ids=bundle['source_ids'],priority_seed=config.get('priority_seed',0))
        write(out/'shared_bundle.json',json.loads((package/'bundle.json').read_text()))
        order=list(job.get('methods',group['methods']))
        order_seed=int(digest(['ccu-method-order-1',job['job_id']])[:32],16)
        random.Random(order_seed).shuffle(order)
        write(out/'method_order.json',{'seed':order_seed,'order':order,'fixed_before_method_outputs':True})
        for number,method in enumerate(order):
            record={'method':method,'status':'planned','requested_precision':config.get('state_precision','FP64'),'solver':solver}
            if method not in KNOWN_METHODS:
                record.update(status='unsupported_method');methods.append(record);continue
            if fp32 and method not in {'O-G','O-T','B-E','B-A','P-I','P-S','P-R','B-F'}:
                record.update(status='unsupported_precision',reason='No FP32 implementation for this exact method; FP64 replacement refused')
                methods.append(record);continue
            implementation=method+'-FP32' if fp32 else method
            run=scratch/f'method_{number:02d}';target=out/f'method_{number:02d}'
            try:
                report=isolated.run_service(package,implementation,batches,run,
                    horizon=len(path['deletion_order']),unit=path['unit'],lambda_reg=lam,policy=policy,
                    persistence=job.get('persistence','final'),solver=solver,panel=job.get('panel_mode','cold_process'))
                record.update(status='completed' if report['success'] else 'worker_failure',service_success=report['success'])
                if report['success']:
                    repairs=json.loads((run/'repair/repair_report.json').read_text());checks=[]
                    for i,k in enumerate(path['checkpoints']):
                        selected,frozen,dead=oracle[k];wanted=frozen if method=='B-F' else selected
                        expected=dual_head(x,y,wanted,lam);got=np.load(run/f'repair/head_{i:04d}.npy',allow_pickle=False)
                        err=float(np.linalg.norm(got-expected));scale=max(1.,float(np.linalg.norm(expected)))
                        numeric=repairs['releases'][i].get('numerical_decoder',{})
                        finite=bool(np.isfinite(got).all());count=repairs['releases'][i]['count']
                        passed=finite and count==len(wanted) and (fp32 or err<=1e-9*scale)
                        pred=ex.astype(np.float64)@got;correct=ex.astype(np.float64)@dual_head(x,y,selected,lam)
                        frozen_pred=ex.astype(np.float64)@dual_head(x,y,frozen,lam)
                        c={'checkpoint':k,'passed':passed,'count':count,'expected_count':len(wanted),
                           'head_frobenius_error':err,'relative_audit_scale':scale,'numerical_gate':numeric,
                           'admissions':len(selected-initial),'zero_admission':not bool(selected-initial),
                           'target':'retained_initial_selection' if method=='B-F' else 'retained_full_curation',
                           'is_secondary_FP32_frontier':fp32,'task_metrics':task.task_metrics(bundle,cell['ey'],pred,cell['thresholds']),
                           'effects':task.metrics.pair_effects(cell['ey'],before,frozen_pred,correct,thresholds=cell['thresholds'])}
                        checks.append(c)
                        np.savez(run/f'prediction_{i:04d}.npz',prediction=pred,oracle=correct,frozen=frozen_pred,before=before,
                            evaluation_y=cell['ey'])
                    record.update(head_checks=checks,independent_target_checks_passed=all(c['passed'] for c in checks))
                    if not all(c['passed'] for c in checks):record['status']='oracle_disagreement'
                    full_state=bool(job.get('complete_state_rebuild_each_checkpoint'))
                    if full_state or implementation in {'B-E','P-I','P-S','P-R'}:
                        from phase6.state_audit import audit_service_state
                        audit=audit_service_state(package,implementation,batches,run,
                            out/f'state_audit_{number:02d}',horizon=len(path['deletion_order']),unit=path['unit'],
                            lambda_reg=lam,policy=policy,solver=solver,full_state=full_state)
                        record['state_audit']=audit
                        if not audit.get('passed'):record['status']='state_audit_failure'
                record['artifact_directory']=target.name
            except Exception as e:record.update(status='failed',exception_type=type(e).__name__,message=str(e))
            finally:
                if run.exists():_copy_outputs(run,target)
            methods.append(record);write(out/'methods.json',methods)
    return {'status':'completed' if all(m['status']=='completed' for m in methods) else 'method_noncompletion',
            'methods':methods,'checkpoint_count':len(path['checkpoints']),
            'analysis_binding':analysis_binding(job,path,group,bundle,cell)}

def convex_job(job,path,group,bundle,cell,out,policy):
    """Run all three convex methods on the same path, including matched record controls."""
    from fractions import Fraction
    graph=cell['graph'];ids=list(graph.record_ids);initial=set(map(int,graph.selected_indices()))
    x,y,ex,ey=cell['x'],cell['y'],cell['ex'],cell['ey'];lam=cell['lambda']
    states={m:convex_program.LogisticPayload(graph,x,y,len(path['deletion_order']),path['unit'])
            for m in ('eligible_payload_retraining','certified_convex_repair')}
    warm={};rows=[];last=0;tol=Fraction(1,10**8);budget=policy.get('convex_certificate_coordinates',2_000_000);blocked=set()
    ix=np.asarray(sorted(initial),dtype=int)
    backend=policy.get('convex_verifier','fraction_reference')
    fit=convex_fit(x[ix],y[ix],lam,None,ex,ey,tol,budget,backend)
    np.save(out/'initial_head.npy',np.asarray(fit['weights'],dtype=np.float64),allow_pickle=False)
    write(out/'initial_fit.json',{k:v for k,v in fit.items() if k!='weights'})
    if not fit.get('released'):return {'status':'initial_certificate_failure','methods':[{'method':m,'status':'initial_certificate_failure'} for m in group['methods']]}
    for method in group['methods']:warm[method]=np.asarray(fit['weights']).copy()
    for k in path['checkpoints']:
        delta=path['deletion_order'][last:k];last=k
        dead=boundary.deleted_indices(path,k,graph);selected=set(map(int,graph.selected_indices([ids[i] for i in dead])))
        independent=independent_selection(cell,dead,fresh=job.get('fresh_graph',False))
        if selected!=set(independent):raise AssertionError('Independent convex retained graph differs')
        for method in group['methods']:
            row={'method':method,'checkpoint':k}
            if method in blocked:
                row.update(status='blocked_after_previous_failure');rows.append(row);continue
            try:
                if method=='fresh_optimum':
                    ix=np.asarray(sorted(selected),dtype=int);fx,fy=x[ix],y[ix];start=None
                else:
                    states[method].delete(delta);fx,fy,chosen=states[method].selected_payload()
                    if set(chosen)!={ids[i] for i in selected}:raise AssertionError('Convex payload selection differs')
                    start=warm[method] if method=='certified_convex_repair' else None
                got=convex_fit(fx,fy,lam,start,ex,ey,tol,budget,backend)
                row.update({key:value for key,value in got.items() if key!='weights'})
                if bundle['task']=='multilabel':
                    entry=bundle['learners'][group['configuration']['learner_encoder']]
                    lock=entry.get('logistic_threshold_lock')
                    if lock is not None:
                        thresholds=np.asarray([np.inf if t['kind']=='never_positive' else t['threshold'] for t in lock['thresholds']],np.float64)
                        row['task_metrics']=task.metrics.multilabel_metrics(ey,np.asarray(got['evaluation']['probabilities']),thresholds)
                    else:row['task_metrics']={'status':'missing_separate_logistic_decision_rule','micro_f1':None,'macro_f1':None}
                if got['released']:
                    warm[method]=np.asarray(got['weights'],dtype=np.float64);np.save(out/f'{method}_{k:06d}.npy',warm[method],allow_pickle=False)
                row.update(status='completed' if got['released'] else 'certificate_failure',selected_count=len(selected),
                    admissions=len(selected-initial),zero_admission=not bool(selected-initial))
            except Exception as e:row.update(status='failed',exception_type=type(e).__name__,message=str(e))
            if row['status']!='completed':blocked.add(method)
            rows.append(row);write(out/'convex.json',rows)
    for method,state in states.items():state.snapshot(out/(method+'.npz'),warm[method])
    return {'status':'completed' if all(r['status']=='completed' for r in rows) else 'method_noncompletion',
            'checkpoint_count':len(path['checkpoints']),'method_releases':len(rows),'artifact':'convex.json'}

def full_refit_group(group,jobs,bundle,cell,out,*,primary):
    """Execute the official or disclosed reference backend and reconcile every job."""
    cfg=refit.Config(**bundle['refit_configuration'])
    if primary and cfg.backend!=refit.OFFICIAL:raise ValueError('Primary SemDeDup requires the pinned official backend')
    allocations={arm:group['allocations'].get(arm,0) for arm in ('R','S','A')}
    summary=refit.run_boundary(cell['cx'],cell['x'],cell['y'],bundle['train_ids'],bundle['source_ids'],
        bundle['source_kinds'],cfg,out/'branch',lambda_reg=cell['lambda'],design=bundle['request_design'],
        allocations=allocations,seed_variants=bundle['refit_seed_variants'],prediction_features=cell['ex'])
    rows=[json.loads(line) for line in (out/'branch/checkpoints.jsonl').read_text().splitlines()]
    seeds=json.loads((out/'branch/seed_audit.json').read_text());lookup={r:i for i,r in enumerate(bundle['train_ids'])}
    original=json.loads((out/'branch/original_head.json').read_text())
    original_check=saved_head_gate(cell['x'],cell['y'],{lookup[r] for r in original['selected_ids']},cell['lambda'],original['weights'])
    write(out/'refit_original_head_check.json',original_check)
    if not original_check['passed']:raise AssertionError('Original fitted head fails independent numerical acceptance')
    seed_checks=[]
    for i,seed in enumerate(seeds):
        fitted=json.loads((out/f'branch/seed_audit/{i}/fitted.json').read_text())
        seed_checks.append(saved_head_gate(cell['x'],cell['y'],{lookup[r] for r in fitted['selected_ids']},cell['lambda'],seed['weights']))
    write(out/'refit_seed_head_checks.json',seed_checks)
    for row in rows:
        row['independent_head_checks']=[]
        if row['status']!='completed':continue
        for method,head in row['weights'].items():
            selected={lookup[r] for r in row['selected_ids'][method]}
            row['independent_head_checks'].append({'method':method,**saved_head_gate(cell['x'],cell['y'],selected,cell['lambda'],head)})
        if not all(c['passed'] for c in row['independent_head_checks']):row['status']='oracle_disagreement'
    write(out/'refit_independent_checks.json',rows);outcomes=[]
    for job in jobs:
        row={'job_id':job['job_id'],'group_id':group['group_id']}
        kind=job.get('kind')
        if kind in {'same_seed_no_deletion','alternative_seed_no_deletion'}:
            offset=0 if kind=='same_seed_no_deletion' else 5
            index=offset+job['repeat'];found=seeds[index]
            row.update(status='completed' if seed_checks[index]['passed'] else 'numerical_gate_failure',seed_audit=found,
                independent_head_check=seed_checks[index])
        else:
            selected=[r for r in rows if r['trajectory_id']==job.get('trajectory_id')]
            if selected:row.update(status='completed' if all(r['status']=='completed' for r in selected) else 'checkpoint_failures',checkpoint_count=len(selected))
            else:
                manifest=json.loads((out/'branch/requests.json').read_text())
                path=next((p for p in manifest['trajectories'] if p['trajectory_id']==job.get('trajectory_id')),None)
                row.update(status='structural_zero_or_unavailable_arm' if path is not None and not path['checkpoints'] else 'blocked_missing_frozen_path')
        row['methods']=[{'method':m,'status':row['status']} for m in job['methods']]
        row['primary_output_accepted']=primary and row['status']=='completed';outcomes.append(row)
    return outcomes,summary

def run_dispatch(registry,jobs,bundles,out_dir,*,mode=ENGINEERING_MODE,design=None,base_dir=None,policy=None):
    """Retain every planned cell. Each status has a concrete cause or output."""
    if registry.get('jobs_content_sha256')!=digest(jobs):raise ValueError('Registry job digest mismatch')
    if len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate job IDs')
    if len({g['group_id'] for g in registry['groups']})!=len(registry['groups']):raise ValueError('Duplicate group IDs')
    if mode==PRIMARY_MODE:
        from phase6.recipes import build_registry as compile_primary
        expected_registry,expected_jobs=compile_primary()
        if registry!=expected_registry or jobs!=expected_jobs:raise ValueError('Primary registry must match the exact accepted recipe compiler')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=False);(ROOT/'tmp').mkdir(exist_ok=True)
    design=json.loads((ROOT.parent/'output/empirical_program/study_design.json').read_text()) if design is None else design
    if mode==PRIMARY_MODE and design!=json.loads((ROOT.parent/'output/empirical_program/study_design.json').read_text()):
        raise ValueError('Primary design override is forbidden; version the protocol before execution')
    from phase5.run_isolated_v2 import DEFAULT_POLICY
    policy=dict(DEFAULT_POLICY if policy is None else policy)
    lock={'schema':SCHEMA,'mode':mode,'registry_sha256':digest(registry),'jobs_content_sha256':digest(jobs),
        'bundle_bindings':task.fingerprint(bundles),'policy':policy,'source_hashes':codes(),
        'primary_results_claimed_before_acceptance':False,'output_policy':'failed and absent cells remain in the ledger',
        'frozen_branch_metadata':'Legacy engine labels remain unchanged. This dispatcher records its own acceptance and output checks.'}
    write(out/'lock.json',lock);ledger=[];group_reports=[];handled=set()
    for gi,group in enumerate(registry['groups']):
        target=out/f'group_{gi:03d}';target.mkdir();gjobs=[j for j in jobs if j.get('group_id')==group['group_id']]
        grow={'group_id':group['group_id'],'planned_jobs':len(gjobs)};group_scratch=None
        try:
            bundle=bundles[group['request_family']]
            if mode==PRIMARY_MODE and bundle.get('execution_policy')!=policy:
                raise ValueError('Common resource policy is not bound by this input dossier')
            accepted=acceptance(group,bundle,design,mode=mode,base_dir=base_dir or ROOT.parent)
            write(target/'acceptance.json',accepted)
            if not accepted['execution_allowed']:raise ValueError('Input acceptance failed; see acceptance.json')
            cell=resolve(group,bundle);manifest=request_for_group(group,bundle,cell)
            cell['_validated_source_bindings']=accepted.get('validated_source_bindings')
            if mode==PRIMARY_MODE and group['configuration'].get('threshold','calibrated')!='calibrated':
                if not cell['geometry'].get('threshold_quality',{}).get('quality_pass'):
                    raise ValueError('The separate threshold sensitivity quality gate did not pass')
            write(target/'requests.json',manifest);paths={p['trajectory_id']:p for p in manifest['trajectories']}
            write(target/'geometry.json',cell['geometry']);grow.update(status='accepted_inputs',primary_inputs_accepted=accepted['primary_inputs_accepted'])
            if group['block']=='E_full_refit':
                outcomes,branch_summary=full_refit_group(group,gjobs,bundle,cell,target,primary=accepted['primary_inputs_accepted'])
                ledger.extend(outcomes);handled.update(r['job_id'] for r in outcomes);grow['branch_summary']=branch_summary
                group_reports.append(grow);write(out/'groups.json',group_reports);write(out/'jobs.json',ledger)
                continue
            if set(group['methods'])<=KNOWN_METHODS and cell['x'] is not None:
                from phase6 import run_isolated as isolated
                group_scratch=tempfile.TemporaryDirectory(prefix='dispatch_group_',dir=ROOT/'tmp')
                package=isolated.prepare_bundle(cell['cx'],cell['x'],cell['y'],bundle['train_ids'],Path(group_scratch.name)/'inputs',
                    threshold=cell['graph'].threshold,source_ids=bundle['source_ids'],priority_seed=cell['priority_seed'])
                cell['_prepared_bundle_path']=str(package)
                write(target/'shared_bundle.json',json.loads((package/'bundle.json').read_text()))
                grow['shared_preparation_charge']='one package construction per group; exact time and bytes in shared_bundle.json'
            for ji,job in enumerate(gjobs):
                row={'job_id':job['job_id'],'group_id':group['group_id'],'methods':[{'method':m,'status':'planned'} for m in job.get('methods',group['methods'])]}
                jobout=target/f'job_{ji:04d}';jobout.mkdir()
                try:
                    path=paths[job['trajectory_id']];_check_path(job,path,bundle)
                    row.update(actual_horizon=len(path['deletion_order']),actual_checkpoint_units=path['checkpoints'])
                    if not path['checkpoints']:
                        row.update(status='structural_zero_or_unavailable_arm',reason=path['status'],checkpoint_count=0)
                    elif group['methods']==['independent_structure_oracle']:
                        row.update(structural_job(job,path,cell,jobout))
                    elif group['block']=='F_convex':row.update(convex_job(job,path,group,bundle,cell,jobout,policy))
                    elif group['block']=='E_full_refit':raise NotImplementedError('Full-refit jobs require branch-specific fit and seed orchestration')
                    elif group['variant']=='ANN':
                        approximate,audit=boundary.audit_ann(cell['cx'],cell['graph'],**bundle['ANN_policy'])
                        comparisons=[]
                        for k in path['checkpoints']:
                            dead=boundary.deleted_indices(path,k,cell['graph']);deleted=[cell['graph'].record_ids[i] for i in dead]
                            exact=set(map(int,cell['graph'].selected_indices(deleted)));approx=set(map(int,approximate.selected_indices(deleted)))
                            comparisons.append({'checkpoint':k,'selected_count':len(exact),'approximate_selected_count':len(approx),
                                'selected_symmetric_difference':len(exact^approx)})
                        audit['checkpoints']=comparisons
                        write(jobout/'ANN.json',audit);row.update(status='completed',artifact='ANN.json',checkpoint_count=len(comparisons))
                    elif set(group['methods'])<=KNOWN_METHODS and cell['x'] is not None:
                        row.update(method_job(job,path,group,bundle,cell,jobout,policy))
                    else:raise NotImplementedError('No supported executor for this explicit group')
                    if row['status']=='completed':
                        row['primary_output_accepted']=accepted['primary_inputs_accepted']
                    if row.get('methods') and all(m['status']=='planned' for m in row['methods']):
                        row['methods']=[{'method':m['method'],'status':row['status']} for m in row['methods']]
                except KeyError as e:row.update(status='blocked_missing_input',message=str(e))
                except NotImplementedError as e:row.update(status='unsupported_executor',message=str(e))
                except Exception as e:row.update(status='failed',exception_type=type(e).__name__,message=str(e))
                for m in row.get('methods',[]):
                    if m['status']=='planned':m['status']=row['status']
                ledger.append(row);handled.add(job['job_id']);write(jobout/'outcome.json',row)
                with (out/'journal.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        except Exception as e:
            status='blocked_missing_input' if isinstance(e,(KeyError,FileNotFoundError)) else 'input_acceptance_failed'
            grow.update(status=status,exception_type=type(e).__name__,message=str(e))
            for job in gjobs:
                if job['job_id'] in handled:continue
                ledger.append({'job_id':job['job_id'],'group_id':group['group_id'],'status':status,'message':str(e),
                    'methods':[{'method':m,'status':status} for m in job.get('methods',group['methods'])]});handled.add(job['job_id'])
        finally:
            if group_scratch is not None:group_scratch.cleanup()
        group_reports.append(grow);write(out/'groups.json',group_reports);write(out/'jobs.json',ledger)
    extension_cache={}
    for job in jobs:
        if job['job_id'] not in handled:
            if not bundles:
                ledger.append({'job_id':job['job_id'],'status':'blocked_missing_input',
                    'recipe_id':job.get('recipe_id'),'executor':job.get('executor'),
                    'required_artifact_keys':job.get('required_artifact_keys',[]),
                    'methods':[{'method':m,'status':'blocked_missing_input'} for m in job.get('methods',[])]})
                continue
            target=out/'extensions'/digest(job['job_id'])[:20];target.mkdir(parents=True)
            try:
                from phase6.extensions import run_extension
                row=run_extension(job,bundles,target,{'dispatch_module':sys.modules[__name__],
                    'policy':policy,'mode':mode,'design':design,'base_dir':base_dir or ROOT.parent,
                    'extension_cache':extension_cache})
                if row.get('job_id')!=job['job_id']:raise ValueError('Extension output job ID differs')
            except Exception as e:
                row={'job_id':job['job_id'],'status':'blocked_recipe_executor_input' if isinstance(e,(KeyError,FileNotFoundError,ImportError)) else 'extension_failure',
                    'recipe_id':job.get('recipe_id'),'executor':job.get('executor'),'message':str(e),
                    'required_artifact_keys':job.get('required_artifact_keys',[]),
                    'methods':[{'method':m,'status':'not_executed'} for m in job.get('methods',[])]}
            ledger.append(row);write(target/'outcome.json',row)
    for temporary in extension_cache.get('_temporary_directories',[]):temporary.cleanup()
    if len(ledger)!=len(jobs) or {r['job_id'] for r in ledger}!={j['job_id'] for j in jobs}:raise AssertionError('A planned job disappeared')
    write(out/'jobs.json',ledger)
    summary={'schema':SCHEMA,'mode':mode,'planned_jobs':len(jobs),'observed_jobs':len(ledger),
        'completed_jobs':sum(r['status']=='completed' for r in ledger),'status_counts':{},
        'primary_outputs_accepted':sum(r.get('primary_output_accepted',False) for r in ledger),
        'every_planned_job_retained':True,'inputs_unchanged':task.fingerprint(bundles)==lock['bundle_bindings'],
        'source_hashes_at_end':codes(),'source_hashes_unchanged':codes()==lock['source_hashes'],
        'exact_physical_memory_traffic_claim':False,'external_provenance_proved':False}
    if mode==PRIMARY_MODE and (not summary['inputs_unchanged'] or not summary['source_hashes_unchanged']):
        summary['primary_outputs_accepted']=0
        for row in ledger:
            if row.get('primary_output_accepted'):
                row['primary_output_accepted']=False;row['acceptance_revoked']='Bound inputs or sources changed during execution'
        write(out/'jobs.json',ledger)
    for row in ledger:summary['status_counts'][row['status']]=summary['status_counts'].get(row['status'],0)+1
    write(out/'summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--jobs',type=Path,required=True);p.add_argument('--bundles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=[ENGINEERING_MODE,PRIMARY_MODE],default=ENGINEERING_MODE)
    a=p.parse_args();registry=json.loads(a.registry.read_text());opener=gzip.open if a.jobs.suffix=='.gz' else open
    with opener(a.jobs,'rt') as stream:jobs=[json.loads(l) for l in stream if l.strip()]
    bundles=task.load_bound_value(json.loads(a.bundles.read_text()),a.bundles.parent)
    print(json.dumps(run_dispatch(registry,jobs,bundles,a.out,mode=a.mode,base_dir=a.bundles.parent)))

if __name__=='__main__':main()
