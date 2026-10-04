"""Execute resolved recipe jobs with real inputs and explicit missing-input states."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import hashlib, json, random, re, time, tempfile
import numpy as np
from phase3 import calibration as cal
from phase3.panels import normalized_text
from phase3.reference_graph import stable_priority
from phase4 import requests as rq
from phase4.execution import independent_edges
from phase5 import boundary, task_program as task, statistics
from phase5.replication import news_chronological_requests
from phase6 import recipes


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def _index(trajectory):
    found=re.search(r'(\d{3})$',trajectory)
    if found is None:raise ValueError('Three-digit trajectory suffix required')
    return int(found.group(1))


def _group(job,bundle,mode):
    conf=deepcopy(recipes.legacy.NATIVE)
    if mode=='natural_text_engineering':
        # Engineering runs retain their actual encoder identity and dimension.
        # Primary runs cannot use this explicit reduced-geometry branch.
        conf['curator_encoder']=bundle['base_curator']
        if bundle.get('learners'):
            learner=bundle.get('base_learner',next(iter(bundle['learners'])))
            conf['learner_encoder']=learner
            conf['learner_dimension']=int(bundle['learners'][learner]['train'].shape[1])
        else:conf['learner_encoder']=bundle['base_curator']
    return {'group_id':'H_recipes/'+job['recipe_id']+'/'+job['corpus']+'/'+job['panel'],
            'block':'H_recipes','corpus':job['corpus'],'panel':job['panel'],
            'variant':job['recipe_id'],'methods':job['methods'],'configuration':conf,
            'request_family':job['corpus']+'/'+job['panel']}


def _cache_key(bundle_fingerprint,group,context):
    """Bind population contracts as well as the accepted in-memory snapshot."""
    return recipes.digest([bundle_fingerprint,group['corpus'],group['panel'],group['configuration'],context['mode'],
                           context['policy'],context['design'],str(Path(context['base_dir']).resolve())])


def _path_from_order(graph,arm,index,unit,horizon,frame,order,checkpoints,*,seed_value,metadata=None):
    ids,owners,blockers,_=rq._graph_state(graph)
    return rq._path(arm,index,seed_value,unit,horizon,frame,order,checkpoints,ids,owners,blockers,metadata)


def extension_path(job,bundle,cell):
    """Construct outcome-blind paths. Only registered A may inspect graph structure."""
    name=job['recipe_id'];graph=cell['graph'];ids,owners,blockers,_=rq._graph_state(graph)
    original=bundle['requests'];paths={p['trajectory_id']:p for p in original['trajectories']}
    if name=='structure_mass_PPS':
        index=_index(job.get('parent_source_trajectory',job['trajectory_id']))
        masses=Counter(owners.values());masses={s:n for s,n in masses.items() if bundle['source_kinds'][s]=='genuine_native'}
        sd=recipes.seed(name,job['corpus'],job['panel'],index)
        order=recipes.pps_order_exact_integer_tickets(masses,draw_count=8,seed_value=sd)
        levels=rq._levels([1,2,4,8],len(order))
        parent=_path_from_order(graph,'S-PPS',index,'source',8,sorted(masses),order,levels,seed_value=sd,
             metadata={'source_masses':masses,'sampling_law':'successive_PPS_without_replacement'})
        if job['arm']=='S-PPS':return parent
        counts=[o['cumulative_deleted_records'] for o in parent['observations']]
        limit=max(counts,default=0);ms=recipes.seed(name,job['corpus'],job['panel'],index,'matched-R')
        order=random.Random(ms).sample(list(ids),limit)
        return _path_from_order(graph,'R-volume-matched-to-S-PPS',index,'record',limit,ids,order,sorted(set(counts)),
            seed_value=ms,metadata={'parent_source_trajectory':parent['trajectory_id'],'separate_record_horizon':True})
    if name=='structure_one_percent_horizon':
        index=_index(job['trajectory_id']);arm=job['arm'];horizon=recipes.one_percent_horizon(len(ids))['initial_horizon']
        design=deepcopy(bundle['request_design']);design.setdefault('requests',{})['record_checkpoints']=sorted(set([1,8,32,128,horizon])-{0})
        allocation={a:(index+1 if a==arm else 0) for a in ('R','S','U','A')}
        manifest=rq.generate_manifest(graph,dataset_id=job['corpus'],panel_id=job['panel']+'/one-percent',
            source_kinds=bundle['source_kinds'],design=design,allocations=allocation,record_horizon=horizon,
            include_excluded_blocker_stress=False,master_seed=recipes.SEED)
        return next(p for p in manifest['trajectories'] if p['trajectory_id']==job['trajectory_id'])
    if name=='excluded_blocker_stress':
        index=_index(job['trajectory_id'])
        manifest=rq.generate_manifest(graph,dataset_id=job['corpus'],panel_id=job['panel'],source_kinds=bundle['source_kinds'],
            design=bundle['request_design'],allocations={'R':0,'S':0,'U':0,'A':index+1},record_horizon=128,
            include_excluded_blocker_stress=True,master_seed=recipes.SEED)
        return next(p for p in manifest['trajectories'] if p['trajectory_id']==job['trajectory_id'])
    if name=='news_chronological':
        records=bundle['train_records']
        if [r['record_id'] for r in records]!=list(graph.record_ids):raise ValueError('Chronological rows must match accepted training IDs')
        chronological=news_chronological_requests([{**r,'partition':'train'} for r in records],horizon=128,resolution='day')
        order=[];levels=[]
        for item in chronological['checkpoints']:
            order.extend(item['batch_record_ids']);levels.append(len(order))
        return _path_from_order(graph,'chronological',0,'record',128,ids,order,levels,
            seed_value=0,metadata={'chronological_manifest':chronological,'independent_stream_count':1})
    tid=job.get('trajectory_id')
    if tid is None:raise KeyError('trajectory_id')
    return deepcopy(paths[tid])


def control_selection(curator,graph,records,deleted):
    """Fresh own-curator oracle. Selected-neighbor greedy is intentionally separate."""
    ids=list(graph.record_ids);lookup={r:i for i,r in enumerate(ids)};dead=set(deleted)
    ordered=[ids[int(i)] for i in graph.priority_indices]
    if curator=='none':return {i for i,r in enumerate(ids) if r not in dead}
    if curator=='exact_text':
        if [r['record_id'] for r in records]!=ids:raise ValueError('Exact-text records must match graph IDs')
        seen=set();selected=set()
        for rid in ordered:
            if rid in dead:continue
            text=normalized_text(records[lookup[rid]]['text'])
            if text not in seen:seen.add(text);selected.add(lookup[rid])
        return selected
    if curator=='fixed_order_greedy':
        selected=set()
        for rid in ordered:
            i=lookup[rid]
            if rid not in dead and not (set(map(int,graph.blockers[i]))&selected):selected.add(i)
        return selected
    raise ValueError('Unknown negative-control curator')


def negative_control(job,bundle,cell,out,dispatch):
    graph=cell['graph'];ids=list(graph.record_ids);curator=job['curator']
    records=bundle['train_records'] if curator=='exact_text' else []
    initial=control_selection(curator,graph,records,set())
    if curator=='none':path=extension_path(job,bundle,cell)
    else:
        frame=sorted(set(ids)-{ids[i] for i in initial});index=_index(job['trajectory_id'])
        sd=recipes.seed('negative_controls',job['corpus'],job['panel'],curator,index)
        order=random.Random(sd).sample(frame,min(128,len(frame)))
        path=_path_from_order(graph,'own-excluded-U',index,'record',128,frame,order,
             rq._levels([1,8,32,128],len(order)),seed_value=sd,
             metadata={'curator':curator,'own_initial_selected_IDs':[ids[i] for i in sorted(initial)]})
    write(out/'requests.json',path);rows=[]
    for k in path['checkpoints']:
        dead=set(path['deletion_order'][:k]);selected=control_selection(curator,graph,records,dead)
        row={'checkpoint':k,'deleted_records':len(dead),'selected_count':len(selected),
             'selection_IDs':[ids[i] for i in sorted(selected)],
             'own_excluded_invariant':selected==initial if curator!='none' else None}
        if cell['x'] is not None:
            w=dispatch.dual_head(cell['x'],cell['y'],selected,cell['lambda'])
            from ccu.core import ridge_moments
            from phase5.decoders import decode
            ix=np.asarray(sorted(selected),dtype=int)
            solved=decode(ridge_moments(cell['x'][ix],cell['y'][ix]),cell['lambda'])
            row['head_gate']=solved.diagnostics
            row['head_dual_error']=float(np.linalg.norm(solved.weights-w)) if solved.weights is not None else None
            if solved.weights is not None:
                np.save(out/f'head_{k}.npy',solved.weights)
                row['metrics']=task.task_metrics(bundle,cell['ey'],cell['ex'].astype(np.float64)@solved.weights,cell['thresholds'])
        rows.append(row)
    write(out/'negative_control.json',rows)
    passed=all(r['own_excluded_invariant'] is not False and r.get('head_gate',{}).get('release_allowed',True) for r in rows)
    return {'status':('completed' if passed else 'failed') if rows else 'structural_zero_or_unavailable_arm',
            'checkpoint_count':len(rows),'artifact':'negative_control.json','not_primary_curator':True}


def envelope_job(job,path,cell,out,dispatch):
    graph=cell['graph'];ids=list(graph.record_ids)
    low,high,audit=boundary.graph_envelope(cell['cx'],graph);rows=[]
    for k in path['checkpoints']:
        dead=boundary.deleted_indices(path,k,graph);deleted=[ids[i] for i in sorted(dead)]
        minimum=set(map(int,high.selected_indices(deleted)));maximum=set(map(int,low.selected_indices(deleted)))
        target=set(map(int,graph.selected_indices(deleted)))
        if not minimum<=target<=maximum:raise AssertionError('Envelope selection containment failed')
        row={'checkpoint':k,'minimum_selected':len(minimum),'target_selected':len(target),
             'maximum_selected':len(maximum),'selection_containment':True}
        if cell['x'] is not None:
            candidate=dispatch.dual_head(cell['x'],cell['y'],minimum,cell['lambda'])
            oracle=dispatch.dual_head(cell['x'],cell['y'],target,cell['lambda'])
            radius=boundary.model_envelope_bound(cell['x'],cell['y'],candidate,minimum,maximum,cell['lambda'])
            row.update(model_bound=radius,head_error_frobenius=float(np.linalg.norm(candidate-oracle)))
            np.save(out/f'envelope_head_{k}.npy',candidate)
        else:row.update(model_bound=None,model_scope='unlabeled_structure_corpus_no_fabricated_targets')
        rows.append(row)
    write(out/'graph_envelope.json',{'uniform_interval_audit':audit,'checkpoints':rows})
    return {'status':'completed','artifact':'graph_envelope.json','checkpoint_count':len(rows),
            'bound_conditional_on_declared_arithmetic_assumptions':True}


def utility_panel(bundle,cell,out,dispatch):
    """Common fixed ridge references and independently certified logistic utility."""
    from scipy.special import expit
    from phase5.convex_multioutput import solve_multioutput
    from phase6.dyadic_convex import certify_multioutput
    x,y,ex,ey,lam=cell['x'],cell['y'],cell['ex'],cell['ey'],cell['lambda']
    ids=bundle['train_ids'];initial=set(map(int,cell['graph'].selected_indices()))
    order=sorted(range(len(ids)),key=lambda i:(hashlib.sha256(('ccu-v1-utility-hash\0'+ids[i]).encode()).digest(),ids[i]))
    rows=[]
    for name,selected in [('curated_ridge',initial),('uncurated_ridge',set(range(len(ids)))),('same_size_hash_ridge',set(order[:len(initial)]))]:
        head,pred=task.ridge_predictions(x,y,selected,ex,lam)
        independent=dispatch.dual_head(x,y,selected,lam)
        error=float(np.linalg.norm(head-independent))
        gate=dispatch.saved_head_gate(x,y,selected,lam,head)
        np.savez(out/(name+'.npz'),weights=head,predictions=pred,selected_indices=np.asarray(sorted(selected),dtype=np.int64))
        rows.append({'method':name,'selected_records':len(selected),
                     'status':'completed' if gate['passed'] else 'oracle_disagreement',
                     'actual_saved_head_gate':gate,'independent_dual_head_error':error,'metrics':task.task_metrics(bundle,ey,pred,cell['thresholds'])})
    budget=2000000;planned=bundle.get('convex_verification_development')
    if planned is not None:
        rebuilt=recipes.select_verification_budget(planned['observations'],rows=len(ids),dimension=x.shape[1],outputs=y.shape[1],resource_policy=planned['resource_policy'])
        if planned['locked_budget']!=rebuilt:raise ValueError('Verification budget does not recompute from development observations')
        write(out/'verification_budget.json',rebuilt)
        if not rebuilt['forecast_feasible']:
            rows.append({'method':'curated_logistic','status':'forecast_infeasible','observed_timeout':False})
            return rows
        budget=rebuilt['max_coordinates']
    row={'method':'curated_logistic','selected_records':len(initial),'verification_coordinate_budget':budget}
    if len(initial)*x.shape[1]*y.shape[1]>budget:
        row.update(status='verification_budget_infeasible',observed_timeout=False);rows.append(row);return rows
    ix=np.asarray(sorted(initial),dtype=int)
    fit=solve_multioutput(x[ix],y[ix],lam,**{k:task.LOGISTIC_POLICY[k] for k in ('gradient_tolerance','max_iterations','max_backtracks')})
    pred=expit(ex.astype(np.float64)@fit['weights'])
    np.savez(out/'curated_logistic.npz',weights=fit['weights'],predictions=pred,selected_indices=ix)
    cert=certify_multioutput(x[ix],y[ix],lam,fit['weights'],max_coordinates=budget,parameter_tolerance=Fraction(1,10**8))
    thresholds=cell['thresholds']
    if bundle['task']=='multilabel':
        from phase6.logistic_selection import apply_logistic_thresholds
        learner=bundle.get('base_learner',next(iter(bundle['learners'])))
        lock=bundle['learners'][learner]['logistic_threshold_lock']
        dispatch.logistic_rule(bundle,learner)
        apply_logistic_thresholds(pred,lock)
        thresholds=np.full(y.shape[1],np.inf)
        for entry in lock['thresholds']:
            if entry['kind']=='score_at_least':thresholds[entry['output_coordinate']]=entry['threshold']
    row.update(status='completed' if cert['meets_parameter_tolerance'] else 'certificate_failed',certificate=cert,
               metrics=task.task_metrics(bundle,ey,pred,thresholds),meter=fit['meter'],
               logistic_task_thresholds='separate_calibration_logistic_lock' if bundle['task']=='multilabel' else 'fixed_Civil_endpoint')
    rows.append(row);return rows


def _narrow_review(dossier,replay,context,questions):
    """Bind supplied accountable testimony. A hash does not authenticate a witness."""
    if context['mode']!=context['dispatch_module'].PRIMARY_MODE:
        if dossier.get('evidence_role')!='natural_text_engineering':raise ValueError('Explicit natural engineering scope required')
        return {'primary_accepted':False,'external_testimony_authenticated_by_code':False}
    review=dossier['external_evidence_review'];cal._verify(review)
    if review.get('schema')!='ccu-stage-external-review-1':raise ValueError('Stage-specific external review required')
    if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():raise ValueError('Accountable reviewer required')
    if datetime.fromisoformat(review['completed_at'].replace('Z','+00:00')).tzinfo is None:raise ValueError('Review timestamp needs timezone')
    data={k:v for k,v in dossier.items() if k!='external_evidence_review'}
    if review.get('reviewed_dossier_sha256')!=recipes.digest(task.fingerprint(data)):raise ValueError('Review does not bind actual stage dossier')
    if set(review.get('evidence',{}))!=set(questions):raise ValueError('Complete stage-specific trust questions required')
    base=Path(context['base_dir']).resolve();receipts={}
    for question,e in review['evidence'].items():
        path=(base/e['path']).resolve()
        if e.get('decision')!='accepted' or not isinstance(e.get('rationale'),str) or not e['rationale'].strip():raise ValueError('Question not accepted: '+question)
        if not path.is_relative_to(base) or not path.is_file() or not path.stat().st_size or task.file_hash(path)!=e['sha256']:raise ValueError('Missing or changed review evidence')
        receipts[question]=e['sha256']
    return {'primary_accepted':True,'review_sha256':review['sha256'],'receipt_hashes':receipts,
            'external_testimony_authenticated_by_code':False,'machine_replay_sha256':recipes.digest(task.fingerprint(replay))}


def human_job(job,bundles,out,context):
    name=job['recipe_id']
    if name=='human_threshold_calibration':
        key='human_threshold/'+job['corpus']+'/'+job['encoder']
        dossier=bundles[key];features=dossier['features'];records=dossier['records'];provenance=dossier['provenance']
        upstream=None
        if context['mode']==context['dispatch_module'].PRIMARY_MODE:
            from phase6.acceptance import accept_calibration_inputs
            upstream=accept_calibration_inputs(job['corpus'],job['encoder'],dossier,base_dir=context['base_dir'],design=context['design'])
            write(out/'source_cache_replay.json',upstream)
            if not upstream['machine_acceptance_passed']:raise ValueError('Calibration original-source/cache replay failed')
        selection=dossier['selection_manifest']
        frame=cal._check_inputs(features,records,provenance,selection['frame']['scorer']['block_size'])
        if frame!=selection['frame']:raise ValueError('Actual calibration frame does not match selection manifest')
        reconstructed=cal.prepare_selection(features,records,provenance,seed=selection['seed'],block_size=selection['frame']['scorer']['block_size'])
        if reconstructed!=selection:raise ValueError('Selection pairs do not reproduce the fixed sampling law')
        lock=cal.select_threshold(selection,dossier['selection_responses'])
        result={'selection_lock':lock,'stage':job['stage'],'source_cache_replay':upstream}
        if job['stage']!='selection':
            if job['stage']=='sensitivity_validation':
                delta=-.02 if job['threshold_sign']=='minus' else .02
                expected=max(0.,min(1.,float(lock['threshold'])+delta))
                entry={'threshold_lock':lock,'calibration':dossier,
                       'sensitivity_quality':dossier['sensitivity_quality']}
                result['quality_report']=task.sensitivity_quality(entry,expected)
            else:
                validation=dossier['validation_manifest'];responses=dossier['validation_responses']
                if validation['frame']!=frame:raise ValueError('Validation frame differs from actual corpus/encoder')
                rebuilt=cal.prepare_validation(features,records,lock,seed=validation['seed'],block_size=selection['frame']['scorer']['block_size'])
                if rebuilt!=validation:raise ValueError('Validation pairs do not reproduce independent sampling')
                result['quality_report']=cal.quality_gate(validation,responses,lock)
        accepted=_narrow_review(dossier,result,context,('original_frame_origin','encoder_origin','genuine_independent_human_collection','permissions_and_content_use'))
        selection_passed=lock['selection_status']=='qualified_development_only'
        quality_passed=result.get('quality_report',{}).get('statistical_and_declared_scope_eligible',result.get('quality_report',{}).get('quality_pass',True))
        status='selection_gate_failed' if not selection_passed else ('completed' if quality_passed else 'quality_gate_failed')
        result.update(analysis_completed=True,selection_gate_passed=selection_passed,quality_gate_passed=quality_passed if job['stage']!='selection' else None)
    else:
        from phase5.human_analysis import analyze_human_audit
        from phase6.human_inference import analyze_intervals
        key='human_admission' if name=='human_admission_main' else 'human_context'
        dossier=bundles[key];manifest=dossier['manifest'];responses=dossier['responses']
        corpus_inputs=dossier['corpus_inputs']
        for corpus,inputs in corpus_inputs.items():
            source_bundle=bundles[corpus+'/primary-10000']
            source_job={'recipe_id':name,'corpus':corpus,'panel':'primary-10000','methods':['independent_structure_oracle']}
            group=_group(source_job,source_bundle,context['mode']);dispatch=context['dispatch_module']
            accepted_source=dispatch.acceptance(group,source_bundle,context['design'],mode=context['mode'],base_dir=context['base_dir'])
            if not accepted_source['execution_allowed']:raise ValueError('Human frame source inputs failed acceptance')
            cell=dispatch.resolve(group,source_bundle)
            if rq.graph_binding(inputs['graph'])!=rq.graph_binding(cell['graph']):raise ValueError('Human frame graph differs from accepted graph')
            if not np.array_equal(inputs['features'],cell['cx']):raise ValueError('Human frame vectors differ from accepted vectors')
            if inputs['records']!=source_bundle['train_records'] or inputs['request_manifest']!=source_bundle['requests']:
                raise ValueError('Human frame records or complete request manifest differ from accepted inputs')
        if context['mode']==context['dispatch_module'].PRIMARY_MODE and set(corpus_inputs)!=set(recipes.CORE):
            raise ValueError('All three genuine corpus frames are required for primary human interpretation')
        config=manifest['configuration']
        if name=='human_admission_main':
            from phase4.admission_audit import prepare_admission_audit
            rebuilt=prepare_admission_audit(corpus_inputs,master_seed=config['master_seed'],evidence_role=manifest['evidence_role'])
        else:
            from phase5.context_audit import prepare_context_audits
            rebuilt=prepare_context_audits(corpus_inputs,master_seed=config['master_seed'],evidence_role=manifest['evidence_role'],
                target_character_limit=config['target_character_limit'],context_character_limit=config['context_character_limit'])
        if rebuilt!=manifest:raise ValueError('Human sampling frame does not reproduce accepted upstream inputs')
        analysis=analyze_human_audit(manifest,responses)
        intervals=analyze_intervals(manifest,responses,alpha=Fraction(1,40))
        result={'analysis':analysis,'intervals':intervals,'context_kind':job['job_id'].split('/')[-1] if name=='human_admission_context' else None}
        accepted=_narrow_review(dossier,result,context,('original_frame_origin','genuine_independent_human_collection','permissions_and_content_use'))
        expected=analysis['expected_assignment_count'];completed=analysis['declared_completed_assignment_count']
        skipped=sum(row['skipped_assignments'] for row in analysis['unit_results'])
        substantive=sum(row['substantive_assignments'] for row in analysis['unit_results'])
        missing=expected-completed
        result.update(analysis_completed=True,collection_complete=expected>0 and missing==0,
            expected_assignments=expected,completed_assignments=completed,substantive_assignments=substantive,
            skipped_assignments=skipped,missing_assignments=missing)
        status=('structural_zero_empty_annotation_frame' if expected==0 else 'completed' if missing==0 else
                'partial_human_collection' if completed else 'awaiting_real_human_responses')
    write(out/'human_analysis.json',result);write(out/'acceptance.json',accepted)
    return {'status':status,'artifact':'human_analysis.json','primary_output_accepted':accepted['primary_accepted'] and status=='completed',
            'external_testimony_authenticated_by_code':False}


def statistics_job(job,bundles,out,context):
    dossier=bundles['statistics'];registry=dossier['registry'];records=dossier['records']
    from phase6.statistics_evidence import bind_statistics_evidence
    source_check=bind_statistics_evidence(dossier,context)
    if registry['replicates']!=10000 or registry['seed']!=20271004:raise ValueError('Locked10000 resamples and seed required')
    result={'input_replay':source_check,'paired':statistics.summarize_registry(registry,records,metrics=dossier['metrics']),
            'crossed':{},'systems':{},'Holm':{}}
    for name,inputs in dossier['crossed'].items():
        result['crossed'][name]=({'status':'unavailable_from_preserved_failed_or_missing_cells',
            'reason':source_check['crossed_unavailable'][name]} if name in source_check['crossed_unavailable'] else
            statistics.crossed_source_bootstrap(**inputs,seed=20271004,replicates=10000))
    for corpus,inputs in dossier['systems'].items():
        result['systems'][corpus]=({'status':'unavailable_from_preserved_failed_or_missing_cells',
            'reason':source_check['systems_unavailable'][corpus]} if corpus in source_check['systems_unavailable'] else
            statistics.equal_rs_lifecycle_bootstrap(**inputs,seed=20271004,replicates=10000))
    for family in ('signed_loss','equal_RS_lifecycle'):
        result['Holm'][family]=statistics.holm_family(dossier['valid_p_values'][family],family=family)
    if context['mode']==context['dispatch_module'].PRIMARY_MODE:
        if set(result['crossed'])!={'civil_comments:R','civil_comments:S','askubuntu:R','askubuntu:S'}:raise ValueError('All four crossed-source analysis cells required')
        if set(result['systems'])!=set(recipes.LABELED):raise ValueError('Both systems corpus analyses required')
    accepted=_narrow_review(dossier,result,context,('preoutcome_analysis_lock','complete_primary_run_evidence','test_source_origin','valid_p_value_derivation'))
    missing_endpoints=sum(not metric['full_registry_estimand_identified']
        for group in result['paired']['groups'] for metric in group['metrics'].values())
    unavailable=bool(missing_endpoints or source_check['crossed_unavailable'] or source_check['systems_unavailable'])
    result.update(analysis_completed=True,unidentified_paired_endpoints=missing_endpoints,
                  complete_required_estimands_available=not unavailable)
    write(out/'statistics.json',result);write(out/'acceptance.json',accepted)
    return {'status':'completed_with_unavailable_estimands' if unavailable else 'completed',
            'analysis_completed':True,'artifact':'statistics.json',
            'primary_output_accepted':accepted['primary_accepted'] and not unavailable,
            'unidentified_paired_endpoints':missing_endpoints,
            'missing_rows_dropped':False,'intervals_called_p_values':False}


def run_extension(job,bundles,out,context):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);name=job['recipe_id'];dispatch=context['dispatch_module']
    row={'job_id':job['job_id'],'recipe_id':name,'primary_output_accepted':False}
    if name.startswith('human_'):result=human_job(job,bundles,out,context)
    elif name=='test_source_bootstrap':result=statistics_job(job,bundles,out,context)
    else:
        bundle=bundles[job['corpus']+'/'+job['panel']];group=_group(job,bundle,context['mode'])
        if context['mode']==dispatch.PRIMARY_MODE and bundle.get('execution_policy')!=context['policy']:
            raise ValueError('Extension inputs do not bind common resource policy')
        before=task.fingerprint(bundle)
        cache=context.setdefault('extension_cache',{})
        key=_cache_key(before,group,context)
        began=time.perf_counter()
        if key not in cache:
            accepted=dispatch.acceptance(group,bundle,context['design'],mode=context['mode'],base_dir=context['base_dir'])
            if not accepted['execution_allowed']:
                write(out/'acceptance.json',accepted)
                raise ValueError('Extension input acceptance failed')
            cell=dispatch.resolve(group,bundle)
            cell['_validated_source_bindings']=accepted.get('validated_source_bindings')
            cache[key]={'accepted':accepted,'cell':cell,'owner_job_id':job['job_id'],
                        'preparation_seconds':time.perf_counter()-began,'prepared_input_fingerprint':before}
        item=cache[key];cell=item['cell'];accepted=deepcopy(item['accepted']);accepted['group_id']=group['group_id']
        accepted['shared_acceptance_owner_job']=item['owner_job_id']
        write(out/'acceptance.json',accepted)
        preparation={'cache_key':key,'acceptance_and_geometry_owner_job':item['owner_job_id'],
                     'acceptance_and_geometry_seconds':item['preparation_seconds'],
                     'shared_not_service_state':True,'charge_once_in_deployment_total':True,
                     'execution_input_scope':'loaded accepted arrays and metadata; original archives and encoders are not reread',
                     'loaded_snapshot_rechecked_after_each_job':True,
                     'unused_source_files_rehashed_per_job':False}
        if name in {'cold_warm_scope','every_release_persistence','common_cached_graph_oracle'} and '_prepared_bundle_path' not in cell:
            from phase6 import run_isolated as isolated
            folder=tempfile.TemporaryDirectory(prefix='extension_shared_',dir=dispatch.ROOT/'tmp')
            cache.setdefault('_temporary_directories',[]).append(folder)
            package=isolated.prepare_bundle(cell['cx'],cell['x'],cell['y'],bundle['train_ids'],Path(folder.name)/'inputs',
                threshold=cell['graph'].threshold,source_ids=bundle['source_ids'],priority_seed=cell['priority_seed'])
            cell['_prepared_bundle_path']=str(package);item['service_package_owner_job']=job['job_id']
            item['service_package_charge']=json.loads((package/'bundle.json').read_text())
        if 'service_package_owner_job' in item:
            preparation.update(service_package_owner_job=item['service_package_owner_job'],service_package_charge=item['service_package_charge'])
        write(out/'shared_preparation.json',preparation)
        if name=='negative_controls':result=negative_control(job,bundle,cell,out,dispatch)
        elif name=='utility_reference':
            rows=utility_panel(bundle,cell,out,dispatch)
            write(out/'utility.json',rows)
            result={'status':'completed' if all(r['status']=='completed' for r in rows) else 'method_noncompletion','artifact':'utility.json',
                    'methods':[{'method':r['method'],'status':r['status']} for r in rows]}
        else:
            path=extension_path(job,bundle,cell);write(out/'requests.json',path)
            if not path['checkpoints']:
                result={'status':'structural_zero_or_unavailable_arm','reason':path['status'],'checkpoint_count':0}
            elif name in {'structure_mass_PPS','structure_one_percent_horizon','excluded_blocker_stress','news_chronological','fresh_graph_audit'}:
                result=dispatch.structural_job(job,path,cell,out)
            elif name in {'cold_warm_scope','every_release_persistence','common_cached_graph_oracle'}:
                execution=deepcopy(job)
                if name=='cold_warm_scope':execution['panel_mode']='warm_service'
                elif name=='every_release_persistence':execution.update(panel_mode='every_release',persistence='every_release')
                else:execution['methods']=['O-G-cached']
                result=dispatch.method_job(execution,path,group,bundle,cell,out,context['policy'])
                if name=='common_cached_graph_oracle':
                    for method in result.get('methods',[]):
                        method.update(method='O-G',implementation='O-G-cached',oracle_tier='cached_exhaustive_graph')
            elif name=='graph_envelope_conditional':result=envelope_job(job,path,cell,out,dispatch)
            elif name=='head_normalization_error_diagnostic':
                from ccu.core import ridge_moments
                k=path['checkpoints'][-1];graph=cell['graph'];ids=list(graph.record_ids)
                dead=boundary.deleted_indices(path,k,graph);selected=set(map(int,graph.selected_indices([ids[i] for i in dead])))
                initial_count=len(graph.selected_indices());ix=np.asarray(sorted(selected),dtype=int)
                mom=ridge_moments(cell['x'][ix],cell['y'][ix]);correct=dispatch.dual_head(cell['x'],cell['y'],selected,cell['lambda'])
                wrong=np.linalg.solve(mom.gram+cell['lambda']*initial_count*np.eye(cell['x'].shape[1]),mom.cross) if initial_count else np.zeros_like(mom.cross)
                error=float(np.linalg.norm(wrong-correct));np.savez(out/'normalization.npz',correct=correct,wrong=wrong)
                report={'checkpoint':k,'initial_selected_count':initial_count,'current_selected_count':len(selected),
                        'head_error_frobenius':error,'zero_count_change':initial_count==len(selected),
                        'wrong_implementation_is_not_competitor':True}
                write(out/'normalization.json',report);result={'status':'completed','artifact':'normalization.json',**report}
            else:raise ValueError('Unknown resolved recipe executor: '+name)
        if task.fingerprint(bundle)!=before:raise ValueError('Accepted extension input mutated during execution')
        result['shared_preparation']='shared_preparation.json'
        result['primary_output_accepted']=accepted['primary_inputs_accepted'] and result['status']=='completed'
    row.update(result)
    if 'methods' not in row:row['methods']=[{'method':m,'status':row['status']} for m in job['methods']]
    return row
