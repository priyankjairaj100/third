"""Complete-cell task/utility executor for frozen registry groups.

This evaluator computes ridge oracle and wrong-target diagnostics. It is not a
repair service, a systems benchmark, or authorization of primary publication.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
import argparse,gzip,hashlib,json,math,platform
from fractions import Fraction
import numpy as np
from scipy.special import expit
from ccu.core import ridge_moments
from phase3 import calibration as cal
from phase3.reference_graph import build_reference_graph
from phase4.execution import array_hash,common_decode,canonical,write
from phase4.requests import verify_manifest
from phase5 import task_metrics as metrics
from phase5.study_registry import digest as registry_digest
from phase5.boundary import deleted_indices
from phase5.convex_multioutput import solve_multioutput,certify_multioutput
from empirical_execution.phase4 import model_selection

SCHEMA='ccu-task-program-1'
ENGINEERING_LOCK='ccu-task-engineering-parameter-1'
SUPPORTED_BLOCKS={'B_relevance','D_replication','G_boundary'}
LOGISTIC_POLICY={'gradient_tolerance':1e-11,'max_iterations':100,'max_backtracks':60,
 'parameter_radius':1e-8,'certificate_max_coordinates':2_000_000,
 'selection':'same initially curated set','lambda':'same frozen ridge lambda; no logistic retuning'}


def fingerprint(value):
    if isinstance(value,np.ndarray):return {'array_sha256':array_hash(value),'shape':list(value.shape),'dtype':str(value.dtype)}
    if isinstance(value,dict):return {k:fingerprint(v) for k,v in sorted(value.items())}
    if isinstance(value,(list,tuple)):return [fingerprint(v) for v in value]
    return value


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def code_bindings():
    modules=[sys.modules[__name__],metrics,sys.modules[solve_multioutput.__module__],sys.modules[certify_multioutput.__module__],sys.modules[build_reference_graph.__module__],model_selection]
    result={str(Path(m.__file__).resolve().relative_to(ROOT)):file_hash(m.__file__) for m in modules}
    for name in ('ccu/core.py','phase3/panels.py','phase3/calibration.py','phase4/convex.py','phase4/requests.py','phase5/boundary.py','phase5/statistics.py','phase5/study_registry.py'):
        result[name]=file_hash(ROOT/name)
    return result


def engineering_parameter(kind,value,*,representation,source='prospective engineering constant, not calibrated'):
    if kind not in {'threshold','lambda'} or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Finite named engineering parameter required')
    if (kind=='threshold' and not -1<=value<=1) or (kind=='lambda' and value<=0):raise ValueError('Invalid parameter range')
    return cal._seal({'schema':ENGINEERING_LOCK,'kind':kind,'value':float(value),'representation':representation,
      'role':'engineering_nonconfirmatory','source':source,'selected_from_test':False,'is_calibration_result':False})


def _parameter(entry,kind,representation):
    lock=entry[kind+'_lock'];cal._verify(lock)
    if lock.get('schema')==ENGINEERING_LOCK:
        if lock.get('kind')!=kind or lock.get('representation')!=representation or lock.get('role')!='engineering_nonconfirmatory' or lock.get('selected_from_test') is not False:raise ValueError('Engineering lock mismatch')
        return float(lock['value']),lock['sha256']
    expected_encoder={'e5':'intfloat/multilingual-e5-base','mpnet':'sentence-transformers/all-mpnet-base-v2'}.get(representation.split(':')[0])
    if expected_encoder is None or entry.get('encoder_id')!=expected_encoder:
        raise ValueError('Real calibration lock requires matching named feature encoder provenance')
    if kind=='lambda':
        dossier=entry.get('calibration')
        if not isinstance(dossier,dict):raise ValueError('Actual calibration features/records/provenance required to recheck lambda')
        if dossier['provenance'].get('encoder_id')!=entry['encoder_id'] or dossier['provenance'].get('encoder_revision')!=entry.get('encoder_revision'):
            raise ValueError('Lambda calibration belongs to a different encoder/revision')
        rebuilt=model_selection.select_ridge_regularization(dossier['features'],dossier['records'],dossier['provenance'],vocabulary_lock=dossier.get('vocabulary_lock'))
        if rebuilt!=lock:raise ValueError('Lambda lock differs from recomputed calibration-only selection')
        return float(lock['lambda']),lock['sha256']
    dossier=entry.get('calibration')
    if not isinstance(dossier,dict):raise ValueError('Actual curator calibration/quality dossier required')
    if dossier['provenance'].get('encoder_id')!=entry['encoder_id'] or dossier['provenance'].get('encoder_revision')!=entry.get('encoder_revision'):
        raise ValueError('Curator calibration belongs to a different encoder/revision')
    selection=dossier['selection_manifest'];validation=dossier['validation_manifest']
    frame=cal._check_inputs(dossier['features'],dossier['records'],dossier['provenance'],selection['frame']['scorer']['block_size'])
    if selection['frame']!=frame or validation['frame']!=frame or lock['frame']!=frame:raise ValueError('Curator calibration frame mismatch')
    rebuilt=cal.select_threshold(selection,dossier['selection_responses'])
    quality=cal.quality_gate(validation,dossier['validation_responses'],lock)
    if rebuilt!=lock or quality!=dossier['quality_report'] or not quality['statistical_and_declared_scope_eligible']:raise ValueError('Curator threshold/quality gate does not recompute')
    return float(lock['threshold']),lock['sha256']


def bound_array(x,n,name):
    x=np.asarray(x)
    if x.dtype!=np.float32 or x.ndim!=2 or len(x)!=n or x.shape[1]<1 or not np.isfinite(x).all():raise ValueError('Finite aligned stored FP32 '+name+' required')
    return x


def validate_bundle(bundle):
    ids=bundle['train_ids'];eval_ids=bundle['evaluation_ids'];sources=bundle['source_ids']
    if not ids or not eval_ids or any(not isinstance(i,str) or not i for i in ids+eval_ids) or len(set(ids+eval_ids))!=len(ids)+len(eval_ids):raise ValueError('Distinct nonempty training/evaluation ID populations required')
    if len(sources)!=len(ids) or any(not isinstance(s,str) or not s for s in sources):raise ValueError('Complete training source ownership required')
    y=np.asarray(bundle['train_y']);ey=np.asarray(bundle['evaluation_y'])
    if y.ndim==1:y=y[:,None]
    if ey.ndim==1:ey=ey[:,None]
    if y.dtype!=np.float64 or ey.dtype!=np.float64 or y.ndim!=2 or ey.ndim!=2 or y.shape[0]!=len(ids) or ey.shape!=(len(eval_ids),y.shape[1]) or not all(np.isfinite(a).all() for a in (y,ey)):raise ValueError('Aligned original FP64 task targets required')
    if np.any(y<0) or np.any(y>1) or np.any(ey<0) or np.any(ey>1):raise ValueError('Fractional or multihot task targets in [0,1] required')
    if bundle['task']=='civil' and y.shape[1]!=1:raise ValueError('Civil uses one toxicity fraction')
    if bundle['task']=='multilabel' and (not np.isin(y,[0,1]).all() or not np.isin(ey,[0,1]).all()):raise ValueError('Multihot target indicators required')
    if bundle['task'] not in {'civil','multilabel'}:raise ValueError('Unsupported labeled task')
    for entry in bundle['curators'].values():
        if entry.get('train_ids')!=ids:raise ValueError('Curator cache row-ID order differs from frozen training panel')
    for entry in bundle['learners'].values():
        if entry.get('train_ids')!=ids or entry.get('evaluation_ids')!=eval_ids:raise ValueError('Learner/evaluation cache row-ID order differs')
    base=bundle['curators'][bundle['base_curator']];cx=bound_array(base['train'],len(ids),'base curator')
    tau,_=_parameter(base,'threshold',bundle['base_curator'])
    graph=build_reference_graph(cx,ids,tau,sources,seed=bundle.get('base_priority_seed',0))
    verify_manifest(bundle['requests'],graph)
    # A calibration-only chooser cannot use either population evaluated here.
    for entry in list(bundle['curators'].values())+list(bundle['learners'].values()):
        dossier=entry.get('calibration')
        if dossier and set(r['record_id'] for r in dossier['records']) & set(ids+eval_ids):raise ValueError('Calibration records overlap training/evaluation')
    return y,ey,graph


def _threshold(base,setting):
    if setting=='calibrated':return base
    if setting=='max(0,calibrated-0.02)':return max(0.,base-.02)
    if setting=='min(1,calibrated+0.02)':return min(1.,base+.02)
    if isinstance(setting,str) and setting.startswith('calibrated'):
        offset=float(setting[len('calibrated'):]);value=base+offset
    elif isinstance(setting,(int,float)) and not isinstance(setting,bool):value=float(setting)
    else:raise ValueError('Unsupported frozen threshold configuration')
    if not math.isfinite(value) or not -1<=value<=1:raise ValueError('Threshold sensitivity exits [-1,1]; cell retained as invalid, not clipped')
    return value


def prepare_sensitivity_validation(entry,tau,*,seed=2027100407):
    main=entry['threshold_lock'];cal._verify(main)
    if main.get('schema')!='ccu-calibration-selection-lock-1':raise ValueError('Actual parent calibration selection lock required')
    if tau not in {max(0.,main['threshold']-.02),min(1.,main['threshold']+.02)}:raise ValueError('Only the two predeclared threshold sensitivities are allowed')
    dossier=entry['calibration'];block=main['frame']['scorer']['block_size']
    frame=cal._check_inputs(dossier['features'],dossier['records'],dossier['provenance'],block)
    if frame!=main['frame']:raise ValueError('Sensitivity calibration frame changed')
    fixed=cal._seal({'schema':'ccu-fixed-threshold-sensitivity-lock-1','parent_selection_lock_sha256':main['sha256'],
      'frame':frame,'threshold':float(tau),'parent_selection_status':main['selection_status'],
      'primary_promotion_permitted':False})
    rng=cal._rng(seed,['threshold-sensitivity',fixed['sha256']]);sample=[];population=0;audit={}
    for i,j,score in cal.iter_pair_scores(dossier['features'],block,audit):
        if score>tau:
            population+=1;cal._reservoir_insert(sample,(i,j,score,0),population,200,rng)
    probabilities={0:Fraction(min(200,population),population)} if population else {}
    context=cal.digest([frame,'threshold-sensitivity',seed,fixed['sha256']])
    pairs=cal._sample_pairs(sample,dossier['records'],'quality_validation',context,probabilities)
    previous=set(main['selection_natural_pair_ids'])
    return cal._seal({'schema':'ccu-calibration-sensitivity-sample-1','protocol_id':cal.PROTOCOL,
      'role':'quality_validation','frame':frame,'seed':seed,'fixed_sensitivity_lock':fixed,
      'threshold':float(tau),'pair_population':population,'sample_count':len(pairs),'census':len(pairs)==population,
      'boundary_score_audit':audit,'sampling':'independent reservoir SRS from complete alternate-above-threshold population',
      'overlap_natural_pair_count':sum(p['natural_pair_id'] in previous for p in pairs),
      'fresh_blinded_assignments':True,'pairs':pairs,'primary_promotion_permitted':False})


def sensitivity_quality(entry,tau):
    main=entry['threshold_lock']
    if main.get('schema')==ENGINEERING_LOCK:
        return {'audited':False,'status':'engineering_lexical_threshold_without_semantic_quality_claim'}
    spec=entry.get('sensitivity_quality',{}).get(float(tau).hex())
    if not isinstance(spec,dict):raise KeyError('Missing separate sensitivity-quality sampling/responses for '+float(tau).hex())
    supplied=spec['validation_manifest'];cal._verify(supplied)
    expected=prepare_sensitivity_validation(entry,tau,seed=supplied['seed'])
    if supplied!=expected:raise ValueError('Sensitivity validation not reconstructed from fixed alternate threshold')
    labels,annotation=cal._majority_labels(supplied,spec['responses'])
    bound=cal.finite_population_lower_bound(supplied['pair_population'],len(supplied['pairs']),sum(labels.values()))
    passed=bool(bound['defined'] and 10*bound['lower_success_count']>=9*bound['N'])
    return {'audited':True,'status':'passed_secondary_quality' if passed else 'failed_secondary_quality',
      'quality_pass':passed,'bound':bound,'annotation_report':annotation,'validation_sha256':supplied['sha256'],
      'threshold':float(tau),'fixed_sensitivity_lock_sha256':supplied['fixed_sensitivity_lock']['sha256'],'main_gate_not_replaced':True}


def seeded_projection(source_dimension,target_dimension,seed):
    if (isinstance(seed,bool) or not isinstance(seed,int) or seed<0 or not 1<=target_dimension<=source_dimension):
        raise ValueError('Valid frozen projection dimension/seed required')
    matrix=np.random.Generator(np.random.PCG64(seed)).standard_normal((source_dimension,target_dimension))
    q,r=np.linalg.qr(matrix,mode='reduced');q*=np.where(np.diag(r)<0,-1.,1.)
    return q

def resolve_cell(group,bundle):
    c=group['configuration'];ids=bundle['train_ids'];ne=len(bundle['evaluation_ids'])
    if c.get('state_precision','FP64')!='FP64' or c.get('decoder','Cholesky')!='Cholesky':raise NotImplementedError('FP32/CG state methods belong to their separate method executor; no silent FP64 substitution')
    curator=bundle['curators'][c['curator_encoder']];cx=bound_array(curator['train'],len(ids),'curator')
    tau,threshold_hash=_parameter(curator,'threshold',c['curator_encoder']);setting=c.get('threshold','calibrated');tau=_threshold(tau,setting)
    quality=sensitivity_quality(curator,tau) if setting!='calibrated' else {'status':'base_threshold_contract'}
    learner=bundle['learners'][c['learner_encoder']];x=bound_array(learner['train'],len(ids),'learner');ex=bound_array(learner['evaluation'],ne,'evaluation')
    if x.shape[1]!=ex.shape[1]:raise ValueError('Training/evaluation feature dimensions differ')
    d=c['learner_dimension'];representation=c['learner_encoder'];entry=learner;projection_hash=None
    if d!=x.shape[1]:
        spec=bundle.get('projections',{}).get(str(d))
        if not isinstance(spec,dict):raise KeyError('Missing fixed projection '+str(d))
        p=np.asarray(spec['matrix'])
        if p.dtype!=np.float64 or p.shape!=(x.shape[1],d) or not np.isfinite(p).all() or not np.allclose(p.T@p,np.eye(d),rtol=0,atol=1e-12):raise ValueError('Declared fixed orthonormal FP64 projection required')
        if spec.get('seed')!=c.get('projection_seed') or spec.get('data_independent') is not True:raise ValueError('Projection seed/data-independence declaration mismatch')
        if not np.array_equal(p,seeded_projection(x.shape[1],d,spec['seed'])):raise ValueError('Projection differs from prospective PCG64 Gaussian signed-QR convention')
        projection_hash=array_hash(p);x=(x.astype(np.float64)@p).astype(np.float32);ex=(ex.astype(np.float64)@p).astype(np.float32)
        representation+=':projection-'+str(d);entry=learner['projection_parameters'][str(d)]
        if entry['lambda_lock'].get('schema')!=ENGINEERING_LOCK:
            expected_cal=(np.asarray(learner['calibration']['features'],dtype=np.float64)@p).astype(np.float32)
            if not np.array_equal(expected_cal,entry['calibration']['features']):raise ValueError('Projection calibration features differ from fixed projection')
    lam,lambda_hash=_parameter(entry,'lambda',representation);factor=c.get('lambda_factor',1.)
    if isinstance(factor,bool) or not isinstance(factor,(float,int)) or not math.isfinite(factor) or factor<=0:raise ValueError('Positive fixed lambda factor required')
    lam*=factor
    thresholds=None;threshold_status='not_applicable'
    if bundle['task']=='multilabel':
        tlock=entry.get('tag_threshold_lock')
        if tlock is not None:
            model_selection.verify(tlock)
            if tlock.get('kind')!='tag_decision_thresholds' or tlock.get('lambda_lock_sha256')!=lambda_hash:raise ValueError('Tag decision thresholds must bind selected lambda lock')
            if model_selection.select_tag_decision_thresholds(entry['lambda_lock'])!=tlock:raise ValueError('Tag thresholds do not reconstruct from locked calibration OOF scores')
            thresholds=np.asarray([np.inf if r['kind']=='never_positive' else float(r['threshold']) for r in tlock['thresholds']],dtype=np.float64)
            threshold_status='locked_calibration_thresholds'
        else:threshold_status='unavailable_missing_calibration_tag_threshold_lock'
    graph=build_reference_graph(cx,ids,tau,bundle['source_ids'],seed=c.get('priority_seed',0))
    return x,ex,graph,lam,thresholds,{'threshold':tau,'lambda':lam,'threshold_lock_sha256':threshold_hash,
      'lambda_lock_sha256':lambda_hash,'projection_sha256':projection_hash,'projection_output_sha256':array_hash(x),
      'tag_threshold_status':threshold_status,'threshold_quality':quality,'reference_graph_code':file_hash(sys.modules[build_reference_graph.__module__].__file__)}


def task_metrics(bundle,y,pred,thresholds):
    if bundle['task']=='civil':return metrics.civil_metrics(y,pred)
    if thresholds is None:
        return {'task':'original_multihot_tags','mse':float(np.mean((y-pred)**2)),
          'micro_f1':None,'macro_f1':None,'mean_average_precision':float(np.mean([metrics.binary_rank_metrics(y[:,i],pred[:,i])['average_precision'] for i in range(y.shape[1])])),
          'threshold_metrics_unavailable':'missing calibration-only decision lock; no test threshold fitted'}
    return metrics.multilabel_metrics(y,pred,thresholds)


def ridge_predictions(x,y,selected,ex,lam):
    ix=np.asarray(sorted(selected),dtype=int);moments=ridge_moments(x[ix],y[ix]);head=common_decode(moments,lam).weights
    if selected:
        matrix=moments.gram+lam*len(selected)*np.eye(x.shape[1])
        residual=float(np.linalg.norm(matrix@head-moments.cross))
        denominator=float(np.linalg.norm(matrix)*np.linalg.norm(head)+np.linalg.norm(moments.cross))
        eta=residual/denominator if denominator else (0. if residual==0 else float('inf'))
        if not math.isfinite(eta) or eta>1e-10:raise RuntimeError('Fixed ridge eta gate failed: '+repr(eta))
    return head,ex.astype(np.float64)@head


def utility_panel(bundle,x,ex,y,ey,graph,lam,thresholds,out):
    initial=set(map(int,graph.selected_indices()));ids=bundle['train_ids']
    ordered=sorted(range(len(ids)),key=lambda i:(hashlib.sha256(('ccu-v1-utility-hash\0'+ids[i]).encode()).digest(),ids[i]))
    rows=[];before=None
    for name,selected in [('curated_ridge',initial),('uncurated_ridge',set(range(len(ids)))),('same_size_hash_ridge',set(ordered[:len(initial)]))]:
        head,pred=ridge_predictions(x,y,selected,ex,lam)
        np.savez(out/(name+'.npz'),weights=head,predictions=pred,selected_indices=np.asarray(sorted(selected),dtype=np.int64))
        rows.append({'method':name,'selected_records':len(selected),'status':'completed','metrics':task_metrics(bundle,ey,pred,thresholds)})
        if name=='curated_ridge':before=pred
    row={'method':'curated_logistic','selected_records':len(initial),'status':'started'}
    try:
        ix=np.asarray(sorted(initial),dtype=int);fit=solve_multioutput(x[ix],y[ix],lam,**{k:LOGISTIC_POLICY[k] for k in ('gradient_tolerance','max_iterations','max_backtracks')})
        pred=expit(ex.astype(np.float64)@fit['weights'])
        np.savez(out/'curated_logistic.npz',weights=fit['weights'],predictions=pred,selected_indices=ix)
        row.update({'status':'candidate_saved_before_certificate',
          'metrics':task_metrics(bundle,ey,pred,None if bundle['task']=='multilabel' else thresholds),
          'logistic_task_thresholds':'unavailable; ridge thresholds are not silently applied to logistic probabilities' if bundle['task']=='multilabel' else 'Civil fixed fractional-toxicity endpoint',
          'output_solver_statuses':[r['status'] for r in fit['output_results']],
          'meter':fit['meter']})
        cert=certify_multioutput(x[ix],y[ix],lam,fit['weights'],max_coordinates=LOGISTIC_POLICY['certificate_max_coordinates'],parameter_tolerance=Fraction(1,10**8))
        row.update(certificate=cert,status='completed' if cert['meets_parameter_tolerance'] else 'certificate_failed')
    except Exception as error:row.update(status='failed',exception_type=type(error).__name__,message=str(error))
    rows.append(row);return before,initial,rows


def task_groups(registry):
    return [g for g in registry['groups'] if g['block'] in SUPPORTED_BLOCKS and g['corpus'] in {'civil_comments','askubuntu','english_stackexchange'}]


def run_task_program(registry,jobs,bundles,output_dir,*,evidence_role='engineering_nonconfirmatory'):
    if evidence_role!='engineering_nonconfirmatory':raise RuntimeError('Primary promotion refused: this evaluator requires separate authentic-data/whole-study acceptance and executes no repair-service timing')
    if registry.get('jobs_content_sha256')!=registry_digest(jobs):raise ValueError('Registry job inventory digest mismatch')
    out=Path(output_dir)
    if out.exists():raise FileExistsError('Frozen task program output exists')
    group_ids=[g['group_id'] for g in registry['groups']]
    if len(set(group_ids))!=len(group_ids) or len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Unique registry group/job IDs required')
    groups=task_groups(registry);applicable={g['group_id'] for g in groups};planned=[j for j in jobs if j.get('group_id') in applicable]
    out.mkdir(parents=True,exist_ok=False)
    lock={'schema':SCHEMA,'role':evidence_role,'primary_study':False,'registry_sha256':cal.digest(registry),
      'jobs_content_sha256':registry['jobs_content_sha256'],'task_group_ids':[g['group_id'] for g in groups],
      'planned_task_jobs':[j['job_id'] for j in planned],'bundle_bindings':fingerprint(bundles),
      'code_bindings':code_bindings(),'numpy':np.__version__,'python':platform.python_version(),
      'ridge_solver':'same FP64 Cholesky on fresh moments; eta <= 1e-10; no rigorous state certificate',
      'logistic_policy':LOGISTIC_POLICY,'same_size_hash_salt':'ccu-v1-utility-hash',
      'evaluation_scope':'oracle/frozen task effects only; methods beyond O-G/B-F retained as outside this evaluator',
      'predictions':'stored FP32 evaluation features promoted FP64, ridge scores never clipped; logistic expit scores',
      'missing_or_failed_cells':'every planned task job retained; no outcome-adaptive reroll or configuration omission'}
    write(out/'run_lock.json',lock);ledger=[];checkpoint_rows=[];group_ledger=[]
    for gi,group in enumerate(groups):
        group_dir=out/f'group_{gi:03d}';group_dir.mkdir();gjobs=[j for j in planned if j['group_id']==group['group_id']]
        grow={'group_id':group['group_id'],'configuration':group['configuration'],'requested_methods':group['methods'],
          'executed_task_methods':['O-G','B-F'],'other_requested_methods_outside_this_evaluator':[m for m in group['methods'] if m not in {'O-G','B-F'}]}
        try:
            bundle=bundles[group['request_family']]
            if bundle.get('dataset_id')!=group['corpus']:raise ValueError('Corpus/bundle mismatch')
            y,ey,base=validate_bundle(bundle);x,ex,graph,lam,thresholds,geometry=resolve_cell(group,bundle)
            before,initial,utility=utility_panel(bundle,x,ex,y,ey,graph,lam,thresholds,group_dir)
            write(group_dir/'utility.json',{'group_id':group['group_id'],'geometry':geometry,'references':utility})
            write(group_dir/'evaluation_ids.json',bundle['evaluation_ids']);np.save(group_dir/'evaluation_y.npy',ey,allow_pickle=False)
            byname={p['trajectory_id']:p for p in bundle['requests']['trajectories']}
            same_graph=(graph.record_ids==base.record_ids and np.array_equal(graph.indices,base.indices) and np.array_equal(graph.indptr,base.indptr) and np.array_equal(graph.priority_indices,base.priority_indices))
            grow.update(status='completed_task_slice',geometry=geometry,utility_failures=sum(r['status']!='completed' for r in utility))
            for ji,job in enumerate(gjobs):
                row={'job_id':job['job_id'],'group_id':group['group_id'],'arm':job.get('arm'),'trajectory_id':job.get('trajectory_id')}
                try:
                    path=byname[job['trajectory_id']]
                    if job.get('kind')=='timing_repeat':raise NotImplementedError('Timing repetitions belong to isolated worker program')
                    if not same_graph and path['arm'] not in {'R','S','R-volume-matched-to-S'}:raise ValueError('Graph-dependent U/A cannot be relabeled across alternative curators')
                    if not path['checkpoints']:
                        row.update(status='structural_zero_or_unavailable_arm',reason=path['status'],checkpoints=0);ledger.append(row);continue
                    count=0;failures=0
                    for checkpoint in path['checkpoints']:
                        crow={'job_id':job['job_id'],'group_id':group['group_id'],'trajectory_id':path['trajectory_id'],'arm':path['arm'],'checkpoint':checkpoint}
                        try:
                            dead=deleted_indices(path,checkpoint,graph);deleted=[graph.record_ids[i] for i in dead]
                            selected=set(map(int,graph.selected_indices(deleted)));frozen=initial-dead
                            wh,oracle=ridge_predictions(x,y,selected,ex,lam);wf,frozen_pred=ridge_predictions(x,y,frozen,ex,lam)
                            target=group_dir/f'job_{ji:04d}_checkpoint_{checkpoint:06d}.npz'
                            np.savez(target,before=before,frozen=frozen_pred,oracle=oracle,oracle_weights=wh,frozen_weights=wf)
                            crow.update(status='completed',admissions=len(selected-initial),zero_admission=not bool(selected-initial),
                              deleted_records=len(dead),deleted_units=checkpoint,unit=path['unit'],
                              admissions_per_deleted_record=len(selected-initial)/len(dead) if dead else None,
                              admissions_per_deleted_source=len(selected-initial)/checkpoint if path['unit']=='source' and checkpoint else None,selected_count=len(selected),frozen_count=len(frozen),
                              empty_target=not selected,correct_selected_ids=[graph.record_ids[i] for i in sorted(selected)],
                              effects=metrics.pair_effects(ey,before,frozen_pred,oracle,thresholds=thresholds),
                              oracle_metrics=task_metrics(bundle,ey,oracle,thresholds),frozen_metrics=task_metrics(bundle,ey,frozen_pred,thresholds),
                              prediction_file=str(target.relative_to(out)),prediction_sha256=file_hash(target))
                        except Exception as error:
                            failures+=1;crow.update(status='failed',exception_type=type(error).__name__,message=str(error))
                        checkpoint_rows.append(crow);count+=1
                    row.update(status='completed' if not failures else 'checkpoint_failures',checkpoints=count,failed_checkpoints=failures)
                except KeyError as error:row.update(status='blocked_missing_frozen_path',message=str(error))
                except NotImplementedError as error:row.update(status='outside_task_executor',message=str(error))
                except Exception as error:row.update(status='failed',exception_type=type(error).__name__,message=str(error))
                ledger.append(row)
        except (KeyError,FileNotFoundError) as error:
            grow.update(status='blocked_missing_input',message=str(error));ledger.extend({'job_id':j['job_id'],'group_id':group['group_id'],'status':'blocked_missing_input','message':str(error)} for j in gjobs)
        except NotImplementedError as error:
            grow.update(status='outside_task_executor',message=str(error));ledger.extend({'job_id':j['job_id'],'group_id':group['group_id'],'status':'outside_task_executor','message':str(error)} for j in gjobs)
        except Exception as error:
            grow.update(status='failed_configuration',exception_type=type(error).__name__,message=str(error));ledger.extend({'job_id':j['job_id'],'group_id':group['group_id'],'status':'failed_configuration','message':str(error)} for j in gjobs)
        group_ledger.append(grow)
        write(out/'group_ledger.json',group_ledger);write(out/'job_ledger.json',ledger);write(out/'checkpoints.json',checkpoint_rows)
    if {r['job_id'] for r in ledger}!={r['job_id'] for r in planned} or len(ledger)!=len(planned):raise AssertionError('Planned task job disappeared or duplicated')
    summary={'schema':SCHEMA,'scope':'engineering evaluator on frozen inputs; no primary promotion, no speed claim',
      'planned_task_groups':len(groups),'observed_task_groups':len(group_ledger),'planned_task_jobs':len(planned),'observed_task_jobs':len(ledger),
      'completed_task_jobs':sum(r['status']=='completed' for r in ledger),'blocked_or_other_jobs':sum(r['status']!='completed' for r in ledger),
      'checkpoint_rows':len(checkpoint_rows),'failed_checkpoint_rows':sum(r['status']!='completed' for r in checkpoint_rows),
      'zero_admission_rows':sum(r.get('zero_admission',False) for r in checkpoint_rows),'all_planned_task_cells_preserved':True,
      'execution_integrity_passed':fingerprint(bundles)==lock['bundle_bindings'] and code_bindings()==lock['code_bindings'],
      'primary_study':False,'run_lock_sha256':file_hash(out/'run_lock.json'),'code_bindings':code_bindings()}
    write(out/'summary.json',summary);return summary


def load_bound_value(value,base):
    if isinstance(value,dict) and set(value)=={'path','sha256','kind'}:
        path=(base/value['path']).resolve()
        if not path.is_relative_to(base.resolve()) or not path.is_file() or file_hash(path)!=value['sha256']:raise ValueError('Artifact absent, outside bundle root, or hash mismatch')
        if value['kind']=='npy':return np.load(path,mmap_mode='r',allow_pickle=False)
        if value['kind']=='json':return load_bound_value(json.loads(path.read_text()),base)
        if value['kind']=='jsonl':return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        raise ValueError('Unsupported bound artifact kind')
    if isinstance(value,dict):return {k:load_bound_value(v,base) for k,v in value.items()}
    if isinstance(value,list):return [load_bound_value(v,base) for v in value]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',type=Path,required=True);p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--bundles',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    registry=json.loads(a.registry.read_text());opener=gzip.open if a.jobs.suffix=='.gz' else open
    with opener(a.jobs,'rt') as stream:jobs=[json.loads(l) for l in stream if l.strip()]
    bundles=load_bound_value(json.loads(a.bundles.read_text()),a.bundles.parent)
    print(json.dumps(run_task_program(registry,jobs,bundles,a.out)))

if __name__=='__main__':main()
