"""Frozen-input utility/sensitivity integration on reused natural Civil82/18.

No semantic model is impersonated by lexical features; absent semantic cells
remain blocked. Small helper arrays below are software checks only.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
import argparse,copy,gzip,json,tempfile
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from phase3 import calibration as cal
from phase3.reference_graph import build_reference_graph
from phase4.requests import generate_manifest
from phase5 import task_program as task
from phase5.study_registry import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'phase5/results/task_program_engineering');a=p.parse_args();out=a.out
    if out.exists():raise FileExistsError('Frozen output exists')
    out.mkdir(parents=True)
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,FileExistsError,KeyError):checks.append(name)
        else:raise AssertionError(name)
    source=ROOT/'data/civil_comments_engineering_preview.jsonl';rows=read_natural_jsonl(source,require_labels=True)
    partition_path=ROOT/'phase2/results/partition_manifest.json';partition=json.loads(partition_path.read_text())
    roles={r['record_id']:r['partition'] for r in partition};tr=[i for i,r in enumerate(rows) if roles[r['record_id']]=='development_train'];te=[i for i,r in enumerate(rows) if roles[r['record_id']]=='development_evaluation']
    ids=[rows[i]['record_id'] for i in tr];eval_ids=[rows[i]['record_id'] for i in te];sources=['unknown:'+i for i in ids]
    features={};metadata={}
    for d in (64,128,256):features[d],metadata[d]=lexical_engineering_features(rows,d)
    graph=build_reference_graph(features[128][tr],ids,.6,sources)
    requests=generate_manifest(graph,dataset_id='civil_comments',panel_id='reused_Civil82_task_software',source_kinds={s:'unknown_singleton' for s in sources},
      allocations={'R':2,'S':0,'U':1,'A':1},record_horizon=8,include_excluded_blocker_stress=False,
      design={'requests':{'record_checkpoints':[1,2,4,8]}},master_seed=2027100407)
    bundle={'dataset_id':'civil_comments','task':'civil','train_ids':ids,'evaluation_ids':eval_ids,'source_ids':sources,
      'train_y':np.asarray([rows[i]['label'] for i in tr],dtype=np.float64)[:,None],
      'evaluation_y':np.asarray([rows[i]['label'] for i in te],dtype=np.float64)[:,None],
      'base_curator':'lexical128','requests':requests,'curators':{},'learners':{},'projections':{},
      'evidence':'reused lexical diagnostic; legacy evaluation split not untouched test; native sources/semantic guard unavailable'}
    for d in (128,256):
        name='lexical'+str(d);bundle['curators'][name]={'train_ids':ids,'train':features[d][tr],
          'threshold_lock':task.engineering_parameter('threshold',.6,representation=name)}
    for d in (64,128):
        name='lexical'+str(d);bundle['learners'][name]={'train_ids':ids,'evaluation_ids':eval_ids,'train':features[d][tr],'evaluation':features[d][te],
          'lambda_lock':task.engineering_parameter('lambda',.01,representation=name)}
    projection=task.seeded_projection(64,32,20271005);bundle['projections']['32']={'matrix':projection,'seed':20271005,'data_independent':True}
    bundle['learners']['lexical64']['projection_parameters']={'32':{'lambda_lock':task.engineering_parameter('lambda',.01,representation='lexical64:projection-32')}}
    groups=[]
    def group(variant,block='G_boundary',alloc=None,**config):
        groups.append({'group_id':block+'/civil_comments/reused-82/'+variant,'block':block,'corpus':'civil_comments','panel':'reused-82','variant':variant,
          'allocations':alloc or {'R':2},'methods':['O-G','B-F'],'request_family':'civil_comments/reused-82',
          'configuration':{'curator_encoder':'lexical128','learner_encoder':'lexical64','learner_dimension':64,
            'threshold':'calibrated','priority_seed':0,'state_precision':'FP64','decoder':'Cholesky',**config}})
    group('native',block='B_relevance',alloc={'R':2,'U':1,'A':1})
    for cd in (128,256):
        for ld in (64,128):group(f'lexical-cross-{cd}-{ld}',curator_encoder='lexical'+str(cd),learner_encoder='lexical'+str(ld),learner_dimension=ld)
    for seed in (1,2):group('priority-'+str(seed),priority_seed=seed)
    group('threshold-minus',threshold='max(0,calibrated-0.02)');group('threshold-plus',threshold='min(1,calibrated+0.02)')
    group('projection-32',learner_dimension=32,projection_seed=20271005)
    group('lambda-lower',lambda_factor=.1);group('lambda-higher',lambda_factor=10.)
    group('missing-semantic-encoder',curator_encoder='e5')
    group('FP32-outside-task-slice',state_precision='FP32')
    jobs=[]
    for g in groups:
        for arm,count in g['allocations'].items():
            for i in range(count):
                path=f'{arm}{i:03d}';jobs.append({'job_id':g['group_id']+'/'+path,'group_id':g['group_id'],'arm':arm,'trajectory_id':path})
    jobs.append({'job_id':groups[0]['group_id']+'/R005','group_id':groups[0]['group_id'],'arm':'R','trajectory_id':'R005'})
    registry={'schema':'ccu-engineering-task-registry-1','groups':groups,'jobs_content_sha256':digest(jobs),
      'scope':'prospective reduced software exercise, not substitution for semantic registry','primary_study':False}
    task.write(out/'engineering_design.json',{'source_sha256':sha256_file(source),'legacy_partition_sha256':sha256_file(partition_path),
      'train_records':len(tr),'evaluation_records':len(te),'source_arm_available':False,'semantic_features':False,
      'new_independent_empirical_evidence':False,'feature_metadata':metadata,'registry':registry,'jobs':jobs})
    summary=task.run_task_program(registry,jobs,{'civil_comments/reused-82':bundle},out/'executed')
    check('all_task_cells_retained',summary['planned_task_jobs']==summary['observed_task_jobs']==len(jobs))
    check('input_and_code_bindings_unchanged',summary['execution_integrity_passed'])
    check('no_checkpoint_failures',summary['failed_checkpoint_rows']==0)
    ledger=json.loads((out/'executed/group_ledger.json').read_text());jl=json.loads((out/'executed/job_ledger.json').read_text());cp=json.loads((out/'executed/checkpoints.json').read_text())
    check('semantic_cell_blocked_not_lexical_relabel',next(g for g in ledger if g['group_id'].endswith('missing-semantic-encoder'))['status']=='blocked_missing_input')
    check('FP32_not_silently_FP64',next(g for g in ledger if g['group_id'].endswith('FP32-outside-task-slice'))['status']=='outside_task_executor')
    check('missing_frozen_path_retained',any(r['status']=='blocked_missing_frozen_path' for r in jl))
    check('four_lexical_cross_cells',sum('lexical-cross-' in r['group_id'] and r['status']=='completed_task_slice' for r in ledger)==4)
    check('all_utility_references_and_hash_budgets',all(len((u:=json.loads(p.read_text())['references']))==4 and u[0]['selected_records']==u[2]['selected_records'] for p in (out/'executed').glob('group_*/utility.json')))
    check('all_logistic_reference_certificates_pass',all(json.loads(p.read_text())['references'][-1]['status']=='completed' for p in (out/'executed').glob('group_*/utility.json')))
    check('zeros_retained',sum(r.get('zero_admission',False) for r in cp)>0)
    check('fixed_lambda_factors',{r['geometry']['lambda'] for r in ledger if r['status']=='completed_task_slice'}=={.001,.01,.1})
    check('threshold_exact_protocol_clamping',task._threshold(.005,'max(0,calibrated-0.02)')==0. and task._threshold(.995,'min(1,calibrated+0.02)')==1.)
    check('projection_columns_orthonormal',np.allclose(projection.T@projection,np.eye(32),atol=1e-12,rtol=0))
    check('no_projection_renormalization',not np.allclose(np.linalg.norm((features[64][tr].astype(float)@projection).astype(np.float32),axis=1),1.))
    reject('primary_promotion_refused',lambda:task.run_task_program(registry,jobs,{},out/'illegal_primary',evidence_role='primary'))
    check('primary_rejection_before_output',not (out/'illegal_primary').exists())
    bad=copy.deepcopy(bundle);bad['evaluation_ids'][0]=bad['train_ids'][0];reject('evaluation_training_overlap_refused',lambda:task.validate_bundle(bad))
    bad=copy.deepcopy(bundle);bad['requests']['manifest_sha256']='0'*64;reject('altered_requests_refused',lambda:task.validate_bundle(bad))
    bad=copy.deepcopy(registry);bad['jobs_content_sha256']='0'*64;reject('altered_job_inventory_refused',lambda:task.run_task_program(bad,jobs,{},out/'illegal_jobs'))
    # Actual protocol registry: every applicable uninstantiated semantic job stays
    # visible; no missing encoder gets replaced by the lexical geometry above.
    original_registry=json.loads((ROOT/'phase5/results/study_registry_release/registry.json').read_text())
    with gzip.open(ROOT/'phase5/results/study_registry_release/jobs.jsonl.gz','rt') as stream:original_jobs=[json.loads(l) for l in stream]
    blocked=task.run_task_program(original_registry,original_jobs,{},out/'registered_missing_inputs')
    check('official_registry_no_unavailable_cell_omitted',blocked['planned_task_jobs']==blocked['observed_task_jobs'] and blocked['completed_task_jobs']==0)
    # No multilabel test threshold invented when the lock is absent.
    my=np.array([[0.,0.],[1.,0.]],dtype=np.float64);ms=np.array([[.1,.2],[.8,.3]],dtype=np.float64)
    mm=task.task_metrics({'task':'multilabel'},my,ms,None)
    check('missing_tag_thresholds_report_null',mm['micro_f1'] is None and mm['macro_f1'] is None)
    result={'status':'passed','check_count':len(checks),'checks':checks,'engineering_summary':summary,
      'registered_missing_inputs_summary':blocked,'code_sha256':task.file_hash(task.__file__),'check_code_sha256':task.file_hash(__file__),
      'reused_natural_lexical_only':True,'primary_study':False,'real_semantic_caches_used':False,'human_responses_created':0}
    task.write(out/'checks.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
