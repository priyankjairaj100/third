"""Natural-text branch integration and refusal checks. No semantic claims."""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features
from phase3.reference_graph import build_reference_graph
from phase4.requests import generate_manifest
from phase5 import task_program as task, refit
from phase6 import dispatch as d

def fixture():
    source=ROOT/'data/civil_comments_engineering_preview.jsonl';rows=read_natural_jsonl(source,require_labels=True)
    x,_=lexical_engineering_features(rows,32);cx,_=lexical_engineering_features(rows,64)
    tr=list(range(20));te=list(range(20,28));ids=[rows[i]['record_id'] for i in tr];evaluation=[rows[i]['record_id'] for i in te]
    owners=['unknown:'+r for r in ids];kinds={s:'unknown_singleton' for s in owners}
    graph=build_reference_graph(cx[tr],ids,.6,owners)
    design={'requests':{'record_horizon':4,'source_horizon':2,'record_checkpoints':[1,2,4],'source_checkpoints':[1,2]}}
    manifest=generate_manifest(graph,dataset_id='civil_comments',panel_id='phase6_natural20',source_kinds=kinds,
        design=design,allocations={'R':1,'S':1,'U':0,'A':0},include_excluded_blocker_stress=False)
    bundle={'dataset_id':'civil_comments','task':'civil','train_ids':ids,'evaluation_ids':evaluation,'source_ids':owners,
        'source_kinds':kinds,'train_y':np.asarray([rows[i]['label'] for i in tr],np.float64)[:,None],
        'evaluation_y':np.asarray([rows[i]['label'] for i in te],np.float64)[:,None],
        'base_curator':'lexical64','request_design':design,'requests':manifest,
        'curators':{'lexical64':{'train_ids':ids,'train':cx[tr],
            'threshold_lock':task.engineering_parameter('threshold',.6,representation='lexical64')}},
        'learners':{'lexical32':{'train_ids':ids,'evaluation_ids':evaluation,'train':x[tr],'evaluation':x[te],
            'lambda_lock':task.engineering_parameter('lambda',.01,representation='lexical32')}},
        'evidence_role':d.ENGINEERING_MODE,'natural_input_path':str(source),'natural_input_sha256':task.file_hash(source),
        'refit_configuration':{'clusters':3,'iterations':4,'epsilon':.4,'seed':7,'backend':refit.REFERENCE,'development_config_id':'fixed-engineering-only'},
        'refit_seed_variants':[11,13,17],'ANN_policy':{'tables':4,'bits':4,'sample_size':30}}
    methods=['O-G','O-T','B-E','B-A','P-I','P-S','P-R','B-F'];groups=[];jobs=[]
    for block,variant,meth,options in [
        ('A_structure','native',['independent_structure_oracle'],{}),
        ('B_relevance','native',['O-G','B-F'],{}),
        ('C_full_methods','native',methods,{}),
        ('D_replication','native',['O-G','B-F'],{}),
        ('E_full_refit','native',['R_full_refit','F_frozen_fitted','B_frozen_selection'],{}),
        ('F_convex','native',['fresh_optimum','eligible_payload_retraining','certified_convex_repair'],{}),
        ('G_boundary','zero-start-CG',methods,{'decoder':'zero_start_CG'}),
        ('G_boundary','fp32-state',methods,{'state_precision':'FP32'}),
        ('G_boundary','ANN',['exhaustive_graph','pinned_ANN_candidates'],{}),
        ('G_boundary','missing-e5',methods,{'curator_encoder':'e5'})]:
        group={'group_id':block+'/civil_comments/natural20/'+variant,'block':block,'corpus':'civil_comments','panel':'natural20',
            'variant':variant,'request_family':'civil_comments/natural20','methods':meth,'allocations':{'R':1,'S':1,'U':0,'A':0},
            'configuration':{'curator_encoder':'lexical64','learner_encoder':'lexical32','learner_dimension':32,
                'priority_seed':0,'threshold':'calibrated','decoder':'Cholesky','state_precision':'FP64',**options}}
        groups.append(group)
        for path in manifest['trajectories']:
            jobs.append({'job_id':group['group_id']+'/'+path['trajectory_id'],'group_id':group['group_id'],
                'trajectory_id':path['trajectory_id'],'arm':path['arm'],'methods':meth})
            if block=='C_full_methods' and path['arm']=='R':jobs[-1]['complete_state_rebuild_each_checkpoint']=True
        if block=='E_full_refit':
            for kind,count in [('same_seed_no_deletion',5),('alternative_seed_no_deletion',3)]:
                for repeat in range(count):jobs.append({'job_id':group['group_id']+'/'+kind+str(repeat),
                    'group_id':group['group_id'],'kind':kind,'repeat':repeat,'methods':meth})
    registry={'schema':'ccu-natural-engineering-dispatch-check-1','groups':groups,'jobs_content_sha256':d.digest(jobs)}
    return registry,jobs,{'civil_comments/natural20':bundle},design

def main():
    out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'phase6/results/dispatch_engineering'
    if out.exists():raise FileExistsError('Preserve existing output')
    registry,jobs,bundles,design=fixture();out.mkdir(parents=True);checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,RuntimeError,FileExistsError):checks.append(name)
        else:raise AssertionError(name)
    d.write(out/'design.json',{'registry':registry,'jobs':jobs,'bundle_bindings':task.fingerprint(bundles),
        'scope':'reused real Civil20/8 lexical engineering; no source, semantic or independent empirical claim'})
    from phase5.run_isolated_v2 import DEFAULT_POLICY
    policy={**DEFAULT_POLICY,'convex_verifier':'exact_dyadic_integer_v1'}
    summary=d.run_dispatch(registry,jobs,bundles,out/'run',design=design,policy=policy)
    ledger=json.loads((out/'run/jobs.json').read_text());groups=json.loads((out/'run/groups.json').read_text())
    check('all_planned_jobs_retained',len(ledger)==len(jobs) and summary['every_planned_job_retained'])
    check('all_A_G_blocks_present',set(g['block'] for g in registry['groups'])=={'A_structure','B_relevance','C_full_methods','D_replication','E_full_refit','F_convex','G_boundary'})
    check('no_primary_promotion',summary['primary_outputs_accepted']==0)
    check('natural_inputs_unchanged',summary['inputs_unchanged'])
    for block in ['A_structure','B_relevance','C_full_methods','D_replication','E_full_refit','F_convex']:
        check(block+'_actual_execution',any(r['status']=='completed' and r['group_id'].startswith(block+'/') and r.get('checkpoint_count',0)>0 for r in ledger))
    check('CG_all_methods_completed',any(r['status']=='completed' and '/zero-start-CG/' in r['job_id'] and len(r['methods'])==8 for r in ledger))
    fp=next(r for r in ledger if '/fp32-state/R000' in r['job_id'])
    check('FP32_variants_not_replaced',len(fp['methods'])==8 and all(m['status']=='completed' for m in fp['methods']) and all(m['requested_precision']=='FP32' for m in fp['methods']))
    check('unknown_sources_preserved_as_unavailable',any(r['status']=='structural_zero_or_unavailable_arm' and '/S000' in r['job_id'] for r in ledger))
    check('missing_encoder_preserved',all(r['status']!='completed' for r in ledger if '/missing-e5/' in r['job_id']))
    check('all_worker_target_checks_pass',all(c['passed'] for r in ledger for m in r.get('methods',[]) for c in m.get('head_checks',[])))
    state_job=next(r for r in ledger if r['job_id']=='C_full_methods/civil_comments/natural20/native/R000')
    check('all_eight_state_audits_pass',len(state_job['methods'])==8 and all(m.get('state_audit',{}).get('passed') for m in state_job['methods']))
    be=[m for row in ledger for m in row.get('methods',[]) if m['method']=='B-E' and m.get('requested_precision')=='FP64']
    check('B_E_membership_audit_at_every_FP64_release',bool(be) and all(m.get('state_audit',{}).get('passed') and len(m['state_audit']['checkpoints'])==len(m['head_checks']) for m in be))
    summaries=[m for row in ledger for m in row.get('methods',[]) if m['method'] in {'P-I','P-S','P-R'} and m.get('requested_precision')=='FP64']
    check('all_summary_moments_audited_at_every_FP64_release',len(summaries)>=6 and all(m.get('state_audit',{}).get('passed') and len(m['state_audit']['checkpoints'])==len(m['head_checks']) and all(c['moments_and_membership']['gram']['passed'] and c['moments_and_membership']['cross']['passed'] for c in m['state_audit']['checkpoints']) for m in summaries))
    forecast=json.loads((out/'run/group_000/job_0000/structure_forecast.json').read_text())
    check('structural_rank_and_byte_forecast_at_actual_horizon',forecast['horizons'][0]['horizon']==4 and 'query_rank' in forecast['horizons'][0] and bool(forecast['horizons'][0]['storage_forecasts']))
    check('statistical_binding_comes_from_actual_output',state_job['analysis_binding']['evaluation_ids']==next(iter(bundles.values()))['evaluation_ids'] and state_job['analysis_binding']['arm']=='R')
    check('method_order_recorded',len(list((out/'run').glob('group_*/job_*/method_order.json')))>=5)
    reject('forged_primary_registry_rejected',lambda:d.run_dispatch(registry,jobs,bundles,out/'forged',mode=d.PRIMARY_MODE))
    reject('changed_job_manifest_rejected',lambda:d.run_dispatch({**registry,'jobs_content_sha256':'0'*64},jobs,bundles,out/'changed'))
    b=copy.deepcopy(next(iter(bundles.values())));b['requests']['trajectories'][0]['deletion_order'].reverse()
    reject('changed_requests_rejected',lambda:d.recompute_requests(b,d._base_graph(b)))
    b=copy.deepcopy(next(iter(bundles.values())));b['requests']['configuration']['master_seed']+=1
    reject('changed_seed_rejected',lambda:d.recompute_requests(b,d._base_graph(b)))
    # Algebra only: absolute closeness cannot replace the actual eta gate.
    algebra=d.saved_head_gate(np.array([[1]],np.float32),np.array([[0]],np.float64),{0},.01,np.array([[1e-11]],np.float64))
    check('saved_head_actual_residual_gate',not algebra['passed'] and algebra['head_frobenius_error']<1e-9 and algebra['normalized_residual_eta']>1e-10)
    source=ROOT/'data/civil_comments_engineering_preview.jsonl';natural=read_natural_jsonl(source,require_labels=True)
    cx,_=lexical_engineering_features(natural,64);ids=[r['record_id'] for r in natural]
    graph=build_reference_graph(cx,ids,.6,['unknown:'+r for r in ids],seed=7)
    c={'cx':cx,'graph':graph,'priority_seed':7}
    for number,dead in enumerate([set(),{0,3,68,70},{*range(100)}]):
        check('independent_block_boundaries_'+str(number),d.independent_selection(c,dead)==d.independent_selection(c,dead,fresh=True))
    full_design=json.loads((ROOT.parent/'output/empirical_program/study_design.json').read_text())
    failed=d.acceptance(registry['groups'][0],next(iter(bundles.values())),full_design,mode=d.PRIMARY_MODE,base_dir=ROOT.parent)
    check('short_engineering_schedule_cannot_activate_primary',not next(c for c in failed['checks'] if c['name']=='authoritative_primary_request_schedule')['passed'])
    from phase6.recipes import build_registry
    primary,pjobs=build_registry();missing=d.run_dispatch(primary,pjobs,{},out/'missing_primary',mode=d.PRIMARY_MODE)
    check('every_primary_job_remains_blocked',missing['planned_jobs']==missing['observed_jobs']==len(pjobs) and missing['completed_jobs']==0)
    d.write(out/'checks.json',{'status':'passed','checks':checks,'check_count':len(checks),'engineering_summary':summary,
        'missing_primary_summary':missing,'dispatch_sha256':task.file_hash(d.__file__),'check_sha256':task.file_hash(__file__),
        'semantic_vectors_created':False,'human_responses_created':0})
    print(json.dumps({'checks':len(checks),'engineering':summary,'missing_primary':missing},indent=2))

if __name__=='__main__':main()
