"""Exercise routes on real Civil text. No semantic or human results are created."""
from pathlib import Path
import argparse, copy, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from ccu.data import read_natural_jsonl, lexical_engineering_features
from phase3.reference_graph import build_reference_graph
from phase4.requests import generate_manifest
from phase5 import task_program as task
from phase6 import dispatch, extensions as e, recipes
from phase5.run_isolated_v2 import DEFAULT_POLICY


def natural_bundle():
    source=ROOT/'data/civil_comments_engineering_preview.jsonl';allrows=read_natural_jsonl(source,require_labels=True)
    train=allrows[:40];evaluation=allrows[40:60];ids=[r['record_id'] for r in train];eval_ids=[r['record_id'] for r in evaluation]
    cx,_=lexical_engineering_features(train,32);xx,_=lexical_engineering_features(train+evaluation,16)
    owners=['unknown:'+r for r in ids];kinds={s:'unknown_singleton' for s in owners}
    graph=build_reference_graph(cx,ids,.6,owners)
    design={'requests':{'record_horizon':8,'source_horizon':8,'record_checkpoints':[1,2,4,8]}}
    manifest=generate_manifest(graph,dataset_id='civil_comments',panel_id='reused-40',source_kinds=kinds,design=design,
        allocations={'R':2,'S':1,'U':2,'A':2},include_excluded_blocker_stress=True)
    bundle={'dataset_id':'civil_comments','task':'civil','train_ids':ids,'evaluation_ids':eval_ids,'source_ids':owners,
        'train_records':train,'source_kinds':kinds,'train_y':np.asarray([r['label'] for r in train],np.float64)[:,None],
        'evaluation_y':np.asarray([r['label'] for r in evaluation],np.float64)[:,None],
        'base_curator':'lexical32','base_learner':'lexical16','requests':manifest,'request_design':design,
        'evidence_role':dispatch.ENGINEERING_MODE,'natural_input_path':str(source),'natural_input_sha256':task.file_hash(source),
        'curators':{'lexical32':{'train_ids':ids,'train':cx,'threshold_lock':task.engineering_parameter('threshold',.6,representation='lexical32')}},
        'learners':{'lexical16':{'train_ids':ids,'evaluation_ids':eval_ids,'train':xx[:40],'evaluation':xx[40:],
            'lambda_lock':task.engineering_parameter('lambda',.01,representation='lexical16')}}}
    return bundle


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=False);bundle=natural_bundle();checks=[];routes=[]
    context={'dispatch_module':dispatch,'policy':DEFAULT_POLICY,'mode':dispatch.ENGINEERING_MODE,
             'design':json.loads(recipes.legacy.DESIGN.read_text()),'base_dir':ROOT.parent}
    reg,jobs=recipes.build_registry()
    candidates=[]
    names=['structure_mass_PPS','structure_one_percent_horizon','excluded_blocker_stress','fresh_graph_audit',
           'graph_envelope_conditional','head_normalization_error_diagnostic','utility_reference',
           'common_cached_graph_oracle','cold_warm_scope','every_release_persistence']
    for name in names:
        job=next(j for j in jobs if j.get('recipe_id')==name and j.get('corpus')=='civil_comments' and (j.get('arm','R') in {'R','S-PPS','A-excluded-blockers'}))
        candidates.append(copy.deepcopy(job))
    for curator in ('exact_text','fixed_order_greedy','none'):
        candidates.append(copy.deepcopy(next(j for j in jobs if j.get('recipe_id')=='negative_controls' and j.get('corpus')=='civil_comments' and j.get('curator')==curator)))
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    cache_group=e._group(candidates[0],bundle,context['mode']);fingerprint=task.fingerprint(bundle)
    cache_key=e._cache_key(fingerprint,cache_group,context)
    check('cache_panel_alias_cannot_reuse_acceptance',cache_key!=e._cache_key(fingerprint,{**cache_group,'panel':'second-disjoint-10000'},context))
    check('cache_corpus_alias_cannot_reuse_acceptance',cache_key!=e._cache_key(fingerprint,{**cache_group,'corpus':'cc_news'},context))
    check('cache_changed_design_cannot_reuse_acceptance',cache_key!=e._cache_key(fingerprint,cache_group,{**context,'design':{'changed':True}}))
    check('cache_changed_base_cannot_reuse_acceptance',cache_key!=e._cache_key(fingerprint,cache_group,{**context,'base_dir':ROOT}))
    bundles={'civil_comments/reused-40':bundle}
    for i,job in enumerate(candidates):
        original=job['job_id'];job['panel']='reused-40';job['job_id']='engineering-route/'+str(i)+'/'+job['recipe_id'];job['registered_template_ID']=original
        target=out/f'route_{i:02d}';target.mkdir()
        result=e.run_extension(job,bundles,target,context);e.write(target/'outcome.json',result)
        expected='structural_zero_or_unavailable_arm' if job['recipe_id']=='structure_mass_PPS' else 'completed'
        check(job['recipe_id']+'_'+str(i)+'_expected_status',result['status']==expected or (job['recipe_id']=='negative_controls' and result['status']=='structural_zero_or_unavailable_arm'))
        check(job['recipe_id']+'_'+str(i)+'_no_primary_promotion',not result['primary_output_accepted'])
        routes.append({'recipe_id':job['recipe_id'],'curator':job.get('curator'),'status':result['status'],
                       'evidence':'natural_Civil40_lexical_route_execution','artifact':str(target.relative_to(out))})
        print(job['recipe_id'],result['status'],flush=True)
    # Regenerate real-text human frames and execute blank-response analysis.
    from phase4.admission_audit import prepare_admission_audit,write_admission_pack
    from phase5.context_audit import prepare_context_audits,write_context_pack
    graph=build_reference_graph(bundle['curators']['lexical32']['train'],bundle['train_ids'],.6,bundle['source_ids'])
    frame={'civil_comments':{'records':bundle['train_records'],'features':bundle['curators']['lexical32']['train'],
        'graph':graph,'request_manifest':bundle['requests'],
        'existing_labels':{r['record_id']:r['label'] for r in bundle['train_records']},
        'provenance':{'corpus_snapshot':'actual_existing_Civil_preview40',
            'representation_id':'lexical32','representation_revision':'frozen_workspace_code',
            'evidence_role':'engineering_nonconfirmatory'}}}
    bundles['civil_comments/primary-10000']=bundle
    for name,prepare,writer,key in [('human_admission_main',prepare_admission_audit,write_admission_pack,'human_admission'),
                                   ('human_admission_context',prepare_context_audits,write_context_pack,'human_context')]:
        manifest=prepare(frame);pack=out/(name+'_blank_pack')
        writer(manifest,{'civil_comments':bundle['train_records']},pack)
        responses=json.loads((pack/'responses.template.json').read_text())
        bundles[key]={'manifest':manifest,'responses':responses,'corpus_inputs':frame,'evidence_role':dispatch.ENGINEERING_MODE}
        job=copy.deepcopy(next(j for j in jobs if j.get('recipe_id')==name));target=out/(name+'_blank_analysis');target.mkdir()
        result=e.run_extension(job,bundles,target,context);e.write(target/'outcome.json',result)
        check(name+'_actual_blank_response_not_completion',result['status']=='awaiting_real_human_responses')
        check(name+'_no_primary_promotion',not result['primary_output_accepted'])
        routes.append({'recipe_id':name,'status':result['status'],'evidence':'actual_Civil_blank_frame_reconstructed_and_analyzed'})
    # Date-group arithmetic uses schema symbols only, never assigned natural provenance.
    from ccu.core import BlockerGraph
    schema_graph=build_reference_graph(np.eye(2,dtype=np.float32),['schema_A','schema_B'],.6)
    schema_bundle={'requests':{'trajectories':[]},'train_records':[{'record_id':'schema_A','original_fields':{'date':'2018-01-01'}},
        {'record_id':'schema_B','original_fields':{'date':'2018-01-02'}}]}
    p=e.extension_path({'recipe_id':'news_chronological'},schema_bundle,{'graph':schema_graph})
    check('chronological_schema_whole_dates',p['checkpoints']==[1,2] and p['deletion_order']==['schema_A','schema_B'])
    routes.append({'recipe_id':'news_chronological','status':'software_path_check_passed','evidence':'two_symbol_date_schema_only'})
    # The threshold route replays real lexical pairs, then refuses absent judgments.
    from phase3 import calibration as cal
    provenance={'dataset_id':'civil_comments_preview40','encoder_id':'lexical_engineering_features',
        'encoder_revision':'existing_frozen_code_not_E5','population_scope':'reused_natural_preview',
        'evidence_role':'engineering_nonconfirmatory'}
    selection=cal.prepare_selection(bundle['curators']['lexical32']['train'],bundle['train_records'],provenance)
    calibration_dossier={'features':bundle['curators']['lexical32']['train'],'records':bundle['train_records'],
        'provenance':provenance,'selection_manifest':selection,
        'selection_responses':{'manifest_sha256':selection['sha256'],'responses':[]},'evidence_role':dispatch.ENGINEERING_MODE}
    threshold_job=copy.deepcopy(next(j for j in jobs if j.get('recipe_id')=='human_threshold_calibration' and j.get('corpus')=='civil_comments' and j.get('stage')=='selection'))
    target=out/'actual_blank_threshold';target.mkdir()
    try:e.run_extension(threshold_job,{'human_threshold/civil_comments/e5':calibration_dossier},target,context)
    except ValueError:check('actual_pair_replay_does_not_create_ratings',True)
    else:raise AssertionError('Blank threshold responses passed')
    altered=copy.deepcopy(calibration_dossier);raw={k:v for k,v in altered['selection_manifest'].items() if k!='sha256'}
    raw['pairs']=raw['pairs'][1:];altered['selection_manifest']=cal._seal(raw)
    try:e.run_extension(threshold_job,{'human_threshold/civil_comments/e5':altered},out/'altered_calibration',context)
    except ValueError as error:check('resigned_but_changed_sample_rejected','sampling law' in str(error))
    else:raise AssertionError('Altered sampling manifest passed')
    routes.append({'recipe_id':'human_threshold_calibration','status':'blank_human_input_refused',
        'evidence':'actual_Civil_pair_sampling_recomputed_no_judgments'})
    # Complete statistics routing uses explicitly modified metadata around unchanged saved predictions.
    from phase6.check_statistics_evidence import build_fixture
    dossier,statistics_context=build_fixture(out/'statistics_metadata_fixture',source=out/'route_09')
    dossier['crossed']={}
    dossier['valid_p_values']={'signed_loss':{k:None for k in ('civil_comments:R','civil_comments:S','askubuntu:R','askubuntu:S')},
        'equal_RS_lifecycle':{k:None for k in ('civil_comments','askubuntu')}}
    stats_job=next(j for j in jobs if j.get('recipe_id')=='test_source_bootstrap')
    target=out/'statistics_analysis';target.mkdir()
    result=e.run_extension(stats_job,{'statistics':dossier},target,statistics_context);e.write(target/'outcome.json',result)
    check('complete_statistics_route_uses_saved_predictions',result['status']=='completed' and not result['primary_output_accepted'])
    summary=json.loads((target/'statistics.json').read_text())
    check('statistics_runs_locked10000_resamples',summary['paired']['bootstrap_replicates']==10000)
    check('statistics_does_not_invent_source_population',summary['crossed']=={})
    check('statistics_does_not_invent_significance',all(not r['reject'] for family in summary['Holm'].values() for r in family['tests'].values()))
    routes.append({'recipe_id':'test_source_bootstrap','status':'software_statistics_route_completed',
        'evidence':'explicit_metadata_fixture_around_unchanged_saved_natural_predictions_no_genuine_source_bootstrap'})
    # Genuine sources and dates are absent. These routes must refuse or preserve emptiness.
    missing_names=['news_chronological','human_threshold_calibration','human_admission_main','human_admission_context','test_source_bootstrap']
    for name in missing_names:
        job=copy.deepcopy(next(j for j in jobs if j.get('recipe_id')==name))
        target=out/('missing_'+name);target.mkdir()
        try:e.run_extension(job,{},target,context)
        except KeyError:check(name+'_missing_actual_inputs_refused',True)
        else:raise AssertionError(name+' invented missing input')
        routes.append({'recipe_id':name,'status':'blocked_missing_actual_input','evidence':'input_refusal_only'})
    # Wrong origin and unknown recipe are rejected before any scientific release.
    primary_context={**context,'mode':dispatch.PRIMARY_MODE}
    try:e._narrow_review({'evidence_role':dispatch.ENGINEERING_MODE},{},primary_context,('actual_question',))
    except KeyError:check('engineering_label_cannot_bypass_primary_review',True)
    else:raise AssertionError('primary provenance bypass')
    for temporary in context.get('extension_cache',{}).get('_temporary_directories',[]):temporary.cleanup()
    coverage={name:{'natural_positive_execution':any(r['recipe_id']==name and r['status']=='completed' and r['evidence']=='natural_Civil40_lexical_route_execution' for r in routes),
        'exercised_evidence':sorted({r['evidence'] for r in routes if r['recipe_id']==name}),
        'observed_statuses':sorted({r['status'] for r in routes if r['recipe_id']==name}),
        'primary_positive_execution':False} for name in reg['recipe_job_counts']}
    check('all16_extension_routes_have_explicit_coverage',len(coverage)==16 and all(v['exercised_evidence'] for v in coverage.values()))
    report={'status':'passed','checks':checks,'check_count':len(checks),'routes':routes,'route_execution_coverage':coverage,
            'source_sha256':{n:task.file_hash(ROOT/'phase6'/n) for n in ('recipes.py','extensions.py','check_extensions.py')},
            'natural_records_train':40,'natural_records_evaluation':20,'curator':'lexical32','learner':'lexical16',
            'genuine_source_withdrawal_evidence':False,'actual_human_responses':0,'primary_semantic_study_started':False,
            'development_evaluation_is_not_untouched_test':True,'timings_not_systems_evidence':True}
    e.write(out/'checks.json',report);print(json.dumps({'status':'passed','check_count':len(checks)},indent=2))
if __name__=='__main__':main()
