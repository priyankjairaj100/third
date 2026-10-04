"""Metadata-only scheduler controls; no corpus, human, or worker execution."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import types
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from phase9 import study_assembly as s


def run(out):
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    source=s.codes(); source['phase9/check_study_assembly.py']=s.file_hash(__file__)
    checks=[]
    def check(name,value):
        if not value: raise AssertionError(name)
        checks.append(name)
    def refuses(name,fn):
        try: fn()
        except (ValueError,KeyError,FileExistsError): checks.append(name);return
        raise AssertionError(name)
    report={'schema':'ccu-study-assembly-checks-1','status':'failed','source_sha256':source,
            'evidence_role':'metadata_only_software_controls_with_explicit_acceptance_and_branch_mocks',
            'corpus_records_created':0,'source_ids_created':0,'human_responses_created':0,
            'workers_executed':0,'primary_study_started':False}
    try:
        registry,jobs=s.recipes.build_registry(); ownership=s.canonical_owners(registry,jobs)
        check('complete_21332_job_bijection',len(ownership)==len(jobs)==21332)
        check('all_43_group_memberships_derived',len({v['group_id'] for v in ownership.values()}-{None})==43)
        check('all_28_immutable_root_keys_derived',len({v['key'] for v in ownership.values()})==28)
        bad=copy.deepcopy(registry);bad['groups'].pop()
        refuses('changed_registry_rejected',lambda:s.canonical_owners(bad,jobs))
        refuses('missing_job_rejected',lambda:s.canonical_owners(registry,jobs[:-1]))
        with tempfile.TemporaryDirectory(prefix='phase9_assembly_checks_') as tmp:
            base=Path(tmp);config=base/'owners.json';s.write(config,{'owners':[]})
            plan=s.assemble(config,base/'empty_plan')
            check('empty_plan_keeps_complete_missing_schedule',plan['planned_jobs']==21332 and plan['selected_jobs']==0)
            result=s.execute(base/'empty_plan/plan.json',base/'empty_run')
            check('unmocked_empty_dispatch_keeps_every_job',result['every_planned_job_retained'] and result['observed_jobs']==21332)
            check('unmocked_empty_dispatch_starts_no_job',result['attempted_jobs']==result['primary_outputs_accepted']==0)
            check('missing_owner_method_statuses_retained',all(m['status']=='blocked_unowned_scope' for r in json.loads((base/'empty_run/jobs.json').read_text()) for m in r['methods']))
            before=s.file_hash(base/'empty_run/jobs.json')
            refuses('existing_execution_directory_rejected',lambda:s.execute(base/'empty_plan/plan.json',base/'empty_run'))
            check('existing_execution_bytes_preserved',s.file_hash(base/'empty_run/jobs.json')==before)
            groups=[next(g for g in registry['groups'] if g['group_id']=='A_structure/'+name+'/primary-10000/native') for name in ('civil_comments','askubuntu')]
            policy={'memory_bytes':64*1024**2,'cpu_seconds':1,'wall_seconds':1,'threads':1}
            specs=[];originals={}
            for index,group in enumerate(groups):
                root=base/f'original_root_{index}';root.mkdir();(root/'same_name.json').write_text(json.dumps({'root_index':index}))
                # No records or source IDs: these are routing metadata only.
                bundle={'_dossier_group_ids':[group['group_id']],'execution_policy':policy,
                        'phase9_scope':{'key':group['request_family'],'group_ids':[group['group_id']],'extension_recipes':[]},
                        'relative_evidence_reference':'same_name.json','test_root_index':index}
                s.write(root/'bundle.json',bundle);originals[str(root)]=s.file_hash(root/'bundle.json')
                specs.append({'key':group['request_family'],'root':str(root),'bundle':{'path':'bundle.json','sha256':s.file_hash(root/'bundle.json')},
                              'group_ids':[group['group_id']],'extension_recipes':[]})
            s.write(base/'two.json',{'owners':specs});r=s.assemble(base/'two.json',base/'two_plan')
            check('two_family_plan_preserves_absolute_roots',r['selected_jobs']==1856)
            calls=[]
            def resource(spec,bundle,reg,planned):
                return {'execution_policy':bundle['execution_policy'],'covered_job_ids':spec['owned_job_ids']}
            def group_executor(group,given,bundle,out,*,base_dir,policy,design,record):
                expected=json.loads((Path(base_dir)/bundle['relative_evidence_reference']).read_text())['root_index']
                check('stable_root_'+str(expected),expected==bundle['test_root_index'])
                for job in given:
                    calls.append(job['job_id']);record(s._row(job,'completed',primary_output_accepted=True))
            with patch.object(s,'verify_owner_resources',side_effect=resource),patch.object(s,'execute_group',side_effect=group_executor):
                result=s.execute(base/'two_plan/plan.json',base/'two_run')
            check('each_selected_job_attempted_exactly_once',len(calls)==len(set(calls))==1856)
            check('shared_ledger_retains_unowned_jobs',result['status_counts']=={'completed':1856,'blocked_unowned_scope':19476})
            check('signed_literal_paths_unchanged',all(s.file_hash(Path(root)/'bundle.json')==h for root,h in originals.items()))
            check('all_primary_claims_in_positive_control_are_explicit_mocks',result['primary_outputs_accepted']==1856)
            mutate_path=base/'mutating_plan.json';mutate_path.write_bytes((base/'two_plan/plan.json').read_bytes())
            original_read=Path.read_bytes;mutated=[]
            def captured_then_mutated(path):
                raw=original_read(path)
                if path==mutate_path and not mutated:
                    path.write_bytes(raw+b' ');mutated.append(True)
                return raw
            with patch.object(Path,'read_bytes',captured_then_mutated),patch.object(s,'verify_owner_resources',side_effect=resource), \
                 patch.object(s,'execute_group',side_effect=group_executor):
                revoked=s.execute(mutate_path,base/'revoked_run')
            check('captured_plan_bytes_detect_postread_mutation',revoked['status']=='completed_acceptance_revoked' and revoked['primary_outputs_accepted']==0)
            check('revocation_clears_all_final_ledger_primary_flags',not any(r['primary_output_accepted'] for r in json.loads((base/'revoked_run/jobs.json').read_text())))
            extension_root=base/'extension_source_root';extension_root.mkdir()
            extended=copy.deepcopy(bundle);extended['_dossier_group_ids']=[groups[0]['group_id']]
            extended['phase9_scope']={'key':groups[0]['request_family'],'group_ids':[groups[0]['group_id']],'extension_recipes':['fresh_graph_audit']}
            s.write(extension_root/'bundle.json',extended)
            human_root=base/'human_stage_root';human_root.mkdir();s.write(human_root/'bundle.json',{'software_dossier_only':True})
            ext_spec={'key':groups[0]['request_family'],'root':str(extension_root),'bundle':{'path':'bundle.json','sha256':s.file_hash(extension_root/'bundle.json')},
                      'group_ids':[],'extension_recipes':['fresh_graph_audit']}
            human_spec={'key':'human_admission','root':str(human_root),'bundle':{'path':'bundle.json','sha256':s.file_hash(human_root/'bundle.json')},
                        'group_ids':[],'extension_recipes':['human_admission_main']}
            s.write(base/'extensions.json',{'owners':[ext_spec,human_spec]});s.assemble(base/'extensions.json',base/'extensions_plan')
            extension_calls=[];dependency_roots=[]
            def extension_executor(job,bundles,out,context):
                extension_calls.append(job['job_id'])
                if job['recipe_id']=='human_admission_main':
                    check('human_coordinator_uses_own_evidence_root',context['base_dir']==str(human_root))
                    context['dispatch_module'].acceptance({'request_family':groups[0]['request_family']},bundles[groups[0]['request_family']],{},mode=s.dispatch.PRIMARY_MODE,base_dir=context['base_dir'])
                else: check('extension_coordinator_uses_task_owner_root_'+str(len(extension_calls)),context['base_dir']==str(extension_root))
                return {'job_id':job['job_id'],'status':'completed','primary_output_accepted':True}
            def dependency_acceptance(*args,**kwargs):
                dependency_roots.append(kwargs['base_dir']);return {'execution_allowed':True,'primary_inputs_accepted':True}
            with patch.object(s,'verify_owner_resources',side_effect=resource),patch.object(s.extensions,'run_extension',side_effect=extension_executor), \
                 patch.object(s.dispatch,'acceptance',side_effect=dependency_acceptance):
                r=s.execute(base/'extensions_plan/plan.json',base/'extensions_run')
            check('extension_and_human_jobs_routed_once',len(extension_calls)==len(set(extension_calls))==r['attempted_jobs'] and len(extension_calls)>1)
            check('human_dependency_keeps_original_task_root',dependency_roots==[str(extension_root)])
            check('extension_methods_get_terminal_status',all(m['status']==row['status'] for row in json.loads((base/'extensions_run/jobs.json').read_text()) for m in row['methods']))
            duplicate={'owners':[specs[0],specs[0]]};s.write(base/'duplicate.json',duplicate)
            r=s.assemble(base/'duplicate.json',base/'duplicate_plan')
            check('duplicate_owner_rejected',r['status']=='blocked' and not (base/'duplicate_plan/plan.json').exists())
            wrong=copy.deepcopy(specs[0]);wrong['group_ids']=[groups[1]['group_id']]
            s.write(base/'wrong.json',{'owners':[wrong]})
            check('cross_family_group_scope_rejected',s.assemble(base/'wrong.json',base/'wrong_plan')['status']=='blocked')
            missing=copy.deepcopy(specs[0]);missing['bundle']['sha256']='0'*64
            s.write(base/'missing.json',{'owners':[missing]});s.assemble(base/'missing.json',base/'missing_plan')
            with patch.object(s,'execute_group',side_effect=AssertionError('must not execute')):
                r=s.execute(base/'missing_plan/plan.json',base/'missing_run')
            check('changed_bundle_blocks_only_its_owned_cells',r['status_counts']['blocked_owner_input']==928 and r['attempted_jobs']==0)
            with patch.object(s,'verify_owner_resources',return_value={'execution_policy':policy,'covered_job_ids':[]}),patch.object(s,'execute_group',side_effect=AssertionError('must not execute')):
                r=s.execute(base/'two_plan/plan.json',base/'unqualified_run')
            check('resource_receipt_must_cover_selected_jobs',r['attempted_jobs']==0 and r['status_counts']['blocked_owner_input']==1856)
            # Direct group adapter controls retain frozen path checks and per-job errors.
            group=groups[0];given=[j for j in jobs if j.get('group_id')==group['group_id']][:2]
            manifest={'trajectories':[{'trajectory_id':j['trajectory_id'],'deletion_order':[],'checkpoints':[],'status':'software_zero_control'} for j in given]}
            accepted={'execution_allowed':True,'primary_inputs_accepted':True}
            seen=[];rows=[]
            with patch.object(s.dispatch,'acceptance',return_value=accepted) as gate,patch.object(s.dispatch,'resolve',return_value={'geometry':{},'x':None}), \
                 patch.object(s.dispatch,'request_for_group',return_value=manifest),patch.object(s.dispatch,'_check_path',side_effect=lambda job,path,bundle:seen.append(job['job_id'])):
                s.execute_group(group,given,{'execution_policy':policy},base/'zero_group',base_dir=base/'original_root_0',policy=policy,design={},record=rows.append)
            check('frozen_acceptance_receives_original_root',gate.call_args.kwargs['base_dir']==base/'original_root_0')
            check('every_zero_still_receives_frozen_path_check',seen==[j['job_id'] for j in given])
            check('zero_method_statuses_filled',all(r['status']=='structural_zero_or_unavailable_arm' and all(m['status']==r['status'] for m in r['methods']) for r in rows))
            rows=[]
            with patch.object(s.dispatch,'acceptance',return_value={'execution_allowed':False,'primary_inputs_accepted':False}),patch.object(s.dispatch,'resolve',side_effect=AssertionError('must not resolve')):
                s.execute_group(group,given,{'execution_policy':policy},base/'failed_group',base_dir=base,policy=policy,design={},record=rows.append)
            check('failed_group_acceptance_preserves_all_rows',len(rows)==2 and all(r['status']=='input_acceptance_failed' for r in rows))
            rows=[]
            s.execute_group(group,given,{'execution_policy':policy},base/'actual_missing_source_group',base_dir=base,policy=policy,design={},record=rows.append)
            check('actual_unmocked_missing_primary_inputs_refuse',len(rows)==2 and all(not r['primary_output_accepted'] and r['status']=='input_acceptance_failed' for r in rows))
            refit=next(g for g in registry['groups'] if g['block']=='E_full_refit')
            refit_jobs=[j for j in jobs if j.get('group_id')==refit['group_id']][:2];rows=[]
            with patch.object(s.dispatch,'acceptance',return_value=accepted),patch.object(s.dispatch,'resolve',return_value={'geometry':{}}), \
                 patch.object(s.dispatch,'request_for_group',return_value={'trajectories':[]}), \
                 patch.object(s.dispatch,'full_refit_group',return_value=([s._row(j,'completed',primary_output_accepted=True) for j in refit_jobs],{})) as runner:
                s.execute_group(refit,refit_jobs,{'execution_policy':policy},base/'refit_group',base_dir=base,policy=policy,design={},record=rows.append)
            check('full_refit_orchestration_invoked_once_for_owned_group',runner.call_count==1 and len(rows)==2)
            # Dependency acceptance routes the exact same object to its own root.
            value={};owner={'bundle':value,'resource_verified':True,'spec':{'root':str(base/'original_root_1')}}
            facade=s.RootedDispatch({'askubuntu/primary-10000':owner})
            with patch.object(s.dispatch,'acceptance',return_value=accepted) as gate:
                facade.acceptance({'request_family':'askubuntu/primary-10000'},value,{},mode=s.dispatch.PRIMARY_MODE,base_dir=base/'stage')
            check('stage_dependency_uses_source_root',gate.call_args.kwargs['base_dir']==owner['spec']['root'])
            refuses('substituted_dependency_object_rejected',lambda:facade.acceptance({'request_family':'askubuntu/primary-10000'},{},{},mode=s.dispatch.PRIMARY_MODE,base_dir=base))
            # Source/version mutation is a hard plan refusal, not a recompile.
            altered=copy.deepcopy(json.loads((base/'two_plan/plan.json').read_text()));altered['source_sha256']['phase9/study_assembly.py']='f'*64
            s.write(base/'changed_plan.json',altered)
            r=s.execute(base/'changed_plan.json',base/'source_changed_run')
            check('changed_plan_source_preserves_full_blocked_ledger',r['status']=='blocked' and r['every_planned_job_retained'] and r['attempted_jobs']==0)
            summary_path=base/'two_run/summary.json';ledger_path=base/'two_run/jobs.json'
            dossier={'phase9_dispatch_evidence':{'path':'two_run/summary.json','sha256':s.file_hash(summary_path)},
                     'job_outcomes':[{'job_id':calls[0],'outcome':{'path':'two_run/jobs.json','sha256':s.file_hash(ledger_path)}}]}
            check('statistics_requires_successful_complete_final_ledger',s.validate_statistics_dispatch(dossier,base)['planned_jobs']==21332)
            stale=copy.deepcopy(dossier);stale['job_outcomes'][0]['outcome']['path']='two_run/provisional_outcome.json'
            refuses('statistics_rejects_provisional_standalone_row',lambda:s.validate_statistics_dispatch(stale,base))
            refused=copy.deepcopy(dossier);refused['phase9_dispatch_evidence']={'path':'source_changed_run/summary.json','sha256':s.file_hash(base/'source_changed_run/summary.json')}
            refuses('statistics_rejects_revoked_or_failed_summary',lambda:s.validate_statistics_dispatch(refused,base))
            # Review preparation is metadata-only; unavailable acceptance and
            # qualification APIs are visibly mocked here, not valid testimony.
            root=base/'review_stages';root.mkdir();(root/'.gitignore').write_text('*\n')
            bundle={'_dossier_group_ids':[groups[0]['group_id']],'source_acceptance':{},'execution_policy':policy}
            descriptor=s.dossiers.dump_value(bundle,root/'candidate.json')
            s.write(root/'receipt.json',{'status':'unsigned_candidate_ready','candidate':descriptor})
            scope={'key':groups[0]['request_family'],'group_ids':[groups[0]['group_id']],'extension_recipes':[]}
            s.write(root/'development.json',{'scope':scope,'development':{'software_control':True},
                                           'external_evidence_review':{'software_control_preliminary':True}})
            r=s.prepare_resource_candidate(root/'candidate.json',root/'development.json')
            check('resource_candidate_binds_scope_before_qualification',r['status']=='resource_candidate_ready' and s.dossiers.load_value(root/'resource_candidate.phase9.json')['phase9_scope']==scope)
            qualification={'execution_policy':policy,'covered_job_ids':s._selected(scope,ownership,jobs),'software_only':True}
            evidence={'software_only':True};s.write(root/'qualification_attachment.json',{'qualification':qualification,'evidence':evidence})
            def verify(q,key,value,observations,*,base_dir):
                check('review_resource_scope_unchanged_'+str(len(checks)),value['phase9_scope']==scope and key==scope['key'] and observations==evidence)
                return q
            mockmodule=types.ModuleType('phase9.resource_profiles');mockmodule.verify_owner_resources=verify
            with patch.dict(sys.modules,{'phase9.resource_profiles':mockmodule}), \
                 patch('phase6.acceptance.accept_source_cache',return_value={'machine_acceptance_passed':True,'replay_binding_sha256':'software_only'}), \
                 patch.object(s.dossiers,'archive_replay',return_value={'software_only':True}), \
                 patch.object(s.dossiers,'_review_request',return_value={'software_only':True}), \
                 patch.object(s.dispatch,'acceptance',return_value=accepted):
                r=s.prepare_review(root/'resource_candidate.phase9.json',root/'qualification_attachment.json')
                check('qualification_attached_before_final_review',r['status']=='review_candidate_ready')
                supplied=s.dossiers.load_value(root/'review_candidate.phase9.json')
                check('reviewed_object_contains_exact_receipt_and_evidence',supplied['resource_policy_qualification'][scope['key']]==qualification and supplied['resource_policy_observations'][scope['key']]==evidence)
                s.write(root/'external_review.json',{'software_control_final':True})
                r=s.finalize_review(root/'review_candidate.phase9.json',root/'external_review.json')
                check('finalization_keeps_same_stable_root',r['status']=='selected_group_inputs_accepted_execution_still_rechecks' and (root/'bundle.phase9.final.json').exists())
                check('review_stages_never_authorize_execution',r['primary_execution_allowed'] is False)
                detached=copy.deepcopy(supplied);detached['resource_policy_observations']={}
                s.write(root/'evidence.json',evidence);s.write(root/'q.json',qualification)
                spec={'key':scope['key'],'root':str(root),'resource':{'receipt':{'path':'q.json','sha256':s.file_hash(root/'q.json')},'request':{'path':'evidence.json','sha256':s.file_hash(root/'evidence.json')}}}
                refuses('unattached_external_resource_evidence_refused',lambda:s.verify_owner_resources(spec,detached,registry,jobs))
        check('all_bound_sources_unchanged',source=={**s.codes(),'phase9/check_study_assembly.py':s.file_hash(__file__)})
        report.update(status='passed',check_count=len(checks),checks=checks)
    except Exception as error:
        report.update(check_count=len(checks),checks=checks,exception_type=type(error).__name__,reason=str(error))
        history=ROOT/'phase9/history'/out.stem;history.mkdir(parents=True,exist_ok=False)
        for name in ('study_assembly.py','check_study_assembly.py'):
            (history/name).write_bytes((ROOT/'phase9'/name).read_bytes())
        report['failed_source_snapshot']=str(history.relative_to(ROOT))
    s.write(out,report);return report


if __name__=='__main__':
    result=run(sys.argv[1]);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2));sys.exit(result['status']!='passed')
