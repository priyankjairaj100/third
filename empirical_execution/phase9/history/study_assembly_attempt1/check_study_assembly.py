"""Metadata-only scheduler controls; no corpus, human, or worker execution."""
from pathlib import Path
import copy
import json
import sys
import tempfile
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
