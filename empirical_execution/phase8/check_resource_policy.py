"""Metadata-only software checks. No measurements, corpora, or benchmarks run.

Positive controls mock upstream source/cache/development acceptance and shape
lineage explicitly. Their numeric metadata are algebraic fixtures, never
native measurements. The real observation reader and policy verifier execute.
"""
from pathlib import Path
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
from unittest.mock import patch
import argparse
import io
import json
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from phase8 import resource_policy as resource
from phase6 import recipes


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def metadata_fixture(base):
    """Untrusted software metadata only; cannot pass real input acceptance."""
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    registry,_=recipes.build_registry()
    group=next(g for g in registry['groups'] if g['group_id']=='C_full_methods/civil_comments/primary-10000/native')
    profiles=[]
    for method in group['methods']:
        for full in ([True,False] if method in resource.SUMMARIES|{'B-E'} else [True]):
            profiles.append(dict(profile_id=method+('-full' if full else '-light'),method=method,unit='record',full_state=full,
                records=10_000,curator_dimension=768,learner_dimension=768,outputs=1,horizon=128,checkpoints=4,
                checkpoint_schedule=[1,8,32,128],
                encoder='e5',encoder_id='intfloat/multilingual-e5-base',encoder_revision='software-fixture-not-a-model-revision',
                solver='cholesky',panel='cold_process',lambda_reg=.01,threshold=.6,batch_rows=1024,compaction_fraction=.5))
    snapshot=dict(physical_available_bytes=8*1024**3,cgroup_remaining_bytes=None,local_cpu_count=1)
    times=[2.,2.]*len(profiles)
    lock=recipes.derive_resource_policy(snapshot,times)
    policy=dict(memory_bytes=lock['per_method_address_space_limit_bytes'],cpu_seconds=lock['cpu_seconds_each_stage'],
                wall_seconds=lock['wall_seconds_each_stage'],threads=1)
    observed_policy={**policy,resource.preflight.PAIR_CAP:40_000_000_000,resource.preflight.COEFFICIENT_CAP:6_000_000}
    hashes={name:resource.digest({'software_only_input_name':name}) for name in ('metadata.json','graph.npz','curator.npy','learner.npy','targets.npy')}
    development=dict(resource_snapshot=snapshot,stage_seconds=times,resource_lock=lock,records=[],
                     service_reports=[],software_fixture_only=True)
    replay={'machine_acceptance_passed':True,'replay_binding_sha256':'a'*64,
            'validated_bindings':{'train_ids':[],'genuine_source_ids':[]}}
    accepted={'development_source_replay_sha256':'b'*64,'pilot_input_file_hashes':hashes,
              'actual_worker_array_lineage_checked':True,'software_fixture_only':True}
    shapes=dict(records=10_000,curator_dimension=768,learner_dimension=768,outputs=1,threshold=.6,
        encoder='e5',encoder_id=profiles[0]['encoder_id'],encoder_revision=profiles[0]['encoder_revision'],
        input_hashes=hashes,graph_edges=7,maximum_blocker_degree=3,feature_density=1.,input_array_bytes=61_520_000)
    codes=resource.source_bindings()
    worker_codes={str(path.relative_to(resource.ROOT)):resource.preflight.sha256(path.read_bytes())
        for folder in ('ccu','phase3','phase4','phase5','phase6') for path in sorted((resource.ROOT/folder).glob('*.py'))}
    def artifact(name,value):
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True);write(path,value)
        return {'path':name,'sha256':resource.preflight.sha256(path.read_bytes())}
    entries=[]
    for index,profile in enumerate(profiles):
        prefix='software_observation_'+str(index)
        requests=[[f'software-unit-{i}' for i in range(a,b)] for a,b in zip([0,1,8,32],[1,8,32,128])]
        request_hash=resource.hashlib.sha256(json.dumps(requests,separators=(',',':')).encode()).hexdigest()
        method=profile['method'];used={k:v for k,v in hashes.items() if k!='curator.npy' or method=='O-G'}
        config={key:profile[key] for key in ('method','unit','solver','panel','horizon','lambda_reg','batch_rows','compaction_fraction')}
        config.update(requests=requests,policy=observed_policy,persistence='every_release',input_hashes=used,code_hashes=worker_codes)
        config_ref=artifact(prefix+'/audit_service/prospective_configuration.json',config)
        stage=dict(exit_code=0,timeout=False,spawn_to_reap_seconds=2.,peak_rss_bytes=10_000_000,
                   sampled_process_tree_peak_rss_sum_bytes=12_000_000,user_cpu_seconds=.5,system_cpu_seconds=.1)
        service={key:profile[key] for key in ('method','unit','solver','panel')}
        service.update(success=True,persistence='every_release',policy=observed_policy,raw_input_files_sha256=used,
            project_python_source_hashes=worker_codes,request_sha256=request_hash,
            construction=stage,repair=stage,artifact_hashes={'prospective_configuration.json':config_ref['sha256']},
            software_fixture_only=True)
        service_ref=artifact(prefix+'/audit_service/service_report.json',service)
        reference_ref=artifact(prefix+'/reference/service_report.json',service)
        development['service_reports'].append(reference_ref)
        rows=[]
        for i in range(4):
            row=dict(index=i,passed=True,reconstructed_actual_snapshot_bound=True,replay_and_measured_heads_bitwise_equal=True,
                     audit_seconds=.1,retained_records=10_000-[1,8,32,128][i],
                     deleted_unique_units=[1,8,32,128][i],remaining_horizon=128-[1,8,32,128][i])
            if method in resource.SUMMARIES and profile['full_state']:
                row['coefficients']={'passed':True,'complete_union_of_keys_compared':True,
                    'rows':[{'expected_designated_key':True} for _ in range(20)]}
            rows.append(row)
        audit=dict(schema='ccu-complete-state-audit-v1',passed=True,status='passed',method=method,
            full_state_requested=profile['full_state'],code_sha256=codes['phase6/state_audit.py'],
            audit_service_report_sha256=service_ref['sha256'],measured_service_report_sha256=reference_ref['sha256'],
            checkpoints=rows,prospective_audit_policy={'absolute_tolerance':1e-11,'relative_tolerance':1e-10,
                'max_pair_coordinates':40_000_000_000,'max_coefficient_coordinates':6_000_000},
            charged_total_audit_seconds=6.,charged_worker_lifecycle_seconds=4.,auditor_process_lifetime_peak_rss_bytes=10_000_000,
            audit_service_snapshot_bytes=100,software_fixture_only=True)
        audit['sha256']=resource.digest(audit)
        audit_ref=artifact(prefix+'/state_audit.json',audit)
        entries.append(dict(observation_id=profile['profile_id'],audit_report=audit_ref,audit_service_report=service_ref,
                            reference_service_report=reference_ref,audit_configuration=config_ref))
    plan={'evidence_role':'actual_disjoint_native_development','development_source_replay_sha256':'b'*64,
          'resource_snapshot_sha256':resource.digest(snapshot),'development_resource_lock_sha256':resource.digest(lock),
          'target_profiles_sha256':resource.digest(profiles),'observation_ids':[e['observation_id'] for e in entries],
          'software_fixture_only':True,'fixture_warning':'Untrusted hypothetical metadata; upstream real acceptance must fail.'}
    manifest={'schema':'ccu-native-audit-observations-1','plan':artifact('software_plan.json',plan),'entries':entries}
    bundle={'development':development,'external_evidence_review':{'development_source_replay_sha256':'b'*64},
        'source_acceptance':{},'_dossier_group_ids':[group['group_id']], 'dataset_id':'civil_comments',
        'software_fixture_only':True}
    return dict(base=base,group=group,profiles=profiles,policy=policy,bundle=bundle,manifest=manifest,
                replay=replay,accepted=accepted,shapes=shapes,codes=codes)


def mocked_inputs(fixture):
    """Only acceptance/shape lineage is mocked; observation parsing is real."""
    stack=ExitStack()
    stack.enter_context(patch.object(resource,'_real_acceptance',return_value=(fixture['replay'],fixture['accepted'])))
    stack.enter_context(patch.object(resource,'_input_shapes',return_value=fixture['shapes']))
    stack.enter_context(patch.object(resource,'_validate_target_bindings',return_value={'software_fixture_only':True}))
    stack.enter_context(patch('phase6.dispatch.development_acceptance',return_value=fixture['accepted']))
    return stack


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args(argv)
    if args.out.exists():parser.exit(2,'Preserve existing result: '+str(args.out)+'\n')
    checks=[]
    def check(name, value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def refuses(name, function):
        try:function()
        except (ValueError,TypeError,KeyError):check(name,True)
        else:raise AssertionError(name)
    initial_codes=resource.source_bindings()
    with tempfile.TemporaryDirectory(prefix='resource-policy-software-only-') as temporary:
        fixture=metadata_fixture(Path(temporary)/'candidate')
        f=fixture
        observations,plan=resource._read_observations(f['manifest'],f['bundle'],f['base'],f['accepted'],f['shapes'],f['policy'],f['codes'],f['profiles'])
        check('metadata_only_reader_retains_all_12_full_light_observations',len(observations)==12)
        recipe=resource.select_observed_envelope(f['profiles'],observations,f['policy'])
        check('common_pair_cap_exact',recipe['execution_policy'][resource.preflight.PAIR_CAP]==38_396_160_000)
        check('common_coefficient_cap_is_observed_minimum',recipe['execution_policy'][resource.preflight.COEFFICIENT_CAP]==5_921_300)
        check('future_symbolic_counts_never_invented',all(row['future_exact_symbolic_keys'] is None for row in recipe['coverage']))
        check('unknown_graph_demand_can_exceed_locked_cap',any(row['coefficient_cap_sufficient_for_all_possible_keys'] is False for row in recipe['coverage']))
        check('convex_caps_never_selected_or_changed',recipe['convex_cap_changed'] is False and resource.preflight.CONVEX_CAP not in recipe['execution_policy'])
        check('full_resource_and_precision_fields_unchanged',all(recipe['execution_policy'][k]==v for k,v in f['policy'].items()))
        rows=deepcopy(observations);rows[0]['records']=9999
        refuses('missing_joint_row_coverage_refuses',lambda:resource.select_observed_envelope(f['profiles'],rows,f['policy']))
        rows=deepcopy(observations);rows[0]['encoder']='mpnet'
        refuses('equal_dimensions_different_encoder_refuses',lambda:resource.select_observed_envelope(f['profiles'],rows,f['policy']))
        rows=deepcopy(observations);rows[0]['encoder_revision']='changed'
        refuses('different_encoder_revision_refuses',lambda:resource.select_observed_envelope(f['profiles'],rows,f['policy']))
        rows=deepcopy(observations);rows[0]['observed_worker_limits_fit']=False
        refuses('reported_limits_failure_refuses',lambda:resource.select_observed_envelope(f['profiles'],rows,f['policy']))
        rows=deepcopy(observations)
        for row in rows:
            if row['method'] in resource.SUMMARIES:row['maximum_fresh_symbol_coordinates']=1
        refuses('lower_than_default_coefficient_coverage_refuses',lambda:resource.select_observed_envelope(f['profiles'],rows,f['policy']))
        target=[f['profiles'][0]];first=deepcopy(observations[0]);second=deepcopy(first)
        first['records']=9999;second['checkpoints']=1
        refuses('separate_maxima_do_not_manufacture_joint_coverage',lambda:resource.select_observed_envelope(target,[first,second],f['policy']))
        profiles=deepcopy(f['profiles']);profiles[0]['full_state']=1
        refuses('boolean_scope_is_strict',lambda:resource.validate_profiles(profiles))
        profiles=deepcopy(f['profiles']);profiles[0]['records']=True
        refuses('boolean_count_refused',lambda:resource.validate_profiles(profiles))

        # Empty arrays communicate dimensions only. No synthetic record,
        # embedding, label, or empirical measurement is constructed here.
        import numpy as np
        from phase5 import task_program as task
        entry={'train':np.empty((0,768),np.float32),'encoder_id':f['profiles'][0]['encoder_id'],
               'encoder_revision':f['profiles'][0]['encoder_revision']}
        target_bundle={'curators':{'e5':entry},'learners':{'e5':entry},'train_y':np.empty((0,1),np.float64),
            'requests':{'configuration':{'requests':{'record_horizon':128,'record_checkpoints':[1,8,32,128]}}}}
        target_replay={'validated_bindings':{'train_ids':range(10_000),'genuine_source_ids':[]}}
        with patch.object(task,'_parameter',side_effect=lambda entry,kind,encoder:(.6 if kind=='threshold' else .01,'software-only')):
            scope=resource._validate_target_bindings(f['profiles'],target_bundle,target_replay,f['group'])
            check('actual_canonical_full_light_profile_contract_passes_metadata_control',scope['covered_profiles']==12)
            for full in (True,False):
                profiles=deepcopy(f['profiles']);profiles.pop(next(i for i,p in enumerate(profiles) if p['full_state'] is full))
                refuses('missing_required_'+str(full)+'_profile_refuses',lambda profiles=profiles:resource._validate_target_bindings(profiles,target_bundle,target_replay,f['group']))
            wrong=deepcopy(f['group']);wrong['block']='F_convex'
            refuses('unsupported_group_cannot_borrow_C_qualification',lambda:resource._validate_target_bindings(f['profiles'],target_bundle,target_replay,wrong))
            for name,value in [('solver','cg'),('encoder_revision','changed'),('threshold',.5),('lambda_reg',.02),('horizon',64),('checkpoint_schedule',[1,2,4,128])]:
                profiles=deepcopy(f['profiles']);profiles[0][name]=value
                refuses('canonical_profile_'+name+'_change_refused',lambda profiles=profiles:resource._validate_target_bindings(profiles,target_bundle,target_replay,f['group']))

        def altered_artifact(entry_key, mutate, check_name):
            manifest=deepcopy(f['manifest']);reference=manifest['entries'][0][entry_key]
            path=f['base']/reference['path'];raw=path.read_bytes();value=json.loads(raw);mutate(value)
            if entry_key=='audit_report':value['sha256']=resource.digest({k:v for k,v in value.items() if k!='sha256'})
            write(path,value);reference['sha256']=resource.preflight.sha256(path.read_bytes())
            try:
                refuses(check_name,lambda:resource._read_observations(manifest,f['bundle'],f['base'],f['accepted'],f['shapes'],f['policy'],f['codes'],f['profiles']))
            finally:path.write_bytes(raw)
        altered_artifact('audit_report',lambda v:v.update(passed=False,status='audit_failed'),'failed_attempt_refuses_policy')
        altered_artifact('audit_report',lambda v:v.update(method='P-S'),'retagged_method_refused')
        altered_artifact('audit_report',lambda v:v.update(full_state_requested=1),'retagged_scope_refused')
        altered_artifact('audit_report',lambda v:v['prospective_audit_policy'].update(max_pair_coordinates=1),'retagged_policy_refused')
        altered_artifact('audit_report',lambda v:v.update(code_sha256='f'*64),'changed_auditor_source_refused')
        altered_artifact('audit_report',lambda v:v['checkpoints'][-1].update(deleted_unique_units=4),'configured_horizon_without_actual_distinct_deletions_refused')
        altered_artifact('audit_service_report',lambda v:v['project_python_source_hashes'].update({'ccu/core.py':'f'*64}),'changed_dependency_refused')
        manifest=deepcopy(f['manifest']);manifest['entries']=manifest['entries'][:-1]
        refuses('missing_planned_observation_refused',lambda:resource._read_observations(manifest,f['bundle'],f['base'],f['accepted'],f['shapes'],f['policy'],f['codes'],f['profiles']))
        manifest=deepcopy(f['manifest']);manifest['entries'][0]['audit_report']['path']='../../escape.json'
        refuses('path_escape_refused',lambda:resource._read_observations(manifest,f['bundle'],f['base'],f['accepted'],f['shapes'],f['policy'],f['codes'],f['profiles']))

        # Actual upstream acceptance rejects this complete-looking fixture.
        blocked=resource.qualify_resource_policy(f['group'],f['bundle'],f['manifest'],f['profiles'],base_dir=f['base'])
        check('unmocked_hypothetical_native_metadata_is_blocked',blocked['execution_policy'] is None and blocked['actual_native_observations_accepted']==0)
        refuses('blocked_receipt_never_verifies',lambda:resource.verify_resource_receipt(blocked,f['group'],f['bundle'],f['manifest'],f['profiles'],base_dir=f['base']))
        with mocked_inputs(f):
            qualified=resource.qualify_resource_policy(f['group'],f['bundle'],f['manifest'],f['profiles'],base_dir=f['base'])
            check('mocked_input_positive_control_uses_real_reader_and_selector',qualified['status']=='qualified_observed_count_policy_pending_external_review')
            attached=deepcopy(f['bundle']);attached['resource_policy_qualification']={f['group']['group_id']:qualified}
            attached['resource_policy_observations']={'development_audits':f['manifest'],'target_profiles':f['profiles']}
            attached['execution_policy']=qualified['execution_policy']
            attached['external_evidence_review']={'development_source_replay_sha256':'b'*64,'software_final_testimony':True}
            verified=resource.verify_resource_receipt(qualified,f['group'],attached,f['manifest'],f['profiles'],base_dir=f['base'])
            check('receipt_stable_after_attachment_and_final_review',verified==qualified)
            bad=deepcopy(attached);bad['external_evidence_review']['development_source_replay_sha256']='c'*64
            refuses('changed_preliminary_replay_digest_refuses',lambda:resource.verify_resource_receipt(qualified,f['group'],bad,f['manifest'],f['profiles'],base_dir=f['base']))
            bad=deepcopy(attached);bad['actual_input_changed']='unexpected'
            refuses('changed_actual_bundle_input_refuses',lambda:resource.verify_resource_receipt(qualified,f['group'],bad,f['manifest'],f['profiles'],base_dir=f['base']))
            with patch.object(resource,'source_bindings',side_effect=[initial_codes,{**initial_codes,'phase8/resource_policy.py':'f'*64}]):
                changed=resource.qualify_resource_policy(f['group'],f['bundle'],f['manifest'],f['profiles'],base_dir=f['base'])
            check('source_change_during_replay_blocks_export',changed['execution_policy'] is None)

            # Real assembler attach/finalize calls; the resource verifier remains real.
            from phase8 import dossiers
            base=f['base'];(base/'.gitignore').write_text('*\n')
            descriptor=dossiers.dump_value(f['bundle'],base/'candidate.json')
            write(base/'groups.json',[f['group']])
            write(base/'receipt.json',{'status':'unsigned_candidate_ready','candidate':descriptor})
            write(base/'attachment.json',{'development':f['bundle']['development'],
                'external_evidence_review':f['bundle']['external_evidence_review'],'resource_qualification':qualified,
                'development_audits':f['manifest'],'target_profiles':f['profiles']})
            with patch.object(dossiers.acceptance,'accept_source_cache',return_value=f['replay']), \
                 patch.object(dossiers,'archive_replay',return_value={'software_fixture_only':True}), \
                 patch.object(dossiers.dispatch,'acceptance',return_value={'primary_inputs_accepted':True,'software_fixture_only':True}):
                attach=dossiers.attach_development(base/'candidate.json',base/'attachment.json')
                check('actual_assembler_attach_roundtrip_with_real_resource_verifier',attach['status']=='review_candidate_ready')
                write(base/'review.json',{'development_source_replay_sha256':'b'*64,'software_final_testimony':True})
                final=dossiers.finalize(base/'review_candidate.json',base/'review.json')
                check('actual_assembler_finalize_roundtrip_with_real_resource_verifier',final['status']=='selected_groups_accepted_dispatch_still_rechecks')
                check('mock_roundtrip_never_authorizes_study',final['primary_execution_allowed'] is False)

        request={'group':f['group'],'bundle':{},'development_audits':{},'target_profiles':f['profiles']}
        write(Path(temporary)/'missing.json',request)
        with redirect_stdout(io.StringIO()):
            result=resource.main(['--request',str(Path(temporary)/'missing.json'),'--out',str(Path(temporary)/'blocked')])
        check('actual_missing_input_CLI_exports_no_policy',result['status']=='blocked' and not (Path(temporary)/'blocked/execution_policy.json').exists())
        try:
            with redirect_stderr(io.StringIO()):resource.main(['--request',str(Path(temporary)/'missing.json'),'--out',str(Path(temporary)/'blocked')])
        except SystemExit as error:check('existing_output_directory_preserved',error.code==2)
        else:raise AssertionError('existing_output_directory_preserved')
    check('all_bound_sources_unchanged',resource.source_bindings()==initial_codes)
    report={'schema':'ccu-phase8-resource-policy-checks-1','status':'passed','passed':len(checks),'checks':checks,
        'evidence_role':'metadata_only_software_checks_with_explicit_upstream_mocks',
        'actual_native_measurements':0,'actual_native_policies_exported':0,'benchmarks_run':0,
        'positive_control_scope':'Real observation reader, selector, verifier, and assembler attach/finalize; source/cache/shape/development/full acceptance mocked.',
        'source_sha256':{**initial_codes,'phase8/check_resource_policy.py':resource.preflight.sha256(Path(__file__).read_bytes()),
                         'phase8/dossiers.py':resource.preflight.sha256((resource.ROOT/'phase8/dossiers.py').read_bytes())}}
    write(args.out,report);print(json.dumps({'status':'passed','checks':len(checks),'report':str(args.out)}))


if __name__=='__main__':main()
