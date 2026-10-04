"""Explicit algebra/metadata software controls; no empirical dataset or benchmark.

Only four toy numerical rows and reserved algebra IDs are used. Upstream source,
cache, calibration and development acceptance are mocked for positive controls;
those same inputs must fail unmocked admission. No semantic encoder is run.
"""
from pathlib import Path
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import argparse,json,sys,tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from phase9 import resource_profiles as r
from phase3.reference_graph import build_reference_graph
from phase8 import dossiers


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def ref(base,name,value):
    write(base/name,value);return {'path':name,'sha256':r.sha(base/name)}


def fixture(base):
    """Small algebra arrays plus untrusted invented metadata, never evidence."""
    ids=['algebra-row-'+str(i) for i in range(4)];owners=['algebra-source-'+str(i) for i in range(4)]
    rows=[{'record_id':i,'source_unit_id':s,'source_kind':'native_algebra_software_only','original_fields':{'toxicity':.5}} for i,s in zip(ids,owners)]
    x=np.ones((4,768),np.float32);x[1]*=-1;x[2,::2]*=-1;x[3,1::2]*=-1
    y=np.full((4,1),.5,np.float64);cache=base.parent/(base.name+'_accepted_source_fixture');cache.mkdir()
    prepared=cache/'records.jsonl';prepared.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    curators={};learners={};caches={}
    for name,values in [('e5',x),('mpnet',np.roll(x,1,axis=1))]:
        directory=cache/name;directory.mkdir();np.save(directory/'vectors.npy',values,allow_pickle=False)
        identity={'encoder_id':name+'-SOFTWARE-ONLY','encoder_revision':'not-an-actual-model'}
        curators[name]={**identity,'train_ids':ids,'train':values,'threshold_lock':r.task.engineering_parameter('threshold',.6,representation=name)}
        learners[name]={**identity,'train_ids':ids,'train':values,'evaluation_ids':['algebra-eval'],'evaluation':values[:1],
            'lambda_lock':r.task.engineering_parameter('lambda',.01,representation=name)}
        caches[name]={'directory':str(directory)}
    snapshot={'physical_available_bytes':8*1024**3,'cgroup_remaining_bytes':None,'local_cpu_count':1}
    lock=r.recipes.derive_resource_policy(snapshot,[2.,2.])
    development={'resource_snapshot':snapshot,'stage_seconds':[2.,2.],'resource_lock':lock,'records':rows,
        'development_source_ids':owners,'input_bundle_directory':'pilot','service_reports':[], 'software_fixture_only':True}
    bundle={'task':'civil','dataset_id':'civil_comments','train_ids':ids,'evaluation_ids':['algebra-eval'],'source_ids':owners,
        'source_kinds':{s:'unknown_singleton' for s in owners},'train_y':y,'evaluation_y':y[:1],
        'curators':curators,'learners':learners,'development':development,'base_curator':'e5','base_learner':'e5',
        'source_acceptance':{'prepared':{'records':str(prepared)},'caches':caches},
        'external_evidence_review':{'development_source_replay_sha256':'b'*64},'evidence_role':'algebra_software_only'}
    group='B_relevance/civil_comments/primary-10000/native'
    bundle['_dossier_group_ids']=[group];bundle['phase9_scope']={'key':'civil_comments/primary-10000','group_ids':[group],'extension_recipes':[]}
    registry,jobs=r.recipes.build_registry();byid={g['group_id']:g for g in registry['groups']}
    paths={}
    for job in jobs:
        if not job.get('trajectory_id') or job.get('recipe_id'):continue
        tid=job['trajectory_id'];arm=job['arm'];unavailable=arm=='S' or 'matched' in arm
        order=[] if unavailable else list(ids)
        planned=job.get('planned_checkpoint_units',[1,8,32,128]);planned=planned if isinstance(planned,list) else [1,8,32,128]
        cp=sorted({min(k,len(order)) for k in planned}) if order else []
        paths[tid]={'trajectory_id':tid,'arm':arm,'unit':'source' if arm=='S' else 'record','deletion_order':order,'checkpoints':cp,'status':'algebra_fixture'}
    accepted={'development_source_replay_sha256':'b'*64,'actual_worker_array_lineage_checked':True}
    replay={'machine_acceptance_passed':True,'replay_binding_sha256':'a'*64,'validated_bindings':{'genuine_source_ids':[]}}
    def request(group,b,c):return {'trajectories':[deepcopy(paths[j['trajectory_id']]) for j in jobs if j.get('group_id')==group['group_id'] and j.get('trajectory_id')]}
    def mocked():
        stack=ExitStack();stack.enter_context(patch.object(r,'_accept_group',return_value=(replay,accepted)))
        stack.enter_context(patch.object(r.dispatch,'request_for_group',side_effect=request))
        stack.enter_context(patch.object(r.dispatch,'development_acceptance',return_value=accepted));return stack
    return bundle,byid,paths,accepted,replay,mocked


def package(base,bundle,profile):
    sig=profile['signature'];ids=bundle['train_ids'];owners=bundle['source_ids'];cx=bundle['curators'][sig['curator']['name']]['train']
    x=bundle['learners'][sig['learner']['name']]['train'] if sig['learner'] else None
    if sig['projection_sha256'] is not None:x=(x.astype(np.float64)@bundle['projections'][str(sig['learner_dimension'])]['matrix']).astype(np.float32)
    graph=build_reference_graph(cx,ids,sig['threshold'],owners,seed=sig['priority_seed'])
    target=base/('package_'+profile['profile_id'][:12]);target.mkdir(exist_ok=True)
    np.save(target/'curator.npy',cx,allow_pickle=False)
    if x is not None:np.save(target/'learner.npy',x,allow_pickle=False);np.save(target/'targets.npy',bundle['train_y'],allow_pickle=False)
    np.savez(target/'graph.npz',priority=graph.priority_indices,indptr=graph.indptr,indices=graph.indices)
    write(target/'metadata.json',{'record_ids':ids,'source_ids':owners,'threshold':sig['threshold'],'priority_seed':sig['priority_seed'],'evidence_role':'algebra_software_only'})
    files={p.name:{'sha256':r.sha(p)} for p in target.iterdir() if p.name!='bundle.json'};write(target/'bundle.json',{'files':files})
    return target.name,files


def worker_entry(base,bundle,profile):
    """No process executed: fabricated metadata exercises rejection contracts."""
    sig=profile['signature'];name,files=package(base,bundle,profile);method=sig['method']+('-FP32' if sig['state_precision']=='FP32' else '')
    policy,_=r._base_policy(bundle);policy.update({r.pf.PAIR_CAP:100_000_000,r.pf.COEFFICIENT_CAP:4_000_000})
    used={k:files[k]['sha256'] for k in ['metadata.json','graph.npz','learner.npy','targets.npy']+(['curator.npy'] if method in {'O-G','O-G-FP32'} else [])}
    schedule=sig['checkpoint_schedule'];order=bundle['train_ids'];last=0;batches=[]
    for end in schedule:batches.append(order[last:end]);last=end
    config={k:sig[k] for k in ('unit','horizon','lambda_reg','solver','panel','persistence','batch_rows','compaction_fraction')}
    config.update(method=method,policy=policy,input_hashes=used,requests=batches,code_hashes=r._worker_sources())
    prefix='observation_'+profile['profile_id'][:12];conf=ref(base,prefix+'/prospective_configuration.json',config)
    stage={'exit_code':0,'timeout':False,'spawn_to_reap_seconds':2.,'user_cpu_seconds':.1,'system_cpu_seconds':.1,'peak_rss_bytes':1000000,'sampled_process_tree_peak_rss_sum_bytes':1000000}
    report={k:config[k] for k in ('method','unit','solver','panel','persistence','policy')}
    report.update(success=True,raw_input_files_sha256=used,project_python_source_hashes=r._worker_sources(),request_sha256=r.hashlib.sha256(json.dumps(batches,separators=(',',':')).encode()).hexdigest(),
        construction=stage,repair=stage,artifact_hashes={'prospective_configuration.json':conf['sha256']},software_fixture_only=True)
    entry={'observation_id':prefix,'profile_id':profile['profile_id'],'input_package':name,'configuration':conf,'service_report':ref(base,prefix+'/service_report.json',report)}
    return entry


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=r.codes();source['phase9/check_resource_profiles.py']=r.sha(__file__)
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError,FileNotFoundError):checks.append(name);return
        raise AssertionError(name)
    report={'schema':'ccu-resource-profile-checks-1','status':'failed','source_sha256':source,
        'evidence_role':'explicit_algebra_and_metadata_software_only_upstream_acceptance_mocks',
        'primary_study_started':False,'native_observations':0,'workers_executed':0,'real_semantic_embeddings':False}
    try:
      with tempfile.TemporaryDirectory(prefix='resource_profiles_software_') as tmp:
        base=Path(tmp)/'owner';base.mkdir();bundle,groups,paths,accepted,replay,mocked=fixture(base);key=bundle['phase9_scope']['key']
        from phase9 import study_assembly as assembly
        registry,jobs=r.recipes.build_registry();ownership=assembly.canonical_owners(registry,jobs)
        for group in registry['groups']:
            given={**bundle,'_dossier_group_ids':[group['group_id']],'phase9_scope':{'key':group['request_family'],'group_ids':[group['group_id']],'extension_recipes':[]}}
            _,_,owned=r.canonical_owner_jobs(group['request_family'],given)
            check('canonical_'+group['group_id'],[j['job_id'] for j in owned]==assembly._selected(given['phase9_scope'],ownership,jobs))
        bad=deepcopy(bundle);bad['phase9_scope']['group_ids']=['not-registered']
        reject('unregistered_group_refused',lambda:r.canonical_owner_jobs(key,bad))
        bad=deepcopy(bundle);bad['phase9_scope']['extension_recipes']=['human_admission_main']
        reject('global_human_owner_not_retagged',lambda:r.canonical_owner_jobs(key,bad))
        base_projection=r.digest(r.projection(bundle));attached=deepcopy(bundle)
        attached['resource_policy_qualification']={key:{'self':'ignored'}};attached['resource_policy_observations']={key:{'later':'ignored'}}
        attached['external_evidence_review']['final_testimony']='software-only'
        check('projection_cycle_free',r.digest(r.projection(attached))==base_projection)
        attached['external_evidence_review']['development_source_replay_sha256']='c'*64
        check('development_replay_change_bound',r.digest(r.projection(attached))!=base_projection)
        attached=deepcopy(bundle);attached['phase9_scope']['extension_recipes']=['negative_controls']
        check('selected_recipe_scope_bound',r.digest(r.projection(attached))!=base_projection)
        with mocked():manifest,context=r.compile_profiles(key,bundle,base_dir=base)
        check('complete_canonical_job_scope',manifest['covered_job_ids']==assembly._selected(bundle['phase9_scope'],ownership,jobs))
        check('unknown_source_arms_retained_unavailable',any(j['disposition']=='unavailable_or_structural_zero' for j in manifest['jobs']))
        check('workers_profiles_exact_methods',{p['signature']['method'] for p in manifest['profiles']}=={'O-G','B-F'})
        check('full_horizon_clipped_from_actual_path',all(p['signature']['checkpoint_schedule']==[1,4] for p in manifest['profiles']))
        observation_entries=[worker_entry(base,bundle,p) for p in manifest['profiles']]
        evidence={'plan':ref(base,'plan.json',{'evidence_role':'actual_disjoint_development','profile_manifest_sha256':manifest['sha256'],'observation_ids':[e['observation_id'] for e in observation_entries],
            'development_resource_lock_sha256':r.digest(manifest['resource_lock'])}),'observations':observation_entries}
        with mocked():q=r.qualify_owner_resources(key,bundle,evidence,base_dir=base)
        check('real_reader_and_selector_positive_under_upstream_mocks',q['execution_policy'] is not None)
        check('positive_control_never_primary_acceptance',q['execution_allowed'] is False and q['complete_primary_acceptance'] is False)
        updated=deepcopy(bundle);updated['execution_policy']=q['execution_policy'];updated['resource_policy_qualification']={key:q};updated['resource_policy_observations']={key:evidence};updated['external_evidence_review']['final_testimony']='software-only'
        with mocked():check('real_verifier_stable_after_attachment_and_final_testimony',r.verify_owner_resources(q,key,updated,evidence,base_dir=base)==q)
        changed=deepcopy(updated);changed['external_evidence_review']['development_source_replay_sha256']='c'*64
        with mocked():reject('real_verifier_rejects_changed_development_binding',lambda:r.verify_owner_resources(q,key,changed,evidence,base_dir=base))
        check('same_algebra_inputs_unmocked_fail_closed',r.qualify_owner_resources(key,bundle,evidence,base_dir=base)['execution_policy'] is None)
        empty={'plan':evidence['plan'],'observations':[]}
        with mocked():check('missing_observation_blocks',r.qualify_owner_resources(key,bundle,empty,base_dir=base)['execution_policy'] is None)
        profile=manifest['profiles'][0];entry=observation_entries[0]
        derived=r.replay_derived_package(profile,entry,bundle,context,base_dir=base)
        check('accepted_absolute_sibling_cache_path_supported',derived['records']==4)
        bad=deepcopy(entry);bad['input_package']='../../escape'
        reject('observation_package_root_escape_refused',lambda:r.replay_derived_package(profile,bad,bundle,context,base_dir=base))
        wrong=deepcopy(profile);wrong['signature']['curator']['revision']='different'
        reject('mixed_cache_revision_retag_refused',lambda:r.replay_derived_package(wrong,entry,bundle,context,base_dir=base))
        # Exact fixed projection from actual toy stored values; no empirical use.
        proj=r.task.seeded_projection(768,32,20271005);projected=deepcopy(bundle)
        projected['projections']={'32':{'matrix':proj,'seed':20271005,'data_independent':True}}
        projected['learners']['e5']['projection_parameters']={'32':{'lambda_lock':r.task.engineering_parameter('lambda',.01,representation='e5:projection-32')}}
        g=groups['G_boundary/civil_comments/primary-10000/projection-32'];cell=r.dispatch.resolve(g,projected)
        sig=r._signature(g,projected,cell,kind='worker',method='O-G',path=paths['R000']);p={**profile,'signature':sig,'profile_id':r.digest(sig)}
        name,_=package(base,projected,p);got=r.replay_derived_package(p,{'input_package':name},projected,context,base_dir=base)
        check('projected_rows_replay_exact_seeded_FP32_transform',np.array_equal(got['arrays']['x'],cell['x']))
        for variant in ('encoder-e5-mpnet','encoder-mpnet-e5','fp32-state','zero-start-CG'):
            g=groups['G_boundary/civil_comments/primary-10000/'+variant];cell=r.dispatch.resolve(g,bundle)
            sig=r._signature(g,bundle,cell,kind='worker',method='P-I',path=paths['R000'])
            if variant=='fp32-state':check('fp32_does_not_inherit_FP64_audit',not sig['audit_requested'])
            elif variant=='zero-start-CG':check('cg_solver_not_retagged_cholesky',sig['solver']=='cg')
            else:check(variant+'_identities_distinct',sig['curator']['name']!=sig['learner']['name'])
        outside=deepcopy(bundle);outside['phase9_scope']['extension_recipes']=['negative_controls']
        with mocked(),patch.object(r.extensions,'extension_path',side_effect=AssertionError('outside branches must not resolve a wrong path')):
            out_manifest,_=r.compile_profiles(key,outside,base_dir=base)
        check('negative_controls_classified_before_wrong_path_resolution',all(row['disposition']=='outside_added_audit_and_verifier_caps' for row in out_manifest['jobs'] if '/negative_controls/' in row['job_id']))
        # Count formulas use metadata only, never an empirical witness.
        example=deepcopy(profile);example['signature']['audit_requested']=True;example['signature']['full_state']=True;example['signature']['method']='P-I';example['original_records']=10000
        one={**manifest,'profiles':[example]};obs={'profile_id':example['profile_id'],'records':10000,'observed_limits_fit':True,'pair_coordinates':38396160000,'symbol_coordinates':5000000,'observation_id':'metadata-only'}
        selected=r.select_policy(one,[obs])
        check('native_pair_cap_exact_formula',selected['execution_policy'][r.pf.PAIR_CAP]==38396160000)
        check('native_symbolic_count_stays_unknown',selected['coverage'][0]['future_symbolic_keys'] is None)
        check('no_runtime_or_RSS_forecast_in_count_selector',selected['runtime_forecast'] is None and selected['RSS_forecast'] is None)
        wrong={**obs,'records':9999};reject('smaller_observation_not_extrapolated',lambda:r.select_policy(one,[wrong]))
        wrong={**obs,'pair_coordinates':38396159999};reject('below_exact_pair_cap_refused',lambda:r.select_policy(one,[wrong]))
        wrong={**obs,'symbol_coordinates':3999999};reject('observed_coefficient_extent_cannot_lower_default',lambda:r.select_policy(one,[wrong]))
        cp=deepcopy(example);cp['signature'].update(kind='convex',audit_requested=False,learner_dimension=768,outputs=20);cp.update(maximum_selected_records=10000,selected_counts=[10000],utility_budget_required=True)
        reject('global_convex_cap_cannot_authorize_unlocked_utility',lambda:r.utility_budget_bindings({**manifest,'profiles':[cp]},[],bundle))
        stage={'exit_code':0,'timeout':False,'spawn_to_reap_seconds':1.,'user_cpu_seconds':.1,'system_cpu_seconds':.1,'peak_rss_bytes':1,'sampled_process_tree_peak_rss_sum_bytes':1}
        check('CPU_RSS_are_reported_indicators_not_bounds',not r._measurements(stage,manifest['base_policy'])['true_peak_or_address_space_bound'])
        reject('noncompletion_not_qualified',lambda:r._measurements({**stage,'exit_code':1},manifest['base_policy']))
        reject('boolean_CPU_rejected',lambda:r._measurements({**stage,'user_cpu_seconds':True},manifest['base_policy']))
      if source!=r.codes()|{'phase9/check_resource_profiles.py':r.sha(__file__)}:raise AssertionError('sources_changed_during_check')
      report.update(status='passed',checks=checks,passed=len(checks),failed=0)
    except Exception as error:
      report.update(checks=checks,passed=len(checks),failed=1,exception_type=type(error).__name__,reason=str(error))
    write(out/'checks.json',report);print(json.dumps({'status':report['status'],'checks':len(checks),'reason':report.get('reason')}));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();run(args.out)
