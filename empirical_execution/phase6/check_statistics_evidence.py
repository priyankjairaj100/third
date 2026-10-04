"""Software binding checks around saved natural-text predictions. No new study."""
from pathlib import Path
import copy,json,sys,tempfile,shutil,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase5 import statistics
from phase5.study_registry import digest
from phase4.execution import array_hash
from phase6 import dispatch
from phase6.statistics_evidence import bind_statistics_evidence
from phase5.run_isolated_v2 import DEFAULT_POLICY

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def descriptor(p,base):return {'path':str(p.relative_to(base)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def build_fixture(destination,source=None):
    """Metadata fixture around unchanged saved Civil predictions; never primary."""
    base=Path(destination).resolve();base.mkdir(parents=True,exist_ok=False)
    source=Path(source or ROOT/'phase6/results/extensions_candidate1/route_09')
    original=json.loads((source/'outcome.json').read_text());accepted=json.loads((source/'acceptance.json').read_text())
    gid='B_relevance/civil_comments/reused-40/native';jid=gid+'/R000';configuration={'learner_encoder':'lexical16','learner_dimension':16}
    group={'group_id':gid,'block':'B_relevance','corpus':'civil_comments','panel':'reused-40','variant':'native','configuration':configuration}
    job={'job_id':jid,'group_id':gid,'arm':'R','trajectory_id':'R000'}
    accepted['group_id']=gid;accepted['software_metadata_fixture']=True
    inputs=accepted['input_bindings'];inputs['evaluation_source_ids']=['unknown:'+r for r in inputs['evaluation_ids']]
    inputs['evaluation_source_kinds']={s:'unknown_singleton' for s in inputs['evaluation_source_ids']}
    manifest=inputs['requests'];path=next(p for p in manifest['trajectories'] if p['trajectory_id']=='R000')
    outcome={'job_id':jid,'group_id':gid,'status':'completed','primary_output_accepted':False,'methods':copy.deepcopy(original['methods']),
        'software_metadata_fixture':True}
    for m in outcome['methods']:
        target=base/'job'/m['artifact_directory'];target.mkdir(parents=True,exist_ok=True)
        for f in (source/m['artifact_directory']).glob('prediction_*.npz'):shutil.copyfile(f,target/f.name)
    outcome['analysis_binding']={'group_id':gid,'dataset_id':'civil_comments','panel_id':'reused-40',
        'configuration_id':digest(configuration),'configuration':configuration,'arm':'R','trajectory_id':'R000',
        'evaluation_ids':inputs['evaluation_ids'],'evaluation_source_ids':inputs['evaluation_source_ids'],
        'evaluation_source_kinds':inputs['evaluation_source_kinds'],'thresholds':None,
        'evaluation_targets_sha256':inputs['evaluation_y']['array_sha256'],'accepted_source_replay':False}
    write(base/'acceptance.json',accepted);write(base/'requests.json',manifest);write(base/'job/outcome.json',outcome)
    cells=[];records=[];bindings=[];predictions=[]
    for i,k in enumerate(path['checkpoints']):
        identity={'dataset_id':'civil_comments','panel_id':'reused-40','configuration_id':digest(configuration),'arm':'R','trajectory_id':'R000','checkpoint':k}
        p=base/'job/method_00'/f'prediction_{i:04d}.npz'
        with np.load(p) as z:values={n:z[n].copy() for n in ('evaluation_y','before','frozen','oracle')}
        predictions.append(values);metrics=statistics.prediction_metrics(values['evaluation_y'],values['before'],values['frozen'],values['oracle'])
        cells.append(identity);records.append({**identity,'status':'success','metrics':{'signed_normalized_loss':metrics['signed_normalized_loss']}})
        bindings.append({'identity':identity,'job_id':jid,'outcome':descriptor(base/'job/outcome.json',base),'method':'B-F','prediction':descriptor(p,base)})
    groups={gid:{'acceptance':descriptor(base/'acceptance.json',base),'requests':descriptor(base/'requests.json',base)}}
    jobs=[job];study={'groups':[group],'jobs_content_sha256':digest(jobs)}
    crossed={'trajectory_ids':['R000'],'checkpoint_ids':path['checkpoints'],'y':predictions[0]['evaluation_y'],'before':predictions[0]['before'],
        'frozen':np.stack([v['frozen'] for v in predictions])[None], 'oracle':np.stack([v['oracle'] for v in predictions])[None],
        'test_source_ids':inputs['evaluation_source_ids'],'source_kinds':inputs['evaluation_source_kinds']}
    dossier={'evidence_role':'natural_text_engineering','software_metadata_fixture':True,'study_registry':study,'study_jobs':jobs,
        'groups':groups,'job_outcomes':[{'job_id':jid,'outcome':descriptor(base/'job/outcome.json',base)}],
        'registry':statistics.lock_registry(cells),'records':records,'record_bindings':bindings,'crossed':{'civil_comments:R':crossed},
        'systems':{},'system_bindings':{},'metrics':['signed_normalized_loss'],
        'valid_p_values':{'signed_loss':{},'equal_RS_lifecycle':{}}}
    context={'base_dir':base,'mode':dispatch.ENGINEERING_MODE,'dispatch_module':dispatch,'policy':DEFAULT_POLICY}
    return dossier,context

def add_system_fixture(dossier,context):
    """Frame two existing measured pairs as a temporary schema test only."""
    base=Path(context['base_dir']);d=copy.deepcopy(dossier)
    gid='C_full_methods/civil_comments/systems-schema/native'
    config={'learner_encoder':'lexical16','learner_dimension':16}
    group={'group_id':gid,'block':'C_full_methods','corpus':'civil_comments','panel':'systems-schema','variant':'native','configuration':config}
    accept=json.loads((base/'acceptance.json').read_text());accept['group_id']=gid
    manifest=copy.deepcopy(accept['input_bindings']['requests']);manifest['panel_id']='systems-schema'
    pairs=[];differences=[]
    for repeat,release in enumerate(('worker_engineering_final','worker_engineering_release_v2')):
        source=ROOT/'phase6/results'/release;verify=json.loads((source/'verification.json').read_text())
        case_by_method={m:next(c for c in verify['cases'] if c['method']==m and c['dimension']==16 and c['solver']=='cholesky' and c['panel']=='cold_process') for m in ('P-I','B-E')}
        jid=gid+'/R000/fresh-process-'+str(repeat);job={'job_id':jid,'group_id':gid,'arm':'R','trajectory_id':'R000','kind':'timing_repeat','repeat':repeat}
        methods=[];pair={'job_id':jid};times=[]
        for number,(role,method) in enumerate((('proposed','P-I'),('baseline','B-E'))):
            original=source/case_by_method[method]['case'];target=base/f'systems/repeat_{repeat}/method_{number:02d}';target.mkdir(parents=True)
            for name in ('service_report.json','persistence_receipt.json'):shutil.copyfile(original/name,target/name)
            conf=json.loads((original/'prospective_configuration.json').read_text());order=[r for batch in conf['requests'] for r in batch]
            checkpoints=np.cumsum([len(v) for v in conf['requests']]).tolist()
            methods.append({'method':method,'status':'completed','artifact_directory':target.name,
                'head_checks':[{'checkpoint':k,'passed':case_by_method[method]['passed']} for k in checkpoints]})
            pair[role]=descriptor(target/'persistence_receipt.json',base)
            times.append(json.loads((target/'persistence_receipt.json').read_text())['charged_total_seconds'])
        outcome={'job_id':jid,'group_id':gid,'status':'completed','primary_output_accepted':False,'methods':methods,'software_metadata_fixture':True}
        op=base/f'systems/repeat_{repeat}/outcome.json';write(op,outcome);pair['outcome']=descriptor(op,base);pairs.append(pair);differences.append(times[0]-times[1])
        d['study_jobs'].append(job)
    manifest['trajectories']=[{'trajectory_id':'R000','arm':'R','unit':'record','deletion_order':order,'checkpoints':checkpoints}]
    accept['input_bindings']['requests']=manifest
    write(base/'systems/acceptance.json',accept);write(base/'systems/requests.json',manifest)
    d['groups'][gid]={'acceptance':descriptor(base/'systems/acceptance.json',base),'requests':descriptor(base/'systems/requests.json',base)}
    d['study_registry']['groups'].append(group);d['study_registry']['jobs_content_sha256']=digest(d['study_jobs'])
    d['system_bindings']={'civil_comments':{'R':pairs,'S':[]}}
    d['systems']={'civil_comments':{'r_differences':[float(np.mean(differences))],'s_differences':[]}}
    return d

def run(destination):
    output=Path(destination)
    if output.exists():raise FileExistsError('Preserve previous evidence')
    checks=[]
    with tempfile.TemporaryDirectory(prefix='statistics-evidence-',dir=ROOT.parent/'tmp') as td:
        d,c=build_fixture(Path(td)/'fixture');base=Path(c['base_dir'])
        result=bind_statistics_evidence(d,c);checks.append({'check':'bound_saved_predictions_pass','passed':result['derived_analysis_cells']==4})
        def refusal(name,change):
            candidate=copy.deepcopy(d);change(candidate)
            try:bind_statistics_evidence(candidate,c)
            except (ValueError,KeyError,StopIteration):checks.append({'check':name,'passed':True});return
            raise AssertionError('Accepted '+name)
        refusal('relabelled_identity_rejected',lambda a:a['record_bindings'][0]['identity'].update(dataset_id='wrong'))
        refusal('wrong_job_rejected',lambda a:a['record_bindings'][0].update(job_id='wrong'))
        refusal('omitted_record_rejected',lambda a:a['records'].pop())
        refusal('omitted_binding_rejected',lambda a:a['record_bindings'].pop())
        refusal('omitted_compiled_outcome_rejected',lambda a:a['job_outcomes'].clear())
        refusal('omitted_planned_cell_rejected',lambda a:a.update(registry=statistics.lock_registry(a['registry']['cells'][:-1])))
        refusal('source_partition_override_rejected',lambda a:a['crossed']['civil_comments:R']['test_source_ids'].__setitem__(0,'wrong'))
        refusal('source_kind_override_rejected',lambda a:a['crossed']['civil_comments:R']['source_kinds'].update({'wrong':'genuine_native'}))
        refusal('decision_threshold_override_rejected',lambda a:a['crossed']['civil_comments:R'].update(thresholds=[.5]))
        refusal('prediction_mutation_rejected',lambda a:a['crossed']['civil_comments:R']['oracle'].__setitem__((0,0,0,0),99.))
        refusal('unbound_success_status_rejected',lambda a:a['records'][0].update(status='failed',metrics={}))
        # Change only a metadata fixture. The underlying natural predictions stay unchanged.
        op=base/'job/outcome.json';original=json.loads(op.read_text());failed=copy.deepcopy(original);failed['status']='method_noncompletion'
        write(op,failed);fd=copy.deepcopy(d);desc=descriptor(op,base);fd['job_outcomes'][0]['outcome']=desc
        for r,b in zip(fd['records'],fd['record_bindings']):r.update(status='failed',metrics={});b['outcome']=desc
        saved=bind_statistics_evidence(fd,c)
        checks.append({'check':'failed_rows_preserve_denominator_without_output_acceptance','passed':saved['derived_analysis_cells']==4 and 'civil_comments:R' in saved['crossed_unavailable']})
        write(op,original)
        # A receipt cannot enter the analysis without its compiled timing job.
        refusal('unplanned_system_receipt_rejected',lambda a:a.update(systems={'civil_comments':{'r_differences':[0.],'s_differences':[0.]}},
            system_bindings={'civil_comments':{'R':[{'job_id':'unknown'}],'S':[]}}))
        sd=add_system_fixture(d,c);positive=bind_statistics_evidence(sd,c)
        checks.append({'check':'paired_receipts_and_repeat_averaging_pass','passed':positive['timing_repetitions_averaged_within_trajectory'] and len(positive['systems_receipts'])==2})
        def system_refusal(name,mutate):
            candidate=copy.deepcopy(sd);mutate(candidate)
            try:bind_statistics_evidence(candidate,c)
            except (ValueError,KeyError):checks.append({'check':name,'passed':True});return
            raise AssertionError('Accepted '+name)
        system_refusal('wrong_method_receipt_rejected',lambda a:a['system_bindings']['civil_comments']['R'][0].update(proposed=a['system_bindings']['civil_comments']['R'][0]['baseline']))
        system_refusal('reused_receipt_rejected',lambda a:a['system_bindings']['civil_comments']['R'][1].update(proposed=a['system_bindings']['civil_comments']['R'][0]['proposed']))
        system_refusal('omitted_timing_repeat_rejected',lambda a:a['system_bindings']['civil_comments']['R'].pop())
        system_refusal('wrong_repeat_mean_rejected',lambda a:a['systems']['civil_comments']['r_differences'].__setitem__(0,999.))
        # Temporary schema mutations never modify the preserved service files.
        pair=sd['system_bindings']['civil_comments']['R'][0];receipt_path=base/pair['proposed']['path'];service_path=receipt_path.with_name('service_report.json')
        old_service=service_path.read_bytes();old_receipt=receipt_path.read_bytes()
        changed_service=json.loads(old_service);changed_service['request_sha256']='0'*64;write(service_path,changed_service)
        changed_receipt=json.loads(old_receipt);changed_receipt['service_report_sha256']=hashlib.sha256(service_path.read_bytes()).hexdigest();write(receipt_path,changed_receipt)
        candidate=copy.deepcopy(sd);candidate['system_bindings']['civil_comments']['R'][0]['proposed']=descriptor(receipt_path,base)
        try:bind_statistics_evidence(candidate,c)
        except ValueError:checks.append({'check':'mismatched_saved_requests_rejected','passed':True})
        else:raise AssertionError('Accepted different method requests')
        service_path.write_bytes(old_service);receipt_path.write_bytes(old_receipt)
        outcome_path=base/pair['outcome']['path'];old_outcome=outcome_path.read_bytes();failed_outcome=json.loads(old_outcome);failed_outcome['status']='method_noncompletion';failed_outcome['methods'][0]['status']='worker_failure';write(outcome_path,failed_outcome)
        candidate=copy.deepcopy(sd);candidate['system_bindings']['civil_comments']['R'][0]['outcome']=descriptor(outcome_path,base)
        candidate['systems']['civil_comments']['r_differences']=[None]
        failed_result=bind_statistics_evidence(candidate,c)
        checks.append({'check':'failed_pair_keeps_trajectory_unavailable','passed':'civil_comments' in failed_result['systems_unavailable']})
        outcome_path.write_bytes(old_outcome)
        # Bind targets to accepted labels, not merely an internally consistent NPZ.
        ap=base/'acceptance.json';aa=json.loads(ap.read_text());aa['input_bindings']['evaluation_y']['array_sha256']='0'*64;write(ap,aa)
        changed=copy.deepcopy(d);changed['groups'][next(iter(changed['groups']))]['acceptance']=descriptor(ap,base)
        try:bind_statistics_evidence(changed,c)
        except ValueError:checks.append({'check':'accepted_label_hash_override_rejected','passed':True})
        else:raise AssertionError('Accepted wrong label hash')
    report={'schema':'ccu-statistics-evidence-checks-1','scope':'software_metadata_fixtures_around_unchanged_saved_Civil_predictions',
        'primary_evidence_created':False,'worker_benchmarks_rerun':False,'check_count':len(checks),'checks':checks,'all_passed':all(v['passed'] for v in checks),
        'source_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ('statistics_evidence.py','check_statistics_evidence.py')}}
    write(output,report);print(json.dumps({k:report[k] for k in ('check_count','all_passed')}));return report
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
