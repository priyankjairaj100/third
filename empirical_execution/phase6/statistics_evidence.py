"""Bind complete planned analyses to actual dispatcher outputs and receipts.

External provenance remains accountable testimony. File hashes only bind bytes.
"""
from pathlib import Path
import json
import numpy as np
from phase5 import statistics,task_program as task
from phase5.study_registry import digest
from phase4.execution import array_hash
from phase6 import recipes


def bound_file(descriptor,base):
    base=Path(base).resolve();p=(base/descriptor['path']).resolve()
    if not p.is_relative_to(base) or not p.is_file() or task.file_hash(p)!=descriptor['sha256']:
        raise ValueError('Missing, escaped or changed analysis evidence file')
    return p


def _json(descriptor,base):
    path=bound_file(descriptor,base);return path,json.loads(path.read_text())


def _read_outcome(descriptor,job_id,base):
    p,value=_json(descriptor,base)
    if isinstance(value,list):
        matches=[v for v in value if v.get('job_id')==job_id]
        if len(matches)!=1:raise ValueError('Ledger must contain exactly one bound job')
        value=matches[0]
    if value.get('job_id')!=job_id:raise ValueError('Outcome job identity differs')
    return p,value


def _thresholds(group,accepted):
    inputs=accepted['input_bindings']
    if inputs.get('task')!='multilabel':return None
    entry=inputs['learners'][group['configuration']['learner_encoder']]
    native=entry['train']['shape'][1];d=group['configuration']['learner_dimension']
    if d!=native:entry=entry['projection_parameters'][str(d)]
    lock=entry['tag_threshold_lock']
    return ['Infinity' if r['kind']=='never_positive' else float(r['threshold']) for r in lock['thresholds']]


def _numeric_thresholds(value):
    return None if value is None else np.asarray([np.inf if v=='Infinity' else float(v) for v in value],np.float64)


def _source_binding(accepted,primary):
    inputs=accepted['input_bindings'];expected={k:inputs.get(k) for k in ('evaluation_ids','evaluation_source_ids','evaluation_source_kinds')}
    if primary:
        rows=[r for r in accepted['checks'] if r['name']=='replayed_source_and_cache_acceptance' and r['passed']]
        if len(rows)!=1:raise ValueError('Accepted evaluation source replay required')
        replay=rows[0]['detail']['validated_bindings']
        for key in expected:
            if key not in replay:raise ValueError('Accepted replay omits '+key)
            expected[key]=replay[key]
    return expected


def _status(outcome):
    """Conservative saved-job status. No failed job becomes a success row."""
    s=outcome.get('status','missing')
    if s=='completed':return 'success'
    if 'timeout' in s:return 'timeout'
    if 'infeasible' in s:return 'infeasible'
    if 'missing' in s or 'blocked' in s or s=='missing':return 'missing'
    if s=='structural_zero_or_unavailable_arm':return 'skipped'
    return 'failed'


def _load_design(dossier,primary):
    if primary:registry,jobs=recipes.build_registry()
    else:registry,jobs=dossier['study_registry'],dossier['study_jobs']
    if registry.get('jobs_content_sha256')!=digest(jobs):raise ValueError('Study jobs do not match registry')
    groups={g['group_id']:g for g in registry['groups']};lookup={j['job_id']:j for j in jobs}
    if len(groups)!=len(registry['groups']) or len(lookup)!=len(jobs):raise ValueError('Duplicate study group or job')
    return registry,jobs,groups,lookup


def bind_statistics_evidence(dossier,context):
    base=context['base_dir'];primary=context['mode']==context['dispatch_module'].PRIMARY_MODE
    _,jobs,allgroups,joblookup=_load_design(dossier,primary)
    taskjobs={j['job_id']:j for j in jobs if j.get('group_id') in allgroups and allgroups[j['group_id']]['block']=='B_relevance'}
    if not taskjobs:raise ValueError('No planned primary relevance jobs')
    required_groups={j['group_id'] for j in taskjobs.values()}
    groups={};group_paths={};group_descriptors=dossier['groups']
    # Systems groups are checked here too, but do not enter relevance denominators.
    for gid,descriptors in group_descriptors.items():
        if gid not in allgroups:raise ValueError('Unplanned statistics group')
        group=allgroups[gid];_,accepted=_json(descriptors['acceptance'],base)
        if accepted.get('group_id')!=gid or not accepted.get('execution_allowed'):raise ValueError('Group inputs lack acceptance')
        if primary and not accepted.get('primary_inputs_accepted'):raise ValueError('Primary group input acceptance required')
        _,manifest=_json(descriptors['requests'],base)
        if manifest!=accepted['input_bindings']['requests']:raise ValueError('Request manifest differs from accepted input binding')
        if manifest['dataset_id']!=group['corpus'] or manifest['panel_id']!=group['panel']:raise ValueError('Request population differs')
        paths={p['trajectory_id']:p for p in manifest['trajectories']}
        if len(paths)!=len(manifest['trajectories']):raise ValueError('Duplicate frozen request path')
        groups[gid]=(group,accepted);group_paths[gid]=paths
    if not required_groups<=set(groups):raise ValueError('Missing complete relevance group evidence')
    outcomes={};outcome_descriptors={}
    for entry in dossier['job_outcomes']:
        jid=entry['job_id']
        if jid not in taskjobs or jid in outcomes:raise ValueError('Duplicate or unplanned relevance job outcome')
        p,outcome=_read_outcome(entry['outcome'],jid,base)
        if outcome.get('group_id')!=taskjobs[jid]['group_id']:raise ValueError('Outcome group differs from compiled job')
        outcomes[jid]=(p,outcome);outcome_descriptors[jid]=entry['outcome']
    if set(outcomes)!=set(taskjobs):raise ValueError('Every compiled relevance job needs an outcome, including matched diagnostics')
    expected={};diagnostic_jobs=[];empty_jobs=[]
    for jid,job in taskjobs.items():
        group,accepted=groups[job['group_id']];path=group_paths[job['group_id']][job['trajectory_id']]
        if job['arm']!=path['arm']:raise ValueError('Compiled arm differs from frozen path')
        if job['arm'] not in ('R','S','U','A'):
            diagnostic_jobs.append(jid);continue
        if not path['checkpoints']:empty_jobs.append(jid)
        for checkpoint in path['checkpoints']:
            key=(group['corpus'],group['panel'],digest(group['configuration']),job['arm'],job['trajectory_id'],checkpoint)
            if key in expected:raise ValueError('Duplicate derived analysis cell')
            expected[key]=jid
    statistics._verify(dossier['registry'])
    planned={statistics._key(r) for r in dossier['registry']['cells']}
    if planned!=set(expected):raise ValueError('Analysis registry omits or adds compiler-derived feasible cells')
    if primary:
        required_metrics={'predelete_mse','predelete_score_sd','frozen_mse','oracle_mse','signed_raw_loss','signed_normalized_loss','absolute_normalized_loss','prediction_rms','normalized_prediction_rms','loss_exceedance_001','prediction_exceedance_001','admissions','any_admission','admissions_per_deleted_record','admissions_per_deleted_source'}
        if any(a['input_bindings'].get('task')=='multilabel' for g,a in groups.values()):required_metrics|={'frozen_macro_f1','oracle_macro_f1','macro_f1_difference'}
        if set(dossier['metrics'])!=required_metrics or len(dossier['metrics'])!=len(required_metrics):raise ValueError('Complete fixed primary metric set required')
    records=dossier['records'];lookup={statistics._key(r):r for r in records}
    if len(lookup)!=len(records) or set(lookup)!=set(expected):raise ValueError('Every derived cell needs an explicit status row')
    bindings=dossier['record_bindings'];found=set();arrays={};evaluation={};receipts=[]
    for binding in bindings:
        key=statistics._key(binding['identity'])
        if key not in expected or key in found:raise ValueError('Unknown or duplicate analysis binding')
        found.add(key);record=lookup[key];jid=expected[key]
        if binding['job_id']!=jid or binding['outcome']!=outcome_descriptors[jid]:raise ValueError('Analysis identity relabeled from another job')
        p,outcome=outcomes[jid];job=taskjobs[jid];group,accepted=groups[job['group_id']]
        if record['status']!=_status(outcome):raise ValueError('Analysis status differs from saved job status')
        if record['status']=='success':
            if primary and not outcome.get('primary_output_accepted'):raise ValueError('Successful primary output lacks acceptance')
            identity=outcome['analysis_binding']
            expected_identity=dict(zip(statistics._FIELDS[:-1],key[:-1]))
            if any(identity.get(k)!=v for k,v in expected_identity.items()) or identity.get('group_id')!=group['group_id']:
                raise ValueError('Saved analysis identity differs from compiled job')
            if identity.get('configuration')!=group['configuration']:raise ValueError('Saved configuration changed')
            source=_source_binding(accepted,primary)
            if any(identity.get(k)!=v for k,v in source.items()):raise ValueError('Saved evaluation sources differ from accepted population')
            if primary and not identity.get('accepted_source_replay'):raise ValueError('Primary source replay absent')
            thresholds=_thresholds(group,accepted)
            if identity.get('thresholds')!=thresholds:raise ValueError('Saved decision thresholds differ from accepted calibration')
            if 'thresholds' in binding and not np.array_equal(_numeric_thresholds(binding['thresholds']),_numeric_thresholds(thresholds)):
                raise ValueError('Caller threshold override rejected')
            candidates=[m for m in outcome['methods'] if m['method']==binding['method']]
            if len(candidates)!=1 or candidates[0]['status']!='completed':raise ValueError('Analysis method did not complete')
            method=candidates[0];index=next(i for i,c in enumerate(method['head_checks']) if c['checkpoint']==key[5]);check=method['head_checks'][index]
            if not check.get('passed'):raise ValueError('Saved head failed its independent check')
            prediction=bound_file(binding['prediction'],base)
            if prediction!=(p.parent/method['artifact_directory']/f'prediction_{index:04d}.npz').resolve():raise ValueError('Wrong prediction artifact')
            with np.load(prediction,allow_pickle=False) as archive:values={k:archive[k].copy() for k in ('evaluation_y','before','frozen','oracle')}
            if array_hash(values['evaluation_y'])!=identity['evaluation_targets_sha256']:raise ValueError('Saved evaluation targets changed')
            if identity['evaluation_targets_sha256']!=accepted['input_bindings']['evaluation_y']['array_sha256']:raise ValueError('Evaluation targets differ from accepted labels')
            if len(values['evaluation_y'])!=len(source['evaluation_ids']):raise ValueError('Evaluation row order length changed')
            computed=statistics.prediction_metrics(values['evaluation_y'],values['before'],values['frozen'],values['oracle'],_numeric_thresholds(thresholds))
            actual_path=group_paths[job['group_id']][job['trajectory_id']]
            observation=next(o for o in actual_path['observations'] if o['checkpoint']==key[5])
            if check['admissions']!=observation['admissions']:raise ValueError('Saved admissions differ from accepted request replay')
            computed.update(statistics.admission_metrics(check['admissions'],observation['cumulative_deleted_records'],observation['cumulative_deleted_units'] if actual_path['unit']=='source' else None))
            if primary and set(record['metrics'])!=set(computed):raise ValueError('Successful primary row omits applicable metrics')
            for name,value in record['metrics'].items():
                if name not in computed or value!=computed[name]:raise ValueError('Metric differs from actual saved prediction: '+name)
            arrays[key]=values;evaluation[key]={**source,'thresholds':thresholds}
        elif record['metrics']:raise ValueError('Failed/missing rows must retain empty metrics')
        receipts.append({'key':list(key),'outcome_sha256':binding['outcome']['sha256'],'prediction_sha256':binding.get('prediction',{}).get('sha256')})
    if found!=set(expected):raise ValueError('A planned analysis cell lacks evidence binding')
    crossed_unavailable={}
    for name,inputs in dossier['crossed'].items():
        corpus,arm=name.split(':');keys=[k for k in expected if k[0]==corpus and k[3]==arm]
        if not keys:raise ValueError('Unplanned crossed analysis population')
        expected_pairs={(k[4],k[5]) for k in keys}
        supplied_pairs={(t,k) for t in inputs['trajectory_ids'] for k in inputs['checkpoint_ids']}
        if supplied_pairs!=expected_pairs or len(supplied_pairs)!=len(inputs['trajectory_ids'])*len(inputs['checkpoint_ids']):raise ValueError('Crossed analysis omits planned trajectories/checkpoints')
        if any(lookup[k]['status']!='success' for k in keys):
            crossed_unavailable[name]='planned_failed_or_missing_trajectory; unconditional_crossed_estimand_unavailable';continue
        match={(k[4],k[5]):k for k in keys};reference=evaluation[keys[0]]
        for k in keys:
            if evaluation[k]!=reference:raise ValueError('Crossed analysis mixes evaluation populations or thresholds')
        if inputs['test_source_ids']!=reference['evaluation_source_ids'] or inputs['source_kinds']!=reference['evaluation_source_kinds']:
            raise ValueError('Crossed source partition differs from accepted evaluation population')
        if not np.array_equal(_numeric_thresholds(inputs.get('thresholds')),_numeric_thresholds(reference['thresholds'])):raise ValueError('Crossed decision thresholds differ')
        for ti,tid in enumerate(inputs['trajectory_ids']):
            for ki,checkpoint in enumerate(inputs['checkpoint_ids']):
                value=arrays[match[tid,checkpoint]]
                if not np.array_equal(inputs['y'],value['evaluation_y']) or not np.array_equal(inputs['before'],value['before']):raise ValueError('Crossed baseline or labels differ')
                if not np.array_equal(inputs['frozen'][ti,ki],value['frozen']) or not np.array_equal(inputs['oracle'][ti,ki],value['oracle']):raise ValueError('Crossed saved predictions differ')
    systems_unavailable={};system_receipts=[];global_receipt_hashes=set()
    if set(dossier['systems'])!=set(dossier['system_bindings']):raise ValueError('Missing systems bindings')
    for corpus,arms in dossier['system_bindings'].items():
        expected_jobs={j['job_id']:j for j in jobs if j.get('group_id') in allgroups and allgroups[j['group_id']]['block']=='C_full_methods'
            and allgroups[j['group_id']]['corpus']==corpus and j.get('kind')=='timing_repeat' and j.get('arm') in ('R','S')}
        if not expected_jobs:raise ValueError('Unplanned systems corpus')
        found_jobs=set();seen_receipts=set();aggregate={'R':{},'S':{}};failed=False
        for arm in ('R','S'):
            for pair in arms[arm]:
                jid=pair['job_id']
                if jid not in expected_jobs or jid in found_jobs or expected_jobs[jid]['arm']!=arm:raise ValueError('Systems job duplicate or wrong arm/corpus')
                found_jobs.add(jid);job=expected_jobs[jid];gid=job['group_id']
                if gid not in groups:raise ValueError('Systems group acceptance missing')
                group,accepted=groups[gid];outcome_path,outcome=_read_outcome(pair['outcome'],jid,base)
                if outcome.get('group_id')!=gid:raise ValueError('Systems outcome group differs')
                times=[];services=[];methods={m['method']:m for m in outcome['methods']}
                complete=all(methods.get(m,{}).get('status')=='completed' for m in ('P-I','B-E'))
                if primary and not outcome.get('primary_output_accepted'):complete=False
                if not complete:
                    failed=True;aggregate[arm].setdefault(job['trajectory_id'],[]).append(None)
                    system_receipts.append({'job_id':jid,'status':'incomplete_pair','outcome_sha256':pair['outcome']['sha256']});continue
                for role,method in (('proposed','P-I'),('baseline','B-E')):
                    receipt_path,receipt=_json(pair[role],base)
                    expected_path=(outcome_path.parent/methods[method]['artifact_directory']/'persistence_receipt.json').resolve()
                    if receipt_path!=expected_path or receipt_path in seen_receipts or pair[role]['sha256'] in global_receipt_hashes:raise ValueError('Receipt reused or assigned to wrong method/job')
                    global_receipt_hashes.add(pair[role]['sha256'])
                    seen_receipts.add(receipt_path);service_path=receipt_path.with_name('service_report.json');service=json.loads(service_path.read_text())
                    if task.file_hash(service_path)!=receipt['service_report_sha256'] or service.get('method')!=method or not service.get('success'):
                        raise ValueError('Receipt does not bind a successful named service')
                    head_checks=methods[method].get('head_checks',[])
                    expected_checkpoints=group_paths[gid][job['trajectory_id']]['checkpoints']
                    if [c.get('checkpoint') for c in head_checks]!=expected_checkpoints or not head_checks or any(not c.get('passed') for c in head_checks):raise ValueError('System head audit is missing, incomplete, or failed')
                    if service['policy']!=context['policy']:raise ValueError('Systems resource policy differs from common lock')
                    if primary and accepted['input_bindings']['execution_policy']!=service['policy']:raise ValueError('Systems policy differs from accepted inputs')
                    duration=receipt['charged_total_seconds']
                    if isinstance(duration,bool) or not isinstance(duration,(float,int)) or not np.isfinite(duration) or duration<0:raise ValueError('Invalid charged lifecycle time')
                    times.append(float(duration));services.append(service)
                for field in ('raw_input_files_sha256','request_sha256','policy','unit','solver','panel','persistence'):
                    if services[0].get(field)!=services[1].get(field):raise ValueError('Systems pair has different '+field)
                if services[0]['panel']!='cold_process' or services[0]['persistence']!='final':raise ValueError('Primary systems contrast requires cold/final panel')
                request_path=group_paths[gid][job['trajectory_id']];batches=[];last=0
                for k in request_path['checkpoints']:batches.append(request_path['deletion_order'][last:k]);last=k
                import hashlib
                expected_request=hashlib.sha256(json.dumps(batches,separators=(',',':')).encode()).hexdigest()
                if services[0]['request_sha256']!=expected_request:raise ValueError('Systems request differs from accepted trajectory')
                aggregate[arm].setdefault(job['trajectory_id'],[]).append(times[0]-times[1])
                system_receipts.append({'job_id':jid,'repeat':job['repeat'],'proposed_sha256':pair['proposed']['sha256'],'baseline_sha256':pair['baseline']['sha256']})
        if found_jobs!=set(expected_jobs):raise ValueError('Systems bindings omit planned timing repetitions')
        expected_values={}
        for arm in ('R','S'):
            expected_values[arm.lower()+'_differences']=[None if any(v is None for v in aggregate[arm][tid]) else float(np.mean(aggregate[arm][tid])) for tid in sorted(aggregate[arm])]
        if expected_values!=dossier['systems'][corpus]:raise ValueError('Systems inputs must average paired repeats within each sorted trajectory')
        if failed:systems_unavailable[corpus]='planned_failed_or_missing_timing_pair; unconditional_systems_estimand_unavailable'
    return {'record_receipts':receipts,'systems_receipts':system_receipts,'derived_analysis_cells':len(expected),
        'bound_relevance_jobs':len(outcomes),'matched_diagnostic_jobs_outside_primary_bootstrap':diagnostic_jobs,
        'empty_request_jobs':empty_jobs,'crossed_unavailable':crossed_unavailable,'systems_unavailable':systems_unavailable,
        'actual_saved_predictions_recomputed':True,'timing_repetitions_averaged_within_trajectory':True,
        'source_authenticity_inferred_from_hashes':False}
