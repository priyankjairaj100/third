"""Stage authentic local inputs into frozen-dispatcher dossiers.

This assembler never creates reviews, human responses, source identities, or
labels. Candidate construction and final external review are separate stages.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
from phase3 import calibration as cal, panels, run_preparation as prep
from phase3.reference_graph import build_reference_graph
from phase4 import calibrate_cache as cc, embeddings as emb, model_selection as ms, news_calendar, requests
from phase5 import task_program as task, replication_encoding
from phase6 import acceptance, dispatch, recipes, logistic_selection
from phase7.prepare_calibration import strict_json, file_hash

DESIGN = ROOT.parent/'output/empirical_program/study_design.json'
SCHEMA = 'ccu-phase8-dossier-stage-1'


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def load_value(path, *, base=None):
    """Strict, hash-bound JSON/JSONL/NPY descriptors rooted in one directory."""
    path = Path(path).resolve(); base = Path(base).resolve() if base else path.parent
    def visit(value, active):
        if isinstance(value, dict) and set(value) == {'path','sha256','kind'}:
            if not isinstance(value['path'], str): raise ValueError('Descriptor path must be text')
            target = (base/value['path']).resolve()
            if not target.is_relative_to(base) or not target.is_file(): raise ValueError('Bound file outside root or absent')
            if target in active: raise ValueError('Cyclic bound reference')
            before = file_hash(target)
            if before != value['sha256']: raise ValueError('Bound file checksum mismatch')
            if value['kind'] == 'npy': result = np.load(target, mmap_mode='r', allow_pickle=False)
            elif value['kind'] == 'json': result = visit(strict_json(target.read_text()), active+(target,))
            elif value['kind'] == 'jsonl': result = [visit(strict_json(line),active+(target,)) for line in target.read_text().splitlines() if line.strip()]
            else: raise ValueError('Unsupported bound file kind')
            if file_hash(target) != before: raise ValueError('Bound file changed while loading')
            return result
        if isinstance(value, dict): return {k:visit(v,active) for k,v in value.items()}
        if isinstance(value, list): return [visit(v,active) for v in value]
        return value
    before = file_hash(path); result = visit(strict_json(path.read_text()),(path,))
    if file_hash(path) != before: raise ValueError('JSON input changed while loading')
    return result


def dump_value(value, path):
    """Write a frozen-loader-compatible descriptor tree; deduplicate array bytes."""
    path = Path(path); root = path.parent; (root/'artifacts').mkdir(exist_ok=True)
    def save(item, key=''):
        if isinstance(item,np.ndarray):
            name = task.fingerprint(item)['array_sha256']+'.npy'; target = root/'artifacts'/name
            if not target.exists():
                with target.open('xb') as stream: np.save(stream,item,allow_pickle=False)
            stored = np.load(target,mmap_mode='r',allow_pickle=False)
            if task.fingerprint(stored) != task.fingerprint(item): raise ValueError('Existing array artifact differs')
            return {'path':'artifacts/'+name,'kind':'npy','sha256':file_hash(target)}
        if isinstance(item,dict): return {k:save(v,k) for k,v in item.items()}
        if isinstance(item,list):
            if key in {'records','train_records','evaluation_records'}:
                name = cal.digest(item)+'.json'; target=root/'artifacts'/name
                if not target.exists(): write(target,item)
                if strict_json(target.read_text()) != item: raise ValueError('Existing record artifact differs')
                return {'path':'artifacts/'+name,'kind':'json','sha256':file_hash(target)}
            return [save(v) for v in item]
        return item
    write(path,save(value))
    if task.fingerprint(load_value(path)) != task.fingerprint(value): raise AssertionError('Serialized dossier changed its values')
    return {'path':path.name,'sha256':file_hash(path),'kind':'json'}


def codes():
    result = dispatch.codes()
    for module in (sys.modules[__name__],sys.modules[strict_json.__module__]):
        result[str(Path(module.__file__).resolve().relative_to(ROOT))] = file_hash(module.__file__)
    for name in ('archive_export.py','resource_policy.py'):
        path=ROOT/'phase8'/name
        if path.exists(): result['phase8/'+name]=file_hash(path)
    return result


def _absolute(value, base):
    p=Path(value); return str((p if p.is_absolute() else Path(base)/p).resolve())


def source_spec(path):
    """Canonical absolute source paths remain stable through later review stages."""
    path=Path(path).resolve(); spec=load_value(path); base=path.parent
    if not isinstance(spec,dict): raise ValueError('Source specification must be an object')
    for parent,names in ((spec.get('adapter',{}),('directory',)),(spec.get('prepared',{}),('records','audit')),
                         (spec.get('panel',{}),('directory',)),(spec.get('archive_export',{}),('directory','archive_path'))):
        for name in names:
            if name in parent: parent[name]=_absolute(parent[name],base)
    for key,value in spec.get('adapter',{}).get('inputs',{}).items(): spec['adapter']['inputs'][key]=_absolute(value,base)
    kw=spec.get('adapter',{}).get('kwargs',{})
    if 'psl_path' in kw: kw['psl_path']=_absolute(kw['psl_path'],base)
    for cache in spec.get('caches',{}).values():
        for key in ('directory','assets','model_dir','tokenizer_dir'):
            if key in cache: cache[key]=_absolute(cache[key],base)
    if 'coverage' in spec: spec['coverage']=_absolute(spec['coverage'],base)
    for item in spec.get('external_evidence',[]): item['path']=_absolute(item['path'],base)
    return spec


def archive_replay(spec, out):
    """Add original archive→export replay, beyond frozen export→cache replay."""
    if spec['dataset_id'] not in {'civil_comments','cc_news'}:
        return {'required':False,'scope':'Original supplied Stack/WCEP parser inputs are replayed by frozen acceptance.'}
    from phase8 import archive_export
    export=spec['archive_export']; directory=Path(export['directory'])
    if Path(spec['adapter']['inputs']['records']).resolve() != (directory/'records.jsonl').resolve():
        raise ValueError('Adapter input must be the replayed archive export records')
    manifest=strict_json((directory/'manifest.json').read_text())
    if manifest.get('dataset_id') != spec['dataset_id']: raise ValueError('Archive export dataset differs')
    expected=manifest['adapter']; kwargs=spec['adapter'].get('kwargs',{})
    for key in ('input_format','input_schema'):
        if kwargs.get(key) != expected[key]: raise ValueError('Adapter parser contract differs from archive exporter')
    report=archive_export.replay_export(directory,export['archive_path'],out)
    if report.get('machine_acceptance_passed') is not True: raise ValueError('Original archive/export replay failed')
    return {'required':True,'manifest_sha256':file_hash(directory/'manifest.json'),
            'export_records_sha256':file_hash(directory/'records.jsonl'),
            'replay_binding_sha256':report['replay_binding_sha256'],
            'exporter_code_sha256':file_hash(archive_export.__file__),
            'source_rights_or_meaning_authenticated':False}


def calibration_values(spec, encoder):
    cache=spec['caches'][encoder]; directory=Path(cache['directory']); prepared=spec['prepared']
    rows,features,provenance=cc.load_aligned_cache(prepared['records'],prepared['audit'],directory/'vectors.npy',
        directory/'cache_manifest.json',cache.get('assets',directory/'assets.json'))
    if provenance['dataset_id'] != spec['dataset_id'] or provenance['encoder_id'] != acceptance.ENCODERS[encoder]:
        raise ValueError('Calibration corpus/encoder mismatch')
    return {'source_acceptance':spec,'records':rows,'features':features,'provenance':provenance}


def _stage(out, kind, action):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    (out/'.gitignore').write_text('*\n')
    report={'schema':SCHEMA,'stage':kind,'status':'blocked','primary_execution_allowed':False,
            'human_responses_created':0,'external_testimony_created':False,'code_sha256':codes(),
            'design_sha256':file_hash(DESIGN)}
    try:
        result=action(out,report)
        if codes()!=report['code_sha256'] or file_hash(DESIGN)!=report['design_sha256']:
            raise ValueError('Assembler/dependencies/design changed during stage')
        commit=result.pop('_commit',None)
        if commit is not None: result.update(commit())
        report.update(result)
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,ImportError) as error:
        report['failure']={'type':type(error).__name__,'message':str(error)}
        report['partial_outputs_are_not_accepted']=True
    report['persisted_stage_directory_bytes_before_receipt']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    report['memory_accounting_scope']={
        'measured_peak_memory_bytes':None,'constant_total_memory_claim':False,
        'shared_preparation_not_repair_measurement':True,
        'resident_inputs':'Complete prepared/guarded row metadata, calibration gathers, selected train/evaluation gathers, graph/requests, and acceptance replay workspaces.',
        'cache_access':'Full NPY files are mapped; selected row and projection gathers allocate new arrays. OS page cache and encoder memory remain additional.'}
    write(out/'receipt.json',report); return report


def build_calibration(source_path, encoder, out):
    def action(out,report):
        if encoder not in acceptance.ENCODERS: raise ValueError('Registered E5/MPNet encoder required')
        spec=source_spec(source_path)
        if spec['dataset_id']=='wcep100': raise ValueError('WCEP inherits News and cannot select its own threshold')
        report['archive_replay']=archive_replay(spec,out/'archive_replay')
        dossier=calibration_values(spec,encoder)
        replay=acceptance.accept_calibration_inputs(spec['dataset_id'],encoder,dossier,
            base_dir=out,design=strict_json(DESIGN.read_text()))
        write(out/'source_cache_replay.json',replay)
        if replay.get('machine_acceptance_passed') is not True: raise ValueError('Prethreshold source/cache acceptance failed')
        bound=dump_value(dossier,out/'dossier.json')
        return {'status':'calibration_dossier_ready_external_review_pending','dossier':bound,
                'threshold_or_human_responses_required':False,'source_cache_replay_binding':replay['replay_binding_sha256']}
    return _stage(out,'calibration',action)


def completed_curator(spec, encoder, evidence, base):
    """Recompute supplied human files; do not generate responses or declarations."""
    dossier=calibration_values(spec,encoder)
    fields=('selection_manifest','selection_responses','validation_manifest','validation_responses')
    for key in fields: dossier[key]=load_value(Path(base)/evidence[key])
    selected=cal.select_threshold(dossier['selection_manifest'],dossier['selection_responses'])
    quality=cal.quality_gate(dossier['validation_manifest'],dossier['validation_responses'],selected)
    dossier['quality_report']=quality
    entry={'encoder_id':dossier['provenance']['encoder_id'],'encoder_revision':dossier['provenance']['encoder_revision'],
           'threshold_lock':selected,'calibration':{k:v for k,v in dossier.items() if k!='source_acceptance'}}
    dispatch.recompute_calibration(entry,encoder)
    if 'sensitivity_quality' in evidence:
        entry['sensitivity_quality']=load_value(Path(base)/evidence['sensitivity_quality'])
        for sign in (-1,1):
            tau=max(0.,selected['threshold']-.02) if sign<0 else min(1.,selected['threshold']+.02)
            gate=task.sensitivity_quality(entry,tau)
            if not gate.get('quality_pass'): raise ValueError('Supplied threshold-sensitivity quality did not pass')
    return entry


def selected_groups(group_ids):
    registry,jobs=recipes.build_registry(); lookup={g['group_id']:g for g in registry['groups']}
    if not group_ids or len(set(group_ids))!=len(group_ids): raise ValueError('Unique explicit registered group IDs required')
    groups=[lookup[g] for g in group_ids]
    if len({g['request_family'] for g in groups})!=1: raise ValueError('One stable request family per candidate')
    return groups,registry


def panel_population(group,spec,curators,design,out):
    contract=acceptance.panel_contract(group,spec)
    if spec['dataset_id']=='wcep100':
        rows,panel,_=replication_encoding.load_wcep_panel(spec['adapter']['directory'],spec['panel']['directory'])
        if panel['target']!=contract['target'] or panel['shortfall']: raise ValueError('WCEP whole-event target unmet')
        return rows,[r['record_id'] for r in rows],[],{'whole_event_panel_sha256':panel['sha256']}
    e5=curators['e5']; cache=spec['caches']['e5']; directory=Path(cache['directory'])
    write(out/'e5_threshold.json',e5['threshold_lock']); write(out/'e5_quality.json',e5['calibration']['quality_report'])
    guard=out/'guarded'
    cc.guard_population(spec['prepared']['records'],spec['prepared']['audit'],directory/'vectors.npy',
        directory/'cache_manifest.json',cache.get('assets',directory/'assets.json'),out/'e5_threshold.json',out/'e5_quality.json',guard)
    retained=prep.read_rows(guard/'records.jsonl'); report={}
    if contract['calendar']:
        coverage=news_calendar.load_coverage(spec['coverage'],file_hash(spec['adapter']['inputs']['records']))
        counts=Counter(r['original_fields']['date'][:7] for r in retained if r['partition']=='train')
        decision=news_calendar.calendar_decision(dict(counts),coverage,target=contract['target'])
        if not decision['target_met']: raise ValueError('Complete News calendar target unmet')
        ids=[r['record_id'] for r in retained if r['partition']=='train' and r['original_fields']['date'][:7] in decision['selected_months']]
        panel_ids={'calendar':ids}; test=[]; report['calendar']=decision
    elif contract['pool']=='separate_refit':
        panel_ids,report=acceptance.separate_refit_panel(retained,design,contract)
        if report['shortfall']: raise ValueError('Disjoint refit whole-source pool target unmet')
        ids,test=panel_ids['separate_refit'],panel_ids['test']
    else:
        panel_ids,report=panels.select_panels(retained,design,target=contract['target'])
        if report[contract['pool']]['shortfall']: raise ValueError('Registered whole-source panel target unmet')
        ids,test=panel_ids[contract['pool']],panel_ids['test']
    write(out/'panel_ids.json',panel_ids)
    spec['guarded']={'records':str(guard/'records.jsonl'),'audit':str(guard/'audit.json'),
        'panel_ids':str(out/'panel_ids.json'),'target':contract['target'],'pool':contract['pool'],
        'selection_lock':str(out/'e5_threshold.json'),'quality_report':str(out/'e5_quality.json'),'block_size':256}
    return retained,ids,test,report


def _learner_parameters(features, records, provenance, vocabulary, encoder, revision, logistic=False):
    dossier={'features':features,'records':records,'provenance':provenance}
    if vocabulary is not None: dossier['vocabulary_lock']=vocabulary
    lock=ms.select_ridge_regularization(features,records,provenance,vocabulary_lock=vocabulary)
    entry={'encoder_id':encoder,'encoder_revision':revision,'calibration':dossier,'lambda_lock':lock}
    if vocabulary is not None:
        entry['tag_threshold_lock']=ms.select_tag_decision_thresholds(lock)
        if logistic:
            entry['logistic_threshold_lock']=logistic_selection.select_logistic_decision_thresholds(
                features,records,provenance,lock,vocabulary)
    return entry


def assemble_bundle(groups,registry,spec,rows,train_ids,evaluation_ids,curators,caches,design):
    """Pure assembly after population selection. All targets derive from original rows."""
    group=groups[0]; dataset=spec['dataset_id']; lookup={r['record_id']:r for r in rows}
    train=[lookup[r] for r in train_ids]; evaluation=[lookup[r] for r in evaluation_ids]
    if len(train_ids)!=len(set(train_ids)) or set(train_ids)&set(evaluation_ids): raise ValueError('Panel IDs overlap or repeat')
    positions={r['record_id']:i for i,r in enumerate(rows)}
    ti=[positions[r] for r in train_ids]; ei=[positions[r] for r in evaluation_ids]
    bundle={'dataset_id':dataset,'train_ids':train_ids,'evaluation_ids':evaluation_ids,
        'train_records':train,'evaluation_records':evaluation,'source_ids':[r['source_unit_id'] for r in train],
        'source_kinds':{r['source_unit_id']:'genuine_native' if r['source_kind'].startswith('native_') else 'unknown_singleton' for r in train},
        'evaluation_source_ids':[r['source_unit_id'] for r in evaluation],
        'evaluation_source_kinds':{r['source_unit_id']:'genuine_native' if r['source_kind'].startswith('native_') else 'unknown_singleton' for r in evaluation},
        'source_acceptance':spec,'base_curator':'e5','base_priority_seed':0,
        '_dossier_group_ids':[g['group_id'] for g in groups],
        'curators':{name:dict(entry) for name,entry in curators.items()},'learners':{},'request_design':design,
        'evidence_role':'prospective_confirmatory_preparation'}
    for name,entry in bundle['curators'].items(): entry.update(train_ids=train_ids,train=caches[name][ti])
    labeled=dataset in {'civil_comments','askubuntu','english_stackexchange'}
    if labeled:
        bundle['task']='civil' if dataset=='civil_comments' else 'multilabel'
        basecal=curators['e5']['calibration']; vocabulary=None
        if bundle['task']=='multilabel':
            vocabulary=ms.select_tag_vocabulary(basecal['records'],{**basecal['provenance'],'feature_contract':'native_normalized_fp32'})
            spec['vocabulary_lock']=vocabulary
            bundle['train_y']=ms.encode_tag_targets(train,vocabulary)
            bundle['evaluation_y']=ms.encode_heldout_tag_targets_for_evaluation(evaluation,vocabulary)
        else:
            bundle['train_y']=np.asarray([[r['original_fields']['toxicity']] for r in train],np.float64)
            bundle['evaluation_y']=np.asarray([[r['original_fields']['toxicity']] for r in evaluation],np.float64)
        needed={g['configuration']['learner_encoder'] for g in groups}
        for name in sorted(needed):
            c=curators[name]['calibration']; provenance={**c['provenance'],'feature_contract':'native_normalized_fp32'}
            logistic=any(g['block']=='F_convex' and g['configuration']['learner_encoder']==name for g in groups)
            entry=_learner_parameters(c['features'],c['records'],provenance,vocabulary,
                curators[name]['encoder_id'],curators[name]['encoder_revision'],logistic)
            entry.update(train_ids=train_ids,evaluation_ids=evaluation_ids,train=caches[name][ti],evaluation=caches[name][ei])
            dimensions={g['configuration']['learner_dimension']:g['configuration'].get('projection_seed')
                        for g in groups if g['configuration']['learner_encoder']==name and g['configuration']['learner_dimension']!=caches[name].shape[1]}
            for dimension,seed in sorted(dimensions.items()):
                matrix=task.seeded_projection(caches[name].shape[1],dimension,seed)
                bundle.setdefault('projections',{})[str(dimension)]={'matrix':matrix,'seed':seed,'data_independent':True}
                pf=(np.asarray(c['features'],np.float64)@matrix).astype(np.float32)
                pp={**c['provenance'],'feature_contract':'fixed_projection_fp32','projection_lineage':{
                    'projection_matrix_sha256':ms._array_hash(matrix),'source_feature_cache_sha256':file_hash(Path(spec['caches'][name]['directory'])/'vectors.npy'),
                    'source_dimension':caches[name].shape[1],'output_dimension':dimension}}
                entry.setdefault('projection_parameters',{})[str(dimension)]=_learner_parameters(pf,c['records'],pp,vocabulary,entry['encoder_id'],entry['encoder_revision'])
            bundle['learners'][name]=entry
    else: bundle['task']='unlabeled'
    family=[g for g in registry['groups'] if g['request_family']==group['request_family']]
    allocations={arm:max(g['allocations'].get(arm,0) for g in family) for arm in ('R','S','U','A')}
    graph=build_reference_graph(bundle['curators']['e5']['train'],train_ids,curators['e5']['threshold_lock']['threshold'],bundle['source_ids'],seed=0)
    bundle['requests']=requests.generate_manifest(graph,dataset_id=dataset,panel_id=group['panel'],
        source_kinds=bundle['source_kinds'],design=design,allocations=allocations,evidence_role='prospective_confirmatory_preparation')
    policies=recipes.build_recipe_book()['supporting_policies']
    if any(g['variant']=='ANN' for g in groups): bundle['ANN_policy']={k:policies['ANN'][k] for k in ('tables','bits','seed','sample_size','sample_seed')}
    if any(g['block']=='E_full_refit' for g in groups):
        from phase5.refit import OFFICIAL
        policy=policies['refit_configuration']
        bundle['refit_configuration']={k:policy[k] for k in ('clusters','iterations','seed','policy','spherical','threads')}
        bundle['refit_configuration'].update(backend=OFFICIAL,epsilon=1-float(curators['e5']['threshold_lock']['threshold']))
        bundle['refit_seed_variants']=policy['alternative_seeds']
    return bundle


def _review_request(bundle,groups,replay,archive):
    return {'schema':'ccu-phase8-review-input-binding-1','reviewer_attestation_created':False,
        'reviewed_bundle_sha256':dispatch.digest(task.fingerprint({k:v for k,v in bundle.items() if k!='external_evidence_review'})),
        'replay_binding_sha256':replay['replay_binding_sha256'],'archive_replay':archive,
        'covered_group_ids':[g['group_id'] for g in groups], 'request_family':groups[0]['request_family'],
        'stable_path_requirement':'Do not relocate or change literal bundle paths before review and dispatch.',
        'remaining_requirements':['supplied_development_and_policy_qualification','supplied_complete_external_review','full_frozen_dispatch_acceptance']}


def build_candidate(config_path,out):
    config_path=Path(config_path).resolve(); base=config_path.parent
    def action(out,report):
        config=load_value(config_path); before=task.fingerprint(config)
        groups,registry=selected_groups(config['group_ids']); group=groups[0]
        spec=source_spec(base/config['source_acceptance'])
        if spec['dataset_id']!=group['corpus']: raise ValueError('Registered group corpus differs from source')
        report['archive_replay']=archive_replay(spec,out/'archive_replay')
        needed={'e5'}|{g['configuration']['curator_encoder'] for g in groups}|{g['configuration']['learner_encoder'] for g in groups}
        inherited=None
        if spec['dataset_id']=='wcep100':
            inherited=load_value(base/config['inherited_news_bundle'])
            curators={name:{k:v for k,v in inherited['curators'][name].items() if k not in {'train','train_ids'}} for name in needed}
            spec['inherited_news_group']=config['inherited_news_group']
        else: curators={name:completed_curator(spec,name,config['calibrations'][name],base) for name in sorted(needed)}
        retained,ids,eids,panel=panel_population(group,spec,curators,strict_json(DESIGN.read_text()),out)
        if spec['dataset_id']=='wcep100': original=retained
        else: original=prep.read_rows(spec['prepared']['records'])
        # Keep full cache order mapped; assemble_bundle gathers selected train/test rows only.
        caches={name:np.load(Path(spec['caches'][name]['directory'])/'vectors.npy',mmap_mode='r',allow_pickle=False) for name in sorted(needed)}
        bundle=assemble_bundle(groups,registry,spec,original,ids,eids,curators,caches,strict_json(DESIGN.read_text()))
        if inherited is not None: bundle['inherited_news_bundle']=inherited
        for key in ('development','execution_policy','external_evidence_review'):
            if key in config: bundle[key]=load_value(base/config[key])
        replay=acceptance.accept_source_cache(group,bundle,strict_json(DESIGN.read_text()),base_dir=out)
        write(out/'source_cache_replay.json',replay)
        if replay.get('machine_acceptance_passed') is not True: raise ValueError('Candidate source/cache replay failed')
        dispatch.recompute_requests(bundle,dispatch._base_graph(bundle),primary=True)
        for g in groups:
            dispatch.resolve(g,bundle)
            if g['block']=='F_convex' and bundle['task']=='multilabel': dispatch.logistic_rule(bundle,g['configuration']['learner_encoder'])
        if before!=task.fingerprint(load_value(config_path)): raise ValueError('Candidate configuration changed')
        descriptor=dump_value(bundle,out/'candidate.json')
        review_request=_review_request(bundle,groups,replay,report['archive_replay']); write(out/'review_request.json',review_request)
        write(out/'groups.json',groups)
        return {'status':'unsigned_candidate_ready','candidate':descriptor,'covered_group_ids':[g['group_id'] for g in groups],
            'request_family':group['request_family'],'panel':panel,'machine_source_cache_replay_passed':True,
            'review_request_sha256':file_hash(out/'review_request.json'),'complete_study_ready':False}
    return _stage(out,'candidate',action)


def candidate_root(candidate_path):
    candidate_path=Path(candidate_path).resolve(); base=candidate_path.parent
    if not (base/'.gitignore').is_file() or (base/'.gitignore').read_text()!='*\n':
        raise ValueError('Candidate root requires its private-output publication guard')
    if candidate_path.name=='candidate.json': receipt_path=base/'receipt.json'; status='unsigned_candidate_ready'
    elif candidate_path.name=='review_candidate.json': receipt_path=base/'development_attachment/receipt.json'; status='review_candidate_ready'
    else: raise ValueError('Only immutable assembler candidate artifacts can advance stages')
    receipt=strict_json(receipt_path.read_text())
    if receipt.get('status')!=status or receipt.get('candidate',{}).get('sha256')!=file_hash(candidate_path):
        raise ValueError('Candidate stage receipt is blocked or no longer binds this file')
    return candidate_path,base


def attach_development(candidate_path,attachment_path):
    """Keep the candidate root fixed. Never synthesize development observations."""
    candidate_path,base=candidate_root(candidate_path)
    def action(out,report):
        from phase8 import resource_policy
        bundle=load_value(candidate_path); groups,_=selected_groups(bundle['_dossier_group_ids'])
        if load_value(base/'groups.json')!=groups: raise ValueError('Saved groups differ from authoritative registry')
        supplied=load_value(attachment_path,base=base)
        bundle['development']=supplied['development']
        bundle['external_evidence_review']=supplied['external_evidence_review']
        qualifications=supplied.get('resource_qualifications')
        if qualifications is None and len(groups)==1:
            qualifications={groups[0]['group_id']:supplied['resource_qualification']}
        if not isinstance(qualifications,dict) or set(qualifications)!={g['group_id'] for g in groups}:
            raise ValueError('Each covered group requires its own reproduced resource qualification')
        bundle['execution_policy']=qualifications[groups[0]['group_id']]['execution_policy']
        if not isinstance(bundle['execution_policy'],dict): raise ValueError('A qualified concrete execution policy is required')
        for group in groups:
            q=qualifications[group['group_id']]
            if q['execution_policy']!=bundle['execution_policy']: raise ValueError('Covered groups need one common execution policy')
            verified=resource_policy.verify_resource_receipt(q,group,bundle,supplied['development_audits'],supplied['target_profiles'],base_dir=base)
            if verified != q: raise ValueError('Resource qualifier did not verify this exact candidate')
        bundle['resource_policy_qualification']=qualifications
        bundle['resource_policy_observations']={'development_audits':supplied['development_audits'],'target_profiles':supplied['target_profiles']}
        replay=acceptance.accept_source_cache(groups[0],bundle,strict_json(DESIGN.read_text()),base_dir=base)
        write(out/'source_cache_replay.json',replay)
        if replay.get('machine_acceptance_passed') is not True: raise ValueError('Updated candidate replay failed')
        archive=archive_replay(bundle['source_acceptance'],out/'archive_replay')
        descriptor=dump_value(bundle,base/'review_candidate.json')
        write(base/'final_review_request.json',_review_request(bundle,groups,replay,archive))
        return {'status':'review_candidate_ready','candidate':descriptor,'complete_study_ready':False}
    return _stage(base/'development_attachment','attach_development',action)


def finalize(candidate_path,review_path):
    candidate_path,base=candidate_root(candidate_path)
    def action(out,report):
        from phase8 import resource_policy
        bundle=load_value(candidate_path); groups,_=selected_groups(bundle['_dossier_group_ids'])
        if load_value(base/'groups.json')!=groups: raise ValueError('Saved groups differ from authoritative registry')
        before=task.fingerprint(bundle); review=load_value(review_path,base=base)
        bundle['external_evidence_review']=review
        report['archive_replay']=archive_replay(bundle['source_acceptance'],out/'archive_replay')
        observations=bundle['resource_policy_observations']
        for group in groups:
            q=bundle['resource_policy_qualification'][group['group_id']]
            if q['execution_policy']!=bundle['execution_policy']: raise ValueError('Resource policy differs from attached qualification')
            verified=resource_policy.verify_resource_receipt(q,group,bundle,
                observations['development_audits'],observations['target_profiles'],base_dir=base)
            if verified != q: raise ValueError('Resource qualification is blocked or changed')
        accepted=[]
        for i,group in enumerate(groups):
            value=dispatch.acceptance(group,bundle,strict_json(DESIGN.read_text()),mode=dispatch.PRIMARY_MODE,base_dir=base)
            write(out/f'acceptance_{i:03d}.json',value); accepted.append(value)
        if not all(v.get('primary_inputs_accepted') is True for v in accepted): raise ValueError('One or more frozen group acceptance gates failed')
        if before!=task.fingerprint(load_value(candidate_path)): raise ValueError('Candidate changed during final review')
        def publish():
            descriptor=dump_value(bundle,base/'bundle.final.json')
            write(base/'bundles.json',{groups[0]['request_family']:descriptor})
            return {'bundles_map':str(base/'bundles.json')}
        return {'status':'selected_groups_accepted_dispatch_still_rechecks','covered_group_ids':[g['group_id'] for g in groups],
            '_commit':publish,'primary_execution_allowed':False,'complete_study_ready':False,
            'all_study_jobs_must_still_be_retained':True}
    return _stage(base/'finalization','finalize',action)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='stage',required=True)
    c=sub.add_parser('calibration'); c.add_argument('--source-spec',type=Path,required=True); c.add_argument('--encoder',choices=['e5','mpnet'],required=True); c.add_argument('--out',type=Path,required=True)
    c=sub.add_parser('candidate'); c.add_argument('--config',type=Path,required=True); c.add_argument('--out',type=Path,required=True)
    c=sub.add_parser('attach-development'); c.add_argument('--candidate',type=Path,required=True); c.add_argument('--attachment',type=Path,required=True)
    c=sub.add_parser('finalize'); c.add_argument('--candidate',type=Path,required=True); c.add_argument('--review',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.stage=='calibration': result=build_calibration(a.source_spec,a.encoder,a.out)
        elif a.stage=='candidate': result=build_candidate(a.config,a.out)
        elif a.stage=='attach-development': result=attach_development(a.candidate,a.attachment)
        else: result=finalize(a.candidate,a.review)
    except (ValueError,OSError) as e: p.exit(2,'BLOCKED: '+str(e)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False)); return 2 if result['status']=='blocked' else 0


if __name__=='__main__': raise SystemExit(main())
