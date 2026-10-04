"""Prospective protocol registry and fail-closed local artifact inventory.

This is a job plan, not a preregistration submission or scientific execution gate.
Data-dependent horizons stay symbolic until authentic frozen manifests exist.
"""
from __future__ import annotations
from pathlib import Path
import argparse,gzip,hashlib,json,math

REPO=Path(__file__).resolve().parents[2]
DESIGN=REPO/'output/empirical_program/study_design.json'
PROTOCOL=REPO/'output/empirical_program/counterfactual_curation_empirical_protocol.tex'
CORE=('civil_comments','askubuntu','cc_news')
LABELED=('civil_comments','askubuntu')
METHODS=['O-G','O-T','B-E','B-A','P-I','P-S','P-R','B-F']
NATIVE=dict(curator_encoder='e5',learner_encoder='e5',learner_dimension=768,
            priority_seed=0,threshold='calibrated',state_precision='FP64',decoder='Cholesky')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def registry_groups(design):
    """Explicit bounded cells; never an unregistered full factorial."""
    groups=[]
    def add(block,corpus,panel,allocation,methods,variant='native',**config):
        groups.append(dict(group_id=f'{block}/{corpus}/{panel}/{variant}',block=block,
            corpus=corpus,panel=panel,variant=variant,allocations=dict(allocation),methods=methods,
            configuration={**NATIVE,**config},request_family=f'{corpus}/{panel}',
            source_condition='verified_original_urls_and_native_sources' if corpus=='wcep100' else 'genuine_native_sources_available'))
    alloc=design['allocations_per_corpus_panel_primary_seed']
    for corpus in CORE:
        for size in design['population']['soft_primary_size_targets']:
            add('A_structure',corpus,f'primary-{size}',alloc['structure'],['independent_structure_oracle'])
    for corpus in LABELED:
        add('B_relevance',corpus,'primary-10000',alloc['task_relevance'],['O-G','B-F'])
        add('C_full_methods',corpus,'primary-10000',alloc['full_methods'],METHODS)
        add('E_full_refit',corpus,'separate-refit-5000',alloc['full_refit'],['R_full_refit','F_frozen_fitted','B_frozen_selection'],
            curator_encoder='e5',curator='pinned_standard_SemDeDup',cluster_policy='development_lock_required')
        add('F_convex',corpus,'primary-10000',alloc['convex_extension'],
            ['fresh_optimum','eligible_payload_retraining','certified_convex_repair'])
    for corpus in CORE:
        add('D_replication',corpus,'second-disjoint-10000',alloc['replication'],
            ['independent_structure_oracle'] if corpus=='cc_news' else ['O-G','B-F'])
    add('D_replication','english_stackexchange','primary-10000',alloc['replication'],['O-G','B-F'])
    add('D_replication','wcep100','whole-events-25000',alloc['replication'],['independent_structure_oracle'])
    # Every alternative changes only its named factor relative to the fixed panel.
    boundary=alloc['each_boundary_configuration']
    for dimension in design['features']['projection_ablation_dimensions']:
        add('G_boundary','civil_comments','primary-10000',boundary,METHODS,
            variant=f'projection-{dimension}',learner_dimension=dimension,projection_seed=design['seeds']['projection_seed'])
    for curator in ('e5','mpnet'):
        for learner in ('e5','mpnet'):
            add('G_boundary','civil_comments','primary-10000',boundary,['O-G','B-F'],
                variant=f'encoder-{curator}-{learner}',curator_encoder=curator,learner_encoder=learner,
                population='same_post_E5_guard_no_MPNet_refilter')
    for seed in design['seeds']['sensitivity_priority_seeds']:
        add('G_boundary','civil_comments','primary-10000',boundary,['O-G','B-F'],
            variant=f'priority-{seed}',priority_seed=seed)
    for sign in (-1,1):
        add('G_boundary','civil_comments','primary-10000',boundary,['O-G','B-F'],
            variant=f'threshold-{sign:+d}',threshold=(f'max(0,calibrated-{design["threshold_calibration"]["sensitivity_delta"]:.2f})'
                if sign<0 else f'min(1,calibrated+{design["threshold_calibration"]["sensitivity_delta"]:.2f})'))
    for factor in (.1,10):
        add('G_boundary','civil_comments','primary-10000',boundary,METHODS,
            variant=f'lambda-x{factor:g}',lambda_factor=factor)
    add('G_boundary','civil_comments','primary-10000',boundary,METHODS,
        variant='fp32-state',state_precision='FP32',scope='separate_accuracy_memory_frontier')
    add('G_boundary','civil_comments','primary-10000',boundary,METHODS,
        variant='zero-start-CG',decoder='zero_start_CG',stopping_rule='development_lock_required')
    add('G_boundary','cc_news','complete-calendar-window',boundary,['exhaustive_graph','pinned_ANN_candidates'],
        variant='ANN',ANN_policy='development_lock_required')
    add('G_boundary','cc_news','complete-calendar-window',boundary,['independent_structure_oracle'],
        variant='natural-window-vs-source-panel',source_panel_comparator='primary-10000')
    return groups

def needs(group):
    corpus=group['corpus']; panel=group['panel']; conf=group['configuration']
    keys=['global.code_lock','global.resource_policy','global.development_precision_analysis',
          'global.compaction_persistence_policy',f'{corpus}.original_corpus',f'{corpus}.parser_audit',
          f'{corpus}.source_partition',f'{corpus}.external_provenance_review',f'{corpus}.{panel}.manifest',
          f'{corpus}.{panel}.requests']
    if corpus!='wcep100':keys += [f'{corpus}.e5.semantic_guard']
    for encoder in sorted({conf['curator_encoder'],conf['learner_encoder']}):
        keys += [f'{corpus}.{encoder}.model_tokenizer_assets',f'{corpus}.{encoder}.cache',f'{corpus}.{encoder}.cache_manifest']
    encoder=conf['curator_encoder']
    calibration_corpus='cc_news' if corpus=='wcep100' else corpus
    keys += [f'{calibration_corpus}.{encoder}.{name}' for name in ('selection_manifest','selection_responses','selection_lock',
             'validation_manifest','validation_responses','quality_report')]
    if corpus in (*LABELED,'english_stackexchange'):
        keys += [f'{corpus}.lambda_lock',f'{corpus}.test_manifest',f'{corpus}.test_cache',f'{corpus}.test_labels']
        if conf['learner_encoder']!='e5':keys += [f'{corpus}.{conf["learner_encoder"]}.lambda_lock',f'{corpus}.{conf["learner_encoder"]}.test_cache']
    if corpus in ('askubuntu','english_stackexchange'):
        keys += [f'{corpus}.tag_vocabulary',f'{corpus}.tag_thresholds',f'{corpus}.duplicate_links']
    if corpus=='cc_news':keys += ['cc_news.pinned_PSL','cc_news.complete_month_evidence','cc_news.calendar_cohort_lock']
    if corpus=='wcep100':keys += ['wcep100.original_event_manifest']
    if group['block']=='E_full_refit':keys += ['global.semdedup_dependency_lock',f'{corpus}.semdedup_development_configuration']
    if group['block']=='F_convex':keys += ['global.convex_three_method_worker_lock']
    if conf['learner_dimension']!=768:keys += [f'global.projection_{conf["learner_dimension"]}_matrix',f'{corpus}.{conf["learner_encoder"]}.lambda_projection_{conf["learner_dimension"]}']
    if group['variant']=='ANN':keys += ['global.ANN_candidate_policy','cc_news.ANN_missed_pair_sampling_lock']
    if group['variant']=='zero-start-CG':keys += ['global.CG_common_stopping_policy']
    if group['variant'].startswith('threshold-'):keys += [f'{corpus}.{encoder}.{group["variant"]}.quality_audit']
    return sorted(set(keys))

def enumerate_jobs(groups,design):
    """One planned trajectory job can contain multiple paired method workers."""
    rq=design['requests']
    for group in groups:
        for arm,count in group['allocations'].items():
            for index in range(count):
                source=arm=='S';path=f'{arm}{index:03d}'
                native_key=f'{group["request_family"]}/{path}'
                row=dict(job_id=f'{group["group_id"]}/{path}',group_id=group['group_id'],
                    arm=arm,path_index=index,trajectory_id=path,request_pairing_key=native_key,
                    planned_checkpoint_units=rq['source_checkpoints'] if source else rq['record_checkpoints'],
                    planned_horizon=rq['source_horizon'] if source else rq['record_horizon'],
                    actual_horizon=None,actual_checkpoint_units=None,actual_request_IDs_available=False,
                    clipping='min horizon with available arm universe; deduplicate clipped checkpoints; retain zero/inapplicable status',
                    reset_initial_state=True,methods=group['methods'],execution_allowed=False,status='planned_blocked')
                if source and group['corpus']=='wcep100':
                    row['additional_required_artifact_keys']=['wcep100.original_URL_verification']
                    row['unavailable_source_evidence_preserves_record_only_jobs']=True
                if group['block']=='C_full_methods':row['subset_of']=f'B_relevance/{group["corpus"]}/{group["panel"]}/native/{path}'
                if group['block']=='B_relevance':row['shared_structure_path']=f'A_structure/{group["corpus"]}/{group["panel"]}/native/{path}'
                if group['block']=='E_full_refit' and arm in ('R','S') and index<2:
                    row['additional_oracle_every_integer_service_step']=True
                if group['block']=='C_full_methods' and arm in ('R','S') and index<16:
                    row['complete_state_rebuild_each_checkpoint']=True
                yield row
                if source:
                    yield dict(job_id=row['job_id']+'/volume-matched-R',group_id=group['group_id'],
                        arm='R-volume-matched-to-S',trajectory_id='R-volume-matched-to-'+path,
                        parent_source_job=row['job_id'],request_pairing_key=native_key+'/volume-matched',
                        planned_checkpoint_units='parent_source_cumulative_raw_record_counts',
                        planned_horizon='parent_source_final_cumulative_raw_record_count',
                        actual_horizon=None,actual_checkpoint_units=None,actual_request_IDs_available=False,
                        own_separately_sufficient_record_horizon=True,methods=group['methods'],
                        additional_required_artifact_keys=row.get('additional_required_artifact_keys',[]),
                        execution_allowed=False,status='planned_blocked')
                if group['block']=='C_full_methods' and arm in ('R','S') and index<8:
                    for repeat in range(design['audit']['fresh_process_timing_repetitions']):
                        yield {**row,'job_id':row['job_id']+f'/fresh-process-{repeat}',
                               'kind':'timing_repeat','repeat':repeat,'parent_trajectory_job':row['job_id']}
        if group['block']=='E_full_refit':
            for kind,count in [('same_seed_no_deletion',5),('alternative_seed_no_deletion',3)]:
                for index in range(count):
                    yield dict(job_id=group['group_id']+f'/{kind}-{index}',group_id=group['group_id'],
                        kind=kind,repeat=index,planned_checkpoint_units=[0],actual_request_IDs_available=False,
                        execution_allowed=False,status='planned_blocked',methods=group['methods'])

# These protocol obligations have no fully specified allocation/recipe. They are
# retained as jobs requiring a prospective completion decision, never guessed away.
UNRESOLVED=[
 ('structure_mass_PPS','Structure-only mass-weighted source sampler; trajectory allocation and panel scope need a development lock'),
 ('structure_one_percent_horizon','Separate ceil(0.01*N) initial state; trajectory allocation/checkpoints need a development lock'),
 ('excluded_blocker_stress','Separate A variant restricted to initially excluded blockers; count/panel allocation needs lock'),
 ('news_chronological','Descriptive chronological date-range withdrawals; exact date granularity/stream length needs lock'),
 ('fresh_graph_audit','16 precommitted paths per primary corpus; arm allocation not specified'),
 ('common_cached_graph_oracle','Separate cached-graph and retained-similarity cost tiers; benchmark scope needs lock'),
 ('utility_reference','Curated/uncurated/hash-matched ridge and converged logistic initial utility; optimization policy needs lock'),
 ('negative_controls','Exact-text and fixed-order greedy own-excluded deletion controls, no-curation floor; request allocation needs lock'),
 ('head_normalization_error_diagnostic','One fixed Civil count-shift diagnostic; specific path/checkpoint needs lock'),
 ('graph_envelope_conditional','Admissible only with valid uniform score bounds; theorem may remain uninstantiated'),
 ('every_release_persistence','Separate persistence sensitivity; trajectory allocation needs lock'),
 ('cold_warm_scope','Cold-process and warm-service panels need a common measured resource/cache policy'),
 ('official_Civil_splits_optional','Optional comparability sensitivity if resources allow; retained as optional, not silently activated'),
 ('human_threshold_calibration','Per-corpus/per-curator 600 max selection and 200 validation pairs, three independent judgments; actual pair frames pending'),
 ('human_admission_main','400 admission and 200 control pairs maximum, equal corpus quotas and registered strata; actual frames pending'),
 ('human_admission_context','Two distinct 100-admission context audits, three independent annotators; actual frames pending'),
 ('boundary_method_scope','Prospective registry allocates all eight methods for dimension/numeric/conditioning/CG alternatives; narrow only through documented pre-confirmation amendment'),
 ('matched_R_method_scope','Prospective registry pairs each S path with the same listed methods at its own sufficient record horizon; do not count as primary R replication'),
 ('test_source_bootstrap','Crossed test-source and trajectory bootstrap, macro-F1 recomputation and Holm contrast families; locked analysis input bindings pending'),
 ('primary_numerical_release_gate','Common normalized eta<=1e-10 and condition diagnostics required; an absolute residual alone is insufficient'),
 ('worker_dependency_and_IO_accounting','Bind actual binary/runtime dependency chain and charge logical payload/metadata reads; file inventory alone does not establish these'),
]

def build_registry():
    design=json.loads(DESIGN.read_text());groups=registry_groups(design)
    for group in groups:
        group['required_artifact_keys']=needs(group)
        group['conditional_artifact_keys']={'S':['wcep100.original_URL_verification']} if group['corpus']=='wcep100' else {}
    jobs=list(enumerate_jobs(groups,design))
    for name,reason in UNRESOLVED:
        jobs.append(dict(job_id='unresolved/'+name,kind='unresolved_protocol_obligation',
                         status='recipe_or_frame_lock_required',reason=reason,execution_allowed=False))
    keys=sorted({key for group in groups for key in group['required_artifact_keys']} |
                {key for group in groups for values in group['conditional_artifact_keys'].values() for key in values})
    counts={group['group_id']:sum(j.get('group_id')==group['group_id'] for j in jobs) for group in groups}
    registry=dict(schema='ccu-prospective-study-registry-1',date='2026-10-04',
        protocol_status=design['status'],preregistration_submitted=False,execution_allowed=False,
        primary_semantic_study_started=False,source_bindings={str(p.relative_to(REPO)):sha(p) for p in
            (DESIGN,PROTOCOL,REPO/'empirical_execution/phase3/PREPARATION_AMENDMENT.txt')},
        registry_code_sha256=sha(Path(__file__)),groups=groups,group_job_counts=counts,
        job_count=len(jobs),jobs_content_sha256=digest(jobs),required_artifact_keys=keys,
        symbolic_horizons_until_authentic_data=True,
        prospective_implementation_choices=['second disjoint panels for all three core structure corpora',
          'all named paired methods receive matched-volume controls, separately sufficing their record horizon',
          'all eight methods in dimension/FP32/lambda/CG boundary cells; narrow only before confirmation',
          'same-E5 encoder cell references existing native outcomes; never independent replication'],
        unresolved_obligations=[dict(id=k,requirement=v) for k,v in UNRESOLVED])
    return registry,jobs

def preflight(registry,inventory,base_dir):
    """Validate supplied bytes; do not infer authenticity or human evidence from files."""
    base=Path(base_dir).resolve(); entries=inventory.get('artifacts',{});results={}
    for key in registry['required_artifact_keys']:
        entry=entries.get(key)
        result=dict(key=key,byte_integrity_verified=False,authenticity_verified=False)
        if not isinstance(entry,dict) or not entry.get('path'):
            result.update(status='missing_descriptor',requirement='Supply actual local artifact and SHA256')
        else:
            try:
                path=(base/entry['path']).resolve()
                if not path.is_relative_to(base):raise ValueError('path escapes inventory root')
                expected=entry.get('sha256')
                if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
                    raise ValueError('explicit lowercase SHA256 required')
                if not path.is_file():raise FileNotFoundError('artifact absent')
                actual=sha(path)
                result.update(path=entry['path'],bytes=path.stat().st_size,actual_sha256=actual)
                if actual!=expected:raise ValueError('artifact SHA256 mismatch')
                result.update(status='bytes_verified_semantic_review_pending',byte_integrity_verified=True)
            except (ValueError,OSError) as error:
                result.update(status='invalid_or_missing_artifact',reason=str(error))
        results[key]=result
    group_status=[]
    for group in registry['groups']:
        missing=[key for key in group['required_artifact_keys'] if not results[key]['byte_integrity_verified']]
        group_status.append(dict(group_id=group['group_id'],planned_jobs=registry['group_job_counts'][group['group_id']],
            missing_or_invalid_artifact_keys=missing,status='blocked_missing_or_invalid_bytes' if missing else 'blocked_semantic_and_execution_acceptance',
            conditional_arm_missing_keys={arm:[key for key in values if not results[key]['byte_integrity_verified']]
                                         for arm,values in group['conditional_artifact_keys'].items()},
            execution_allowed=False,actual_human_ratings_verified=0))
    return dict(schema='ccu-study-preflight-1',execution_allowed=False,actual_semantic_study_started=False,
        artifact_count=len(results),byte_verified_count=sum(r['byte_integrity_verified'] for r in results.values()),
        artifact_results=results,group_status=group_status,planned_jobs_retained=registry['job_count'],
        required_external_acceptance=['Corpus authenticity/source completeness and access review',
          'Actual pinned encoder derivation, cache/frame binding and dimension verification',
          'Recompute selection and independent validation from genuine blinded human responses',
          'Calibration-only task settings and post-E5 panel/request locks',
          'Code/runtime/resource/isolation/persistence and branch-specific worker acceptance',
          'Resolve every applicable recipe obligation before confirmation'],
        capability_declarations=inventory.get('capabilities',{}),capability_declarations_are_not_verified=True)

def export(output_dir,inventory=None,base_dir=None):
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('registry export is immutable')
    out.mkdir(parents=True,exist_ok=True);registry,jobs=build_registry()
    write(out/'registry.json',registry)
    payload=b''.join(canonical(job)+b'\n' for job in jobs)
    (out/'jobs.jsonl.gz').write_bytes(gzip.compress(payload,mtime=0))
    template=dict(schema='ccu-study-artifact-inventory-1',artifacts={key:dict(path=None,sha256=None)
        for key in registry['required_artifact_keys']},capabilities={})
    write(out/'artifact_inventory_template.json',template)
    report=preflight(registry,inventory or template,base_dir or out)
    write(out/'preflight.json',report)
    write(out/'export_manifest.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        expanded_job_count=len(jobs),registry_digest=digest(registry)))
    return registry,report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--inventory',type=Path);args=parser.parse_args()
    inv=json.loads(args.inventory.read_text()) if args.inventory else None
    registry,report=export(args.output_dir,inv,args.inventory.resolve().parent if args.inventory else None)
    print(json.dumps(dict(planned_jobs=registry['job_count'],groups=len(registry['groups']),
         verified_artifacts=report['byte_verified_count'],required_artifacts=report['artifact_count'],
         execution_allowed=False),indent=2))

if __name__=='__main__':main()
