"""Sealed human-admission audit sampling and BLANK local annotation packs.

No human judgments are produced or imputed. Sampling probabilities condition on
the frozen declared checkpoint union and one fixed former-blocker draw per record.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import platform
import random
import numpy as np
from phase3.panels import graph_normalize, reference_cosines
from phase4.requests import digest, derive_seed, verify_manifest, _graph_state

SCHEMA='ccu-human-admission-audit-preparation-1'
CORPORA=('civil_comments','askubuntu','cc_news')
QUOTAS={'civil_comments':{'R/S':67,'U/A':67,'control':67},
        'askubuntu':{'R/S':67,'U/A':66,'control':67},
        'cc_news':{'R/S':67,'U/A':66,'control':66}}
TASKS={'civil_comments':'Assess whether a textual distinction could change the perceived toxicity of a comment.',
       'askubuntu':'Assess whether a textual distinction could change the technical tags appropriate for this question.',
       'cc_news':'Assess whether the texts contain distinct factual assertions.'}


def _fraction(value):
    return {'numerator':value.numerator,'denominator':value.denominator}


def _label_relation(a,b):
    if a is None or b is None: return 'unavailable'
    return 'equal' if a==b else 'unequal'


def _target(value,corpus):
    if value is None: return None
    if corpus=='civil_comments':
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Civil original toxicity fraction must be finite in [0,1] or missing')
        return float(value)
    if corpus=='askubuntu':
        if not isinstance(value,(list,tuple)) or not value or any(type(x) not in (int,bool) or x not in (0,1) for x in value):
            raise ValueError('Ask Ubuntu labels must be original-tag vocabulary binary vectors or missing')
        return list(value)
    raise ValueError('News has no task label; do not invent one')


def _band(score,tau):
    if not math.isfinite(score) or not -1<=tau<1 or not score>tau:
        raise ValueError('former-blocker score must satisfy the fixed strict graph threshold below one')
    margin=(score-tau)/(1-tau)
    return sum(margin>=boundary for boundary in (.25,.5,.75)),margin


def _code_hashes():
    root=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),root/'phase4/requests.py',root/'phase3/reference_graph.py',root/'phase3/panels.py']}


def _one_corpus_frame(corpus,inputs,master_seed):
    required={'records','features','graph','request_manifest','provenance'}
    if not isinstance(inputs,dict) or not required<=inputs.keys(): raise ValueError('incomplete corpus inputs')
    records=inputs['records'];graph=inputs['graph'];requests=inputs['request_manifest']
    verify_manifest(requests,graph)
    ids,owners,blockers,binding=_graph_state(graph)
    record_ids=[r.get('record_id') for r in records]
    if tuple(record_ids)!=tuple(graph.record_ids): raise ValueError('records/features must align with graph input ID order')
    if any(not isinstance(r.get('text'),str) or not r['text'].strip() for r in records): raise ValueError('original nonempty natural texts required')
    x=np.asanyarray(inputs['features'])
    if x.dtype!=np.dtype('float32') or x.ndim!=2 or x.shape[0]!=len(records): raise ValueError('aligned FP32 feature cache required')
    normalized=graph_normalize(x)
    labels=inputs.get('existing_labels',{})
    if not isinstance(labels,dict) or not set(labels)<=set(ids): raise ValueError('labels must map only current record IDs')
    if corpus=='cc_news' and any(v is not None for v in labels.values()): raise ValueError('News cannot have invented task labels')
    labels={r:_target(labels.get(r),corpus) for r in ids} if corpus!='cc_news' else {r:None for r in ids}
    if corpus=='askubuntu' and len({len(v) for v in labels.values() if v is not None})>1: raise ValueError('one frozen tag vocabulary dimension required')
    provenance=inputs['provenance']
    if not isinstance(provenance,dict) or not all(isinstance(provenance.get(k),str) and provenance[k] for k in
        ('corpus_snapshot','representation_id','representation_revision','evidence_role')):
        raise ValueError('explicit corpus/representation/evidence provenance declaration required')
    initial={r for r in ids if not blockers[r]};events=defaultdict(list);control_events=defaultdict(list)
    eligible_paths=[];ignored=Counter()
    for path in requests['trajectories']:
        arm=path['arm']
        if arm not in {'R','S','U','A'}: ignored[arm]+=1;continue
        eligible_paths.append(path['trajectory_id'])
        for observation in path['observations']:
            k=observation['checkpoint'];prefix=set(path['deletion_order'][:k])
            deleted=prefix if path['unit']=='record' else {r for r in ids if owners[r] in prefix}
            current={r for r in ids if r not in deleted and blockers[r]<=deleted}
            admitted=current-initial
            if sorted(admitted)!=observation['admitted_record_ids']: raise ValueError('request diagnostic does not match complete graph admissions')
            event=dict(trajectory_id=path['trajectory_id'],arm=arm,checkpoint=k)
            for rid in admitted: events[rid].append(event)
            for rid in set(ids)-initial-deleted-current: control_events[rid].append(event)
    control_ids=set(control_events)-set(events)
    lookup={r:i for i,r in enumerate(graph.record_ids)};frame=[]
    for rid in sorted(set(events)|control_ids):
        candidates=sorted(blockers[rid])
        seed=derive_seed(master_seed,'ccu-admission-former-blocker-v1',corpus,requests['panel_id'],rid)
        former=random.Random(seed).choice(candidates)
        score=float(reference_cosines(normalized[lookup[rid]:lookup[rid]+1],normalized[lookup[former]:lookup[former]+1])[0,0])
        band,margin=_band(score,float(graph.threshold))
        relation=_label_relation(labels[rid],labels[former])
        roles=sorted({'R/S' if e['arm'] in {'R','S'} else 'U/A' for e in events[rid]}) if rid in events else ['control']
        memberships=[[corpus,role,band,relation] for role in roles]
        record_events=events[rid] if rid in events else control_events[rid]
        frame.append(dict(pair_id=digest(['human-admission-pair-v1',corpus,rid,former]),
            corpus=corpus,record_id=rid,former_blocker_id=former,roles=roles,strata=memberships,
            score=score,score_hex=score.hex(),score_excess_fraction=margin,similarity_band=band,
            existing_label_relation=relation,former_blocker_count=len(candidates),
            former_blocker_choice_seed=seed,former_blocker_draw_probability=_fraction(Fraction(1,len(candidates))),
            occurrence_multiplicity=len(record_events),occurrence_by_arm=dict(Counter(e['arm'] for e in record_events)),
            occurrences=record_events))
    vh=hashlib.sha256();vh.update(digest({'shape':list(x.shape),'dtype':'float32-little-endian'}).encode())
    vh.update(np.asarray(x,dtype='<f4',order='C').tobytes())
    bindings=dict(graph=binding,request_manifest_sha256=requests['manifest_sha256'],
        records_sha256=digest([{'record_id':r['record_id'],'text':r['text']} for r in records]),
        features_sha256=vh.hexdigest(),existing_labels_sha256=digest(labels),provenance=provenance,
        eligible_trajectory_ids=eligible_paths,ignored_noncore_arm_counts=dict(ignored),
        admitted_unique_records=len(events),control_unique_records=len(control_ids),
        original_initially_excluded_records=len(ids)-len(initial))
    return frame,bindings


def prepare_admission_audit(corpus_inputs,*,master_seed=20271003,evidence_role='engineering_nonconfirmatory'):
    if isinstance(master_seed,bool) or not isinstance(master_seed,int) or master_seed<0: raise ValueError('nonnegative integer seed required')
    if not isinstance(corpus_inputs,dict) or set(corpus_inputs)-set(CORPORA): raise ValueError('only the three prespecified corpora may enter')
    if evidence_role not in {'engineering_nonconfirmatory','prospective_confirmatory_preparation'}: raise ValueError('explicit evidence role required')
    frame=[];bindings={};corpus_status={}
    for corpus in CORPORA:
        if corpus not in corpus_inputs:
            corpus_status[corpus]='missing_corpus_no_quota_reallocation';continue
        rows,binding=_one_corpus_frame(corpus,corpus_inputs[corpus],master_seed)
        if evidence_role=='prospective_confirmatory_preparation' and binding['provenance']['evidence_role']!='prospective_confirmatory_preparation':
            raise ValueError('engineering corpus cannot be relabeled as primary preparation')
        frame.extend(rows);bindings[corpus]=binding;corpus_status[corpus]='frame_prepared_provenance_declared_only'
    frame.sort(key=lambda p:p['pair_id']);by_id={p['pair_id']:p for p in frame}
    if len(by_id)!=len(frame): raise ValueError('duplicate pair identity')
    buckets=defaultdict(list)
    for p in frame:
        for stratum in p['strata']: buckets[tuple(stratum)].append(p['pair_id'])
    strata=[];selected_memberships=defaultdict(list);probability_by_stratum={}
    for corpus in CORPORA:
        relations=['unavailable'] if corpus=='cc_news' else ['equal','unequal','unavailable']
        cells=[(band,relation) for band in range(4) for relation in relations]
        for role in ['R/S','U/A','control']:
            budget=QUOTAS[corpus][role]
            for offset,(band,relation) in enumerate(cells):
                key=(corpus,role,band,relation);quota=budget//len(cells)+int(offset<budget%len(cells))
                population=sorted(buckets[key]);sample_size=min(quota,len(population))
                seed=derive_seed(master_seed,'ccu-human-admission-stratum-v1',corpus,role,str(band),relation)
                sample=sorted(random.Random(seed).sample(population,sample_size))
                probability=Fraction(sample_size,len(population)) if population else Fraction(0)
                probability_by_stratum[key]=probability
                for pid in sample: selected_memberships[pid].append(list(key))
                strata.append(dict(stratum=list(key),quota=quota,frame_pair_ids=population,
                    population_size=len(population),sample_size=sample_size,shortfall=quota-sample_size,
                    status='empty_frame' if not population else 'census' if sample_size==len(population) else 'sampled',
                    sampling_seed=seed,conditional_inclusion_probability=_fraction(probability),selected_pair_ids=sample))
    for pair in frame:
        nonselection=Fraction(1)
        for stratum in pair['strata']: nonselection*=1-probability_by_stratum[tuple(stratum)]
        pair['conditional_union_inclusion_probability']=_fraction(1-nonselection)
    selected=[]
    assignment_context=digest({'input_bindings':bindings,'complete_frame':frame,'master_seed':master_seed})
    for pid in sorted(selected_memberships):
        p=by_id[pid];assignment_ids=[digest(['admission-audit-assignment-v1',assignment_context,pid,j]) for j in range(3)]
        selected.append(dict(pair_id=pid,selected_in_strata=selected_memberships[pid],
            conditional_union_inclusion_probability=p['conditional_union_inclusion_probability'],assignment_ids=assignment_ids))
    config=dict(master_seed=master_seed,python_version=platform.python_version(),corpora=list(CORPORA),quotas=QUOTAS,
        admission_max=400,control_max=200,annotators_per_unique_pair=3,
        similarity_bands='(score-tau)/(1-tau): [0,.25),[.25,.5),[.5,.75),[.75,infinity)',
        label_relation='exact existing target equality/inequality or unavailable; no imputation',
        control_population='initially excluded, retained and excluded at least once, never admitted within declared core checkpoint union',
        stratum_quotas='equal quotient with fixed-order remainder across four bands times label-relation cells',
        missing_or_scarce_strata='census available units; no favorable refill and no cross-stratum transfer',
        pair_rule='one seeded uniform former blocker per record; fixed across all request occurrences',
        repeated_pairs='sample role strata independently; deduplicate human assignments; union probability one minus product nonselection',
        inclusion_scope='conditional on complete frozen core checkpoint union and realized fixed former-blocker choices',
        target_scope='unique pairs in declared frame; not all possible deletions, not all blocker pairs, not a causal control')
    result=dict(schema=SCHEMA,evidence_role=evidence_role,configuration=config,configuration_sha256=digest(config),
        corpus_status=corpus_status,input_bindings=bindings,complete_frame=frame,complete_frame_sha256=digest(frame),
        strata=strata,selected_pairs=selected,unique_selected_pairs=len(selected),blank_assignment_count=3*len(selected),
        selected_admission_pairs=sum(bool(set(by_id[p['pair_id']]['roles'])&{'R/S','U/A'}) for p in selected),
        selected_control_pairs=sum('control' in by_id[p['pair_id']]['roles'] for p in selected),
        conditional_union_probability_formula='1 - product_h(1 - n_h/N_h), independent stratum samples',
        code_sha256=_code_hashes(),source_provenance_verified_by_this_module=False,
        confirmatory_study_ready=False,human_dispatch_ready=False,completed_human_responses=0,
        required_before_dispatch=['real corpus/model provenance review','independent fluent annotator recruitment',
            'payment rate and estimated time','consent/withdrawal workflow','permission to distribute protected text'],
        neighborhood_audit={'status':'pending_real_context_and_fixed_context_limit','target_admissions':100,
            'requires':['complete former-blocker texts','predeclared uniform subset of admission records','logged truncation/skip policy']},
        surviving_selected_audit={'status':'pending_real_retained_checkpoint_context','target_admissions':100,
            'requires':['predeclared record/checkpoint rule','actual surviving selected population','exact nearest similarity with stable-ID ties'],
            'scope':'exploratory interpretation; not corpus recovery or global novelty guarantee'})
    result['manifest_sha256']=digest(result);return result


def verify_admission_audit(manifest):
    if not isinstance(manifest,dict) or manifest.get('schema')!=SCHEMA: raise ValueError('wrong admission-audit schema')
    if digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})!=manifest.get('manifest_sha256'):
        raise ValueError('admission-audit manifest seal mismatch')
    if digest(manifest['complete_frame'])!=manifest['complete_frame_sha256']: raise ValueError('complete frame hash mismatch')
    if digest(manifest['configuration'])!=manifest['configuration_sha256']: raise ValueError('configuration hash mismatch')
    return True


def write_admission_pack(manifest,records_by_corpus,directory):
    """Create local blank files only, refusing any existing output directory."""
    verify_admission_audit(manifest);texts={}
    for corpus,binding in manifest['input_bindings'].items():
        records=records_by_corpus.get(corpus)
        if not isinstance(records,list): raise ValueError('complete aligned natural records required')
        if digest([{'record_id':r['record_id'],'text':r['text']} for r in records])!=binding['records_sha256']:
            raise ValueError('annotation texts differ from frozen corpus frame')
        texts[corpus]={r['record_id']:r['text'] for r in records}
    lookup={p['pair_id']:p for p in manifest['complete_frame']};assignments=[];responses=[]
    for sample in manifest['selected_pairs']:
        p=lookup[sample['pair_id']];left=texts[p['corpus']][p['record_id']];right=texts[p['corpus']][p['former_blocker_id']]
        for assignment in sample['assignment_ids']:
            a,b=(right,left) if int(assignment[:2],16)%2 else (left,right)
            assignments.append(dict(assignment_id=assignment,pair_id=p['pair_id'],text_a=a,text_b=b,
                                    task_definition=TASKS[p['corpus']]))
            responses.append(dict(assignment_id=assignment,pair_id=p['pair_id'],annotator_id='',response_id='',
                meaning_relation='',consequential_distinction='',distinction_types=[],task_relevance='',
                skipped=False,skip_reason='',human_completed=False,completed_at=''))
    assignments.sort(key=lambda a:a['assignment_id']);responses.sort(key=lambda a:a['assignment_id'])
    instructions=dict(status='blank_pack_not_dispatched',independence='Three distinct fluent humans judge independently before any adjudication.',
        blinding='No source record IDs, method, arm, gold labels, similarity score or model outcomes are supplied.',
        questions={'meaning_relation':['exact_copy','substantially_same_meaning','overlapping_information','merely_related','unrelated','uncertain'],
                   'consequential_distinction':['yes','no','uncertain'],
                   'distinction_types':['entity','number','negation','time','assertion','other'],
                   'task_relevance':['could_matter','unlikely_to_matter','uncertain']},
        content_warning='Some texts may contain toxic, disturbing or offensive language. Skip any item without providing a substantive rating.',
        consent_withdrawal='Must be finalized before dispatch; no raters recruited or consent assumed.',
        payment_and_time='Pending before dispatch; never inferred or advertised as paid.',
        skipped_items='Preserve all skips and reasons; do not replace with convenient pairs or impute judgments.',
        release='Publish only permitted annotations, stable IDs and short permitted excerpts; not full publisher text.')
    bundle=dict(manifest_sha256=manifest['manifest_sha256'],human_only=False,independent=False,blinded=False,
                responsible_collector='',responses=responses)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    for name,value in [('private_sampling_manifest.json',manifest),('blinded_assignments.json',assignments),
                       ('responses.template.json',bundle),('annotator_instructions.json',instructions)]:
        (directory/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    return dict(directory=str(directory),unique_pairs=len(manifest['selected_pairs']),assignments=len(assignments),
                completed_human_responses=0,human_dispatch_ready=False)
