"""Two independently sampled, BLANK, contextual admission audits.

Complete private frames and deterministic context limitations are preserved.
No human judgments, provenance authenticity, annotation dispatch or global
semantic novelty is inferred from this preparation workflow.
"""
from pathlib import Path
from fractions import Fraction
import hashlib,json,random
import numpy as np
from phase3.panels import graph_normalize,reference_cosines
from phase4.requests import digest,derive_seed,_graph_state
from phase4.admission_audit import _one_corpus_frame,TASKS,CORPORA
from phase5.convex_multioutput import retained_selection

SCHEMA='ccu-context-admission-audit-v1'
KINDS=('former_neighborhood','nearest_surviving_selected')
QUOTAS={'civil_comments':34,'askubuntu':33,'cc_news':33}

def _fraction(value):return dict(numerator=value.numerator,denominator=value.denominator)

def _positive_int(v,name):
    if type(v)!=int or v<1:raise ValueError(name+' must be a positive integer')
    return v

def exact_nearest(features,record_ids,query_id,candidate_ids):
    """Shared finite-precision scorer, exhaustive candidates, stable ID score ties."""
    ids=list(record_ids);lookup={r:i for i,r in enumerate(ids)}
    if len(lookup)!=len(ids) or query_id not in lookup:raise ValueError('unique aligned IDs/query required')
    candidates=sorted(set(candidate_ids))
    if len(candidates)!=len(candidate_ids) or any(r not in lookup or r==query_id for r in candidates):
        raise ValueError('unique known candidates excluding the query required')
    x=np.asarray(features)
    if x.dtype!=np.float32 or x.ndim!=2 or len(x)!=len(ids):raise ValueError('aligned FP32 features required')
    if not candidates:return None,None
    # Do not allocate an all-candidate score matrix or rely on BLAS tie differences.
    query=graph_normalize(x[lookup[query_id]:lookup[query_id]+1]);best=None;best_score=-float('inf')
    for rid in candidates:
        i=lookup[rid];score=float(reference_cosines(query,graph_normalize(x[i:i+1]))[0,0])
        if score>best_score:best,best_score=rid,score
    return best,best_score

def prepare_context_audits(corpus_inputs,*,master_seed=20271003,
                          evidence_role='engineering_nonconfirmatory',
                          target_character_limit=8000,context_character_limit=20000):
    if type(master_seed)!=int or master_seed<0:raise ValueError('nonnegative integer seed required')
    if not isinstance(corpus_inputs,dict) or set(corpus_inputs)-set(CORPORA):raise ValueError('prespecified corpora only')
    if evidence_role not in {'engineering_nonconfirmatory','prospective_confirmatory_preparation'}:
        raise ValueError('explicit evidence role required')
    _positive_int(target_character_limit,'target_character_limit');_positive_int(context_character_limit,'context_character_limit')
    complete=[];bindings={};sample=[];strata=[];status={};checkpoint_states={}
    for corpus in CORPORA:
        if corpus not in corpus_inputs:
            status[corpus]='missing_corpus_no_quota_transfer'
            for kind in KINDS:strata.append(dict(corpus=corpus,kind=kind,quota=QUOTAS[corpus],population_size=0,
              sample_size=0,shortfall=QUOTAS[corpus],frame_unit_ids=[],sample_unit_ids=[],inclusion_probability=_fraction(Fraction(0))))
            continue
        inputs=corpus_inputs[corpus];frame,binding=_one_corpus_frame(corpus,inputs,master_seed)
        if evidence_role=='prospective_confirmatory_preparation' and binding['provenance']['evidence_role']!=evidence_role:
            raise ValueError('engineering inputs cannot be promoted to confirmatory preparation')
        bindings[corpus]=binding;status[corpus]='declared_frame_prepared_not_authenticated'
        records=inputs['records'];texts={r['record_id']:r['text'] for r in records};ids=[r['record_id'] for r in records]
        _,owners,blockers,_=_graph_state(inputs['graph'])
        admissions=sorted((p for p in frame if p['roles']!=['control']),key=lambda p:p['record_id'])
        for kind in KINDS:
            pop=sorted(digest(['context-unit-v1',corpus,kind,p['record_id']]) for p in admissions)
            size=min(QUOTAS[corpus],len(pop));seed=derive_seed(master_seed,'context-subset-v1',corpus,kind)
            chosen=sorted(random.Random(seed).sample(pop,size));prob=Fraction(size,len(pop)) if pop else Fraction(0)
            units=[]
            for pair in admissions:
                rid=pair['record_id'];unit_id=digest(['context-unit-v1',corpus,kind,rid])
                events=sorted(pair['occurrences'],key=lambda e:(e['trajectory_id'],e['checkpoint']))
                event_seed=derive_seed(master_seed,'context-occurrence-v1',corpus,kind,rid)
                event=random.Random(event_seed).choice(events)
                common=dict(unit_id=unit_id,corpus=corpus,kind=kind,record_id=rid,
                  occurrences=events,occurrence_multiplicity=len(events),chosen_occurrence=event,
                  occurrence_choice_seed=event_seed,conditional_occurrence_probability=_fraction(Fraction(1,len(events))))
                if unit_id not in chosen:
                    units.append(dict(**common,context_prepared=False));continue
                checkpoint_key=digest([corpus,event['trajectory_id'],event['checkpoint']])
                if checkpoint_key not in checkpoint_states:
                    selected,deleted=retained_selection(inputs['graph'],inputs['request_manifest'],event['trajectory_id'],event['checkpoint'])
                    checkpoint_states[checkpoint_key]=dict(corpus=corpus,trajectory_id=event['trajectory_id'],checkpoint=event['checkpoint'],
                      retained_selected_ids=selected,deleted_record_ids=deleted)
                state=checkpoint_states[checkpoint_key];selected=state['retained_selected_ids'];deleted=state['deleted_record_ids']
                if rid not in selected or rid in deleted or not blockers[rid]<=set(deleted):raise AssertionError('invalid admitted checkpoint')
                candidate_ids=sorted(set(selected)-{rid})
                nearest,score=(None,None) if kind=='former_neighborhood' else exact_nearest(inputs['features'],ids,rid,candidate_ids)
                full_context=sorted(blockers[rid]) if kind=='former_neighborhood' else ([] if nearest is None else [nearest])
                # Context order is independent of semantic scores, labels and eventual responses.
                display_order=sorted(full_context,key=lambda r:(digest(['context-order-v1',corpus,rid,r]),r))
                remaining=context_character_limit;shown=[];omitted=[]
                for context_id in display_order:
                    count=min(len(texts[context_id]),remaining)
                    if count:shown.append(dict(record_id=context_id,displayed_characters=count,total_characters=len(texts[context_id])))
                    if count<len(texts[context_id]):omitted.append(context_id)
                    remaining-=count
                units.append(dict(**common,context_prepared=True,checkpoint_state_key=checkpoint_key,
                  complete_former_blocker_ids=sorted(blockers[rid]),
                  nearest_candidate_definition='checkpoint retained_selected_ids minus query' if kind=='nearest_surviving_selected' else None,
                  nearest_candidate_count=len(candidate_ids) if kind=='nearest_surviving_selected' else None,
                  nearest_record_id=nearest,nearest_reference_score=score,
                  complete_context_ids=full_context,display_order=display_order,displayed_context=shown,
                  omitted_or_truncated_context_ids=omitted,
                  target_total_characters=len(texts[rid]),target_displayed_characters=min(len(texts[rid]),target_character_limit),
                  complete_context_displayed=not omitted and len(texts[rid])<=target_character_limit,
                  missing_comparison=kind=='nearest_surviving_selected' and nearest is None))
            strata.append(dict(corpus=corpus,kind=kind,quota=QUOTAS[corpus],population_size=len(pop),sample_size=size,
              shortfall=QUOTAS[corpus]-size,sampling_seed=seed,frame_unit_ids=pop,sample_unit_ids=chosen,
              inclusion_probability=_fraction(prob)))
            for unit in units:
                unit['inclusion_probability']=_fraction(prob)
                unit['horvitz_thompson_weight']=_fraction(1/prob) if prob else None
                unit['selected_for_annotation']=unit['unit_id'] in chosen
            complete.extend(units)
            sample.extend(chosen)
    complete.sort(key=lambda p:p['unit_id']);config=dict(master_seed=master_seed,quotas_per_kind=QUOTAS,
      subset_independence='separate independent SRSWOR streams by corpus and kind; overlap is retained',
      record_frame='all unique admitted records in complete logged core R/S/U/A checkpoint union',
      checkpoint_rule='one uniformly seeded actual occurrence independently fixed per record and audit kind',
      nearest_rule='all surviving selected records at chosen checkpoint EXCLUDING query; largest reference cosine, stable ID tie',
      target_character_limit=target_character_limit,context_character_limit=context_character_limit,
      context_limit_units='Unicode code points, not model tokens',
      missing_or_scarce='no refill, no other-corpus transfer, no replacement for truncation, empty comparison, skip or uncertainty')
    assignment_context=digest(dict(bindings=bindings,complete_frame=complete,checkpoint_states=checkpoint_states,configuration=config))
    assignments=[dict(unit_id=unit,assignment_ids=[digest(['context-assignment-v1',assignment_context,unit,i]) for i in range(3)]) for unit in sorted(sample)]
    result=dict(schema=SCHEMA,evidence_role=evidence_role,configuration=config,corpus_status=status,
      input_bindings=bindings,complete_frame=complete,complete_frame_sha256=digest(complete),strata=strata,
      checkpoint_states=checkpoint_states,
      selected_units=assignments,blank_assignment_count=3*len(sample),completed_human_responses=0,
      human_dispatch_ready=False,confirmatory_study_ready=False,source_authenticity_verified=False,
      code_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),Path(__file__).with_name('convex_multioutput.py'),
                  Path(__file__).resolve().parents[1]/'phase4/admission_audit.py',
                  Path(__file__).resolve().parents[1]/'phase4/requests.py',
                  Path(__file__).resolve().parents[1]/'phase3/panels.py']})
    result['manifest_sha256']=digest(result);return result

def verify_context_manifest(manifest):
    if manifest.get('schema')!=SCHEMA or manifest.get('manifest_sha256')!=digest({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
        raise ValueError('context manifest seal/schema mismatch')
    if digest(manifest['complete_frame'])!=manifest['complete_frame_sha256']:raise ValueError('complete frame changed')

def write_context_pack(manifest,records_by_corpus,directory):
    verify_context_manifest(manifest);texts={}
    for corpus,binding in manifest['input_bindings'].items():
        rows=records_by_corpus.get(corpus)
        if not isinstance(rows,list) or digest([dict(record_id=r['record_id'],text=r['text']) for r in rows])!=binding['records_sha256']:
            raise ValueError('original text frame binding mismatch')
        texts[corpus]={r['record_id']:r['text'] for r in rows}
    units={u['unit_id']:u for u in manifest['complete_frame']};assignments=[];responses=[]
    for chosen in manifest['selected_units']:
        unit=units[chosen['unit_id']];corpus=unit['corpus'];text=texts[corpus]
        for aid in chosen['assignment_ids']:
            assignments.append(dict(assignment_id=aid,unit_id=unit['unit_id'],
              target_text=text[unit['record_id']][:unit['target_displayed_characters']],
              comparison_texts=[text[s['record_id']][:s['displayed_characters']] for s in unit['displayed_context']],
              task_definition=TASKS[corpus],comparison_question=(
                'Does the target add information beyond the comparison texts considered together?' if unit['kind']=='former_neighborhood' else
                'Does the target contain distinct information relative to this comparison text?'),
              context_complete=unit['complete_context_displayed'],comparison_available=not unit['missing_comparison']))
            responses.append(dict(assignment_id=aid,unit_id=unit['unit_id'],annotator_id='',response_id='',
              distinct_information='',consequential_distinction='',distinction_types=[],task_relevance='',
              skipped=False,skip_reason='',human_completed=False,completed_at=''))
    instructions=dict(status='blank_local_preparation_not_dispatched',
      independence='Three distinct fluent humans independently; no model-generated answers or reused pair judgments.',
      blinding='Methods, arm, scores, labels, outcomes, source/record IDs and sampling weights are hidden.',
      questions=dict(distinct_information=['yes','no','uncertain'],consequential_distinction=['yes','no','uncertain'],
        distinction_types=['entity','number','negation','time','assertion','other'],task_relevance=['could_matter','unlikely_to_matter','uncertain']),
      context_limits='Character limits fixed in manifest. Context-complete=false means the original neighborhood/target is not fully shown. Judge only displayed information.',
      empty_comparison='If comparison_available=false, mark skipped and explain unavailable comparison; never infer novelty.',
      skip_policy='Offensive/toxic/disturbing text may be skipped. Preserve skips, uncertainty, missingness and truncation. Never refill or impute.',
      missingness_analysis='Report weighted response/skip/truncation rates by audit/corpus. Full-frame novelty rates require explicit missing-outcome bounds, not complete-case claims.',
      interpretation='Former neighborhood is not all prior selected data. Nearest selected is one current text and excludes the target; neither proves global novelty.',
      before_dispatch=['recruitment and fluency','fair payment and duration','consent and withdrawal','permission for protected text','content protections'],
      release='Do not redistribute publisher article bodies.')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    payloads={'private_sampling_manifest.json':manifest,'blinded_assignments.json':assignments,
      'responses.template.json':dict(manifest_sha256=manifest['manifest_sha256'],human_only=False,independent=False,
        blinded=False,responsible_collector='',responses=responses),'annotator_instructions.json':instructions}
    for name,value in payloads.items():(directory/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    return dict(selected_units=len(manifest['selected_units']),blank_assignments=len(assignments),completed_human_responses=0)
