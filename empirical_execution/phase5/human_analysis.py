"""Validated, missingness-aware analysis of real three-rater admission audits.

Blank packs remain missing. Declaration fields do not authenticate human origin,
independence or collection ethics. No collection, filling or dispatch occurs here.
"""
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math

from phase4.admission_audit import verify_admission_audit, CORPORA
from phase5.context_audit import verify_context_manifest
from phase4.requests import digest

PAIR_SCHEMA='ccu-human-admission-audit-preparation-1'
CONTEXT_SCHEMA='ccu-context-admission-audit-v1'
MEANING=['exact_copy','substantially_same_meaning','overlapping_information','merely_related','unrelated','uncertain']
BINARY=['yes','no','uncertain']
RELEVANCE=['could_matter','unlikely_to_matter','uncertain']
DISTINCTION_TYPES=['entity','number','negation','time','assertion','other']


def _fraction(raw):
    if not isinstance(raw,dict) or set(raw)!={'numerator','denominator'} or type(raw['numerator'])!=int or type(raw['denominator'])!=int or raw['denominator']<=0:
        raise ValueError('exact rational inclusion probability required')
    return Fraction(raw['numerator'],raw['denominator'])


def _timestamp(value):
    if not isinstance(value,str) or not value:raise ValueError('recorded timezone-aware completion time required')
    try:dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError as exc:raise ValueError('invalid ISO completion timestamp') from exc
    if dt.tzinfo is None or dt.utcoffset() is None:raise ValueError('completion timestamp requires timezone')
    return dt


def _frame(manifest):
    """Check seals plus internal sampling/assignment probability closure."""
    ispair=manifest.get('schema')==PAIR_SCHEMA
    if ispair:verify_admission_audit(manifest)
    elif manifest.get('schema')==CONTEXT_SCHEMA:verify_context_manifest(manifest)
    else:raise ValueError('unsupported admission/context manifest')
    key='pair_id' if ispair else 'unit_id'
    selected_key='selected_pairs' if ispair else 'selected_units'
    units={r[key]:copy.deepcopy(r) for r in manifest['complete_frame']}
    if len(units)!=len(manifest['complete_frame']):raise ValueError('duplicate human observational unit')
    selected={r[key]:r for r in manifest[selected_key]}
    if len(selected)!=len(manifest[selected_key]) or not set(selected)<=set(units):raise ValueError('invalid selected unit registry')
    union=set();stratum_probabilities={};context_frame_union=set();context_strata=set()
    for st in manifest['strata']:
        frame_ids=st['frame_pair_ids'] if ispair else st['frame_unit_ids']
        sample_ids=st['selected_pair_ids'] if ispair else st['sample_unit_ids']
        n,N=st['sample_size'],st['population_size']
        if type(n)!=int or type(N)!=int or not 0<=n<=N or N!=len(frame_ids) or n!=len(sample_ids) or len(set(frame_ids))!=N or len(set(sample_ids))!=n or not set(sample_ids)<=set(frame_ids) or not set(frame_ids)<=set(units):raise ValueError('invalid sampling stratum')
        p=_fraction(st['conditional_inclusion_probability'] if ispair else st['inclusion_probability'])
        if p!=(Fraction(n,N) if N else 0):raise ValueError('stratum probability differs from SRS design')
        if ispair:
            sk=tuple(st['stratum'])
            if sk in stratum_probabilities:raise ValueError('duplicate sampling stratum')
            stratum_probabilities[sk]=(p,set(frame_ids),set(sample_ids))
        else:
            cell=(st['corpus'],st['kind'])
            if cell in context_strata or context_frame_union.intersection(frame_ids):raise ValueError('duplicate context stratum/unit')
            context_strata.add(cell);context_frame_union.update(frame_ids)
            for item in frame_ids:
                if units[item]['corpus']!=st['corpus'] or units[item]['kind']!=st['kind'] or _fraction(units[item]['inclusion_probability'])!=p:raise ValueError('context frame probability/stratum mismatch')
                if units[item].get('selected_for_annotation')!=(item in sample_ids):raise ValueError('context selected flag mismatch')
        union.update(sample_ids)
    if union!=set(selected):raise ValueError('selected assignment union differs from sampled strata')
    if not ispair and context_frame_union!=set(units):raise ValueError('context frame stratum coverage incomplete')
    assignments={};probabilities={}
    for item,u in units.items():
        if u.get('corpus') not in CORPORA:raise ValueError('unit corpus outside prescribed frame')
        if ispair:
            if not u.get('roles') or len(set(u['roles']))!=len(u['roles']) or not set(u['roles'])<={'R/S','U/A','control'} or ('control' in u['roles'] and len(u['roles'])!=1):raise ValueError('invalid admitted/control role')
            if any(len(sk)!=4 or sk[0]!=u['corpus'] or sk[1] not in u['roles'] for sk in u['strata']) or {sk[1] for sk in u['strata']}!=set(u['roles']):raise ValueError('pair roles and sampling strata disagree')
            nonselection=Fraction(1)
            if len({tuple(s) for s in u['strata']})!=len(u['strata']):raise ValueError('duplicate unit stratum membership')
            actual={sk for sk,(_,frame_ids,_) in stratum_probabilities.items() if item in frame_ids}
            if actual!={tuple(s) for s in u['strata']}:raise ValueError('unit/stratum frame incidence mismatch')
            for sk in actual:nonselection*=1-stratum_probabilities[sk][0]
            p=1-nonselection
            if p!=_fraction(u['conditional_union_inclusion_probability']):raise ValueError('pair union probability inconsistent')
            if item in selected:
                if p!=_fraction(selected[item]['conditional_union_inclusion_probability']):raise ValueError('sample union probability inconsistent')
                routes={sk for sk,(_,_,chosen) in stratum_probabilities.items() if item in chosen}
                if {tuple(sk) for sk in selected[item]['selected_in_strata']}!=routes:raise ValueError('sampled pair route metadata disagree')
        else:p=_fraction(u['inclusion_probability'])
        if not 0<=p<=1 or (item in selected and p==0):raise ValueError('invalid unit inclusion probability')
        probabilities[item]=p
        if item in selected:
            aids=selected[item]['assignment_ids']
            if len(aids)!=3 or len(set(aids))!=3 or any(not isinstance(a,str) or not a for a in aids):raise ValueError('three unique assignment slots required')
            for aid in aids:
                if aid in assignments:raise ValueError('assignment reused across observational units')
                assignments[aid]=item
    return ispair,key,units,selected,assignments,probabilities


def validate_responses(manifest,bundle,*,software_fixture_only=False):
    ispair,key,units,selected,assignments,probabilities=_frame(manifest)
    if not isinstance(bundle,dict) or bundle.get('manifest_sha256')!=manifest['manifest_sha256'] or not isinstance(bundle.get('responses'),list):raise ValueError('response bundle must bind the frozen sampling manifest')
    if bundle.get('evidence_role')=='software_fixture_only' and not software_fixture_only:raise ValueError('software ratings cannot enter human analysis')
    if software_fixture_only and bundle.get('evidence_role')!='software_fixture_only':raise ValueError('fixture analysis requires explicit fixture bundle')
    questions={('meaning_relation' if ispair else 'distinct_information'):(MEANING if ispair else BINARY),
               'consequential_distinction':BINARY,'task_relevance':RELEVANCE}
    supplied={};response_ids=set();humans=defaultdict(set);completed_times=defaultdict(list)
    required={'assignment_id',key,'annotator_id','response_id',*questions,'distinction_types','skipped','skip_reason','human_completed','completed_at'}
    for response in bundle['responses']:
        if not isinstance(response,dict) or set(response)!=required:raise ValueError('response fields differ from frozen assignment schema')
        aid=response['assignment_id']
        if aid not in assignments or aid in supplied or response[key]!=assignments[aid]:raise ValueError('unknown, repeated or misbound assignment')
        if type(response['human_completed'])!=bool or type(response['skipped'])!=bool:raise ValueError('boolean completion/skip flags required')
        if not isinstance(response['distinction_types'],list) or len(set(response['distinction_types']))!=len(response['distinction_types']) or not set(response['distinction_types'])<=set(DISTINCTION_TYPES):raise ValueError('invalid distinction types')
        if not response['human_completed']:
            if response['skipped'] or response['distinction_types'] or any(response[f]!='' for f in [*questions,'annotator_id','response_id','skip_reason','completed_at']):raise ValueError('incomplete rows must be untouched blanks, not partial implied ratings')
            supplied[aid]=copy.deepcopy(response);continue
        if not software_fixture_only and (not all(bundle.get(k) is True for k in ('human_only','independent','blinded')) or not isinstance(bundle.get('responsible_collector'),str) or not bundle['responsible_collector'].strip()):raise ValueError('completed human bundle needs collector and human/independence/blinding declarations')
        rater,rid=response['annotator_id'],response['response_id'];item=assignments[aid]
        if not isinstance(rater,str) or not rater.strip() or rater in humans[item] or not isinstance(rid,str) or not rid.strip() or rid in response_ids:raise ValueError('distinct raters per item and unique response IDs required')
        humans[item].add(rater);response_ids.add(rid);completed_times[item].append(_timestamp(response['completed_at']))
        if response['skipped']:
            if any(response[q]!='' for q in questions) or response['distinction_types'] or not isinstance(response['skip_reason'],str) or not response['skip_reason'].strip():raise ValueError('skip requires reason and no substantive ratings')
        else:
            if response['skip_reason']!='' or any(response[q] not in choices for q,choices in questions.items()):raise ValueError('invalid substantive category or skip reason')
            if response['consequential_distinction']=='no' and response['distinction_types']:raise ValueError('no distinction cannot specify distinction types')
            if response['consequential_distinction']=='yes' and not response['distinction_types']:raise ValueError('yes distinction requires at least one type')
            if not ispair and units[item]['missing_comparison']:raise ValueError('unavailable comparator requires skip, never novelty judgment')
        supplied[aid]=copy.deepcopy(response)
    rows=[];outcomes={};agreement={}
    for item,sample in selected.items():
        actual=[supplied.get(a) for a in sample['assignment_ids']]
        rated=[r for r in actual if r is not None and r['human_completed'] and not r['skipped']]
        complete=len(rated)==3
        majority={};detail={}
        for q,choices in questions.items():
            tally=Counter(r[q] for r in rated)
            majority[q]=(next((c for c,n in tally.items() if n>=2),'no_majority') if complete else None)
            detail[q]={'rating_counts':dict(tally),'pairwise_agreement':sum(n*(n-1) for n in tally.values())/6 if complete else None,
                       'uncertain_rating_fraction':tally['uncertain']/3 if complete else None}
        for kind in DISTINCTION_TYPES:
            majority['distinction_type:'+kind]=float(sum(kind in r['distinction_types'] for r in rated)>=2) if complete else None
        outcomes[item]=majority;agreement[item]=detail
        rows.append({'item_id':item,'expected_assignments':3,'present_response_rows':sum(r is not None for r in actual),
                     'completed_assignments':sum(r is not None and r['human_completed'] for r in actual),
                     'skipped_assignments':sum(r is not None and r['human_completed'] and r['skipped'] for r in actual),
                     'substantive_assignments':len(rated),'complete_three_rater_unit':complete,
                     'original_majority':majority,'original_agreement':detail})
    return {'is_pair':ispair,'questions':questions,'units':units,'selected':selected,'probabilities':probabilities,
            'supplied':supplied,'outcomes':outcomes,'agreement':agreement,'unit_results':rows,
            'completed_times':completed_times,'response_bundle_sha256':digest(bundle),
            'completed_assignment_count':len(response_ids),'actual_human_response_count':0 if software_fixture_only else len(response_ids),
            'schema_acceptance_authenticates_humans':False}


def _weighted_binary(domain,selected,probabilities,values):
    """HT and Hájek estimators; missingness envelopes are not sampling CIs."""
    N=len(domain);sample=[i for i in domain if i in selected]
    if any(v is not None and (not math.isfinite(v) or not 0<=v<=1) for v in values.values()):raise ValueError('binary/rate outcomes must be finite in [0,1]')
    weights={i:1/probabilities[i] for i in sample}
    known=[i for i in sample if values.get(i) is not None]
    unknown=[i for i in sample if values.get(i) is None]
    W=sum(weights.values(),Fraction(0));known_w=sum((weights[i] for i in known),Fraction(0))
    yes=sum((weights[i]*Fraction(str(values[i])) for i in known),Fraction(0));missing_w=W-known_w
    complete=bool(N) and len(known)==len(sample) and all(probabilities[i]>0 for i in domain) and bool(sample)
    observations=sum(float(values[i]) for i in known)
    return {'frame_units':N,'sampled_units':len(sample),'observed_units':len(known),'missing_sampled_units':len(unknown),
            'sample_weight_total':float(W),'observed_weight_total':float(known_w),
            'ht_prevalence':float(yes/N) if complete else None,
            'hajek_prevalence':float(yes/W) if complete and W else None,
            'observed_only_hajek_descriptive':float(yes/known_w) if known_w else None,
            'weighted_missing_fraction':float(missing_w/W) if W else None,
            'hajek_missing_outcome_extremes':[float(yes/W),float((yes+missing_w)/W)] if W else None,
            'ht_missing_outcome_extremes':[float(yes/N),float((yes+missing_w)/N)] if N and W else None,
            'finite_frame_bounded_outcome_identification_bounds':[observations/N,(observations+N-len(known))/N] if N else None,
            'sampling_confidence_interval':None,'intervals_above_are_sampling_confidence_intervals':False,
            'zero_frame_prevalence_undefined':N==0,'unobserved_unsampled_units':N-len(sample)}


def _domains(ispair,units):
    result={}
    for corpus in CORPORA:
        roles=['admission_union','R/S','U/A','control'] if ispair else ['former_neighborhood','nearest_surviving_selected']
        for role in roles:
            result[(corpus,role)]=[i for i,u in units.items() if u['corpus']==corpus and
                ((role=='admission_union' and bool(set(u['roles'])&{'R/S','U/A'})) or (role in u['roles']))] if ispair else [i for i,u in units.items() if u['corpus']==corpus and u['kind']==role]
    return result


def _summaries(data,outcomes):
    result=[]
    status={r['item_id']:r for r in data['unit_results']}
    for (corpus,role),domain in _domains(data['is_pair'],data['units']).items():
        sample=[i for i in domain if i in data['selected']]
        scopes=['pair'] if data['is_pair'] else ['complete_original_context','displayed_context_only']
        for scope in scopes:
            allowed={i for i in sample if data['is_pair'] or (not data['units'][i]['missing_comparison'] and (scope=='displayed_context_only' or data['units'][i]['complete_context_displayed']))}
            category={};agreements={}
            for q,choices in data['questions'].items():
                category[q]={}
                for choice in [*choices,'no_majority']:
                    values={i:float(outcomes[i][q]==choice) if i in allowed and outcomes[i][q] is not None else None for i in sample}
                    category[q][choice]=_weighted_binary(domain,data['selected'],data['probabilities'],values)
                complete=[i for i in sample if i in allowed and data['agreement'][i][q]['pairwise_agreement'] is not None]
                weights=[1/data['probabilities'][i] for i in complete];total=sum(weights,Fraction(0))
                observed_agreement=sum((w*Fraction(sum(n*(n-1) for n in data['agreement'][i][q]['rating_counts'].values()),6) for i,w in zip(complete,weights)),Fraction(0))/total if total else None
                margins={c:sum((w*Fraction(data['agreement'][i][q]['rating_counts'].get(c,0),3) for i,w in zip(complete,weights)),Fraction(0))/total if total else None for c in choices}
                chance=sum((v*v for v in margins.values()),Fraction(0)) if total else None
                agreements[q]={'complete_units':len(complete),'weighted_pairwise_agreement':float(observed_agreement) if total else None,
                  'weighted_fleiss_kappa':float((observed_agreement-chance)/(1-chance)) if total and chance!=1 else None,
                  'weighted_uncertain_rating_fraction':float(margins['uncertain']) if total else None,
                  'scope':'complete three-substantive-rating units only; missingness not corrected by sampling weights'}
            types={k:_weighted_binary(domain,data['selected'],data['probabilities'],
                    {i:outcomes[i]['distinction_type:'+k] if i in allowed else None for i in sample}) for k in DISTINCTION_TYPES}
            missingness={}
            for label,values in {
                'three_substantive_ratings':{i:float(status[i]['complete_three_rater_unit']) for i in sample},
                'any_skip':{i:float(status[i]['skipped_assignments']>0) for i in sample},
                'assignment_completion_fraction':{i:status[i]['completed_assignments']/3 for i in sample},
                'assignment_skip_fraction':{i:status[i]['skipped_assignments']/3 for i in sample},
                'incomplete_context':{i:float(not data['units'][i]['complete_context_displayed']) for i in sample} if not data['is_pair'] else {},
                'missing_comparison':{i:float(data['units'][i]['missing_comparison']) for i in sample} if not data['is_pair'] else {}}.items():
                if values:missingness[label]=_weighted_binary(domain,data['selected'],data['probabilities'],values)
            result.append({'corpus':corpus,'population':role,'context_scope':scope,
                           'categories':category,'distinction_types':types,'original_rater_agreement':agreements,'collection_rates':missingness})
    return result


def _adjudicate(data,manifest,bundle,adjudication):
    if adjudication is None:return None
    if not isinstance(adjudication,dict) or adjudication.get('manifest_sha256')!=manifest['manifest_sha256'] or adjudication.get('response_bundle_sha256')!=digest(bundle):raise ValueError('adjudication must bind exact preserved original responses')
    policy=adjudication.get('policy',{})
    if not isinstance(policy,dict) or not all(isinstance(policy.get(k),str) and policy[k].strip() for k in ('policy_id','policy_text')) or policy.get('fixed_before_adjudication') is not True or not isinstance(adjudication.get('decisions'),list):raise ValueError('explicit separately recorded adjudication policy required')
    outcomes=copy.deepcopy(data['outcomes']);seen=set();status={r['item_id']:r for r in data['unit_results']}
    for row in adjudication['decisions']:
        item=row.get('item_id')
        if item not in outcomes or item in seen or not status[item]['complete_three_rater_unit']:raise ValueError('adjudication requires complete original independent judgments, once per item')
        if not isinstance(row.get('adjudicator_id'),str) or not row['adjudicator_id'].strip() or not isinstance(row.get('reason'),str) or not row['reason'].strip():raise ValueError('adjudicator identity and rationale required')
        if _timestamp(row.get('completed_at'))<max(data['completed_times'][item]):raise ValueError('adjudication preceded an original rating')
        if not isinstance(row.get('answers'),dict) or set(row['answers'])!={*data['questions'],'distinction_types'}:raise ValueError('complete separate adjudicated answers required')
        for q,choices in data['questions'].items():
            if row['answers'][q] not in choices:raise ValueError('invalid adjudicated category')
            outcomes[item][q]=row['answers'][q]
        types=row['answers']['distinction_types']
        if not isinstance(types,list) or len(set(types))!=len(types) or not set(types)<=set(DISTINCTION_TYPES):raise ValueError('invalid adjudicated distinction types')
        if (row['answers']['consequential_distinction']=='yes' and not types) or (row['answers']['consequential_distinction']=='no' and types):raise ValueError('inconsistent adjudicated distinction types')
        for kind in DISTINCTION_TYPES:outcomes[item]['distinction_type:'+kind]=float(kind in types)
        seen.add(item)
    return {'policy':copy.deepcopy(policy),'adjudication_sha256':digest(adjudication),'decisions':copy.deepcopy(adjudication['decisions']),
            'original_ratings_replaced':False,'supplementary_estimator_rule':'use adjudication only on declared resolved units; retain original majority elsewhere',
            'supplementary_summaries':_summaries(data,outcomes)}


def analyze_human_audit(manifest,bundle,*,adjudication=None,software_fixture_only=False):
    data=validate_responses(manifest,bundle,software_fixture_only=software_fixture_only)
    summary=_summaries(data,data['outcomes'])
    # Only a complete, explicitly balanced three-corpus target may receive this
    # average. Missing corpus frames are not silently reweighted away.
    balanced=[];by={(s['corpus'],s['population'],s['context_scope']):s for s in summary}
    for role,scope in sorted({(s['population'],s['context_scope']) for s in summary}):
        cat={}
        for q,choices in data['questions'].items():
            cat[q]={}
            for c in [*choices,'no_majority']:
                values=[by[corpus,role,scope]['categories'][q][c]['ht_prevalence'] for corpus in CORPORA]
                cat[q][c]=sum(values)/3 if all(v is not None for v in values) else None
        balanced.append({'population':role,'context_scope':scope,'target':'equal one-third weight for each prespecified corpus',
                         'ht_category_estimates':cat,'missing_corpora_not_reweighted':True})
    return {'schema':'ccu-human-admission-analysis-1','manifest_sha256':manifest['manifest_sha256'],
      'response_bundle_sha256':data['response_bundle_sha256'],'evidence_role':'software_fixture_only' if software_fixture_only else manifest['evidence_role'],
      'actual_human_response_count':data['actual_human_response_count'],
      'fixture_completed_response_rows':data['completed_assignment_count'] if software_fixture_only else 0,
      'declared_completed_assignment_count':data['completed_assignment_count'],'expected_assignment_count':3*len(data['selected']),
      'observational_unit':'unique sampled record/pair or context item; repeated request occurrences not independent ratings',
      'independent_majority_rule':'three substantive original ratings required; category count >=2 else no_majority; uncertain is preserved',
      'estimand_conditioning':'frozen unique-unit frame and realized blocker/context choices; sampling weights do not fix skip bias',
      'source_and_human_authenticity_verified':False,'confirmatory_claims_established':False,
      'unit_results':data['unit_results'],'original_summaries':summary,'equal_corpus_balanced':balanced,
      'adjudication':_adjudicate(data,manifest,bundle,adjudication),
      'original_response_bundle':copy.deepcopy(bundle),
      'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}



def lock_illustration_policy(manifest,categories,*,seed=20271003,per_category=1):
    """Lock outcome-free example categories; actual preoutcome timing is external."""
    ispair,_,units,_,_,_=_frame(manifest)
    if type(seed)!=int or seed<0 or type(per_category)!=int or per_category<1:
        raise ValueError('nonnegative seed and positive example count required')
    questions={('meaning_relation' if ispair else 'distinct_information'):(MEANING if ispair else BINARY),
               'consequential_distinction':BINARY,'task_relevance':RELEVANCE}
    domains=_domains(ispair,units);keys=set()
    for category in categories:
        if not isinstance(category,dict) or set(category)!={'corpus','population','context_scope','question','category'}:
            raise ValueError('complete prospective example category required')
        if (category['corpus'],category['population']) not in domains or category['question'] not in questions or category['category'] not in questions[category['question']]+['no_majority']:
            raise ValueError('unknown example domain/category')
        allowed=['pair'] if ispair else ['complete_original_context','displayed_context_only']
        if category['context_scope'] not in allowed:raise ValueError('unknown example context scope')
        k=digest(category)
        if k in keys:raise ValueError('repeated example category')
        keys.add(k)
    if not categories:raise ValueError('at least one prospective example category required')
    policy={'schema':'ccu-human-illustration-policy-1','manifest_sha256':manifest['manifest_sha256'],
      'seed':seed,'per_category':per_category,'categories':copy.deepcopy(categories),
      'selection_rule':'first eligible original-majority units by seeded stable-ID hash; no task-loss/effect ranking',
      'external_preoutcome_timing_verified':False}
    policy['sha256']=digest(policy);return policy


def select_illustrative_ids(manifest,bundle,policy,*,software_fixture_only=False):
    """Return IDs only, without publisher excerpts or fabricated interpretations."""
    if policy.get('sha256')!=digest({k:v for k,v in policy.items() if k!='sha256'}) or policy.get('manifest_sha256')!=manifest['manifest_sha256']:
        raise ValueError('illustration policy seal/manifest mismatch')
    expected=lock_illustration_policy(manifest,policy['categories'],seed=policy['seed'],per_category=policy['per_category'])
    if expected!=policy:raise ValueError('illustration policy schema/content mismatch')
    data=validate_responses(manifest,bundle,software_fixture_only=software_fixture_only)
    domains=_domains(data['is_pair'],data['units']);output=[]
    for category in policy['categories']:
        candidates=[]
        for item in domains[category['corpus'],category['population']]:
            if item not in data['outcomes'] or data['outcomes'][item][category['question']]!=category['category']:continue
            unit=data['units'][item]
            if not data['is_pair'] and (unit['missing_comparison'] or (category['context_scope']=='complete_original_context' and not unit['complete_context_displayed'])):continue
            candidates.append(item)
        candidates.sort(key=lambda item:(digest(['human-illustration-v1',policy['seed'],manifest['manifest_sha256'],category,item]),item))
        output.append({'category':copy.deepcopy(category),'eligible_units':len(candidates),
                       'selected_item_ids':candidates[:policy['per_category']],
                       'shortfall':max(0,policy['per_category']-len(candidates))})
    return {'schema':'ccu-human-illustration-ids-1','policy_sha256':policy['sha256'],
      'response_bundle_sha256':digest(bundle),'role':'software_fixture_only' if software_fixture_only else 'permitted_id_export_only',
      'publisher_text_exported':False,'examples':output,'same_unit_across_categories_not_independent':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--responses',required=True,type=Path)
    p.add_argument('--adjudication',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('analysis output exists; preserve previous release')
    result=analyze_human_audit(json.loads(a.manifest.read_text()),json.loads(a.responses.read_text()),
       adjudication=json.loads(a.adjudication.read_text()) if a.adjudication else None)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(a.output),'actual_human_response_count':result['actual_human_response_count']}))

if __name__=='__main__':main()
