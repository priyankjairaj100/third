#!/usr/bin/env python3
"""Schema/algebra rating fixtures plus genuine blank-pack missingness checks.

All filled response values below are in-memory SOFTWARE fixtures, never stored
as human observations or used to fill the real prepared annotation packs.
"""
from pathlib import Path
import sys,json,copy,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase4.requests import digest
from phase5.human_analysis import analyze_human_audit,validate_responses,lock_illustration_policy,select_illustrative_ids,PAIR_SCHEMA


def fraction(n,d):return {'numerator':n,'denominator':d}

def fixture():
    st0=['civil_comments','R/S',0,'equal'];st1=['civil_comments','U/A',0,'equal']
    frame=[{'pair_id':f'fixture-u{i}','corpus':'civil_comments','roles':['R/S','U/A'],
            'strata':[st0,st1],'conditional_union_inclusion_probability':fraction(3,4)} for i in range(2)]
    strata=[{'stratum':st,'frame_pair_ids':['fixture-u0','fixture-u1'],'selected_pair_ids':['fixture-u0'],
             'population_size':2,'sample_size':1,'conditional_inclusion_probability':fraction(1,2)} for st in [st0,st1]]
    m={'schema':PAIR_SCHEMA,'evidence_role':'engineering_nonconfirmatory','configuration':{'fixture':'algebra only'},
       'complete_frame':frame,'strata':strata,'selected_pairs':[{'pair_id':'fixture-u0',
       'selected_in_strata':[st0,st1],'conditional_union_inclusion_probability':fraction(3,4),
       'assignment_ids':['fixture-a0','fixture-a1','fixture-a2']}]}
    m['configuration_sha256']=digest(m['configuration']);m['complete_frame_sha256']=digest(frame);m['manifest_sha256']=digest(m)
    responses=[{'assignment_id':f'fixture-a{i}','pair_id':'fixture-u0','annotator_id':f'fixture-rater{i}',
       'response_id':f'fixture-response{i}','meaning_relation':'exact_copy','consequential_distinction':'yes',
       'distinction_types':['entity'],'task_relevance':'could_matter','skipped':False,'skip_reason':'',
       'human_completed':True,'completed_at':'2026-01-01T12:00:00+00:00'} for i in range(3)]
    b={'manifest_sha256':m['manifest_sha256'],'evidence_role':'software_fixture_only','human_only':False,
       'independent':False,'blinded':False,'responsible_collector':'SOFTWARE FIXTURE ONLY','responses':responses}
    return m,b


def primary(summary):
    return next(s for s in summary['original_summaries'] if s['corpus']=='civil_comments' and s['population']=='admission_union')


def main():
    tests={};m,b=fixture();r=analyze_human_audit(m,b,software_fixture_only=True)
    rate=primary(r)['categories']['meaning_relation']['exact_copy']
    tests['overlap_union_pi_weight_exact']=abs(rate['ht_prevalence']-2/3)<1e-15 and rate['hajek_prevalence']==1
    tests['sample_vs_frame_bounds_distinguished']=rate['finite_frame_bounded_outcome_identification_bounds']==[.5,1.] and rate['sampling_confidence_interval'] is None
    illustration=lock_illustration_policy(m,[{'corpus':'civil_comments','population':'admission_union','context_scope':'pair','question':'meaning_relation','category':'exact_copy'}])
    chosen=select_illustrative_ids(m,b,illustration,software_fixture_only=True)
    tests['illustration_ids_only_fixed_rule']=chosen['examples'][0]['selected_item_ids']==['fixture-u0'] and not chosen['publisher_text_exported']
    tests['fixture_never_counted_as_human']=r['actual_human_response_count']==0 and r['fixture_completed_response_rows']==3
    tests['missing_corpus_not_reweighted']=all(v is None for group in r['equal_corpus_balanced'] for q in group['ht_category_estimates'].values() for v in q.values())
    bad=copy.deepcopy(b);bad['responses'][1]['annotator_id']=bad['responses'][0]['annotator_id']
    try:validate_responses(m,bad,software_fixture_only=True)
    except ValueError:tests['duplicate_human_refused']=True
    else:tests['duplicate_human_refused']=False
    try:validate_responses(m,b)
    except ValueError:tests['fixture_to_real_promotion_refused']=True
    else:tests['fixture_to_real_promotion_refused']=False
    partial=copy.deepcopy(b);partial['responses'].pop()
    p=analyze_human_audit(m,partial,software_fixture_only=True)
    missing=primary(p)['categories']['meaning_relation']['exact_copy']
    tests['two_raters_not_full_three_rater_outcome']=missing['ht_prevalence'] is None and missing['finite_frame_bounded_outcome_identification_bounds']==[0.,1.]
    skipped=copy.deepcopy(b);s=skipped['responses'][2];s.update(meaning_relation='',consequential_distinction='',distinction_types=[],task_relevance='',skipped=True,skip_reason='software fixture skip')
    sk=analyze_human_audit(m,skipped,software_fixture_only=True)
    tests['skips_preserved_not_imputed']=primary(sk)['categories']['meaning_relation']['exact_copy']['ht_prevalence'] is None and primary(sk)['collection_rates']['any_skip']['hajek_prevalence']==1
    tie=copy.deepcopy(b)
    for row,meaning,consequence,task in zip(tie['responses'],['exact_copy','overlapping_information','unrelated'],['yes','no','uncertain'],['could_matter','unlikely_to_matter','uncertain']):
        row.update(meaning_relation=meaning,consequential_distinction=consequence,task_relevance=task,distinction_types=['entity'] if consequence=='yes' else [])
    raw=copy.deepcopy(tie)
    t=analyze_human_audit(m,tie,software_fixture_only=True)
    tests['three_way_tie_not_fabricated_majority']=t['unit_results'][0]['original_majority']['meaning_relation']=='no_majority'
    agreement=primary(t)['original_rater_agreement']['meaning_relation']
    tests['agreement_and_uncertainty_preserved']=abs(agreement['weighted_fleiss_kappa']+.5)<1e-15 and primary(t)['original_rater_agreement']['task_relevance']['weighted_uncertain_rating_fraction']==1/3
    adj={'manifest_sha256':m['manifest_sha256'],'response_bundle_sha256':digest(tie),
       'policy':{'policy_id':'software_fixture_policy','policy_text':'Software-only later adjudication example; not a human record','fixed_before_adjudication':True},
       'decisions':[{'item_id':'fixture-u0','adjudicator_id':'fixture-adjudicator','reason':'software algebra check',
       'completed_at':'2026-01-02T12:00:00+00:00','answers':{'meaning_relation':'exact_copy','consequential_distinction':'yes','distinction_types':['entity'],'task_relevance':'could_matter'}}]}
    a=analyze_human_audit(m,tie,software_fixture_only=True,adjudication=adj)
    tests['adjudication_never_overwrites_original']=tie==raw and a['original_response_bundle']==raw and a['unit_results'][0]['original_majority']['meaning_relation']=='no_majority' and not a['adjudication']['original_ratings_replaced']
    badadj=copy.deepcopy(adj);badadj['decisions'][0]['completed_at']='2025-12-31T12:00:00+00:00'
    try:analyze_human_audit(m,tie,software_fixture_only=True,adjudication=badadj)
    except ValueError:tests['premature_adjudication_refused']=True
    else:tests['premature_adjudication_refused']=False
    wrong=copy.deepcopy(m);wrong['complete_frame'][0]['conditional_union_inclusion_probability']=fraction(1,2)
    wrong['complete_frame_sha256']=digest(wrong['complete_frame']);wrong['manifest_sha256']=digest({k:v for k,v in wrong.items() if k!='manifest_sha256'})
    wrongbundle=copy.deepcopy(b);wrongbundle['manifest_sha256']=wrong['manifest_sha256']
    try:analyze_human_audit(wrong,wrongbundle,software_fixture_only=True)
    except ValueError:tests['rehashed_wrong_probability_refused']=True
    else:tests['rehashed_wrong_probability_refused']=False
    # Actual local packs stay blank. Their outputs report missingness only.
    locations={'pair':ROOT/'phase4/results/human_admission_engineering_pack',
      'context':ROOT/'phase5/results/context_audit_engineering/blank_pack'}
    actual_counts={}
    outdir=ROOT/'phase5/results/human_analysis_blank'
    outdir.mkdir(exist_ok=True)
    for name,directory in locations.items():
        manifest=json.loads((directory/'private_sampling_manifest.json').read_text())
        bundle=json.loads((directory/'responses.template.json').read_text())
        result=analyze_human_audit(manifest,bundle)
        tests[name+'_real_pack_has_zero_humans']=result['actual_human_response_count']==0
        tests[name+'_real_pack_semantics_undefined']=all(v['ht_prevalence'] is None for g in result['original_summaries'] for categories in g['categories'].values() for v in categories.values())
        actual_counts[name]={'expected_assignments':result['expected_assignment_count'],'actual_human_responses':0,'sampled_units':len(result['unit_results'])}
        (outdir/f'{name}_missingness.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    # Context completeness is a scientific scope gate independent of valid ratings.
    cm=json.loads((locations['context']/'private_sampling_manifest.json').read_text())
    cb=json.loads((locations['context']/'responses.template.json').read_text());cb['evidence_role']='software_fixture_only'
    item=cm['selected_units'][0]['unit_id'];chosen=next(u for u in cm['complete_frame'] if u['unit_id']==item)
    # Test only a schema mutation in memory, preserving actual pack files untouched.
    chosen['missing_comparison']=True;cm['complete_frame_sha256']=digest(cm['complete_frame']);cm['manifest_sha256']=digest({k:v for k,v in cm.items() if k!='manifest_sha256'});cb['manifest_sha256']=cm['manifest_sha256']
    for i,row in enumerate(cb['responses']):
        if row['unit_id']==item:row.update(annotator_id=f'fixture-{i}',response_id=f'fixture-response-{i}',distinct_information='yes',consequential_distinction='yes',distinction_types=['entity'],task_relevance='could_matter',human_completed=True,completed_at='2026-01-01T12:00:00+00:00')
    try:analyze_human_audit(cm,cb,software_fixture_only=True)
    except ValueError:tests['no_comparator_cannot_be_novelty']=True
    else:tests['no_comparator_cannot_be_novelty']=False
    audit={'schema':'ccu-human-analysis-checks-1','status':'passed' if all(tests.values()) else 'failed',
      'checks':tests,'passed':sum(tests.values()),'total':len(tests),'actual_blank_packs':actual_counts,
      'filled_fixture_answers_stored_as_human_records':False,'collection_or_dispatch_performed':False,
      'code_sha256':hashlib.sha256((Path(__file__).with_name('human_analysis.py')).read_bytes()).hexdigest(),
      'check_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'phase5/results/human_analysis_checks.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2));assert all(tests.values())

if __name__=='__main__':main()
