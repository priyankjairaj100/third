#!/usr/bin/env python3
"""Real Civil-100 lexical frame; only blank annotation packs are created."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import copy
import itertools
import json
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from phase3.reference_graph import build_reference_graph
from phase4.requests import generate_manifest
from phase4.admission_audit import prepare_admission_audit,verify_admission_audit,write_admission_pack,QUOTAS
from phase4.check_requests import expect_error


def main():
    data=ROOT/'data/civil_comments_engineering_preview.jsonl';records=read_natural_jsonl(data)
    x,_=lexical_engineering_features(records,128);ids=[r['record_id'] for r in records]
    graph=build_reference_graph(x,ids,.6,block_size=17)
    design={'requests':{'record_horizon':8,'record_checkpoints':[1,2,4,8]}}
    requests=generate_manifest(graph,dataset_id='civil_comments_engineering_preview',
        panel_id='all100_lexical128_human_pack_software_check',source_kinds={s:'unknown_singleton' for s in graph.source_ids},
        design=design,allocations={'R':8,'S':0,'U':8,'A':4},include_excluded_blocker_stress=False)
    inputs={'civil_comments':{'records':records,'features':x,'graph':graph,'request_manifest':requests,
        'existing_labels':{r['record_id']:r['label'] for r in records},
        'provenance':{'corpus_snapshot':'existing_connector_preview_100_not_original_snapshot',
                      'representation_id':'lexical_engineering_128','representation_revision':'existing_workspace_code',
                      'evidence_role':'engineering_nonconfirmatory'}}}
    manifest=prepare_admission_audit(inputs);verify_admission_audit(manifest)
    assert manifest==prepare_admission_audit(inputs)
    assert sum(QUOTAS[c]['R/S']+QUOTAS[c]['U/A'] for c in QUOTAS)==400
    assert sum(QUOTAS[c]['control'] for c in QUOTAS)==200
    assert manifest['corpus_status']['askubuntu']=='missing_corpus_no_quota_reallocation'
    assert manifest['corpus_status']['cc_news']=='missing_corpus_no_quota_reallocation'
    frame={p['pair_id']:p for p in manifest['complete_frame']};prob={}
    observed=set();retained_excluded=set();multiplicity=Counter();family= {}
    initial={graph.record_ids[i] for i in range(len(ids)) if graph.indptr[i]==graph.indptr[i+1]}
    b={r:set(graph.record_ids[int(j)] for j in graph.blockers[i]) for i,r in enumerate(graph.record_ids)}
    for path in requests['trajectories']:
        for o in path['observations']:
            deleted=set(path['deletion_order'][:o['checkpoint']])
            current={r for r in ids if r not in deleted and not (b[r]-deleted)}
            admissions=current-initial;observed|=admissions
            retained_excluded|=set(ids)-initial-deleted-current
            for r in admissions:
                multiplicity[r]+=1;family.setdefault(r,set()).add('R/S' if path['arm'] in {'R','S'} else 'U/A')
    expected_controls=retained_excluded-observed
    actual_admissions={p['record_id'] for p in frame.values() if set(p['roles'])&{'R/S','U/A'}}
    actual_controls={p['record_id'] for p in frame.values() if p['roles']==['control']}
    assert actual_admissions==observed and actual_controls==expected_controls
    assert not actual_admissions&actual_controls
    for p in frame.values():
        assert p['former_blocker_id'] in b[p['record_id']]
        assert p['former_blocker_draw_probability']=={'numerator':1,'denominator':len(b[p['record_id']])}
        if p['record_id'] in observed:
            assert p['occurrence_multiplicity']==multiplicity[p['record_id']]
            assert set(p['roles'])==family[p['record_id']]
    selected=set()
    for stratum in manifest['strata']:
        key=tuple(stratum['stratum']);pop=stratum['frame_pair_ids'];sample=stratum['selected_pair_ids']
        assert len(sample)==min(stratum['quota'],len(pop))
        assert set(sample)<=set(pop);selected.update(sample)
        assert stratum['shortfall']==stratum['quota']-len(sample)
        prob[key]=Fraction(len(sample),len(pop)) if pop else Fraction(0)
        for pid in pop: assert list(key) in frame[pid]['strata']
    for p in frame.values():
        miss=Fraction(1)
        for h in p['strata']: miss*=1-prob[tuple(h)]
        expected=1-miss;actual=p['conditional_union_inclusion_probability']
        assert expected==Fraction(actual['numerator'],actual['denominator'])
    assert selected=={p['pair_id'] for p in manifest['selected_pairs']}
    assert len(manifest['selected_pairs'])==len(selected)
    # Pure combinatorial validation of overlapping independent role samples.
    first=list(itertools.combinations(('p','q','r'),1));second=list(itertools.combinations(('p','s'),1))
    picked=sum('p' in a or 'p' in b for a in first for b in second)
    assert Fraction(picked,len(first)*len(second))==1-(1-Fraction(1,3))*(1-Fraction(1,2))
    bad=copy.deepcopy(manifest);bad['complete_frame'][0]['record_id']='tampered'
    expect_error(lambda:verify_admission_audit(bad))
    missing_labels={**inputs['civil_comments'],'existing_labels':{}}
    m=prepare_admission_audit({'civil_comments':missing_labels})
    assert all(p['existing_label_relation']=='unavailable' for p in m['complete_frame'])
    # No ratings are filled. Temporary pack checks do not dispatch annotations.
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'pack';write_admission_pack(manifest,{'civil_comments':records},p)
        expect_error(lambda:write_admission_pack(manifest,{'civil_comments':records},p))
        assignments=json.loads((p/'blinded_assignments.json').read_text())
        responses=json.loads((p/'responses.template.json').read_text())['responses']
        assert len(assignments)==len(responses)==3*len(selected)
        assert Counter(a['pair_id'] for a in assignments)=={pid:3 for pid in selected}
        permitted={'assignment_id','pair_id','text_a','text_b','task_definition'}
        assert all(set(a)==permitted for a in assignments)
        assert all(not r['human_completed'] and not r['annotator_id'] and not r['response_id'] and
                   not r['meaning_relation'] and not r['consequential_distinction'] and not r['task_relevance'] for r in responses)
    out=ROOT/'phase4/results';out.mkdir(parents=True,exist_ok=True)
    pack=out/'human_admission_engineering_pack'
    if pack.exists():
        # Keep frozen packs immutable; repeated check writes a fresh temporary
        # pack above and verifies the checked-in pack's exact content below.
        assert json.loads((pack/'private_sampling_manifest.json').read_text())==manifest
    else:write_admission_pack(manifest,{'civil_comments':records},pack)
    report=dict(status='passed',evidence_role='engineering_nonconfirmatory',natural_records=100,
        representation='lexical_engineering_128_not_semantic_encoder',declared_request_paths=20,
        complete_unique_pair_frame=len(frame),unique_admitted_records=len(actual_admissions),
        unique_disjoint_observed_controls=len(actual_controls),selected_unique_pairs=len(selected),
        blank_assignments=3*len(selected),completed_human_responses=0,human_dispatch_ready=False,
        all_frame_occurrences_controls_quotas_and_probabilities_independently_checked=True,
        overlap_probability_combinatorial_cases=len(first)*len(second),
        no_missing_label_imputation=True,no_favorable_shortfall_refill=True,
        full_neighborhood_and_nearest_survivor_audits='pending_real_context_not_completed',
        manifest_sha256=manifest['manifest_sha256'],data_sha256=sha256_file(data),
        code_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in
                     [Path(__file__),ROOT/'phase4/admission_audit.py',ROOT/'phase4/requests.py']})
    (out/'human_admission_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
