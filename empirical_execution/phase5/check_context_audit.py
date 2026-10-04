#!/usr/bin/env python3
"""Natural Civil lexical contextual frame; ratings remain blank."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import argparse,copy,json,math,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from phase3.reference_graph import build_reference_graph
from phase4.requests import generate_manifest
from phase5.context_audit import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'phase5/results/context_audit_engineering')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/civil_comments_engineering_preview.jsonl';records=read_natural_jsonl(source)
    x,_=lexical_engineering_features(records,128);ids=[r['record_id'] for r in records];lookup={r:i for i,r in enumerate(ids)}
    graph=build_reference_graph(x,ids,.6,block_size=17)
    design={'requests':{'record_horizon':8,'record_checkpoints':[1,2,4,8]}}
    requests=generate_manifest(graph,dataset_id='civil_comments_engineering_preview',panel_id='all100_lexical128_human_pack_software_check',
      source_kinds={s:'unknown_singleton' for s in graph.source_ids},design=design,
      allocations={'R':8,'S':0,'U':8,'A':4},include_excluded_blocker_stress=False)
    inputs={'civil_comments':dict(records=records,features=x,graph=graph,request_manifest=requests,
      existing_labels={r['record_id']:r['label'] for r in records},
      provenance=dict(corpus_snapshot='existing_connector_preview_100_not_original_snapshot',representation_id='lexical_engineering_128',
        representation_revision='existing_workspace_code',evidence_role='engineering_nonconfirmatory'))}
    manifest=prepare_context_audits(inputs);verify_context_manifest(manifest)
    assert manifest==prepare_context_audits(inputs)
    checks=['deterministic_full_manifest']
    # Independent raw-neighbor test across all logged prefixes, no preparer frame reuse.
    blockers={rid:{ids[int(j)] for j in graph.blockers[i]} for i,rid in enumerate(ids)}
    initially_selected={rid for rid in ids if not blockers[rid]};admitted=set();events={}
    for path in requests['trajectories']:
        for obs in path['observations']:
            dead=set(path['deletion_order'][:obs['checkpoint']]);selected={r for r in ids if r not in dead and blockers[r]<=dead}
            for r in selected-initially_selected:
                admitted.add(r);events.setdefault(r,[]).append((path['trajectory_id'],obs['checkpoint']))
    for kind in KINDS:
        frame=[u for u in manifest['complete_frame'] if u['kind']==kind]
        assert {u['record_id'] for u in frame}==admitted
        assert len(frame)==len(admitted)
        for unit in frame:
            assert sorted((e['trajectory_id'],e['checkpoint']) for e in unit['occurrences'])==sorted(events[unit['record_id']])
            assert unit['occurrence_multiplicity']==len(events[unit['record_id']])
    checks.append('complete_unique_admission_union_and_multiplicity')
    # Scalar normalization/dot oracle independently follows the declared coordinate order.
    normalized={}
    for r in ids:
        squared=0.
        for v in x[lookup[r]]:squared+=float(v)*float(v)
        normalized[r]=[float(v)/math.sqrt(squared) for v in x[lookup[r]]]
    def cosine(a,b):
        value=0.
        for u,v in zip(normalized[a],normalized[b]):value+=u*v
        return value
    for unit in manifest['complete_frame']:
        assert unit['selected_for_annotation']==unit['context_prepared']
        if not unit['context_prepared']:continue
        rid=unit['record_id'];state=manifest['checkpoint_states'][unit['checkpoint_state_key']]
        dead=set(state['deleted_record_ids']);selected={r for r in ids if r not in dead and blockers[r]<=dead}
        assert set(state['retained_selected_ids'])==selected
        assert set(unit['complete_former_blocker_ids'])==blockers[rid]
        assert blockers[rid]<=dead
        if unit['kind']=='former_neighborhood':assert set(unit['complete_context_ids'])==blockers[rid]
        else:
            candidates=sorted(selected-{rid});assert unit['nearest_candidate_count']==len(candidates)
            nearest=min(candidates,key=lambda r:(-cosine(rid,r),r)) if candidates else None
            assert unit['nearest_record_id']==nearest
            if nearest is not None:assert unit['nearest_reference_score']==cosine(rid,nearest)
        assert sum(s['displayed_characters'] for s in unit['displayed_context'])<=20000
    checks.extend(['complete_former_blocker_sets_at_actual_admission','nearest_surviving_selected_excludes_self_matches_scalar_oracle'])
    for stratum in manifest['strata']:
        n=stratum['sample_size'];N=stratum['population_size'];p=stratum['inclusion_probability']
        assert Fraction(p['numerator'],p['denominator'])==(Fraction(n,N) if N else 0)
        assert n==min(N,stratum['quota']) and stratum['shortfall']==stratum['quota']-n
    assert sum(QUOTAS.values())==100
    checks.append('fixed100_quotas_complete_frames_exact_weights_no_refill')
    # Tiny tie fixture checks software behavior only, not fabricated empirical text.
    fixture=np.asarray([[1.,0.],[1.,0.],[1.,0.]],dtype=np.float32)
    assert exact_nearest(fixture,['q','z','a'],'q',['z','a'])==('a',1.)
    assert exact_nearest(fixture,['q','z','a'],'q',[])==(None,None)
    checks.append('software_tie_id_order_and_empty_comparison')
    truncated=prepare_context_audits(inputs,target_character_limit=5,context_character_limit=7)
    assert any(u.get('omitted_or_truncated_context_ids') for u in truncated['complete_frame'])
    assert all(u['target_displayed_characters']<=5 and sum(s['displayed_characters'] for s in u['displayed_context'])<=7
      for u in truncated['complete_frame'] if u['context_prepared'])
    assert {u['unit_id'] for u in truncated['complete_frame']}=={u['unit_id'] for u in manifest['complete_frame']}
    assert [u['unit_id'] for u in truncated['selected_units']]==[u['unit_id'] for u in manifest['selected_units']]
    assert not any(u.get('complete_context_displayed',True) for u in truncated['complete_frame'])
    checks.append('logged_truncation_does_not_reselect_or_refill_units')
    pack=write_context_pack(manifest,{'civil_comments':records},out/'blank_pack')
    assignments=json.loads((out/'blank_pack/blinded_assignments.json').read_text())
    responses=json.loads((out/'blank_pack/responses.template.json').read_text())['responses']
    allowed={'assignment_id','unit_id','target_text','comparison_texts','task_definition','comparison_question','context_complete','comparison_available'}
    assert all(set(a)==allowed for a in assignments)
    assert len({a['assignment_id'] for a in assignments})==len(assignments)
    assert all(v==3 for v in Counter(a['unit_id'] for a in assignments).values())
    assert all(not r['human_completed'] and not r['annotator_id'] and not r['distinct_information'] for r in responses)
    checks.append('three_blinded_blank_independent_assignment_slots_per_unit')
    for name,call in [
      ('tampered_manifest',lambda:verify_context_manifest(dict(manifest,completed_human_responses=1))),
      ('overwrite',lambda:write_context_pack(manifest,{'civil_comments':records},out/'blank_pack')),
      ('changed_text',lambda:write_context_pack(manifest,{'civil_comments':[dict(records[0],text='changed')]+records[1:]},out/'changed')),
      ('engineering_relabel',lambda:prepare_context_audits(inputs,evidence_role='prospective_confirmatory_preparation'))]:
        try:call()
        except (ValueError,FileExistsError):checks.append(name+'_refused')
        else:raise AssertionError(name+' accepted')
    summary=dict(status='passed',checks=checks,complete_unique_admissions=len(admitted),
      selected_neighborhoods=sum(u['kind']=='former_neighborhood' and u['selected_for_annotation'] for u in manifest['complete_frame']),
      selected_nearest_selected=sum(u['kind']=='nearest_surviving_selected' and u['selected_for_annotation'] for u in manifest['complete_frame']),
      blank_assignments=pack['blank_assignments'],completed_human_responses=0,missing_corpora=['askubuntu','cc_news'],
      semantic_study_result=False,corpus_sha256=sha256_file(source),manifest_sha256=manifest['manifest_sha256'],
      source_service_empirical=False,human_dispatch_ready=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
