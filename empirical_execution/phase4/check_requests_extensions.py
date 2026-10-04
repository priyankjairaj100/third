#!/usr/bin/env python3
"""Tiny natural replay and exact sampler-law software checks, no invented evidence."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
from unittest.mock import patch
import copy
import json
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from ccu.core import BlockerGraph
from ccu.data import read_natural_jsonl, lexical_engineering_features, sha256_file
from phase3.reference_graph import build_reference_graph
from phase4.requests_extensions import generate_structure_extensions, verify_structure_extensions, weighted_source_prefix
from phase4.check_requests import audit_manifest, expect_error


class ScriptedRng:
    def __init__(self, values): self.values=iter(values)
    def randrange(self, maximum):
        value=next(self.values); assert 0<=value<maximum; return value


def exact_pps_law_check():
    weights={'A':1,'B':2,'C':3}; probability=Counter(); traces=0
    for first_draw in range(6):
        first='A' if first_draw==0 else 'B' if first_draw<3 else 'C'
        rest=6-weights[first]
        for second_draw in range(rest):
            scripted=ScriptedRng([first_draw,second_draw])
            with patch('phase4.requests_extensions.random.Random',return_value=scripted):
                order, draws=weighted_source_prefix(list(weights),weights,2,0)
            assert order[0]==first and len(set(order))==2
            assert draws[0]['conditional_probability_denominator']==6
            assert draws[1]['conditional_probability_denominator']==rest
            probability[tuple(order)]+=Fraction(1,6*rest); traces+=1
    for (a,b),actual in probability.items():
        assert actual==Fraction(weights[a],6)*Fraction(weights[b],6-weights[a])
    assert sum(probability.values())==1
    return traces,len(probability)


def main():
    traces,orderings=exact_pps_law_check()
    data=ROOT/'data/civil_comments_engineering_preview.jsonl'
    rows=read_natural_jsonl(data); ids=[r['record_id'] for r in rows]
    x,_=lexical_engineering_features(rows,128)
    graph=build_reference_graph(x,ids,.6,block_size=17)
    args=dict(dataset_id='civil_comments_engineering_preview',panel_id='all100_lexical128_structure_extensions',
              source_kinds={s:'unknown_singleton' for s in graph.source_ids},
              mass_source_paths=4,one_percent_allocations=dict(R=4,S=0,U=4,A=2))
    natural=generate_structure_extensions(graph,**args)
    verify_structure_extensions(natural,graph)
    assert natural==generate_structure_extensions(graph,**args)
    assert natural['configuration']['one_percent_record_horizon']==1
    assert all(p['status']=='structural_zero_empty_sampling_frame' for p in natural['mass_source_trajectories'])
    observations,candidates=audit_manifest(natural['one_percent_record_manifest'],graph)
    for p in natural['one_percent_record_manifest']['trajectories']:
        assert p['initial_horizon']==1 and p['checkpoints']==[1]
        assert p['observations'][0]['remaining_horizon']==0
    bad=copy.deepcopy(natural);bad['source_mass_weights']['fake']=1
    expect_error(lambda:verify_structure_extensions(bad))
    expect_error(lambda:weighted_source_prefix(['A','B'],{'A':1,'B':0},1,0))
    expect_error(lambda:weighted_source_prefix(['A'],{'A':1},2,0))

    # Algebra-only ownership/adjacency fixture; no factual source annotation.
    fixture=BlockerGraph(('a','b','c','d'),('P','P','unknown:c','Q'),np.arange(4),
                         np.array([0,0,1,2,2]),np.array([0,1]),.5)
    fkinds={'P':'genuine_native','Q':'genuine_native','unknown:c':'unknown_singleton'}
    fargs=dict(dataset_id='algebra_fixture_not_corpus',panel_id='source_math_only',source_kinds=fkinds,
               mass_source_paths=8,one_percent_allocations=dict(R=4,S=0,U=0,A=0))
    fm=generate_structure_extensions(fixture,**fargs)
    verify_structure_extensions(fm,fixture)
    assert fm['source_mass_weights']=={'P':2,'Q':1}
    assert fm['source_service_universe']==['P','Q','unknown:c']
    source_obs=0
    for p in fm['mass_source_trajectories']:
        assert set(p['deletion_order'])=={'P','Q'}
        assert p['service_universe_size']==3 and p['initial_horizon']==2
        assert p['observations'][-1]['deleted_record_ids']==['a','b','d']
        assert p['observations'][-1]['selected_record_ids']==['c']
        assert not p['uniform_source_admission_formula_applicable']
        source_obs+=len(p['observations'])
    changed=BlockerGraph(fixture.record_ids,fixture.source_ids,np.arange(4),
                         np.zeros(5,dtype=int),np.array([],dtype=int),.8)
    alternate=generate_structure_extensions(changed,**fargs)
    assert [p['deletion_order'] for p in fm['mass_source_trajectories']]==[p['deletion_order'] for p in alternate['mass_source_trajectories']]
    assert [p['deletion_order'] for p in fm['one_percent_record_manifest']['trajectories']]==[p['deletion_order'] for p in alternate['one_percent_record_manifest']['trajectories']]
    expect_error(lambda:verify_structure_extensions(fm,changed))
    # Pure integer boundary arithmetic under actual generator configuration,
    # using an edgeless adjacency object only; these are not empirical runs.
    boundary_checks=[]
    for n,expected in [(0,0),(1,1),(100,1),(101,2),(200,2),(201,3),(12801,129)]:
        names=tuple(f'algebra-index-{j}' for j in range(n))
        edgeless=BlockerGraph(names,names,np.arange(n),np.zeros(n+1,dtype=int),np.array([],dtype=int),.9)
        m=generate_structure_extensions(edgeless,dataset_id='integer_boundary_algebra',panel_id=str(n),
             source_kinds={r:'unknown_singleton' for r in names},mass_source_paths=0,
             one_percent_allocations=dict(R=0,S=0,U=0,A=0))
        assert m['configuration']['one_percent_record_horizon']==expected
        if expected: assert expected in m['configuration']['one_percent_requested_checkpoints']
        boundary_checks.append(dict(records=n,horizon=expected))
    out=ROOT/'phase4/results';out.mkdir(parents=True,exist_ok=True)
    (out/'requests_extensions_natural_engineering.json').write_text(json.dumps(natural,indent=2)+'\n')
    result=dict(status='passed',natural_records=100,natural_dimension=128,
                natural_curator='lexical_engineering_reference_FP64_tau_0.6',
                natural_one_percent_record_horizon=1,natural_checkpoint_observations_checked=observations,
                natural_candidate_scores_checked=candidates,natural_mass_source_paths_with_no_genuine_sources=4,
                mathematical_PPS_draw_traces_exhaustively_checked=traces,mathematical_PPS_order_probabilities_checked=orderings,
                source_fixture_observations_checked=source_obs,integer_horizon_boundaries_checked=boundary_checks,
                genuine_source_empirical_experiments_run=False,semantic_encoder_experiments_run=False,
                source_fixture_scope='mathematical ownership and integer arithmetic only; no source evidence',
                separate_namespace_and_state=True,within_branch_paired_curator_randomness=True,
                manifest_sha256=natural['manifest_sha256'],data_sha256=sha256_file(data),
                code_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in
                    [Path(__file__),ROOT/'phase4/requests_extensions.py',ROOT/'phase4/requests.py']})
    (out/'requests_extensions_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
