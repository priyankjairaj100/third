#!/usr/bin/env python3
"""Natural-record reference and blank-pack audit; no labels are supplied or inferred."""
from __future__ import annotations
import bisect
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ccu.data import lexical_engineering_features, read_natural_jsonl
from phase3 import panels, calibration, reference_graph


def scalar_normalize(matrix):
    result=[]
    for raw in matrix:
        row=[float(v) for v in raw]
        total=0.0
        for value in row:
            total=total+value*value
        norm=math.sqrt(total)
        result.append([value/norm for value in row])
    return result


def scalar_dot(left,right):
    total=0.0
    for a,b in zip(left,right):
        total=total+a*b
    return total


def run():
    rows=read_natural_jsonl(ROOT/'data/civil_comments_engineering_preview.jsonl',require_labels=True)
    score_counts={}
    graph_checks=[]
    oracle128=None
    for dimension in (128,768):
        x,_=lexical_engineering_features(rows,dimension)
        oracle_norm=scalar_normalize(x)
        actual_norm=panels.graph_normalize(x)
        assert np.array_equal(actual_norm,np.array(oracle_norm))
        oracle={(i,j):scalar_dot(oracle_norm[i],oracle_norm[j]).hex()
                for i in range(len(rows)) for j in range(i+1,len(rows))}
        for tile in (7,17,32,256):
            actual={(i,j):score.hex() for i,j,score in calibration.iter_pair_scores(x,tile)}
            assert actual==oracle,(dimension,tile)
        retained=[i for i in range(len(rows)) if i%4!=0]
        fresh={(retained[i],retained[j]):score.hex()
               for i,j,score in calibration.iter_pair_scores(x[retained],13)}
        assert fresh=={ij:score for ij,score in oracle.items() if ij[0] in retained and ij[1] in retained}
        reversed_ids=list(reversed(range(len(rows))))
        reverse={(min(reversed_ids[i],reversed_ids[j]),max(reversed_ids[i],reversed_ids[j])):score.hex()
                 for i,j,score in calibration.iter_pair_scores(x[reversed_ids],23)}
        assert reverse==oracle
        score_counts[str(dimension)]={'pairs_independently_scored':len(oracle),'tile_sizes':[7,17,32,256],
                                     'retained_rebuild_pairs':len(fresh),'reversed_order_pairs':len(reverse),
                                     'exact_hex_agreement':True}
        ids=[row['record_id'] for row in rows]
        for seed in (0,1,2):
            priority=sorted(ids,key=lambda rid:(hashlib.sha256(f'priority-v1|{seed}|{rid}'.encode()).digest(),rid))
            ranks={rid:i for i,rid in enumerate(priority)}
            expected=set()
            for (i,j),score in oracle.items():
                if float.fromhex(score)>.6:
                    expected.add((ids[i],ids[j]) if ranks[ids[i]]>ranks[ids[j]] else (ids[j],ids[i]))
            for tile in (7,19,256):
                graph=reference_graph.build_reference_graph(x,ids,.6,seed=seed,block_size=tile)
                actual={(graph.record_ids[i],graph.record_ids[int(j)]) for i,b in enumerate(graph.blockers) for j in b}
                assert actual==expected,(dimension,seed,tile)
            fresh=reference_graph.build_reference_graph(x[retained],[ids[i] for i in retained],.6,seed=seed,block_size=13)
            kept={ids[i] for i in retained}
            actual={(fresh.record_ids[i],fresh.record_ids[int(j)]) for i,b in enumerate(fresh.blockers) for j in b}
            assert actual=={(a,b) for a,b in expected if a in kept and b in kept}
            graph_checks.append({'dimension':dimension,'priority_seed':seed,'directed_edges':len(expected),'tile_sizes':[7,19,256],
                                 'fresh_retained_edges':len(actual),'scalar_oracle_agreement':True})
        if dimension==128:oracle128=oracle
    pack=ROOT/'phase3/results/calibration_engineering_pack'
    manifest=json.loads((pack/'private_sampling_manifest.json').read_text())
    blank=json.loads((pack/'responses.template.json').read_text())
    assignments=json.loads((pack/'blinded_assignments.json').read_text())
    actual_code=hashlib.sha256(Path(calibration.__file__).read_bytes()).hexdigest()
    assert manifest['frame']['scorer']['code_sha256']==actual_code
    actual_shared=hashlib.sha256(Path(panels.__file__).read_bytes()).hexdigest()
    assert manifest['frame']['scorer']['shared_scorer_sha256']==actual_shared
    ids={row['record_id']:i for i,row in enumerate(rows)}
    edges=[float(Fraction(-10+i,10)) for i in range(21)]
    bins=[0]*20
    def bin_of(score):return min(19,max(0,bisect.bisect_right(edges,score)-1))
    for score in oracle128.values(): bins[bin_of(float.fromhex(score))]+=1
    assert bins==manifest['bin_populations']
    assert manifest['bin_sample_counts']==[min(30,n) for n in bins]
    for pair in manifest['pairs']:
        ij=tuple(sorted((ids[pair['left_id']],ids[pair['right_id']])))
        assert pair['score_hex']==oracle128[ij]
        b=bin_of(float.fromhex(pair['score_hex']))
        assert pair['bin']==b
        fp=pair['inclusion_probability_exact']
        assert Fraction(fp['numerator'],fp['denominator'])==Fraction(min(30,bins[b]),bins[b])
    required={a:p for p in manifest['pairs'] for a in p['assignment_ids']}
    assert len(required)==3*len(manifest['pairs'])==len(assignments)==len(blank['responses'])
    by_id={row['record_id']:row['text'] for row in rows}
    for assignment in assignments:
        assert set(assignment)=={'assignment_id','pair_id','text_a','text_b'}
        pair=required[assignment['assignment_id']]
        assert sorted([assignment['text_a'],assignment['text_b']])==sorted([by_id[pair['left_id']],by_id[pair['right_id']]])
    for response in blank['responses']:
        assert response['human_completed'] is False and response['label']=='' and response['category']==''
        assert response['annotator_id']=='' and response['response_id']==''
    try:calibration.select_threshold(manifest,blank)
    except ValueError:pass
    else:raise AssertionError('Blank responses accepted')
    return {'passed':True,'scope':'independent scalar finite-precision reference; natural lexical engineering data only',
            'natural_records':len(rows),'score_checks':score_counts,'graph_checks':graph_checks,'selection_population_pairs':sum(bins),
            'sampled_pairs':len(manifest['pairs']),'blank_assignments':len(assignments),
            'human_judgments_created_or_present':0,'semantic_quality_pass':False,
            'production_hashes':{'calibration.py':actual_code,'panels.py':actual_shared,'reference_graph.py':hashlib.sha256(Path(reference_graph.__file__).read_bytes()).hexdigest()}}

if __name__=='__main__':
    result=run()
    Path(__file__).with_name('independent_scoring_pack_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
