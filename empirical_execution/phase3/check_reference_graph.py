#!/usr/bin/env python3
"""Graph context invariance on existing natural-text vectors; no invented corpus."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ccu.data import read_natural_jsonl,lexical_engineering_features,sha256_file
from phase3.reference_graph import build_reference_graph,id_edges
from phase3.panels import graph_normalize,reference_cosines

def main():
    p=ROOT/'data/civil_comments_engineering_preview.jsonl'
    rows=read_natural_jsonl(p);ids=[r['record_id'] for r in rows]
    x,_=lexical_engineering_features(rows,128)
    base=build_reference_graph(x,ids,.6,block_size=17);edges=id_edges(base)
    checks=0
    for block in [1,7,32,256]:
        assert id_edges(build_reference_graph(x,ids,.6,block_size=block))==edges;checks+=1
    for removed in [set(range(1)),set(range(8)),set(range(0,len(ids),3)),set(range(len(ids)))]:
        ix=np.asarray([i for i in range(len(ids)) if i not in removed],dtype=int)
        surviving={ids[i] for i in ix}
        fresh=build_reference_graph(x[ix],[ids[i] for i in ix],.6,block_size=7)
        assert id_edges(fresh)=={(a,b) for a,b in edges if a in surviving and b in surviving};checks+=1
    order=np.arange(len(ids))[::-1]
    assert id_edges(build_reference_graph(x[order],[ids[i] for i in order],.6,block_size=32))==edges;checks+=1
    normalized=graph_normalize(x)
    big=reference_cosines(normalized,normalized)
    small=np.empty_like(big)
    for i in range(0,len(ids),17):
        for j in range(0,len(ids),13):small[i:i+17,j:j+13]=reference_cosines(normalized[i:i+17],normalized[j:j+13])
    assert np.array_equal(big,small);checks+=1
    result={'status':'passed','natural_records':len(rows),'dimension':128,'edge_count':len(edges),'checks':checks,
      'bit_exact_score_entries_checked':int(big.size),'tile_row_order_and_retained_context_invariant':True,
      'scope':'lexical engineering natural fixture; not semantic quality or real-arithmetic score certificate',
      'data_sha256':sha256_file(p),'code_sha256':{p.name:sha256_file(p) for p in [Path(__file__),ROOT/'phase3/reference_graph.py',ROOT/'phase3/panels.py']}}
    out=ROOT/'phase3/results/reference_graph_checks.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
