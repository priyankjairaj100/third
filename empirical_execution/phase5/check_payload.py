#!/usr/bin/env python3
"""Algebra-only software fixtures plus reused natural Civil100 integration.

No fixture is an empirical corpus. Source partitions in algebra fixtures are
mathematical inputs and never stand in for missing native Civil sources.
"""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from ccu.core import BlockerGraph, ridge_moments, solve_ridge
from ccu.data import read_natural_jsonl, lexical_engineering_features, sha256_file
from phase3.reference_graph import build_reference_graph
from phase5.payload import PayloadState

checks = 0
max_moment = 0.0
max_head = 0.0


def graph_from_groups(groups, sources=None, ids=None):
    n = len(groups)
    ids = tuple(ids or [f'r{i}' for i in range(n)])
    ptr = np.asarray([0] + list(itertools.accumulate(map(len, groups))), dtype=np.int64)
    index = np.asarray([i for group in groups for i in group], dtype=np.int64)
    return BlockerGraph(ids, tuple(sources or ids), np.arange(n, dtype=np.int64), ptr, index, .5)


def retained_graph(graph, gone, unit):
    owners = graph.record_ids if unit == 'record' else graph.source_ids
    alive = [i for i, owner in enumerate(owners) if owner not in gone]
    remap = {old: new for new, old in enumerate(alive)}
    groups = [[remap[int(v)] for v in graph.indices[graph.indptr[i]:graph.indptr[i+1]] if int(v) in remap] for i in alive]
    old_rank = {int(v): k for k, v in enumerate(graph.priority_indices)}
    rank = np.asarray(sorted(range(len(alive)), key=lambda j: old_rank[alive[j]]), dtype=np.int64)
    g = graph_from_groups(groups, [graph.source_ids[i] for i in alive], [graph.record_ids[i] for i in alive])
    return BlockerGraph(g.record_ids, g.source_ids, rank, g.indptr, g.indices, graph.threshold), alive


def verify(state, graph, x, y, gone):
    global checks, max_moment, max_head
    state.check_invariants(True)
    # Scalar independent retained-selection predicate directly over original
    # graph blockers; never use update code to obtain the expected membership.
    owners = graph.record_ids if state.unit == 'record' else graph.source_ids
    selected = [i for i in range(len(x)) if owners[i] not in gone and all(owners[int(b)] in gone for b in graph.indices[graph.indptr[i]:graph.indptr[i+1]])]
    expected_ids = tuple(sorted(graph.record_ids[i] for i in selected))
    assert state.selected_ids() == expected_ids
    expected = ridge_moments(x[selected], y[selected])
    actual = state.moments()
    assert actual.count == expected.count
    mg = float(np.max(np.abs(actual.gram-expected.gram), initial=0))
    mh = float(np.max(np.abs(actual.cross-expected.cross), initial=0))
    max_moment = max(max_moment, mg, mh)
    np.testing.assert_allclose(actual.gram, expected.gram, atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(actual.cross, expected.cross, atol=1e-9, rtol=1e-10)
    head = state.decode(.01).weights
    oracle = solve_ridge(expected,.01).weights
    max_head = max(max_head, float(np.max(np.abs(head-oracle), initial=0)))
    np.testing.assert_allclose(head, oracle, atol=1e-8, rtol=1e-9)
    fresh_graph, alive = retained_graph(graph, gone, state.unit)
    fresh = PayloadState(fresh_graph, x[alive], y[alive], state.horizon,
                         unit=state.unit, retain_all=state.retain_all,
                         compaction_fraction=state.compaction_fraction)
    assert state.logical_membership() == fresh.logical_membership()
    if state.retain_all:
        assert state.accounting()['live_payload_rows'] == len(alive)
    else:
        assert all(len(r['blockers']) <= state.horizon and r['owner'] not in r['blockers'] for r in state.logical_membership().values())
    checks += 1


def algebra_checks():
    global checks
    # Exhaust every earlier-neighbor graph on four ordered vertices, both service
    # universes, every permitted cumulative deletion set, both access strata.
    pairs = [(i,j) for i in range(4) for j in range(i)]
    x = np.asarray([[1,.25],[.5,1],[1,-.5],[.25,.75]], dtype=np.float32)
    y = np.asarray([[0,1],[.25,.75],[.5,.5],[1,0]], dtype=np.float64)
    configs = 0
    for mask in range(1 << len(pairs)):
        groups = [[] for _ in range(4)]
        for k, (i,j) in enumerate(pairs):
            if mask >> k & 1: groups[i].append(j)
        graph = graph_from_groups(groups, ['s0','s0','s1','s2'])
        for unit in ['record','source']:
            universe = sorted(set(graph.record_ids if unit == 'record' else graph.source_ids))
            for h in range(len(universe)+1):
                for all_payload in [False, True]:
                    for count in range(h+1):
                        for request in itertools.combinations(universe,count):
                            state = PayloadState(graph,x,y,h,unit=unit,retain_all=all_payload)
                            result = state.delete(request)
                            assert result['fresh_deletions'] == count
                            verify(state,graph,x,y,set(request))
                            configs += 1
    # Sequential/batch/reverse, duplicate retries, resume and validation atomicity.
    graph = graph_from_groups([[],[0],[1],[0,2]], ['s0','s0','s1','s2'])
    for unit, path in [('record',['r0','r1','r2','r3']), ('source',['s0','s1','s2'])]:
        for all_payload in [False,True]:
            state = PayloadState(graph,x,y,len(path),unit=unit,retain_all=all_payload,compaction_fraction=1)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)/'state.checkpoint'
                original_x=x.copy();original_y=y.copy()
                assert not np.shares_memory(state.x,x) and not np.shares_memory(state.y,y)
                gone=set()
                for i,u in enumerate(path):
                    state.delete([u,u]);gone.add(u)
                    verify(state,graph,x,y,gone)
                    s=state.snapshot(p);assert s['serialized_bytes']==p.stat().st_size
                    loaded=PayloadState.load(p)
                    assert loaded.logical_membership()==state.logical_membership()
                    assert np.array_equal(loaded.gram,state.gram)
                    assert np.array_equal(loaded.cross,state.cross)
                    assert loaded.accounting()['allocated_array_bytes']==sum(getattr(loaded,n).nbytes for n in loaded.accounting()['array_bytes_by_field'])
                    state=loaded
                    previous_h=state.horizon
                    assert state.delete([u])['fresh_deletions']==0 and state.horizon==previous_h
                    frozen=state.snapshot(p)['sha256']
                    try: state.delete(['unknown',u])
                    except ValueError: pass
                    else: raise AssertionError('unknown request accepted')
                    assert state.snapshot(p)['sha256']==frozen
                    checks += 1
                assert not state.gram.any() and not state.cross.any() and state.count==0
                assert np.array_equal(x,original_x) and np.array_equal(y,original_y)
            batch=PayloadState(graph,x,y,len(path),unit=unit,retain_all=all_payload)
            batch.delete(list(reversed(path)))
            assert batch.logical_membership()==state.logical_membership()
    state=PayloadState(graph,x,y,1)
    before=state.logical_membership();g=state.gram.copy();c=state.cumulative.copy()
    try:state.delete(['r0','r1'])
    except ValueError:pass
    else:raise AssertionError('overbudget accepted')
    assert state.horizon==1 and before==state.logical_membership() and np.array_equal(g,state.gram) and c==state.cumulative
    state.delete(['r0'])
    try:state.delete(['r1'])
    except ValueError:pass
    else:raise AssertionError('extra request after exhaustion accepted')
    # Legitimate stale retained bytes until the locked threshold is reached.
    stale=PayloadState(graph,x,y,4,retain_all=True,compaction_fraction=1)
    stale.delete(['r3'])
    ledger=stale.accounting()
    assert ledger['stale_payload_rows']==1 and ledger['stale_payload_bytes']==x.shape[1]*4+y.shape[1]*8
    # A large disjoint source batch is accumulated in bounded BLAS blocks.
    n=2051
    big=graph_from_groups([[] for _ in range(n)], ['bulk']*n)
    xx=np.ones((n,3),dtype=np.float32);yy=np.full((n,1),.5,dtype=np.float64)
    bulk=PayloadState(big,xx,yy,1,unit='source',batch_rows=128)
    assert bulk.cumulative['blas_batches']==17
    op=bulk.delete(['bulk'])
    assert op['operations']['blas_batches']==17 and bulk.count==0 and len(bulk.x)==0
    bulk.check_invariants(True)
    checks+=5
    return configs


def natural_checks():
    source=ROOT/'data/civil_comments_engineering_preview.jsonl'
    rows=read_natural_jsonl(source,require_labels=True)
    ids=[r['record_id'] for r in rows]
    e,_=lexical_engineering_features(rows,128)
    graph=build_reference_graph(e,ids,.6,block_size=32)
    x,_=lexical_engineering_features(rows,64)
    y=np.asarray([r['label'] for r in rows],dtype=np.float64)[:,None]
    paths=[sorted(ids,key=lambda r:hashlib.sha256(f'payload-check-{seed}|{r}'.encode()).digest())[:8] for seed in range(4)]
    paths.append([ids[i] for i in range(len(ids)) if graph.blocker_counts[i]>0][:8])
    logs=[]
    for keep_all in [False,True]:
        for j,path in enumerate(paths):
            s=PayloadState(graph,x,y,8,retain_all=keep_all)
            gone=set()
            for u in path:
                op=s.delete([u]);gone.add(u)
                verify(s,graph,x,y,gone)
                logs.append({'method':'B-A' if keep_all else 'B-E','path':j,'deleted':len(gone),'admitted':len(op['admitted_ids']),'count':s.count,'accounting':s.accounting()})
    return {'scope':'reused natural Civil100 lexical implementation check; no semantic/source evidence',
            'data_sha256':sha256_file(source),'records':len(rows),'source_arm_executed':False,
            'native_source_fields_absent':True,'checkpoints':len(logs),'rows':logs}


def main():
    configs=algebra_checks()
    natural=natural_checks()
    result={'schema':'ccu-payload-checks-1','status':'passed','software_fixture_configurations':configs,
            'verified_states':checks,'maximum_moment_abs_difference':max_moment,
            'maximum_head_abs_difference':max_head,'natural':natural,
            'source_sha256':sha256_file(Path(__file__).with_name('payload.py')),
            'check_sha256':sha256_file(Path(__file__)),
            'primary_study':False,'rigorous_numeric_certificate':False}
    output=ROOT/'phase5/results/payload_checks.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='natural'},indent=2))
    print('Natural check states:',natural['checkpoints'])

if __name__=='__main__':main()
