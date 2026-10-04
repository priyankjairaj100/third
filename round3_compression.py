#!/usr/bin/env python3
"""Exact finite checks for factor-two storage and canonical gauge compression.

Python standard library only. Test oracles keep original records; GaugeState
does not. Statistics and coefficient arithmetic are exact integer tuples.
Run: python round3_compression.py --output round3_compression_verification.json
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from itertools import combinations, permutations
import json
import math
from pathlib import Path
import random
import time

GROUND = "ground"


class DSU:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        for y in path:
            self.parent[y] = x
        return x

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def geometry(meta, h):
    """Current eligible metadata -> nodes/components and deterministic basis."""
    dsu = DSU()
    # Tuple hashing is charged once per represented endpoint occurrence;
    # union/find operates only on interned integer handles.
    node_ids = {}

    def intern(key):
        if key not in node_ids:
            node_ids[key] = len(node_ids)
        return node_ids[key]

    edges = []
    for record, (owner, blockers) in meta.items():
        assert owner not in blockers and len(blockers) <= h
        tail = blockers
        head = add_owner(blockers, owner) if len(blockers) < h else GROUND
        dsu.join(intern(tail), intern(head))
        edges.append((record, tail, head))
    components = {}
    for node, handle in node_ids.items():
        components.setdefault(dsu.find(handle), []).append(node)
    nodes, omitted, basis = set(), {}, set()
    for component in components.values():
        nonground = [x for x in component if x != GROUND]
        nodes.update(nonground)
        if GROUND in component:
            basis.update(nonground)
        else:
            assert len(nonground) >= 2
            distinguished = min(nonground)
            others = tuple(x for x in nonground if x != distinguished)
            omitted[distinguished] = others
            basis.update(others)
    assert len(nodes) <= 2 * len(basis)
    return nodes, omitted, basis, edges


def add_owner(blockers, owner):
    """Insert one source/record ID into an already sorted tuple in linear time."""
    at = bisect_left(blockers, owner)
    return blockers[:at] + (owner,) + blockers[at:]


def full_coeff(meta, h, stats, dim):
    out = {}
    for record, (owner, blockers) in meta.items():
        for key, sign in ((blockers, 1), (add_owner(blockers, owner), -1)):
            if len(key) <= h:
                old = out.get(key, (0,) * dim)
                out[key] = tuple(x + sign*y for x, y in zip(old, stats[record]))
    return {k: v for k, v in out.items() if any(v)}


def eligible(records, alive, h, canonical_input=False):
    out = {}
    for record, (owner, original_blockers) in records.items():
        if owner not in alive:
            continue
        b = tuple(u for u in original_blockers if u in alive)
        # Initial construction sorts input once; repairs preserve sorted order.
        if not canonical_input:
            b = tuple(sorted(b))
        if owner not in b and len(b) <= h:
            out[record] = (owner, b)
    return out


class GaugeState:
    """No original graph/statistics: only current eligible metadata and gauge."""
    __slots__ = ("alive", "h", "dim", "meta", "values")

    def __init__(self, records, alive, h, stats, dim):
        self.alive, self.h, self.dim = set(alive), h, dim
        self.meta = eligible(records, self.alive, h)
        coefficients = full_coeff(self.meta, h, stats, dim)
        basis = geometry(self.meta, h)[2]
        self.values = {key: coefficients.get(key, (0,) * dim) for key in basis}

    def reconstruct(self):
        nodes, omitted, basis, edges = geometry(self.meta, self.h)
        assert set(self.values) == basis
        out = dict(self.values)
        for key, others in omitted.items():
            # Fetch each tuple-keyed vector once, not once per scalar coordinate.
            total = [0] * self.dim
            for other in others:
                value = out[other]
                for j, scalar in enumerate(value):
                    total[j] += scalar
            out[key] = tuple(-scalar for scalar in total)
        assert set(out) == nodes
        return {key: value for key, value in out.items() if any(value)}

    def delete(self, request):
        requested = set(request)
        if not requested <= self.alive:
            raise ValueError("Requests must be fresh current identifiers")
        if len(requested) > self.h:
            raise ValueError("Budget exceeded")
        new_h = self.h - len(requested)
        coeff = {}
        for key, value in self.reconstruct().items():
            target = tuple(v for v in key if v not in requested)
            if len(target) <= new_h:
                old = coeff.get(target, (0,) * self.dim)
                coeff[target] = tuple(a+b for a, b in zip(old, value))
        self.alive -= requested
        self.meta = eligible(self.meta, self.alive, new_h, canonical_input=True)
        self.h = new_h
        basis = geometry(self.meta, self.h)[2]
        self.values = {key: coeff.get(key, (0,) * self.dim) for key in basis}
        # No full map, old basis, original record graph, or deleted value persists.

    def snapshot(self):
        return (tuple(sorted(self.alive)), self.h, self.dim,
                tuple(sorted((v, own, tuple(sorted(b)))
                             for v, (own, b) in self.meta.items())),
                tuple(sorted(self.values.items())))


def modular_rank(rows, prime):
    a = [[x % prime for x in row] for row in rows]
    rank = 0
    for j in range(len(a[0]) if a else 0):
        pivot = next((i for i in range(rank, len(a)) if a[i][j]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        inv = pow(a[rank][j], -1, prime)
        a[rank] = [(x*inv) % prime for x in a[rank]]
        for i in range(rank+1, len(a)):
            c = a[i][j]
            a[i] = [(x-c*y) % prime for x, y in zip(a[i], a[rank])]
        rank += 1
    return rank


def record_graph(n, mask):
    b = [set() for _ in range(n)]
    for bit, (u, v) in enumerate(combinations(range(n), 2)):
        if (mask >> bit) & 1:
            b[v].add(u)
    return {v: (v, frozenset(b[v])) for v in range(n)}


def assert_state(state, records, stats, original_alive, deleted):
    fresh = GaugeState(records, original_alive-set(deleted), state.h, stats, state.dim)
    assert state.snapshot() == fresh.snapshot()
    assert state.reconstruct() == full_coeff(fresh.meta, fresh.h, stats, fresh.dim)
    p, omit, basis, _ = geometry(state.meta, state.h)
    assert len(state.reconstruct()) <= len(p) <= 2*len(basis)


def run():
    start, rng = time.time(), random.Random(20261003)
    counts = dict(graphs=0, graph_horizons=0, field_rank_checks=0,
                  exhaustive_sequences=0, exact_rebuild_comparisons=0,
                  random_document_sequences=0, random_source_sequences=0,
                  source_rank_checks=0, batch_order_checks=0)
    for n in range(6):
        for mask in range(1 << (n*(n-1)//2)):
            records, alive = record_graph(n, mask), set(range(n))
            stats = {v: (rng.randrange(-5, 6), rng.randrange(-5, 6), 1)
                     for v in records}
            counts['graphs'] += 1
            for h in range(n+1):
                state = GaugeState(records, alive, h, stats, 3)
                nodes, omitted, basis, edges = geometry(state.meta, h)
                rows = [[int(tail == key)-int(head == key)
                         for v, tail, head in edges] for key in sorted(nodes)]
                for prime in (2, 3, 101):
                    assert modular_rank(rows, prime) == len(basis)
                    counts['field_rank_checks'] += 1
                assert len(nodes) <= 2*len(basis)
                assert_state(state, records, stats, alive, set())
                counts['graph_horizons'] += 1
                if n <= 4:
                    for length in range(1, h+1):
                        for order in permutations(range(n), length):
                            current = GaugeState(records, alive, h, stats, 3)
                            deleted = set()
                            for u in order:
                                current.delete({u})
                                deleted.add(u)
                                assert_state(current, records, stats, alive, deleted)
                                counts['exact_rebuild_comparisons'] += 1
                            batched = GaugeState(records, alive, h, stats, 3)
                            batched.delete(deleted)
                            assert batched.snapshot() == current.snapshot()
                            counts['batch_order_checks'] += 1
                            counts['exhaustive_sequences'] += 1
    for source_mode in (False, True):
        for trial in range(180):
            n = rng.randrange(2, 25)
            nvar = rng.randrange(1, n+1) if source_mode else n
            owner = {v: rng.randrange(nvar) if source_mode else v for v in range(n)}
            records = {}
            for v in range(n):
                b = frozenset(owner[u] for u in range(v) if rng.random() < .3)
                records[v] = owner[v], b
            alive = set(range(nvar))
            h = rng.randrange(nvar+1)
            stats = {v: tuple(rng.randrange(-10, 11) for _ in range(4))
                     for v in range(n)}
            state = GaugeState(records, alive, h, stats, 4)
            if source_mode:
                nodes, omitted, basis, edges = geometry(state.meta, h)
                rows = [[int(tail == key)-int(head == key)
                         for v, tail, head in edges] for key in sorted(nodes)]
                for prime in (2, 3, 101):
                    assert modular_rank(rows, prime) == len(basis)
                    counts['source_rank_checks'] += 1
            order = rng.sample(sorted(alive), h)
            deleted = set()
            while order:
                amount = rng.randrange(1, min(4, len(order))+1)
                request, order = set(order[:amount]), order[amount:]
                oldrank = len(state.values)
                state.delete(request)
                assert len(state.values) <= oldrank
                deleted |= request
                assert_state(state, records, stats, alive, deleted)
                counts['exact_rebuild_comparisons'] += 1
            counts['random_source_sequences' if source_mode else
                   'random_document_sequences'] += 1
    tight = []
    for q in (3, 4, 8, 16, 32, 64):
        records = {i: (i, frozenset()) for i in range(q)}
        vectors = {i: {i: 1.0} for i in range(q)}
        for i, j in combinations(range(q), 2):
            v = len(records)
            records[v] = v, frozenset((i, j))
            vectors[v] = {i: 1/math.sqrt(2), j: 1/math.sqrt(2)}
        stats = {v: (1,) for v in records}
        state = GaugeState(records, set(records), 3, stats, 1)
        nodes, omitted, basis, edges = geometry(state.meta, 3)
        m, p, r = q*(q-1)//2, len(nodes), len(basis)
        assert (p, r, len(state.reconstruct())) == (q+1+2*m, q+m, p)
        # All dot products for q<=16; larger cases use the already proved form.
        geom_checks = 0
        if q <= 16:
            for u, v in combinations(range(len(records)), 2):
                dot = sum(value*vectors[v].get(i, 0.)
                          for i, value in vectors[u].items())
                assert (dot > .6) == (u in records[v][1])
                geom_checks += 1
        tight.append(dict(anchors=q, records=q+m, nonzero_coefficients=p,
                          rank=r, ratio=p/r, cosine_pairs_checked=geom_checks))
    return {"status": "passed", "seed": 20261003, "arithmetic": "exact integers",
            "counts": counts, "sharp_fixed_cosine_family": tight,
            "seconds": round(time.time()-start, 3),
            "limitations": ["Finite implementation checks, not proofs",
                            "No natural-language corpus or neural training",
                            "Rank optimality counts arbitrary independent statistic coordinates",
                            "Graph metadata and floating point precision charged separately"]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='round3_compression_verification.json')
    args = parser.parse_args()
    result = run()
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
