#!/usr/bin/env python3
"""Finite checks for newly derived rank/source structure, using stdlib only.

These checks supplement proofs; they are not NLP experiments. Graphs, record
priority, source partition, and statistic values are fixed under restriction.
Run: python round2_structure_verification.py
"""

from __future__ import annotations

import itertools
import json
import math
import random
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


SEED = 3032026


def subsets(values, limit):
    values = tuple(sorted(values))
    for size in range(min(limit, len(values)) + 1):
        yield from (frozenset(x) for x in itertools.combinations(values, size))


def rational_rank(rows):
    a = [[Fraction(x) for x in row] for row in rows]
    rank = 0
    for column in range(len(a[0]) if a else 0):
        pivot = next((i for i in range(rank, len(a)) if a[i][column]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][column]
        a[rank] = [x / scale for x in a[rank]]
        for i in range(rank + 1, len(a)):
            scale = a[i][column]
            if scale:
                a[i] = [x - scale * y for x, y in zip(a[i], a[rank])]
        rank += 1
    return rank


def rank_mod_two(rows):
    pivots = {}
    for row in rows:
        bits = sum((int(x) & 1) << i for i, x in enumerate(row))
        while bits:
            pivot = bits.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = bits
                break
            bits ^= pivots[pivot]
    return len(pivots)


def record_rank_curve(blockers, max_budget, unknown=None):
    """Reduced-signature algorithm; no query matrix or per-budget DSU."""
    if unknown is None:
        unknown = set(range(len(blockers)))
    head_root = {}
    counts = [0] * (max_budget + 1)
    roots = [set() for _ in range(max_budget + 1)]
    for v, b in enumerate(blockers):
        if v not in unknown or len(b) > max_budget:
            continue
        root = head_root.get(b, b)
        counts[len(b)] += 1
        roots[len(b)].add(root)
        if len(b) < max_budget:
            head_root[b | {v}] = root
    prefix = 0
    result = []
    for k in range(max_budget + 1):
        result.append(prefix + len(roots[k]))
        prefix += counts[k]
    return result


class DSU:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, value):
        if value not in self.parent:
            self.parent[value] = value
            self.size[value] = 1
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return False
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return True


def source_blockers(blockers, source):
    return [frozenset(source[u] for u in b) for b in blockers]


def source_rank_curve(blockers, source, max_budget):
    """Incidence-DSU sweep, including source columns that cancel to zero."""
    sb = source_blockers(blockers, source)
    buckets = defaultdict(list)
    for v, b in enumerate(sb):
        if source[v] not in b and len(b) <= max_budget:
            buckets[len(b)].append((b, b | {source[v]}))
    dsu = DSU()
    interior_rank = 0
    result = []
    for k in range(max_budget + 1):
        if k:
            for tail, head in buckets[k - 1]:
                interior_rank += int(dsu.union(tail, head))
        boundary_roots = {dsu.find(tail) for tail, _ in buckets[k]}
        result.append(interior_rank + len(boundary_roots))
    return result


def source_coefficients(blockers, source, values, horizon, live=None):
    """Build canonical scalar coefficient map by source variables."""
    if live is None:
        live = frozenset(range(len(blockers)))
    out = defaultdict(int)
    for v in live:
        b = frozenset(source[u] for u in blockers[v] if u in live)
        head = b | {source[v]}
        if len(b) <= horizon:
            out[b] += values[v]
        if len(head) <= horizon:
            out[head] -= values[v]
    return {key: value for key, value in out.items() if value}


def substitute(coefficients, deleted, horizon):
    new_horizon = horizon - len(deleted)
    assert new_horizon >= 0
    out = defaultdict(int)
    for key, value in coefficients.items():
        new_key = key - deleted
        if len(new_key) <= new_horizon:
            out[new_key] += value
    return {key: value for key, value in out.items() if value}


def query(coefficients, deleted):
    return sum(value for key, value in coefficients.items() if key <= deleted)


def direct_source_selection(blockers, source, deleted):
    live = {v for v in range(len(blockers)) if source[v] not in deleted}
    return {v for v in live if not (blockers[v] & live)}


def run():
    started = time.perf_counter()
    rng = random.Random(SEED)
    counts = defaultdict(int)
    for n in range(8):
        for _ in range(30):
            blockers = [
                frozenset(u for u in range(v) if rng.random() < 0.4)
                for v in range(n)
            ]
            counts["random_ordered_graphs"] += 1
            unknown = {v for v in range(n) if rng.random() < 0.6}
            for unknown_columns in (set(range(n)), unknown):
                curve = record_rank_curve(blockers, n, unknown_columns)
                columns = sorted(unknown_columns)
                for k in range(n + 1):
                    rows = [
                        [int(v not in deleted and blockers[v] <= deleted) for v in columns]
                        for deleted in subsets(range(n), k)
                    ]
                    assert rational_rank(rows) == curve[k]
                    assert rank_mod_two(rows) == curve[k]
                    counts["reduced_signature_rational_rank_checks"] += 1
                    counts["reduced_signature_binary_rank_checks"] += 1

            if n == 0:
                m, source = 0, []
            else:
                m = rng.randint(1, min(n, 4))
                source = list(range(m)) + [rng.randrange(m) for _ in range(n - m)]
                rng.shuffle(source)
            source_sets = source_blockers(blockers, source)
            values = [rng.randint(-10, 10) for _ in range(n)]
            curve = source_rank_curve(blockers, source, m)
            for k in range(m + 1):
                requests = list(subsets(range(m), k))
                rows = [
                    [int(source[v] not in h and source_sets[v] <= h) for v in range(n)]
                    for h in requests
                ]
                assert rational_rank(rows) == curve[k]
                assert rank_mod_two(rows) == curve[k]
                counts["source_dsu_rational_rank_checks"] += 1
                counts["source_dsu_binary_rank_checks"] += 1
                coefficients = source_coefficients(blockers, source, values, k)
                for h in requests:
                    selected = direct_source_selection(blockers, source, h)
                    assert query(coefficients, h) == sum(values[v] for v in selected)
                    counts["source_polynomial_query_checks"] += 1
                    for v in range(n):
                        if source[v] in source_sets[v]:
                            assert v not in selected
                            counts["same_source_blocker_zero_checks"] += 1
                    live = frozenset(v for v in range(n) if source[v] not in h)
                    updated = substitute(coefficients, h, k)
                    rebuilt = source_coefficients(blockers, source, values, k - len(h), live)
                    assert updated == rebuilt
                    counts["source_canonical_rebuild_checks"] += 1

                    remaining_sources = set(range(m)) - h
                    for j in subsets(remaining_sources, k - len(h)):
                        sequential = substitute(updated, j, k - len(h))
                        direct = substitute(coefficients, h | j, k)
                        assert sequential == direct
                        counts["source_path_independence_checks"] += 1

            # Exact logical coefficient-key count cannot grow on substitution.
            coefficients = source_coefficients(blockers, source, values, m)
            for h in subsets(range(m), m):
                assert len(substitute(coefficients, h, m)) <= len(coefficients)
                counts["source_logical_key_count_checks"] += 1

    # Explicit source diamond: a(A), b(B), w(B), z(A), edges a-w and b-z.
    diamond_b = [frozenset(), frozenset(), frozenset({0}), frozenset({1})]
    diamond_s = [0, 1, 1, 0]
    diamond_curve = source_rank_curve(diamond_b, diamond_s, 2)
    assert diamond_curve == [1, 3, 3]
    diamond_rows = [
        [int(diamond_s[v] not in h and source_blockers(diamond_b, diamond_s)[v] <= h)
         for v in range(4)]
        for h in subsets(range(2), 2)
    ]
    assert diamond_rows == [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 0, 0]]
    relation = [1, -1, 1, -1]
    assert all(sum(x * c for x, c in zip(row, relation)) == 0 for row in diamond_rows)
    counts["explicit_source_diamond_checks"] = 1

    # Released SemDeDup-style zero-masked upper-triangle maximum uses strict >.
    # For tau>=0, padded zeros do not alter whether any earlier cosine exceeds tau.
    # For tau<0, even the singleton zero-padded matrix is a counterexample.
    for n in range(1, 13):
        for _ in range(10):
            angles = [rng.uniform(-math.pi, math.pi) for _ in range(n)]
            similarity = [[math.cos(a - b) for b in angles] for a in angles]
            thresholds = [0.0, 0.2, 0.8, 1.0]
            if n > 1 and similarity[0][1] >= 0:
                thresholds.append(similarity[0][1])  # exact floating-point tie
            for tau in thresholds:
                matrix_removed = [max([0.0] + [similarity[i][j] for i in range(j)]) > tau
                                  for j in range(n)]
                graph_removed = [any(similarity[i][j] > tau for i in range(j))
                                 for j in range(n)]
                assert matrix_removed == graph_removed
                counts["semdedup_nonnegative_strict_threshold_checks"] += 1
    assert (0.0 > -0.1) is True
    assert any([]) is False
    counts["semdedup_negative_threshold_counterexamples"] = 1

    report = {
        "status": "all_passed",
        "seed": SEED,
        "scope": "fixed graph/order, partition sources, exact logical scalar states",
        "counts": dict(sorted(counts.items())),
        "source_diamond_rank_curve": diamond_curve,
        "semdedup_equivalence_condition": "strict similarity > tau with conventional tau in [0,1]",
        "limitations": [
            "Finite randomized-graph checks supplement proofs; not exhaustive universal proof.",
            "No NLP corpus or model training experiment is performed.",
            "Logical-map equality does not assert history-free machine handles or whole-system erasure.",
            "Source membership is a partition; overlapping ownership is outside this theorem.",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 6),
    }
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
