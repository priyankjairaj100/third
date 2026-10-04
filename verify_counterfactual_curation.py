#!/usr/bin/env python3
"""Reproducible checks for bounded-deletion counterfactual curation.

Selector studied (not greedy maximal independent set): vertices have a fixed
order; v survives deletion F iff v not in F and every earlier neighbor of v
is in F. Graph/embedding/order are frozen. A blocker set is denoted B(v).

For additive scalar or vector records t_v, the selected aggregate is
  P(x) = sum_v t_v (1-x_v) product_{u in B(v)} x_u.
The coefficient of x_J is
  alpha_J = sum_{B(v)=J} t_v - sum_{B(v) union {v}=J} t_v.
Keeping degrees <= k gives exact answers for every |F| <= k. After deleting
F, substitute x_u=1 for u in F, collect equal remaining monomials, and keep
degrees <= k-|F|. The resulting coefficients equal rebuilding on retained
vertices, with that remaining horizon.

Let A[J,v] be the coefficient of t_v in alpha_J, and Q[F,v] be v's selection
indicator after F. Rows range over all subsets of cardinality <= k.
Then Q=Z A, where Z[F,J]=1[J subset F]. Z is unitriangular in any ordering by
cardinality and has a Mobius inverse over every field.

Each nonzero column of A is an edge-incidence column: +1 at B(v), -1 at
B(v) union {v}; a missing degree-(k+1) endpoint is a common ground vertex.
If S is the set of incident non-ground vertices and c is the number of
connected components not containing ground, rank(A)=|S|-c over every field.
Therefore this rank is the minimum number of reals in an exact LINEAR sketch
of arbitrary real scalar records; a row basis attains it. For arbitrary
exact deterministic sketches of records over F_q, state cardinality is at
least q**rank(A), i.e. storage at least ceil(rank(A)*log2(q)) bits. An arbitrary
real-valued nonlinear encoder has no analogous dimension bound without
regularity/precision restrictions (one real can encode many reals).

All checks are finite computational corroboration, not proofs. The script
uses only Python's standard library and NumPy. It does not test adaptive
embeddings, refitted clustering, approximate nearest neighbors, or greedy
MIS curation. Run:
  python verify_counterfactual_curation.py --output curation_verification.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from math import comb
from pathlib import Path
import platform
import time

import numpy as np


def masks(n: int, k: int):
    """All masks of size at most k, in cardinality then integer order."""
    return sorted((m for m in range(1 << n) if m.bit_count() <= k),
                  key=lambda m: (m.bit_count(), m))


def blockers_from_graph(n: int, graph_code: int):
    blockers = [0] * n
    for bit, (u, v) in enumerate(combinations(range(n), 2)):
        if (graph_code >> bit) & 1:
            blockers[v] |= 1 << u
    return blockers


def selected(blockers, deleted: int):
    return np.array([not ((deleted >> v) & 1) and (b & ~deleted) == 0
                     for v, b in enumerate(blockers)], dtype=np.int64)


def coefficient_map(blockers, k: int, retained_mask=None):
    """Integer coefficients of each original record, including zero columns.

    With retained_mask, refit the frozen graph selector on retained vertices.
    Deleted record columns remain zero so exact equality can be checked.
    """
    n = len(blockers)
    if retained_mask is None:
        retained_mask = (1 << n) - 1
    result = {}
    for v, old_b in enumerate(blockers):
        if not (retained_mask & (1 << v)):
            continue
        b = old_b & retained_mask
        for key, sign in ((b, 1), (b | (1 << v), -1)):
            if key.bit_count() <= k:
                if key not in result:
                    result[key] = np.zeros(n, dtype=np.int64)
                result[key][v] += sign
    return {key: value for key, value in result.items() if np.any(value)}


def update_coefficients(coeff, deleted: int, remaining_k: int):
    result = {}
    for key, value in coeff.items():
        new_key = key & ~deleted
        if new_key.bit_count() <= remaining_k:
            if new_key not in result:
                result[new_key] = np.zeros_like(value)
            result[new_key] += value
    return {key: value for key, value in result.items() if np.any(value)}


def maps_equal(left, right):
    return left.keys() == right.keys() and all(
        np.array_equal(left[key], right[key]) for key in left)


def modular_rank(matrix, prime: int):
    a = np.asarray(matrix, dtype=np.int64).copy() % prime
    if a.ndim != 2:
        raise ValueError("Matrix must be two-dimensional")
    row = 0
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(a[row:, col])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        a[[row, pivot]] = a[[pivot, row]]
        a[row] = a[row] * pow(int(a[row, col]), -1, prime) % prime
        for other in range(row + 1, len(a)):
            if a[other, col]:
                a[other] = (a[other] - a[other, col] * a[row]) % prime
        row += 1
        if row == len(a):
            break
    return row


def incidence_rank(blockers, k):
    """rank = incident non-ground vertices - components without ground."""
    ground = -1
    parent = {ground: ground}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        parent[find(a)] = find(b)

    for v, b in enumerate(blockers):
        degree = b.bit_count()
        if degree <= k:
            union(b, b | (1 << v) if degree < k else ground)
    ground_root = find(ground)
    non_ground_components = {find(node) for node in list(parent)
                             if find(node) != ground_root}
    return len(parent) - 1 - len(non_ground_components)


def row_matrix(coeff, rows, n):
    return np.array([coeff.get(key, np.zeros(n, dtype=np.int64))
                     for key in rows], dtype=np.int64).reshape(len(rows), n)


def exhaustive_checks(max_n):
    counts = defaultdict(int)
    rank_histogram = defaultdict(int)
    for n in range(max_n + 1):
        full_mask = (1 << n) - 1
        all_queries = list(range(1 << n))
        for code in range(1 << comb(n, 2)):
            counts["graphs"] += 1
            blockers = blockers_from_graph(n, code)
            q_full = np.array([selected(blockers, f) for f in all_queries],
                              dtype=np.int64).reshape(1 << n, n)
            for k in range(n + 1):
                counts["graph_horizon_pairs"] += 1
                rows = masks(n, k)
                coeff = coefficient_map(blockers, k)
                a = row_matrix(coeff, rows, n)
                zeta = np.array([[int((j & ~f) == 0) for j in rows]
                                 for f in rows], dtype=np.int64)
                q = q_full[rows]
                assert np.array_equal(zeta @ a, q), ("zeta", n, code, k)
                counts["query_indicator_vectors"] += len(rows)
                assert np.all(np.diag(zeta) == 1)
                assert not np.any(np.triu(zeta, 1))
                target_rank = incidence_rank(blockers, k)
                rank_histogram[f"n={n},k={k},rank={target_rank}"] += 1
                for prime in (2, 3, 5):
                    assert modular_rank(a, prime) == target_rank
                    assert modular_rank(q, prime) == target_rank
                    counts["modular_rank_equalities"] += 2
                for f in rows:
                    remaining_k = k - f.bit_count()
                    actual = update_coefficients(coeff, f, remaining_k)
                    target = coefficient_map(blockers, remaining_k,
                                             retained_mask=full_mask & ~f)
                    assert maps_equal(actual, target), ("update", n, code, k, f)
                    counts["canonical_sequential_update_equalities"] += 1
                if k == n:
                    assert target_rank == n
                    counts["full_horizon_full_rank"] += 1
    return {"counts": dict(counts), "rank_histogram": dict(rank_histogram)}


def rank_examples(max_n=24):
    examples = {"star": [], "clique": []}
    for n in range(3, max_n + 1):
        star = [0] + [1] * (n - 1)
        for k, expected in ((1, 2), (2, n)):
            coeff = coefficient_map(star, k)
            a = row_matrix(coeff, sorted(coeff), n)
            actual = incidence_rank(star, k)
            assert actual == expected == modular_rank(a, 101)
            examples["star"].append({"vertices": n, "leaves": n - 1,
                                     "horizon": k, "rank": actual})
    for n in range(1, max_n + 1):
        clique = [(1 << v) - 1 for v in range(n)]
        for k in range(n + 1):
            coeff = coefficient_map(clique, k)
            a = row_matrix(coeff, sorted(coeff), n)
            actual = incidence_rank(clique, k)
            expected = min(n, k + 1)
            assert actual == expected == modular_rank(a, 101)
            examples["clique"].append({"vertices": n, "horizon": k,
                                       "rank": actual})
    return examples


def random_checks(seed=20261003, trials=300):
    rng = np.random.default_rng(seed)
    counts = defaultdict(int)
    max_stat_error = 0.0
    max_model_error = 0.0
    max_sequential_stat_error = 0.0
    for _ in range(trials):
        n = int(rng.integers(6, 25))
        d = int(rng.integers(1, 7))
        k = int(rng.integers(0, min(n, 7) + 1))
        density = float(rng.uniform(0.05, 0.95))
        blockers = [sum(1 << u for u in range(v) if rng.random() < density)
                    for v in range(n)]
        coeff = coefficient_map(blockers, k)
        features = rng.normal(size=(n, d))
        labels = rng.normal(size=n)
        t = np.concatenate([
            np.einsum("vi,vj->vij", features, features).reshape(n, d*d),
            features * labels[:, None], np.ones((n, 1))], axis=1)
        sketch = {key: vector @ t for key, vector in coeff.items()}
        lam = 0.25
        for _query in range(24):
            r = int(rng.integers(0, k + 1))
            deleted = sum(1 << int(v) for v in rng.choice(n, r, replace=False))
            recovered = sum((value for key, value in sketch.items()
                             if (key & ~deleted) == 0), start=np.zeros(t.shape[1]))
            indicator = selected(blockers, deleted)
            direct = indicator @ t
            error = float(np.max(np.abs(recovered - direct)))
            max_stat_error = max(max_stat_error, error)
            assert np.allclose(recovered, direct, atol=1e-10, rtol=1e-10)
            gram = recovered[:d*d].reshape(d, d)
            response = recovered[d*d:d*d+d]
            theta = np.linalg.solve(gram + lam * np.eye(d), response)
            selected_x = features[indicator.astype(bool)]
            selected_y = labels[indicator.astype(bool)]
            oracle = np.linalg.solve(selected_x.T @ selected_x + lam*np.eye(d),
                                     selected_x.T @ selected_y)
            model_error = float(np.max(np.abs(theta - oracle)))
            max_model_error = max(max_model_error, model_error)
            assert np.allclose(theta, oracle, atol=1e-9, rtol=1e-9)
            counts["ridge_query_equalities"] += 1
        # A genuinely sequential request stream, consuming the horizon.
        active = dict(sketch)
        total_deleted = 0
        remaining = k
        sequence = rng.permutation(n)[:k]
        for raw_v in sequence:
            v = int(raw_v)
            request = 1 << v
            remaining -= 1
            total_deleted |= request
            active = update_coefficients(active, request, remaining)
            rebuilt = coefficient_map(blockers, remaining,
                                      ((1 << n) - 1) & ~total_deleted)
            rebuilt_stats = {key: value @ t for key, value in rebuilt.items()}
            # Floating cancellation can leave mathematically zero entries.
            for key in set(active) | set(rebuilt_stats):
                actual = active.get(key, np.zeros(t.shape[1]))
                expected = rebuilt_stats.get(key, np.zeros(t.shape[1]))
                error = float(np.max(np.abs(actual - expected)))
                max_sequential_stat_error = max(max_sequential_stat_error, error)
                assert np.allclose(actual, expected, atol=1e-10, rtol=1e-10)
            counts["sequential_ridge_state_equalities"] += 1
        counts["random_graphs"] += 1
    return {"seed": seed, "counts": dict(counts),
            "maximum_absolute_aggregate_error": max_stat_error,
            "maximum_absolute_ridge_parameter_error": max_model_error,
            "maximum_absolute_sequential_state_error": max_sequential_stat_error}


def examples_of_scope():
    # An earlier blocker can itself be excluded. This distinguishes the model
    # from greedy MIS and shows deletion of a never-trained point adds a point.
    path = [0, 1, 2]
    original = selected(path, 0).tolist()
    after_middle_delete = selected(path, 1 << 1).tolist()
    assert original == [1, 0, 0]
    assert after_middle_delete == [1, 0, 1]
    # Curation changes can be nonmonotone: a selected record is deleted, while
    # previously blocked retained records enter. Retained selected records
    # never leave under this frozen blocker selector.
    return {"path_edges": [[0, 1], [1, 2]], "original_selected": [0],
            "deleted_vertex": 1, "new_selected": [0, 2],
            "greedy_mis_original_selected_for_comparison": [0, 2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-exhaustive-n", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output", type=Path,
                        default=Path("curation_verification.json"))
    args = parser.parse_args()
    if not 0 <= args.max_exhaustive_n <= 6:
        parser.error("Exhaustive n must be between 0 and 6 (n=6 is costly).")
    started = time.monotonic()
    result = {"status": "passed", "python_version": platform.python_version(),
              "numpy_version": np.__version__,
              "scope": "Frozen ordered graph, all earlier neighbors are blockers",
              "exhaustive": exhaustive_checks(args.max_exhaustive_n),
              "rank_examples": rank_examples(),
              "random": random_checks(args.seed, args.random_trials),
              "scope_example": examples_of_scope()}
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items()
               if key not in ("exhaustive", "rank_examples", "scope_example")}
    summary["exhaustive_counts"] = result["exhaustive"]["counts"]
    summary["rank_example_count"] = sum(map(len, result["rank_examples"].values()))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
