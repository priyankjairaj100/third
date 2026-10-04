"""Bounded exact algebra checks; no empirical or synthetic corpus experiment."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))


def norm2(matrix):
    return sum((x * x for row in matrix for x in row), F(0))


def solve(a, b):
    n, q = len(a), len(b[0])
    rows = [list(x) + list(y) for x, y in zip(a, b)]
    for col in range(n):
        pivot = next(i for i in range(col, n) if rows[i][col])
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [x / scale for x in rows[col]]
        for i in range(n):
            if i != col:
                scale = rows[i][col]
                rows[i] = [x - scale * y for x, y in zip(rows[i], rows[col])]
    return [row[n:n + q] for row in rows]


def rank(matrix):
    rows = [list(map(F, row)) for row in matrix]
    out = 0
    for col in range(len(rows[0])):
        pivot = next((i for i in range(out, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[out], rows[pivot] = rows[pivot], rows[out]
        scale = rows[out][col]
        rows[out] = [x / scale for x in rows[out]]
        for i in range(len(rows)):
            if i != out:
                scale = rows[i][col]
                rows[i] = [x - scale * y for x, y in zip(rows[i], rows[out])]
        out += 1
        if out == len(rows):
            break
    return out


def head(features, labels, kept, lam=F(1)):
    d, q = len(features[0]), len(labels[0])
    a = [[sum((features[v][i] * features[v][j] for v in kept), F(0)) + (lam * len(kept) if i == j else 0)
          for j in range(d)] for i in range(d)]
    b = [[sum((features[v][i] * labels[v][j] for v in kept), F(0)) for j in range(q)] for i in range(d)]
    return solve(a, b) if kept else [[F(0)] * q for _ in range(d)]


def selected(blockers, removed):
    return [v for v in range(len(blockers)) if v not in removed and blockers[v] <= removed]


def run_checks():
    checks = []
    counts = {"scalar_side_information_answers": 0, "frontier_graph_instances": 0,
              "low_budget_requests": 0, "isolated_hard_heads": 0, "rank_one_distance_identities": 0,
              "integer_codewords": 0}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    # Exact stacked-head rank does not apply unchanged when deleted labels are
    # supplied for free with the request. One stored sum suffices in this case.
    n = 3
    requests = [frozenset()] + [frozenset([i]) for i in range(n)]
    maps = [[F(0) if i in removed else F(1, 2 * (n - len(removed))) for i in range(n)] for removed in requests]
    check("stacked_identifier_only_head_rank_is_three", rank(maps) == 3)
    for labels in product((0, 1), repeat=n):
        total = sum(labels)
        for removed, row in zip(requests, maps):
            true = dot(row, labels)
            with_payload = F(total - sum(labels[i] for i in removed), 2 * (n - len(removed)))
            assert true == with_payload
            counts["scalar_side_information_answers"] += 1
    check("one_sum_and_forget_payload_answer_all_small_queries", counts["scalar_side_information_answers"] == 32)
    check("sum_state_has_four_possibilities_not_eight", len({sum(y) for y in product((0, 1), repeat=n)}) == 4)

    # Different realizable moments can yield the same sole requested head.
    x1, y1 = [[F(1), F(0)], [F(0), F(1)]], [[F(3, 10)], [F(3, 10)]]
    x2, y2 = [[F(1, 2), F(0)], [F(0), F(1)]], [[F(9, 20)], [F(3, 10)]]
    check("distinct_realizable_grams_same_initial_head", x1 != x2 and head(x1, y1, [0, 1]) == head(x2, y2, [0, 1]) == [[F(1, 10)], [F(1, 10)]])
    check("same_initial_head_not_sufficient_after_one_deletion", head(x1, y1, [0]) != head(x2, y2, [0]))

    # Shared unit orthogonal features: exact rank is n, while every binary-label
    # target is within 1/5 of zero for n=25 and at most one deletion.
    n = 25
    check("shared_unit_feature_zero_state_approximation", all(F(s, (s + 1) ** 2) < F(1, 25) for s in (n, n - 1)))
    check("initial_exact_map_in_that_example_has_full_rank", rank([[F(i == j, n + 1) for j in range(n)] for i in range(n)]) == n)
    # One real/integer coordinate is not one bounded word.
    for bits in product((0, 1), repeat=8):
        message = sum(bit << i for i, bit in enumerate(bits))
        assert tuple((message >> i) & 1 for i in range(8)) == bits
        counts["integer_codewords"] += 1
    check("one_integer_coordinate_still_carries_eight_bits", counts["integer_codewords"] == 256)

    # Review the owner's original radical construction through conservative
    # rational margin inequalities, without substituting rounded cosines.
    r = F(1, 100)
    check("original_candidate_pair_margin_survives_two_caps", F(3, 8) + 2 * r < F(2, 5))
    check("rational_lower_bound_on_sqrt_three", F(433, 250) ** 2 < 3)
    check("original_common_blocker_margin_survives_cap", F(433, 1000) - r > F(2, 5))
    check("original_own_private_blocker_margin_survives_cap", F(3, 4) - r > F(2, 5))
    check("original_wrong_private_blocker_nonedge_survives_cap", F(1, 8) + r < F(2, 5))
    check("original_anchor_nonedge_survives_cap", r < F(2, 5))

    # A separate, explicitly rational realization checks the full ridge algebra
    # and raw-blocker semantics. It is not a claim about natural embeddings or
    # a finite check of the logarithmic-dimensional code-existence proof.
    m, b, d, q = 3, 2, 5, 2
    e = [[F(i == j) for j in range(d)] for i in range(d)]
    anchor = e[0]
    common = [[F(3, 5) * e[0][j] + F(4, 5) * e[1][j] for j in range(d)] for _ in range(b)]
    private = [[F(3, 5) * e[0][j] + F(4, 5) * e[i + 2][j] for j in range(d)] for i in range(m)]
    base = [[F(3, 5) * e[1][j] + F(4, 5) * e[i + 2][j] for j in range(d)] for i in range(m)]
    public_features = [anchor] + common + private
    expected_blocks = None
    for chart_values in product((F(-1, 1000), F(0), F(1, 1000)), repeat=m):
        hidden, betas, ys = [], [], []
        for i, t in enumerate(chart_values):
            cosine, sine = (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)
            w = [cosine * base[i][j] + sine * anchor[j] for j in range(d)]
            assert dot(w, w) == 1 and dot([x - z for x, z in zip(w, base[i])], [x - z for x, z in zip(w, base[i])]) < r * r
            beta = [F(1, 6), F(1, 3)] if t <= 0 else [F(1, 3), F(1, 4)]
            cinvw = [w[0] / 3] + [x / 2 for x in w[1:]]
            scalar = 1 + dot(w, cinvw)
            y = [scalar * x for x in beta]
            assert all(0 < x <= F(1, 2) for x in y)
            hidden.append(w); betas.append(beta); ys.append(y)
        features = public_features + hidden
        labels = [[F(0)] * q for _ in public_features] + ys
        blockers = [{u for u in range(v) if dot(features[u], features[v]) > F(2, 5)} for v in range(len(features))]
        if expected_blocks is None:
            expected_blocks = blockers
        assert blockers == expected_blocks
        counts["frontier_graph_instances"] += 1
        assert selected(blockers, set()) == [0]
        assert head(features, labels, [0]) == [[F(0)] * q for _ in range(d)]
        for size in range(b + 1):
            for removed_tuple in combinations(range(len(features)), size):
                kept = selected(blockers, set(removed_tuple))
                assert all(v < len(public_features) for v in kept)
                counts["low_budget_requests"] += 1
        for i, w in enumerate(hidden):
            removed = set(range(1, b + 1)) | {1 + b + i}
            wi = len(public_features) + i
            kept = selected(blockers, removed)
            assert kept == [0, wi]
            for other in range(m):
                if other != i:
                    pj = 1 + b + other
                    wj = len(public_features) + other
                    assert pj not in removed and pj not in kept and pj in blockers[wj]
            # The requested blockers are fixed and carry zero labels.
            assert all(v < len(public_features) and labels[v] == [F(0)] * q for v in removed)
            cinvw = [w[0] / 3] + [x / 2 for x in w[1:]]
            expected = [[x * y for y in betas[i]] for x in cinvw]
            assert head(features, labels, kept) == expected
            counts["isolated_hard_heads"] += 1
            # C W = w beta^T gives the claimed unique positive-label chart.
            recovered = [[value * (3 if j == 0 else 2) for value in row] for j, row in enumerate(expected)]
            assert [row[0] / betas[i][0] for row in recovered] == w
            assert all(sum((row[j] ** 2 for row in recovered), F(0)) == betas[i][j] ** 2 for j in range(q))
        for i, j in combinations(range(m), 2):
            w, wp, beta, bp = hidden[i], hidden[j], betas[i], betas[j]
            delta = [[x * y - xp * yp for y, yp in zip(beta, bp)] for x, xp in zip(w, wp)]
            rhs = sum(((x - y) ** 2 for x, y in zip(beta, bp)), F(0)) + dot(beta, bp) * sum(((x - y) ** 2 for x, y in zip(w, wp)), F(0))
            assert norm2(delta) == rhs
            transformed = [[value / (3 if row == 0 else 2) for value in entries] for row, entries in enumerate(delta)]
            assert norm2(transformed) >= rhs / 9
            counts["rank_one_distance_identities"] += 1
    check("private_feature_caps_preserve_one_graph", counts["frontier_graph_instances"] == 27)
    check("finite_low_budget_exhaustion_contains_no_hidden_candidate", counts["low_budget_requests"] == 1242)
    check("all_hard_queries_leave_anchor_and_only_requested_candidate", counts["isolated_hard_heads"] == 81)
    check("rank_one_distance_and_C_inverse_lower_bound_exact", counts["rank_one_distance_identities"] == 81)
    c, beta0 = F(3), F(1, 6)
    eta = r * beta0 / (32 * c)
    feature_separation = 4 * c * eta / beta0
    label_separation = 4 * c * eta
    check("explicit_feature_packing_volume_ratio_four", (r / 2) / feature_separation == 4)
    check("explicit_label_grid_has_801_levels", int(beta0 / label_separation) + 1 == 801)
    check("explicit_codeword_head_separation_exceeds_twice_error", beta0 * feature_separation / c == label_separation / c == 4 * eta)

    source_paths = ["round2_lower_bounds.md", "round2_structural.md", "round3_lower.md", "round3_compression.md",
                    "empirical_execution/memory_theory_addendum.tex", "empirical_execution/canonical_theory_addendum.tex",
                    "empirical_execution/certified_ridge_contract.txt"]
    return {"schema": "ccu-phase9-theory-review-checks-1", "status": "passed", "checks": checks, "passed": len(checks), "case_counts": counts,
            "scope": "bounded_exact_algebra_and_counterexample_checks_only", "empirical_benchmark": False,
            "global_novelty_proved": False, "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reviewed_prior_sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        try:
            result = run_checks()
        except Exception as exc:
            json.dump({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "scope": "algebra_only"}, stream, indent=2)
            stream.write("\n")
            raise
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "case_counts": result["case_counts"], "scope": result["scope"]}))


if __name__ == "__main__":
    main()
