"""Independent bounded algebra/format review; not empirical data or a benchmark."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import struct


def dot(x, y):
    return sum((a * b for a, b in zip(x, y)), F(0))


def edge(x, y):
    """Exact positive-threshold cosine decision without square roots."""
    xy = dot(x, y)
    return xy > 0 and 25 * xy * xy > 4 * dot(x, x) * dot(y, y)


def run():
    checks, counts = [], {}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def count(name, n=1):
        counts[name] = counts.get(name, 0) + n

    # Representability is verified algebraically and by an actual binary32
    # round trip. This is scalar format verification, not a large array run.
    for rk, rd in product(range(11), repeat=2):
        k, d_hidden = 4**rk, 4**rd
        if 2 + k + d_hidden > 2**20:
            continue
        for value in (F(1, 2), F(7, 8 * 2**rk), F(1, 128 * 2**rd)):
            for sign in (-1, 1):
                x = value * sign
                roundtrip = struct.unpack(">f", struct.pack(">f", float(x)))[0]
                assert F(roundtrip) == x
                count("binary32_scalar_roundtrips")
        count("admitted_dimension_pairs")
    check("all_admitted_dyadic_coordinate_types_binary32_exact", counts["admitted_dimension_pairs"] > 0)

    n_public, n_hidden = F(65, 64), F(16641, 16384)
    # Threshold inequalities are squared only after establishing positivity.
    check("anchor_blocker_edge_margin", F(1, 4) > F(4, 25) * n_public)
    check("common_private_nonedge", F(1, 4) ** 2 < F(4, 25) * n_public**2)
    check("common_candidate_edge_margin", F(7, 16) ** 2 > F(4, 25) * n_public * n_hidden)
    check("own_private_candidate_edge_margin", F(49, 64) ** 2 > F(4, 25) * n_public * n_hidden)
    check("wrong_private_candidate_nonedge", F(49, 384) ** 2 < F(4, 25) * n_public * n_hidden)
    check("private_private_nonedge", (F(1, 4) + F(49, 384)) / n_public < F(2, 5))
    check("candidate_candidate_nonedge_even_with_identical_private_signs", (F(1, 4) + F(49, 384) + F(1, 16384)) / n_hidden < F(2, 5))

    # m=2, b=2, k=D=4. Public Hadamard signatures are orthogonal.
    # All 256 hidden sign assignments are mathematical family instances.
    d, b, k, hidden, horizon = 10, 2, 4, 4, 3
    a = (F(1),) + (F(0),) * (d - 1)
    s = (F(1, 2), F(7, 8)) + (F(0),) * (d - 2)
    signatures = [(1, 1, 1, 1), (1, 1, -1, -1)]
    p = [(F(1, 2), F(0)) + tuple(F(7 * t, 16) for t in u) + (F(0),) * hidden for u in signatures]
    blockers = (frozenset((1, 2, 3)), frozenset((1, 2, 4)))
    universe = tuple(range(7))
    requests = [frozenset(x) for n in range(horizon + 1) for x in combinations(universe, n)]
    head_families = set()
    state_images = {request: set() for request in requests}
    fixed_graph = None
    for bits in product((0, 1), repeat=2 * hidden):
        signs = [tuple(1 if bit else -1 for bit in bits[i * hidden:(i + 1) * hidden]) for i in range(2)]
        w = [(F(0), F(1, 2)) + tuple(F(7 * t, 16) for t in signatures[i]) + tuple(F(t, 256) for t in signs[i]) for i in range(2)]
        x = [a, s, s, p[0], p[1], w[0], w[1]]
        graph = tuple(frozenset(j for j in range(i) if edge(x[i], x[j])) for i in universe)
        assert graph[5:] == blockers
        assert dot(w[0], w[0]) == dot(w[1], w[1]) == n_hidden
        if fixed_graph is None:
            fixed_graph = graph
        assert graph == fixed_graph
        count("full_private_sign_assignments")
        heads = []
        for i in range(2):
            theta = tuple(v / (n_hidden + 2) for v in w[i])
            # Fresh exact normal equations with two selected records and λ=1.
            residual = tuple(a[j] * dot(a, theta) + w[i][j] * dot(w[i], theta) + 2 * theta[j] - w[i][j] for j in range(d))
            assert not any(residual)
            assert tuple(1 if value > 0 else -1 for value in theta[-hidden:]) == signs[i]
            heads.append(theta)
            count("independent_exact_normal_equations")
        head_families.add(tuple(heads))
        for removed in requests:
            selected = [i for i in universe if i not in removed and graph[i] <= removed]
            admitted = [i for i in range(2) if i + 5 in selected]
            if len(removed) <= b:
                assert not admitted
                count("low_budget_queries")
            assert len(admitted) <= 1
            if admitted:
                i = admitted[0]
                assert removed == blockers[i] and selected == [0, i + 5]
                count("hard_pair_queries")
            eligible = tuple(i for i in range(2) if i + 5 not in removed and len(blockers[i] - removed) <= horizon - len(removed))
            simplified = tuple(i for i in range(2) if i + 5 not in removed and removed <= blockers[i])
            assert eligible == simplified
            private_state = tuple(bits[i * hidden:(i + 1) * hidden] for i in eligible)
            state_images[removed].add(private_state)
            if len(removed) < horizon:
                for next_id in set(universe) - removed:
                    after = removed | {next_id}
                    after_eligible = tuple(i for i in range(2) if i + 5 not in after and after <= blockers[i])
                    assert set(after_eligible) <= set(eligible)
                    transitioned = tuple(private_state[eligible.index(i)] for i in after_eligible)
                    fresh = tuple(bits[i * hidden:(i + 1) * hidden] for i in after_eligible)
                    assert transitioned == fresh
                    count("private_state_drop_transition_checks")
    check("graph_independent_of_all_private_signs", counts["full_private_sign_assignments"] == 256)
    check("hard_probe_families_identify_every_private_bit", len(head_families) == 2**8)
    check("exact_heads_satisfy_actual_normal_equations", counts["independent_exact_normal_equations"] == 512)
    check("all_low_budget_queries_have_zero_cross_moment", counts["low_budget_queries"] == 256 * (1 + 7 + 21))
    check("every_admitting_query_is_an_isolated_hard_pair", counts["hard_pair_queries"] == 512)
    check("eligibility_simplification_and_monotone_private_drop", counts["private_state_drop_transition_checks"] > 0)
    for removed, images in state_images.items():
        eligible = [i for i in range(2) if i + 5 not in removed and removed <= blockers[i]]
        assert len(images) == 2 ** (len(eligible) * hidden)
        count("conditional_state_alphabets")
    check("conditional_private_state_cardinalities_exact", counts["conditional_state_alphabets"] == len(requests))

    # Posterior variance identity fixes the approximate theorem's D scaling.
    alpha, t = F(1, 128), n_hidden + 2
    probs = (F(0), F(1, 8), F(1, 4), F(1, 2))
    hidden_mse = sum((4 * alpha**2 / (t**2 * hidden) * q * (1 - q) for q in probs), F(0))
    check("posterior_bit_variance_has_correct_dimension_normalization", t**2 * hidden_mse / (4 * alpha**2) == sum((q * (1 - q) for q in probs), F(0)) / hidden)
    check("uniform_certified_release_radius_is_below_each_hidden_coordinate", n_hidden + 2 < 4 and F(1, 2**20) < alpha / (4 * 2**10))
    for radius in (F(0), F(1, 8), F(1, 4), F(1, 2)):
        for error_fraction in (F(0), radius / 2, radius):
            squared_error = (1 - error_fraction) * (2 * radius)**2 + error_fraction * (2 - 2 * radius)**2
            assert squared_error == 4 * (radius**2 + error_fraction * (1 - 2 * radius))
            assert squared_error <= 4 * radius * (1 - radius)
            count("covering_code_reconstruction_distortion_cases")
    check("shrunk_covering_code_reconstruction_distortion", counts["covering_code_reconstruction_distortion_cases"] == 12)

    # A numerical vector is not the same release contract as its IEEE bytes.
    # At large, exactly representable λ every coordinate underflows in FP32.
    # Signed zero still exposes signs if raw bytes are the official target.
    lam = F(2**150)
    small = F(1, 256) / (n_hidden + 2 * lam)
    pos = struct.pack(">f", float(small))
    neg = struct.pack(">f", float(-small))
    check("rounded_numerical_vectors_can_collapse", struct.unpack(">f", pos)[0] == struct.unpack(">f", neg)[0] == 0)
    check("signed_zero_byte_targets_do_not_collapse", pos != neg)
    check("public_lambda_in_release_counterexample_binary64_exact", F(float(lam)) == lam)

    return {"schema": "ccu-phase10-independent-algebra-checks-1", "status": "passed", "checks": checks,
            "assertions_passed": len(checks), "case_counts": counts, "empirical_benchmark": False,
            "scope": "bounded exact algebra and scalar IEEE-format checks only; no entropy or novelty proof by enumeration"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve()
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    try:
        result = run()
    except Exception as exc:
        result = {"schema": "ccu-phase10-independent-algebra-checks-1", "status": "failed", "error": repr(exc), "empirical_benchmark": False}
    result["checker_sha256"] = digest
    if source.read_bytes() != source_bytes:
        result["status"] = "failed"
        result["error"] = "checker source changed during execution"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.out.with_name(args.out.stem + "_checker.py")
    with snapshot.open("xb") as f:
        f.write(source_bytes)
    result["checker_snapshot"] = str(snapshot)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"status": result["status"], "assertions_passed": result.get("assertions_passed"), "out": str(args.out)}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
