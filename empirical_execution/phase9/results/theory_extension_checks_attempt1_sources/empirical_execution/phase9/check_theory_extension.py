"""Bounded exact-rational proof/software checks, never empirical evidence.

Only the Python standard library is needed. The theorems about topology,
packing existence, and entropy are proved in the companion text, not by this
finite checker. No frozen pipeline or corpus is imported or executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as Q
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
BOUND = [
    "empirical_execution/phase9/check_theory_extension.py",
    "empirical_execution/phase9/THEORY_EXTENSION.md",
    "empirical_execution/phase9/theory_extension.tex",
    "output/pdf/counterfactual_curation_theory.tex",
    "empirical_execution/memory_theory_addendum.tex",
    "empirical_execution/canonical_theory_addendum.tex",
    "round3_lower.md",
]


def dot(x, y):
    return sum((a * b for a, b in zip(x, y)), Q(0))


def vecadd(x, y):
    return [a + b for a, b in zip(x, y)]


def scale(a, x):
    return [a * b for b in x]


def basis(d, j):
    return [Q(int(i == j)) for i in range(d)]


def transpose(A):
    return [list(row) for row in zip(*A)]


def matmul(A, B):
    return [[dot(row, col) for col in transpose(B)] for row in A]


def outer(x, y):
    return [[a * b for b in y] for a in x]


def matsub(A, B):
    return [[a - b for a, b in zip(x, y)] for x, y in zip(A, B)]


def norm2(A):
    return sum((dot(row, row) for row in A), Q(0))


def rank(A):
    if not A:
        return 0
    T = [list(row) for row in A]
    r = 0
    for j in range(len(T[0])):
        pivot = next((i for i in range(r, len(T)) if T[i][j]), None)
        if pivot is None:
            continue
        T[r], T[pivot] = T[pivot], T[r]
        T[r] = [x / T[r][j] for x in T[r]]
        for i in range(len(T)):
            if i != r and T[i][j]:
                alpha = T[i][j]
                T[i] = [a - alpha * b for a, b in zip(T[i], T[r])]
        r += 1
        if r == len(T):
            break
    return r


def solve(A, B):
    n = len(A)
    T = [list(A[i]) + list(B[i]) for i in range(n)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if T[i][j])
        T[j], T[pivot] = T[pivot], T[j]
        divisor = T[j][j]
        T[j] = [x / divisor for x in T[j]]
        for i in range(n):
            if i != j and T[i][j]:
                alpha = T[i][j]
                T[i] = [a - alpha * b for a, b in zip(T[i], T[j])]
    return [row[n:] for row in T]


def subsets(n, h):
    for count in range(h + 1):
        for ids in combinations(range(n), count):
            yield frozenset(ids)


def selected(Z, F, tau=Q(2, 5)):
    # Direct pairwise definition, independent of saved blocker sets.
    return [v for v in range(len(Z)) if v not in F and
            not any(u not in F and dot(Z[u], Z[v]) > tau for u in range(v))]


def ridge(Z, Y, S, lam, d, q):
    if not S:
        return [[Q(0)] * q for _ in range(d)]
    A = [[sum((Z[v][i] * Z[v][j] for v in S), Q(0)) +
          (lam * len(S) if i == j else 0) for j in range(d)] for i in range(d)]
    H = [[sum((Z[v][i] * Y[v][j] for v in S), Q(0))
          for j in range(q)] for i in range(d)]
    return solve(A, H)


def family(m, b, q, lam, variant, extra=0):
    """Rational orthogonal signatures: a bounded case of the theorem.

    The logarithmic signature existence claim is not numerically tested.
    Rational stereographic points are exactly on the stated square-root
    chart, and permit all subsequent checks with Fraction arithmetic.
    """
    d = m + 2 + extra
    e = [basis(d, i) for i in range(d)]
    A, B = Q(8, 17), Q(15, 17)
    a = e[0]
    s = vecadd(scale(A, e[0]), scale(B, e[1]))
    centers = [vecadd(scale(A, e[1]), scale(B, e[i + 2])) for i in range(m)]
    private = [vecadd(scale(A, e[0]), scale(B, e[i + 2])) for i in range(m)]
    frames, W, T, Betas, Ys = [], [], [], [], []
    c = 1 + 2 * lam
    beta0 = lam / (2 * c)
    invdiag = [1 / c] + [1 / (2 * lam)] * (d - 1)
    for i, w0 in enumerate(centers):
        frame = [e[0], vecadd(scale(B, e[1]), scale(-A, e[i + 2]))]
        frame += [e[j] for j in range(2, d) if j != i + 2]
        z = [Q(0)] * (d - 1)
        if variant:
            z[(i + variant - 1) % (d - 1)] = Q((-1) ** (i + variant), 2000)
            if variant == 2:
                z[0] = Q(1, 3000)
        z2 = dot(z, z)
        coeff = (1 - z2) / (1 + z2)
        t = [2 * x / (1 + z2) for x in z]
        w = scale(coeff, w0)
        for tj, vj in zip(t, frame):
            w = vecadd(w, scale(tj, vj))
        beta = [beta0 * (Q(5, 4) + Q((i + j + variant) % 3, 8)) for j in range(q)]
        factor = 1 + sum((invdiag[j] * w[j] ** 2 for j in range(d)), Q(0))
        frames.append(frame)
        W.append(w)
        T.append(t)
        Betas.append(beta)
        Ys.append(scale(factor, beta))
    Z = [a] + [s] * b + private + W
    Y = [[Q(0)] * q for _ in range(1 + b + m)] + Ys
    return dict(m=m, b=b, q=q, d=d, lam=lam, c=c, beta0=beta0, Z=Z, Y=Y,
                a=a, centers=centers, frames=frames, W=W, T=T, beta=Betas,
                invdiag=invdiag, start=1 + b + m)


class Checks:
    def __init__(self):
        self.counts = Counter()

    def check(self, name, condition, context=None):
        if not condition:
            raise AssertionError({"check": name, "context": context})
        self.counts[name] += 1


def run_checks(checks):
    ch = checks.check
    r = Q(1, 100)
    ch("rational_common_margin", Q(120, 289) - Q(2, 5) == Q(22, 1445) > r)
    ch("rational_pair_margin", Q(2, 5) - Q(203, 578) == Q(141, 2890) > 2 * r)
    ch("private_cross_bound", Q(15, 17) ** 2 / 6 == Q(75, 578) < Q(2, 5) - r)
    ch("private_match_margin", Q(225, 289) - Q(2, 5) > r)
    ch("anchor_blocker_margin", Q(8, 17) > Q(2, 5))
    ch("common_private_nonedge", Q(64, 289) < Q(2, 5))
    cases = 0
    for m, b, q, lam, extra in product([2, 3], [1, 2], [1, 3], [Q(1, 2), Q(2)], [0]):
        reference = family(m, b, q, lam, 0, extra)
        saved_heads = []
        for variant in range(3):
            f = family(m, b, q, lam, variant, extra)
            cases += 1
            d, Z, Y, start, c = f["d"], f["Z"], f["Y"], f["start"], f["c"]
            fixed_payload = (Z[:start], Y[:start])
            ch("fixed_forgotten_payloads", fixed_payload == (reference["Z"][:start], reference["Y"][:start]))
            ch("initial_selected_anchor", selected(Z, frozenset()) == [0])
            ch("initial_head_zero", not norm2(ridge(Z, Y, [0], lam, d, q)))
            for v, zv in enumerate(Z):
                ch("exact_unit_feature", dot(zv, zv) == 1)
                actual_blockers = {u for u in range(v) if dot(Z[u], zv) > Q(2, 5)}
                ref_blockers = {u for u in range(v) if dot(reference["Z"][u], reference["Z"][v]) > Q(2, 5)}
                ch("whole_graph_preserved", actual_blockers == ref_blockers)
                if v >= start:
                    i = v - start
                    ch("exact_candidate_blockers", actual_blockers == set(range(1, b + 1)) | {1 + b + i})
            for i in range(m):
                w, w0, t, frame, beta = f["W"][i], f["centers"][i], f["T"][i], f["frames"][i], f["beta"][i]
                ch("feature_cap_displacement", dot(vecadd(w, scale(-1, w0)), vecadd(w, scale(-1, w0))) < r * r)
                ch("chart_ball", dot(t, t) < (r / 2) ** 2)
                ch("chart_reconstruction", [dot(v, w) for v in frame] == t)
                ch("tangent_orthonormal", all(dot(v, w0) == 0 for v in frame) and
                   all(dot(u, v) == int(j == k) for j, u in enumerate(frame) for k, v in enumerate(frame)))
                ch("response_coordinate_bound", all(0 < y <= Q(1, 2) for y in Y[start + i]))
                ch("beta_positive_interval", all(f["beta0"] < x < 2 * f["beta0"] for x in beta))
                F = frozenset(range(1, b + 1)) | {1 + b + i}
                ch("hard_selected_pair", selected(Z, F) == [0, start + i])
                expected = outer([w[j] * f["invdiag"][j] for j in range(d)], beta)
                actual = ridge(Z, Y, selected(Z, F), lam, d, q)
                ch("independent_exact_ridge_solve", actual == expected)
                transformed = [[x / f["invdiag"][j] for x in actual[j]] for j in range(d)]
                ch("ridge_rank_one_recovery", transformed == outer(w, beta))
                ch("positive_factor_identifiability", all(dot(col, col) == beta[j] ** 2 for j, col in enumerate(transpose(transformed))))
                if variant == 0:
                    saved_heads.append(actual)
                else:
                    ch("different_private_parameters_change_probe_head", actual != saved_heads[i])
                # Analytic Jacobian at cap center; columns are exact derivatives
                # of C^-1*w(t)*beta^T, for every independent chart coordinate.
                cols = []
                for v in frame:
                    cols.append([x for row in outer([v[j] * f["invdiag"][j] for j in range(d)], beta) for x in row])
                for j in range(q):
                    cols.append([x for row in outer([w0[k] * f["invdiag"][k] for k in range(d)], basis(q, j)) for x in row])
                ch("local_parameter_rank", rank(transpose(cols)) == d - 1 + q)
                if variant:
                    wp, bp = reference["W"][i], reference["beta"][i]
                    diff = matsub(outer(w, beta), outer(wp, bp))
                    dw = vecadd(w, scale(-1, wp))
                    db = vecadd(beta, scale(-1, bp))
                    identity = dot(db, db) + dot(beta, bp) * dot(dw, dw)
                    ch("rank_one_distance_identity", norm2(diff) == identity)
                    lower = dot(db, db) + f["beta0"] ** 2 * dot(dw, dw)
                    ch("positive_beta_separation", identity >= lower)
                    head_diff = matsub(actual, saved_heads[i])
                    ch("C_inverse_separation", norm2(head_diff) * c * c >= lower)
                    # Separate public binary-label corollary: actual response
                    # is exactly one, not the coupled main-family response.
                    binary_y = [[Q(0)] for _ in range(len(Z))]
                    for hidden in range(start, len(Z)):
                        binary_y[hidden] = [Q(1)]
                    binary_head = ridge(Z, binary_y, [0, start + i], lam, d, 1)
                    factor = 1 + sum((f["invdiag"][j] * w[j] ** 2 for j in range(d)), Q(0))
                    beta_binary = 1 / factor
                    expected_binary = [[f["invdiag"][j] * w[j] * beta_binary] for j in range(d)]
                    ch("binary_label_exact_head", binary_head == expected_binary)
                    ch("binary_label_beta_lower", beta_binary >= 2 * lam / c)
                    reference_binary_y = [[Q(0)] for _ in range(len(Z))]
                    for hidden in range(start, len(Z)):
                        reference_binary_y[hidden] = [Q(1)]
                    previous_binary = ridge(reference["Z"], reference_binary_y, [0, start + i], lam, d, 1)
                    ch("binary_label_head_separation", norm2(matsub(binary_head, previous_binary)) >=
                       (2 * lam / (c * c)) ** 2 * dot(dw, dw))
            hard_sets = {frozenset(range(1, b + 1)) | {1 + b + i}: i for i in range(m)}
            for F in subsets(len(Z), b + 1):
                S = selected(Z, F)
                admitted = [v for v in S if v >= start]
                if len(F) <= b:
                    ch("every_low_budget_query_has_zero_cross_moment", not admitted and
                       all(sum((Z[v][j] * Y[v][k] for v in S), Q(0)) == 0 for j in range(d) for k in range(q)))
                ch("only_hard_queries_admit", admitted == ([start + hard_sets[F]] if F in hard_sets else []))
            eta = r * f["beta0"] / (32 * c)
            feature_spacing = 4 * c * eta / f["beta0"]
            beta_spacing = 4 * c * eta
            ch("packing_feature_ratio", (r / 2) / feature_spacing == 4)
            ch("packing_grid_count", int(f["beta0"] // beta_spacing) + 1 > 2)
            ch("packing_output_separation", f["beta0"] * feature_spacing / c == 4 * eta and beta_spacing / c == 4 * eta)
    # This variation genuinely leaves the anchor-orthogonal subspace.
    f = family(2, 1, 1, Q(1), 2)
    ch("full_sphere_not_only_anchor_orthogonal", all(dot(w, f["a"]) != 0 for w in f["W"]))
    # Error scale cannot silently lose its dependence on regularization.
    for lam in [Q(1, 100), Q(1, 2), Q(1), Q(100)]:
        c = 1 + 2 * lam
        beta0 = lam / (2 * c)
        ch("response_upper_bound_algebra", (1 + 1 / (2 * lam)) * 2 * beta0 == Q(1, 2))
        ch("fixed_error_range_positive", r * beta0 / (16 * c) > 0)
    return {"rational_family_instances": cases, "signature_scope": "orthogonal_signatures_only; logarithmic_signature_existence_is_proved_not_tested",
            "floating_point_tolerance_used": False,
            "topology_entropy_or_packing_existence_proved_by_finite_tests": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Preserve the existing report: {out}")
    snapshot = out.parent / (out.stem + "_sources")
    if snapshot.exists():
        raise FileExistsError(f"Preserve the existing source snapshot: {snapshot}")
    out.parent.mkdir(parents=True, exist_ok=True)
    bindings = {}
    for relative in BOUND:
        path = ROOT / relative
        data = path.read_bytes()
        bindings[relative] = {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    checks = Checks()
    report = {"schema": "phase9_theory_extension_algebra_checks_v1", "scope": "bounded_exact_rational_proof_software_checks_not_empirical_evidence",
              "source_bindings": bindings, "source_snapshot": str(snapshot.relative_to(ROOT)) if snapshot.is_relative_to(ROOT) else str(snapshot)}
    try:
        report.update(run_checks(checks))
        for relative, binding in bindings.items():
            checks.check("bound_source_unchanged_during_check", sha256((ROOT / relative).read_bytes()).hexdigest() == binding["sha256"])
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        report["checks_by_category"] = dict(sorted(checks.counts.items()))
        report["checks"] = sum(checks.counts.values())
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "checks": report["checks"], "output": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
