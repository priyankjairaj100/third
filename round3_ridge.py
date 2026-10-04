#!/usr/bin/env python3
"""Ridge decoder checks using only the current canonical statistic coefficient.

The exact checks use fractions, including CG and an independent Gaussian
elimination oracle. The floating checks test residual inequalities numerically;
they are not a machine-roundoff certificate. No retained-record reads occur in
the decoder. Raw records below are held only by the external rebuild oracle.
"""
from fractions import Fraction as F
import argparse
import json
from pathlib import Path
import random
import numpy as np
from algorithm_round2 import IndexedState, rebuild


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def mv(A, x):
    return [dot(row, x) for row in A]


def exact_rank(A):
    a = [[F(x) for x in row] for row in A]
    rows, cols = len(a), len(a[0]) if a else 0
    pivot = 0
    for j in range(cols):
        at = next((i for i in range(pivot, rows) if a[i][j]), None)
        if at is None:
            continue
        a[pivot], a[at] = a[at], a[pivot]
        scale = a[pivot][j]
        a[pivot] = [x/scale for x in a[pivot]]
        for i in range(pivot+1, rows):
            scale = a[i][j]
            a[i] = [x-scale*y for x, y in zip(a[i], a[pivot])]
        pivot += 1
        if pivot == rows:
            break
    return pivot


def exact_solve(A, b):
    n = len(b)
    a = [[F(x) for x in row]+[F(v)] for row, v in zip(A, b)]
    for j in range(n):
        p = next(i for i in range(j, n) if a[i][j])
        a[j], a[p] = a[p], a[j]
        s = a[j][j]
        a[j] = [x/s for x in a[j]]
        for i in range(n):
            if i != j:
                s = a[i][j]
                a[i] = [x-s*y for x, y in zip(a[i], a[j])]
    return [row[-1] for row in a]


def exact_cg(A, b, x0=None):
    """Standard exact-arithmetic CG; no inverse, roots, rank computation, payloads."""
    d = len(b)
    x = [F(0)]*d if x0 is None else [F(v) for v in x0]
    r = [F(v)-w for v, w in zip(b, mv(A, x))]
    p = r[:]
    rr = dot(r, r)
    if not rr:
        return x, 0
    for t in range(1, d+1):
        ap = mv(A, p)
        alpha = rr/dot(p, ap)
        x = [v+alpha*w for v, w in zip(x, p)]
        r = [v-alpha*w for v, w in zip(r, ap)]
        rrnew = dot(r, r)
        if not rrnew:
            assert mv(A, x) == [F(v) for v in b]
            return x, t
        beta = rrnew/rr
        p = [v+beta*w for v, w in zip(r, p)]
        rr = rrnew
    raise AssertionError('CG failed exact finite termination')


def record_stats(z, y):
    return tuple(a*b for a in z for b in z)+tuple(a*y for a in z)+(1,)


def unpack_moments(value, d):
    M = [list(value[i*d:(i+1)*d]) for i in range(d)]
    h = list(value[d*d:d*d+d])
    n = value[-1]
    return M, h, n


def ridge_decode_exact(value, d, lam):
    M, h, n = unpack_moments(value, d)
    if not n:
        assert not any(value)
        return [F(0)]*d, 0
    A = [[F(M[i][j])+(lam*n if i == j else 0)
          for j in range(d)] for i in range(d)]
    return exact_cg(A, h)


def cg_float(A, h, nu, tolerance, x0=None):
    """Illustrative numerical solver; recomputes true residual for stopping.

    The tolerance certifies a real-arithmetic bound if this residual norm is
    rigorously enclosed. This ordinary float demo does not supply that enclosure.
    """
    x = np.zeros_like(h) if x0 is None else x0.copy()
    r = h-A@x
    p = r.copy()
    rr = r@r
    if np.linalg.norm(r) <= nu*tolerance:
        return x, 0, False
    for t in range(1, 3*len(h)+1):
        ap = A@p
        denom = p@ap
        if denom <= 0 or not np.isfinite(denom):
            break
        alpha = rr/denom
        x = x+alpha*p
        r = r-alpha*ap
        actual = h-A@x
        if np.linalg.norm(actual) <= nu*tolerance:
            return x, t, False
        rrnew = r@r
        if not rrnew:
            break
        p = r+(rrnew/rr)*p
        rr = rrnew
    # A costed O(d^3) fallback; certify its residual as well at the call site.
    return np.linalg.solve(A, h), t, True


def run(seed=20261003):
    rng = random.Random(seed)
    out = {'seed': seed, 'exact_graph_cases': 0, 'exact_releases': 0,
           'exact_warm_releases': 0, 'exact_positive_count_releases': 0,
           'max_zero_iterations': 0, 'max_warm_iterations': 0,
           'floating_cases': 0, 'floating_fallbacks': 0,
           'max_floating_reference_error': 0.0,
           'max_float_certificate_ratio': 0.0}
    for case in range(120):
        N, d = rng.randint(0, 10), rng.randint(1, 6)
        k = rng.randint(0, N)
        blockers = [tuple(u for u in range(v) if rng.random() < .30)
                    for v in range(N)]
        z = [[rng.randint(-2, 2) for _ in range(d)] for _ in range(N)]
        y = [rng.randint(-2, 2) for _ in range(N)]
        records = [record_stats(a, b) for a, b in zip(z, y)]
        dim = d*d+d+1
        coef = rebuild(blockers, records, set(range(N)), k)
        state = IndexedState(N, k, dim, range(N), coef)
        meter = state.meter()
        lam = F(rng.randint(1, 5), rng.randint(1, 7))
        previous = [F(rng.randint(-2, 2)) for _ in range(d)]
        order = list(range(N))
        rng.shuffle(order)
        for j in range(k+1):
            value = state.constant()
            answer, steps = ridge_decode_exact(value, d, lam)
            selected = [v for v in state.alive
                        if not set(blockers[v]) & state.alive]
            oracle = tuple(sum(records[v][a] for v in selected)
                           for a in range(dim))
            assert value == oracle
            M, h, n = unpack_moments(value, d)
            if n:
                A = [[F(M[a][b])+(lam*n if a == b else 0)
                      for b in range(d)] for a in range(d)]
                reference = exact_solve(A, h)
                rank = exact_rank(M)
                assert answer == reference
                assert steps <= rank <= min(d, n)
                warm, warm_steps = exact_cg(A, h, previous)
                assert warm == reference
                assert warm_steps <= min(d, rank+1)
                out['exact_warm_releases'] += 1
                out['exact_positive_count_releases'] += 1
                out['max_warm_iterations'] = max(out['max_warm_iterations'], warm_steps)
            else:
                assert answer == [F(0)]*d and steps == 0
            previous = answer
            out['exact_releases'] += 1
            out['max_zero_iterations'] = max(out['max_zero_iterations'], steps)
            if j < k:
                state.delete([order[j]], meter)
        out['exact_graph_cases'] += 1

    ng = np.random.default_rng(seed)
    for _ in range(160):
        n, d = int(ng.integers(1, 91)), int(ng.integers(1, 49))
        Z = ng.normal(size=(n, d))
        Z = Z/np.linalg.norm(Z, axis=1, keepdims=True)
        y = ng.uniform(-1, 1, n)
        lam = 10.0**ng.uniform(-3, 1)
        M, h = Z.T@Z, Z.T@y
        A, nu = M+(lam*n)*np.eye(d), lam*n
        eigen = np.linalg.eigvalsh(A)
        kappa = eigen[-1]/eigen[0]
        bound = 1+np.trace(M)/(lam*n)
        assert kappa <= bound*(1+2e-11)
        reference = np.linalg.solve(A, h)
        # Both a zero start and arbitrary start exercise the residual guarantee.
        for x0 in (None, ng.normal(size=d)):
            answer, iterations, fallback = cg_float(A, h, nu, 1e-9, x0)
            r = A@answer-h
            error = np.linalg.norm(answer-reference)
            cert = np.linalg.norm(r)/nu
            assert error <= cert+2e-11
            assert cert <= 1e-9+2e-11
            gap = ((answer-reference)@A@(answer-reference))/(2*n)
            assert gap <= np.linalg.norm(r)**2/(2*lam*n*n)+1e-20
            out['floating_cases'] += 1
            out['floating_fallbacks'] += int(fallback)
            out['max_floating_reference_error'] = max(out['max_floating_reference_error'], float(error))
            if cert > 1e-12:
                out['max_float_certificate_ratio'] = max(out['max_float_certificate_ratio'], float(error/cert))

    d = 5
    lam = 1.0
    M0 = np.diag([1.0]+[0.0]*(d-1))
    deltaM = np.diag([0.0, 1.0]+[0.0]*(d-2))
    A0 = M0+lam*np.eye(d)
    A1 = M0+deltaM+2*lam*np.eye(d)
    deltaA = A1-A0
    h1 = np.array([1.0, 1.0]+[0.0]*(d-2))
    actual = np.linalg.solve(A1, h1)
    incorrect = np.linalg.solve(A0+deltaM, h1)
    assert np.linalg.matrix_rank(deltaA) == d
    assert np.max(np.abs(actual-incorrect)) > .16
    out['normalization_counterexample'] = {
        'dimension': d, 'gram_update_rank': int(np.linalg.matrix_rank(deltaM)),
        'normal_equation_update_rank': int(np.linalg.matrix_rank(deltaA)),
        'max_parameter_error_ignoring_count_shift': float(np.max(np.abs(actual-incorrect)))}
    out['status'] = 'all checks passed'
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='round3_ridge_check.json')
    args = parser.parse_args()
    result = run()
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
