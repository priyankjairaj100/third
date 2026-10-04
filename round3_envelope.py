"""Exhaustive and numerical checks for the graph-envelope certificate.

Run: python3 round3_envelope.py
Dependencies: Python 3, NumPy. Writes round3_envelope.json next to this file.
This validates finite instances; proofs are in round3_envelope.md.
"""

from collections import defaultdict
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np


def powerset(xs):
    xs = tuple(xs)
    for r in range(len(xs) + 1):
        for ys in combinations(xs, r):
            yield frozenset(ys)


def select(blockers, deleted):
    return frozenset(v for v, bs in enumerate(blockers)
                     if v not in deleted and bs <= deleted)


def count_coefficients(blockers):
    coefficients = defaultdict(int)
    for v, bs in enumerate(blockers):
        coefficients[bs] += 1
        coefficients[bs | {v}] -= 1
    return {key: value for key, value in coefficients.items() if value}


def query(coefficients, deleted):
    return sum(value for key, value in coefficients.items() if key <= deleted)


def tv(left, right):
    assert left and right
    return 1.0 - len(left & right) / max(len(left), len(right))


def certificate(inner, proxy, outer):
    assert inner <= proxy <= outer
    assert inner
    a, p, c = len(inner), len(proxy), len(outer - proxy)
    return 1.0 - a / max(p, a + c)


def main():
    counts = defaultdict(int)
    max_tv_slack_violation = 0.0
    for n in range(1, 5):
        pairs = list(combinations(range(n), 2))
        # 0: forbidden, 1: uncertain, 2: mandatory.
        for states in product(range(3), repeat=len(pairs)):
            low = [set() for _ in range(n)]
            high = [set() for _ in range(n)]
            optional = []
            for (u, v), state in zip(pairs, states):
                if state == 2:
                    low[v].add(u)
                if state:
                    high[v].add(u)
                if state == 1:
                    optional.append((u, v))
            low = [frozenset(bs) for bs in low]
            high = [frozenset(bs) for bs in high]
            low_coeff, high_coeff = count_coefficients(low), count_coefficients(high)
            counts['ordered_graph_envelopes'] += 1
            for deleted in powerset(range(n)):
                inner, outer = select(high, deleted), select(low, deleted)
                assert query(high_coeff, deleted) == len(inner)
                assert query(low_coeff, deleted) == len(outer)
                assert inner <= outer
                counts['polynomial_count_queries'] += 2
                if len(deleted) == n:
                    assert not inner and not outer
                    continue
                assert inner  # The earliest retained vertex has no surviving blocker.
                possible_sets = set()
                for chosen in powerset(range(len(optional))):
                    middle = [set(bs) for bs in low]
                    for j in chosen:
                        u, v = optional[j]
                        middle[v].add(u)
                    chosen_set = select([frozenset(bs) for bs in middle], deleted)
                    assert inner <= chosen_set <= outer
                    possible_sets.add(chosen_set)
                    counts['graph_selection_envelope_checks'] += 1
                interval_sets = {inner | extra for extra in powerset(outer - inner)}
                assert possible_sets == interval_sets
                counts['interval_realizability_checks'] += 1
                for proxy in possible_sets:
                    bound = certificate(inner, proxy, outer)
                    exact = max(tv(proxy, actual) for actual in possible_sets)
                    max_tv_slack_violation = max(max_tv_slack_violation, exact - bound)
                    assert abs(exact - bound) < 1e-14
                    counts['sharp_tv_certificates'] += 1

    # Verify the maximum squared coefficient norm used by the task-gradient bound.
    for a in range(1, 9):
        for b in range(9):
            for c in range(9):
                p = a + b
                h = ((1 / a - 1 / p) if 2 * a <= p
                     else 1 / p + (p - 2 * a) / (p * (a + c)))
                actual = max(1 / p + (p - 2 * q) / (p * (q + r))
                             for q in range(a, p + 1) for r in range(c + 1))
                assert abs(h - actual) < 1e-14
                counts['sharp_coefficient_norm_certificates'] += 1

    rng = np.random.default_rng(20261003)
    max_gradient_slack_violation = 0.0
    for _ in range(300):
        a = int(rng.integers(1, 5))
        b = int(rng.integers(0, 4))
        c = int(rng.integers(0, 4))
        p, o = a + b, a + b + c
        gradients = rng.normal(size=(o, 5))
        centered = gradients - gradients.mean(axis=0)
        covariance_sum = centered.T @ centered
        spectral = max(0.0, np.linalg.eigvalsh(covariance_sum)[-1])
        h = ((1 / a - 1 / p) if 2 * a <= p
             else 1 / p + (p - 2 * a) / (p * (a + c)))
        bound = np.sqrt(max(0.0, spectral * h))
        proxy_gradient = gradients[:p].mean(axis=0)
        for extra in powerset(range(a, o)):
            chosen = list(range(a)) + sorted(extra)
            error = np.linalg.norm(gradients[chosen].mean(axis=0) - proxy_gradient)
            max_gradient_slack_violation = max(max_gradient_slack_violation, error - bound)
            assert error <= bound + 1e-12
            counts['gradient_second_moment_checks'] += 1

    # Unit-vector and threshold drift implication, including equality boundaries.
    for _ in range(2000):
        vec = rng.normal(size=(4, 7))
        vec /= np.linalg.norm(vec, axis=1)[:, None]
        z_u, z_v, new_u, new_v = vec
        tau, new_tau = rng.uniform(-1, 1, size=2)
        delta = (np.linalg.norm(new_u - z_u) + np.linalg.norm(new_v - z_v)
                 + abs(new_tau - tau))
        old_margin = np.dot(z_u, z_v) - tau
        new_margin = np.dot(new_u, new_v) - new_tau
        assert old_margin - delta <= new_margin + 1e-12
        assert new_margin <= old_margin + delta + 1e-12
        if old_margin > delta:
            assert new_margin > 0
        if new_margin > 0:
            assert old_margin > -delta
        counts['score_drift_checks'] += 1

    report = dict(counts)
    report['max_tv_bound_violation'] = max_tv_slack_violation
    report['max_gradient_bound_violation'] = max_gradient_slack_violation
    report['status'] = 'all checks passed'
    output = Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
