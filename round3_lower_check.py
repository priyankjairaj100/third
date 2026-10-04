"""Finite checks of fixed-test prediction, ridge excess loss, and coded repair.

These checks validate the algebra and a small symmetrized coding example;
the entropy, asymptotic covering, and dimension statements are proofs.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np


def bit_vectors(m):
    return np.array(list(itertools.product([0, 1], repeat=m)), dtype=int)


def covering_code(m, r):
    words = bit_vectors(m)
    distances = np.count_nonzero(words[:, None, :] != words[None, :, :], axis=2)
    uncovered = np.ones(len(words), dtype=bool)
    chosen = []
    while np.any(uncovered):
        counts = np.sum((distances <= r) & uncovered[None, :], axis=1)
        index = int(np.argmax(counts))
        chosen.append(index)
        uncovered &= distances[index] > r
    code = words[chosen]
    nearest = np.argmin(
        np.count_nonzero(words[:, None, :] != code[None, :, :], axis=2), axis=1
    )
    return words, code, code[nearest]


def main():
    rng = np.random.default_rng(20261003)
    m, b, lam = 7, 2, 0.7
    c = 1 + 2 * lam
    # A regular simplex hits the allowed negative-correlation boundary.
    U = np.eye(m) - np.ones((m, m)) / m
    U /= np.linalg.norm(U, axis=1)[:, None]
    assert np.max(np.abs((U @ U.T)[~np.eye(m, dtype=bool)])) <= 1 / 6 + 1e-14
    d = U.shape[1]
    e0, e1 = np.eye(d + 2)[:2]
    W = .5 * e1 + math.sqrt(3) / 2 * np.c_[np.zeros((m, 2)), U]
    candidate_energies = np.mean((W @ W.T / c) ** 2, axis=0)
    lower_energy = (1 + (m - 1) / 64) / (m * c * c)
    assert np.min(candidate_energies) >= lower_energy - 1e-14
    probabilities = np.array([.25, .25] + [1 / (2 * d)] * d)
    expected_energy = (1 / 16 + 3 / (8 * d)) / (c * c)
    energies = (W * W) @ probabilities / (c * c)
    max_energy_error = float(np.max(np.abs(energies - expected_energy)))
    assert max_energy_error < 1e-14

    min_excess_gap = float('inf')
    max_scalar_excess_error = 0.0
    quadratic_cases = 0
    for i in range(m):
        H = lam * np.eye(d + 2) + .5 * (np.outer(e0, e0) + np.outer(W[i], W[i]))
        for yi in [0, 1]:
            optimum = yi * W[i] / c
            for _ in range(100):
                theta = rng.normal(size=d + 2)
                err = theta - optimum
                excess = .5 * err @ H @ err
                y_hat = c * W[i] @ theta
                scalar_bound = (y_hat - yi) ** 2 / (4 * c)
                min_excess_gap = min(min_excess_gap, excess - scalar_bound)
                assert excess >= scalar_bound - 1e-12
                scalar_hat = rng.uniform(-1, 2)
                scalar_theta = scalar_hat * W[i] / c
                scalar_err = scalar_theta - optimum
                exact_excess = .5 * scalar_err @ H @ scalar_err
                max_scalar_excess_error = max(
                    max_scalar_excess_error,
                    abs(exact_excess - (scalar_hat - yi) ** 2 / (4 * c)),
                )
                quadratic_cases += 1

    # Exhaust every fixed label vector, public mask, public permutation, and
    # query coordinate for a nontrivial radius-one binary covering code.
    cm, radius = 4, 1
    words, code, reconstruction = covering_code(cm, radius)
    q = radius / cm
    D = q * (1 - q)
    squared_by_y_i = np.zeros((len(words), cm))
    bit_error_by_y_i = np.zeros_like(squared_by_y_i)
    powers = 2 ** np.arange(cm - 1, -1, -1)
    perm_count = math.factorial(cm)
    cases = 0
    for iy, y in enumerate(words):
        for mask in words:
            for order in itertools.permutations(range(cm)):
                order = np.array(order)
                transformed = (y ^ mask)[order]
                word_index = int(transformed @ powers)
                binary = reconstruction[word_index]
                transformed_hat = q + (1 - 2 * q) * binary
                raw_hat = np.empty(cm)
                raw_binary = np.empty(cm, dtype=int)
                raw_hat[order] = transformed_hat
                raw_binary[order] = binary
                estimate = np.where(mask, 1 - raw_hat, raw_hat)
                binary_estimate = raw_binary ^ mask
                squared_by_y_i[iy] += (estimate - y) ** 2
                bit_error_by_y_i[iy] += binary_estimate != y
                cases += cm
    squared_by_y_i /= len(words) * perm_count
    bit_error_by_y_i /= len(words) * perm_count
    assert np.max(squared_by_y_i) <= D + 1e-14
    assert np.max(bit_error_by_y_i) <= q + 1e-14
    assert np.ptp(squared_by_y_i) < 1e-14
    assert np.ptp(bit_error_by_y_i) < 1e-14
    result = {
        'checks_passed': True,
        'fixed_test_candidate_count': m,
        'signature_dimension': d,
        'candidate_min_template_energy': float(np.min(candidate_energies)),
        'candidate_energy_lower_bound': lower_energy,
        'full_rank_test_max_energy_formula_error': max_energy_error,
        'quadratic_objective_cases_checked': quadratic_cases,
        'min_excess_minus_projection_bound': float(min_excess_gap),
        'scalar_excess_identity_max_error': float(max_scalar_excess_error),
        'cover_code_word_length': cm,
        'cover_code_radius': radius,
        'cover_code_cardinality': len(code),
        'symmetrization_coordinate_cases_checked': cases,
        'worst_fixed_y_query_expected_squared_error': float(np.max(squared_by_y_i)),
        'guaranteed_expected_squared_error': D,
        'worst_fixed_y_query_bit_failure_probability': float(np.max(bit_error_by_y_i)),
        'guaranteed_bit_failure_probability': q,
        'limitations': 'Finite algebra/coding checks; no NLP experiment or efficient optimal codec claim.',
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
