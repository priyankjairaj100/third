"""Small deterministic checks of the second-pass storage lower-bound construction."""
import itertools
import json
import math
from pathlib import Path

import numpy as np


def frontier(m, b):
    # Signature correlations hit the allowed upper boundary exactly.
    u = np.zeros((m, m + 1))
    u[:, 0] = math.sqrt(1 / 6)
    for i in range(m):
        u[i, i + 1] = math.sqrt(5 / 6)
    d = m + 3
    e0, e1 = np.eye(d)[:2]
    ids = ['a'] + [f's{j}' for j in range(b)] + [f'p{i}' for i in range(m)] + [f'w{i}' for i in range(m)]
    vecs = [e0]
    vecs += [.5 * e0 + math.sqrt(3) / 2 * e1 for _ in range(b)]
    vecs += [.5 * e0 + math.sqrt(3) / 2 * np.r_[0, 0, ui] for ui in u]
    vecs += [.5 * e1 + math.sqrt(3) / 2 * np.r_[0, 0, ui] for ui in u]
    E = np.array(vecs)
    B = {ids[j]: {ids[i] for i in range(j) if E[i] @ E[j] > .4} for j in range(len(ids))}
    return ids, E, B


def selected(ids, B, F):
    return [v for v in ids if v not in F and B[v] <= F]


def main():
    m, b, lam = 5, 2, .7
    ids, E, B = frontier(m, b)
    ix = {v: i for i, v in enumerate(ids)}
    common = {f's{j}' for j in range(b)}
    assert np.max(np.abs(np.linalg.norm(E, axis=1) - 1)) < 1e-12
    assert selected(ids, B, set()) == ['a']
    for i in range(m):
        assert B[f'w{i}'] == common | {f'p{i}'}
    low_requests = [set(F) for r in range(b + 1) for F in itertools.combinations(ids, r)]
    for F in low_requests:
        assert not any(v.startswith('w') for v in selected(ids, B, F))
    max_full_error = 0.
    max_scalar_error = 0.
    max_projection_error = 0.
    labels_checked = 0
    alpha = 1 / math.sqrt(2)
    Ep = np.c_[np.full(len(ids), alpha), math.sqrt(1 - alpha**2) * E]
    Bp = {ids[j]: {ids[i] for i in range(j) if Ep[i] @ Ep[j] > .7} for j in range(len(ids))}
    assert Bp == B
    for Y in itertools.product([0, 1], repeat=m):
        labels = {v: 0 for v in ids}
        labels.update({f'w{i}': yi for i, yi in enumerate(Y)})
        for i, yi in enumerate(Y):
            F = common | {f'p{i}'}
            assert all(labels[v] == 0 for v in F)
            T = selected(ids, B, F)
            assert T == ['a', f'w{i}']
            X = E[[ix[v] for v in T]]
            y = np.array([labels[v] for v in T])
            theta = np.linalg.solve(X.T @ X + lam * len(T) * np.eye(E.shape[1]), X.T @ y)
            expected = yi * E[ix[f'w{i}']] / (1 + 2 * lam)
            max_full_error = max(max_full_error, float(np.linalg.norm(theta - expected)))
            scalar = float(sum(y) / (len(T) * (1 + lam)))
            max_scalar_error = max(max_scalar_error, abs(scalar - yi / (2 * (1 + lam))))
            projected = float(alpha * sum(y) / (len(T) * (alpha**2 + lam)))
            max_projection_error = max(max_projection_error, abs(projected - alpha * yi / (2 * (alpha**2 + lam))))
        labels_checked += 1
    result = {
        'm': m, 'b': b, 'N': len(ids), 'embedding_dimension': E.shape[1],
        'signature_pair_inner_product': 1/6, 'cosine_threshold': .4,
        'low_budget_requests_checked': len(low_requests),
        'binary_label_vectors_checked': labels_checked,
        'indexed_high_budget_targets_checked': labels_checked*m,
        'same_full_embedding_ridge_max_l2_error': max_full_error,
        'scalar_ridge_max_abs_error': max_scalar_error,
        'common_coordinate_scalar_max_abs_error': max_projection_error,
        'full_embedding_decoding_accuracy_threshold': 1/(2*(1+2*lam)),
        'constant_feature_decoding_accuracy_threshold': 1/(4*(1+lam)),
        'checks_passed': True,
        'limitations': 'Deterministic finite checks validate geometry/optimizer algebra; entropy and logarithmic dimension are proofs, not empirical claims.'
    }
    out = Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
