"""Bounded algebra/software checks for the analytic FP32 scorer-transfer proof.

No empirical data, semantic embeddings, model training, or hardware certification.
The all-sign/all-dimension conclusion is proved in SCORER_TRANSFER.md.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from phase3.panels import graph_normalize, reference_cosines
from phase3.reference_graph import build_reference_graph, id_edges, stable_priority

PINNED = {
    'phase3/panels.py': 'bd2409958ac9a9e27faf463b44b2f839260a3195c7cde32fb3bfb76aeb14a04b',
    'phase3/reference_graph.py': 'efc41498648f6f7f7969768fbcd050619d7ed5d6e82eafadb1b9a3a2158e9d1c',
    'ccu/core.py': 'd89f6affd566103c65aa92f10e02ac86e394f7a1f88ff2e6b098720b8cef7df3',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_bindings():
    names = list(PINNED) + ['phase10/SCORER_TRANSFER.md', 'phase10/check_scorer_transfer.py']
    return {name: sha(ROOT / name) for name in names}


def fixture(case):
    """Eleven reserved algebra IDs, with roles assigned after actual SHA sorting."""
    k = D = 16
    m, b, seed = 4, 2, 41
    ids = tuple('scorer-algebra-only-' + str(i) for i in reversed(range(1+b+2*m)))
    priority = stable_priority(ids, seed)
    anchor = priority[0]
    common = priority[1:1+b]
    private = priority[1+b:1+b+m]
    candidate = priority[1+b+m:]
    role = {anchor: ('a', 0)}
    role.update({name: ('s', i) for i, name in enumerate(common)})
    role.update({name: ('p', i) for i, name in enumerate(private)})
    role.update({name: ('w', i) for i, name in enumerate(candidate)})
    signatures = np.asarray([[(-1 if (i & j).bit_count() % 2 else 1)/4
                              for j in range(k)] for i in range(m)], dtype=np.float32)
    x = np.zeros((len(ids), 2+k+D), np.float32)
    for row, name in enumerate(ids):
        kind, i = role[name]
        if kind == 'a':
            x[row, 0] = 1
        elif kind == 's':
            x[row, :2] = [.5, .875]
        else:
            x[row, 0 if kind == 'p' else 1] = .5
            x[row, 2:2+k] = np.float32(.875) * signatures[i]
            if kind == 'w':
                signs = [1 if case == 0 else -1 if case == 1 else
                         (-1 if ((j ^ (i*7+case*3)).bit_count()+case//4) % 2 else 1)
                         for j in range(D)]
                x[row, 2+k:] = np.asarray(signs, np.float32)/512
    edges = {(s, anchor) for s in common} | {(p, anchor) for p in private}
    edges |= {(common[j], common[i]) for j in range(b) for i in range(j)}
    edges |= {(w, s) for w in candidate for s in common}
    edges |= {(candidate[i], private[i]) for i in range(m)}
    return x, ids, priority, role, edges, common, private, candidate, seed, signatures


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = source_bindings()
    checks, cases = [], []
    report = {'schema': 'ccu-fp32-scorer-transfer-checks-1', 'status': 'failed',
              'evidence_role': 'bounded_algebra_software_only', 'source_sha256': sources,
              'empirical_benchmark': False, 'primary_study_started': False,
              'human_responses_created': 0, 'model_training_performed': False,
              'universal_conclusion_from_enumeration': False,
              'runtime_conformance_proved': False,
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'machine': platform.machine(), 'platform': platform.platform()}}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    try:
        check('frozen_scorer_sources_exact', all(sources[k] == v for k, v in PINNED.items()))
        check('binary32_binary64_format_metadata', np.finfo(np.float32).nmant == 23 and
              np.finfo(np.float64).nmant == 52 and np.dtype('float32').itemsize == 4 and
              np.dtype('float64').itemsize == 8)
        u = F(1, 2**53)
        rho = 2*u/(1-u)
        n = 2**20+1
        gamma = n*u/(1-n*u)
        error = 2*rho+rho*rho+gamma*(1+rho)**2
        tau = F.from_float(.4)
        check('threshold_exact_binary64', tau == F(3602879701896397, 2**53))
        check('threshold_offset_exact', tau-F(2, 5) == F(1, 5*2**53))
        check('base_norm_square', F(1, 4)+F(49, 64) == F(65, 64))
        check('candidate_norm_square_and_exact_root', F(1, 4)+F(49, 64)+F(1, 16384) == F(129, 128)**2)
        check('anchor_edge_exceeds_common_candidate', F(4, 1) > F(448, 129))
        check('matching_edge_exceeds_common_candidate', 784 > 448)
        check('common_candidate_edge_gt_043_squared', F(448**2, 129**2*65) > F(43, 100)**2)
        check('common_private_nonedge_lt_candidate_bound', F(16, 65) < F(18563, 49923))
        check('private_pair_bound_exact', (F(1, 4)+F(49, 384))/F(65, 64) == F(29, 78))
        check('private_pair_bound_lt_candidate_bound', F(29, 78) < F(18563, 49923))
        check('nonmatching_pair_bound_squared', F(392**2, 387**2*65) < F(18563, 49923)**2)
        check('candidate_pair_bound_exact', (F(1, 4)+F(49, 384)+F(1, 16384))/F(16641, 16384) == F(18563, 49923))
        check('all_nonedges_lt_0372', F(18563, 49923) < F(93, 250))
        check('uniform_error_lt_2powminus32', error < F(1, 2**32))
        check('displayed_analytic_error_bound', rho < 3*u and gamma < F(9, 7*2**33) and
              (1+rho)**2 < F(9, 8) and 2*rho+rho*rho < 7*u and
              F(81, 56*2**33)+7*u < F(1, 2**32))
        check('edge_stored_threshold_margin_gt_029', F(43, 100)-F(1, 2**32) > tau+F(29, 1000))
        check('nonedge_stored_threshold_margin_gt_027', F(93, 250)+F(1, 2**32) < tau-F(27, 1000))
        exponent_pairs = []
        for r in range(11):
            for s in range(11):
                k, D = 4**r, 4**s
                if 2+k+D > 2**20:
                    continue
                coordinates = [F(1), F(1, 2), F(7, 8), F(7, 2**(r+3)), F(1, 2**(s+7))]
                assert all(F.from_float(float(np.float32(float(v)))) == v for v in coordinates)
                assert all((v*v*2**34).denominator == 1 for v in coordinates)
                assert F(16641, 16384)*2**34 < 2**35 < 2**53
                assert min(coordinates) >= F(1, 2**17)
                exponent_pairs.append([r, s])
        check('every_admissible_exponent_pair_exact_FP32_and_norm_grid', len(exponent_pairs) == 100)
        for case in range(8):
            x, ids, priority, role, edges, common, private, candidate, seed, signatures = fixture(case)
            check(f'case_{case}_public_orthogonal_codebook', np.array_equal(signatures.astype(np.float64) @ signatures.T, np.eye(4)))
            square = np.zeros(len(x), np.float64)
            for coordinate in range(x.shape[1]):
                square += x[:, coordinate].astype(np.float64)**2
            expected = [1. if role[name][0] == 'a' else float(F(16641, 16384))
                        if role[name][0] == 'w' else float(F(65, 64)) for name in ids]
            check(f'case_{case}_ordered_norm_squares_exact', np.array_equal(square, expected))
            scores = reference_cosines(graph_normalize(x), graph_normalize(x))
            lookup = {name: i for i, name in enumerate(ids)}
            pairs = [(priority[j], priority[i]) for j in range(len(ids)) for i in range(j)]
            check(f'case_{case}_actual_scores_have_proved_margins', all(
                scores[lookup[a], lookup[b]] > .4+.029 if (a, b) in edges else
                scores[lookup[a], lookup[b]] < .4-.027 for a, b in pairs))
            for tile in (1, 3, 256):
                graph = build_reference_graph(x, ids, .4, seed=seed, block_size=tile)
                check(f'case_{case}_tile_{tile}_actual_blockers', id_edges(graph) == edges)
            removed = set(common) | {private[0]}
            retained = [i for i, name in enumerate(ids) if name not in removed]
            reduced = build_reference_graph(x[retained], [ids[i] for i in retained], .4, seed=seed, block_size=3)
            check(f'case_{case}_retained_graph_restriction', id_edges(reduced) ==
                  {(a, b) for a, b in edges if a not in removed and b not in removed})
            selected = {reduced.record_ids[int(i)] for i in reduced.selected_indices()}
            check(f'case_{case}_public_probe_selects_only_anchor_and_candidate', selected == {priority[0], candidate[0]})
            cases.append({'case': case, 'records': len(ids), 'dimension': x.shape[1],
                          'edges': len(edges), 'tile_sizes': [1, 3, 256], 'private_payloads_exhaustive': False})
        check('sources_unchanged', sources == source_bindings())
        report.update(status='passed', admissible_exponent_pairs=exponent_pairs,
                      error_bound_exact=str(error), error_bound_decimal=float(error),
                      uniform_error_upper_bound='1/4294967296', threshold_exact=str(tau),
                      clip_in_frozen_path=False, algebra_cases=cases)
    except Exception as error:
        report.update(exception_type=type(error).__name__, reason=str(error))
    report.update(checks=checks, passed_checks=len(checks))
    (out/'checks.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({'status': result['status'], 'passed_checks': result['passed_checks']}))
    raise SystemExit(result['status'] != 'passed')
