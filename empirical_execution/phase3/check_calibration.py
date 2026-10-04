#!/usr/bin/env python3
"""Software checks and natural-data plumbing only; produces no human judgments."""
from __future__ import annotations
import itertools
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccu.data import read_natural_jsonl, lexical_engineering_features
from phase3 import calibration as c


def software_checks():
    checks = []
    # Enumerate the exact random choices of Algorithm R on integer SOFTWARE IDs.
    # These objects are not documents, labels, or synthetic empirical datasets.
    for N in range(1, 8):
        for k in range(1, N + 1):
            distribution = Counter()
            ranges = [range(t) for t in range(k + 1, N + 1)]
            for path in itertools.product(*ranges):
                choices = iter(path)
                class ChoiceStream:
                    def randrange(self, stop):
                        value = next(choices)
                        assert 0 <= value < stop
                        return value
                sample = []
                rng = ChoiceStream()
                for index in range(N):
                    c._reservoir_insert(sample, index, index + 1, k, rng)
                assert len(sample) == k and len(set(sample)) == k
                distribution[tuple(sorted(sample))] += 1
            assert len(distribution) == math.comb(N, k)
            assert len(set(distribution.values())) == 1
            checks.append(f"exact_uniform_reservoir_N{N}_k{k}")
    for edge_index, edge in enumerate(c.BIN_EDGES):
        assert c._bin(edge) == min(edge_index, 19)
        if edge_index:
            assert c._bin(float(np.nextafter(edge, -np.inf))) == edge_index - 1
    checks.append("all_bin_endpoints_and_immediate_predecessors")
    assert c.finite_population_lower_bound(0, 0, 0)["defined"] is False
    assert c.finite_population_lower_bound(100, 0, 0)["defined"] is False
    for N in range(1, 15):
        for x in range(N + 1):
            assert c.finite_population_lower_bound(N, N, x)["lower_success_count"] == x
    checks.append("empty_precision_undefined_and_census_exact")
    for bad in [(True, 1, 1), (1, True, 0), (1, 1, True), (3, 4, 2), (3, 2, -1)]:
        try:
            c.finite_population_lower_bound(*bad)
        except ValueError:
            continue
        raise AssertionError("Malformed counts accepted")
    checks.append("invalid_exact_count_inputs_rejected")
    return checks


def natural_plumbing_checks():
    rows = read_natural_jsonl(ROOT / "data/civil_comments_engineering_preview.jsonl", require_labels=True)
    vectors, metadata = lexical_engineering_features(rows, 128)
    vectors = vectors.astype(np.float32)
    provenance = {"dataset_id": "civil_comments_100_existing_preview",
                  "encoder_id": "lexical_engineering_features",
                  "encoder_revision": "existing_frozen_code_not_E5",
                  "population_scope": "reused_natural_Civil100_development_fixture",
                  "evidence_role": "engineering_nonconfirmatory"}
    manifest = c.prepare_selection(vectors, rows, provenance, block_size=32)
    replay = c.prepare_selection(vectors, rows, provenance, block_size=32)
    assert manifest == replay
    assert manifest["pair_population"] == len(rows) * (len(rows) - 1) // 2
    counts = [0] * 20
    all_pairs = set()
    for i, j, score in c.iter_pair_scores(vectors, 32):
        assert i < j and (i, j) not in all_pairs and math.isfinite(score)
        all_pairs.add((i, j))
        counts[c._bin(score)] += 1
    assert len(all_pairs) == manifest["pair_population"]
    assert counts == manifest["bin_populations"]
    assert {(i, j): score.hex() for i, j, score in c.iter_pair_scores(vectors, 32)} == {
        (i, j): score.hex() for i, j, score in c.iter_pair_scores(vectors, 17)}
    assert manifest["bin_sample_counts"] == [min(30, k) for k in counts]
    assert len({p["natural_pair_id"] for p in manifest["pairs"]}) == len(manifest["pairs"])
    assert len({p["pair_id"] for p in manifest["pairs"]}) == len(manifest["pairs"])
    for pair in manifest["pairs"]:
        f = pair["inclusion_probability_exact"]
        expected = Fraction(min(30, counts[pair["bin"]]), counts[pair["bin"]])
        assert Fraction(f["numerator"], f["denominator"]) == expected
        assert float.fromhex(pair["score_hex"]) == pair["score"]
    # No human responses exist. A blank intake is required to fail, not auto-fill.
    try:
        c.select_threshold(manifest, {"manifest_sha256": manifest["sha256"], "responses": []})
    except ValueError:
        missing_response_rejected = True
    else:
        raise AssertionError("Missing human responses accepted")
    # Test validation sampling ONLY with a marked internal software contract fixture.
    # It has no invented human observations and is never saved as a selection output.
    software_lock = c._seal({"schema": "ccu-calibration-selection-lock-1",
        "protocol_id": c.PROTOCOL, "frame": manifest["frame"], "threshold": .5,
        "selection_status": "failed_diagnostic", "software_test_only": True,
        "selection_natural_pair_ids": [p["natural_pair_id"] for p in manifest["pairs"]]})
    validation = c.prepare_validation(vectors, rows, software_lock, block_size=32)
    above = {(i, j) for i, j, s in c.iter_pair_scores(vectors, 32) if s > .5}
    assert validation["pair_population"] == len(above)
    assert validation["sample_count"] == min(200, len(above))
    assert len({p["natural_pair_id"] for p in validation["pairs"]}) == validation["sample_count"]
    selection_assignments = {a for p in manifest["pairs"] for a in p["assignment_ids"]}
    assert not selection_assignments & {a for p in validation["pairs"] for a in p["assignment_ids"]}
    try:
        c.quality_gate(validation, {"manifest_sha256": validation["sha256"], "responses": []}, software_lock)
    except ValueError:
        pass
    else:
        raise AssertionError("Uncollected validation responses accepted")
    pack_directory = ROOT / "phase3" / "results" / "calibration_engineering_pack"
    if not pack_directory.exists():
        c.write_blinded_pack(manifest, rows, pack_directory)
    else:
        old = json.loads((pack_directory / "private_sampling_manifest.json").read_text())
        if old != manifest:
            raise RuntimeError("Existing pack refers to an earlier code hash; choose a new version path")
    return {"status": "passed_software_and_natural_plumbing_only",
            "natural_records": len(rows), "natural_pairs_scored": len(all_pairs),
            "selection_sample_pairs": len(manifest["pairs"]),
            "bin_populations": counts, "bin_sample_counts": manifest["bin_sample_counts"],
            "selection_replay_exact": True, "all_pairs_once": True,
            "shared_reference_tile_size_invariant": True,
            "exact_inclusion_probabilities": True, "fresh_validation_assignment_ids": True,
            "validation_plumbing_population_at_software_test_tau_point5": len(above),
            "missing_human_responses_rejected": missing_response_rejected,
            "human_judgments_created": 0, "semantic_encoder_used": False,
            "semantic_threshold_selected": False, "semantic_quality_claim": False,
            "boundary_score_audit": manifest["boundary_score_audit"],
            "manifest_code_sha256": manifest["frame"]["scorer"]["code_sha256"]}


if __name__ == "__main__":
    result = natural_plumbing_checks()
    result["software_checks"] = software_checks()
    out = ROOT / "phase3" / "results" / "calibration_software_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
