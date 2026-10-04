#!/usr/bin/env python3
"""Panel engineering audit on cached natural records, plus algebraic unit fixtures.

No constructed fixture is an empirical dataset. Lexical features are explicitly
diagnostic and never passed off as E5, native source provenance or semantic truth.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "empirical_execution"))
from ccu.data import lexical_engineering_features
from phase3.panels import (HASH_DENOMINATOR, digest_partition, fixed_guard, graph_normalize,
                           hash_integer, prepare_partitions, prepare_population,
                           reference_cosines, select_panels, source_unit, semantic_guard)

OUT = Path(__file__).resolve().parent
DESIGN = json.loads((ROOT / "output/empirical_program/study_design.json").read_text())
checks = []


def check(name, ok):
    if not ok:
        raise AssertionError(name)
    checks.append(name)


def rejected(name, callback):
    try:
        callback()
    except (ValueError, TypeError):
        checks.append(name)
    else:
        raise AssertionError(name)


def run():
    # Algebraic software boundary fixtures, deliberately not study observations.
    check("hash_serialization_known_answer", hash_integer("salt", "unit") == int(hashlib.sha256(b"salt\0unit").hexdigest(), 16))
    for numerator, denominator, before, after in [(7, 10, "train", "calibration"), (17, 20, "calibration", "test")]:
        ceiling = (HASH_DENOMINATOR * numerator + denominator - 1) // denominator
        check(f"exact_quantile_{numerator}_{denominator}", digest_partition(ceiling - 1) == before and digest_partition(ceiling) == after)
    rejected("bool_digest_rejected", lambda: digest_partition(True))
    check("community_owner_singleton", source_unit({"record_id": "a", "original_fields": {"site": "askubuntu", "OwnerUserId": -1}}, "askubuntu")[0] == "unknown:a")
    rejected("bool_source_rejected", lambda: source_unit({"record_id": "a", "original_fields": {"site": "askubuntu", "OwnerUserId": True}}, "askubuntu"))
    rejected("false_source_rejected", lambda: source_unit({"record_id": "a", "original_fields": {"site": "askubuntu", "OwnerUserId": False}}, "askubuntu"))
    rejected("nonfinite_source_rejected", lambda: source_unit({"record_id": "a", "original_fields": {"publication_id": "p", "article_id": float("nan")}}, "civil_comments"))
    check("blank_source_singleton", source_unit({"record_id": "a", "original_fields": {"publication_id": " ", "article_id": 1}}, "civil_comments")[0] == "unknown:a")
    check("negative_source_singleton", source_unit({"record_id": "a", "original_fields": {"site": "askubuntu", "OwnerUserId": "-2"}}, "askubuntu")[0] == "unknown:a")
    fixture = [
        {"record_id": "t", "partition": "train", "text": "same", "original_fields": {}},
        {"record_id": "c", "partition": "calibration", "text": "same", "original_fields": {}},
        {"record_id": "e", "partition": "test", "text": "same", "original_fields": {}},
        {"record_id": "tc", "partition": "train", "text": "unique1", "original_fields": {"parent_id": "e"}},
        {"record_id": "tp", "partition": "train", "text": "unique2", "original_fields": {}},
        {"record_id": "ec", "partition": "test", "text": "unique3", "original_fields": {"parent_id": "tp"}},
        {"record_id": "cl", "partition": "calibration", "text": "unique4", "original_fields": {}},
        {"record_id": "s", "partition": "train", "text": "unique5", "original_fields": {"parent_id": -1}},
    ]
    guard, ga = fixed_guard(fixture, [("cl", "e")])
    keep = {r["record_id"] for r in guard}
    check("exact_precedence_and_calibration_duplicate_link", keep == {"e", "tp", "ec", "s"})
    check("parent_direction_and_sentinel", "tp" in keep and "tc" not in keep and not ga["unresolved_native_links"])
    check("guard_counts_disjoint", ga["removed_training_rows"] == 2 and ga["removed_calibration_rows"] == 2)
    check("guard_input_not_mutated", len(fixture) == 8)
    unit_vectors = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    unit_rows = [dict(record_id=str(i), partition=p) for i, p in enumerate(["train", "test", "calibration"])]
    sg, _, _ = semantic_guard(unit_rows, unit_vectors, 1.0, diagnostic_only=True)
    check("threshold_strict_equality_kept", len(sg) == 3)
    sg, _, _ = semantic_guard(unit_rows, unit_vectors, float(np.nextafter(1., 0.)), diagnostic_only=True)
    check("threshold_strict_above_removed", [r["record_id"] for r in sg] == ["1", "2"])
    rejected("lexical_cannot_satisfy_E5", lambda: semantic_guard(unit_rows, unit_vectors, .5))
    rejected("FP64_cache_rejected", lambda: semantic_guard(unit_rows, unit_vectors.astype(np.float64), .5, diagnostic_only=True))
    bad = unit_vectors.copy(); bad[0] = 0
    rejected("zero_cache_rejected", lambda: semantic_guard(unit_rows, bad, .5, diagnostic_only=True))

    outputs = {}
    for name, size in [("civil_comments", 100), ("cc_news", 90)]:
        path = ROOT / "empirical_execution/data" / (name + "_engineering_preview.jsonl")
        cached = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        check(name + "_natural_fixture_size", len(cached) == size)
        rows = [{"record_id": r["record_id"], "text": r["text"],
                 "labels": [] if r["label"] is None else [r["label"]],
                 "original_fields": r["fields"], "provenance": r["provenance"]} for r in cached]
        arr, features = lexical_engineering_features(cached, 768)
        assigned, split_audit = prepare_partitions(rows, name, DESIGN, allow_preview_sources=(name == "cc_news"))
        original = json.dumps(rows, sort_keys=True)
        grouped = {}
        for row in assigned:
            grouped.setdefault(row["source_unit_id"], set()).add(row["partition"])
        check(name + "_source_disjoint", all(len(p) == 1 for p in grouped.values()))
        check(name + "_no_authentic_native_claim", split_audit["native_source_units"] == 0)
        final, panels, audit = prepare_population(rows, name, DESIGN, fp32_cache=arr, tau=.6,
                                                 allow_preview_sources=(name == "cc_news"),
                                                 diagnostic_only=True, encoder_id="lexical_hash_engineering_only", block_size=7)
        check(name + "_input_rows_immutable", original == json.dumps(rows, sort_keys=True))
        check(name + "_confirmatory_gate_closed", audit["confirmatory_ready"] is False)
        fixed, _ = fixed_guard(assigned)
        idx = {r["record_id"]: i for i, r in enumerate(assigned)}
        fx = arr[[idx[r["record_id"]] for r in fixed]]
        alternate, _, alt_audit = semantic_guard(fixed, fx, .6, diagnostic_only=True, block_size=19, encoder_id="lexical_hash_engineering_only")
        check(name + "_chunk_size_same_population", [r["record_id"] for r in final] == [r["record_id"] for r in alternate])
        check(name + "_chunk_size_same_witness_scores", audit["semantic_guard"]["removed"] == alt_audit["removed"])
        check(name + "_all_pairs_scored", audit["semantic_guard"]["pairs_scored"] == audit["semantic_guard"]["exhaustive_pairs_expected"])
        panels2, _ = select_panels(list(reversed(final)), DESIGN)
        check(name + "_row_order_invariant_panels", panels2 == panels)
        check(name + "_disjoint_pools", not (set(panels["primary"]) & set(panels["replication"])))
        # Each selected source contains its entire surviving training group.
        for panel in ("primary", "replication"):
            selected = set(panels[panel])
            source_ids = {r["source_unit_id"] for r in final if r["record_id"] in selected}
            check(name + "_whole_groups_" + panel, all(r["record_id"] in selected for r in final if r["partition"] == "train" and r["source_unit_id"] in source_ids))
        if name == "civil_comments":
            check("civil_only_unknown_singletons", set(split_audit["source_kind_counts"]) == {"unknown_singleton"})
        else:
            check("missing_test_partition_reported", "missing_required_partitions:test" in audit["remaining_gates"])
            tiny, tiny_audit = select_panels(final, DESIGN, target=1)
            check("whole_boundary_group_overshoot_reported", any(v.get("overshoot", 0) > 0 for v in tiny_audit.values() if isinstance(v, dict)))
        # Pure numerical independent scalar check on actual natural vectors.
        norm = graph_normalize(arr[:4])
        score = reference_cosines(norm, norm)
        scalar = np.empty((4, 4))
        for i in range(4):
            for j in range(4):
                value = 0.
                for k in range(norm.shape[1]):
                    value = value + float(norm[i, k]) * float(norm[j, k])
                scalar[i, j] = value
        check(name + "_reference_scalar_dot_identical", np.array_equal(score, scalar))
        outputs[name] = {"scope": "nonconfirmatory_natural_preview_software_engineering_only",
                         "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "features": features, "audit": audit,
                         "panels": panels}
    result = {"passed": True, "checks_count": len(checks), "checks": checks,
              "synthetic_empirical_datasets_used": False,
              "algebra_fixtures_are_software_boundary_tests_only": True,
              "corpora": outputs,
              "code_sha256": hashlib.sha256((OUT / "panels.py").read_bytes()).hexdigest(),
              "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "panel_engineering_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "checks": len(checks),
                      "corpora": {k: {"split": v["audit"]["split"]["partition_counts"],
                                      "semantic_pairs": v["audit"]["semantic_guard"]["pairs_scored"],
                                      "semantic_removed": v["audit"]["semantic_guard"]["removed_training_rows"],
                                      "primary": len(v["panels"]["primary"]),
                                      "replication": len(v["panels"]["replication"])}
                                  for k, v in outputs.items()}}, indent=2))


if __name__ == "__main__":
    run()
