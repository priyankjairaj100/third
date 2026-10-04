#!/usr/bin/env python3
"""Small local audits on unchanged natural-text engineering previews.

The hash representation and thresholds are unvalidated software fixtures, not
semantic-deduplication measurements. No labels, synthetic text, injected
similarities, neural encoder, or test-set utility measures are used.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
import random
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ccu.core import build_blocker_graph, stable_priority
from ccu.data import (lexical_engineering_features, normalize_text,
                      read_natural_jsonl, sha256_file)
from ccu.structure import (derive_seed, expected_admissions,
                           generate_request_trajectory, trajectory_observations)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "structure_audit.json"
MASTER_SEED = 20261004
THRESHOLDS = (0.35, 0.60, 0.85)
DIMENSION = 128
HORIZON = 8
CHECKPOINTS = (1, 2, 4, 8)
TRAJECTORIES_PER_ARM = 64


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def ids_hash(values):
    return hashlib.sha256(canonical_json(sorted(values))).hexdigest()


def fresh_select(features, record_ids, priority, deleted_ids, threshold,
                 mode="raw-neighbor"):
    """Independent retained-row scorer; rebuild normalization and all scores.

    This reads no BlockerGraph, blocker cache, or previous score matrix. It uses
    the fixed full-priority order restricted to retained IDs. In raw-neighbor
    mode every earlier retained row can block; in greedy mode only a previously
    selected retained row can block. Output is stable record IDs.
    """
    if mode not in {"raw-neighbor", "greedy"}:
        raise ValueError("unsupported fresh selector")
    positions = {record_id: i for i, record_id in enumerate(record_ids)}
    deleted = set(deleted_ids)
    if not deleted <= set(positions):
        raise ValueError("unknown deletion ID")
    retained = [record_id for record_id in priority if record_id not in deleted]
    if not retained:
        return set()
    rows = np.array(features[[positions[x] for x in retained]], dtype=np.float64,
                    order="C", copy=True)
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero input row")
    rows /= norms[:, None]
    scores = rows @ rows.T
    selected = []
    for i in range(len(retained)):
        earlier = range(i) if mode == "raw-neighbor" else selected
        if not any(float(scores[i, j]) > threshold for j in earlier):
            selected.append(i)
    return {retained[i] for i in selected}


def minimum_threshold_margin(features, threshold):
    """Diagnostic only; this is not an interval-certified edge guarantee."""
    rows = np.asarray(features, dtype=np.float64).copy()
    rows /= np.linalg.norm(rows, axis=1)[:, None]
    scores = rows @ rows.T
    values = scores[np.tril_indices(len(rows), -1)]
    return float(np.min(np.abs(values - threshold))) if len(values) else None


def exact_text_select(records, priority, deleted_ids):
    """One fixed-priority representative per exactly normalized text class."""
    deleted = set(deleted_ids)
    text = {record["record_id"]: normalize_text(record["text"]) for record in records}
    seen = set()
    selected = set()
    for record_id in priority:
        if record_id in deleted:
            continue
        value = text[record_id]
        if value not in seen:
            seen.add(value)
            selected.add(record_id)
    return selected


def cached_source_selection(graph, deleted_sources):
    """Distinct surviving-source counts, independent of graph.selected_indices."""
    withdrawn = set(deleted_sources)
    selected = set()
    for i, record_id in enumerate(graph.record_ids):
        source = graph.source_ids[i]
        blocker_sources = {graph.source_ids[int(j)] for j in
                           graph.indices[graph.indptr[i]:graph.indptr[i + 1]]}
        surviving_source_count = sum(source_id not in withdrawn
                                     for source_id in blocker_sources)
        if source not in withdrawn and surviving_source_count == 0:
            selected.add(record_id)
    return selected


def news_source_audit(records, features):
    ids = [record["record_id"] for record in records]
    sources = [record["source_id"] for record in records]
    universe = sorted(set(sources))
    if len(universe) != 5:
        raise ValueError("registered source audit requires the acquired five-host preview")
    runs = []
    failures = []
    for threshold in THRESHOLDS:
        graph = build_blocker_graph(features, ids, threshold, source_ids=sources)
        initial = fresh_select(features, ids, graph.priority, (), threshold)
        if initial != cached_source_selection(graph, ()):
            failures.append({"threshold": threshold, "stage": "initial_selection"})
        queries = []
        by_size = {k: [] for k in range(len(universe) + 1)}
        for size in range(len(universe) + 1):
            for deleted_sources in itertools.combinations(universe, size):
                dead = {ids[i] for i, source in enumerate(sources)
                        if source in deleted_sources}
                fresh = fresh_select(features, ids, graph.priority, dead, threshold)
                cached = cached_source_selection(graph, deleted_sources)
                equality = fresh == cached
                if not equality:
                    failures.append({"threshold": threshold, "stage": "source_subset",
                                     "deleted_sources": list(deleted_sources),
                                     "only_fresh_ids": sorted(fresh - cached),
                                     "only_cached_ids": sorted(cached - fresh)})
                admissions = len(fresh - initial)
                by_size[size].append(admissions)
                queries.append({
                    "deleted_sources": list(deleted_sources), "deleted_source_count": size,
                    "deleted_record_count": len(dead), "retained_record_count": len(ids) - len(dead),
                    "fresh_selected_count": len(fresh), "cached_selected_count": len(cached),
                    "selected_sets_equal": equality, "admissions": admissions,
                    "selected_ids_sha256": ids_hash(fresh),
                    "empty_retained_target": not bool(len(ids) - len(dead)),
                })
        expectations = []
        for size, admissions in by_size.items():
            empirical_exact_mean = math.fsum(admissions) / len(admissions)
            formula = expected_admissions(graph, size, "source", universe)
            error = abs(empirical_exact_mean - formula["expected_admissions"])
            passed = error <= 1e-12
            if not passed:
                failures.append({"threshold": threshold, "stage": "exact_expectation",
                                 "source_count": size, "absolute_error": error})
            expectations.append({
                "source_count": size, "enumerated_subsets": len(admissions),
                "expected_subsets_binomial": math.comb(len(universe), size),
                "enumerated_mean_admissions": empirical_exact_mean,
                "formula_mean_admissions": formula["expected_admissions"],
                "absolute_difference": error, "passed_diagnostic_tolerance": passed,
                "subsets_with_admissions": sum(value > 0 for value in admissions),
                "scope": "exhaustive finite-preview mean count; not population prevalence",
            })
        runs.append({"threshold": threshold, "initial_selected_count": len(initial),
                     "minimum_pair_threshold_margin": minimum_threshold_margin(features, threshold),
                     "subsets_tested": len(queries),
                     "selected_set_mismatches": sum(not row["selected_sets_equal"] for row in queries),
                     "queries": queries, "expectations_by_request_size": expectations})
    return {"records": len(ids), "source_type": "observed_original_url_host",
            "sources": universe, "source_sizes": dict(sorted(Counter(sources).items())),
            "not_registrable_domain_or_verified_publisher_ownership": True,
            "source_subset_checks": sum(run["subsets_tested"] for run in runs),
            "expectation_checks": sum(len(run["expectations_by_request_size"]) for run in runs),
            "failures": failures, "runs": runs}


def civil_request_audit(records, features):
    ids = [record["record_id"] for record in records]
    runs = []
    failures = []
    for threshold in THRESHOLDS:
        graph = build_blocker_graph(features, ids, threshold)
        initial = fresh_select(features, ids, graph.priority, (), threshold)
        initial_excluded = sorted(set(ids) - initial)
        for arm in ("R", "U", "A"):
            paths = []
            for path_index in range(TRAJECTORIES_PER_ARM):
                seed = derive_seed(MASTER_SEED, "civil_structure_audit", arm, str(path_index))
                trajectory = generate_request_trajectory(
                    graph, arm, seed, HORIZON, CHECKPOINTS)
                cached_rows = trajectory_observations(graph, trajectory)
                observations = []
                previous_dead = set()
                for row in cached_rows:
                    checkpoint = row["cumulative_deleted_units"]
                    dead = set(trajectory.prefix(checkpoint))
                    fresh = fresh_select(features, ids, graph.priority, dead, threshold)
                    cached_indices = graph.selected_indices(dead)
                    cached_ids = {graph.record_ids[int(i)] for i in cached_indices}
                    admissions = len(fresh - initial)
                    consistency = (fresh == cached_ids
                                   and len(fresh) == row["current_selected_records"]
                                   and admissions == row["admissions"]
                                   and previous_dead <= dead
                                   and row["remaining_horizon"] == trajectory.initial_horizon - len(dead))
                    if arm == "U":
                        consistency = consistency and dead <= set(initial_excluded) and initial <= fresh
                    if not consistency:
                        failures.append({"threshold": threshold, "arm": arm,
                                         "path_index": path_index, "checkpoint": checkpoint})
                    observations.append({**row, "fresh_selected_ids_sha256": ids_hash(fresh),
                                         "fresh_selected_count": len(fresh),
                                         "fresh_admissions": admissions,
                                         "passed_selection_and_cumulative_checks": consistency})
                    previous_dead = dead
                path_dict = trajectory.to_dict()
                hash_request = {key: value for key, value in path_dict.items()
                                if key != "stress_search_seconds"}
                paths.append({"path_index": path_index, "request": path_dict,
                              "request_hash_scope": "request metadata excluding stress_search_seconds",
                              "request_sha256": hashlib.sha256(canonical_json(hash_request)).hexdigest(),
                              "checkpoints": observations})
            aggregate = []
            all_levels = sorted({row["cumulative_deleted_units"] for path in paths
                                 for row in path["checkpoints"]})
            for level in all_levels:
                values = [row["fresh_admissions"] for path in paths for row in path["checkpoints"]
                          if row["cumulative_deleted_units"] == level]
                observed_mean = math.fsum(values) / len(values)
                expectation = None
                if arm in {"R", "U"}:
                    frame = None if arm == "R" else initial_excluded
                    expectation = expected_admissions(graph, level, "record", frame)["expected_admissions"]
                aggregate.append({
                    "checkpoint": level, "independent_trajectories": len(values),
                    "mean_admissions": observed_mean,
                    "trajectories_with_admissions": sum(v > 0 for v in values),
                    "zero_admission_trajectories": sum(v == 0 for v in values),
                    "maximum_admissions": max(values, default=0),
                    "finite_preview_uniform_mean_prediction": expectation,
                    "monte_carlo_minus_prediction": observed_mean - expectation if expectation is not None else None,
                    "population": "targeted structural stress" if arm == "A" else
                                  "uniform initially excluded records" if arm == "U" else "uniform records",
                })
            runs.append({"threshold": threshold, "arm": arm,
                         "initial_selected_count": len(initial),
                         "initial_excluded_count": len(initial_excluded),
                         "minimum_pair_threshold_margin": minimum_threshold_margin(features, threshold),
                         "trajectories": paths, "aggregate_by_checkpoint": aggregate,
                         "checkpoint_checks": sum(len(path["checkpoints"]) for path in paths),
                         "distinct_ordered_trajectories": len({tuple(path["request"]["deletion_order"]) for path in paths}),
                         "distinct_checkpoint_deleted_sets": len({tuple(sorted(path["request"]["deletion_order"][:row["cumulative_deleted_units"]]))
                                                                  for path in paths for row in path["checkpoints"]}),
                         "zero_checkpoint_results": sum(row["fresh_admissions"] == 0
                                                        for path in paths for row in path["checkpoints"]),
                         "stress_paths_without_candidate": sum(path["request"]["selected_candidate"] is None
                                                               for path in paths) if arm == "A" else None,
                         "monte_carlo_agreement_is_descriptive_not_an_acceptance_test": True})
    return {"records": len(ids), "source_withdrawal_not_tested_missing_native_source_fields": True,
            "trajectories_per_arm_per_threshold": TRAJECTORIES_PER_ARM,
            "trajectory_count": sum(len(run["trajectories"]) for run in runs),
            "checkpoint_checks": sum(run["checkpoint_checks"] for run in runs),
            "failures": failures, "runs": runs}


def negative_controls(records, features):
    ids = [record["record_id"] for record in records]
    priority = stable_priority(ids)
    results = []
    failures = []
    configurations = [("exact_normalized_text", None), *[("fixed_order_greedy", t) for t in THRESHOLDS]]
    for name, threshold in configurations:
        def select(dead):
            if name == "exact_normalized_text":
                return exact_text_select(records, priority, dead)
            return fresh_select(features, ids, priority, dead, threshold, mode="greedy")
        initial = select(())
        # Crucial: use the CONTROL's own excluded pool, not suppression-U.
        excluded = sorted(set(ids) - initial)
        horizon = min(HORIZON, len(excluded))
        levels = sorted({min(k, horizon) for k in CHECKPOINTS}) if horizon else []
        paths = []
        for path_index in range(TRAJECTORIES_PER_ARM):
            seed = derive_seed(MASTER_SEED, "control_excluded_only", name,
                               str(threshold), str(path_index))
            order = random.Random(seed).sample(excluded, horizon)
            rows = []
            for level in levels:
                dead = set(order[:level])
                fresh = select(dead)
                passed = fresh == initial
                if not passed:
                    failures.append({"control": name, "threshold": threshold,
                                     "path_index": path_index, "checkpoint": level})
                rows.append({"checkpoint": level, "selected_set_unchanged": passed,
                             "added_selected_records": len(fresh - initial),
                             "removed_selected_records": len(initial - fresh),
                             "deleted_only_initially_excluded_by_this_control": dead <= set(excluded)})
            paths.append({"path_index": path_index, "seed": seed,
                          "deletion_order": order, "checkpoints": rows})
        results.append({"control": name, "threshold": threshold,
                        "text_equality": "NFC + whitespace normalization; otherwise exact" if threshold is None else None,
                        "initial_selected_count": len(initial), "initial_excluded_count": len(excluded),
                        "status": "tested" if excluded else "no_excluded_pool_not_empirically_testable",
                        "planned_trajectories": TRAJECTORIES_PER_ARM,
                        "nonempty_trajectories_tested": len(paths) if horizon else 0,
                        "checkpoint_cases_tested": sum(len(path["checkpoints"]) for path in paths),
                        "count_unit": "checkpoint executions; independently sampled paths can repeat the same deleted set",
                        "distinct_nonempty_deleted_sets": len({tuple(sorted(path["deletion_order"][:row["checkpoint"]]))
                                                              for path in paths for row in path["checkpoints"]}),
                        "zero_change_cases": sum(row["selected_set_unchanged"] for path in paths for row in path["checkpoints"]),
                        "trajectories": paths})
    return {"scope": "control-specific excluded-only pools; absent pools are not counted as passed cases",
            "failures": failures, "controls": results,
            "checkpoint_cases_tested": sum(result["checkpoint_cases_tested"] for result in results),
            "zero_change_cases": sum(result["zero_change_cases"] for result in results),
            "distinct_control_config_and_deleted_set_cases": sum(result["distinct_nonempty_deleted_sets"] for result in results)}


def input_provenance(path, records):
    raw = {}
    for record in records:
        provenance = record["provenance"]
        response_name = provenance.get("response_file")
        if response_name:
            response_path = ROOT / "data" / response_name
            actual_hash = sha256_file(response_path) if response_path.exists() else None
            declared_hash = provenance.get("response_sha256")
            if actual_hash != declared_hash:
                raise ValueError(f"raw provenance hash mismatch: {response_name}")
            raw[response_name] = actual_hash
    return {"relative_path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
            "records": len(records), "record_ids_sha256": ids_hash(r["record_id"] for r in records),
            "raw_connector_response_sha256": dict(sorted(raw.items())),
            "source_bytes_and_repository_revision_verified": False,
            "text_fidelity": sorted({r["provenance"].get("text_fidelity", "unspecified") for r in records})}


def main():
    started = time.perf_counter()
    civil_path = ROOT / "data" / "civil_comments_engineering_preview.jsonl"
    news_path = ROOT / "data" / "cc_news_engineering_preview.jsonl"
    civil = read_natural_jsonl(civil_path)
    news = read_natural_jsonl(news_path, require_sources=True)
    civil_features, civil_meta = lexical_engineering_features(civil, DIMENSION)
    news_features, news_meta = lexical_engineering_features(news, DIMENSION)
    with threadpool_limits(limits=1):
        news_results = news_source_audit(news, news_features)
        civil_results = civil_request_audit(civil, civil_features)
        control_results = negative_controls(civil, civil_features)
    failure_count = sum(len(result["failures"]) for result in
                        (news_results, civil_results, control_results))
    script_paths = [Path(__file__), ROOT / "ccu" / "core.py", ROOT / "ccu" / "data.py",
                    ROOT / "ccu" / "structure.py"]
    report = {
        "schema_version": 1, "status": "passed_software_audit" if not failure_count else "failed_software_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "nonconfirmatory local software checks on natural connector-rendered previews",
        "confirmatory": False, "NLP_semantic_quality_claim": False,
        "no_synthetic_documents_labels_or_edges": True,
        "no_news_targets_or_learner": True, "no_remote_compute": True,
        "design": {"master_seed": MASTER_SEED, "dimension": DIMENSION, "thresholds": THRESHOLDS,
                   "record_horizon": HORIZON, "checkpoints": CHECKPOINTS,
                   "trajectories_per_arm_per_threshold": TRAJECTORIES_PER_ARM,
                   "thresholds_semantically_calibrated": False,
                   "one_fixed_full_priority_restricted_after_deletion": True,
                   "threads_per_numerical_backend": 1},
        "inputs": {"civil": input_provenance(civil_path, civil), "news": input_provenance(news_path, news)},
        "features": {"civil": civil_meta, "news": news_meta},
        "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in script_paths},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "method_API": {
            "encoder": "ccu.data.lexical_engineering_features(records, 128)",
            "graph": "ccu.core.build_blocker_graph(features, ids, threshold, source_ids=...)",
            "source_cached": "run_structure_audit.cached_source_selection(graph, deleted_sources)",
            "fresh_oracle": "run_structure_audit.fresh_select(features, ids, priority, deleted_ids, threshold)",
            "source_expectation": "ccu.structure.expected_admissions(graph, k, 'source', genuine_host_frame)",
            "record_sampler": "ccu.structure.generate_request_trajectory(graph, arm, seed, 8, [1,2,4,8])",
            "control": "control-specific initially excluded pool, independent fresh control rerun"},
        "news_source_audit": news_results, "civil_request_audit": civil_results,
        "negative_controls": control_results, "failure_count": failure_count,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["small convenience previews, not a representative natural corpus audit",
                        "connector-rendered text and unverified immutable upstream revisions",
                        "unvalidated lexical hashing and thresholds, not registered E5 semantic geometry",
                        "source groups are observed URL hosts, not verified publisher ownership",
                        "fixed-point selections tested; no transcript privacy or full-machine erasure claim",
                        "ordinary floating arithmetic and margin diagnostics, not an interval certificate",
                        "Monte Carlo checkpoints within a trajectory are dependent; no pooled inferential claim",
                        "any control with no excluded records would be explicitly inapplicable"],
    }
    report["integrity"] = {"algorithm": "sha256",
                           "scope": "canonical sorted compact UTF-8 JSON of complete report excluding integrity",
                           "content_sha256": hashlib.sha256(canonical_json(report)).hexdigest()}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    summary = {"status": report["status"], "failure_count": failure_count,
               "news_source_subset_checks": news_results["source_subset_checks"],
               "news_expectation_checks": news_results["expectation_checks"],
               "civil_trajectories": civil_results["trajectory_count"],
               "civil_checkpoint_checks": civil_results["checkpoint_checks"],
               "negative_control_checkpoint_checks": control_results["checkpoint_cases_tested"],
               "negative_control_zero_change_cases": control_results["zero_change_cases"],
               "elapsed_seconds": report["elapsed_seconds"], "output": str(OUTPUT),
               "output_file_sha256": sha256_file(OUTPUT)}
    print(json.dumps(summary, indent=2))
    return int(failure_count > 0)


if __name__ == "__main__":
    sys.exit(main())
