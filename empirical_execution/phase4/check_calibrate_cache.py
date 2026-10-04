#!/usr/bin/env python3
"""Per-encoder calibration plumbing fixtures; no real model or human judgments."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from phase3 import calibration as cal, panels, run_preparation
from phase4 import calibrate_cache as c, embeddings as e


def main():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def reject(name, function):
        try:
            function()
        except (ValueError, FileNotFoundError, FileExistsError):
            checks.append(name)
        else:
            raise AssertionError("Invalid input accepted: " + name)
    def save(path, value):
        Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="ccu_calibration_SOFTWARE_FIXTURES_") as temporary:
        root = Path(temporary)
        rows = [{"record_id": f"SOFTWARE_FIXTURE_{i}", "text": f"SOFTWARE FIXTURE NONLINGUISTIC RECORD {i}",
                 "source_unit_id": f"unknown:SOFTWARE_FIXTURE_{i}",
                 "partition": part, "original_fields": {}}
                for i, part in enumerate(["train", "calibration", "test", "calibration", "calibration"])]
        prepared = root / "prepared"
        run_preparation.write_population(prepared, rows, {}, {
            "dataset_id": "SOFTWARE_FIXTURE_NOT_EMPIRICAL_CORPUS",
            "stage": "fixed_guards_done_semantic_guard_pending", "confirmatory_study_ready": False})
        records, audit = prepared / "records.jsonl", prepared / "audit.json"
        preparation = e.read_json(audit)
        arguments, selections, locks = {}, {}, {}
        vectors = np.zeros((len(rows), 768), dtype=np.float32)
        vectors[:, 0] = 1
        vectors[:, 1] = np.arange(len(rows), dtype=np.float32) / 10
        for encoder in (e.E5, e.MPNET):
            short = "e5" if encoder == e.E5 else "mpnet"
            directory = root / short
            directory.mkdir()
            np.save(directory / "vectors.npy", vectors)
            save(directory / "row_ids.json", [r["record_id"] for r in rows])
            (directory / "chunk_log.jsonl").write_text("SOFTWARE_FIXTURE_LOG_NOT_REAL_ENCODING\n")
            save(directory / "encoding_summary.json", {"fixture": True, "semantic_model_forward_executed": False})
            assets = cal._seal({"schema": e.ASSET_SCHEMA, "encoder_id": encoder,
                "encoder_revision": "a" * 40, "tokenizer_revision": "b" * 40,
                "fixture_only": True,
                "model_files": [{"path": "config.json", "bytes": 2, "sha256": "c" * 64},
                                {"path": "model.safetensors", "bytes": 1, "sha256": "d" * 64}],
                "tokenizer_files": [{"path": "tokenizer.json", "bytes": 2, "sha256": "e" * 64}],
                "library_versions": {v: "SOFTWARE_FIXTURE_NOT_ACTUAL_RUNTIME" for v in e.VERSIONS}})
            save(directory / "assets.json", assets)
            manifest = cal._seal({"schema": e.SCHEMA, "asset_manifest_sha256": assets["sha256"],
                "preparation_audit_sha256": preparation["sha256"], "prepared_rows_sha256": cal.digest(rows),
                "records_file_sha256": e.file_hash(records),
                "normalized_records_sha256": run_preparation.text_binding(rows),
                "row_ids": [r["record_id"] for r in rows], "code_sha256": e.file_hash(e.__file__),
                "dtype": "float32", "shape": [len(rows), 768],
                "cache_file_sha256": e.file_hash(directory / "vectors.npy"),
                "chunk_log_sha256": e.file_hash(directory / "chunk_log.jsonl"),
                "encoding_summary_sha256": e.file_hash(directory / "encoding_summary.json"),
                "implementation_lock": {"library_versions": assets["library_versions"]},
                "preprocessing": {"chunk_content_tokens": 256, "chunking": "nonoverlapping_all_chunks",
                    "pooling": "content_token_weighted_mean_then_normalize", "prefix": e.ENCODERS[encoder],
                    "storage_dtype": "float32", "learner_normalization": "no_renormalization_after_FP32_storage"},
                "provenance": {"dataset_id": "SOFTWARE_FIXTURE_NOT_EMPIRICAL_CORPUS", "encoder_id": encoder,
                    "encoder_revision": "a" * 40, "tokenizer_revision": "b" * 40,
                    "encoder_role": "primary" if encoder == e.E5 else "robustness",
                    "evidence_role": "engineering_nonconfirmatory", "population_scope": "complete_prepared_source_disjoint_population"}})
            save(directory / "cache_manifest.json", manifest)
            args = (records, audit, directory / "vectors.npy", directory / "cache_manifest.json", directory / "assets.json")
            arguments[short] = args
            crows, cvectors, prov = c.load_aligned_cache(*args)
            check(short + "_calibration_only", [r["record_id"] for r in crows] == [rows[i]["record_id"] for i in [1, 3, 4]])
            check(short + "_exact_row_gather", np.array_equal(cvectors, vectors[[1, 3, 4]]))
            check(short + "_no_refilter", prov["population_refilter_performed"] is False)
            selection = c.selection_pack(*args, directory / "selection", block_size=2)
            selections[short] = selection
            check(short + "_three_pairs_only", selection["pair_population"] == 3)
            responses = e.read_json(directory / "selection/responses.template.json")
            check(short + "_nine_blank_assignments", len(responses["responses"]) == 9 and
                  all(not r["human_completed"] and not r["label"] for r in responses["responses"]))
            reject(short + "_blank_threshold_blocked", lambda: c.threshold_lock(directory / "selection/private_sampling_manifest.json",
                directory / "selection/responses.template.json", directory / "threshold.json"))
            check(short + "_no_failed_threshold_file", not (directory / "threshold.json").exists())
            reject(short + "_pack_overwrite_blocked", lambda: c.selection_pack(*args, directory / "selection"))
            # A declared numerical software lock exercises the post-selection
            # path; there are no fabricated human responses or semantic pass.
            lock = cal._seal({"schema": "ccu-calibration-selection-lock-1", "protocol_id": cal.PROTOCOL,
                "frame": selection["frame"], "selection_status": "failed_diagnostic", "threshold": .6,
                "selection_natural_pair_ids": [p["natural_pair_id"] for p in selection["pairs"]],
                "selection_pair_ids": [p["pair_id"] for p in selection["pairs"]],
                "software_fixture_only": True, "human_annotation_collected": False})
            lock_path = directory / "SOFTWARE_FIXTURE_lock.json"
            save(lock_path, lock)
            locks[short] = lock_path
            validation = c.validation_pack(*args, lock_path, directory / "validation")
            check(short + "_validation_frame_unchanged", validation["frame"] == selection["frame"])
            check(short + "_validation_uses_locked_tile", validation["frame"]["scorer"]["block_size"] == 2)
            select_ids = {a for p in selection["pairs"] for a in p["assignment_ids"]}
            validate_ids = {a for p in validation["pairs"] for a in p["assignment_ids"]}
            check(short + "_fresh_assignments", not select_ids & validate_ids)
            reject(short + "_blank_quality_blocked", lambda: c.quality_report(directory / "validation/private_sampling_manifest.json",
                directory / "validation/responses.template.json", lock_path, directory / "quality.json"))
            check(short + "_no_failed_quality_file", not (directory / "quality.json").exists())
        check("per_encoder_manifest_separation", selections["e5"]["sha256"] != selections["mpnet"]["sha256"])
        reject("E5_lock_cannot_validate_MPNet", lambda: c.validation_pack(*arguments["mpnet"], locks["e5"], root / "bad_cross_encoder"))
        reject("cross_encoder_quality_frame_refused", lambda: c.quality_report(root / "mpnet/validation/private_sampling_manifest.json",
            root / "mpnet/validation/responses.template.json", locks["e5"], root / "bad_quality.json"))
        reject("MPNet_cannot_refilter_population", lambda: c.guard_population(*arguments["mpnet"],
            locks["mpnet"], root / "does_not_exist_quality.json", root / "bad_MPNet_guard"))
        failed_quality = cal._seal({"schema": "ccu-calibration-quality-report-1",
            "selection_lock_sha256": e.read_json(locks["e5"])["sha256"], "threshold": .6,
            "statistical_and_declared_scope_eligible": False, "software_fixture_only": True})
        save(root / "failed_quality.json", failed_quality)
        reject("failed_quality_cannot_trigger_E5_guard", lambda: c.guard_population(*arguments["e5"],
            locks["e5"], root / "failed_quality.json", root / "bad_E5_guard"))
        check("failed_guard_leaves_no_population", not (root / "bad_E5_guard").exists() and not (root / "bad_MPNet_guard").exists())

        directory = root / "e5"
        original = e.read_json(directory / "cache_manifest.json")
        for field, altered in [("row_ids", original["row_ids"][::-1]), ("shape", [5, 767]),
                               ("records_file_sha256", "0" * 64), ("code_sha256", "0" * 64),
                               ("asset_manifest_sha256", "0" * 64), ("chunk_log_sha256", "0" * 64)]:
            changed = {k: v for k, v in copy.deepcopy(original).items() if k != "sha256"}
            changed[field] = altered
            save(directory / "cache_manifest.json", cal._seal(changed))
            reject("rebound_but_inconsistent_" + field, lambda: c.load_aligned_cache(*arguments["e5"]))
        save(directory / "cache_manifest.json", original)
        for field, value in [("dataset_id", "WRONG_DATASET"), ("population_scope", "WRONG_SCOPE")]:
            changed = {k: v for k, v in copy.deepcopy(original).items() if k != "sha256"}
            changed["provenance"][field] = value
            save(directory / "cache_manifest.json", cal._seal(changed))
            reject("provenance_" + field + "_mismatch_refused", lambda: c.load_aligned_cache(*arguments["e5"]))
        save(directory / "cache_manifest.json", original)
        missing_args = list(arguments["e5"]); missing_args[2] = root / "missing.npy"
        reject("missing_vectors_refused", lambda: c.load_aligned_cache(*missing_args))
        (directory / "vectors.npy").write_bytes(b"changed cache bytes")
        reject("mutated_cache_bytes_refused", lambda: c.load_aligned_cache(*arguments["e5"]))
        # MPNet CLI works without importing any transformer runtime because the
        # temporary file is a marked numerical plumbing fixture, not model data.
        args = arguments["mpnet"]
        command = [sys.executable, str(c.__file__), "selection", "--records", str(args[0]),
            "--preparation-audit", str(args[1]), "--vectors", str(args[2]),
            "--cache-manifest", str(args[3]), "--asset-manifest", str(args[4]),
            "--out", str(root / "mpnet_cli")]
        completed = subprocess.run(command, capture_output=True, text=True)
        check("MPNet_CLI_selection_path", completed.returncode == 0)
        check("CLI_keeps_readiness_false", json.loads(completed.stdout)["confirmatory_study_ready"] is False)
        completed = subprocess.run(command, capture_output=True, text=True)
        check("CLI_overwrite_rejected", completed.returncode == 2 and "BLOCKED" in completed.stderr)
        check("fixtures_still_not_human_evidence", not list(root.rglob("quality.json")) and not list(root.rglob("threshold.json")))

    result = {"passed": True, "checks": len(checks), "check_names": checks,
        "evidence_role": "temporary_software_fixtures_only_not_semantic_empirical_evidence",
        "semantic_model_forward_executed": False, "genuine_human_ratings_collected": 0,
        "semantic_threshold_selected": False, "semantic_quality_gate_passed": False,
        "production_code_sha256": e.file_hash(c.__file__), "audit_code_sha256": e.file_hash(__file__),
        "unchanged_phase3_core_sha256": e.file_hash(cal.__file__)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/calibrate_cache_software_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
