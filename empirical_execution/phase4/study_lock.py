"""Provisional artifact-integrity review, never permission for primary execution.

The program verifies actual files, cross-artifact consistency and the statistical
calculation. It cannot prove historical provenance or that a human filled a file.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "empirical_execution"))
from phase3 import calibration
from phase4 import model_selection, requests as request_module

SCHEMA = "ccu-primary-execution-lock-1"
BASE_ARTIFACTS = ("original_corpus", "parser_audit", "records", "prepared_audit",
    "vectors", "cache_manifest", "study_design", "selection_manifest",
    "selection_responses", "selection_lock", "validation_manifest",
    "validation_responses", "quality_report", "lambda_lock", "requests",
    "resource_policy", "external_review")
INPUT_BINDINGS = ("curator_array_sha256", "learner_array_sha256", "target_array_sha256",
    "record_ids_sha256", "request_manifest_sha256", "threshold_hex", "lambda_reg_hex",
    "execution_code_sha256")
REVIEW_ITEMS = ("original_snapshot_and_native_sources", "parser_dates_links_and_attribution",
    "semantic_embedding_derivation", "genuine_independent_blinded_human_collection",
    "complete_pre_execution_configuration")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def validate_execution_lock(lock, actual_inputs, *, base_dir=None):
    """Review supplied artifact consistency. Primary execution always stays blocked.

Runtime bindings here are compared with caller-supplied values. This helper does
not independently reconstruct those arrays, post-guard panels or source lineage.
Even a fully matching dossier would therefore not authorize primary execution.
"""
    base = Path(base_dir or ROOT).resolve()
    if isinstance(lock, (str, Path)):
        lock = json.loads(Path(lock).read_text())
    checks = []
    loaded = {}
    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    try:
        calibration._verify(lock)
        sealed = True
    except (ValueError, TypeError, AttributeError):
        sealed = False
    check("sealed_lock", sealed)
    if not isinstance(lock, dict):
        lock = {}
    check("schema", lock.get("schema") == SCHEMA)
    check("scope", lock.get("evidence_role") == "confirmatory_input_contract")
    dataset = lock.get("dataset_id")
    check("implemented_primary_labeled_corpus", dataset in ("civil_comments", "askubuntu", "english_stackexchange"))
    artifacts = lock.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    required = BASE_ARTIFACTS + (("tag_vocabulary", "tag_thresholds") if dataset in ("askubuntu", "english_stackexchange") else ())
    hashes = {}
    for name in required:
        descriptor = artifacts.get(name, {})
        try:
            relative = Path(descriptor["path"])
            path = (base / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(base) or not path.is_file():
                raise ValueError("Expected an existing file inside the declared artifact root")
            actual_hash = file_sha256(path)
            if descriptor.get("sha256") != actual_hash:
                raise ValueError("File checksum mismatch")
            hashes[name] = actual_hash
            if name not in ("original_corpus", "vectors", "records"):
                loaded[name] = json.loads(path.read_text())
            check("artifact." + name, True)
        except (KeyError, ValueError, TypeError, OSError) as error:
            check("artifact." + name, False, str(error))
    declared = lock.get("input_bindings", {})
    if not isinstance(declared, dict):
        declared = {}
    actual_inputs = actual_inputs if isinstance(actual_inputs, dict) else {}
    for key in INPUT_BINDINGS:
        check("runtime_binding." + key, isinstance(declared.get(key), str)
              and bool(declared[key]) and declared[key] == actual_inputs.get(key))
    for name in ("selection_manifest", "selection_lock", "validation_manifest", "quality_report"):
        try:
            calibration._verify(loaded[name])
            check("sealed." + name, True)
        except (KeyError, ValueError, TypeError) as error:
            check("sealed." + name, False, str(error))
    try:
        selection = calibration.select_threshold(loaded["selection_manifest"], loaded["selection_responses"])
        check("recomputed_selection_lock", selection == loaded["selection_lock"])
        quality = calibration.quality_gate(loaded["validation_manifest"],
            loaded["validation_responses"], loaded["selection_lock"])
        check("recomputed_quality_report", quality == loaded["quality_report"])
        check("validation_same_frame", loaded["validation_manifest"]["frame"] == selection["frame"])
        check("independent_statistical_quality_pass", quality["statistical_and_declared_scope_eligible"])
        check("runtime_threshold", declared.get("threshold_hex") == float(selection["threshold"]).hex())
        provenance = selection["frame"]["provenance"]
        check("primary_E5", provenance.get("encoder_id") == "intfloat/multilingual-e5-base")
        check("calibration_dataset", provenance.get("dataset_id") == dataset)
    except (KeyError, ValueError, TypeError) as error:
        check("complete_calibration_chain", False, str(error))
    try:
        cache = loaded["cache_manifest"]
        check("cache_file_binding", cache["cache_file_sha256"] == hashes["vectors"])
        check("cache_dataset", cache["provenance"]["dataset_id"] == dataset)
        check("cache_semantic_role", cache["provenance"]["evidence_role"] == "confirmatory_calibration")
        selection_provenance = loaded["selection_lock"]["frame"]["provenance"]
        check("same_calibration_representation", all(cache["provenance"].get(key) == selection_provenance.get(key)
              for key in ("dataset_id", "encoder_id", "encoder_revision", "evidence_role")))
        # The phase-4 wrapper enriches calibration-subset provenance without
        # mutating the full cache manifest. Its parent is the semantic content
        # seal, not the byte checksum of the pretty-printed JSON file.
        if "source_cache_manifest_sha256" in selection_provenance:
            calibration._verify(cache)
            check("calibration_parent_cache", selection_provenance["source_cache_manifest_sha256"] == cache["sha256"])
    except (KeyError, ValueError, TypeError) as error:
        check("cache_chain", False, str(error))
    try:
        requests = loaded["requests"]
        request_module.verify_manifest(requests)
        check("request_manifest_binding", requests["manifest_sha256"] == declared["request_manifest_sha256"])
        check("request_prospective_role", requests.get("evidence_role") == "prospective_confirmatory_preparation")
        check("request_dataset", requests.get("dataset_id") == dataset)
    except (KeyError, ValueError, TypeError) as error:
        check("request_chain", False, str(error))
    try:
        regularization = loaded["lambda_lock"]
        model_selection.verify(regularization)
        selected = regularization.get("lambda")
        check("runtime_lambda", selected is not None and float(selected).hex() == declared.get("lambda_reg_hex"))
        check("lambda_kind", regularization.get("kind") == "ridge_regularization")
        check("lambda_calibration_role", regularization.get("provenance", {}).get("evidence_role") == "confirmatory_calibration")
        check("lambda_dataset", regularization.get("provenance", {}).get("dataset_id") == dataset)
    except (KeyError, ValueError, TypeError) as error:
        check("lambda_chain", False, str(error))
    try:
        resources = loaded["resource_policy"]
        for key in ("threads", "per_method_memory_cap_bytes", "common_timeout_seconds", "timing_repetitions"):
            value = resources.get(key)
            check("resource." + key, isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value) and value > 0
                  and (key not in ("threads", "timing_repetitions") or isinstance(value, int)))
        check("resource.fixed_before_confirmation", resources.get("fixed_before_confirmation") is True)
        check("resource.local_execution", resources.get("execution_location") == "current_workspace")
    except (KeyError, ValueError, TypeError) as error:
        check("resource_policy", False, str(error))
    try:
        review = loaded["external_review"]
        check("external_reviewer_named", isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip()))
        date = datetime.fromisoformat(review.get("completed_at", "").replace("Z", "+00:00"))
        check("external_review_timestamp", date.tzinfo is not None)
        expected = {k: v for k, v in hashes.items() if k != "external_review"}
        check("external_review_exact_artifacts", review.get("reviewed_artifact_sha256") == expected)
        check("external_review_exact_input_bindings", review.get("reviewed_input_bindings") == declared)
        for key in REVIEW_ITEMS:
            check("external_attestation." + key, review.get("attestations", {}).get(key) is True)
    except (KeyError, ValueError, TypeError) as error:
        check("external_review_record", False, str(error))
    passed = all(c["passed"] for c in checks)
    return {"schema": "ccu-execution-lock-validation-1", "mechanical_integrity_passed": passed,
        "actual_input_bindings_match": all(c["passed"] for c in checks if c["check"].startswith("runtime_binding.")),
        "declared_artifact_consistency_passed": passed,
        "execution_allowed": False,
        "external_provenance_verified_by_software": False, "confirmatory_claims_established": False,
        "condition": "Matching hashes and review declarations alone cannot establish genuine data/human provenance or the complete empirical protocol",
        "unimplemented_validations": [
            "independent reconstruction of runtime vectors, targets and ID hashes from original artifacts",
            "complete parser, post-E5-guard and whole-source-panel lineage",
            "all method branches, isolation and study-wide stopping/resource contract"],
        "checks": checks, "blockers": [c["check"] for c in checks if not c["passed"]]}


def empty_template():
    return {"schema": SCHEMA, "dataset_id": "civil_comments", "evidence_role": "confirmatory_input_contract",
        "artifacts": {key: {"path": "", "sha256": ""} for key in BASE_ARTIFACTS},
        "input_bindings": {key: "" for key in INPUT_BINDINGS},
        "note": "Unsealed incomplete template; do not fill attestations without actual independent review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("template"); a.add_argument("--out", required=True)
    a = sub.add_parser("check"); a.add_argument("--lock", required=True); a.add_argument("--actual-inputs", required=True)
    a.add_argument("--base-dir", default=str(ROOT)); a.add_argument("--out", required=True)
    args = parser.parse_args()
    value = empty_template() if args.command == "template" else validate_execution_lock(
        args.lock, json.loads(Path(args.actual_inputs).read_text()), base_dir=args.base_dir)
    with Path(args.out).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"output": args.out, "mechanical_integrity_passed": value.get("mechanical_integrity_passed", False)}))


if __name__ == "__main__":
    main()
