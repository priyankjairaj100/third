#!/usr/bin/env python3
"""Local E5/MPNet calibration and a separate E5-only population guard.

All sampling, human-response validation, threshold selection and finite-population
statistics use unchanged phase3.calibration. Metadata checks establish internal
consistency, not historical model provenance or human annotation authenticity.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "empirical_execution"))
from phase3 import calibration as cal, panels, run_preparation
from phase4 import embeddings as emb

WORKFLOW = "ccu-per-encoder-calibration-v1"


def load_aligned_cache(records_path, preparation_audit_path, vectors_path,
                       cache_manifest_path, asset_manifest_path):
    """Keep complete row order; explicitly gather only the calibration partition."""
    rows, preparation = emb.load_prepared_records(records_path, preparation_audit_path)
    manifest, assets = emb.read_json(cache_manifest_path), emb.read_json(asset_manifest_path)
    cal._verify(manifest)
    cal._verify(assets)
    if manifest.get("schema") != emb.SCHEMA or assets.get("schema") != emb.ASSET_SCHEMA:
        raise ValueError("Expected versioned semantic-cache and local-asset manifests")
    if manifest.get("asset_manifest_sha256") != assets["sha256"]:
        raise ValueError("Asset manifest does not match cache derivation")
    if (manifest.get("preparation_audit_sha256") != preparation["sha256"]
            or manifest.get("prepared_rows_sha256") != cal.digest(rows)
            or manifest.get("records_file_sha256") != emb.file_hash(records_path)
            or manifest.get("normalized_records_sha256") != run_preparation.text_binding(rows)):
        raise ValueError("Cache text/source/partition/full-record binding mismatch")
    ids = [r["record_id"] for r in rows]
    if manifest.get("row_ids") != ids:
        raise ValueError("Complete ordered cache IDs differ from prepared IDs")
    if manifest.get("code_sha256") != emb.file_hash(emb.__file__):
        raise ValueError("Cache derivation code differs from this frozen encoder implementation")
    provenance = manifest.get("provenance", {})
    if (not isinstance(preparation.get("dataset_id"), str)
            or provenance.get("dataset_id") != preparation["dataset_id"]
            or provenance.get("population_scope") != "complete_prepared_source_disjoint_population"):
        raise ValueError("Cache dataset identity or complete-prepared scope differs from preparation")
    encoder = provenance.get("encoder_id")
    if encoder not in emb.ENCODERS or assets.get("encoder_id") != encoder:
        raise ValueError("Cache and assets must name the same supported E5 or MPNet encoder")
    for key in ("encoder_revision", "tokenizer_revision"):
        revision = provenance.get(key)
        if (not isinstance(revision, str) or not emb.REVISION.fullmatch(revision)
                or assets.get(key) != revision):
            raise ValueError("Missing or inconsistent full model/tokenizer revision")
    if provenance.get("encoder_role") != ("primary" if encoder == emb.E5 else "robustness"):
        raise ValueError("Incorrect primary/robustness encoder role")
    if provenance.get("evidence_role") not in {"engineering_nonconfirmatory", "confirmatory_calibration"}:
        raise ValueError("Declared calibration evidence role required")
    # Cached-vector reuse does not need a model runtime, but its immutable
    # derivation manifest must still be structurally complete and consistent.
    for key in ("model_files", "tokenizer_files"):
        files = assets.get(key)
        if not isinstance(files, list) or not files:
            raise ValueError("Complete model/tokenizer byte inventory required")
        names = []
        for item in files:
            if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                    or not item["path"] or Path(item["path"]).is_absolute()
                    or ".." in Path(item["path"]).parts
                    or not isinstance(item.get("bytes"), int) or isinstance(item["bytes"], bool)
                    or item["bytes"] < 0 or not isinstance(item.get("sha256"), str)
                    or len(item["sha256"]) != 64
                    or any(v not in "0123456789abcdef" for v in item["sha256"])):
                raise ValueError("Malformed local asset byte inventory")
            names.append(item["path"])
        if len(set(names)) != len(names):
            raise ValueError("Repeated asset inventory path")
        if key == "model_files" and ("config.json" not in names or not any(p.endswith(".safetensors") for p in names)):
            raise ValueError("Model inventory lacks required config and safetensors weights")
    versions = assets.get("library_versions", {})
    if (set(versions) != set(emb.VERSIONS) or any(not isinstance(v, str) or not v for v in versions.values())
            or manifest.get("implementation_lock", {}).get("library_versions") != versions):
        raise ValueError("Cache/runtime version declarations differ from pinned assets")
    expected_preprocessing = {"chunk_content_tokens": 256, "chunking": "nonoverlapping_all_chunks",
        "pooling": "content_token_weighted_mean_then_normalize", "prefix": emb.ENCODERS[encoder],
        "storage_dtype": "float32", "learner_normalization": "no_renormalization_after_FP32_storage"}
    if manifest.get("preprocessing") != expected_preprocessing:
        raise ValueError("Cache preprocessing differs from the frozen full-document contract")
    if (manifest.get("cache_file_sha256") != emb.file_hash(vectors_path)
            or manifest.get("dtype") != "float32"
            or manifest.get("shape") != [len(rows), 768]):
        raise ValueError("Frozen vector bytes, dtype or shape differ")
    directory = Path(cache_manifest_path).parent
    for key, filename in (("chunk_log_sha256", "chunk_log.jsonl"),
                          ("encoding_summary_sha256", "encoding_summary.json")):
        if manifest.get(key) != emb.file_hash(directory / filename):
            raise ValueError("Encoder derivation log bytes differ: " + filename)
    if emb.read_json(directory / "row_ids.json") != ids:
        raise ValueError("Encoder row-ID artifact is not aligned")
    cache = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.dtype("float32") or cache.shape != (len(rows), 768):
        raise ValueError("Primary and robustness calibration require aligned N-by-768 FP32 vectors")
    for start in range(0, len(rows), 256):
        panels.graph_normalize(cache[start:start + 256])
    indices = [i for i, row in enumerate(rows) if row["partition"] == "calibration"]
    if len(indices) < 2:
        raise ValueError("At least two calibration-only records are required")
    calibration_rows = [rows[i] for i in indices]
    calibration_vectors = cache[indices]  # Explicit O(N_cal * 768) FP32 gather.
    derived_provenance = {**provenance,
        "population_scope": "source_disjoint_calibration_partition_of_complete_prepared_population",
        "workflow_id": WORKFLOW,
        "workflow_code_sha256": emb.file_hash(__file__),
        "source_cache_manifest_sha256": manifest["sha256"],
        "source_preparation_audit_sha256": preparation["sha256"],
        "source_full_records_sha256": cal.digest(rows),
        "calibration_record_ids_sha256": cal.digest([r["record_id"] for r in calibration_rows]),
        "calibration_source_ids_sha256": cal.digest([r["source_unit_id"] for r in calibration_rows]),
        "population_refilter_performed": False,
        "external_provenance_verified_by_this_module": False,
        "confirmatory_study_ready": False}
    return calibration_rows, calibration_vectors, derived_provenance


def verify_workflow_scope(value):
    cal._verify(value)
    frame = value.get("frame", {})
    provenance = frame.get("provenance", {})
    shape = frame.get("shape", [])
    if (provenance.get("workflow_id") != WORKFLOW
            or provenance.get("workflow_code_sha256") != emb.file_hash(__file__)
            or provenance.get("encoder_id") not in emb.ENCODERS
            or provenance.get("population_refilter_performed") is not False
            or len(shape) != 2 or shape[1] != 768):
        raise ValueError("Sampling frame is not bound to this per-encoder workflow")
    return provenance


def selection_pack(records, audit, vectors, cache_manifest, asset_manifest, out,
                   *, seed=20271003, block_size=256):
    if Path(out).exists():
        raise FileExistsError("Output already exists; annotation packs are immutable")
    rows, cache, provenance = load_aligned_cache(records, audit, vectors, cache_manifest, asset_manifest)
    manifest = cal.prepare_selection(cache, rows, provenance, seed=seed, block_size=block_size)
    cal.write_blinded_pack(manifest, rows, out)
    return manifest


def validation_pack(records, audit, vectors, cache_manifest, asset_manifest, lock, out,
                    *, seed=20271004):
    if Path(out).exists():
        raise FileExistsError("Output already exists; annotation packs are immutable")
    rows, cache, provenance = load_aligned_cache(records, audit, vectors, cache_manifest, asset_manifest)
    selection_lock = emb.read_json(lock)
    locked_provenance = verify_workflow_scope(selection_lock)
    if provenance != locked_provenance:
        raise ValueError("Validation encoder, cache or population differs from the frozen selection")
    # Block size belongs to the sealed scoring frame, rather than a new CLI choice.
    block_size = selection_lock["frame"]["scorer"]["block_size"]
    manifest = cal.prepare_validation(cache, rows, selection_lock, seed=seed, block_size=block_size)
    cal.write_blinded_pack(manifest, rows, out)
    return manifest


def threshold_lock(manifest_path, responses_path, out):
    if Path(out).exists():
        raise FileExistsError("Output already exists; threshold locks are immutable")
    manifest = emb.read_json(manifest_path)
    verify_workflow_scope(manifest)
    value = cal.select_threshold(manifest, emb.read_json(responses_path))
    emb.write_json(out, value)
    return value


def quality_report(manifest_path, responses_path, selection_lock_path, out):
    if Path(out).exists():
        raise FileExistsError("Output already exists; quality reports are immutable")
    manifest, lock = emb.read_json(manifest_path), emb.read_json(selection_lock_path)
    verify_workflow_scope(manifest)
    verify_workflow_scope(lock)
    if manifest["frame"] != lock["frame"]:
        raise ValueError("Quality validation and threshold selection use different encoder frames")
    value = cal.quality_gate(manifest, emb.read_json(responses_path), lock)
    emb.write_json(out, value)
    return value


def guard_population(records, audit, vectors, cache_manifest, asset_manifest, lock,
                     quality, out, *, block_size=256):
    """Apply the ONE E5 population guard; MPNet never filters this population."""
    if Path(out).exists():
        raise FileExistsError("Output already exists; guarded populations are immutable")
    crows, cvectors, provenance = load_aligned_cache(records, audit, vectors, cache_manifest, asset_manifest)
    if provenance["encoder_id"] != emb.E5:
        raise ValueError("Population filtering is E5-only; MPNet must use the same resulting population")
    selection_lock, quality_result = emb.read_json(lock), emb.read_json(quality)
    locked_provenance = verify_workflow_scope(selection_lock)
    cal._verify(quality_result)
    if (locked_provenance != provenance
            or quality_result.get("schema") != "ccu-calibration-quality-report-1"
            or quality_result.get("selection_lock_sha256") != selection_lock["sha256"]
            or quality_result.get("threshold") != selection_lock["threshold"]
            or quality_result.get("statistical_and_declared_scope_eligible") is not True):
        raise ValueError("Matching frozen E5 selection and passed declared-scope statistical quality are required")
    frame = cal._check_inputs(cvectors, crows, provenance, selection_lock["frame"]["scorer"]["block_size"])
    if frame != selection_lock["frame"]:
        raise ValueError("Current E5 calibration frame differs from the selected threshold")
    rows, preparation = emb.load_prepared_records(records, audit)
    cache = np.load(vectors, mmap_mode="r", allow_pickle=False)
    retained, indices, semantic = panels.semantic_guard(rows, cache, selection_lock["threshold"],
        block_size=block_size, encoder_id=emb.E5)
    index_list = [int(i) for i in indices]
    if [rows[i]["record_id"] for i in index_list] != [r["record_id"] for r in retained]:
        raise AssertionError("Retained index map and row IDs disagree")
    result = {"schema": "ccu-phase4-guarded-population-1",
        "stage": "semantic_guard_done_external_review_pending", "dataset_id": preparation["dataset_id"],
        "input_preparation_audit_sha256": preparation["sha256"],
        "input_cache_manifest_sha256": provenance["source_cache_manifest_sha256"],
        "selection_lock_sha256": selection_lock["sha256"],
        "quality_report_sha256": quality_result["sha256"], "semantic_guard": semantic,
        "retained_original_row_indices_sha256": cal.digest(index_list),
        "retained_record_ids_sha256": cal.digest([r["record_id"] for r in retained]),
        "guard_code_sha256": emb.file_hash(__file__),
        "cross_encoder_population": "same_retained_original_row_indices_for_E5_and_MPNet_no_MPNet_refilter",
        "panel_selection_pending": True, "external_provenance_verified_by_this_module": False,
        "confirmatory_study_ready": False}
    run_preparation.write_population(out, retained, {}, result)
    emb.write_json(Path(out) / "retained_original_row_indices.json", index_list)
    return emb.read_json(Path(out) / "audit.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("selection", "validation", "guard"):
        target = sub.add_parser(command)
        for name in ("records", "preparation-audit", "vectors", "cache-manifest", "asset-manifest", "out"):
            target.add_argument("--" + name, required=True)
        if command != "guard":
            target.add_argument("--seed", type=int, default=20271003 if command == "selection" else 20271004)
        if command == "selection":
            target.add_argument("--block-size", type=int, default=256)
        else:
            target.add_argument("--lock", required=True)
        if command == "guard":
            target.add_argument("--quality", required=True)
            target.add_argument("--block-size", type=int, default=256)
    for command in ("threshold", "quality"):
        target = sub.add_parser(command)
        for name in ("manifest", "responses", "out"):
            target.add_argument("--" + name, required=True)
        if command == "quality":
            target.add_argument("--lock", required=True)
    args = parser.parse_args()
    try:
        if args.command in {"selection", "validation", "guard"}:
            common = (args.records, args.preparation_audit, args.vectors, args.cache_manifest, args.asset_manifest)
            if args.command == "selection":
                value = selection_pack(*common, args.out, seed=args.seed, block_size=args.block_size)
            elif args.command == "validation":
                value = validation_pack(*common, args.lock, args.out, seed=args.seed)
            else:
                value = guard_population(*common, args.lock, args.quality, args.out, block_size=args.block_size)
        elif args.command == "threshold":
            value = threshold_lock(args.manifest, args.responses, args.out)
        else:
            value = quality_report(args.manifest, args.responses, args.lock, args.out)
        print(__import__("json").dumps({"stage": args.command, "sha256": value["sha256"],
            "output": args.out, "population_refilter_performed": args.command == "guard",
            "confirmatory_study_ready": False}))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.exit(2, "BLOCKED: " + str(error) + "\n")


if __name__ == "__main__":
    main()
