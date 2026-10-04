"""Local-only population and calibration CLI. Never downloads or supplies ratings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "empirical_execution"))
from phase3 import calibration as cal
from phase3 import panels


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def text_binding(rows):
    return cal.digest([{k: row[k] for k in ("record_id", "text")} for row in rows])


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_population(directory, rows, ids, audit):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "records.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_new_json(directory / "panel_ids.json", ids)
    audit["prepared_records_file_sha256"] = file_hash(directory / "records.jsonl")
    audit["prepared_rows_sha256"] = cal.digest(rows)
    audit["preparation_code_sha256"] = file_hash(__file__)
    audit["source_and_scoring_code_sha256"] = file_hash(panels.__file__)
    write_new_json(directory / "audit.json", cal._seal(audit))


def load_cache(args):
    """Require the complete, exact row order; no hidden reorder or dtype cast."""
    rows = read_rows(args.records)
    manifest = read_json(args.cache_manifest)
    preparation = read_json(args.preparation_audit)
    cal._verify(preparation)
    if (preparation.get("stage") != "fixed_guards_done_semantic_guard_pending"
            or preparation.get("prepared_records_file_sha256") != file_hash(args.records)
            or preparation.get("prepared_rows_sha256") != cal.digest(rows)
            or preparation.get("preparation_code_sha256") != file_hash(__file__)
            or preparation.get("source_and_scoring_code_sha256") != file_hash(panels.__file__)):
        raise ValueError("Records do not match the sealed population preparation audit")
    if manifest.get("prepared_rows_sha256") != cal.digest(rows):
        raise ValueError("Cache source/partition/full-record binding mismatch")
    ids = [row["record_id"] for row in rows]
    if len(set(ids)) != len(ids) or manifest.get("row_ids") != ids:
        raise ValueError("Cache row_ids must equal the complete unique records order")
    if manifest.get("normalized_records_sha256") != text_binding(rows):
        raise ValueError("Cache text/ID binding mismatch")
    if any(row["text"] != panels.normalized_text(row["text"]) for row in rows):
        raise ValueError("Prepared texts must use protocol normalization")
    if manifest.get("cache_file_sha256") != file_hash(args.vectors):
        raise ValueError("Cache file checksum mismatch")
    vectors = np.load(args.vectors, mmap_mode="r", allow_pickle=False)
    if vectors.dtype != np.dtype("float32") or vectors.ndim != 2 or len(vectors) != len(rows):
        raise ValueError("Expected aligned N-by-D FP32 .npy cache")
    provenance = manifest["provenance"]
    role = provenance.get("evidence_role")
    if role == "confirmatory_calibration" and provenance.get("encoder_id") != panels.E5:
        raise ValueError("This primary workflow requires the pinned E5 encoder")
    if any(row.get("partition") not in panels.PARTITIONS for row in rows):
        raise ValueError("Prepared partitions are required")
    sources = {}
    for row in rows:
        source = row.get("source_unit_id")
        if not isinstance(source, str) or not source:
            raise ValueError("Prepared source IDs are required")
        if source in sources and sources[source] != row["partition"]:
            raise ValueError("Source group spans multiple partitions")
        sources[source] = row["partition"]
    indices = [i for i, row in enumerate(rows) if row["partition"] == "calibration"]
    # Explicit O(N_cal*D) FP32 gather; bounded pair scoring does not erase this cost.
    calibration_vectors = vectors[indices]
    calibration_rows = [rows[i] for i in indices]
    if len(calibration_rows) < 2:
        raise ValueError("At least two calibration records are required")
    return rows, vectors, calibration_rows, calibration_vectors, provenance


def execute(args):
    if Path(args.out).exists():
        raise ValueError("Output already exists; frozen artifacts are never overwritten")
    if args.command == "prepare":
        rows, ids, audit = panels.prepare_population(
            read_rows(args.records), args.dataset, read_json(args.design),
            duplicate_links=read_json(args.duplicate_links) if args.duplicate_links else (),
            target=args.target, allow_preview_sources=args.engineering_preview)
        audit["input_records_file_sha256"] = file_hash(args.records)
        audit["study_design_sha256"] = file_hash(args.design)
        audit["text_binding"] = text_binding(rows)
        audit["stage"] = "fixed_guards_done_semantic_guard_pending"
        write_population(args.out, rows, ids, audit)
        return {"status": "prepared_pending_semantic_calibration", "rows": len(rows)}
    if args.command in ("selection", "validation", "guard"):
        rows, vectors, crows, cvectors, provenance = load_cache(args)
        if args.command == "selection":
            manifest = cal.prepare_selection(cvectors, crows, provenance,
                seed=args.seed, block_size=args.block_size)
            cal.write_blinded_pack(manifest, crows, args.out)
            return {"status": "blank_selection_pack_created", "pairs": len(manifest["pairs"]),
                    "completed_human_ratings": 0, "confirmatory_study_ready": False}
        lock = read_json(args.lock)
        cal._verify(lock)
        if lock.get("schema") != "ccu-calibration-selection-lock-1":
            raise ValueError("Expected a frozen selection lock")
        block_size = lock["frame"]["scorer"]["block_size"]
        frame = cal._check_inputs(cvectors, crows, provenance, block_size)
        if frame != lock["frame"]:
            raise ValueError("Cache/scoring/calibration population differs from the selection lock")
        if args.command == "validation":
            manifest = cal.prepare_validation(cvectors, crows, lock, seed=args.seed,
                block_size=block_size)
            cal.write_blinded_pack(manifest, crows, args.out)
            return {"status": "blank_independent_validation_pack_created",
                    "pairs": len(manifest["pairs"]), "completed_human_ratings": 0}
        quality = read_json(args.quality)
        cal._verify(quality)
        if (quality.get("schema") != "ccu-calibration-quality-report-1"
                or quality.get("selection_lock_sha256") != lock["sha256"]
                or quality.get("threshold") != lock["threshold"]
                or not quality.get("statistical_and_declared_scope_eligible")):
            raise ValueError("Primary guard requires the matching passed statistical/scope report")
        if provenance["encoder_id"] != panels.E5:
            raise ValueError("Primary E5 leakage guard required")
        retained, indices, audit = panels.semantic_guard(rows, vectors, lock["threshold"],
            block_size=block_size, encoder_id=panels.E5)
        ids, panel_audit = panels.select_panels(retained, read_json(args.design), target=args.target)
        audit.update({"retained_input_indices": indices, "panels": panel_audit,
                      "quality_report_sha256": quality["sha256"],
                      "cache_file_sha256": file_hash(args.vectors),
                      "stage": "semantic_guard_done_external_review_pending",
                      "confirmatory_study_ready": False,
                      "external_provenance_verified_by_this_module": False,
                      "calibration_FP32_gather_bytes": int(cvectors.nbytes),
                      "missing_required_partitions": [p for p in panels.PARTITIONS
                          if not any(row["partition"] == p for row in retained)],
                      "remaining_gates": ["original_source_and_link_completeness_review",
                          "authentic_embedding_derivation", "authentic_independent_human_collection",
                          "complete_pre_execution_lock_and_required_panel_sizes"]})
        write_population(args.out, retained, ids, audit)
        return {"status": "guarded_population_external_review_pending", "rows": len(retained),
                "confirmatory_study_ready": False}
    if args.command == "threshold":
        result = cal.select_threshold(read_json(args.manifest), read_json(args.responses))
    else:
        result = cal.quality_gate(read_json(args.manifest), read_json(args.responses), read_json(args.lock))
    write_new_json(args.out, result)
    return {"status": "analysis_written", "schema": result["schema"],
            "confirmatory_study_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    default_design = str(ROOT / "output/empirical_program/study_design.json")
    for command in ("prepare", "selection", "threshold", "validation", "quality", "guard"):
        p = sub.add_parser(command)
        p.add_argument("--out", required=True)
        if command in ("prepare", "selection", "validation", "guard"):
            p.add_argument("--records", required=True)
        if command in ("selection", "validation", "guard"):
            p.add_argument("--vectors", required=True)
            p.add_argument("--cache-manifest", required=True)
            p.add_argument("--preparation-audit", required=True)
        if command in ("selection", "validation"):
            p.add_argument("--seed", type=int, default=20271003 if command == "selection" else 20271004)
        if command == "selection":
            p.add_argument("--block-size", type=int, default=256)
        if command in ("prepare", "guard"):
            p.add_argument("--design", default=default_design)
            p.add_argument("--target", type=int, default=10000)
        if command == "prepare":
            p.add_argument("--dataset", required=True, choices=["civil_comments", "askubuntu", "english_stackexchange", "cc_news"])
            p.add_argument("--duplicate-links")
            p.add_argument("--engineering-preview", action="store_true")
        if command in ("threshold", "quality"):
            p.add_argument("--manifest", required=True)
            p.add_argument("--responses", required=True)
        if command in ("validation", "quality", "guard"):
            p.add_argument("--lock", required=True)
        if command == "guard":
            p.add_argument("--quality", required=True)
    try:
        print(json.dumps(execute(parser.parse_args()), indent=2))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error),
                          "confirmatory_study_ready": False}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
