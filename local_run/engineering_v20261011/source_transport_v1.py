#!/usr/bin/env python3
"""Make a lossless ASCII JSONL engineering view for the frozen splitlines reader.

Original preparation files, audits, and frozen code remain unchanged. Every
original full-file and complete-row binding must pass before a view is written.
This repairs serialization interoperability only; it grants no primary input,
source-origin, semantic-cache, human-calibration, or study acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback


DATASETS = ("askubuntu", "cc_news")
DELIMITERS = ("\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


class ListDigest:
    def __init__(self, canonical):
        self.canonical = canonical
        self.h = hashlib.sha256(b"[")
        self.count = 0

    def add(self, value):
        if self.count:
            self.h.update(b",")
        self.h.update(self.canonical(value))
        self.count += 1

    def finish(self):
        h = self.h.copy()
        h.update(b"]")
        return h.hexdigest()


def source_scan(path, cal, *, destination=None):
    """Binary iteration splits on literal LF only; preserves all JSON strings."""
    raw_digest = hashlib.sha256()
    full, texts, ids = (ListDigest(cal._canonical) for _ in range(3))
    delimiters, partitions = Counter(), Counter()
    reproduction = None
    with Path(path).open("rb") as stream:
        for line_number, physical_bytes in enumerate(stream, 1):
            raw_digest.update(physical_bytes)
            require(physical_bytes.strip(), f"Blank JSONL physical line {line_number}")
            physical_text = physical_bytes.decode("utf-8")
            row = json.loads(physical_text)
            require(isinstance(row, dict), f"Non-object JSONL physical line {line_number}")
            for delimiter in DELIMITERS:
                count = physical_text.count(delimiter)
                if count:
                    delimiters[f"U+{ord(delimiter):04X}"] += count
            if reproduction is None and any(d in physical_text for d in DELIMITERS):
                try:
                    # Isolated reproduction of the frozen reader on one valid
                    # original physical line. Never include any source text.
                    [json.loads(part) for part in physical_text.splitlines() if part.strip()]
                except json.JSONDecodeError as error:
                    reproduction = {"physical_line": line_number,
                                    "original_physical_line_sha256": hashlib.sha256(physical_bytes).hexdigest(),
                                    "exception_type": type(error).__name__, "decoder_message": error.msg,
                                    "column": error.colno, "position": error.pos,
                                    "source_text_disclosed": False}
            full.add(row)
            texts.add({k: row[k] for k in ("record_id", "text")})
            ids.add(row["record_id"])
            partitions[str(row.get("partition"))] += 1
            if destination is not None:
                destination.write(json.dumps(row, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")
    return {"raw_file_sha256": raw_digest.hexdigest(), "rows": full.count,
            "prepared_rows_sha256": full.finish(), "normalized_records_sha256": texts.finish(),
            "ordered_record_ids_sha256": ids.finish(), "partition_counts": dict(sorted(partitions.items())),
            "literal_non_lf_splitlines_delimiter_counts": dict(sorted(delimiters.items())),
            "frozen_reader_error_reproduction": reproduction}


def process_source(repo, workspace, out, dataset, e, cal, progress):
    start = time.perf_counter()
    original_dir = workspace / "prepared" / dataset / "a02/population"
    records_path, audit_path = original_dir / "records.jsonl", original_dir / "audit.json"
    progress.update({"dataset": dataset, "stage": "original_full_source_verification"})
    audit_binding = binding(audit_path)
    audit = json.loads(audit_path.read_bytes())
    cal._verify(audit)
    require(audit.get("dataset_id") == dataset, "Original dataset ID mismatch")
    require(audit.get("stage") == "fixed_guards_done_semantic_guard_pending", "Original preparation stage mismatch")
    require(audit.get("source_and_scoring_code_sha256") == file_hash(repo / "empirical_execution/phase3/panels.py"),
            "Original frozen scoring code binding mismatch")
    if dataset == "cc_news":
        require(audit.get("schema") == "ccu-news-preparation-1", "Unexpected original News audit schema")
        for key, rel in (("news_preparation_code_sha256", "empirical_execution/phase4/news_calendar.py"),
                         ("adapter_code_sha256", "empirical_execution/phase4/adapters.py")):
            require(audit.get(key) == file_hash(repo / rel), f"Original {key} mismatch")
        require(audit.get("source_split_salt") == "ccu-v1-source-split", "Original News source salt mismatch")
    else:
        require(audit.get("preparation_code_sha256") == file_hash(repo / "empirical_execution/phase3/run_preparation.py"),
                "Original preparation code binding mismatch")
    original = source_scan(records_path, cal)
    require(original["rows"] > 0, "Empty original prepared population")
    require(original["raw_file_sha256"] == audit.get("prepared_records_file_sha256"),
            "Original raw file does not match its preparation audit")
    require(original["prepared_rows_sha256"] == audit.get("prepared_rows_sha256"),
            "Original complete canonical row digest mismatch")
    if dataset == "cc_news":
        require(original["normalized_records_sha256"] == audit.get("normalized_records_sha256"),
                "Original News complete normalized-record digest mismatch")
    if "text_binding" in audit:
        require(original["normalized_records_sha256"] == audit["text_binding"],
                "Original complete text binding mismatch")
    require(binding(audit_path) == audit_binding, "Original audit changed during verification")
    source_binding = {"path": str(records_path), "bytes": records_path.stat().st_size,
                      "sha256": original["raw_file_sha256"]}

    # No transport data or derived audit is written before all original seals,
    # raw bytes, canonical rows and available text/code bindings verify above.
    progress["stage"] = "lossless_transport_derivation"
    target = out / dataset
    target.mkdir(exist_ok=False)
    write_json(target / "original_validation.json", {"original_records": source_binding,
        "original_audit_file": audit_binding, "original_audit_seal_sha256": audit["sha256"], **original})
    with (target / "records.jsonl").open("xb") as stream:
        repeated_original = source_scan(records_path, cal, destination=stream)
    require(repeated_original == original, "Original source changed while deriving transport view")
    derived_binding = binding(target / "records.jsonl")
    # The transport view has a distinct audit seal. Logical population, source,
    # partition, guard, date, and frozen code bindings are copied unchanged.
    derived = {k: v for k, v in audit.items() if k != "sha256"}
    derived["prepared_records_file_sha256"] = derived_binding["sha256"]
    lineage = {"schema": "third-lossless-jsonl-transport-v1",
               "evidence_role": "engineering_transport_view_only",
               "original_preparation_directory": str(original_dir),
               "original_records": source_binding, "original_audit_file": audit_binding,
               "original_audit_seal_sha256": audit["sha256"], "converter": binding(__file__),
               "operation": "LF-only JSON decode then ensure_ascii=True JSON encode; no object changes",
               "original_full_bindings_verified_before_derivation": True,
               "prepared_objects_or_order_changed": False, "primary_input_acceptance": False,
               "full_population_cache_acceptance": False, "confirmatory_study_ready": False}
    if isinstance(derived.get("outputs"), dict):
        # Other output files still belong to the original preparation directory.
        # They are never silently claimed to have been copied beside this view.
        lineage["original_output_bindings"] = derived["outputs"]
        derived["outputs"] = {"records.jsonl": {"sha256": derived_binding["sha256"],
                                                  "bytes": derived_binding["bytes"]}}
    derived["transport_lineage"] = lineage
    derived = cal._seal(derived)
    write_json(target / "audit.json", derived)
    progress["stage"] = "unchanged_frozen_loader_validation"
    decoded, accepted_audit = e.load_prepared_records(target / "records.jsonl", target / "audit.json")
    require(accepted_audit == derived, "Frozen loader audit mismatch")
    require(len(decoded) == original["rows"], "Transport row count mismatch")
    require(cal.digest(decoded) == original["prepared_rows_sha256"], "Transport full canonical object digest mismatch")
    decoded_ids, decoded_texts = ListDigest(cal._canonical), ListDigest(cal._canonical)
    original_replay_raw = hashlib.sha256()
    with records_path.open("rb") as stream:
        for original_bytes, view_row in zip(stream, decoded, strict=True):
            original_replay_raw.update(original_bytes)
            original_row = json.loads(original_bytes.decode("utf-8"))
            require(original_row == view_row, "Decoded full object differs at an original row position")
            decoded_ids.add(view_row["record_id"])
            decoded_texts.add({k: view_row[k] for k in ("record_id", "text")})
    require(original_replay_raw.hexdigest() == original["raw_file_sha256"], "Original raw bytes changed during replay")
    require(decoded_ids.finish() == original["ordered_record_ids_sha256"], "Transport record order changed")
    require(decoded_texts.finish() == original["normalized_records_sha256"], "Transport record text changed")
    del decoded, accepted_audit, original_row, view_row
    gc.collect()
    require(binding(audit_path) == audit_binding, "Original audit changed during transport validation")
    require(binding(target / "records.jsonl") == derived_binding, "Derived file changed during validation")
    receipt = {"schema": "third-lossless-jsonl-transport-receipt-v1", "passed": True,
               "dataset": dataset, "evidence_role": "engineering_transport_view_only",
               "original_records": source_binding, "original_audit_file": audit_binding,
               "original_audit_seal_sha256": audit["sha256"], "original_validation": original,
               "derived_records": derived_binding, "derived_audit_file": binding(target / "audit.json"),
               "derived_audit_seal_sha256": derived["sha256"], "converter": binding(__file__),
               "complete_decoded_row_equality_and_order_verified": True,
               "unchanged_frozen_load_prepared_records_passed": True,
               "original_preparation_files_modified": False, "primary_input_acceptance": False,
               "external_source_origin_authenticated": False, "full_population_cache_acceptance": False,
               "confirmatory_study_ready": False, "seconds": time.perf_counter() - start}
    write_json(target / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "workspace", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    repo, workspace, out = (Path(getattr(args, key)).resolve() for key in ("repo", "workspace", "out"))
    out.mkdir(parents=True, exist_ok=False)
    (out / ".gitignore").write_text("*\n", encoding="utf-8")
    progress = {"stage": "startup"}
    receipts = []
    started = time.perf_counter()
    try:
        require(platform.system() == "Linux" and bool(os.environ.get("SLURM_JOB_ID")),
                "Run this complete-source workload in a Linux Slurm allocation")
        sys.path.insert(0, str(repo / "empirical_execution"))
        from phase3 import calibration as cal
        from phase4 import embeddings as e
        initial_code = {rel: binding(repo / rel) for rel in (
            "empirical_execution/phase3/run_preparation.py", "empirical_execution/phase3/calibration.py",
            "empirical_execution/phase3/panels.py", "empirical_execution/phase4/embeddings.py",
            "empirical_execution/phase4/news_calendar.py", "empirical_execution/phase4/adapters.py")}
        initial_code["converter"] = binding(__file__)
        for dataset in DATASETS:
            receipts.append(process_source(repo, workspace, out, dataset, e, cal, progress))
        require(any(r["original_validation"]["frozen_reader_error_reproduction"] for r in receipts),
                "No original splitlines JSON error reproduced; do not claim this diagnosed failure")
        for item in initial_code.values():
            require(binding(item["path"]) == item, "Frozen code or converter changed during transport")
        result = {"schema": "third-lossless-jsonl-transport-result-v1", "passed": True,
                  "status": "completed", "evidence_role": "engineering_transport_view_only",
                  "datasets": [{"dataset": r["dataset"], "rows": r["original_validation"]["rows"],
                                "records": r["derived_records"], "audit": r["derived_audit_file"],
                                "receipt": binding(out / r["dataset"] / "receipt.json")} for r in receipts],
                  "code_bindings": initial_code, "original_files_modified": False,
                  "primary_input_acceptance": False, "full_population_cache_acceptance": False,
                  "confirmatory_study_ready": False, "seconds": time.perf_counter() - started,
                  "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": platform.node()}
        exit_code = 0
    except Exception as error:
        result = {"schema": "third-lossless-jsonl-transport-result-v1", "passed": False,
                  "status": "failed", "evidence_role": "engineering_transport_view_only", **progress,
                  "exception_type": type(error).__name__, "message": str(error),
                  "completed_dataset_receipts": [r["dataset"] for r in receipts],
                  "primary_input_acceptance": False, "full_population_cache_acceptance": False,
                  "confirmatory_study_ready": False, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        with (out / "traceback.private.txt").open("x", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        write_json(out / "FAILED.json", result)
        exit_code = 1
    write_json(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "code_bindings"}, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
