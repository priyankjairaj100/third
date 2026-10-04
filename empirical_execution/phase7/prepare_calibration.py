#!/usr/bin/env python3
"""Replay local calibration inputs, then create a blank selection pack.

This command does not download files, select a threshold, or supply ratings.
It preserves the frozen Phase 6 acceptance implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "empirical_execution"))
from phase3 import calibration as cal
from phase5 import task_program as task
from phase6 import acceptance

DATASETS = ("civil_comments", "askubuntu", "english_stackexchange", "cc_news")
FIELDS = {"source_acceptance", "records", "features", "provenance"}
DESIGN = ROOT / "output/empirical_program/study_design.json"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(text):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)


def load_dossier(path):
    """Load hash-bound leaves without unbound external paths or reference cycles."""
    path = Path(path).resolve()
    base = path.parent
    inventory = {}

    def visit(value, active=()):
        if isinstance(value, dict) and set(value) == {"path", "sha256", "kind"}:
            if not isinstance(value["path"], str):
                raise ValueError("Descriptor path must be a string")
            target = (base / value["path"]).resolve()
            if not target.is_relative_to(base) or not target.is_file():
                raise ValueError("Descriptor is absent or outside the dossier directory")
            if target in active:
                raise ValueError("Cyclic artifact reference")
            before = file_hash(target)
            if before != value["sha256"]:
                raise ValueError("Bound artifact hash differs")
            inventory[target.relative_to(base).as_posix()] = before
            kind = value["kind"]
            if kind == "json":
                result = visit(strict_json(target.read_text()), (*active, target))
            elif kind == "jsonl":
                result = [strict_json(line) for line in target.read_text().splitlines() if line.strip()]
            elif kind == "npy":
                result = task.load_bound_value(value, base)
            else:
                raise ValueError("Unsupported bound artifact kind")
            if file_hash(target) != before:
                raise ValueError("Artifact changed while loading")
            return result
        if isinstance(value, dict):
            return {key: visit(item, active) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, active) for item in value]
        return value

    original = file_hash(path)
    result = visit(strict_json(path.read_text()), (path,))
    if not isinstance(result, dict) or set(result) != FIELDS:
        raise ValueError("Dossier must contain exactly source_acceptance, records, features, and provenance")
    if file_hash(path) != original:
        raise ValueError("Dossier changed while loading")
    return result, {"dossier_sha256": original, "bound_files": inventory,
                    "loaded_value_sha256": cal.digest(task.fingerprint(result))}


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def code_bindings():
    result = acceptance.code_bindings()
    for path in (Path(__file__), Path(cal.__file__), Path(task.__file__)):
        result[str(path.resolve().relative_to(ROOT / "empirical_execution"))] = file_hash(path)
    return result


def run(dataset, encoder, dossier_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "ccu-calibration-intake-cli-1", "status": "blocked",
              "dataset_id": dataset, "encoder": encoder, "machine_acceptance_passed": False,
              "selection_pack_created": False, "primary_execution_allowed": False,
              "source_authenticity_verified": False, "human_responses_created": 0,
              "human_collection_dispatched": False, "code_bindings": code_bindings(),
              "study_design_sha256": file_hash(DESIGN)}
    stage = "arguments"
    try:
        if dataset not in DATASETS or encoder not in {"e5", "mpnet"}:
            raise ValueError("Registered calibration corpus and encoder required; WCEP inherits News")
        dossier_path = Path(dossier_path).resolve()
        stage = "bound_dossier"
        dossier, before = load_dossier(dossier_path)
        report["input_binding"] = before
        stage = "original_source_and_complete_encoder_replay"
        replay = acceptance.accept_calibration_inputs(
            dataset, encoder, dossier, base_dir=dossier_path.parent,
            design=strict_json(DESIGN.read_text()))
        write_new(out / "acceptance.json", replay)
        report["acceptance_sha256"] = file_hash(out / "acceptance.json")
        if replay.get("machine_acceptance_passed") is not True:
            raise ValueError("Original source and complete encoder replay did not pass")
        stage = "blank_selection_preparation"
        manifest = cal.prepare_selection(dossier["features"], dossier["records"],
                                         dossier["provenance"], seed=20271003, block_size=256)
        stage = "unchanged_inputs_and_code"
        _, after = load_dossier(dossier_path)
        if before != after or code_bindings() != report["code_bindings"]:
            raise ValueError("Inputs or implementation changed during intake")
        if file_hash(DESIGN) != report["study_design_sha256"]:
            raise ValueError("Study design changed during intake")
        stage = "write_blank_pack"
        pack = cal.write_blinded_pack(manifest, dossier["records"], out / "selection")
        report.update(status="blank_selection_pack_ready_external_review_pending",
                      machine_acceptance_passed=True, selection_pack_created=True,
                      selection_manifest_sha256=manifest["sha256"],
                      assignment_count=pack["assignment_count"], pair_count=pack["pair_count"],
                      remaining_requirements=["archive_and_encoder_authenticity_review",
                                              "independent_human_collection",
                                              "threshold_selection_and_fresh_validation"])
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
        report["failure"] = {"stage": stage, "type": type(error).__name__, "reason": str(error)}
        report["partial_output_not_accepted"] = (out / "selection").exists()
    write_new(out / "receipt.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--encoder", required=True, choices=("e5", "mpnet"))
    p.add_argument("--dossier", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    try:
        result = run(args.dataset, args.encoder, args.dossier, args.out)
    except (OSError, ValueError) as error:
        p.exit(2, "BLOCKED: " + str(error) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["selection_pack_created"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
