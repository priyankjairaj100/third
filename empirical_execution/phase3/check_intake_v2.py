#!/usr/bin/env python3
"""Regression checks; malformed/algebraic metadata is not an empirical dataset."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys

from intake_v2 import (CATEGORIES, SCHEMA_VERSION, Validator, expected_source,
                       norm_text, response_label_consistent, sha256)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
checks = []


def require(name, passed):
    if not passed:
        raise AssertionError(name)
    checks.append(name)


class MemoryRows(Validator):
    """Inject transient software fixtures, never save them as corpus records."""
    def __init__(self, rows):
        super().__init__(ROOT)
        self._rows = rows

    def jsonl_artifact(self, descriptor, code):
        return self._rows


def row_check(rows, suffix):
    validator = MemoryRows(rows)
    validator.rows({"dataset_id": "civil_comments", "outputs": 1, "rows": {}}, "software")
    matches = [x["passed"] for x in validator.checks if x["check"].endswith(suffix)]
    require("check_exists_" + suffix, len(matches) == 1)
    return matches[0]


def main():
    phase2_path = ROOT / "empirical_execution/phase2/intake_validate.py"
    phase2_hash = sha256(phase2_path)
    require("version_is_two", SCHEMA_VERSION == "ccu-local-intake-2")
    natural = [json.loads(s) for s in (ROOT / "empirical_execution/data/civil_comments_engineering_preview.jsonl").read_text().splitlines()]
    # Unit metadata below is deliberately fictional, exists only in memory and
    # checks parser branches. Text and labels come from actual cached records.
    rows = []
    for i, part in enumerate(("train", "calibration", "test")):
        original = {"id": str(i), "publication_id": "software_fixture", "article_id": i,
                    "parent_id": None, "created_date": "software_fixture_not_a_date",
                    "text": "\t" + natural[i]["text"] + "\n", "toxicity": natural[i]["label"]}
        rows.append({"record_id": str(i), "source_unit_id": "civil:" + json.dumps(["software_fixture", i], separators=(",", ":")),
                     "partition": part, "text": norm_text(original["text"]),
                     "labels": [original["toxicity"]], "original_fields": original})
    require("normalized_original_text_accepted", row_check(rows, ".original_fields_internal_consistency"))
    changed = copy.deepcopy(rows); changed[0]["text"] += " altered"
    require("changed_original_text_rejected", not row_check(changed, ".original_fields_internal_consistency"))
    require("normalization_preserves_original_field", rows[0]["original_fields"]["text"].startswith("\t"))
    # IDs 0/1/2 are fixture IDs. 0 is intentionally a declared parent sentinel.
    for child, parent, expected in ((0, "1", False), (0, "2", False), (1, "2", True), (2, "1", True), (2, "0", True)):
        changed = copy.deepcopy(rows); changed[child]["original_fields"]["parent_id"] = parent
        require(f"parent_direction_child{child}_parent{parent}", row_check(changed, ".parent_partition_guard") is expected)
    # Prove the permitted reverse direction with a nonsentinel training ID.
    reverse = copy.deepcopy(rows); reverse[0]["record_id"] = "train-parent"
    reverse[0]["original_fields"]["id"] = "train-parent"
    reverse[2]["original_fields"]["parent_id"] = "train-parent"
    require("heldout_child_training_parent_allowed", row_check(reverse, ".parent_partition_guard"))
    for sentinel in (None, "", 0, "0", -1, "-1"):
        changed = copy.deepcopy(rows); changed[0]["original_fields"]["parent_id"] = sentinel
        require("parent_sentinel_" + repr(sentinel), row_check(changed, ".parent_partition_guard"))
    for owner in (None, "", " ", 0, "0", -1, "-1", -2, "-2"):
        row = {"record_id": "id", "original_fields": {"site": "askubuntu", "OwnerUserId": owner}}
        require("community_missing_owner_" + repr(owner), expected_source(row, "askubuntu") == ("unknown:id", False))
    for owner in (True, False, float("nan"), [], {}):
        require("malformed_owner_" + repr(owner), expected_source({"record_id": "id", "original_fields": {"site": "askubuntu", "OwnerUserId": owner}}, "askubuntu") == (None, False))
    require("positive_owner_is_native", expected_source({"record_id": "id", "original_fields": {"site": "askubuntu", "OwnerUserId": 7}}, "askubuntu") == ('account:["askubuntu",7]', True))
    for category, target in CATEGORIES.items():
        for label in ("duplicate", "not_duplicate", "unsure"):
            require("category_map_" + category + "_" + label,
                    response_label_consistent({"category": category, "label": label}) is (label == target))
    require("invalid_category_rejected", not response_label_consistent({"category": "invented", "label": "duplicate"}))
    require("absent_optional_category_keeps_legacy_mapping", response_label_consistent({"label": "duplicate"}))
    require("empty_present_category_rejected", not response_label_consistent({"category": "", "label": "duplicate"}))

    # Actual unmodified preview rows must still fail source-rich study intake.
    fixture = ROOT / "empirical_execution/data/civil_comments_engineering_preview.jsonl"
    design = ROOT / "output/empirical_program/study_design.json"
    manifest = {"schema_version": SCHEMA_VERSION, "synthetic_data_allowed": False,
                "study_design": {"path": str(design.relative_to(ROOT)), "sha256": sha256(design)},
                "datasets": [{"dataset_id": "civil_comments", "outputs": 1,
                              "rows": {"path": str(fixture.relative_to(ROOT)), "sha256": sha256(fixture)}}]}
    result = Validator(ROOT).validate(manifest)
    require("unmodified_natural_preview_still_fails_mechanical_intake", not result["mechanical_intake_passed"])
    require("human_authenticity_gate_never_inferred", result["confirmatory_ready"] is False)
    require("phase2_original_file_preserved", sha256(phase2_path) == phase2_hash)
    out = {"passed": True, "checks_count": len(checks), "checks": checks,
           "scope": "software_regression_only; transient fictional metadata fixtures are not empirical datasets",
           "human_responses_generated": False, "confirmatory_ready": False,
           "unmodified_natural_fixture_validation_failed_checks": result["failed_checks"],
           "source_sha256": sha256(HERE / "intake_v2.py"),
           "audit_sha256": sha256(Path(__file__)), "preserved_phase2_sha256": phase2_hash}
    (HERE / "intake_v2_audit.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "checks": len(checks), "human_responses_generated": False}))


if __name__ == "__main__":
    main()
