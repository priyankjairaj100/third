"""Versioned News source/date cohorts and declared-coverage calendar windows.

Coverage means completeness of the pinned extraction, never all publisher/crawl
holdings. Local evidence bindings do not authenticate those declarations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3.panels import digest_partition, hash_integer, source_unit
from phase3 import calibration as cal
from phase3 import panels
from phase4 import adapters
from phase4.adapters import _rows, _dump, _date, file_sha256, verify_adapter_output

SCHEMA = "ccu-news-calendar-1"
MONTHS = [f"{year}-{month:02}" for year in (2017, 2018, 2019) for month in range(1, 13)]
ANALYSIS_MONTHS = [month for month in MONTHS if month >= "2018-01"]


def load_coverage(path, input_sha256):
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get("schema") != "ccu-news-calendar-coverage-1" or value.get("input_file_sha256") != input_sha256:
        raise ValueError("Coverage must bind the exact parsed extraction bytes")
    if value.get("completeness_unit") != "all_records_in_pinned_extraction_for_recorded_month":
        raise ValueError("Coverage unit must describe the pinned extraction, not all web/news")
    evidence = value.get("evidence", [])
    if not evidence:
        raise ValueError("Local acquisition/coverage evidence is required; row counts are not evidence")
    bindings = []
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Malformed coverage evidence")
        filename = Path(item["path"])
        if not filename.is_absolute():
            filename = path.parent / filename
        if file_sha256(filename) != item.get("sha256"):
            raise ValueError("Coverage evidence SHA256 mismatch")
        bindings.append({"name": filename.name, "sha256": item["sha256"]})
    months = value.get("months", {})
    if set(months) != set(MONTHS):
        raise ValueError("All 36 months need explicit complete/incomplete/unavailable statuses")
    for month, item in months.items():
        if (not isinstance(item, dict) or item.get("status") not in {"complete", "incomplete", "unavailable"}
                or not isinstance(item.get("basis"), str) or not item["basis"].strip()):
            raise ValueError("Each month requires explicit status and evidence basis")
    return {"schema": value["schema"], "input_file_sha256": input_sha256,
            "manifest_sha256": file_sha256(path), "months": months,
            "evidence_bindings": bindings, "declarations_authenticated": False}


def calendar_decision(month_counts, coverage, *, target=10000):
    """Select from declared complete months only. No outcome-dependent selection.

    Call only with counts for the final immutable post-guard analysis population.
    This pure metadata decision does not certify that its caller supplied it.
    """
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError("Positive whole-record target required")
    if any(month not in ANALYSIS_MONTHS or isinstance(count, bool) or not isinstance(count, int) or count < 0
           for month, count in month_counts.items()):
        raise ValueError("Counts must be nonnegative integers for 2018–19 calendar months")
    if set(coverage.get("months", {})) != set(MONTHS):
        raise ValueError("Explicit coverage for all 36 months required")
    if any(not isinstance(item, dict) or item.get("status") not in {"complete", "incomplete", "unavailable"}
           for item in coverage["months"].values()):
        raise ValueError("Calendar coverage statuses must be explicit and recognized")
    complete = [m for m in ANALYSIS_MONTHS if coverage["months"][m]["status"] == "complete"]
    qualifying = [m for m in complete if month_counts.get(m, 0) >= target]
    selected, reason, gap = [], None, None
    if qualifying:
        selected = [qualifying[0]]
        reason = "earliest_declared_complete_month_meeting_target"
    else:
        reason = "consecutive_complete_whole_month_prefix_from_2018_01"
        total = 0
        for month in ANALYSIS_MONTHS:
            if coverage["months"][month]["status"] != "complete":
                gap = month
                break
            selected.append(month)
            total += month_counts.get(month, 0)
            if total >= target:
                break
    total = sum(month_counts.get(m, 0) for m in selected)
    return {"schema": SCHEMA, "selection_rule": reason, "selected_months": selected,
            "target": target, "selected_count": total, "target_met": total >= target,
            "shortfall": max(0, target - total), "overshoot": max(0, total - target),
            "prefix_stopped_at_coverage_gap": gap,
            "all_month_counts": {m: month_counts.get(m, 0) for m in ANALYSIS_MONTHS},
            "coverage_statuses": {m: coverage["months"][m]["status"] for m in ANALYSIS_MONTHS},
            "calendar_completeness_authenticated": False,
            "scope": "Conditional on declared extraction coverage and caller-provided post-guard counts"}


def prepare_news_cohorts(adapter_directory, coverage_path, output_dir, *, source_split_salt="ccu-v1-source-split"):
    """Prospective completion: 2017 calibration domains;2018–19 train domains.

    Source assignment is unchanged 70/15/15. Test-domain records and domain/year
    combinations outside those roles are retained only in the exclusion ledger.
    Whole-source panel selection occurs later, within the resulting period.
    """
    adapter_directory = Path(adapter_directory)
    verification = verify_adapter_output(adapter_directory)
    audit = json.loads((adapter_directory / "audit.json").read_text())
    if audit.get("dataset_id") != "cc_news":
        raise ValueError("News adapter output required")
    coverage = load_coverage(coverage_path, audit["inputs"][0]["sha256"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    role_sources = {"calibration": set(), "analysis": set()}
    analysis_months = Counter()
    with (output_dir / "calibration.jsonl").open("x", encoding="utf-8") as calibration, (output_dir / "analysis.jsonl").open("x", encoding="utf-8") as analysis, (output_dir / "excluded_cohort_ids.jsonl").open("x", encoding="utf-8") as excluded:
        for _, row in _rows(adapter_directory / "records.jsonl"):
            date = _date(row["original_fields"]["date"])
            source, kind = source_unit(row, "cc_news")
            if row["source_unit_id"] != source:
                raise ValueError("Adapter source mismatch")
            part = digest_partition(hash_integer(source_split_salt, source))
            counts["input_rows"] += 1
            counts["assigned_partition:" + part] += 1
            role = "calibration" if date.year == 2017 and part == "calibration" else "analysis" if date.year in (2018, 2019) and part == "train" else None
            if role is None:
                reason = "test_domain_reserved" if part == "test" else "outside_source_date_role"
                _dump(excluded, {"record_id": row["record_id"], "source_unit_id": source,
                                 "source_partition": part, "recorded_date": date.isoformat(), "reason": reason})
                counts["excluded:" + reason] += 1
                continue
            row["partition"] = part
            row["news_cohort"] = role
            _dump(calibration if role == "calibration" else analysis, row)
            counts[role + "_rows"] += 1
            counts[role + "_source_kind:" + kind] += 1
            role_sources[role].add(source)
            if role == "analysis":
                analysis_months[date.isoformat()[:7]] += 1
    if role_sources["calibration"] & role_sources["analysis"]:
        raise AssertionError("Source-disjoint cohort invariant violated")
    # Calibration is held out; remove train-side exact normalized copies without
    # keeping the whole text population in RAM. News has no parent/native links.
    text_index = sqlite3.connect(output_dir / "calibration_text_index.sqlite3")
    text_index.execute("PRAGMA cache_size=-2048")
    text_index.execute("CREATE TABLE texts(text TEXT PRIMARY KEY)")
    for _, row in _rows(output_dir / "calibration.jsonl"):
        text_index.execute("INSERT OR IGNORE INTO texts VALUES (?)", (row["text"],))
    text_index.commit()
    full_digest, text_digest = hashlib.sha256(), hashlib.sha256()
    full_digest.update(b"["); text_digest.update(b"[")
    first = True
    analysis_months = Counter()
    with (output_dir / "records.jsonl").open("x", encoding="utf-8") as prepared, (output_dir / "exact_guard_removed_ids.jsonl").open("x") as removed:
        for filename in ("calibration.jsonl", "analysis.jsonl"):
            for _, row in _rows(output_dir / filename):
                if row["partition"] == "train" and text_index.execute("SELECT 1 FROM texts WHERE text=?", (row["text"],)).fetchone():
                    _dump(removed, {"record_id": row["record_id"], "reason": "crosssplit_normalized_exact_text"})
                    counts["exact_guard_removed_training_rows"] += 1
                    continue
                _dump(prepared, row)
                if not first:
                    full_digest.update(b","); text_digest.update(b",")
                first = False
                full_digest.update(cal._canonical(row))
                text_digest.update(cal._canonical({k: row[k] for k in ("record_id", "text")}))
                counts["post_fixed_guard:" + row["partition"]] += 1
                if row["partition"] == "train":
                    analysis_months[_date(row["original_fields"]["date"]).isoformat()[:7]] += 1
    full_digest.update(b"]"); text_digest.update(b"]")
    text_index.close()
    decision = calendar_decision(dict(analysis_months), coverage)
    decision["status"] = "provisional_after_fixed_guard_before_semantic_guard_not_locked_window"
    result = {"schema": "ccu-news-preparation-1", "dataset_id": "cc_news", "stage": "fixed_guards_done_semantic_guard_pending",
              "counts": dict(sorted(counts.items())), "cohort_source_counts": {k: len(v) for k,v in role_sources.items()},
              "cross_cohort_source_overlap": 0, "adapter_verification": verification,
              "adapter_audit_sha256": file_sha256(adapter_directory / "audit.json"),
              "source_split_salt": source_split_salt, "coverage": coverage,
              "all_2017_months_declared_complete": all(coverage["months"][m]["status"] == "complete" for m in MONTHS[:12]),
              "provisional_calendar_counts": decision,
              "news_preparation_code_sha256": file_sha256(__file__),
              "adapter_code_sha256": file_sha256(adapters.__file__),
              "source_and_scoring_code_sha256": file_sha256(panels.__file__),
              "prepared_records_file_sha256": file_sha256(output_dir / "records.jsonl"),
              "prepared_rows_sha256": full_digest.hexdigest(),
              "normalized_records_sha256": text_digest.hexdigest(),
              "fixed_guard": "Remove only train-side normalized exact copies of calibration; no native News parent/duplicate-link metadata; test domains excluded by cohort policy",
              "prospective_completion": "Existing 70/15/15 source hash; calibration sources in2017; training sources in2018–19; test sources reserved. Calendar selection uses final guarded counts; prefix never crosses a coverage gap.",
              "confirmatory_study_ready": False}
    result["outputs"] = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in output_dir.iterdir() if p.is_file()}
    result = cal._seal(result)
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def select_post_guard_window(records_path, coverage_path, input_sha256, post_guard_audit_path, quality_report_path, output_dir, *, target=10000, source_split_salt="ccu-v1-source-split"):
    """Lock calendar membership after a versioned upstream population audit.

    Consume the News-compatible guard CLI's sealed audit and passed quality report.
    This interface binds upstream consistency evidence; it cannot establish its
    underlying authenticity or independence of human judgments.
    """
    records_path = Path(records_path)
    binding = json.loads(Path(post_guard_audit_path).read_text())
    quality = json.loads(Path(quality_report_path).read_text())
    cal._verify(binding)
    cal._verify(quality)
    digest = file_sha256(records_path)
    if (binding.get("stage") != "semantic_guard_done_external_review_pending" or binding.get("prepared_records_file_sha256") != digest
            or binding.get("quality_report_sha256") != quality.get("sha256")
            or quality.get("schema") != "ccu-calibration-quality-report-1"
            or not quality.get("statistical_and_declared_scope_eligible")):
        raise ValueError("Post-guard records and quality-lock bindings required")
    coverage = load_coverage(coverage_path, input_sha256)
    months = Counter()
    for _, row in _rows(records_path):
        if row.get("news_cohort") == "calibration" and row.get("partition") == "calibration":
            continue
        date = _date(row["original_fields"]["date"])
        source, _ = source_unit(row, "cc_news")
        if date.year not in (2018, 2019) or row.get("partition") != "train" or row.get("news_cohort") != "analysis" or row.get("source_unit_id") != source:
            raise ValueError("Only final analysis-cohort records may enter the window")
        if digest_partition(hash_integer(source_split_salt, source)) != "train":
            raise ValueError("Analysis source belongs to a heldout source partition")
        months[date.isoformat()[:7]] += 1
    decision = calendar_decision(dict(months), coverage, target=target)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "window_record_ids.jsonl").open("x") as output:
        for _, row in _rows(records_path):
            if row.get("news_cohort") == "analysis" and _date(row["original_fields"]["date"]).isoformat()[:7] in decision["selected_months"]:
                _dump(output, row["record_id"])
    decision.update({"status": "conditional_calendar_membership_bound_to_upstream_audit", "records_sha256": digest,
                     "post_guard_audit_sha256": file_sha256(post_guard_audit_path),
                     "coverage_manifest_sha256": coverage["manifest_sha256"],
                     "window_ids_sha256": file_sha256(directory / "window_record_ids.jsonl"),
                     "quality_report_sha256": quality["sha256"], "source_split_salt": source_split_salt,
                     "upstream_quality_and_provenance_verified_by_this_module": False,
                     "confirmatory_study_ready": False, "code_sha256": file_sha256(__file__)})
    (directory / "audit.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_directory")
    parser.add_argument("coverage_path")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = prepare_news_cohorts(args.adapter_directory, args.coverage_path, args.output_dir)
    print(json.dumps({"counts": result["counts"], "confirmatory_study_ready": False}, indent=2))


if __name__ == "__main__":
    main()
