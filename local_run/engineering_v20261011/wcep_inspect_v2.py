#!/usr/bin/env python3
"""Inspect preserved WCEP test bytes without altering them or inventing fields.

Transfer identity, full gzip/JSON integrity, and frozen-parser compatibility
are separate findings. No downloads, adapters, article text, or IDs are emitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re

EXPECTED_ID = "1qsd5pOCpeSXsaqNobXCrcAzhcjtG1wA1"
EXPECTED_BYTES = 51528345
EXPECTED_SHA256 = "aa931187172ea63ae1e3f98c94a44a02e1dbcc36b76e9e140bbfde7f4391c412"
LINE_CAP = 64 * 1024 ** 2
EXPANDED_CAP = 64 * 1024 ** 3
EVENT_CAP = 1000000


class InspectionError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def typename(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    return {str: "string", int: "integer", float: "number", dict: "object", list: "array"}.get(type(value), "other")


def keyname(key):
    # Field inventories contain schema names only, never arbitrary long keys.
    return key if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key) else "<nonstandard_key>"


def bad_constant(_):
    raise ValueError("nonfinite_JSON_number")


def verify_inputs(archive, receipt, receipt_hash=None):
    observed_receipt_hash = sha(receipt)
    if receipt_hash is not None and observed_receipt_hash != receipt_hash:
        raise InspectionError("Original download receipt changed during inspection")
    r = json.loads(receipt.read_text(encoding="utf-8"))
    required = {"split": "test", "file": "test.jsonl.gz", "source_id": EXPECTED_ID,
                "author_url": "https://drive.google.com/uc?id=" + EXPECTED_ID,
                "status": "downloaded_pending_validation",
                "compressed_bytes": EXPECTED_BYTES, "compressed_sha256": EXPECTED_SHA256}
    if any(r.get(k) != v for k, v in required.items()):
        raise InspectionError("Receipt does not match the explicitly observed test download binding")
    if archive.stat().st_size != EXPECTED_BYTES or sha(archive) != EXPECTED_SHA256:
        raise InspectionError("Compressed archive differs from the exact observed receipt binding")
    return observed_receipt_hash


def inspect(archive):
    counts = Counter()
    issues = Counter()
    event_keys = Counter()
    article_keys = Counter()
    event_types = defaultdict(Counter)
    article_types = defaultdict(Counter)
    collection_counts = Counter()
    first_issue_lines = []
    native_ids = set()
    expanded = hashlib.sha256()
    sizes = Counter()
    gzip_complete = False
    fatal = None

    def issue(reason, line):
        issues[reason] += 1
        if len(first_issue_lines) < 20:
            first_issue_lines.append({"physical_line": line, "reason": reason})

    try:
        with gzip.open(archive, "rb") as stream:
            while True:
                line = stream.readline(LINE_CAP + 1)
                if not line:
                    gzip_complete = True
                    break
                counts["physical_lines"] += 1
                ln = counts["physical_lines"]
                counts["uncompressed_bytes_read"] += len(line)
                if len(line) > LINE_CAP or counts["uncompressed_bytes_read"] > EXPANDED_CAP:
                    raise InspectionError("Expanded archive/line bound exceeded; inspection incomplete")
                expanded.update(line)
                if not line.strip():
                    counts["blank_lines"] += 1
                    continue
                counts["nonblank_lines"] += 1
                try:
                    value = json.loads(line.decode("utf-8", "strict"), parse_constant=bad_constant)
                except UnicodeDecodeError:
                    issue("invalid_utf8", ln)
                    continue
                except (ValueError, RecursionError):
                    issue("invalid_json", ln)
                    continue
                counts["valid_json_lines"] += 1
                if not isinstance(value, dict):
                    issue("event_not_object", ln)
                    continue
                counts["event_objects"] += 1
                if counts["event_objects"] > EVENT_CAP:
                    raise InspectionError("Event-count bound exceeded; inspection incomplete")
                for key, item in value.items():
                    safe_key = keyname(key)
                    event_keys[safe_key] += 1
                    event_types[safe_key][typename(item)] += 1
                for key in ("id", "date", "collection", "articles"):
                    if key not in value:
                        issue("event_missing_" + key, ln)
                eid = value.get("id")
                if isinstance(eid, bool) or not isinstance(eid, (str, int)) or not str(eid).strip() or "\0" in str(eid):
                    issue("event_invalid_native_id", ln)
                elif str(eid) in native_ids:
                    issue("duplicate_native_event_id", ln)
                else:
                    native_ids.add(str(eid))
                if not isinstance(value.get("date"), str):
                    issue("event_date_not_string", ln)
                else:
                    try:
                        date.fromisoformat(value["date"])
                    except ValueError:
                        issue("event_date_not_iso_date", ln)
                if "collection" not in value:
                    collection_counts["absent"] += 1
                elif not isinstance(value["collection"], str):
                    collection_counts["nonstring"] += 1
                    issue("event_collection_not_string", ln)
                elif value["collection"] == "test":
                    collection_counts["equals_author_filename_split"] += 1
                else:
                    collection_counts["different_from_author_filename_split"] += 1
                    issue("event_collection_mismatches_expected_split", ln)
                articles = value.get("articles")
                if not isinstance(articles, list):
                    issue("event_articles_not_array", ln)
                    continue
                counts["events_with_article_array"] += 1
                sizes[len(articles)] += 1
                for article in articles:
                    counts["article_instances"] += 1
                    if not isinstance(article, dict):
                        issue("article_not_object", ln)
                        continue
                    counts["article_objects"] += 1
                    for key, item in article.items():
                        safe_key = keyname(key)
                        article_keys[safe_key] += 1
                        article_types[safe_key][typename(item)] += 1
                    for key in ("title", "text"):
                        if key not in article:
                            issue("article_missing_" + key, ln)
                        if not isinstance(article.get(key), str):
                            issue("article_" + key + "_not_string", ln)
                    counts["articles_with_nonempty_url"] += isinstance(article.get("url"), str) and bool(article["url"].strip())
    except Exception as exc:
        fatal = {"error_type": type(exc).__name__,
                 "reason": str(exc) if isinstance(exc, InspectionError) else "Stream inspection failed; source values omitted"}

    integrity = gzip_complete and not issues["invalid_utf8"] and not issues["invalid_json"]
    compatible = integrity and bool(counts["event_objects"]) and not sum(issues.values())
    return {"schema": "wcep-preserved-archive-inspection-v2", "completed_utc": now(),
            "gzip_crc_complete": gzip_complete, "all_rows_inspected": gzip_complete,
            "utf8_and_json_integrity_passed": integrity,
            "frozen_parser_compatible": compatible, "counts": dict(counts),
            "schema_issue_counts": dict(issues), "first_20_schema_issue_locations": first_issue_lines,
            "event_field_presence_counts": dict(event_keys),
            "article_field_presence_counts": dict(article_keys),
            "event_field_type_counts": {k: dict(v) for k, v in event_types.items()},
            "article_field_type_counts": {k: dict(v) for k, v in article_types.items()},
            "native_collection_observation_counts": dict(collection_counts),
            "article_count_per_event_histogram": dict(sizes),
            "uncompressed_sha256_or_complete_read_prefix": expanded.hexdigest(),
            "uncompressed_digest_covers_complete_stream": gzip_complete, "fatal_error": fatal,
            "article_text_or_native_ids_in_report": False, "fields_added_or_repaired": False,
            "adapter_executed": False, "input_authenticity_verified": False,
            "complete_author_release_verified": False, "confirmatory_study_ready": False,
            "scope": "Integrity and schema census of exact currently acquired author-linked test bytes only; no historical release identity or study acceptance"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    archive, receipt, out = args.archive.resolve(), args.receipt.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chmod(out, 0o700)
    (out / ".gitignore").write_text("*\n", encoding="ascii")
    save(out / "inspection_manifest.json", {"schema": "wcep-inspection-request-v2", "started_utc": now(),
        "archive_path": str(archive), "original_receipt_path": str(receipt),
        "expected_compressed_bytes": EXPECTED_BYTES, "expected_compressed_sha256": EXPECTED_SHA256,
        "expected_source_id": EXPECTED_ID, "driver_sha256": sha(Path(__file__)),
        "limits": {"line_bytes": LINE_CAP, "expanded_bytes": EXPANDED_CAP, "event_count": EVENT_CAP},
        "no_download_or_adapter": True, "confirmatory_study_ready": False})
    os.chmod(out / "inspection_manifest.json", 0o400)
    try:
        receipt_hash = verify_inputs(archive, receipt)
        result = inspect(archive)
        # Both original files are rebound after even an incomplete/failed inspection.
        verify_inputs(archive, receipt, receipt_hash)
        result["input_binding_verified_before_and_after"] = True
        result["original_receipt_sha256"] = receipt_hash
        result["compressed_sha256"] = EXPECTED_SHA256
        result["compressed_bytes"] = EXPECTED_BYTES
        save(out / "inspection_result.json", result)
        print(json.dumps({k: result[k] for k in ("gzip_crc_complete", "all_rows_inspected",
              "utf8_and_json_integrity_passed", "frozen_parser_compatible", "counts", "schema_issue_counts",
              "native_collection_observation_counts", "confirmatory_study_ready")}), flush=True)
        # A complete inspection of incompatible schema is useful successful work.
        return 0 if result["utf8_and_json_integrity_passed"] else 2
    except Exception as exc:
        failure = {"status": "inspection_failed_preserved", "failed_utc": now(),
                   "error_type": type(exc).__name__,
                   "reason": str(exc) if isinstance(exc, InspectionError) else "Input binding or inspection failed; source values omitted",
                   "confirmatory_study_ready": False}
        save(out / "FAILED.json", failure)
        print(json.dumps(failure), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
