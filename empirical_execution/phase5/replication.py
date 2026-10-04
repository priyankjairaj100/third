"""Event-preserving WCEP intake and descriptive News date withdrawals.

Original article instances remain distinct. Summary/event metadata never enters
the encoded text. This module verifies construction, not external authenticity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from empirical_execution.phase3 import calibration as cal
from empirical_execution.phase3.panels import hash_integer, normalized_text
from empirical_execution.phase4.adapters import news_text, file_sha256

UPSTREAM_COMMIT = "8aa8418d55a4e6710277bed18bfc5349b50a5d96"
UPSTREAM_URL = "https://github.com/complementizer/wcep-mds-dataset/tree/" + UPSTREAM_COMMIT
SCHEMA = "ccu-event-replication-1"


def dump_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _events(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Event at line {index} is not an object")
                yield index, value


def _native_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip() or "\0" in str(value):
        raise ValueError("Native event IDs must be nonempty integer/string identifiers")
    return str(value)


def parse_wcep(input_paths, output_dir, *, snapshot_revision, collection_scope):
    """Read original author extracted event JSONL[.gz], never processed summaries.

Each input must have native event id/date/articles and article title/text.
Instance identity is snapshot-file checksum + native event ID + article ordinal.
Duplicate event IDs across supplied files are an error; duplicate article strings,
URLs and article IDs remain separate instances, including across events.
"""
    if not isinstance(snapshot_revision, str) or not snapshot_revision.strip():
        raise ValueError("Declared original release/revision is required")
    if collection_scope not in {"train", "val", "test", "all_official_splits", "declared_subset"}:
        raise ValueError("Explicit extraction collection scope required")
    paths = [Path(p).resolve() for p in input_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Nonempty unique original input paths required")
    bindings = [{"name": p.name, "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in paths]
    if len({x["name"] for x in bindings}) != len(bindings):
        raise ValueError("Input basenames must be unique for portable binding")
    if collection_scope == "all_official_splits" and len(paths) != 3:
        raise ValueError("All-splits declaration requires three supplied split files")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    index = sqlite3.connect(out / "event_index.sqlite3")
    index.execute("PRAGMA cache_size=-2048")
    index.execute("CREATE TABLE events(id TEXT PRIMARY KEY, supplied INTEGER, retained INTEGER)")
    counts = Counter()
    collection_counts = Counter()
    try:
        with (out / "records.jsonl").open("x", encoding="utf-8") as records, (out / "events.jsonl").open("x", encoding="utf-8") as events, (out / "exclusions.jsonl").open("x", encoding="utf-8") as exclusions:
            for path, binding in zip(paths, bindings):
                for line, event in _events(path):
                    eid = _native_id(event.get("id"))
                    if not isinstance(event.get("date"), str):
                        raise ValueError("Original event date required")
                    date.fromisoformat(event["date"])
                    collection = event.get("collection")
                    if collection not in {"train", "val", "test"}:
                        raise ValueError("Original author collection field required")
                    if collection_scope in {"train", "val", "test"} and collection != collection_scope:
                        raise ValueError("Supplied event violates declared collection scope")
                    articles = event.get("articles")
                    if not isinstance(articles, list):
                        raise ValueError("Original extracted article array required; concatenated processed documents refused")
                    index.execute("INSERT INTO events VALUES (?,?,0)", (eid, len(articles)))
                    retained = 0
                    for ordinal, article in enumerate(articles):
                        counts["article_instances_supplied"] += 1
                        rid = "wcep:" + cal.digest([binding["sha256"], eid, ordinal])
                        if not isinstance(article, dict) or not isinstance(article.get("title"), str) or not isinstance(article.get("text"), str):
                            raise ValueError("Original article title/text strings required")
                        text, repeated = news_text(article["title"], article["text"])
                        if not text:
                            _line(exclusions, {"record_id": rid, "event_id": eid, "article_ordinal": ordinal, "reason": "empty_normalized_article"})
                            counts["empty_exclusions"] += 1
                            continue
                        # URL evidence is retained, but no hostname/domain guess
                        # is silently promoted to a genuine withdrawal source.
                        row = {"record_id": rid, "text": text, "labels": [], "event_id": eid,
                               "event_date": event["date"], "collection": collection,
                               "article_ordinal": ordinal, "original_fields": article,
                               "source_unit_id": "unknown:" + rid, "source_kind": "unknown_singleton",
                               "source_policy": "record_only_until_original_URLs_and_PSL_independently_verified",
                               "adapter_input": {"file_sha256": binding["sha256"], "event_line": line,
                                   "exact_repeated_title_removed": bool(repeated)}}
                        _line(records, row)
                        retained += 1
                        counts["retained_article_instances"] += 1
                        counts["article_original_URL_present"] += isinstance(article.get("url"), str) and bool(article["url"].strip())
                    index.execute("UPDATE events SET retained=? WHERE id=?", (retained, eid))
                    counts["events"] += 1
                    counts["empty_events_after_parse"] += retained == 0
                    collection_counts[collection] += 1
                    # Original event fields, including summary/category, remain
                    # in this separate lineage ledger, never an encoder input.
                    _line(events, {"event_id": eid, "supplied_articles": len(articles), "retained_articles": retained,
                        "input_sha256": binding["sha256"], "input_line": line,
                        "original_event_fields": {k: v for k, v in event.items() if k != "articles"}})
        index.commit()
        if collection_scope == "all_official_splits" and set(collection_counts) != {"train", "val", "test"}:
            raise ValueError("All-splits declaration missing an observed official split")
        if any(file_sha256(p) != b["sha256"] for p, b in zip(paths, bindings)):
            raise ValueError("Original input changed during parsing")
        audit = cal._seal({"schema": SCHEMA, "stage": "parsed_original_events", "dataset_id": "wcep100",
            "inputs": bindings, "snapshot_revision": snapshot_revision, "collection_scope": collection_scope,
            "collection_event_counts": dict(collection_counts), "counts": dict(counts),
            "records_sha256": file_sha256(out / "records.jsonl"), "events_sha256": file_sha256(out / "events.jsonl"),
            "exclusions_sha256": file_sha256(out / "exclusions.jsonl"), "code_sha256": file_sha256(__file__),
            "text_adapter_code_sha256": file_sha256(sys.modules[news_text.__module__].__file__),
            "upstream_schema_reference": UPSTREAM_URL + "/dataset_reproduction/combine_and_split.py",
            "input_authenticity_verified": False, "complete_author_release_verified": False,
            "semantic_or_human_evidence": False, "confirmatory_study_ready": False,
            "scope": "Complete surviving article instances within supplied events; no corpus-wide completeness claim"})
        dump_json(out / "audit.json", audit)
        return audit
    except Exception as error:
        dump_json(out / "FAILED.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error), "counts_before_failure": dict(counts)})
        raise
    finally:
        index.close()


def event_panel(adapter_dir, output_dir, *, target=25000, salt="ccu-v1-wcep-event-panel"):
    """Nested ID-hash-ordered whole-event prefix; never use summary/category."""
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("Positive article target required")
    root = Path(adapter_dir)
    audit = json.loads((root / "audit.json").read_text()); cal._verify(audit)
    if (audit.get("schema") != SCHEMA or audit.get("stage") != "parsed_original_events"
            or audit.get("code_sha256") != file_sha256(__file__) or (root / "FAILED.json").exists()):
        raise ValueError("Current successfully parsed WCEP input required")
    for name in ("records", "events", "exclusions"):
        if file_sha256(root / (name + ".jsonl")) != audit[name + "_sha256"]:
            raise ValueError("Altered WCEP " + name)
    events = [value for _, value in _events(root / "events.jsonl")]
    ordered = sorted(events, key=lambda e: (hash_integer(salt, e["event_id"]), e["event_id"]))
    chosen, total = [], 0
    for event in ordered:
        if total >= target:
            break
        chosen.append(event["event_id"]); total += event["retained_articles"]
    selected = set(chosen)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=False)
    ids = []
    with (out / "records.jsonl").open("x", encoding="utf-8") as stream:
        for _, row in _events(root / "records.jsonl"):
            if row["event_id"] in selected:
                _line(stream, row); ids.append(row["record_id"])
    if len(ids) != total or len(set(ids)) != total:
        raise ValueError("Whole-event counts/instance identity disagree")
    manifest = cal._seal({"schema": SCHEMA, "stage": "whole_event_panel", "dataset_id": "wcep100",
        "adapter_audit_sha256": audit["sha256"], "code_sha256": file_sha256(__file__), "salt": salt,
        "target": target, "selected_articles": total, "selected_events": chosen,
        "overshoot": max(0, total-target), "shortfall": max(0,target-total),
        "record_ids": ids, "records_sha256": file_sha256(out / "records.jsonl"),
        "whole_boundary_event_included": True, "event_membership_is_duplicate_gold": False,
        "threshold_policy": "reuse_frozen_same_encoder_CC_News_threshold_no_WCEP_retuning",
        "source_arm_available": False, "confirmatory_study_ready": False})
    dump_json(out / "panel.json", manifest)
    return manifest


def news_chronological_requests(rows, *, date_field="date", resolution="day", horizon=128):
    """Single observed ordering of whole recorded-date groups; not random trials.

Keep whole dates. Stop before a group would exceed the locked record horizon;
record that blocked next group, never cut a date or quietly enlarge the state.
"""
    if resolution not in {"day", "month"} or isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("Day/month resolution and nonnegative integer horizon required")
    ids = [r.get("record_id") for r in rows]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Unique record IDs required")
    if any(row.get("partition") != "train" for row in rows):
        raise ValueError("Only the already prepared News training population may receive deletion requests")
    groups = defaultdict(list)
    for row in rows:
        value = row.get("original_fields", {}).get(date_field)
        if not isinstance(value, str):
            raise ValueError("Original recorded dates required; no inferred dates")
        # News extraction dates may include timestamp suffixes. Only the
        # explicitly recorded calendar date is used; no timezone is invented.
        day = date.fromisoformat(value[:10]).isoformat()
        if day[:4] not in {"2018", "2019"}:
            raise ValueError("Chronological analysis is confined to the declared 2018–2019 News population")
        groups[day if resolution == "day" else day[:7]].append(row["record_id"])
    deleted = []; checkpoints = []; blocked = None
    for interval in sorted(groups):
        batch = sorted(groups[interval])
        if len(deleted) + len(batch) > horizon:
            blocked = {"date_group": interval, "records": len(batch), "remaining_budget": horizon-len(deleted)}
            break
        deleted.extend(batch)
        checkpoints.append({"date_group": interval, "batch_record_ids": batch,
            "cumulative_record_ids": list(deleted), "cumulative_records": len(deleted)})
    return cal._seal({"schema": "ccu-news-chronological-request-1", "resolution": resolution,
        "record_horizon": horizon, "population_binding": cal.digest(rows), "record_ids": ids,
        "code_sha256": file_sha256(__file__), "checkpoints": checkpoints, "blocked_next_group": blocked,
        "single_descriptive_stream": True, "independent_random_replicates": 0,
        "source_withdrawal_claim": False, "semantic_or_quality_claim": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("wcep"); p.add_argument("inputs", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--snapshot-revision", required=True); p.add_argument("--collection-scope", required=True)
    p = sub.add_parser("panel"); p.add_argument("adapter"); p.add_argument("--out", required=True); p.add_argument("--target", type=int, default=25000)
    p = sub.add_parser("chronological"); p.add_argument("records"); p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, required=True); p.add_argument("--resolution", choices=["day", "month"], default="day")
    args = parser.parse_args()
    if args.command == "wcep":
        value = parse_wcep(args.inputs, args.out, snapshot_revision=args.snapshot_revision, collection_scope=args.collection_scope)
    elif args.command == "panel":
        value = event_panel(args.adapter, args.out, target=args.target)
    else:
        value = news_chronological_requests([r for _, r in _events(args.records)], horizon=args.horizon, resolution=args.resolution)
        dump_json(args.out, value)
    print(json.dumps({"stage": value.get("stage", "chronological"), "sha256": value["sha256"], "confirmatory_study_ready": False}))


if __name__ == "__main__":
    main()
