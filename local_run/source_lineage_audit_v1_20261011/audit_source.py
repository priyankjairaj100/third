#!/usr/bin/env python3
"""Prospective CPU-only AskUbuntu local byte-lineage inventory; never admission."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time


class AuditError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise AuditError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pairs_hook(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def parse(data):
    def bad_constant(_):
        raise AuditError("nonfinite_json_number")
    try:
        return json.loads(data, object_pairs_hook=pairs_hook, parse_constant=bad_constant)
    except (UnicodeError, json.JSONDecodeError):
        raise AuditError("invalid_utf8_or_json") from None


def read_json(path):
    return parse(Path(path).read_bytes())


def hash_stream(stream):
    h, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(1 << 20), b""):
        h.update(block)
        size += len(block)
    return {"bytes": size, "sha256": h.hexdigest()}


def binding(path):
    with Path(path).open("rb") as stream:
        return hash_stream(stream)


def match(actual, expected, code):
    require(all(actual.get(k) == v for k, v in expected.items()), code)


def verify_seal(value):
    require(value.get("sha256") == digest({k: v for k, v in value.items() if k != "sha256"}),
            "canonical_seal_mismatch")


def jsonl(path):
    # Binary iteration splits only on physical LF, never U+2028 or U+2029.
    with Path(path).open("rb") as stream:
        for line in stream:
            require(line.strip(), "blank_physical_jsonl_line")
            row = parse(line)
            require(isinstance(row, dict), "jsonl_row_not_object")
            yield row


class ListDigest:
    def __init__(self):
        self.h, self.count = hashlib.sha256(b"["), 0

    def add(self, value):
        if self.count:
            self.h.update(b",")
        self.h.update(canonical(value))
        self.count += 1

    def finish(self):
        h = self.h.copy()
        h.update(b"]")
        return h.hexdigest()


def ascii_transport(source, target):
    """Verify exact historical ensure_ascii=True serialization and all objects."""
    expected, size, count = hashlib.sha256(), 0, 0
    sentinel = object()
    for left, right in itertools.zip_longest(jsonl(source), jsonl(target), fillvalue=sentinel):
        require(left is not sentinel and right is not sentinel, "transport_row_count_mismatch")
        require(canonical(left) == canonical(right), "transport_row_or_order_mismatch")
        encoded = json.dumps(left, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
        expected.update(encoded)
        size += len(encoded)
        count += 1
    match(binding(target), {"bytes": size, "sha256": expected.hexdigest()}, "transport_bytes_mismatch")
    return {"rows": count, "complete_object_and_order_equality": True,
            "exact_ascii_serialization": True, "physical_lf_only": True}


TAG_VALUE = re.compile(r"^\|(?:[^|<>]+\|)+$")
TAG_ATTRIBUTE = re.compile(r'Tags="([^"]*)"')


def transform_line(line, counts):
    def replace(found):
        value = found.group(1)
        if TAG_VALUE.fullmatch(value):
            counts["tags_converted"] += 1
            value = "".join("&lt;" + tag + "&gt;" for tag in value.split("|") if tag)
        else:
            counts["tags_unchanged"] += 1
        return 'Tags="' + value + '"'
    return TAG_ATTRIBUTE.sub(replace, line)


def tag_bridge(original, transformed):
    counts = Counter({"tags_converted": 0, "tags_unchanged": 0})
    h, size = hashlib.sha256(), 0
    with Path(original).open("r", encoding="utf-8", newline="") as stream:
        for line in stream:
            encoded = transform_line(line, counts).encode("utf-8")
            h.update(encoded)
            size += len(encoded)
    match(binding(transformed), {"sha256": h.hexdigest(), "bytes": size}, "tag_only_transform_mismatch")
    return {**counts, "complete_expected_transformed_bytes_equal": True}


def member_binding(executable, archive, member, timeout=1800):
    require(member in ("Posts.xml", "PostLinks.xml"), "unexpected_archive_member")
    # No extraction writes; stderr is private and is never copied to the report.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(executable), "x", "-so", "-y", str(archive), member],
                                   stdout=subprocess.PIPE, stderr=errors)
        expired = threading.Event()
        def cancel():
            expired.set()
            process.kill()
        timer = threading.Timer(timeout, cancel)
        timer.start()
        try:
            result = hash_stream(process.stdout)
            returncode = process.wait()
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
        require(not expired.is_set(), "archive_member_timeout")
        require(returncode == 0, "archive_member_extraction_failed")
        return result


def population_scan(path, audit, design):
    full, texts, ids = ListDigest(), ListDigest(), ListDigest()
    seen, sources, partitions = set(), {}, Counter()
    for row in jsonl(path):
        rid = row.get("record_id")
        require(isinstance(rid, str) and rid and rid not in seen, "nonunique_population_record_id")
        seen.add(rid)
        fields = row["original_fields"]
        require(fields.get("site") == "askubuntu.com", "native_site_mismatch")
        owner = fields.get("OwnerUserId")
        require(owner is None or type(owner) is int, "native_owner_type_mismatch")
        source = ("account:" + json.dumps(["askubuntu.com", owner], ensure_ascii=False, separators=(",", ":"))
                  if owner is not None and owner > 0 else "unknown:" + rid)
        require(row.get("source_unit_id") == source, "native_source_id_mismatch")
        value = int.from_bytes(hashlib.sha256((design["seeds"]["source_split_salt"] + "\0" + source).encode()).digest(), "big")
        part = "train" if value * 10 < (1 << 256) * 7 else "calibration" if value * 20 < (1 << 256) * 17 else "test"
        require(row.get("partition") == part, "native_source_split_mismatch")
        require(sources.setdefault(source, part) == part, "cross_partition_source")
        partitions[part] += 1
        full.add(row)
        texts.add({k: row[k] for k in ("record_id", "text")})
        ids.add(rid)
    require(full.count > 0, "empty_population")
    require(full.finish() == audit.get("prepared_rows_sha256"), "population_rows_binding_mismatch")
    require(texts.finish() == audit.get("text_binding"), "population_text_binding_mismatch")
    return {"rows": full.count, "prepared_rows_sha256": full.finish(),
            "text_binding": texts.finish(), "ordered_record_ids_sha256": ids.finish(),
            "source_units": len(sources), "unknown_source_units": sum(s.startswith("unknown:") for s in sources),
            "partition_counts": dict(partitions), "independent_native_source_split_verified": True}


def safe_input(base, relative):
    candidate = (base / relative).resolve(strict=True)
    require(candidate.is_relative_to(base) and candidate.is_file(), "input_outside_workspace_or_not_file")
    return candidate


def run(args, report):
    repo, workspace, out = (Path(getattr(args, k)).resolve() for k in ("repo", "workspace", "out"))
    require(out.is_relative_to(workspace / "qualification"), "output_must_be_private_qualification_subdirectory")
    require(not out.exists(), "output_already_exists")
    require(os.environ.get("SLURM_JOB_ID"), "slurm_allocation_required")
    require(socket.gethostname().split(".")[:2] != ["csis", "mn1"], "login_node_prohibited")
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=False)
    report["out"] = str(out)
    expected_path = Path(__file__).with_name("expectations.json")
    expected = read_json(expected_path)
    for value in (expected["archive_sha256"], expected["design_sha256"], expected["population_audit_seal"],
                  *expected["code"].values(), *(v["sha256"] for v in expected["files"].values() if "sha256" in v)):
        require(re.fullmatch("[a-f0-9]{64}", value), "invalid_expected_sha256")
    watched = {}
    def watch(path, declared=None):
        value = binding(path)
        if path in watched:
            require(value == watched[path], "input_changed_between_stages")
        if declared:
            match(value, declared, "historical_binding_mismatch")
        watched[path] = value
        return value
    def stage(name, value):
        report["completed_stages"][name] = value
        with (out / (name + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    report["stage"] = "input_inventory"
    watch(Path(__file__).resolve())
    watch(expected_path)
    paths = {relative: safe_input(workspace, relative) for relative in expected["files"]}
    for relative, path in paths.items():
        watch(path, expected["files"][relative])
    for relative, sha in expected["code"].items():
        watch(safe_input(repo, relative), {"sha256": sha})
    design_path = safe_input(repo, "output/empirical_program/study_design.json")
    watch(design_path, {"sha256": expected["design_sha256"]})
    design = read_json(design_path)
    archive = safe_input(workspace, "inputs/askubuntu/askubuntu.com.7z")
    watch(archive, {"bytes": expected["archive_bytes"], "sha256": expected["archive_sha256"]})
    stage("input_inventory", {str(path): value for path, value in watched.items()})
    report["stage"] = "archive_members"
    executable = shutil.which(args.seven_zip)
    require(executable is not None, "seven_zip_unavailable_no_download_attempted")
    members = {}
    for member in ("Posts.xml", "PostLinks.xml"):
        value = member_binding(executable, archive, member)
        match(value, watched[paths["inputs/askubuntu/" + member]], "archive_member_local_file_mismatch")
        members[member] = value
    stage("archive_members", {"members": members, "local_archive_member_byte_equality": True,
                              "external_source_authenticity": False})
    report["stage"] = "tag_transform"
    tags = tag_bridge(paths["inputs/askubuntu/Posts.xml"], paths["inputs/askubuntu/Posts.angle.xml"])
    match(tags, {key: expected[key] for key in ("tags_converted", "tags_unchanged")}, "tag_counts_mismatch")
    stage("tag_transform", tags)
    report["stage"] = "adapter_bindings"
    adapter = workspace / "prepared/askubuntu/a02/adapter"
    adapter_audit_path = safe_input(workspace, "prepared/askubuntu/a02/adapter/audit.json")
    watch(adapter_audit_path)
    a = read_json(adapter_audit_path)
    require(a.get("schema") == "ccu-corpus-adapters-1" and a.get("dataset_id") == "askubuntu"
            and a.get("status") == "parsed_pending_external_verification", "adapter_schema_or_status_mismatch")
    require(a.get("code_sha256") == expected["code"]["empirical_execution/phase4/adapters.py"]
            and a.get("source_helper_sha256") == expected["code"]["empirical_execution/phase3/panels.py"], "adapter_code_binding_mismatch")
    require(a["settings"].get("archive_sha256_declared_unverified") == expected["archive_sha256"]
            and a["settings"].get("archive_release_declared") == expected["archive_release"], "adapter_archive_declaration_mismatch")
    require(len(a["inputs"]) == 2, "adapter_input_count_mismatch")
    for spec, name in zip(a["inputs"], ("Posts.angle.xml", "PostLinks.xml")):
        require(spec.get("name") == name, "adapter_input_name_mismatch")
        match(watched[paths["inputs/askubuntu/" + name]], {k: spec[k] for k in ("bytes", "sha256")}, "adapter_input_binding_mismatch")
    required_outputs = {"records.jsonl", "exclusions.jsonl", "identity_index.sqlite3", "duplicate_links.jsonl",
                        "duplicate_links.json", "duplicate_link_metadata.jsonl"}
    require(required_outputs.issubset(a["outputs"]), "missing_adapter_output_binding")
    for name, spec in a["outputs"].items():
        require(Path(name).name == name, "unsafe_adapter_output_name")
        watch(safe_input(workspace, "prepared/askubuntu/a02/adapter/" + name), {k: spec[k] for k in ("bytes", "sha256")})
    require(a["counts"].get("retained") == expected["adapter_rows"], "adapter_row_count_mismatch")
    stage("adapter_bindings", {"all_declared_output_hashes_verified": True, "frozen_code_binding_verified": True,
                               "input_bindings_verified": True, "full_adapter_replay_executed": False})
    report["stage"] = "physical_jsonl_transport"
    bridge = ascii_transport(adapter / "records.jsonl", adapter / "records.line_safe.jsonl")
    require(bridge["rows"] == expected["adapter_rows"], "transport_adapter_count_mismatch")
    stage("physical_jsonl_transport", bridge)
    report["stage"] = "population_bindings"
    population = workspace / "prepared/askubuntu/a02/population"
    p = read_json(population / "audit.json")
    verify_seal(p)
    require(p["sha256"] == expected["population_audit_seal"], "population_historical_seal_mismatch")
    require(p.get("dataset_id") == "askubuntu" and p.get("stage") == "fixed_guards_done_semantic_guard_pending", "population_stage_mismatch")
    for key, value in {"input_records_file_sha256": watched[paths["prepared/askubuntu/a02/adapter/records.line_safe.jsonl"]]["sha256"],
                       "prepared_records_file_sha256": watched[paths["prepared/askubuntu/a02/population/records.jsonl"]]["sha256"],
                       "study_design_sha256": expected["design_sha256"],
                       "preparation_code_sha256": expected["code"]["empirical_execution/phase3/run_preparation.py"],
                       "source_and_scoring_code_sha256": expected["code"]["empirical_execution/phase3/panels.py"]}.items():
        require(p.get(key) == value, "population_input_or_code_binding_mismatch")
    scan = population_scan(population / "records.jsonl", p, design)
    require(scan["rows"] == expected["population_rows"], "population_row_count_mismatch")
    stage("population_bindings", scan)
    report["stage"] = "immutability"
    for path, before in watched.items():
        require(binding(path) == before, "input_changed_during_audit")
    stage("immutability", {"all_watched_input_final_bytes_equal_initial": True,
                           "watched_files": len(watched), "source_mutations_performed": False})
    report["passed"] = True
    report["stage"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "workspace", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seven-zip", default="7z")
    args = parser.parse_args()
    start = time.monotonic()
    report = {"schema": "third-askubuntu-source-lineage-audit-v1", "passed": False,
              "evidence_role": "prospective_local_byte_lineage_inventory", "completed_stages": {},
              "external_source_origin_authenticated": False, "primary_input_acceptance": False,
              "full_population_cache_acceptance": False, "confirmatory_study_ready": False,
              "deferred": ["full_frozen_adapter_replay", "full_frozen_preparation_replay",
                           "prepared_ascii_view_receipt_binding", "publisher_archive_authenticity_and_completeness",
                           "archive_attribution_review", "full_frozen_CPU_E5_cache", "semantic_guard_and_human_gates"]}
    try:
        run(args, report)
    except Exception as error:
        # Never disclose an input row or arbitrary parser exception message.
        report["error"] = str(error) if isinstance(error, AuditError) else type(error).__name__
    report["seconds"] = time.monotonic() - start
    report["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if "out" in report:
        with (Path(report["out"]) / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k: report[k] for k in ("passed", "seconds")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
