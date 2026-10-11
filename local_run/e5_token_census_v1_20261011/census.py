#!/usr/bin/env python3
"""Read-only, tokenizer-only census of the complete replayed AskUbuntu frame."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time


class Refusal(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def strict_parse(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    def bad(_):
        raise Refusal("nonfinite_json_number")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    except (UnicodeError, json.JSONDecodeError):
        raise Refusal("invalid_utf8_or_json") from None


def read_json(path):
    return strict_parse(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def binding(path):
    h, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
            size += len(part)
    return {"bytes": size, "sha256": h.hexdigest()}


class Watch:
    def __init__(self):
        self.files = {}

    def add(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = binding(path)
        if path in self.files:
            require(actual == self.files[path], "input_changed_between_stages")
        if expected is not None:
            require(all(actual.get(k) == v for k, v in expected.items()), "input_binding_mismatch")
        self.files[path] = actual
        return actual

    def verify(self):
        for path, expected in self.files.items():
            require(binding(path) == expected, "input_or_code_changed_during_census")


def inside(path, root):
    path, root = Path(path).resolve(strict=True), Path(root).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "file_outside_required_root_or_absent")
    return path


def verify_package(folder, expected_sha, watch):
    manifest = folder / "package_manifest.json"
    watch.add(manifest, {"sha256": expected_sha})
    package = read_json(manifest)
    require(package.get("files"), "empty_package_manifest")
    for name, expected in package["files"].items():
        require(Path(name).name == name and name not in (".", ".."), "unsafe_package_member")
        watch.add(inside(folder / name, folder), expected)


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


def physical_rows(path):
    with Path(path).open("rb") as stream:
        for line in stream:
            require(line.strip(), "blank_physical_jsonl_line")
            row = strict_parse(line)
            require(isinstance(row, dict), "nonobject_jsonl_row")
            yield row


def validate_replay(report):
    require(report.get("schema") == "third-askubuntu-source-replay-v1"
            and report.get("passed") is True and report.get("stage") == "complete", "source_replay_not_passed")
    stages = report["completed_stages"]
    require(stages["final_immutability"].get("all_watched_files_equal_initial_bytes") is True,
            "source_replay_immutability_not_passed")
    for key in ("all_prepared_objects_and_order_equal", "panel_ids_equal", "all_rebuilt_population_audit_fields_equal", "unchanged_frozen_loader_passed"):
        require(stages["frozen_preparation_replay"].get(key) is True, "source_replay_population_check_missing")
    require(stages["prepared_ascii_view"].get("complete_object_order_and_ascii_bytes_rechecked") is True,
            "source_replay_ascii_check_missing")
    return stages


def row_cost(chunks, detail, pad_length, batch_size=8):
    n = detail["content_tokens"]
    count = detail["chunks"]
    require(type(n) is int and n > 0 and count == len(chunks) == (n + 255) // 256, "chunk_count_mismatch")
    expected_lengths = [min(256, n - start) for start in range(0, n, 256)]
    require(detail["chunk_content_tokens"] == expected_lengths, "chunk_length_coverage_mismatch")
    overhead = detail["prefix_tokens_per_chunk"] + detail["special_tokens_per_chunk"]
    require(pad_length == 256 + overhead, "fixed_padding_contract_mismatch")
    attended = 0
    for index, (chunk, length) in enumerate(zip(chunks, expected_lengths)):
        require(chunk["content_start"] == index * 256 and chunk["content_stop"] == index * 256 + length
                and chunk["content_tokens"] == length, "chunk_boundaries_mismatch")
        require(len(chunk["input_ids"]) == length + overhead
                and chunk["attention_mask"] == [1] * (length + overhead), "chunk_special_or_mask_mismatch")
        attended += len(chunk["input_ids"])
    padded = count * pad_length
    return {"batch_count": (count + batch_size - 1) // batch_size,
            "full_batches_of_8": count // batch_size, "tail_batch_size": count % batch_size,
            "padded_token_slots": padded, "attended_token_slots": attended,
            "padding_token_slots": padded - attended,
            "maximum_batch_size": min(batch_size, count),
            "maximum_batch_padded_token_slots": min(batch_size, count) * pad_length}


def census_rows(rows, tokenizer, chunker, encoder_id, max_length, pad_length, ledger):
    complete, texts, ids = ListDigest(), ListDigest(), ListDigest()
    seen, partitions, chunk_hist, token_hist, tail_hist = set(), Counter(), Counter(), Counter(), Counter()
    totals = Counter()
    max_tokens = max_chunks = max_batches = max_chars = 0
    tokenization_seconds = ledger_seconds = validation_seconds = 0.0
    for index, row in enumerate(rows):
        started = time.perf_counter()
        rid, text = row.get("record_id"), row.get("text")
        require(isinstance(rid, str) and rid and rid not in seen, "duplicate_or_missing_record_id")
        require(isinstance(text, str) and text, "empty_record_text")
        require(row.get("partition") in ("train", "calibration", "test"), "unexpected_partition")
        seen.add(rid)
        complete.add(row)
        texts.add({"record_id": rid, "text": text})
        ids.add(rid)
        validation_seconds += time.perf_counter() - started
        started = time.perf_counter()
        chunks, detail = chunker(tokenizer, text, encoder_id, max_length)
        cost = row_cost(chunks, detail, pad_length)
        tokenization_seconds += time.perf_counter() - started
        chunk_count, content = detail["chunks"], detail["content_tokens"]
        partitions[row["partition"]] += 1
        chunk_hist[chunk_count] += 1
        token_hist[256 * ((content - 1) // 256 + 1)] += 1
        if cost["tail_batch_size"]:
            tail_hist[cost["tail_batch_size"]] += 1
        totals.update({key: cost[key] for key in ("batch_count", "full_batches_of_8", "padded_token_slots", "attended_token_slots", "padding_token_slots")})
        totals["rows"] += 1
        totals["content_tokens"] += content
        totals["chunks"] += chunk_count
        totals["first_chunk_truncated_records"] += detail["first_chunk_baseline_omitted_tokens"] > 0
        totals["first_window_truncated_records"] += detail["model_first_window_omitted_tokens"] > 0
        max_tokens, max_chunks = max(max_tokens, content), max(max_chunks, chunk_count)
        max_batches, max_chars = max(max_batches, cost["batch_count"]), max(max_chars, len(text))
        started = time.perf_counter()
        ledger.write(json.dumps({"row_index": index, "record_id": rid, "partition": row["partition"],
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "detail": detail, **cost},
            sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        ledger_seconds += time.perf_counter() - started
        del chunks, detail
    require(totals["rows"] > 0, "empty_census")
    return {"totals": dict(totals), "partition_counts": dict(partitions),
            "chunks_per_record_histogram": dict(sorted(chunk_hist.items())),
            "content_token_bins_256_upper_inclusive": dict(sorted(token_hist.items())),
            "tail_batch_size_histogram_excluding_zero": dict(sorted(tail_hist.items())),
            "maximum_record_content_tokens": max_tokens, "maximum_record_chunks": max_chunks,
            "maximum_record_batch_count": max_batches, "maximum_record_characters": max_chars,
            "maximum_batch_padded_token_slots": min(8, max_chunks) * pad_length,
            "full_prepared_rows_sha256": complete.finish(), "text_binding": texts.finish(),
            "ordered_record_ids_sha256": ids.finish(),
            "measured_inner_seconds": {"canonical_row_validation": validation_seconds,
                "frozen_tokenization_chunking_and_cost_checks": tokenization_seconds,
                "ledger_serialization_and_buffered_writes": ledger_seconds},
            "future_vectors_FP32_payload_bytes_excluding_npy_header": totals["rows"] * 768 * 4,
            "future_calibration_gather_FP32_payload_bytes": partitions["calibration"] * 768 * 4,
            "batch_size": 8, "fixed_padding_tokens": pad_length,
            "batching_scope": "chunks_of_one_record_only_no_cross_record_batches",
            "encoding_walltime_projection": None,
            "tokenization_speed_is_not_encoder_throughput": True}


def run(args, report):
    repo, workspace, view, out = (Path(getattr(args, name)).resolve() for name in ("repo", "workspace", "view", "out"))
    require(platform.system() == "Linux" and os.environ.get("SLURM_JOB_ID"), "linux_slurm_allocation_required")
    require(socket.gethostname().split(".")[:2] != ["csis", "mn1"], "login_node_prohibited")
    require(out.is_relative_to(workspace / "qualification") and not out.exists(), "new_private_output_required")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=False)
    (out / ".gitignore").write_text("*\n")
    report["out"] = str(out)
    phases, watch = {}, Watch()
    started = time.perf_counter()
    folder = Path(__file__).resolve().parent
    verify_package(folder, args.package_sha256, watch)
    pins = read_json(folder / "expectations.json")
    for relative, expected in read_json(folder / "frozen_code.json")["files"].items():
        watch.add(inside(repo / relative, repo), expected)
    replay_path = inside(args.replay_result, workspace)
    watch.add(replay_path, {"sha256": pins["replay_result_sha256"]})
    replay = read_json(replay_path)
    stages = validate_replay(replay)
    require(stages["frozen_preparation_replay"]["rows"] == pins["rows"], "replay_population_size_mismatch")
    recorded = stages["final_immutability"]["bindings"]
    for name in ("records.jsonl", "audit.json", "receipt.json"):
        path = inside(view / name, workspace)
        require(str(path) in recorded, "view_not_bound_by_passed_replay")
        watch.add(path, recorded[str(path)])
    for key, value in pins["view"].items():
        watch.add(view / key, value)
    sys.path.insert(0, str(repo / "empirical_execution"))
    from phase4 import embeddings as e
    from phase3 import calibration as cal
    audit = read_json(view / "audit.json")
    cal._verify(audit)
    require(audit["sha256"] == stages["prepared_ascii_view"]["derived_audit_seal"]
            and audit["dataset_id"] == "askubuntu", "view_seal_or_dataset_mismatch")
    require(e.installed_versions() == pins["library_versions"]
            and platform.python_version() == pins["python"], "pinned_CPU_runtime_mismatch")
    model_dir, tokenizer_dir = workspace / "models/e5/model", workspace / "models/e5/tokenizer"
    assets_path = inside(workspace / "models/e5/assets.json", workspace)
    watch.add(assets_path, pins["assets_file"])
    assets = read_json(assets_path)
    require(assets.get("sha256") == pins["asset_seal_sha256"]
            and assets.get("encoder_id") == e.E5
            and assets.get("encoder_revision") == pins["revision"]
            and assets.get("tokenizer_revision") == pins["revision"], "asset_identity_mismatch")
    e.verify_assets(assets, model_dir, tokenizer_dir, verify_runtime=True)
    require([x for x in assets["model_files"] if x["path"] == "model.safetensors"] == [pins["model_weight"]], "publisher_weight_binding_mismatch")
    report["runtime"] = {"python": platform.python_version(), "executable": sys.executable,
                         "libraries": e.installed_versions(), "platform": platform.platform(),
                         "slurm_job_id": os.environ["SLURM_JOB_ID"]}
    phases["initial_source_code_asset_and_runtime_verification"] = time.perf_counter() - started
    report["phase_seconds"] = phases
    report["stage"] = "offline_tokenizer_load"
    started = time.perf_counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir.resolve()),
        local_files_only=True, trust_remote_code=False, use_fast=True)
    require(tokenizer.is_fast is True, "fast_tokenizer_required")
    max_length = min(512, int(tokenizer.model_max_length))
    pad_length = 256 + tokenizer.num_special_tokens_to_add(pair=False) + len(tokenizer.encode("query: ", add_special_tokens=False, truncation=False))
    require(max_length == pins["max_length"] and pad_length == pins["padding_tokens"], "qualified_tokenizer_capacity_or_padding_mismatch")
    phases["tokenizer_load_and_capacity_check"] = time.perf_counter() - started
    report["stage"] = "complete_population_census"
    started = time.perf_counter()
    ledger_path = out / "row_token_chunk_ledger.private.jsonl"
    with ledger_path.open("x", encoding="utf-8") as ledger:
        aggregate = census_rows(physical_rows(view / "records.jsonl"), tokenizer, e.chunk_inputs,
                                e.E5, max_length, pad_length, ledger)
        flush_started = time.perf_counter()
        ledger.flush()
        os.fsync(ledger.fileno())
        phases["ledger_flush_and_fsync"] = time.perf_counter() - flush_started
    phases["complete_scan_including_decode_chunking_ledger_close"] = time.perf_counter() - started
    require(aggregate["totals"]["rows"] == pins["rows"] and aggregate["partition_counts"] == pins["partition_counts"], "incomplete_population_census")
    require(aggregate["full_prepared_rows_sha256"] == audit["prepared_rows_sha256"]
            and aggregate["text_binding"] == audit["text_binding"]
            and aggregate["ordered_record_ids_sha256"] == pins["ordered_ids_sha256"], "complete_frame_or_text_binding_mismatch")
    report["stage"] = "finalization"
    started = time.perf_counter()
    ledger_binding = binding(ledger_path)
    e.verify_assets(assets, model_dir, tokenizer_dir, verify_runtime=True)
    watch.verify()
    require(e.installed_versions() == pins["library_versions"], "runtime_changed_during_census")
    phases["ledger_hash_and_final_input_code_asset_runtime_verification"] = time.perf_counter() - started
    started = time.perf_counter()
    aggregate.update({"schema": "third-e5-token-census-aggregate-v1", "dataset": "askubuntu", "encoder_id": e.E5,
                      "model_forwards": 0, "model_instantiated": False,
                      "cache_created": False, "full_population_cache_acceptance": False,
                      "confirmatory_study_ready": False})
    write_json(out / "aggregate.json", aggregate)
    write_json(out / "input_bindings.json", {"files": {str(path): value for path, value in watch.files.items()},
                                             "assets_seal_sha256": assets["sha256"],
                                             "assets_full_inventory_verified_initial_and_final": True})
    phases["aggregate_and_binding_receipts_durable_write"] = time.perf_counter() - started
    report.update({"passed": True, "stage": "complete", "rows": aggregate["totals"]["rows"],
                   "ledger": ledger_binding, "aggregate": binding(out / "aggregate.json"),
                   "input_bindings": binding(out / "input_bindings.json"),
                   "initial_final_source_code_and_assets_equal": True,
                   "replay_result_sha256": pins["replay_result_sha256"]})


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "workspace", "view", "replay-result", "out", "package-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    report = {"schema": "third-e5-token-census-result-v1", "passed": False, "stage": "preflight",
              "evidence_role": "complete_population_tokenization_resource_census_only",
              "model_instantiated": False, "model_forwards": 0, "downloads_performed": False,
              "source_authenticity_verified": False, "primary_input_acceptance": False,
              "cache_created": False, "full_population_cache_acceptance": False,
              "confirmatory_study_ready": False, "encoding_walltime_projection": None}
    try:
        run(args, report)
    except Exception as error:
        report["error"] = str(error) if isinstance(error, Refusal) else type(error).__name__
    report["driver_seconds_through_all_data_and_binding_finalization"] = time.perf_counter() - started
    report["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["timing_boundary"] = "Driver time includes ledger close/hash, final assets and input checks, aggregate and binding writes; the Slurm process_completion receipt measures the entire child including this final result write and process exit."
    if "out" in report:
        write_json(Path(report["out"]) / "result.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "stage")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
