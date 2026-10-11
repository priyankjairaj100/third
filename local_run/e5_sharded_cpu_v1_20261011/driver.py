#!/usr/bin/env python3
"""Plan, qualify and assemble versioned CPU shards; never submit bulk jobs."""
import argparse
from bisect import bisect_left
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import common as m


def weighted_ranges(costs, shard_count):
    m.require(type(shard_count) is int and 1 <= shard_count <= len(costs)
              and all(math.isfinite(v) and v > 0 for v in costs), "invalid_shard_costs")
    cumulative = [0.0]
    for cost in costs:
        cumulative.append(cumulative[-1] + cost)
    boundaries = [0]
    for index in range(1, shard_count):
        cut = bisect_left(cumulative, cumulative[-1] * index / shard_count)
        boundaries.append(max(boundaries[-1] + 1, min(cut, len(costs) - (shard_count - index))))
    boundaries.append(len(costs))
    return [{"index": i, "start": start, "stop": stop,
        "modeled_forward_seconds": sum(costs[start:stop])}
        for i, (start, stop) in enumerate(zip(boundaries, boundaries[1:]))]


def make_plan(ctx, kind):
    entries = list(m.c.physical_rows(ctx["census_dir"] / "row_token_chunk_ledger.private.jsonl"))
    small, _, _ = m.b.select_ledger(entries, ctx["aggregate"])
    if kind == "qualification":
        calibration = [row for row in entries if row["partition"] == "calibration"]
        longest = max(row["detail"]["chunks"] for row in calibration)
        m.require(longest == 51, "pinned_longest_calibration_bucket_changed")
        selected = small + [min((r for r in calibration if r["detail"]["chunks"] == k),
                               key=lambda r: m.b.selection_key(r["record_id"])) for k in (9, 17, longest)]
        indices = sorted(row["row_index"] for row in selected)
        shard_count = 4
    else:
        m.require(kind == "full", "invalid_plan_kind")
        indices, shard_count = list(range(len(entries))), 32
    frame = [entries[i] for i in indices]
    weights = {int(k): value["max"] for k, value in ctx["projection"]["model_forward_seconds_by_batch_size"].items()}
    costs = [row["full_batches_of_8"] * weights[8] + (weights[row["tail_batch_size"]] if row["tail_batch_size"] else 0.0) for row in frame]
    shards = weighted_ranges(costs, shard_count)
    m.require(all(s["modeled_forward_seconds"] < 8 * 3600 for s in shards), "modeled_shard_exceeds_allocation_cap")
    plan = m.seal({"schema": "third-e5-cpu-shard-plan-v1", "kind": kind, "fingerprint": ctx["fingerprint"],
        "full_source_frame_sha256": ctx["aggregate"]["full_prepared_rows_sha256"],
        "census_ledger": ctx["pins"]["census_files"]["row_token_chunk_ledger.private.jsonl"],
        "budget_projection": ctx["pins"]["budget_files"]["budget_projection.json"],
        "qualification_result": ctx.get("qualified_result_binding") if kind == "full" else None,
        "frame_row_indices": indices, "frame_ordered_ids_sha256": hashlib.sha256(m.c.canonical([r["record_id"] for r in frame])).hexdigest(),
        "shards": shards, "documents": len(frame), "chunks_per_pass": sum(r["detail"]["chunks"] for r in frame),
        "forward_batches_per_pass": sum(r["batch_count"] for r in frame),
        "cost_rule": "sum full size-eight batches and remainder using each shape's observed maximum model-forward seconds; cumulative equal-cost cuts in original row order, whole documents only",
        "qualification_selection_salt": m.b.SALT if kind == "qualification" else None,
        "allocation_cap_hours": 8, "guaranteed_walltime_bound": False,
        "original_monolithic_writer_executed": False, "original_monolithic_replay_executed": False,
        "full_population_cache_acceptance": False, "confirmatory_study_ready": False})
    m.validate_plan(plan, ctx["fingerprint"], ctx["pins"]["rows"])
    return plan


def compare_cache_files(left, right):
    """Byte-exact vector and full parsed-log equality, including row order."""
    import numpy as np
    left, right = Path(left), Path(right)
    a, b = np.load(left / "vectors.npy", mmap_mode="r", allow_pickle=False), np.load(right / "vectors.npy", mmap_mode="r", allow_pickle=False)
    m.require(a.dtype == b.dtype == np.dtype("float32") and a.shape == b.shape and len(a.shape) == 2 and a.shape[1] == 768,
              "cache_shape_or_dtype_mismatch")
    for index in range(0, len(a), 256):
        m.require(a[index:index+256].tobytes() == b[index:index+256].tobytes(), "cache_vector_bytes_differ")
    from itertools import zip_longest
    count = 0
    for x, y in zip_longest(m.c.physical_rows(left / "chunk_log.jsonl"), m.c.physical_rows(right / "chunk_log.jsonl")):
        m.require(x is not None and y is not None and x == y, "cache_complete_chunk_log_differs")
        count += 1
    m.require(count == len(a), "cache_chunk_log_count_mismatch")


def compare_shard_slice(assembled_dir, encoded_dir, start, stop):
    import numpy as np
    assembled_dir, encoded_dir = Path(assembled_dir), Path(encoded_dir)
    full = np.load(assembled_dir / "vectors.npy", mmap_mode="r", allow_pickle=False)
    part = np.load(encoded_dir / "vectors.npy", mmap_mode="r", allow_pickle=False)
    m.require(full.dtype == part.dtype == np.dtype("float32") and part.shape == (stop - start, 768)
              and full.ndim == 2 and full.shape[1] == 768 and stop <= len(full), "assembled_slice_shape_mismatch")
    for index in range(0, len(part), 256):
        m.require(full[start+index:min(start+index+256, stop)].tobytes() == part[index:index+256].tobytes(), "assembled_shard_vector_bytes_differ")
    from itertools import islice
    wanted = list(m.c.physical_rows(encoded_dir / "chunk_log.jsonl"))
    actual = list(islice(m.c.physical_rows(assembled_dir / "chunk_log.jsonl"), start, stop))
    m.require(len(wanted) == len(actual) == stop - start and wanted == actual, "assembled_shard_log_slice_differs")


def validate_replay_receipt(ctx, plan, index, replay_dir, encoded_dir, encoded):
    import worker
    worker._exact_inventory(Path(replay_dir), worker.FILES | {"completion.json"})
    path = Path(replay_dir) / "completion.json"
    ctx["watch"].add(path)
    report = m.c.read_json(path)
    for key, expected in {"schema": "third-e5-cpu-shard-completion-v1", "kind": "replay", "passed": True,
        "fingerprint": ctx["fingerprint"], "plan_sha256": plan["sha256"], "shard_index": index,
        "start": encoded["start"], "stop": encoded["stop"], "rows": encoded["rows"]}.items():
        m.require(report.get(key) == expected, "replay_receipt_identity_mismatch")
    m.require(report.get("encoded_completion") == m.binding(Path(encoded_dir) / "completion.json"), "replay_does_not_bind_encoded_completion")
    if plan["kind"] == "full":
        m.require(report.get("qualification_result") == plan.get("qualification_result")
                  and plan.get("qualification_result") is not None, "full_replay_qualification_binding_mismatch")
    m.require(report.get("exact_vector_bytes_equal") is True and report.get("complete_chunk_log_equal") is True
              and all(report.get(name) is False for name in worker.FALSE_FLAGS), "replay_checks_or_claim_boundaries_missing")
    m.require(all(type(report.get(key)) is int and report[key] == encoded[key]
                  for key in ("content_tokens", "chunks", "model_forward_batches")), "replay_work_counters_mismatch")
    m.require(set(report["files"]) == {"vectors.npy", "chunk_log.jsonl"}, "replay_output_inventory_mismatch")
    for name, expected in report["files"].items():
        ctx["watch"].add(m.c.inside(Path(replay_dir) / name, Path(replay_dir)), expected)
    compare_cache_files(encoded_dir, replay_dir)
    return report


def assemble(ctx, plan, encoded_dirs, replay_dirs, out, golden_dir=None):
    import worker
    np = ctx["np"]
    m.require(len(encoded_dirs) == len(replay_dirs) == len(plan["shards"]), "assembly_requires_every_shard_and_replay")
    rows, expected = m.rows_for_indices(ctx, plan["frame_row_indices"])
    m.require(hashlib.sha256(m.c.canonical([r["record_id"] for r in rows])).hexdigest() == plan["frame_ordered_ids_sha256"], "assembly_frame_order_mismatch")
    receipts = []
    for index, shard in enumerate(plan["shards"]):
        start, stop = shard["start"], shard["stop"]
        encoded = worker.validate_encoded(ctx, plan, index, encoded_dirs[index], rows=rows[start:stop], expected=expected[start:stop])
        ctx["watch"].add(Path(encoded_dirs[index]) / "completion.json")
        for name, expected_binding in encoded["files"].items():
            ctx["watch"].add(Path(encoded_dirs[index]) / name, expected_binding)
        validate_replay_receipt(ctx, plan, index, replay_dirs[index], encoded_dirs[index], encoded)
        receipts.append({"index": index, "encoded_completion": m.binding(Path(encoded_dirs[index]) / "completion.json"),
            "replay_completion": m.binding(Path(replay_dirs[index]) / "completion.json")})
    out = m.new_output(ctx, out)
    vectors = np.lib.format.open_memmap(out / "vectors.npy", mode="w+", dtype=np.float32, shape=(len(rows), 768))
    with (out / "chunk_log.jsonl").open("x", encoding="utf-8") as log:
        count = 0
        for index, shard in enumerate(plan["shards"]):
            source = np.load(Path(encoded_dirs[index]) / "vectors.npy", mmap_mode="r", allow_pickle=False)
            vectors[shard["start"]:shard["stop"]] = source
            for entry in m.c.physical_rows(Path(encoded_dirs[index]) / "chunk_log.jsonl"):
                m.require(entry["row_index"] == count, "assembly_row_index_gap_or_reorder")
                log.write(json.dumps(entry, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
                count += 1
        log.flush(); os.fsync(log.fileno())
    m.require(count == len(rows), "assembly_count_mismatch")
    vectors.flush(); del vectors
    with (out / "vectors.npy").open("rb") as stream:
        os.fsync(stream.fileno())
    for index, shard in enumerate(plan["shards"]):
        compare_shard_slice(out, encoded_dirs[index], shard["start"], shard["stop"])
    if golden_dir is not None:
        compare_cache_files(out, golden_dir)
    m.verify_final(ctx)
    manifest = m.seal({"schema": "third-e5-sharded-cpu-cache-v1", "passed": True, "plan_sha256": plan["sha256"],
        "fingerprint": ctx["fingerprint"], "runtime": ctx["runtime"], "rows": len(rows), "shape": [len(rows), 768],
        "dtype": "float32", "frame_ordered_ids_sha256": plan["frame_ordered_ids_sha256"],
        "full_source_frame_sha256": plan["full_source_frame_sha256"], "shard_receipts": receipts,
        "files": {name: m.binding(out / name) for name in ("vectors.npy", "chunk_log.jsonl")},
        "all_rows_fresh_process_replayed_under_this_version": True,
        "original_monolithic_full_writer_executed": False, "original_monolithic_full_replay_executed": False,
        "source_authenticity_verified": False, "primary_input_acceptance": False,
        "full_population_cache_acceptance": False, "confirmatory_study_ready": False})
    m.atomic_completion(out / "manifest.json", manifest)
    return manifest


def run_child(command, receipt_dir):
    receipt_dir = Path(receipt_dir); receipt_dir.mkdir(exist_ok=False)
    start = time.perf_counter()
    with (receipt_dir / "stdout.txt").open("x") as stdout, (receipt_dir / "stderr.txt").open("x") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        process.wait()
        for stream in (stdout, stderr):
            stream.flush(); os.fsync(stream.fileno())
    m.write_json(receipt_dir / "process.json", {"command": command, "returncode": process.returncode,
        "child_pid": process.pid, "launcher_pid": os.getpid(),
        "child_wall_seconds_through_exit": time.perf_counter() - start})
    m.require(process.returncode == 0, "fresh_child_failed_see_private_receipt")


def qualify(args, ctx, out, report):
    # Import availability is checked before any model load/forward. The module's
    # unrelated source/task routines are not executed or claimed as acceptance.
    from phase6 import acceptance
    report["phase6_acceptance_import_preflight_passed"] = True
    plan = make_plan(ctx, "qualification")
    m.write_json(out / "plan.json", plan)
    ctx["watch"].add(out / "plan.json")
    rows, expected = m.rows_for_indices(ctx, plan["frame_row_indices"])
    m.require(len(rows) == 19 and plan["chunks_per_pass"] == 149, "bounded_golden_population_mismatch")
    subset = out / "engineering_subset"; subset.mkdir()
    with (subset / "records.jsonl").open("x", encoding="ascii") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    audit = ctx["cal"]._seal({"schema": "third-natural-engineering-subset-audit-v1",
        "dataset_id": "askubuntu_engineering_subset_19_v1", "stage": "fixed_guards_done_semantic_guard_pending",
        "prepared_records_file_sha256": m.binding(subset / "records.jsonl")["sha256"],
        "prepared_rows_sha256": ctx["cal"].digest(rows), "text_binding": ctx["e"].run_preparation.text_binding(rows),
        "preparation_code_sha256": ctx["e"].file_hash(ctx["e"].run_preparation.__file__),
        "source_and_scoring_code_sha256": ctx["e"].file_hash(ctx["e"].panels.__file__),
        "rows": 19, "partition_counts": {"calibration": 19}, "upstream_full_preparation_audit_sha256": ctx["audit"]["sha256"],
        "upstream_full_prepared_rows_sha256": plan["full_source_frame_sha256"], "source_row_indices": plan["frame_row_indices"],
        "subset_derivation": "Unchanged objects selected from the fully replayed population; fixed guards inherited, original preparation not rerun on this subset.",
        "subset_derivation_code": m.binding(Path(__file__)), "engineering_only": True,
        "full_population_cache_acceptance": False, "confirmatory_study_ready": False})
    m.write_json(subset / "audit.json", audit)
    ctx["watch"].add(subset / "records.jsonl"); ctx["watch"].add(subset / "audit.json")
    loaded, _ = ctx["e"].load_prepared_records(subset / "records.jsonl", subset / "audit.json")
    m.require(loaded == rows and all(row["partition"] == "calibration" for row in rows), "golden_subset_binding_mismatch")
    golden = out / "golden"
    report["stage"] = "golden_unmodified_writer"
    run_child([sys.executable, str(ctx["repo"] / "empirical_execution/phase4/embeddings.py"), "encode",
        "--records", str(subset / "records.jsonl"), "--preparation-audit", str(subset / "audit.json"),
        "--asset-manifest", str(ctx["assets_path"]), "--model-dir", str(ctx["model_dir"]),
        "--tokenizer-dir", str(ctx["tokenizer_dir"]), "--out", str(golden), "--batch-size", "8", "--threads", "1",
        "--evidence-role", "engineering_nonconfirmatory"], out / "golden_process")
    golden_manifest = m.c.read_json(golden / "cache_manifest.json"); ctx["cal"]._verify(golden_manifest)
    m.require(golden_manifest["shape"] == [19, 768] and golden_manifest["provenance"]["dataset_id"] == audit["dataset_id"], "golden_scope_mismatch")
    for path in golden.iterdir():
        if path.is_file():
            ctx["watch"].add(path)
    encoded_dirs, replay_dirs = [], []
    for mode in ("encode", "replay"):
        for index in range(4):
            destination = out / f"{mode}-{index}"
            command = [sys.executable, str(ctx["folder"] / "worker.py"), "--mode", mode,
                "--plan", str(out / "plan.json"), "--shard-index", str(index), "--out", str(destination), *m.common_arguments(args)]
            if mode == "replay":
                command += ["--encoded-dir", str(encoded_dirs[index])]
                replay_dirs.append(destination)
            else:
                encoded_dirs.append(destination)
            report["stage"] = f"fresh_{mode}_shard_{index}"
            run_child(command, out / f"{mode}-{index}-process")
    report["stage"] = "canonical_assembly_and_golden_comparison"
    assemble(ctx, plan, encoded_dirs, replay_dirs, out / "assembled", golden_dir=golden)
    m.verify_final(ctx)
    evidence = {path.relative_to(out).as_posix(): m.binding(path) for path in sorted(out.rglob("*")) if path.is_file()}
    report.update({"passed": True, "stage": "complete", "fingerprint": ctx["fingerprint"], "runtime": ctx["runtime"],
        "plan_sha256": plan["sha256"], "qualification_rows": 19, "chunks_per_pass": 149,
        "golden_all_vector_bytes_equal": True, "golden_all_chunk_log_rows_equal": True,
        "all_shards_fresh_process_replayed": True, "logical_passes": 3, "document_encodings": 57,
        "chunks_processed": 447, "forward_batches_per_pass": 28, "model_forward_batches": 84,
        "original_writer_executed_only_on_named_19_row_engineering_subset": True,
        "evidence_files": evidence})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "plan-full", "assemble"))
    m.add_arguments(parser)
    parser.add_argument("--out", required=True)
    parser.add_argument("--qualification-result")
    parser.add_argument("--plan")
    parser.add_argument("--encoded-dirs", nargs="+")
    parser.add_argument("--replay-dirs", nargs="+")
    args = parser.parse_args(); start = time.perf_counter()
    report = {"schema": "third-e5-sharded-cpu-qualification-v1", "passed": False, "stage": "preflight",
        "source_authenticity_verified": False, "primary_input_acceptance": False,
        "original_monolithic_full_writer_executed": False, "original_monolithic_full_replay_executed": False,
        "full_population_cache_acceptance": False, "confirmatory_study_ready": False,
        "full_population_launch_authorized_by_this_report": False}
    out = None
    try:
        ctx = m.context(args)
        if args.command == "qualify":
            out = m.new_output(ctx, args.out)
            qualify(args, ctx, out, report)
        elif args.command == "plan-full":
            m.require_qualification(ctx, args.qualification_result)
            out = m.new_output(ctx, args.out)
            plan = make_plan(ctx, "full")
            m.write_json(out / "plan.json", plan); m.verify_final(ctx)
            report.update({"schema": "third-e5-full-plan-build-v1", "passed": True, "stage": "complete", "plan_sha256": plan["sha256"]})
        else:
            m.require_qualification(ctx, args.qualification_result)
            plan = m.load_plan(ctx, args.plan)
            m.require(plan["kind"] == "full", "full_assembly_requires_full_plan")
            assemble(ctx, plan, args.encoded_dirs, args.replay_dirs, args.out)
            report.update({"schema": "third-e5-full-assembly-build-v1", "passed": True, "stage": "complete"})
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    report["driver_seconds_through_evidence_finalization"] = time.perf_counter() - start
    report["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if out is not None:
        m.atomic_completion(out / "result.json", m.seal(report))
    print(json.dumps({key: report[key] for key in ("passed", "stage", "error_type", "error") if key in report}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
