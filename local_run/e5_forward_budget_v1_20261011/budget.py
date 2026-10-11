#!/usr/bin/env python3
"""Bounded, source-bound CPU E5 timing qualification; never a full cache run."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import statistics
import sys
import time

# Check the already sealed helper before executing it. Its entire package is
# checked again by run(), and all imported frozen encoder files are pinned.
HELPER_SHA = "24158f2b32a8324b47f1c4c304e72fd79dcda31401438864476fc85e4198956c"
HELPER_PACKAGE_SHA = "205a8c7d254cc76e7c1cd3c3c4064b2c56a7311776d6cc6ba99ee38e7125b6fb"
helper_path = Path(__file__).resolve().parent.parent / "e5_token_census_v1_20261011/census.py"
if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError("sealed_census_helper_changed")
spec = importlib.util.spec_from_file_location("e5_budget_census_helper", helper_path)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
require = c.require
SALT = "third-e5-forward-budget-v1-20261011"
PASSES = ("baseline", "repeat", "reverse")


def selection_key(record_id):
    return hashlib.sha256((SALT + "\0" + record_id).encode("utf-8")).hexdigest(), record_id


def select_ledger(rows, aggregate):
    """Validate full ledger identity/counts and retain only 16 natural records."""
    seen, ids, partitions, chunks_hist = set(), c.ListDigest(), Counter(), Counter()
    totals, batches, candidates = Counter(), Counter(), Counter()
    selected = defaultdict(list)
    for index, row in enumerate(rows):
        rid = row.get("record_id")
        require(isinstance(rid, str) and rid and rid not in seen, "duplicate_or_missing_census_record")
        require(row.get("row_index") == index, "census_row_order_mismatch")
        require(row.get("partition") in ("train", "calibration", "test"), "invalid_census_partition")
        detail = row["detail"]
        n, k = detail["content_tokens"], detail["chunks"]
        require(type(n) is int and n > 0 and type(k) is int and k == (n + 255) // 256,
                "invalid_census_chunk_count")
        lengths = [min(256, n - start) for start in range(0, n, 256)]
        require(detail["chunk_content_tokens"] == lengths
                and detail["prefix_tokens_per_chunk"] == 3 and detail["special_tokens_per_chunk"] == 2,
                "invalid_census_chunk_coverage")
        expected_cost = {"batch_count": (k + 7) // 8, "full_batches_of_8": k // 8,
            "tail_batch_size": k % 8, "padded_token_slots": k * 261,
            "attended_token_slots": n + 5 * k, "padding_token_slots": k * 256 - n,
            "maximum_batch_size": min(k, 8), "maximum_batch_padded_token_slots": min(k, 8) * 261}
        require(all(row.get(key) == value for key, value in expected_cost.items()), "census_cost_mismatch")
        seen.add(rid); ids.add(rid)
        partitions[row["partition"]] += 1
        chunks_hist[k] += 1
        totals.update({key: row[key] for key in ("batch_count", "full_batches_of_8", "padded_token_slots", "attended_token_slots", "padding_token_slots")})
        totals.update({"rows": 1, "content_tokens": n, "chunks": k})
        batches[8] += k // 8
        if k % 8:
            batches[k % 8] += 1
        if row["partition"] == "calibration" and 1 <= k <= 8:
            candidates[k] += 1
            selected[k].append(row)
            selected[k].sort(key=lambda item: selection_key(item["record_id"]))
            del selected[k][2:]
    require(all(totals[key] == value for key, value in aggregate["totals"].items()
                if key in totals), "census_aggregate_totals_mismatch")
    require(totals["rows"] == aggregate["totals"]["rows"]
            and dict(partitions) == aggregate["partition_counts"]
            and {str(k): v for k, v in chunks_hist.items()} == aggregate["chunks_per_record_histogram"]
            and ids.finish() == aggregate["ordered_record_ids_sha256"], "census_population_identity_mismatch")
    expected_batches = {int(k): v for k, v in aggregate["tail_batch_size_histogram_excluding_zero"].items()}
    expected_batches[8] = aggregate["totals"]["full_batches_of_8"]
    require(dict(batches) == expected_batches, "census_batch_distribution_mismatch")
    require(all(len(selected[k]) == 2 for k in range(1, 9)), "two_calibration_records_per_bucket_required")
    return [row for k in range(1, 9) for row in selected[k]], dict(candidates), dict(batches)


def bind_source(rows, selected, aggregate):
    complete, texts, ids = c.ListDigest(), c.ListDigest(), c.ListDigest()
    seen, found, partitions = set(), {}, Counter()
    lookup = {row["record_id"]: row for row in selected}
    require(len(lookup) == 16, "exactly_16_distinct_samples_required")
    for index, row in enumerate(rows):
        rid, text = row.get("record_id"), row.get("text")
        require(isinstance(rid, str) and rid and rid not in seen, "duplicate_or_missing_source_record")
        require(isinstance(text, str) and text, "empty_source_text")
        require(row.get("partition") in ("train", "calibration", "test"), "invalid_source_partition")
        seen.add(rid); complete.add(row); texts.add({"record_id": rid, "text": text}); ids.add(rid)
        partitions[row["partition"]] += 1
        if rid in lookup:
            expected = lookup[rid]
            require(index == expected["row_index"] and row["partition"] == "calibration"
                    and hashlib.sha256(text.encode("utf-8")).hexdigest() == expected["text_sha256"],
                    "selected_source_census_binding_mismatch")
            found[rid] = {**expected, "text": text}
    require(len(seen) == aggregate["totals"]["rows"] and dict(partitions) == aggregate["partition_counts"]
            and complete.finish() == aggregate["full_prepared_rows_sha256"]
            and texts.finish() == aggregate["text_binding"] and ids.finish() == aggregate["ordered_record_ids_sha256"],
            "complete_source_frame_mismatch")
    require(set(found) == set(lookup), "selected_source_record_missing")
    return [found[row["record_id"]] for row in selected]


def validate_census(report, aggregate, pins):
    require(report.get("schema") == "third-e5-token-census-result-v1" and report.get("passed") is True
            and report.get("stage") == "complete" and report.get("initial_final_source_code_and_assets_equal") is True,
            "passed_immutable_census_required")
    require(report.get("rows") == pins["rows"] and report.get("model_forwards") == 0
            and report.get("replay_result_sha256") == pins["replay_result_sha256"], "census_scope_or_replay_mismatch")
    for name in ("source_authenticity_verified", "primary_input_acceptance", "full_population_cache_acceptance", "confirmatory_study_ready"):
        require(report.get(name) is False, "census_claim_boundary_changed")
    require(aggregate["totals"] == pins["census_totals"]
            and aggregate["ordered_record_ids_sha256"] == pins["ordered_ids_sha256"]
            and aggregate["full_prepared_rows_sha256"] == pins["full_prepared_rows_sha256"]
            and aggregate["text_binding"] == pins["text_binding"], "census_full_frame_binding_mismatch")


def validate_detail(actual, expected):
    require(all(actual.get(key) == value for key, value in expected["detail"].items()),
            "selected_chunk_or_token_ids_differ_from_census")
    require(actual.get("batch_count") == expected["batch_count"] == 1, "selected_forward_count_mismatch")


def projections(observations, documents, aggregate, cold_load_seconds, fixed_seconds):
    """Transparent shape-weighted planning scenarios, not statistical bounds."""
    timings = defaultdict(list)
    for item in observations:
        value = item["model_forward_seconds"]
        require(math.isfinite(value) and value > 0, "invalid_forward_timing")
        timings[item["shape"][0]].append(value)
    require(set(timings) == set(range(1, 9)) and all(len(v) == 6 for v in timings.values()),
            "six_forward_measurements_per_batch_size_required")
    costs = {int(k): v for k, v in aggregate["tail_batch_size_histogram_excluding_zero"].items()}
    costs[8] = aggregate["totals"]["full_batches_of_8"]
    require(sum(costs.values()) == aggregate["totals"]["batch_count"]
            and sum(k * n for k, n in costs.items()) == aggregate["totals"]["chunks"], "projection_cost_population_mismatch")
    overhead = []
    for row in documents:
        seconds = row["encode_document_seconds"] - row["model_forward_seconds"]
        require(math.isfinite(seconds) and seconds >= 0 and row["chunks"] in range(1, 9), "invalid_nonforward_timing")
        overhead.append(seconds / row["chunks"])
    require(len(overhead) == 48, "complete_document_timing_required")
    scenarios = {}
    for name, fn in (("observed_min_scenario", min), ("observed_median_scenario", statistics.median), ("observed_max_scenario", max)):
        forward = sum(costs[k] * fn(timings[k]) for k in range(1, 9))
        other = aggregate["totals"]["chunks"] * fn(overhead)
        one = forward + other + cold_load_seconds + fixed_seconds
        scenarios[name] = {"transformer_forward_seconds_per_pass": forward,
            "instrumented_nonforward_seconds_per_pass_extrapolated_per_chunk": other,
            "cold_model_load_seconds_per_pass": cold_load_seconds,
            "observed_wrapper_input_scan_checks_and_output_finalization_seconds_per_pass": fixed_seconds,
            "one_full_pass_modeled_seconds": one, "two_separate_full_passes_modeled_seconds": 2 * one,
            "one_full_pass_modeled_hours": one / 3600, "two_separate_full_passes_modeled_hours": 2 * one / 3600}
    return {"schema": "third-e5-forward-budget-projection-v1", "batch_shape_counts": costs,
        "model_forward_seconds_by_batch_size": {k: {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v), "observations": v} for k, v in sorted(timings.items())},
        "scenarios": scenarios, "guaranteed_walltime_bound": False, "confidence_interval": False,
        "full_encoding_launch_authorized_by_this_report": False,
        "scope": "Shape-weighted model forward estimate plus observed instrumented nonforward cost per chunk, cold load and measured qualification fixed costs; two passes assumes two separate identical full passes.",
        "limitations": ["Only two calibration documents per batch size, repeated three times in one fresh process shared by all eight batch sizes on this node; CPU contention, frequency, node and run variability are unmeasured.",
            "Documents above eight chunks are extrapolated as their exact within-record size-eight batches plus remainder; no long document was encoded in this bounded test.",
            "Per-chunk nonforward extrapolation includes timing checks and duplicate padding but approximates per-document costs; it is not a statistically fitted scaling model.",
            "Full-cache serialization, calibration gathers, extra replay acceptance checks, and future filesystem contention are not measured. Qualification fixed costs cannot guarantee their coverage.",
            "Minimum and maximum are observed sample scenarios, not confidence intervals or safe scheduler limits; no full nonresumable job should launch solely on a point estimate."]}


def run(args, report):
    repo, workspace, view, census_dir, out = (Path(getattr(args, name)).resolve() for name in ("repo", "workspace", "view", "census_dir", "out"))
    require(platform.system() == "Linux" and os.environ.get("SLURM_JOB_ID"), "linux_slurm_allocation_required")
    require(socket.gethostname().split(".")[:2] != ["csis", "mn1"], "login_node_prohibited")
    require(out.is_relative_to(workspace / "qualification") and not out.exists(), "new_private_output_required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.umask(0o077); out.mkdir(parents=True, exist_ok=False)
    (out / ".gitignore").write_text("*\n")
    report["out"] = str(out)
    phases, watch = {}, c.Watch()
    report["phase_seconds"] = phases
    start = time.perf_counter()
    folder = Path(__file__).resolve().parent
    c.verify_package(folder, args.package_sha256, watch)
    c.verify_package(helper_path.parent, HELPER_PACKAGE_SHA, watch)
    pins = c.read_json(folder / "expectations.json")
    for name, expected in c.read_json(folder / "frozen_code.json")["files"].items():
        watch.add(c.inside(repo / name, repo), expected)
    replay_path = c.inside(args.replay_result, workspace)
    watch.add(replay_path, {"sha256": pins["replay_result_sha256"]})
    replay_stages = c.validate_replay(c.read_json(replay_path))
    for name, expected in pins["census_files"].items():
        watch.add(c.inside(census_dir / name, workspace), expected)
    census_report = c.read_json(census_dir / "result.json")
    aggregate = c.read_json(census_dir / "aggregate.json")
    validate_census(census_report, aggregate, pins)
    for name, result_key in (("aggregate.json", "aggregate"), ("row_token_chunk_ledger.private.jsonl", "ledger"), ("input_bindings.json", "input_bindings")):
        require(census_report[result_key] == pins["census_files"][name], "census_receipt_binding_mismatch")
    census_bound = c.read_json(census_dir / "input_bindings.json")["files"]
    replay_bound = replay_stages["final_immutability"]["bindings"]
    for name, expected in pins["view"].items():
        path = c.inside(view / name, workspace)
        require(census_bound.get(str(path)) == expected and replay_bound.get(str(path)) == expected,
                "view_not_bound_by_census_and_replay")
        watch.add(path, expected)
    sys.path.insert(0, str(repo / "empirical_execution"))
    from phase4 import embeddings as e
    from phase3 import calibration as cal
    import numpy as np
    audit = c.read_json(view / "audit.json")
    cal._verify(audit)
    require(audit["dataset_id"] == "askubuntu" and audit["sha256"] == replay_stages["prepared_ascii_view"]["derived_audit_seal"]
            and audit["prepared_rows_sha256"] == aggregate["full_prepared_rows_sha256"]
            and audit["text_binding"] == aggregate["text_binding"], "source_audit_identity_mismatch")
    require(platform.python_version() == pins["python"] and e.installed_versions() == pins["library_versions"], "pinned_CPU_runtime_mismatch")
    model_dir, tokenizer_dir = workspace / "models/e5/model", workspace / "models/e5/tokenizer"
    assets_path = c.inside(workspace / "models/e5/assets.json", workspace)
    watch.add(assets_path, pins["assets_file"])
    require(census_bound.get(str(assets_path)) == pins["assets_file"], "assets_not_bound_by_passed_census")
    assets = c.read_json(assets_path)
    require(assets.get("sha256") == pins["asset_seal_sha256"] and assets.get("encoder_id") == e.E5
            and assets.get("encoder_revision") == pins["revision"] and assets.get("tokenizer_revision") == pins["revision"], "asset_identity_mismatch")
    e.verify_assets(assets, model_dir, tokenizer_dir, verify_runtime=True)
    require([x for x in assets["model_files"] if x["path"] == "model.safetensors"] == [pins["model_weight"]], "model_weight_binding_mismatch")
    phases["initial_pinned_input_code_assets_runtime_verification"] = time.perf_counter() - start
    report["runtime"] = {"python": platform.python_version(), "libraries": e.installed_versions(), "executable": sys.executable,
        "platform": platform.platform(), "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"]}
    report["stage"] = "full_ledger_selection_and_source_frame_scan"
    start = time.perf_counter()
    selected, candidates, batches = select_ledger(c.physical_rows(census_dir / "row_token_chunk_ledger.private.jsonl"), aggregate)
    samples = bind_source(c.physical_rows(view / "records.jsonl"), selected, aggregate)
    phases["complete_ledger_and_source_frame_verification"] = time.perf_counter() - start
    planned_chunks = sum(row["detail"]["chunks"] for row in samples) * 3
    require(len(samples) == 16 and planned_chunks == 216, "bounded_workload_mismatch")
    manifest = {"schema": "third-e5-forward-budget-sample-v1", "selection_salt": SALT,
        "selection_rule": "Two lowest SHA256(salt + NUL + record_id), tie-broken by record_id, among calibration rows for each exact chunk count 1 through 8; output bucket order then hash order.",
        "candidate_counts": candidates, "samples": selected, "passes": PASSES,
        "complete_source_frame_rechecked": True, "full_prepared_rows_sha256": aggregate["full_prepared_rows_sha256"],
        "ordered_record_ids_sha256": aggregate["ordered_record_ids_sha256"], "text_binding": aggregate["text_binding"],
        "census_result": pins["census_files"]["result.json"], "census_ledger": pins["census_files"]["row_token_chunk_ledger.private.jsonl"],
        "planned_document_encodings": 48, "planned_model_forwards": 48, "planned_chunks": planned_chunks,
        "planned_padded_token_slots": planned_chunks * 261, "source_authenticity_verified": False,
        "package_manifest": c.binding(folder / "package_manifest.json"), "asset_seal_sha256": assets["sha256"]}
    c.write_json(out / "sample_manifest.json", manifest)
    watch.add(out / "sample_manifest.json")
    report["stage"] = "cold_frozen_model_load"
    start = time.perf_counter()
    adapter = e.LocalTransformer(model_dir, tokenizer_dir, e.E5, threads=1)
    torch = adapter.torch
    require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in adapter.model.parameters()), "CPU_FP32_parameters_required")
    require(not adapter.model.training and not torch.backends.mkldnn.enabled
            and torch.get_num_threads() == torch.get_num_interop_threads() == 1
            and torch.are_deterministic_algorithms_enabled(), "frozen_CPU_runtime_contract_mismatch")
    require(getattr(adapter.model.config, "_attn_implementation", None) == "eager"
            and adapter.max_length == pins["max_length"] and adapter.input_pad_length == pins["padding_tokens"], "attention_or_padding_contract_mismatch")
    phases["cold_model_and_tokenizer_load_including_runtime_checks"] = time.perf_counter() - start
    # Freeze complete selected tokens/chunks before the first model call.
    start = time.perf_counter()
    sample_chunks = []
    token_ledger = out / "selected_tokens.private.jsonl"
    with token_ledger.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(samples):
            chunks, detail = e.chunk_inputs(adapter.tokenizer, row["text"], e.E5, adapter.max_length)
            require(detail == row["detail"] and c.row_cost(chunks, detail, adapter.input_pad_length) == {key: row[key] for key in c.row_cost(chunks, detail, adapter.input_pad_length)}, "selected_token_or_cost_mismatch")
            content = adapter.tokenizer.encode(row["text"], add_special_tokens=False, truncation=False)
            require(cal.digest(content) == detail["content_token_ids_sha256"], "selected_content_token_ids_mismatch")
            sample_chunks.append(chunks)
            stream.write(json.dumps({"sample_index": index, "record_id": row["record_id"], "text_sha256": row["text_sha256"],
                "detail": detail, "content_token_ids": content, "chunks": chunks}, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    watch.add(token_ledger)
    phases["pre_forward_token_and_chunk_ledger_validation_and_durable_write"] = time.perf_counter() - start
    report["stage"] = "bounded_three_pass_encoding"
    observations, documents, current, all_vectors, pass_summaries = [], [], {}, {}, []

    def pre_hook(module, positional, keywords):
        ids, mask = keywords["input_ids"], keywords["attention_mask"]
        require(ids.device.type == mask.device.type == "cpu" and tuple(ids.shape) == tuple(mask.shape)
                and tuple(ids.shape) == (current["chunks"], 261), "actual_forward_shape_or_device_mismatch")
        require(torch.equal(ids, current["expected_ids"]) and torch.equal(mask, current["expected_mask"]), "actual_forward_token_ids_or_mask_mismatch")
        current["observation"] = {"pass": current["pass"], "sample_index": current["sample_index"],
            "record_id": current["record_id"], "shape": list(ids.shape), "input_device": str(ids.device),
            "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            "attention_mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
            "attended_tokens": mask.sum(dim=1).tolist()}
        current["forward_started"] = time.perf_counter()

    def post_hook(module, positional, keywords, output):
        stopped = time.perf_counter()
        item = current.pop("observation")
        item["model_forward_seconds"] = stopped - current.pop("forward_started")
        require(output.last_hidden_state.device.type == "cpu" and output.last_hidden_state.dtype == torch.float32,
                "actual_forward_output_not_CPU_FP32")
        observations.append(item)

    handles = [adapter.model.register_forward_pre_hook(pre_hook, with_kwargs=True),
               adapter.model.register_forward_hook(post_hook, with_kwargs=True)]
    start = time.perf_counter()
    try:
        for pass_name in PASSES:
            indices = list(range(16))
            if pass_name == "reverse":
                indices.reverse()
            vectors = np.empty((16, e.DIMENSION), dtype=np.float32)
            pass_started = time.perf_counter()
            for index in indices:
                row = samples[index]
                current.update({"pass": pass_name, "sample_index": index, "record_id": row["record_id"], "chunks": row["detail"]["chunks"]})
                backend_seconds, calls, before = [], [], len(observations)

                def checked_backend(parts):
                    require(parts == sample_chunks[index], "frozen_encoder_chunks_changed_since_pre_forward_ledger")
                    padded = adapter.tokenizer.pad([{"input_ids": p["input_ids"], "attention_mask": p["attention_mask"]} for p in parts],
                        padding="max_length", max_length=adapter.input_pad_length, return_tensors="pt")
                    current["expected_ids"], current["expected_mask"] = padded["input_ids"], padded["attention_mask"]
                    backend_started = time.perf_counter()
                    means = adapter(parts)
                    backend_seconds.append(time.perf_counter() - backend_started)
                    require(means.dtype == np.float32 and means.shape == (len(parts), e.DIMENSION)
                            and np.isfinite(means).all(), "invalid_frozen_backend_means")
                    calls.append(1)
                    return means

                doc_started = time.perf_counter()
                vector, detail = e.encode_document(adapter.tokenizer, checked_backend, row["text"], e.E5, adapter.max_length, batch_size=8)
                doc_seconds = time.perf_counter() - doc_started
                validate_detail(detail, row)
                require(len(calls) == len(observations) - before == 1, "exactly_one_forward_per_selected_document_required")
                require(vector.dtype == np.float32 and vector.shape == (768,) and np.isfinite(vector).all(), "invalid_document_vector")
                vectors[index] = vector
                documents.append({"pass": pass_name, "sample_index": index, "record_id": row["record_id"],
                    "chunks": detail["chunks"], "encode_document_seconds": doc_seconds,
                    "backend_including_padding_and_masked_mean_seconds": sum(backend_seconds),
                    "model_forward_seconds": observations[-1]["model_forward_seconds"],
                    "detail": detail, "vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest()})
            with (out / ("vectors_" + pass_name + ".npy")).open("xb") as stream:
                np.save(stream, vectors, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            all_vectors[pass_name] = vectors
            pass_summaries.append({"pass": pass_name, "document_order": indices, "seconds_including_vector_durable_write": time.perf_counter() - pass_started})
    finally:
        for handle in handles:
            handle.remove()
    phases["three_passes_including_vector_durable_writes"] = time.perf_counter() - start
    require(len(observations) == 48 and sum(item["shape"][0] for item in observations) == 216, "executed_workload_mismatch")
    comparisons = {name: all_vectors["baseline"].tobytes() == all_vectors[name].tobytes() for name in PASSES[1:]}
    require(all(comparisons.values()), "repeat_or_reverse_vector_bytes_differ")
    reference = {row["sample_index"]: row for row in observations if row["pass"] == "baseline"}
    require(all(all(row[key] == reference[row["sample_index"]][key] for key in ("shape", "input_ids_sha256", "attention_mask_sha256", "attended_tokens")) for row in observations), "repeat_or_reverse_actual_tokens_differ")
    report["stage"] = "finalization"
    start = time.perf_counter()
    c.write_json(out / "actual_forward_timings.json", observations)
    c.write_json(out / "document_timings.json", documents)
    e.verify_assets(assets, model_dir, tokenizer_dir, verify_runtime=True)
    watch.verify()
    require(e.installed_versions() == pins["library_versions"], "runtime_changed_during_qualification")
    c.write_json(out / "input_bindings.json", {"files": {str(path): value for path, value in watch.files.items()},
        "assets_full_inventory_verified_initial_and_final": True, "assets_seal_sha256": assets["sha256"]})
    report["artifacts"] = {p.name: c.binding(p) for p in sorted(out.iterdir()) if p.is_file()}
    phases["timing_and_binding_writes_final_input_asset_checks_and_output_hashes"] = time.perf_counter() - start
    fixed = sum(phases[key] for key in ("initial_pinned_input_code_assets_runtime_verification", "complete_ledger_and_source_frame_verification",
        "timing_and_binding_writes_final_input_asset_checks_and_output_hashes"))
    start = time.perf_counter()
    projection = projections(observations, documents, aggregate, phases["cold_model_and_tokenizer_load_including_runtime_checks"], fixed)
    c.write_json(out / "budget_projection.json", projection)
    report["artifacts"]["budget_projection.json"] = c.binding(out / "budget_projection.json")
    phases["projection_generation_durable_write_and_hash"] = time.perf_counter() - start
    report.update({"passed": True, "stage": "complete", "selected_documents": 16, "document_encodings": 48,
        "model_forwards": 48, "chunks_processed": 216, "padded_token_slots": 56376,
        "exact_vector_byte_comparisons": comparisons, "actual_token_ids_masks_and_shapes_equal_across_passes": True,
        "selected_chunk_and_content_token_id_digests_equal_passed_census": True,
        "full_source_frame_verified": True, "initial_final_input_code_assets_equal": True,
        "model_parameter_bytes": adapter.model_bytes, "model_buffer_bytes": adapter.model_buffer_bytes,
        "passes": pass_summaries, "census_result_sha256": pins["census_files"]["result.json"]["sha256"],
        "replay_result_sha256": pins["replay_result_sha256"]})


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "workspace", "view", "census-dir", "replay-result", "out", "package-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    report = {"schema": "third-e5-forward-budget-result-v1", "passed": False, "stage": "preflight",
        "evidence_role": "bounded_CPU_throughput_planning_and_same_process_byte_replay_only",
        "downloads_performed": False, "source_authenticity_verified": False, "primary_input_acceptance": False,
        "complete_population_cache_created": False, "full_population_cache_acceptance": False,
        "confirmatory_study_ready": False, "cross_machine_bit_identity_claimed": False,
        "full_encoding_launch_authorized_by_this_report": False}
    try:
        run(args, report)
    except Exception as error:
        report["error"] = str(error) if isinstance(error, c.Refusal) else type(error).__name__
    report["driver_seconds_through_all_artifact_finalization"] = time.perf_counter() - started
    report["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["timing_boundary"] = "Driver includes source scans, cold load, full selected encodings, durable vector/timing/token-ledger writes, final input and asset hashing, projection write and artifact hashes. Slurm envelope additionally measures this final result write and process exit. Inner forward/backend/document times overlap and must not be summed."
    if "out" in report:
        c.write_json(Path(report["out"]) / "result.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "stage")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
