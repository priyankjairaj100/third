#!/usr/bin/env python3
"""Bounded real-text check of the unchanged frozen CPU FP32 encoder.

This creates private runtime evidence, not a complete-population cache. Run one
fresh Slurm process per encoder. It never downloads, re-inventories assets,
changes source audits, or substitutes a different encoder implementation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback


SALT = "third-cpu-runtime-qualification-v1-20261011"
PUBLISHED = {
    "e5": {
        "encoder_id": "intfloat/multilingual-e5-base",
        "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
        "weights_sha256": "a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7",
        "weights_bytes": 1112201288,
    },
    "mpnet": {
        "encoder_id": "sentence-transformers/all-mpnet-base-v2",
        "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "weights_sha256": "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658",
        "weights_bytes": 437971872,
    },
}
SOURCE_FILES = (
    "empirical_execution/phase4/embeddings.py",
    "empirical_execution/phase3/calibration.py",
    "empirical_execution/phase3/panels.py",
    "empirical_execution/phase3/run_preparation.py",
    "empirical_execution/phase4/news_calendar.py",
    "empirical_execution/phase4/adapters.py",
)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path)}


def sample_sources(e, workspace, source_specs):
    selected, sources = [], []
    for dataset, records_path, audit_path in source_specs:
        records_path, audit_path = Path(records_path), Path(audit_path)
        # Verification is over the entire existing file and its original audit.
        # In particular, never synthesize a reduced-population preparation audit.
        before = {"records": binding(records_path), "audit_file": binding(audit_path)}
        rows, audit = e.load_prepared_records(records_path, audit_path)
        require(audit.get("dataset_id") == dataset, "Prepared dataset ID mismatch")
        row_count = len(rows)
        calibration_count = sum(r["partition"] == "calibration" for r in rows)
        source_info = {"dataset": dataset, **before, "prepared_rows": row_count,
                       "calibration_rows": calibration_count,
                       "preparation_audit_seal_sha256": audit["sha256"],
                       "full_prepared_record_validation_passed": True,
                       "buckets": {}}
        for bucket, lower, upper in (("short", 1, 1600), ("long", 12000, 24000)):
            candidates = []
            for index, row in enumerate(rows):
                if row["partition"] != "calibration" or not lower <= len(row["text"]) <= upper:
                    continue
                key = hashlib.sha256(
                    f"{SALT}|{dataset}|{bucket}|{row['record_id']}".encode("utf-8")).hexdigest()
                candidates.append((key, str(row["record_id"]), index))
            source_info["buckets"][bucket] = {"character_min": lower, "character_max": upper,
                                              "eligible_rows": len(candidates), "selected_rows": 2}
            require(len(candidates) >= 2, f"{dataset}/{bucket}: fewer than two eligible calibration texts")
            for key, _, index in sorted(candidates)[:2]:
                row = rows[index]
                selected.append({"dataset": dataset, "bucket": bucket,
                                 "source_row_index": index, "selection_hash": key,
                                 "record_id": row["record_id"],
                                 "source_unit_id": row["source_unit_id"],
                                 "partition": row["partition"], "characters": len(row["text"]),
                                 "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                                 "text": row["text"]})
            del candidates
        sources.append(source_info)
        del rows, audit, row
        gc.collect()
    require(2 <= len(selected) <= 8, "Expected two to eight bounded qualification texts")
    require(len({(r["dataset"], r["record_id"]) for r in selected}) == len(selected),
            "Qualification sample IDs must be unique")
    return selected, sources


def run(args, out, progress):
    start = time.perf_counter()
    repo, workspace = Path(args.repo).resolve(), Path(args.workspace).resolve()
    require(platform.system() == "Linux", "Qualification is specified for Linux CPU Slurm")
    require(bool(os.environ.get("SLURM_JOB_ID")), "Run this numerical workload through Slurm")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, str(repo / "empirical_execution"))
    import numpy as np
    from phase4 import embeddings as e

    progress["stage"] = "source_and_asset_validation"
    expected = PUBLISHED[args.encoder]
    model_dir = workspace / "models" / args.encoder / "model"
    tokenizer_dir = workspace / "models" / args.encoder / "tokenizer"
    asset_path = Path(args.asset_manifest) if args.asset_manifest else workspace / "models" / args.encoder / "assets.json"
    code_bindings = {rel: binding(repo / rel) for rel in SOURCE_FILES}
    code_bindings["qualification_driver"] = binding(__file__)
    asset_binding = binding(asset_path)
    assets = e.verify_assets(e.read_json(asset_path), model_dir, tokenizer_dir)
    require(assets["encoder_id"] == expected["encoder_id"], "Wrong encoder asset identity")
    require(assets["encoder_revision"] == expected["revision"] and
            assets["tokenizer_revision"] == expected["revision"], "Wrong immutable publisher revision")
    weight = [f for f in assets["model_files"] if f["path"] == "model.safetensors"]
    require(len(weight) == 1 and weight[0]["sha256"] == expected["weights_sha256"] and
            weight[0]["bytes"] == expected["weights_bytes"], "Published weight hash/size mismatch")
    progress["published_weight_hash_matched"] = True
    if args.source:
        source_specs = args.source
    else:
        source_specs = [(d, workspace / "prepared" / d / "a02/population/records.jsonl",
                        workspace / "prepared" / d / "a02/population/audit.json")
                       for d in ("askubuntu", "cc_news")]
    require(len(source_specs) <= 2, "At most two sources keep this qualification bounded")
    require(len({s[0] for s in source_specs}) == len(source_specs), "Repeated source dataset")
    samples, source_bindings = sample_sources(e, workspace, source_specs)
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "schema": "third-cpu-runtime-qualification-sample-v1",
        "evidence_role": "engineering_nonconfirmatory_runtime_qualification",
        "population_scope": "bounded_calibration_text_sample_not_complete_population",
        "selection_salt": SALT, "selection_rule": "two lowest hash keys per source and character bucket",
        "selection_uses_model_outputs_or_labels": False,
        "created_before_any_model_forward": True,
        "sample": [{k: v for k, v in row.items() if k != "text"} for row in samples],
        "sources": source_bindings, "code": code_bindings,
        "repository_commit": commit, "asset_manifest": asset_binding,
        "asset_seal_sha256": assets["sha256"], "publisher_weight_identity": expected,
        "runtime_versions_observed": e.installed_versions(),
        "runtime_versions_match_existing_asset_manifest": True,
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "machine": platform.machine(), "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "full_population_cache_created": False, "full_population_cache_replay_passed": False,
        "external_source_authenticity_established": False, "confirmatory_study_ready": False,
        "mpnet_scope": "runtime test only; creates no non-Civil MPNet curator or human-calibration arm",
        "batch_size": 8, "threads": 1, "passes": ["baseline", "repeat", "reverse_interleaving"],
    }
    write_json(out / "sample_manifest.json", manifest)
    with (out / "sample_records.private.jsonl").open("x", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    progress["sample_manifest_sha256"] = sha_file(out / "sample_manifest.json")
    progress["stage"] = "model_load"
    model_load_start = time.perf_counter()
    adapter = e.LocalTransformer(model_dir, tokenizer_dir, expected["encoder_id"], threads=1)
    torch = adapter.torch
    require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in adapter.model.parameters()),
            "Model parameters are not CPU FP32")
    require(not adapter.model.training and not torch.backends.mkldnn.enabled,
            "Model eval/MKLDNN contract mismatch")
    require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1 and
            torch.are_deterministic_algorithms_enabled(), "Thread/determinism contract mismatch")
    require(getattr(adapter.model.config, "_attn_implementation", None) == "eager",
            "Expected eager attention")
    model_load_seconds = time.perf_counter() - model_load_start
    progress["stage"] = "three_encoding_passes"
    all_vectors = {}
    batch_observations = []
    current = {}

    def inspect_actual_forward(module, positional, keywords):
        ids, mask = keywords["input_ids"], keywords["attention_mask"]
        require(ids.device.type == "cpu" and mask.device.type == "cpu", "Forward inputs not on CPU")
        require(tuple(ids.shape) == tuple(mask.shape) and ids.shape[1] == adapter.input_pad_length,
                "Actual padding width/mask shape mismatch")
        require(torch.equal(ids, current["expected_ids"]) and torch.equal(mask, current["expected_mask"]),
                "Actual model inputs differ from frozen padded chunk inputs")
        batch_observations.append({"pass": current["pass"], "sample_index": current["sample_index"],
                                   "batch_ordinal": current["batch_ordinal"], "shape": list(ids.shape),
                                   "attended_tokens": mask.sum(dim=1).tolist(),
                                   "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                                   "attention_mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                                   "input_device": str(ids.device), "model_dtype": "float32"})

    handle = adapter.model.register_forward_pre_hook(inspect_actual_forward, with_kwargs=True)
    ledger_path = out / "chunk_ledger.private.jsonl"
    pass_summaries = []
    try:
        with ledger_path.open("x", encoding="utf-8") as ledger:
            for pass_name in ("baseline", "repeat", "reverse_interleaving"):
                indices = list(range(len(samples)))
                if pass_name == "reverse_interleaving":
                    indices.reverse()
                vectors = np.empty((len(samples), e.DIMENSION), dtype=np.float32)
                pass_start = time.perf_counter()
                for sample_index in indices:
                    row = samples[sample_index]
                    chunks, detail = e.chunk_inputs(adapter.tokenizer, row["text"], expected["encoder_id"], adapter.max_length)
                    content = adapter.tokenizer.encode(row["text"], add_special_tokens=False, truncation=False)
                    require(sum(c["content_tokens"] for c in chunks) == len(content), "Lost content tokens")
                    require([c["content_start"] for c in chunks] == list(range(0, len(content), 256)),
                            "Nonconsecutive chunk boundaries")
                    require(chunks[-1]["content_stop"] == len(content), "Incomplete final chunk")
                    if row["bucket"] == "long":
                        require(len(chunks) > 1, "Long-text sample did not exercise multichunk encoding")
                    current.update({"pass": pass_name, "sample_index": sample_index, "batch_ordinal": 0})

                    def checked_backend(parts):
                        padded = adapter.tokenizer.pad(
                            [{"input_ids": p["input_ids"], "attention_mask": p["attention_mask"]} for p in parts],
                            padding="max_length", max_length=adapter.input_pad_length, return_tensors="pt")
                        require(1 <= len(parts) <= 8, "Unexpected intradocument batch size")
                        require(padded["attention_mask"].sum(dim=1).tolist() == [len(p["input_ids"]) for p in parts],
                                "Padding changed attended token counts")
                        current["expected_ids"], current["expected_mask"] = padded["input_ids"], padded["attention_mask"]
                        means = adapter(parts)
                        require(means.dtype == np.float32 and means.shape == (len(parts), e.DIMENSION) and
                                np.isfinite(means).all(), "Invalid actual CPU FP32 masked means")
                        current["batch_ordinal"] += 1
                        return means

                    vector, actual_detail = e.encode_document(adapter.tokenizer, checked_backend, row["text"],
                        expected["encoder_id"], adapter.max_length, batch_size=8)
                    require(all(actual_detail[k] == v for k, v in detail.items()), "Chunk audit changed on replay")
                    require(current["batch_ordinal"] == actual_detail["batch_count"], "Missing forward observations")
                    vectors[sample_index] = vector
                    ledger.write(json.dumps({"pass": pass_name, "sample_index": sample_index,
                        "dataset": row["dataset"], "record_id": row["record_id"],
                        "text_sha256": row["text_sha256"], "vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                        "vector_l2_float64": float(np.linalg.norm(vector.astype(np.float64))),
                        "detail": actual_detail, "content_token_ids": content, "chunks": chunks},
                        sort_keys=True, allow_nan=False) + "\n")
                with (out / f"vectors_{pass_name}.npy").open("xb") as f:
                    np.save(f, vectors, allow_pickle=False)
                all_vectors[pass_name] = vectors
                pass_summaries.append({"pass": pass_name, "document_order": indices,
                                       "seconds": time.perf_counter() - pass_start,
                                       "vectors_file": binding(out / f"vectors_{pass_name}.npy")})
    finally:
        handle.remove()
    baseline = all_vectors["baseline"]
    comparisons = {}
    for pass_name in ("repeat", "reverse_interleaving"):
        comparison = {"all_fp32_bytes_equal": baseline.tobytes() == all_vectors[pass_name].tobytes(),
                      "maximum_absolute_difference": float(np.max(np.abs(
                          baseline.astype(np.float64) - all_vectors[pass_name].astype(np.float64))))}
        comparisons[pass_name] = comparison
    write_json(out / "actual_forward_observations.json", batch_observations)
    write_json(out / "comparisons.json", comparisons)
    require(all(v["all_fp32_bytes_equal"] for v in comparisons.values()), "Repeated/interleaved FP32 bytes differ")
    progress["stage"] = "final_source_bindings"
    e.verify_assets(assets, model_dir, tokenizer_dir)
    require(binding(asset_path) == asset_binding, "Asset manifest changed during qualification")
    for source in source_bindings:
        for key in ("records", "audit_file"):
            require(binding(source[key]["path"]) == source[key], "Source/audit bytes changed during qualification")
    for item in code_bindings.values():
        require(binding(item["path"]) == item, "Encoding/qualification source changed during qualification")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"schema": "third-cpu-runtime-qualification-result-v1", "passed": True,
            "status": "completed", "evidence_role": manifest["evidence_role"],
            "encoder": args.encoder, "rows": len(samples), "model_forwards": len(batch_observations),
            "published_weight_hash_matched": True, "existing_asset_runtime_match": True,
            "actual_frozen_cpu_fp32_runtime_check_passed": True,
            "semantic_runtime_qualified_for_complete_study": False,
            "scope": "eight-or-fewer calibration texts; repeated and reversed replay within one fresh encoder process",
            "source_origin_independently_authenticated": False,
            "full_population_source_replay_passed": False, "full_population_cache_replay_passed": False,
            "complete_population_cache_created": False, "cross_machine_bit_identity_claimed": False,
            "genuine_human_responses_created": 0, "primary_study_jobs_executed": 0,
            "confirmatory_study_ready": False,
            "sample_manifest": binding(out / "sample_manifest.json"),
            "chunk_ledger": binding(ledger_path), "forward_observations": binding(out / "actual_forward_observations.json"),
            "comparisons": comparisons, "passes": pass_summaries,
            "fixed_padding_tokens": adapter.input_pad_length, "model_max_length": adapter.max_length,
            "model_parameter_bytes": adapter.model_bytes, "model_buffer_bytes": adapter.model_buffer_bytes,
            "model_load_seconds": model_load_seconds, "total_seconds": time.perf_counter() - start,
            "peak_process_rss_kib": rss,
            "rss_scope": "Linux process high-water mark, includes complete source validation and model execution",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": platform.node()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("repo", "workspace", "out"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--encoder", choices=tuple(PUBLISHED), required=True)
    parser.add_argument("--asset-manifest", help="Existing manifest only; never regenerated")
    parser.add_argument("--source", nargs=3, action="append", metavar=("DATASET", "RECORDS", "AUDIT"),
                        help="Existing complete prepared population; defaults to AskUbuntu and News a02")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / ".gitignore").write_text("*\n", encoding="utf-8")
    progress = {"stage": "startup", "encoder": args.encoder}
    try:
        result = run(args, out, progress)
        exit_code = 0
    except Exception as error:
        result = {"schema": "third-cpu-runtime-qualification-result-v1", "passed": False,
                  "status": "failed", "evidence_role": "engineering_nonconfirmatory_runtime_qualification",
                  **progress, "exception_type": type(error).__name__, "message": str(error),
                  "actual_frozen_cpu_fp32_runtime_check_passed": False,
                  "semantic_runtime_qualified_for_complete_study": False,
                  "full_population_cache_replay_passed": False, "confirmatory_study_ready": False,
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": platform.node()}
        with (out / "traceback.private.txt").open("x", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        write_json(out / "FAILED.json", result)
        exit_code = 1
    write_json(out / "result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
