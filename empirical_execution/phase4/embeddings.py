#!/usr/bin/env python3
"""Offline, hash-bound, record-local semantic encoding; never fetches a model.

The model adapter is optional. Numerical software fixtures may exercise pooling
without torch, but cannot create a semantic cache through the production CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "empirical_execution"))
from phase3 import calibration, panels, run_preparation

E5 = "intfloat/multilingual-e5-base"
MPNET = "sentence-transformers/all-mpnet-base-v2"
ENCODERS = {E5: "query: ", MPNET: ""}
SCHEMA = "ccu-local-semantic-encoding-v1"
ASSET_SCHEMA = "ccu-offline-encoder-assets-v1"
CONTENT_LIMIT = 256
DIMENSION = 768
REVISION = re.compile(r"[a-f0-9]{40}(?:[a-f0-9]{24})?\Z")
VERSIONS = ("numpy", "torch", "transformers", "tokenizers", "safetensors")


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            value.update(data)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def local_inventory(directory):
    """Bind every local file; forbid symlinks, injected scripts, and external paths."""
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Model/tokenizer directory must be an existing local nonsymlink directory")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Materialize model/tokenizer symlinks into local regular files first")
        if path.is_file():
            if path.suffix in {".py", ".pyc"}:
                raise ValueError("Custom model/tokenizer scripts are not permitted")
            files.append({"path": path.relative_to(directory).as_posix(),
                          "bytes": path.stat().st_size, "sha256": file_hash(path)})
    if not files:
        raise ValueError("Local model/tokenizer directory is empty")
    return files


def installed_versions():
    versions = {}
    for name in VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def asset_inventory(model_dir, tokenizer_dir, encoder_id, encoder_revision,
                    tokenizer_revision, expected_versions):
    if encoder_id not in ENCODERS:
        raise ValueError("Only the two protocol encoders are supported")
    if not all(isinstance(x, str) and REVISION.fullmatch(x)
               for x in (encoder_revision, tokenizer_revision)):
        raise ValueError("Supply immutable full model and tokenizer revision identifiers")
    if (set(expected_versions) != set(VERSIONS)
            or any(not isinstance(v, str) or not v for v in expected_versions.values())):
        raise ValueError("Pin exact versions for numpy, torch, transformers, tokenizers and safetensors")
    return calibration._seal({"schema": ASSET_SCHEMA, "encoder_id": encoder_id,
        "encoder_revision": encoder_revision, "tokenizer_revision": tokenizer_revision,
        "model_files": local_inventory(model_dir), "tokenizer_files": local_inventory(tokenizer_dir),
        "library_versions": expected_versions,
        "revision_to_publisher_provenance_verified": False,
        "provenance_note": "Local bytes are bound; revision declarations require independent snapshot provenance review."})


def verify_assets(manifest, model_dir, tokenizer_dir, *, verify_runtime=True):
    calibration._verify(manifest)
    if manifest.get("schema") != ASSET_SCHEMA:
        raise ValueError("Wrong asset manifest schema")
    expected = asset_inventory(model_dir, tokenizer_dir, manifest["encoder_id"],
        manifest["encoder_revision"], manifest["tokenizer_revision"], manifest["library_versions"])
    for key in ("model_files", "tokenizer_files"):
        if manifest.get(key) != expected[key]:
            raise ValueError("Local byte inventory differs from the pinned " + key)
    model_names = {x["path"] for x in manifest["model_files"]}
    if "config.json" not in model_names or not any(p.endswith(".safetensors") for p in model_names):
        raise ValueError("A local model config and safetensors weights are required; no pickle fallback")
    for item in manifest["model_files"]:
        if item["path"].endswith(".safetensors.index.json"):
            shards = read_json(Path(model_dir) / item["path"]).get("weight_map", {}).values()
            if any(not isinstance(name, str) or name not in model_names
                   or Path(name).is_absolute() or ".." in Path(name).parts for name in shards):
                raise ValueError("Safetensors index references a file outside the bound local inventory")
    for directory in (model_dir, tokenizer_dir):
        for name in ("config.json", "tokenizer_config.json"):
            path = Path(directory) / name
            if path.exists() and read_json(path).get("auto_map"):
                raise ValueError("Custom auto_map implementations are not permitted")
            if path.exists():
                for key, value in read_json(path).items():
                    if key.endswith("_file") and isinstance(value, str):
                        target = Path(value)
                        if target.is_absolute() or ".." in target.parts or not (Path(directory) / target).is_file():
                            raise ValueError("Configuration refers to an unbound external asset file")
    if verify_runtime and installed_versions() != manifest["library_versions"]:
        raise RuntimeError("Pinned local runtime unavailable or mismatched: " +
                           json.dumps({"required": manifest["library_versions"], "installed": installed_versions()}))
    return manifest


def load_prepared_records(records_path, audit_path):
    rows = run_preparation.read_rows(records_path)
    audit = read_json(audit_path)
    calibration._verify(audit)
    if (audit.get("stage") != "fixed_guards_done_semantic_guard_pending"
            or audit.get("prepared_records_file_sha256") != file_hash(records_path)
            or audit.get("prepared_rows_sha256") != calibration.digest(rows)
            or audit.get("source_and_scoring_code_sha256") != file_hash(panels.__file__)):
        raise ValueError("Records are not bound to the fixed-guard preparation audit")
    news_prepared = audit.get("schema") == "ccu-news-preparation-1"
    if news_prepared:
        from phase4 import news_calendar, adapters
        if (audit.get("dataset_id") != "cc_news"
                or audit.get("news_preparation_code_sha256") != file_hash(news_calendar.__file__)
                or audit.get("adapter_code_sha256") != file_hash(adapters.__file__)
                or audit.get("normalized_records_sha256") != run_preparation.text_binding(rows)
                or audit.get("source_split_salt") != "ccu-v1-source-split"):
            raise ValueError("News records differ from the versioned date/source preparation")
    elif audit.get("preparation_code_sha256") != file_hash(run_preparation.__file__):
        raise ValueError("Records are not bound to the unchanged phase3 preparation code")
    ids = [r.get("record_id") for r in rows]
    if (not rows or any(not isinstance(v, str) or not v for v in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Nonempty, unique stable record IDs required")
    source_parts = {}
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text or text != panels.normalized_text(text):
            raise ValueError("Every text must already have protocol record-local normalization")
        source, part = row.get("source_unit_id"), row.get("partition")
        if not isinstance(source, str) or not source or part not in panels.PARTITIONS:
            raise ValueError("Prepared source IDs and partitions required")
        if source in source_parts and source_parts[source] != part:
            raise ValueError("A source appears in multiple partitions")
        source_parts[source] = part
        if news_prepared:
            from phase4.adapters import _date
            date = _date(row["original_fields"]["date"])
            derived_source, _ = panels.source_unit(row, "cc_news")
            expected_part = panels.digest_partition(panels.hash_integer(audit["source_split_salt"], source))
            correct_role = ((part == "calibration" and date.year == 2017 and row.get("news_cohort") == "calibration")
                            or (part == "train" and date.year in (2018, 2019) and row.get("news_cohort") == "analysis"))
            if source != derived_source or part != expected_part or not correct_role:
                raise ValueError("News source/date/cohort assignment is inconsistent")
    if news_prepared:
        caltexts = {r["text"] for r in rows if r["partition"] == "calibration"}
        if any(r["partition"] == "train" and r["text"] in caltexts for r in rows):
            raise ValueError("News fixed exact-text guard is incomplete")
    return rows, audit


def chunk_inputs(tokenizer, text, encoder_id, model_max_length):
    """Content tokenization happens once; add separately tokenized prefix per chunk.

    No decode/re-tokenize step or first-window truncation is allowed. This explicit
    token boundary convention is a declared study design, not an assertion that
    concatenate-then-tokenize is identical for every tokenizer.
    """
    if encoder_id not in ENCODERS:
        raise ValueError("Unknown protocol encoder")
    if not isinstance(model_max_length, int) or model_max_length <= 0:
        raise ValueError("Finite positive encoder sequence capacity required")
    content = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    prefix = tokenizer.encode(ENCODERS[encoder_id], add_special_tokens=False,
                              truncation=False) if ENCODERS[encoder_id] else []
    if not content:
        raise ValueError("Nonempty text produced no content tokens; report this exclusion explicitly")
    special = tokenizer.num_special_tokens_to_add(pair=False)
    if not isinstance(special, int) or special < 0:
        raise ValueError("Invalid special-token overhead")
    capacity = model_max_length - len(prefix) - special
    if capacity < CONTENT_LIMIT:
        raise ValueError("Pinned tokenizer/model cannot fit all 256 content tokens plus prefix/special overhead")
    chunks = []
    for start in range(0, len(content), CONTENT_LIMIT):
        piece = list(content[start:start + CONTENT_LIMIT])
        prepared = tokenizer.prepare_for_model(list(prefix) + piece, add_special_tokens=True,
            padding=False, truncation=False, return_attention_mask=True, return_token_type_ids=False)
        ids = list(prepared["input_ids"])
        if len(ids) != len(prefix) + len(piece) + special or len(ids) > model_max_length:
            raise ValueError("Tokenizer special-token accounting differs from the pinned contract")
        mask = list(prepared.get("attention_mask", []))
        if len(mask) != len(ids) or any(v != 1 for v in mask):
            raise ValueError("Unexpected token mask before explicit padding")
        chunks.append({"input_ids": ids, "attention_mask": mask,
                       "content_tokens": len(piece), "content_start": start,
                       "content_stop": start + len(piece)})
    details = {"content_tokens": len(content), "chunks": len(chunks),
        "chunk_content_tokens": [c["content_tokens"] for c in chunks],
        "prefix_tokens_per_chunk": len(prefix), "special_tokens_per_chunk": special,
        "maximum_encoded_chunk_tokens": max(len(c["input_ids"]) for c in chunks),
        "first_chunk_baseline_omitted_tokens": max(0, len(content) - CONTENT_LIMIT),
        "model_first_window_omitted_tokens": max(0, len(content) - capacity),
        "model_first_window_content_capacity": capacity,
        "content_token_ids_sha256": calibration.digest(list(content))}
    return chunks, details


def normalize_reference(vector):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Expected one finite vector")
    squared = 0.0
    for value in vector:
        squared += float(value) * float(value)
    if not math.isfinite(squared) or squared <= 0:
        raise ValueError("Zero or nonfinite embedding")
    return vector / math.sqrt(squared)


def encode_document(tokenizer, backend, text, encoder_id, max_length, *, batch_size=8,
                    dimension=DIMENSION):
    """Batches contain chunks of ONE document, preserving record-local context."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive chunk batch size required")
    chunks, detail = chunk_inputs(tokenizer, text, encoder_id, max_length)
    total = np.zeros(dimension, dtype=np.float64)
    for start in range(0, len(chunks), batch_size):
        part = chunks[start:start + batch_size]
        means = np.asarray(backend(part))
        if means.dtype != np.dtype("float32") or means.shape != (len(part), dimension):
            raise ValueError("Backend must return aligned FP32 masked means")
        for mean, chunk in zip(means, part):
            # Normalize chunk first, then content-count weight, in document order.
            total += normalize_reference(mean) * chunk["content_tokens"]
    result = normalize_reference(total / detail["content_tokens"]).astype(np.float32)
    if not np.isfinite(result).all() or not np.any(result):
        raise ValueError("Stored FP32 embedding is invalid")
    detail["batch_count"] = (len(chunks) + batch_size - 1) // batch_size
    return result, detail


class LocalTransformer:
    def __init__(self, model_dir, tokenizer_dir, encoder_id, *, threads=1):
        # Environment is set before imports and every loader also requires local files.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Offline semantic encoding needs already installed torch and transformers; no runtime download is attempted") from error
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("Positive CPU thread count required")
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise RuntimeError("Run encoding in a fresh process so CPU interop threads can be pinned") from error
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        torch.backends.mkldnn.enabled = False
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(Path(tokenizer_dir).resolve()),
            local_files_only=True, trust_remote_code=False, use_fast=True)
        self.model = AutoModel.from_pretrained(str(Path(model_dir).resolve()),
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.float32, attn_implementation="eager")
        self.model.to(device="cpu", dtype=torch.float32)
        self.model.eval()
        expected_model = "xlm-roberta" if encoder_id == E5 else "mpnet"
        if self.model.config.model_type != expected_model or self.model.config.hidden_size != DIMENSION:
            raise ValueError("Local model architecture does not match the declared protocol encoder")
        # The declared cap is conservative for both models. The tokenizer must
        # advertise at least this capacity; model position overhead is validated
        # by the actual model forward, never repaired by truncating input.
        self.max_length = min(512, int(self.tokenizer.model_max_length))
        if self.max_length < CONTENT_LIMIT + self.tokenizer.num_special_tokens_to_add(pair=False):
            raise ValueError("Tokenizer model capacity is too small for the study chunk rule")
        self.input_pad_length = CONTENT_LIMIT + self.tokenizer.num_special_tokens_to_add(pair=False)
        if ENCODERS[encoder_id]:
            self.input_pad_length += len(self.tokenizer.encode(ENCODERS[encoder_id], add_special_tokens=False))
        if self.input_pad_length > self.max_length:
            raise ValueError("Prefix and specials exceed pinned sequence capacity")
        self.model_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.model_buffer_bytes = sum(p.numel() * p.element_size() for p in self.model.buffers())

    def __call__(self, chunks):
        clean = [{"input_ids": c["input_ids"], "attention_mask": c["attention_mask"]} for c in chunks]
        batch = self.tokenizer.pad(clean, padding="max_length", max_length=self.input_pad_length,
                                   return_tensors="pt")
        torch = self.torch
        with torch.inference_mode():
            states = self.model(**batch).last_hidden_state
            # As in the official cards, include prefix and special tokens in
            # the attended-token mean; exclude padding, even if it is nonzero.
            states = states.masked_fill(~batch["attention_mask"][..., None].bool(), 0.0)
            means = states.sum(dim=1) / batch["attention_mask"].sum(dim=1)[..., None]
        return means.to(dtype=torch.float32).cpu().numpy()


def encode_local(records_path, preparation_audit_path, asset_manifest_path, model_dir,
                 tokenizer_dir, out_dir, *, batch_size=8, threads=1,
                 evidence_role="engineering_nonconfirmatory"):
    """Production entry point. Output directory is new; successful manifest is last."""
    out = Path(out_dir)
    if out.exists():
        raise ValueError("Output already exists; frozen cache is never overwritten")
    rows, preparation = load_prepared_records(records_path, preparation_audit_path)
    assets = verify_assets(read_json(asset_manifest_path), model_dir, tokenizer_dir)
    encoder_id = assets["encoder_id"]
    if evidence_role not in {"engineering_nonconfirmatory", "confirmatory_calibration"}:
        raise ValueError("Explicit valid evidence role required")
    # A missing runtime or failed model load occurs before any output cache exists.
    adapter = LocalTransformer(model_dir, tokenizer_dir, encoder_id, threads=threads)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    array_path = out / "vectors.npy"
    cache = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32,
                                      shape=(len(rows), DIMENSION))
    token_total = chunks_total = first_chunk_loss = first_window_loss = 0
    max_tokens = max_chunks = 0
    try:
        with (out / "chunk_log.jsonl").open("x", encoding="utf-8") as log:
            for index, row in enumerate(rows):
                vector, detail = encode_document(adapter.tokenizer, adapter, row["text"],
                    encoder_id, adapter.max_length, batch_size=batch_size)
                cache[index] = vector
                log.write(json.dumps({"row_index": index, "record_id": row["record_id"],
                    "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(), **detail},
                    sort_keys=True, allow_nan=False) + "\n")
                token_total += detail["content_tokens"]
                chunks_total += detail["chunks"]
                first_chunk_loss += detail["first_chunk_baseline_omitted_tokens"] > 0
                first_window_loss += detail["model_first_window_omitted_tokens"] > 0
                max_tokens = max(max_tokens, detail["content_tokens"])
                max_chunks = max(max_chunks, detail["chunks"])
        cache.flush()
        del cache
        # Protect against concurrent modification during loading or encoding.
        verify_assets(assets, model_dir, tokenizer_dir)
        rows_again, _ = load_prepared_records(records_path, preparation_audit_path)
        if rows_again != rows:
            raise ValueError("Prepared inputs changed during encoding")
        write_json(out / "row_ids.json", [r["record_id"] for r in rows])
        summary = {"rows": len(rows), "dimension": DIMENSION, "content_tokens": token_total,
            "chunks": chunks_total, "maximum_record_content_tokens": max_tokens,
            "maximum_record_chunks": max_chunks,
            "first_chunk_baseline_truncated_records": first_chunk_loss,
            "first_chunk_baseline_truncated_fraction": first_chunk_loss / len(rows),
            "model_first_window_truncated_records": first_window_loss,
            "model_first_window_truncated_fraction": first_window_loss / len(rows),
            "seconds": time.perf_counter() - start,
            "peak_process_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_units": "KiB on Linux, bytes on macOS; includes earlier process work",
            "model_parameter_bytes": adapter.model_bytes, "model_buffer_bytes": adapter.model_buffer_bytes,
            "output_npy_bytes": array_path.stat().st_size,
            "memory_accounting": "Full input record list and tokenizer/model remain resident; all token IDs/chunk lists of one document are resident; one intradocument batch and its transformer activations are resident; output is a memory map whose resident pages and OS cache are additional memory. No constant-total-memory claim."}
        write_json(out / "encoding_summary.json", summary)
        prefix = ENCODERS[encoder_id]
        manifest = calibration._seal({"schema": SCHEMA,
            "row_ids": [r["record_id"] for r in rows],
            "prepared_rows_sha256": calibration.digest(rows),
            "normalized_records_sha256": run_preparation.text_binding(rows),
            "cache_file_sha256": file_hash(array_path), "shape": [len(rows), DIMENSION],
            "dtype": "float32", "preparation_audit_sha256": preparation["sha256"],
            "asset_manifest_sha256": assets["sha256"],
            "records_file_sha256": file_hash(records_path),
            "chunk_log_sha256": file_hash(out / "chunk_log.jsonl"),
            "encoding_summary_sha256": file_hash(out / "encoding_summary.json"),
            "code_sha256": file_hash(__file__),
            "provenance": {"dataset_id": preparation.get("dataset_id", "unknown"),
                "encoder_id": encoder_id, "encoder_revision": assets["encoder_revision"],
                "encoder_role": "primary" if encoder_id == E5 else "robustness",
                "tokenizer_revision": assets["tokenizer_revision"],
                "population_scope": "complete_prepared_source_disjoint_population",
                "evidence_role": evidence_role,
                "external_provenance_verified_by_this_module": False,
                "confirmatory_study_ready": False},
            "preprocessing": {"chunk_content_tokens": CONTENT_LIMIT,
                "chunking": "nonoverlapping_all_chunks", "pooling": "content_token_weighted_mean_then_normalize",
                "prefix": prefix, "storage_dtype": "float32",
                "learner_normalization": "no_renormalization_after_FP32_storage"},
            "implementation_lock": {"tokenization": "content_once_without_specials; prefix_tokenized_separately_per_chunk; add_model_specials; no_decode_or_truncation",
                "chunk_pooling": "FP32_attention_masked_mean_including_prefix_and_specials_then_FP64_coordinate_order_L2_normalize",
                "document_pooling": "FP64_content_count_weighted_sum_in_chunk_order_then_FP64_coordinate_order_L2_normalize_cast_FP32",
                "graph": "phase3.panels.graph_normalize_and_reference_cosines_on_frozen_FP32",
                "learner": "exact_promotion_of_stored_FP32_without_renormalization",
                "batch_scope": "chunks_of_one_record_only", "batch_size": batch_size,
                "padding": "fixed_CONTENT_LIMIT_plus_prefix_and_specials",
                "device": "cpu", "model_dtype": "float32", "threads": threads,
                "interop_threads": 1, "seed": 0, "mkldnn": False,
                "deterministic_algorithms": True, "attention_implementation": "eager",
                "library_versions": installed_versions(), "python": platform.python_version(),
                "platform": platform.platform(),
                "cross_machine_bit_identity_claimed": False,
                "empirical_model_execution_tested_in_current_development_workspace": False},
            "confirmatory_study_ready": False})
        write_json(out / "cache_manifest.json", manifest)
        return manifest
    except Exception as error:
        # Partial vectors are never accompanied by a successful cache manifest.
        write_json(out / "FAILED.json", {"successful_cache": False,
            "exception_type": type(error).__name__, "message": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    for argument in ("model-dir", "tokenizer-dir", "encoder-id", "encoder-revision", "tokenizer-revision", "versions", "out"):
        inventory.add_argument("--" + argument, required=True)
    encode = sub.add_parser("encode")
    for argument in ("records", "preparation-audit", "asset-manifest", "model-dir", "tokenizer-dir", "out"):
        encode.add_argument("--" + argument, required=True)
    encode.add_argument("--batch-size", type=int, default=8)
    encode.add_argument("--threads", type=int, default=1)
    encode.add_argument("--evidence-role", choices=["engineering_nonconfirmatory", "confirmatory_calibration"],
                        default="engineering_nonconfirmatory")
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            value = asset_inventory(args.model_dir, args.tokenizer_dir, args.encoder_id,
                args.encoder_revision, args.tokenizer_revision, read_json(args.versions))
            write_json(args.out, value)
            print(json.dumps({"asset_manifest": args.out, "sha256": value["sha256"]}))
        else:
            value = encode_local(args.records, args.preparation_audit, args.asset_manifest,
                args.model_dir, args.tokenizer_dir, args.out, batch_size=args.batch_size, threads=args.threads,
                evidence_role=args.evidence_role)
            print(json.dumps({"cache_manifest": str(Path(args.out) / "cache_manifest.json"),
                "rows": len(value["row_ids"]), "sha256": value["sha256"], "confirmatory_study_ready": False}))
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        parser.exit(2, "BLOCKED: " + str(error) + "\n")


if __name__ == "__main__":
    main()
