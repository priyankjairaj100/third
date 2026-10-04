"""Replay local source bytes and every semantic cache row before execution.

Replay establishes derivation from supplied bytes. It cannot establish publisher
identity, archive completeness, collection rights, or independent human work.
The caller must review those separate evidence requirements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from phase3 import calibration as cal, panels, run_preparation as prep
from phase4 import adapters, embeddings as emb, news_calendar, calibrate_cache, model_selection
from phase5 import replication, replication_encoding, task_program

SCHEMA = "ccu-source-cache-acceptance-1"
ENCODERS = {"e5": emb.E5, "mpnet": emb.MPNET}
TRUST_BOUNDARY = [
    "Supplied archive bytes do not establish their publisher or completeness.",
    "Source metadata needs review of its meaning and original coverage.",
    "Local model bytes need evidence linking them to the named publisher revision.",
    "Human response files cannot establish independent human collection.",
    "Replay does not establish data permissions or annotation permissions.",
]


def code_bindings():
    modules = [sys.modules[__name__], adapters, emb, panels, prep, news_calendar,
               calibrate_cache, replication, replication_encoding, model_selection]
    return {str(Path(m.__file__).resolve().relative_to(ROOT)): emb.file_hash(m.__file__) for m in modules}


def _path(value, base):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("A local file or directory path is required")
    p = Path(value)
    return p if p.is_absolute() else Path(base) / p


def _json(value, base):
    return value if isinstance(value, dict) else emb.read_json(_path(value, base))


def _equal(a, b, what):
    if task_program.fingerprint(a) != task_program.fingerprint(b):
        raise ValueError(what + " differs from replay")


def _array(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
        raise ValueError(what + " differs in dtype, shape, or exact stored values")


def _file_binding(path):
    path = Path(path)
    return {"name": path.name, "sha256": emb.file_hash(path), "bytes": path.stat().st_size}


def replay_adapter(spec, *, base_dir=".", work_dir=None):
    """Rerun the parser on original bytes and compare every derived JSON row.

Temporary replay outputs are removed when this function returns. No original
input or frozen adapter output is modified. The returned rows stay in memory.
"""
    dataset = spec["dataset_id"]
    adapter = spec["adapter"]
    directory = _path(adapter["directory"], base_dir)
    inputs = {k: _path(v, base_dir) for k, v in adapter["inputs"].items()}
    before = {k: _file_binding(p) for k, p in inputs.items()}
    kwargs = dict(adapter.get("kwargs", {}))
    if dataset == "cc_news":
        kwargs["psl_path"] = _path(kwargs["psl_path"], base_dir)
        before["public_suffix_list"] = _file_binding(kwargs["psl_path"])
    with tempfile.TemporaryDirectory(prefix="source-replay-", dir=work_dir) as temporary:
        output = Path(temporary) / "adapter"
        if dataset == "civil_comments":
            adapters.adapt_civil(inputs["records"], output, **kwargs)
        elif dataset in {"askubuntu", "english_stackexchange"}:
            if kwargs.get("dataset_id", dataset) != dataset:
                raise ValueError("Stack site does not match the declared dataset")
            kwargs["dataset_id"] = dataset
            adapters.adapt_stack(inputs["posts"], inputs["links"], output, **kwargs)
        elif dataset == "cc_news":
            adapters.adapt_news(inputs["records"], output, **kwargs)
        elif dataset == "wcep100":
            replication.parse_wcep(list(inputs.values()), output, **kwargs)
        else:
            raise ValueError("Unsupported original dataset")
        supplied = emb.read_json(directory / "audit.json")
        rebuilt = emb.read_json(output / "audit.json")
        if dataset == "wcep100":
            cal._verify(supplied)
            cal._verify(rebuilt)
            if (directory / "FAILED.json").exists():
                raise ValueError("WCEP adapter has a failure marker")
            names = ["records.jsonl", "events.jsonl", "exclusions.jsonl"]
            for key in ("inputs", "counts", "dataset_id", "stage", "code_sha256"):
                _equal(supplied.get(key), rebuilt.get(key), "WCEP " + key)
        else:
            adapters.verify_adapter_output(directory)
            names = ["records.jsonl", "exclusions.jsonl"]
            if dataset in {"askubuntu", "english_stackexchange"}:
                names += ["duplicate_links.jsonl", "duplicate_link_metadata.jsonl"]
                _equal(emb.read_json(directory / "duplicate_links.json"),
                       emb.read_json(output / "duplicate_links.json"), "Complete duplicate links")
            for key in ("inputs", "counts", "dataset_id", "settings", "status", "code_sha256", "source_helper_sha256"):
                _equal(supplied.get(key), rebuilt.get(key), "Adapter " + key)
        for name in names:
            _equal(emb.file_hash(directory / name), emb.file_hash(output / name), name)
        rows = prep.read_rows(output / "records.jsonl")
        links = emb.read_json(output / "duplicate_links.json") if (output / "duplicate_links.json").exists() else []
        for key, path in inputs.items():
            _equal(before[key], _file_binding(path), "Input after replay: " + key)
        if dataset == "cc_news":
            _equal(before["public_suffix_list"], _file_binding(kwargs["psl_path"]), "Public suffix list after replay")
        report = {"dataset_id": dataset, "rows": len(rows), "native_duplicate_links": len(links),
                  "input_bindings": before, "adapter_audit_sha256": emb.file_hash(directory / "audit.json"),
                  "records_sha256": emb.file_hash(directory / "records.jsonl"),
                  "all_derived_rows_replayed": True, "source_authenticity_verified": False,
                  "engineering_preview": kwargs.get("input_schema") == "engineering_preview"}
        return rows, links, report


def _independent_sources(rows, dataset, design):
    """Recompute native ownership and split hashes without stored source fields."""
    seen = {}
    for row in rows:
        rid = row["record_id"]
        fields = row["original_fields"]
        if dataset == "civil_comments":
            values = [fields.get("publication_id"), fields.get("article_id")]
            prefix = "civil:"
        elif dataset in {"askubuntu", "english_stackexchange"}:
            values = [fields.get("site"), fields.get("OwnerUserId")]
            prefix = "account:"
        elif dataset == "cc_news":
            values = []
            prefix = "domain:"
        else:
            raise ValueError("Source partition audit does not apply to WCEP record-only rows")
        valid = all(v is not None and not isinstance(v, bool) and
                    (not isinstance(v, str) or bool(v.strip())) for v in values)
        if dataset in {"askubuntu", "english_stackexchange"} and valid:
            owner = values[1]
            if isinstance(owner, int) or (isinstance(owner, str) and owner.strip().lstrip("+-").isdigit()):
                valid = int(owner) > 0
        if dataset == "cc_news":
            domain = row.get("registrable_domain")
            source = prefix + domain if isinstance(domain, str) and domain.strip() else "unknown:" + rid
        else:
            source = prefix + json.dumps(values, ensure_ascii=False, separators=(",", ":")) if valid else "unknown:" + rid
        _equal(row["source_unit_id"], source, "Native source for " + rid)
        value = int.from_bytes(hashlib.sha256((design["seeds"]["source_split_salt"] + "\0" + source).encode()).digest(), "big")
        part = "train" if value * 10 < (1 << 256) * 7 else "calibration" if value * 20 < (1 << 256) * 17 else "test"
        _equal(row["partition"], part, "Source split for " + rid)
        if source in seen and seen[source] != part:
            raise ValueError("A source spans two partitions")
        seen[source] = part
    return {"source_units": len(seen), "genuine_source_units": sum(not s.startswith("unknown:") for s in seen),
            "unknown_source_units": sum(s.startswith("unknown:") for s in seen),
            "partitions": dict(Counter(r["partition"] for r in rows))}


def replay_preparation(spec, design, raw_rows, links, *, base_dir=".", work_dir=None):
    dataset = spec["dataset_id"]
    prepared = spec["prepared"]
    records_path, audit_path = _path(prepared["records"], base_dir), _path(prepared["audit"], base_dir)
    actual, actual_audit = emb.load_prepared_records(records_path, audit_path)
    _equal(actual_audit.get("dataset_id"), dataset, "Prepared dataset")
    if dataset == "cc_news":
        with tempfile.TemporaryDirectory(prefix="news-preparation-replay-", dir=work_dir) as temporary:
            output = Path(temporary) / "prepared"
            rebuilt = news_calendar.prepare_news_cohorts(_path(spec["adapter"]["directory"], base_dir),
                _path(spec["coverage"], base_dir), output,
                source_split_salt=design["seeds"]["source_split_salt"])
            _equal(actual, prep.read_rows(output / "records.jsonl"), "Complete News source/date preparation")
            for key in ("counts", "coverage", "source_split_salt", "cohort_source_counts", "all_2017_months_declared_complete"):
                _equal(actual_audit.get(key), rebuilt.get(key), "News preparation " + key)
    else:
        rebuilt, _, audit = panels.prepare_population(raw_rows, dataset, design, duplicate_links=links,
            target=prepared.get("target", 10000), allow_preview_sources=False)
        _equal(actual, rebuilt, "Complete source/fixed-guard preparation")
        for key in ("split", "fixed_guard"):
            _equal(actual_audit.get(key), audit.get(key), "Preparation " + key)
    source_report = _independent_sources(actual, dataset, design)
    source_report.update({"records_file_sha256": emb.file_hash(records_path), "rows_sha256": cal.digest(actual),
                          "audit_sha256": actual_audit["sha256"], "all_rows_replayed": True})
    return actual, source_report


def _replay_cache_in_process(cache_spec, rows, prepared_spec, *, base_dir=".", wcep_spec=None):
    """Replay every row using actual local tokenizer/model bytes.

    Exact equality applies within the recorded CPU runtime contract. A different
    runtime or differing FP32 value causes refusal. There is no tolerance fallback.
    """
    directory = _path(cache_spec["directory"], base_dir)
    assets_path = _path(cache_spec["assets"], base_dir) if "assets" in cache_spec else directory / "assets.json"
    manifest = emb.read_json(directory / "cache_manifest.json")
    assets = emb.read_json(assets_path)
    if (directory / "FAILED.json").exists():
        raise ValueError("Cache has a failure marker")
    if wcep_spec is None:
        crows, cfeatures, cprovenance = calibrate_cache.load_aligned_cache(
            _path(prepared_spec["records"], base_dir), _path(prepared_spec["audit"], base_dir),
            directory / "vectors.npy", directory / "cache_manifest.json", assets_path)
        cache = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
        encoder = manifest["provenance"]["encoder_id"]
    else:
        wrows, cache, manifest, wassets = replication_encoding.load_wcep_cache(
            _path(wcep_spec["adapter"]["directory"], base_dir), _path(wcep_spec["panel"]["directory"], base_dir), directory)
        _equal(rows, wrows, "WCEP replay population")
        _equal(assets, wassets, "WCEP local assets")
        encoder = manifest["encoder_id"]
        crows, cfeatures, cprovenance = [], None, None
    before = {name: _file_binding(directory / name) for name in
              ("vectors.npy", "cache_manifest.json", "row_ids.json", "chunk_log.jsonl", "encoding_summary.json")}
    model_dir = _path(cache_spec["model_dir"], base_dir)
    tokenizer_dir = _path(cache_spec["tokenizer_dir"], base_dir)
    emb.verify_assets(assets, model_dir, tokenizer_dir, verify_runtime=True)
    lock = manifest["implementation_lock"]
    expected = replication_encoding.implementation(lock["batch_size"], lock["threads"])
    _equal(lock, expected, "Complete CPU implementation/runtime lock")
    if manifest.get("row_ids") != [r["record_id"] for r in rows]:
        raise ValueError("Cache does not cover every prepared row in order")
    backend = emb.LocalTransformer(model_dir, tokenizer_dir, encoder, threads=lock["threads"])
    logs = prep.read_rows(directory / "chunk_log.jsonl")
    if len(logs) != len(rows):
        raise ValueError("Chunk log omits or repeats a row")
    details = []
    for i, row in enumerate(rows):
        vector, detail = emb.encode_document(backend.tokenizer, backend, row["text"], encoder,
            backend.max_length, batch_size=lock["batch_size"])
        _array(cache[i], vector, "Full encoder replay row " + str(i))
        wanted = {"row_index": i, "record_id": row["record_id"],
                  "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(), **detail}
        _equal(logs[i], wanted, "Complete token/chunk log row " + str(i))
        details.append(detail)
    summary = emb.read_json(directory / "encoding_summary.json")
    totals = {"rows": len(rows), "dimension": 768,
              "content_tokens": sum(d["content_tokens"] for d in details),
              "chunks": sum(d["chunks"] for d in details),
              "maximum_record_content_tokens": max(d["content_tokens"] for d in details),
              "model_parameter_bytes": backend.model_bytes,
              "model_buffer_bytes": backend.model_buffer_bytes,
              "output_npy_bytes": (directory / "vectors.npy").stat().st_size}
    if wcep_spec is None:
        totals.update({"maximum_record_chunks": max(d["chunks"] for d in details),
            "first_chunk_baseline_truncated_records": sum(d["first_chunk_baseline_omitted_tokens"] > 0 for d in details),
            "first_chunk_baseline_truncated_fraction": sum(d["first_chunk_baseline_omitted_tokens"] > 0 for d in details) / len(rows),
            "model_first_window_truncated_records": sum(d["model_first_window_omitted_tokens"] > 0 for d in details),
            "model_first_window_truncated_fraction": sum(d["model_first_window_omitted_tokens"] > 0 for d in details) / len(rows)})
    for key, value in totals.items():
        _equal(summary.get(key), value, "Recomputed encoding summary " + key)
    emb.verify_assets(assets, model_dir, tokenizer_dir, verify_runtime=True)
    for name, binding in before.items():
        _equal(binding, _file_binding(directory / name), "Cache after replay: " + name)
    return cache, {"encoder_id": encoder, "encoder_revision": assets["encoder_revision"],
        "tokenizer_revision": assets["tokenizer_revision"], "rows_replayed": len(rows),
        "rows_in_cache": len(rows), "sample_only": False, "exact_FP32_match": True,
        "cache_sha256": manifest["cache_file_sha256"], "manifest_sha256": manifest["sha256"],
        "assets_sha256": assets["sha256"], "runtime": lock, "replayed_rows_sha256": cal.digest(rows),
        "recomputed_encoding_summary": totals,
        "publisher_revision_authenticity_verified": False}, (crows, cfeatures, cprovenance)


def replay_cache(cache_spec, rows, prepared_spec, *, base_dir=".", wcep_spec=None):
    """Use one fresh process per full cache replay, including repeated encoders."""
    directory = _path(cache_spec["directory"], base_dir)
    with tempfile.TemporaryDirectory(prefix="encoder-replay-") as temporary:
        job_path, report_path = Path(temporary) / "job.json", Path(temporary) / "report.json"
        job = {"cache_spec": cache_spec, "prepared_spec": prepared_spec,
               "base_dir": str(Path(base_dir).resolve()), "wcep_spec": wcep_spec,
               "expected_rows_sha256": cal.digest(rows)}
        emb.write_json(job_path, job)
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_cache-replay",
            "--job", str(job_path), "--report", str(report_path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, close_fds=True)
        if not report_path.is_file():
            raise RuntimeError("Encoder replay process did not return a report: " + process.stderr[-2000:])
        outcome = emb.read_json(report_path)
        if process.returncode or not outcome.get("passed"):
            raise RuntimeError("Encoder replay refused: " + outcome.get("reason", "process failure"))
        report = outcome["report"]
        _equal(report["replayed_rows_sha256"], cal.digest(rows), "Fresh process replay row binding")
    # Return only the already accepted stored cache and calibration partition.
    assets = _path(cache_spec["assets"], base_dir) if "assets" in cache_spec else directory / "assets.json"
    cache = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
    _equal(emb.file_hash(directory / "vectors.npy"), report["cache_sha256"], "Cache bytes after fresh process replay")
    if wcep_spec is None:
        calibration = calibrate_cache.load_aligned_cache(_path(prepared_spec["records"], base_dir),
            _path(prepared_spec["audit"], base_dir), directory / "vectors.npy",
            directory / "cache_manifest.json", assets)
    else:
        calibration = ([], None, None)
    report["fresh_process_per_encoder"] = True
    return cache, report, calibration


def _cache_worker(job_path, report_path):
    try:
        job = emb.read_json(job_path)
        if job["wcep_spec"] is None:
            rows, _ = emb.load_prepared_records(_path(job["prepared_spec"]["records"], job["base_dir"]),
                _path(job["prepared_spec"]["audit"], job["base_dir"]))
        else:
            spec = job["wcep_spec"]
            rows, _, _ = replication_encoding.load_wcep_panel(_path(spec["adapter"]["directory"], job["base_dir"]),
                _path(spec["panel"]["directory"], job["base_dir"]))
        _equal(cal.digest(rows), job["expected_rows_sha256"], "Parent/child complete row binding")
        _, report, _ = _replay_cache_in_process(job["cache_spec"], rows, job["prepared_spec"],
            base_dir=job["base_dir"], wcep_spec=job["wcep_spec"])
        emb.write_json(report_path, {"passed": True, "report": report})
        return 0
    except Exception as error:
        emb.write_json(report_path, {"passed": False, "error_type": type(error).__name__, "reason": str(error)})
        return 2


def _bind_entries(bundle, caches, rows, train_ids, evaluation_ids, calibration_data):
    positions = {r["record_id"]: i for i, r in enumerate(rows)}
    ti = [positions[r] for r in train_ids]
    ei = [positions[r] for r in evaluation_ids]
    for family in ("curators", "learners"):
        for name, entry in bundle.get(family, {}).items():
            encoder = entry.get("encoder_id")
            short = next((k for k, v in ENCODERS.items() if v == encoder), None)
            if short not in caches:
                raise ValueError("No fully replayed cache for " + family + ":" + name)
            cache = caches[short]
            _equal(entry.get("train_ids"), train_ids, "Training cache IDs")
            _array(entry["train"], cache[ti], family + " training features")
            if family == "learners":
                _equal(entry.get("evaluation_ids"), evaluation_ids, "Evaluation cache IDs")
                _array(entry["evaluation"], cache[ei], "Evaluation learner features")
            dossier = entry.get("calibration")
            if dossier is not None:
                cr, cf, cp = calibration_data[short]
                _equal(dossier["records"], cr, "Complete calibration rows")
                _array(dossier["features"], cf, "Calibration encoder features")
                expected_provenance = cp if family == "curators" else {**cp, "feature_contract": "native_normalized_fp32"}
                _equal(dossier["provenance"], expected_provenance, "Calibration cache derivation")
                if family == "learners":
                    for dimension, projected in entry.get("projection_parameters", {}).items():
                        pd = projected["calibration"]
                        _equal(pd["records"], cr, "Projected calibration record population")
                        projection = np.asarray(bundle["projections"][dimension]["matrix"])
                        expected_features = (cf.astype(np.float64) @ projection).astype(np.float32)
                        _array(pd["features"], expected_features, "Projected calibration features")
                        expected_provenance = {**cp, "feature_contract": "fixed_projection_fp32",
                            "projection_lineage": {"projection_matrix_sha256": model_selection._array_hash(projection),
                                "source_feature_cache_sha256": emb.file_hash(cache.filename),
                                "source_dimension": cache.shape[1], "output_dimension": projection.shape[1]}}
                        _equal(pd["provenance"], expected_provenance, "Projected calibration byte lineage")


def bind_stack_vocabulary(bundle, vocabulary):
    """Bind every output coordinate, lambda choice, and decision rule together."""
    for name, native in bundle.get("learners", {}).items():
        entries = [(name, native)] + [(name + ":" + d, e) for d, e in native.get("projection_parameters", {}).items()]
        for key, entry in entries:
            _equal(entry["calibration"]["vocabulary_lock"], vocabulary, "Learner calibration vocabulary: " + key)
            lock = entry["lambda_lock"]
            _equal(lock["vocabulary_lock_sha256"], vocabulary["sha256"], "Lambda vocabulary binding: " + key)
            threshold = entry.get("tag_threshold_lock")
            if threshold is None:
                raise ValueError("Stack learner requires calibration-only tag decisions: " + key)
            _equal(threshold, model_selection.select_tag_decision_thresholds(lock), "Exact Stack decision rule: " + key)
            _equal(threshold["vocabulary_lock_sha256"], vocabulary["sha256"], "Tag decision vocabulary: " + key)
            if "logistic_threshold_lock" in entry:
                logistic = entry["logistic_threshold_lock"]
                # The dispatcher recomputes the actual logistic rule separately.
                _equal(logistic.get("vocabulary_lock_sha256"), vocabulary["sha256"], "Logistic vocabulary: " + key)


def _bind_labels(bundle, dataset, train_rows, evaluation_rows, calibration_rows, *, spec, base_dir):
    if dataset == "civil_comments":
        y = np.asarray([[r["original_fields"]["toxicity"]] for r in train_rows], dtype=np.float64)
        ey = np.asarray([[r["original_fields"]["toxicity"]] for r in evaluation_rows], dtype=np.float64)
    elif dataset in {"askubuntu", "english_stackexchange"}:
        vocab = _json(spec["vocabulary_lock"], base_dir)
        _equal(vocab["provenance"]["dataset_id"], dataset, "Vocabulary dataset")
        rebuilt = model_selection.select_tag_vocabulary(calibration_rows, vocab["provenance"])
        _equal(vocab, rebuilt, "Calibration-only top-20 tag vocabulary")
        bind_stack_vocabulary(bundle, vocab)
        y = model_selection.encode_tag_targets(train_rows, vocab)
        ey = model_selection.encode_heldout_tag_targets_for_evaluation(evaluation_rows, vocab)
    else:
        if "train_y" in bundle or "evaluation_y" in bundle:
            raise ValueError("Unlabeled corpora cannot acquire artificial task targets")
        return {"task_labels_present": False}
    _array(bundle["train_y"], y, "Original training targets")
    _array(bundle["evaluation_y"], ey, "Original evaluation targets")
    return {"task_labels_present": True, "outputs": y.shape[1],
            "test_labels_used_for_vocabulary_or_guard": False,
            "training_target_sha256": task_program.fingerprint(y),
            "evaluation_target_sha256": task_program.fingerprint(ey)}


def _external_evidence(spec, base_dir):
    """Bind evidence for review without interpreting declarations as approval."""
    evidence = []
    for item in spec.get("external_evidence", []):
        path = _path(item["path"], base_dir)
        binding = _file_binding(path)
        if binding["sha256"] != item["sha256"] or not isinstance(item.get("purpose"), str) or not item["purpose"]:
            raise ValueError("External evidence needs a purpose and matching actual bytes")
        evidence.append({**binding, "purpose": item["purpose"], "authenticated_by_software": False})
    return evidence


def panel_contract(group, spec):
    """Derive the population target from the registered group, never its bundle."""
    name = group.get("panel", "")
    if spec["dataset_id"] == "wcep100":
        if name != "whole-events-25000":
            raise ValueError("WCEP requires the registered whole-events-25000 panel")
        return {"target": 25000, "pool": "whole_events", "calendar": False}
    if name == "complete-calendar-window":
        if spec["dataset_id"] != "cc_news":
            raise ValueError("Calendar window is specific to News")
        return {"target": 10000, "pool": "calendar", "calendar": True}
    match = re.fullmatch(r"(primary|second-disjoint)-(10000|25000|100000|200000)", name)
    if match:
        if match[1] == "second-disjoint" and match[2] != "10000":
            raise ValueError("Unregistered secondary pool target")
        return {"target": int(match[2]), "pool": "primary" if match[1] == "primary" else "replication", "calendar": False}
    if name == "separate-refit-5000":
        return {"target": 5000, "pool": "separate_refit", "calendar": False,
                "exclude_primary_target": 200000, "exclude_replication_target": 10000,
                "source_order_salt": "ccu-v1-refit-panel"}
    raise ValueError("Unrecognized registered panel contract")


def separate_refit_panel(rows, design, contract):
    """Keep complete source groups outside all registered main/replication panels."""
    main, _ = panels.select_panels(rows, design, target=contract["exclude_primary_target"])
    secondary, _ = panels.select_panels(rows, design, target=contract["exclude_replication_target"])
    lookup = {r["record_id"]: r for r in rows}
    excluded_ids = set(main["primary"]) | set(secondary["replication"])
    excluded_sources = {lookup[r]["source_unit_id"] for r in excluded_ids}
    groups = {}
    for row in rows:
        if row["partition"] == "train" and row["source_unit_id"] not in excluded_sources:
            groups.setdefault(row["source_unit_id"], []).append(row["record_id"])
    selected, selected_sources = [], []
    for source in sorted(groups, key=lambda s: (panels.hash_integer(contract["source_order_salt"], s), s)):
        if len(selected) >= contract["target"]:
            break
        selected_sources.append(source)
        selected.extend(sorted(groups[source]))
    return {"separate_refit": selected, "test": sorted(r["record_id"] for r in rows if r["partition"] == "test"),
            "calibration": sorted(r["record_id"] for r in rows if r["partition"] == "calibration")}, {
            "target": contract["target"], "actual": len(selected), "shortfall": max(0, contract["target"] - len(selected)),
            "overshoot": max(0, len(selected) - contract["target"]), "whole_surviving_source_groups": True,
            "source_order_salt": contract["source_order_salt"], "selected_source_units": selected_sources,
            "excluded_primary_target": contract["exclude_primary_target"],
            "excluded_replication_target": contract["exclude_replication_target"],
            "excluded_source_units_sha256": cal.digest(sorted(excluded_sources)),
            "no_selected_primary_or_replication_source_reused": not (set(selected_sources) & excluded_sources)}


def accept_calibration_inputs(dataset_id, encoder, dossier, *, base_dir=".", design=None, work_dir=None):
    """Replay source and encoder inputs before any threshold or response exists.

    This route avoids a threshold dependency cycle. It never selects a threshold
    or treats blank packs as completed human evidence. WCEP has no such route.
    """
    stage = "calibration_specification"
    result = {"schema": "ccu-calibration-input-acceptance-1", "dataset_id": dataset_id,
        "execution_allowed": False, "machine_acceptance_passed": False, "checks": [],
        "trust_boundary": TRUST_BOUNDARY, "source_authenticity_verified": False,
        "human_authenticity_verified": False, "code_bindings": code_bindings()}
    try:
        if dataset_id not in {"civil_comments", "askubuntu", "english_stackexchange", "cc_news"}:
            raise ValueError("Only registered calibration corpora can select a threshold; WCEP inherits News")
        short = encoder if encoder in ENCODERS else next((k for k, v in ENCODERS.items() if v == encoder), None)
        if short is None:
            raise ValueError("Unknown registered calibration encoder")
        spec = dossier["source_acceptance"]
        _equal(spec["dataset_id"], dataset_id, "Calibration source dataset")
        design = design if design is not None else emb.read_json(ROOT.parent / "output/empirical_program/study_design.json")
        stage = "original_calibration_archive_replay"
        raw_rows, links, result["source"] = replay_adapter(spec, base_dir=base_dir, work_dir=work_dir)
        result["checks"].append({"check": stage, "passed": True})
        if result["source"]["engineering_preview"]:
            raise ValueError("Engineering previews cannot supply original semantic calibration provenance")
        stage = "calibration_fixed_population_replay"
        rows, result["preparation"] = replay_preparation(spec, design, raw_rows, links, base_dir=base_dir, work_dir=work_dir)
        if dataset_id == "cc_news":
            news_audit = _json(spec["prepared"]["audit"], base_dir)
            if news_audit.get("all_2017_months_declared_complete") is not True:
                raise ValueError("The registered News calibration extraction requires all 2017 months")
        result["checks"].append({"check": stage, "passed": True})
        stage = "complete_local_calibration_encoder_replay"
        _, result["cache"], calibration = replay_cache(spec["caches"][short], rows, spec["prepared"], base_dir=base_dir)
        _equal(result["cache"]["encoder_id"], ENCODERS[short], "Actual calibration encoder")
        result["checks"].append({"check": stage, "passed": True})
        stage = "complete_calibration_partition_binding"
        cr, cf, cp = calibration
        _equal(dossier["records"], cr, "Every source-disjoint calibration record")
        _array(dossier["features"], cf, "Every calibration vector")
        _equal(dossier["provenance"], cp, "Actual calibration cache provenance")
        result["calibration_binding"] = {"records_sha256": cal.digest(cr),
            "record_ids": [r["record_id"] for r in cr], "source_ids": [r["source_unit_id"] for r in cr],
            "features": task_program.fingerprint(cf), "provenance": cp,
            "threshold_required": False, "human_responses_required": False,
            "test_or_training_rows_enter_calibration": False}
        result["checks"].append({"check": stage, "passed": True})
        result["machine_acceptance_passed"] = True
        result["blockers"] = ["external_archive_encoder_rights_review_required_before_primary_interpretation"]
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as error:
        result["checks"].append({"check": stage, "passed": False, "error_type": type(error).__name__, "reason": str(error)})
        result["blockers"] = [stage + ": " + str(error)]
    result["replay_binding_sha256"] = cal.digest({key: result.get(key) for key in
        ("dataset_id", "source", "preparation", "cache", "calibration_binding", "code_bindings", "machine_acceptance_passed")})
    return result


def accept_source_cache(group, bundle, design, *, base_dir=".", work_dir=None):
    """Return a fail-closed machine replay report for one dispatcher bundle.

    execution_allowed is always False here. Machine replay is necessary but not
    sufficient. The study dispatcher also needs substantive external reviews,
    recomputed human gates, development locks, and complete registry acceptance.
    """
    result = {"schema": SCHEMA, "group_id": group.get("group_id", group.get("id")),
        "execution_allowed": False, "machine_acceptance_passed": False, "checks": [],
        "source_authenticity_verified": False, "human_authenticity_verified": False,
        "trust_boundary": TRUST_BOUNDARY, "code_bindings": code_bindings(),
        "scope": "actual-byte replay; external scientific acceptance remains separate"}
    checks = result["checks"]
    stage = "source_specification"
    try:
        spec = bundle["source_acceptance"]
        dataset = spec["dataset_id"]
        result["dataset_id"] = dataset
        _equal(group.get("corpus", group.get("dataset", dataset)), dataset, "Registry corpus")
        stage = "original_archive_replay"
        raw_rows, links, source = replay_adapter(spec, base_dir=base_dir, work_dir=work_dir)
        result["source"] = source
        checks.append({"check": stage, "passed": True})
        result["external_evidence"] = _external_evidence(spec, base_dir)
        if source["engineering_preview"]:
            result["blockers"] = ["engineering_preview_is_not_an_original_source_rich_archive"]
            result["replay_binding_sha256"] = cal.digest({"source": source, "code_bindings": result["code_bindings"]})
            return result
        contract = panel_contract(group, spec)
        result["panel_contract"] = contract
        stage = "fixed_population_replay"
        if dataset == "wcep100":
            rows, panel, _ = replication_encoding.load_wcep_panel(
                _path(spec["adapter"]["directory"], base_dir), _path(spec["panel"]["directory"], base_dir))
            if panel["target"] != contract["target"] or panel["shortfall"]:
                raise ValueError("Registered WCEP whole-event target is unmet or changed")
            with tempfile.TemporaryDirectory(prefix="wcep-panel-replay-", dir=work_dir) as temporary:
                rebuilt = replication.event_panel(_path(spec["adapter"]["directory"], base_dir),
                    Path(temporary) / "panel", target=panel["target"], salt=panel["salt"])
                _equal(panel, rebuilt, "Complete WCEP event panel")
            prepared_spec = None
            result["preparation"] = {"rows": len(rows), "panel_sha256": panel["sha256"], "whole_events": True}
        else:
            rows, result["preparation"] = replay_preparation(spec, design, raw_rows, links,
                base_dir=base_dir, work_dir=work_dir)
            prepared_spec = spec["prepared"]
        checks.append({"check": stage, "passed": True})
        caches, calibration_data = {}, {}
        result["caches"] = {}
        needed = {"e5"} | {next((k for k, v in ENCODERS.items() if v == e.get("encoder_id")), "unknown")
                 for f in ("curators", "learners") for e in bundle.get(f, {}).values()}
        for short in sorted(needed):
            stage = "complete_local_encoder_replay:" + short
            if short not in spec.get("caches", {}):
                raise ValueError("Missing local model/cache specification for " + short)
            cache, report, calibration = replay_cache(spec["caches"][short], rows, prepared_spec,
                base_dir=base_dir, wcep_spec=spec if dataset == "wcep100" else None)
            _equal(report["encoder_id"], ENCODERS[short], "Named encoder")
            caches[short], result["caches"][short], calibration_data[short] = cache, report, calibration
            checks.append({"check": stage, "passed": True})
        stage = "guard_and_whole_source_panel_replay"
        if dataset == "wcep100":
            news_bundle = bundle["inherited_news_bundle"]
            news_group = spec["inherited_news_group"]
            if news_group.get("corpus") != "cc_news" or news_bundle["source_acceptance"]["dataset_id"] != "cc_news":
                raise ValueError("WCEP must inherit an actual CC-News acceptance bundle")
            news_acceptance = accept_source_cache(news_group, news_bundle, design, base_dir=base_dir, work_dir=work_dir)
            if not news_acceptance["machine_acceptance_passed"]:
                raise ValueError("Inherited News source/cache acceptance failed: " + json.dumps(news_acceptance["blockers"]))
            result["inherited_news_replay_binding_sha256"] = news_acceptance["replay_binding_sha256"]
            news_spec = news_bundle["source_acceptance"]
            for short in sorted(needed):
                if short not in news_acceptance["caches"]:
                    raise ValueError("The inherited News encoder was not replayed")
                for key in ("assets_sha256", "encoder_revision", "tokenizer_revision", "runtime"):
                    _equal(result["caches"][short][key], news_acceptance["caches"][short][key], "WCEP/News same encoder " + key)
                ncache = news_spec["caches"][short]
                ndir = _path(ncache["directory"], base_dir)
                nassets = _path(ncache["assets"], base_dir) if "assets" in ncache else ndir / "assets.json"
                calibration_data[short] = calibrate_cache.load_aligned_cache(
                    _path(news_spec["prepared"]["records"], base_dir), _path(news_spec["prepared"]["audit"], base_dir),
                    ndir / "vectors.npy", ndir / "cache_manifest.json", nassets)
                if short in bundle.get("curators", {}):
                    _equal(bundle["curators"][short]["threshold_lock"], news_bundle["curators"][short]["threshold_lock"], "Inherited News threshold lock")
                    _equal(bundle["curators"][short]["calibration"], news_bundle["curators"][short]["calibration"], "Inherited News calibration dossier")
            retained = rows
            train_ids = [r["record_id"] for r in rows]
            evaluation_ids = []
        else:
            guard = spec["guarded"]
            if guard.get("target", 10000) != contract["target"] or guard.get("pool", "primary") != contract["pool"]:
                raise ValueError("Guard panel target/pool differs from its registered group")
            lock = _json(guard["selection_lock"], base_dir)
            quality = _json(guard["quality_report"], base_dir)
            cal._verify(lock); cal._verify(quality)
            e5_entry = bundle["curators"]["e5"]
            _equal(lock, e5_entry["threshold_lock"], "Guard threshold and accepted E5 curator threshold")
            _equal(quality, e5_entry["calibration"]["quality_report"], "Guard quality and accepted E5 human dossier")
            if (quality.get("selection_lock_sha256") != lock["sha256"] or
                    quality.get("threshold") != lock["threshold"] or
                    quality.get("statistical_and_declared_scope_eligible") is not True):
                raise ValueError("E5 guard needs the matching passed quality report")
            cr, cf, cp = calibration_data["e5"]
            frame = cal._check_inputs(cf, cr, cp, lock["frame"]["scorer"]["block_size"])
            _equal(frame, lock["frame"], "E5 guard calibration frame")
            retained, indices, semantic = panels.semantic_guard(rows, caches["e5"], lock["threshold"],
                block_size=guard.get("block_size", 256), encoder_id=emb.E5)
            _equal(prep.read_rows(_path(guard["records"], base_dir)), retained, "Every post-guard record")
            audit = _json(guard["audit"], base_dir); cal._verify(audit)
            _equal(audit.get("semantic_guard"), semantic, "Complete E5 guard audit")
            _equal(audit.get("retained_original_row_indices_sha256"), cal.digest(indices), "Retained index map")
            _equal(audit.get("prepared_records_file_sha256"), emb.file_hash(_path(guard["records"], base_dir)), "Guarded records bytes")
            if dataset == "cc_news":
                coverage = news_calendar.load_coverage(_path(spec["coverage"], base_dir), source["input_bindings"]["records"]["sha256"])
                if not all(coverage["months"][m]["status"] == "complete" for m in news_calendar.MONTHS[:12]):
                    raise ValueError("The 2017 News calibration extraction is incomplete")
            if contract["calendar"]:
                counts = Counter(r["original_fields"]["date"][:7] for r in retained if r["partition"] == "train")
                decision = news_calendar.calendar_decision(dict(counts), coverage, target=guard.get("target", 10000))
                if not decision["target_met"]:
                    raise ValueError("News whole-month target is unmet")
                window = set(decision["selected_months"])
                panel_population = [r for r in retained if r["partition"] != "train" or r["original_fields"]["date"][:7] in window]
                result["calendar"] = decision
            else:
                panel_population = retained
            if contract["calendar"]:
                train_ids = [r["record_id"] for r in panel_population if r["partition"] == "train"]
                evaluation_ids = []
                _equal(_json(guard["panel_ids"], base_dir), {"calendar": train_ids}, "Complete calendar record IDs")
                result["panel"] = {"pool": "calendar", "target": contract["target"], "actual": len(train_ids),
                                   "complete_window": True, "panel_ids_sha256": cal.digest(train_ids)}
            else:
                if contract["pool"] == "separate_refit":
                    ids, refit_report = separate_refit_panel(panel_population, design, contract)
                    panel_report = {"separate_refit": refit_report}
                else:
                    ids, panel_report = panels.select_panels(panel_population, design, target=contract["target"])
                _equal(_json(guard["panel_ids"], base_dir), ids, "Whole source panel IDs")
                pool = contract["pool"]
                if panel_report[pool]["shortfall"]:
                    raise ValueError("Required whole-source panel target is unmet")
                train_ids, evaluation_ids = ids[pool], ids["test"]
                result["panel"] = {"pool": pool, **panel_report[pool], "panel_ids_sha256": cal.digest(ids)}
            result["guard"] = {"semantic_guard_sha256": cal.digest(semantic), "retained_rows": len(retained),
                               "selection_lock_sha256": lock["sha256"], "quality_report_sha256": quality["sha256"]}
        checks.append({"check": stage, "passed": True})
        stage = "bundle_population_features_and_targets"
        _equal(bundle["train_ids"], train_ids, "Complete ordered training panel")
        _equal(bundle.get("evaluation_ids", []), evaluation_ids, "Complete untouched evaluation population")
        lookup = {r["record_id"]: r for r in retained}
        train_rows = [lookup[r] for r in train_ids]
        evaluation_rows = [lookup[r] for r in evaluation_ids]
        _equal(bundle["train_records"], train_rows, "Complete training records, text, dates, and native fields")
        if "evaluation_records" in bundle:
            _equal(bundle["evaluation_records"], evaluation_rows, "Complete evaluation records")
        sources = [r["source_unit_id"] for r in train_rows]
        _equal(bundle["source_ids"], sources, "Training source ownership")
        _equal(bundle.get("evaluation_source_ids"), [r["source_unit_id"] for r in evaluation_rows], "Complete evaluation source ownership")
        expected_kinds = {r["source_unit_id"]: "genuine_native" if r.get("source_kind", "").startswith("native_")
                          else "unknown_singleton" for r in train_rows}
        _equal(bundle["source_kinds"], expected_kinds, "Complete native/unknown request source frame")
        _bind_entries(bundle, caches, rows, train_ids, evaluation_ids, calibration_data)
        result["labels"] = _bind_labels(bundle, dataset, train_rows, evaluation_rows,
            calibration_data["e5"][0], spec=spec, base_dir=base_dir)
        genuine = sorted({r["source_unit_id"] for r in train_rows if r.get("source_kind", "").startswith("native_")})
        result["validated_bindings"] = {"train_ids": train_ids, "evaluation_ids": evaluation_ids,
            "source_ids": sources, "source_kinds": [r["source_kind"] for r in train_rows],
            "genuine_source_ids": genuine,
            "evaluation_source_ids": [r["source_unit_id"] for r in evaluation_rows],
            "evaluation_source_kinds": {r["source_unit_id"]: "genuine_native" if r.get("source_kind", "").startswith("native_")
                                         else "unknown_singleton" for r in evaluation_rows},
            "calibration_ids": [r["record_id"] for r in calibration_data["e5"][0]]}
        checks.append({"check": stage, "passed": True})
        result["machine_acceptance_passed"] = True
        result["blockers"] = ["substantive_external_source_model_rights_and_human_reviews_required",
                              "dispatcher_must_recompute_human_gates_and_complete_study_acceptance"]
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as error:
        checks.append({"check": stage, "passed": False, "error_type": type(error).__name__, "reason": str(error)})
        result["blockers"] = [stage + ": " + str(error)]
    result["replay_binding_sha256"] = cal.digest({key: result.get(key) for key in
        ("dataset_id", "source", "preparation", "caches", "guard", "panel", "panel_contract", "calendar", "labels",
         "inherited_news_replay_binding_sha256",
         "validated_bindings", "external_evidence", "code_bindings", "machine_acceptance_passed")})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("group", "bundle", "design", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--base-dir", default=".")
    args = p.parse_args()
    bundle = task_program.load_bound_value(emb.read_json(args.bundle), Path(args.base_dir))
    result = accept_source_cache(emb.read_json(args.group), bundle, emb.read_json(args.design), base_dir=args.base_dir)
    emb.write_json(args.out, result)
    print(json.dumps({"machine_acceptance_passed": result["machine_acceptance_passed"],
                      "execution_allowed": False, "report": args.out}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_cache-replay":
        worker = argparse.ArgumentParser()
        worker.add_argument("--job", required=True)
        worker.add_argument("--report", required=True)
        values = worker.parse_args(sys.argv[2:])
        raise SystemExit(_cache_worker(values.job, values.report))
    main()
