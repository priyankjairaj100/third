#!/usr/bin/env python3
"""Fail-closed, offline intake checks. Never certifies human work or provenance.

A well-formed file can be fabricated. This utility verifies internal consistency,
not historical truth, copyright, source completeness, annotation authorship, or
whether a vector actually came from a declared encoder. Those remain review gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np

SCHEMA_VERSION = "ccu-local-intake-1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HUMAN_GATES = (
    "original_source_snapshot_and_sampling_unit_completeness",
    "original_label_authenticity_and_no_label_fabrication",
    "encoder_tokenizer_and_embedding_derivation_provenance",
    "human_annotation_authenticity_blinding_pay_and_independence",
    "threshold_selection_and_independent_quality_gate_review",
    "full_protocol_pre_execution_lock_and_source_guard_review",
)
PARTITIONS = {"train", "calibration", "test"}
PREPROCESSING = {
    "chunk_content_tokens": 256,
    "chunking": "nonoverlapping_all_chunks",
    "pooling": "content_token_weighted_mean_then_normalize",
    "prefix": "query: ",
    "storage_dtype": "float32",
    "learner_normalization": "no_renormalization_after_FP32_storage",
}


def sha256(path: Path) -> str:
    out = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            out.update(block)
    return out.hexdigest()


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def norm_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


class Validator:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.checks = []
        self.datasets = []
        self.artifact_hashes = {}

    def check(self, code: str, ok: bool, details: Any = None) -> bool:
        self.checks.append({"check": code, "passed": bool(ok), "details": details})
        return bool(ok)

    def artifact(self, descriptor: Any, code: str) -> Path | None:
        if not isinstance(descriptor, dict):
            self.check(code, False, "Expected {path, sha256}; missing artifact.")
            return None
        pathname, digest = descriptor.get("path"), descriptor.get("sha256")
        if not isinstance(pathname, str) or not pathname or not isinstance(digest, str) or not HEX64.fullmatch(digest):
            self.check(code, False, "Path and lowercase SHA256 are required.")
            return None
        path = (self.workspace / pathname).resolve()
        if not path.is_relative_to(self.workspace) or not path.is_file():
            self.check(code, False, "Path must resolve to an existing regular file inside workspace.")
            return None
        actual = sha256(path)
        self.artifact_hashes[str(path.relative_to(self.workspace))] = actual
        if not self.check(code, actual == digest, {"sha256_actual": actual}):
            return None
        return path

    def json_artifact(self, descriptor: Any, code: str) -> Any:
        path = self.artifact(descriptor, code)
        if path is None:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as err:
            self.check(code + ".parse", False, str(err))
            return None
        return data

    def jsonl_artifact(self, descriptor: Any, code: str) -> list | None:
        path = self.artifact(descriptor, code)
        if path is None:
            return None
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not all(isinstance(row, dict) for row in rows):
                raise ValueError("Each line must be a JSON object.")
        except (OSError, UnicodeError, ValueError) as err:
            self.check(code + ".parse", False, str(err))
            return None
        return rows

    def provenance(self, spec: dict, prefix: str) -> None:
        origin = spec.get("origin")
        self.check(prefix + ".origin_metadata", isinstance(origin, dict) and all(
            isinstance(origin.get(key), str) and bool(origin[key].strip())
            for key in ("repository_or_archive", "immutable_revision", "original_release", "sampling_rule")),
            "Metadata presence only; historical provenance and source completeness require review.")
        if not isinstance(origin, dict):
            origin = {}
        archives = origin.get("original_files", [])
        self.check(prefix + ".original_files_nonempty", isinstance(archives, list) and bool(archives))
        if isinstance(archives, list):
            for index, item in enumerate(archives):
                self.artifact(item, f"{prefix}.original_file.{index}")
        self.artifact(origin.get("extraction_code"), prefix + ".extraction_code")
        self.artifact(origin.get("extraction_log"), prefix + ".extraction_log")

    def rows(self, spec: dict, prefix: str) -> list | None:
        rows = self.jsonl_artifact(spec.get("rows"), prefix + ".rows")
        if rows is None:
            return None
        self.check(prefix + ".nonempty", bool(rows))
        ids = [row.get("record_id") for row in rows]
        ids_valid = all(isinstance(value, str) and bool(value) for value in ids)
        self.check(prefix + ".record_ids", ids_valid and len(set(map(str, ids))) == len(ids), "Nonempty unique stable IDs required.")
        partitions = [row.get("partition") for row in rows]
        self.check(prefix + ".partitions_present", set(map(str, partitions)) == PARTITIONS,
                   {"counts": dict(Counter(map(str, partitions)))})
        self.check(prefix + ".texts", all(isinstance(row.get("text"), str) and row["text"].strip() for row in rows))
        groups = defaultdict(set)
        textgroups = defaultdict(set)
        expected_sources = {}
        source_ok = True
        label_ok = True
        original_ok = True
        dataset_id = spec.get("dataset_id")
        outputs = spec.get("outputs")
        source_count = 0
        for row in rows:
            original = row.get("original_fields")
            original = original if isinstance(original, dict) else {}
            rid = row.get("record_id")
            partition = str(row.get("partition"))
            src = row.get("source_unit_id")
            source_ok &= isinstance(src, str) and bool(src)
            if isinstance(src, str):
                groups[src].add(partition)
            if isinstance(row.get("text"), str):
                textgroups[norm_text(row["text"])].add(partition)
            if dataset_id == "civil_comments":
                required = {"id", "publication_id", "article_id", "parent_id", "created_date", "text", "toxicity"}
                original_ok &= required.issubset(original) and original.get("text") == row.get("text") and str(original.get("id")) == rid
                native = [original.get("publication_id"), original.get("article_id")]
                complete = all(value is not None and str(value) for value in native)
                expected = "civil:" + json.dumps(native, ensure_ascii=False, separators=(",", ":")) if complete else "unknown:" + str(rid)
                source_ok &= src == expected
                label = original.get("toxicity")
                label_ok &= outputs == 1 and number(label) and 0 <= label <= 1 and row.get("labels") == [label]
            elif dataset_id in {"askubuntu", "english_stackexchange"}:
                required = {"Id", "OwnerUserId", "site", "PostTypeId", "CreationDate", "Title", "Body", "Tags"}
                original_ok &= required.issubset(original) and str(original.get("Id")) == rid and original.get("PostTypeId") in (1, "1")
                native = [original.get("site"), original.get("OwnerUserId")]
                complete = all(value is not None and str(value) for value in native)
                expected = "account:" + json.dumps(native, ensure_ascii=False, separators=(",", ":")) if complete else "unknown:" + str(rid)
                source_ok &= src == expected
                vocabulary = spec.get("tag_vocabulary", [])
                tags = original.get("Tags", "")
                tags = re.findall(r"<([^>]+)>", tags) if isinstance(tags, str) else tags
                target = [int(tag in tags) for tag in vocabulary] if isinstance(tags, list) and isinstance(vocabulary, list) else None
                label_ok &= outputs == 20 and isinstance(vocabulary, list) and len(vocabulary) == 20 and len(set(map(str, vocabulary))) == 20 and row.get("labels") == target
            elif dataset_id == "cc_news":
                original_ok &= {"url", "domain", "date", "text"}.issubset(original) and original.get("text") == row.get("text")
                complete = isinstance(row.get("registrable_domain"), str) and bool(row.get("registrable_domain"))
                expected = "domain:" + row["registrable_domain"] if complete else "unknown:" + str(rid)
                source_ok &= src == expected
                label_ok &= outputs == 0 and row.get("labels") == []
            else:
                complete = False
                original_ok = source_ok = label_ok = False
            source_count += bool(complete)
            expected_sources[str(rid)] = str(src)
        self.check(prefix + ".original_fields_internal_consistency", original_ok,
                   "Checks original-field presence and copying; cannot authenticate historical originals.")
        self.check(prefix + ".source_id_derivation", source_ok,
                   "Civil uses [publication_id, article_id]; accounts use [site, OwnerUserId]; missing metadata are record singletons.")
        self.check(prefix + ".native_source_rows_nonzero", source_count > 0, {"rows_with_native_source_metadata": source_count})
        self.check(prefix + ".source_partition_disjoint", all(len(value) == 1 for value in groups.values()),
                   {"cross_partition_sources": sum(len(value) > 1 for value in groups.values())})
        self.check(prefix + ".exact_text_partition_guard", all(len(value) == 1 for value in textgroups.values()),
                   {"cross_partition_normalized_texts": sum(len(value) > 1 for value in textgroups.values())})
        self.check(prefix + ".labels_original_field_consistency", label_ok,
                   "Original-label authenticity and calibration-only vocabulary selection remain review gates.")
        byid = {str(row.get("record_id")): row for row in rows}
        parent_leaks = 0
        for row in rows:
            original = row.get("original_fields", {})
            if isinstance(original, dict) and original.get("parent_id") is not None:
                parent = byid.get(str(original["parent_id"]))
                if parent is not None and parent.get("partition") != row.get("partition"):
                    parent_leaks += 1
        self.check(prefix + ".parent_partition_guard", parent_leaks == 0, {"cross_partition_parent_links": parent_leaks})
        if dataset_id == "cc_news":
            self.artifact(spec.get("public_suffix_list"), prefix + ".public_suffix_list")
            self.artifact(spec.get("domain_extraction_code"), prefix + ".domain_extraction_code")
        self.datasets.append({"dataset_id": dataset_id, "rows": len(rows), "partitions": dict(Counter(map(str, partitions))),
                              "source_units": len(groups), "native_source_metadata_rows": source_count})
        return rows

    def embeddings(self, spec: dict, rows: list | None, prefix: str) -> None:
        emb = spec.get("embeddings")
        if not isinstance(emb, dict):
            self.check(prefix + ".embeddings", False, "Missing frozen semantic embedding cache and provenance.")
            return
        self.check(prefix + ".encoder", emb.get("encoder_id") == "intfloat/multilingual-e5-base")
        self.check(prefix + ".preprocessing", emb.get("preprocessing") == PREPROCESSING)
        self.check(prefix + ".revisions", all(isinstance(emb.get(key), str) and re.fullmatch(r"[0-9a-f]{40,64}", emb[key])
                                            for key in ("encoder_revision", "tokenizer_revision")),
                   "Immutable revision syntax only; provenance requires review.")
        for kind in ("encoder_components", "tokenizer_components"):
            parts = emb.get(kind)
            ok = isinstance(parts, list) and bool(parts) and all(isinstance(item, dict)
                and isinstance(item.get("filename"), str) and item["filename"]
                and isinstance(item.get("sha256"), str) and HEX64.fullmatch(item["sha256"]) for item in parts)
            self.check(prefix + "." + kind + ".declared_hashes", ok,
                       "Content hashes are declared identities, not locally verified component bytes unless path is supplied.")
            if isinstance(parts, list):
                for index, item in enumerate(parts):
                    if isinstance(item, dict) and "path" in item:
                        self.artifact(item, f"{prefix}.{kind}.{index}.local_bytes")
        for kind in ("embedding_code", "embedding_run_log"):
            self.artifact(emb.get(kind), prefix + "." + kind)
        row_ids = self.json_artifact(emb.get("row_ids"), prefix + ".embedding_row_ids")
        if rows is not None:
            self.check(prefix + ".embedding_row_alignment", row_ids == [row.get("record_id") for row in rows])
        path = self.artifact(emb.get("array"), prefix + ".embedding_array")
        if path is None:
            return
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            declared = emb.get("shape")
            shape_ok = isinstance(declared, list) and len(declared) == 2 and all(type(x) is int for x in declared)
            shape_ok = shape_ok and tuple(declared) == array.shape and len(array.shape) == 2 and array.shape[1] == 768
            shape_ok = shape_ok and (rows is None or array.shape[0] == len(rows))
            self.check(prefix + ".embedding_shape", shape_ok, {"actual": list(array.shape), "declared": declared})
            self.check(prefix + ".embedding_dtype", array.dtype == np.dtype("float32"), str(array.dtype))
            finite = zero = 0
            if len(array.shape) == 2:
                for start in range(0, len(array), 2048):
                    batch = np.asarray(array[start:start + 2048])
                    finite += int(np.count_nonzero(~np.isfinite(batch)))
                    zero += int(np.count_nonzero(np.all(batch == 0, axis=1)))
                self.check(prefix + ".embedding_finite_nonzero", finite == 0 and zero == 0,
                           {"nonfinite_entries": finite, "allzero_vectors": zero})
            else:
                self.check(prefix + ".embedding_finite_nonzero", False, "Expected rank-two array.")
        except (OSError, ValueError, AttributeError) as err:
            self.check(prefix + ".embedding_load", False, str(err))

    def calibration(self, spec: dict, rows: list | None, prefix: str) -> None:
        calibration = spec.get("calibration")
        if not isinstance(calibration, dict):
            self.check(prefix + ".calibration", False, "Missing pair frames, human responses and calibration lock.")
            return
        pairs = self.jsonl_artifact(calibration.get("pairs"), prefix + ".calibration_pairs")
        responses = self.jsonl_artifact(calibration.get("responses"), prefix + ".calibration_responses")
        self.artifact(calibration.get("instructions"), prefix + ".annotation_instructions")
        self.artifact(calibration.get("sampling_code"), prefix + ".pair_sampling_code")
        self.artifact(calibration.get("sampling_log"), prefix + ".pair_sampling_log")
        report = self.json_artifact(calibration.get("quality_report"), prefix + ".quality_report")
        if not isinstance(pairs, list) or not isinstance(responses, list) or rows is None:
            return
        calids = {row.get("record_id") for row in rows if row.get("partition") == "calibration"}
        pair_ids = [pair.get("pair_id") for pair in pairs]
        pairidsok = all(isinstance(x, str) and x for x in pair_ids) and len(set(map(str, pair_ids))) == len(pair_ids)
        self.check(prefix + ".calibration_pair_ids", pairidsok)
        valid = bool(pairs)
        roles = Counter()
        pairmap = {}
        for pair in pairs:
            role = pair.get("role")
            roles[str(role)] += 1
            valid &= role in {"threshold_selection", "quality_validation"}
            valid &= pair.get("left_id") in calids and pair.get("right_id") in calids and pair.get("left_id") != pair.get("right_id")
            valid &= number(pair.get("inclusion_probability")) and 0 < pair["inclusion_probability"] <= 1
            valid &= number(pair.get("score")) and -1.000001 <= pair["score"] <= 1.000001
            pairmap[str(pair.get("pair_id"))] = pair
        self.check(prefix + ".calibration_pair_population", valid and all(roles[role] > 0 for role in ("threshold_selection", "quality_validation")), dict(roles))
        response_ids = [row.get("response_id") for row in responses]
        self.check(prefix + ".calibration_response_ids", all(isinstance(x, str) and x for x in response_ids)
                   and len(set(map(str, response_ids))) == len(response_ids))
        response_ok = bool(responses)
        raters = defaultdict(set)
        counts = Counter()
        labels = defaultdict(list)
        for response in responses:
            pid, aid = response.get("pair_id"), response.get("annotator_id")
            response_ok &= isinstance(pid, str) and pid in pairmap and isinstance(aid, str) and bool(aid)
            response_ok &= response.get("label") in {"duplicate", "not_duplicate", "unsure"}
            raters[str(pid)].add(str(aid))
            counts[str(pid)] += 1
            labels[str(pid)].append(response.get("label"))
        response_ok &= all(counts[pid] == 3 and len(raters[pid]) == 3 for pid in pairmap)
        self.check(prefix + ".calibration_responses_structure", response_ok,
                   {"responses": len(responses), "requires": "Exactly three distinct pseudonymous annotators per assigned pair; authenticity is not inferred."})
        validation_ids = [pid for pid, pair in pairmap.items() if pair.get("role") == "quality_validation"]
        majority_positive = sum(labels[pid].count("duplicate") >= 2 for pid in validation_ids)
        self.check(prefix + ".quality_report_arithmetic", isinstance(report, dict)
                   and report.get("validation_pairs") == len(validation_ids)
                   and report.get("majority_duplicate_pairs") == majority_positive,
                   {"validation_pairs": len(validation_ids), "majority_duplicate_pairs": majority_positive,
                    "note": "Reported lower confidence bound is not accepted as proof; independent design/quality review required."})
        self.check(prefix + ".human_gate_cannot_be_automated", True,
                   "Nonempty response rows, names, approvals or quality claims do not prove human authorship, blinding or representative sampling.")

    def validate(self, manifest: dict) -> dict:
        self.check("schema_version", manifest.get("schema_version") == SCHEMA_VERSION)
        self.check("synthetic_not_permitted", manifest.get("synthetic_data_allowed") is False)
        self.artifact(manifest.get("study_design"), "study_design_identity")
        specs = manifest.get("datasets", [])
        self.check("datasets_nonempty", isinstance(specs, list) and bool(specs))
        if isinstance(specs, list):
            identities = [spec.get("dataset_id") for spec in specs if isinstance(spec, dict)]
            self.check("dataset_ids_unique", len(identities) == len(specs) and len(set(map(str, identities))) == len(identities))
            for index, spec in enumerate(specs):
                if not isinstance(spec, dict):
                    self.check(f"dataset.{index}.object", False)
                    continue
                prefix = f"dataset.{index}.{spec.get('dataset_id')}"
                self.provenance(spec, prefix)
                rows = self.rows(spec, prefix)
                self.embeddings(spec, rows, prefix)
                self.calibration(spec, rows, prefix)
        mechanical = all(check["passed"] for check in self.checks)
        return {"schema_version": SCHEMA_VERSION, "local_only": True,
                "mechanical_intake_passed": mechanical, "confirmatory_ready": False,
                "status": "BLOCKED_REQUIRES_REVIEW" if mechanical else "BLOCKED_ASSETS_OR_CONSISTENCY",
                "human_and_design_gates": [{"gate": gate, "status": "NOT_VERIFIED_BY_THIS_PROGRAM"} for gate in HUMAN_GATES],
                "human_approval_strings_accepted_as_evidence": False,
                "scope": "Mechanical intake consistency only; no permission to start confirmation is inferred.",
                "datasets": self.datasets, "artifact_hashes": self.artifact_hashes, "checks": self.checks,
                "failed_checks": [check["check"] for check in self.checks if not check["passed"]]}


def inventory(workspace: Path) -> dict:
    observations = []
    expected = (
        ("civil_comments", "empirical_execution/data/civil_comments_engineering_preview.jsonl"),
        ("cc_news", "empirical_execution/data/cc_news_engineering_preview.jsonl"),
    )
    for name, relative in expected:
        path = workspace / relative
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        observations.append({"dataset_id": name, "path": relative, "sha256": sha256(path), "rows": len(rows),
            "nonmissing_source_ids": sum(row.get("source_id") is not None for row in rows),
            "nonmissing_labels": sum(row.get("label") is not None for row in rows),
            "all_rows_report_immutable_revision_verified": all(row.get("provenance", {}).get("hub_revision_verified") is True for row in rows),
            "source_grouped_partition_column_present": all("partition" in row for row in rows),
            "permitted_role": "nonconfirmatory_natural_data_engineering_fixture"})
    artifact_candidates = []
    for path in sorted(workspace.rglob("*")):
        if path.is_file() and path.resolve().is_relative_to(workspace.resolve()) and path.suffix in {".npy", ".npz", ".safetensors", ".pt", ".pth", ".onnx"}:
            artifact_candidates.append({"path": str(path.relative_to(workspace)), "bytes": path.stat().st_size,
                                        "classification": "unverified_candidate_not_assumed_semantic"})
    return {"schema_version": SCHEMA_VERSION, "status": "BLOCKED_CONFIRMATORY_ASSETS_MISSING", "confirmatory_ready": False,
            "inventory_scope": "Current conversation workspace only; no library or other-conversation access.",
            "natural_fixture_inventory": observations, "array_or_model_file_candidates": artifact_candidates,
            "missing_or_unverified": [
                "Original immutable source-rich Civil Comments / AskUbuntu corpus bytes and acquisition/extraction evidence",
                "Predeclared source-disjoint train/calibration/test row manifests with original labels",
                "FP32 E5 semantic embeddings aligned to those IDs, immutable encoder/tokenizer hashes and derivation evidence",
                "Threshold calibration and fresh blinded independent human quality validation artifacts",
                "Source guards, source completeness, original-label authenticity and full pre-execution design lock"],
            "human_and_design_gates": [{"gate": gate, "status": "NOT_VERIFIED_BY_THIS_PROGRAM"} for gate in HUMAN_GATES],
            "preview_text_is_not_original_archive_bytes": True,
            "lexical_hash_features_are_not_semantic_encoder_embeddings": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.inventory == bool(args.manifest):
        parser.error("Supply exactly one of --inventory or --manifest.")
    workspace = args.workspace.resolve()
    if args.inventory:
        result = inventory(workspace)
    else:
        try:
            if not args.manifest.resolve().is_relative_to(workspace):
                raise ValueError("Manifest must be inside the declared workspace.")
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("Manifest must be a JSON object.")
            result = Validator(workspace).validate(manifest)
        except (OSError, ValueError, TypeError, KeyError) as err:
            result = {"schema_version": SCHEMA_VERSION, "mechanical_intake_passed": False,
                      "confirmatory_ready": False, "status": "BLOCKED_INVALID_MANIFEST", "error": str(err)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.out), "status": result["status"],
                      "mechanical_intake_passed": result.get("mechanical_intake_passed"), "confirmatory_ready": False}))
    return 0 if args.inventory or result.get("mechanical_intake_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
