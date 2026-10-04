"""Offline source-disjoint panels and fixed training-side leakage guards.

This is population preparation, never an authenticity/semantic-quality approval.
Every digest hashes UTF-8(salt + NUL + unit), compared as an unsigned 256-bit
integer. Quantiles use exact rational cutoffs, avoiding float boundary drift.
Unknown IDs are explicit record singletons and never native source evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import unicodedata
from typing import Any, Iterable

import numpy as np

E5 = "intfloat/multilingual-e5-base"
PARTITIONS = ("train", "calibration", "test")
HASH_DENOMINATOR = 1 << 256


def hash_integer(salt: str, unit: str) -> int:
    if not isinstance(salt, str) or not isinstance(unit, str) or "\0" in salt or "\0" in unit:
        raise ValueError("Hash salt and unit must be strings without NUL.")
    return int.from_bytes(hashlib.sha256((salt + "\0" + unit).encode("utf-8")).digest(), "big")


def digest_partition(digest: int) -> str:
    if isinstance(digest, bool) or not isinstance(digest, int) or not 0 <= digest < HASH_DENOMINATOR:
        raise ValueError("Expected an unsigned SHA256 integer.")
    return "train" if digest * 10 < HASH_DENOMINATOR * 7 else (
        "calibration" if digest * 20 < HASH_DENOMINATOR * 17 else "test")


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def source_unit(row: dict, dataset_id: str, *, allow_preview_sources: bool = False) -> tuple[str, str]:
    """Derive ID from actual metadata; no author/source imputation.

    CC-News registrable_domain is supplied by a separately pinned PSL extractor;
    this function checks its use, not the unseen domain extraction provenance.
    Preview host groups may be used ONLY under the explicit engineering flag.
    """
    rid = row["record_id"]
    fields = row.get("original_fields", {})
    if not isinstance(fields, dict):
        raise ValueError("original_fields must be an object")
    if dataset_id == "civil_comments":
        values = [fields.get("publication_id"), fields.get("article_id")]
        prefix = "civil:"
    elif dataset_id in ("askubuntu", "english_stackexchange"):
        values = [fields.get("site"), fields.get("OwnerUserId")]
        prefix = "account:"
    elif dataset_id == "cc_news":
        domain = row.get("registrable_domain")
        if isinstance(domain, str) and domain.strip():
            return "domain:" + domain, "native_metadata_requires_provenance_review"
        host = fields.get("domain")
        if allow_preview_sources and isinstance(host, str) and host.strip():
            return "preview-host:" + host, "engineering_host_not_native_source"
        return "unknown:" + rid, "unknown_singleton"
    else:
        raise ValueError("Unsupported dataset_id")
    for value in values:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            raise ValueError("Native source identifiers must be nonboolean integers, strings, or missing")
    complete = all(value is not None and (not isinstance(value, str) or bool(value.strip())) for value in values)
    if dataset_id in ("askubuntu", "english_stackexchange") and complete:
        owner = values[1]
        if isinstance(owner, int) or (isinstance(owner, str) and owner.strip().lstrip("+-").isdigit()):
            if int(owner) <= 0:
                return "unknown:" + rid, "unknown_singleton"
    if complete:
        return prefix + json.dumps(values, ensure_ascii=False, separators=(",", ":")), "native_metadata_requires_provenance_review"
    return "unknown:" + rid, "unknown_singleton"


def prepare_partitions(rows: list[dict], dataset_id: str, design: dict, *,
                       allow_preview_sources: bool = False) -> tuple[list[dict], dict]:
    """Assign all raw rows before filtering; reject conflicting supplied IDs/splits."""
    salt = design["seeds"]["source_split_salt"]
    ids = [row.get("record_id") for row in rows]
    if not all(isinstance(rid, str) and rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Stable, nonempty, unique record IDs required")
    output = []
    for row in rows:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Nonempty original record text is required")
        source, kind = source_unit(row, dataset_id, allow_preview_sources=allow_preview_sources)
        if "source_unit_id" in row and row["source_unit_id"] != source:
            raise ValueError("Supplied source_unit_id does not match original metadata")
        partition = digest_partition(hash_integer(salt, source))
        if "partition" in row and row["partition"] != partition:
            raise ValueError("Supplied partition does not match protocol SHA256 assignment")
        output.append({**row, "text": normalized_text(row["text"]), "source_unit_id": source, "source_kind": kind,
                       "partition": partition})
    groups = defaultdict(set)
    for row in output:
        groups[row["source_unit_id"]].add(row["partition"])
    audit = {"rows": len(output), "source_units": len(groups),
             "partition_counts": dict(Counter(row["partition"] for row in output)),
             "source_kind_counts": dict(Counter(row["source_kind"] for row in output)),
             "source_disjoint": all(len(parts) == 1 for parts in groups.values()),
             "native_source_units": len({r["source_unit_id"] for r in output if r["source_kind"].startswith("native_")}),
             "hash_rule": "SHA256(UTF8(salt + NUL + source_unit_id)); unsigned digest / 2^256",
             "source_split_salt": salt, "boundaries": ["0", "7/10", "17/20", "1"],
             "normalized_row_texts": sum(a["text"] != b["text"] for a, b in zip(rows, output)),
             "original_fields_preserved": True,
             "confirmatory_source_provenance_verified": False}
    return output, audit


def fixed_guard(rows: list[dict], duplicate_links: Iterable[tuple[str, str]] = ()) -> tuple[list[dict], dict]:
    """Exact/link precedence test > calibration > train; directed Civil parent guard.

    Simultaneous decisions refer to the original fixed partition. Exact text and
    native duplicate links discard lower-precedence endpoints; test is untouched.
    Parent links discard only TRAIN CHILDREN with heldout parents, as the original
    narrative protocol specifies. Calibration/test parent links are only logged.
    """
    byid = {r["record_id"]: r for r in rows}
    if len(byid) != len(rows) or any(r.get("partition") not in PARTITIONS for r in rows):
        raise ValueError("Unique IDs and prepared partitions required")
    reasons: dict[str, set[str]] = defaultdict(set)
    texts = defaultdict(list)
    for row in rows:
        texts[normalized_text(row["text"])].append(row["record_id"])
    calibration_test_duplicate_groups = 0
    precedence = {"train": 0, "calibration": 1, "test": 2}
    for ids in texts.values():
        parts = {byid[rid]["partition"] for rid in ids}
        if {"calibration", "test"}.issubset(parts):
            calibration_test_duplicate_groups += 1
        if len(parts) > 1:
            winner = max(parts, key=precedence.get)
            for rid in ids:
                if byid[rid]["partition"] != winner:
                    reasons[rid].add("crosssplit_normalized_exact_text")
    unresolved = []
    residual_heldout_links = []
    def inspect_link(left: str, right: str, reason: str) -> None:
        if left not in byid or right not in byid:
            unresolved.append({"left": left, "right": right, "kind": reason})
            return
        lp, rp = byid[left]["partition"], byid[right]["partition"]
        if lp == rp:
            return
        if reason == "crosssplit_parent_link":
            if lp == "train" and rp in ("calibration", "test"):
                reasons[left].add(reason)
            else:
                residual_heldout_links.append({"left": left, "right": right, "kind": reason})
        elif precedence[lp] < precedence[rp]:
            reasons[left].add(reason)
        else:
            reasons[right].add(reason)
    for row in rows:
        parent = row.get("original_fields", {}).get("parent_id")
        if parent not in (None, "", 0, "0", -1, "-1"):
            inspect_link(row["record_id"], str(parent), "crosssplit_parent_link")
    for pair in duplicate_links:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(isinstance(x, str) for x in pair):
            raise ValueError("duplicate_links must be stable record-ID pairs")
        inspect_link(pair[0], pair[1], "crosssplit_native_duplicate_link")
    retained = [row for row in rows if row["record_id"] not in reasons]
    return retained, {"input_rows": len(rows), "retained_rows": len(retained),
                      "removed_training_rows": sum(byid[r]["partition"] == "train" for r in reasons),
                      "removed_calibration_rows": sum(byid[r]["partition"] == "calibration" for r in reasons),
                      "removed": [{"record_id": rid, "reasons": sorted(reasons[rid])} for rid in sorted(reasons)],
                      "unresolved_native_links": unresolved,
                      "initial_calibration_test_duplicate_groups": calibration_test_duplicate_groups,
                      "parent_links_outside_directed_training_child_guard": residual_heldout_links,
                      "normalization": "NFC_then_whitespace_collapse_case_sensitive",
                      "guard_scope": "exact_and_duplicate_link_precedence_test_then_calibration_then_train; parent_training_child_only",
                      "reference_population": "simultaneous_guards_against_original_fixed_partitions_no_iterative_reassignment",
                      "parent_sentinels": [None, "", 0, "0", -1, "-1"],
                      "link_manifest_completeness_verified": False}


def graph_normalize(block: np.ndarray) -> np.ndarray:
    value = np.asarray(block, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite embedding cache")
    # Explicit coordinate order pins the FP64 reference independently of BLAS.
    squared = np.zeros(len(value), dtype=np.float64)
    for k in range(value.shape[1]):
        squared += value[:, k] * value[:, k]
    if np.any(squared <= 0) or not np.isfinite(squared).all():
        raise ValueError("Zero/overflow embedding norm")
    return value / np.sqrt(squared)[:, None]


def reference_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """FP64 dot products with explicit coordinate order; strict threshold uses >."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Dimension mismatch")
    scores = np.zeros((len(left), len(right)), dtype=np.float64)
    for k in range(left.shape[1]):
        scores += left[:, k, None] * right[None, :, k]
    return scores


def semantic_guard(rows: list[dict], fp32_cache: np.ndarray, tau: float, *,
                   block_size: int = 256, encoder_id: str = E5,
                   diagnostic_only: bool = False) -> tuple[list[dict], list[int], dict]:
    """Exhaustive train-v-heldout scoring with O(block_size*d+block_size^2) workspace.

    Rows/cache must already be aligned after fixed_guard. Caller supplies a
    memory-mapped FP32 .npy; no persistent N*d FP64 conversion is made. Learner
    vectors are untouched. A pinned E5 derivation is still a separate intake gate.
    The returned row IDs define the population for BOTH E5 and MPNet.
    """
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("Positive integer block_size required")
    if isinstance(tau, bool) or not isinstance(tau, (float, int)) or not math.isfinite(tau) or not -1 <= tau <= 1:
        raise ValueError("Finite threshold in [-1,1] required")
    cache = fp32_cache
    if not isinstance(cache, np.ndarray) or cache.ndim != 2 or cache.shape[0] != len(rows) or cache.shape[1] < 1:
        raise ValueError("Cache must be rank two and align with all rows")
    if cache.dtype != np.dtype(np.float32):
        raise ValueError("Cache must be native/little-endian FP32, not converted FP64 input")
    if not diagnostic_only and (encoder_id != E5 or cache.shape[1] != 768):
        raise ValueError("Primary semantic guard requires E5 dimension 768")
    if any(row.get("partition") not in PARTITIONS for row in rows):
        raise ValueError("Prepared partitions required")
    for start in range(0, len(rows), block_size):
        graph_normalize(cache[start:start + block_size])
    train = [i for i, row in enumerate(rows) if row["partition"] == "train"]
    heldout = [i for i, row in enumerate(rows) if row["partition"] != "train"]
    removed: dict[int, dict] = {}
    scored = 0
    max_score_shape = (0, 0)
    for start in range(0, len(train), block_size):
        ti = train[start:start + block_size]
        left = graph_normalize(cache[ti])
        for other in range(0, len(heldout), block_size):
            hi = heldout[other:other + block_size]
            scores = reference_cosines(left, graph_normalize(cache[hi]))
            scored += scores.size
            max_score_shape = max(max_score_shape, scores.shape, key=lambda s: s[0] * s[1])
            for local in np.flatnonzero(np.any(scores > tau, axis=1)):
                idx = ti[int(local)]
                witnesses = [hi[int(j)] for j in np.flatnonzero(scores[int(local)] > tau)]
                # Canonical stable-ID witness, independent of row/chunk order.
                witness = min(witnesses, key=lambda j: rows[j]["record_id"])
                candidate = {"record_id": rows[idx]["record_id"],
                             "heldout_witness_id": rows[witness]["record_id"],
                             "score": float(scores[int(local), hi.index(witness)])}
                if idx not in removed or candidate["heldout_witness_id"] < removed[idx]["heldout_witness_id"]:
                    removed[idx] = candidate
    retained_indices = [i for i in range(len(rows)) if i not in removed]
    return [rows[i] for i in retained_indices], retained_indices, {
        "input_rows": len(rows), "retained_rows": len(retained_indices),
        "training_rows": len(train), "heldout_rows": len(heldout),
        "removed_training_rows": len(removed), "removed": sorted(removed.values(), key=lambda r: r["record_id"]),
        "pairs_scored": scored, "exhaustive_pairs_expected": len(train) * len(heldout),
        "encoder_id": encoder_id, "tau": float(tau), "operator": "strict_greater_than",
        "reference": "FP32_promote_FP64_ordered_sum_squares_normalize_ordered_coordinate_dot_v1",
        "block_size": block_size, "largest_score_block_shape": list(max_score_shape),
        "workspace_order": "O(block_size*dimension + block_size^2), excluding input cache and row metadata",
        "diagnostic_only": diagnostic_only,
        "cross_encoder_population": "same_returned_record_ids_no_MPNet_refilter",
        "semantic_derivation_and_threshold_quality_verified": False}


def select_panels(rows: list[dict], design: dict, *, target: int = 10000) -> tuple[dict, dict]:
    """Whole surviving source groups: disjoint 80/20 pools, stable group order."""
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("Positive target required")
    groups = defaultdict(list)
    for row in rows:
        if row["partition"] == "train":
            groups[row["source_unit_id"]].append(row["record_id"])
    pool_salt = design["seeds"]["replication_pool_salt"]
    order_salt = design["seeds"]["primary_panel_salt"]
    pools = {"primary": [], "replication": []}
    for source in groups:
        pool = "primary" if hash_integer(pool_salt, source) * 5 < HASH_DENOMINATOR * 4 else "replication"
        pools[pool].append(source)
    panels, info = {}, {}
    for name, units in pools.items():
        selected, selected_sources = [], []
        for source in sorted(units, key=lambda unit: (hash_integer(order_salt, unit), unit)):
            if len(selected) >= target:
                break
            selected_sources.append(source)
            selected.extend(sorted(groups[source]))
        panels[name] = selected
        info[name] = {"target": target, "actual": len(selected), "overshoot": max(0, len(selected) - target),
                      "shortfall": max(0, target - len(selected)), "available_rows": sum(len(groups[s]) for s in units),
                      "available_source_units": len(units), "selected_source_units": selected_sources,
                      "whole_surviving_source_groups": True}
    panels["calibration"] = sorted(r["record_id"] for r in rows if r["partition"] == "calibration")
    panels["test"] = sorted(r["record_id"] for r in rows if r["partition"] == "test")
    info["hash_rule"] = "SHA256(UTF8(protocol_salt + NUL + source_unit_id)); exact 4/5 pool boundary"
    info["pool_salt"], info["group_order_salt"] = pool_salt, order_salt
    return panels, info


def prepare_population(rows: list[dict], dataset_id: str, design: dict, *,
                       duplicate_links: Iterable[tuple[str, str]] = (),
                       fp32_cache: np.ndarray | None = None, tau: float | None = None,
                       target: int = 10000, allow_preview_sources: bool = False,
                       diagnostic_only: bool = False, encoder_id: str = E5,
                       block_size: int = 256) -> tuple[list[dict], dict, dict]:
    assigned, split = prepare_partitions(rows, dataset_id, design, allow_preview_sources=allow_preview_sources)
    guarded, fixed = fixed_guard(assigned, duplicate_links)
    original_indices = {r["record_id"]: i for i, r in enumerate(assigned)}
    retained_indices = [original_indices[r["record_id"]] for r in guarded]
    semantic = {"executed": False, "reason": "aligned_FP32_E5_cache_and_locked_tau_required"}
    if fp32_cache is not None and tau is not None:
        if fp32_cache.shape[0] != len(assigned):
            raise ValueError("Raw cache must align with original rows")
        # Selecting fixed-guard rows may allocate O(N*d) FP32, never N^2 or N*d FP64.
        # Call semantic_guard directly with a prepared memmap to avoid this copy.
        view = fp32_cache if len(guarded) == len(assigned) else fp32_cache[retained_indices]
        guarded, kept, semantic = semantic_guard(guarded, view, tau, block_size=block_size,
                                                 encoder_id=encoder_id, diagnostic_only=diagnostic_only)
        retained_indices = [retained_indices[i] for i in kept]
        semantic["executed"] = True
    panels, panel_audit = select_panels(guarded, design, target=target)
    blockers = ["historical_source_and_snapshot_authenticity_review",
                "source_completeness_and_date_sampling_review", "native_link_manifest_completeness_review",
                "pinned_E5_embedding_derivation_review", "genuine_independent_human_threshold_quality_gate",
                "complete_pre_execution_lock"]
    if not semantic["executed"] or diagnostic_only:
        blockers.append("primary_E5_semantic_guard_not_executed")
    if split["native_source_units"] == 0:
        blockers.append("no_native_source_groups_no_source_bootstrap_claim")
    missing_partitions = [part for part in PARTITIONS if not any(r["partition"] == part for r in guarded)]
    if missing_partitions:
        blockers.append("missing_required_partitions:" + ",".join(missing_partitions))
    if any(panel_audit[p]["shortfall"] for p in ("primary", "replication")):
        blockers.append("declared_whole_group_panel_size_not_reached")
    return guarded, panels, {"status": "prepared_engineering_only_pending_gates", "confirmatory_ready": False,
                            "dataset_id": dataset_id, "split": split, "fixed_guard": fixed,
                            "semantic_guard": semantic, "panels": panel_audit,
                            "retained_original_indices": retained_indices, "remaining_gates": blockers}
