"""Offline, fail-closed threshold calibration. No labels or model downloads occur here.

The inferential target is the specified independent three-rater majority protocol,
not an unobservable semantic ground truth. Selection's ratio estimator and the
conservative unsure coding are prospective operational clarifications of study v1.
Sampling uses the seeded Python MT19937 reference implementation; exact SRS claims
refer to its uniform-draw algorithmic model, not cryptographic randomness.
"""
from __future__ import annotations

import hashlib
import bisect
import json
import math
import platform
import random
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
from . import panels as _scoring

PROTOCOL = "ccu-calibration-three-rater-v1"
GRID = tuple([Fraction(i, 100) for i in range(50, 100)] + [Fraction(199, 200)])
LABELS = {"duplicate", "not_duplicate", "unsure"}
CATEGORIES = {
    "exact_copy": "duplicate",
    "substantially_same_meaning": "duplicate",
    "overlapping_information": "not_duplicate",
    "merely_related": "not_duplicate",
    "unrelated": "not_duplicate",
    "uncertain": "unsure",
}
BIN_EDGES = tuple(float(Fraction(i, 10)) for i in range(-10, 11))


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _seal(value):
    value = dict(value)
    value["sha256"] = digest(value)
    return value


def _verify(value):
    if not isinstance(value, dict) or "sha256" not in value:
        raise ValueError("A sealed manifest is required")
    body = {k: v for k, v in value.items() if k != "sha256"}
    if digest(body) != value["sha256"]:
        raise ValueError("Manifest hash mismatch")


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric parameter")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite numeric parameter")
        return Fraction(str(value))
    return Fraction(value)


def _fjson(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _rng(seed, stage):
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise ValueError("Seed must be a recorded integer or string")
    return random.Random(int(hashlib.sha256(_canonical([PROTOCOL, stage, seed])).hexdigest(), 16))


def _check_inputs(vectors, records, provenance, block_size):
    a = np.asanyarray(vectors)
    if a.ndim != 2 or a.dtype != np.dtype("float32") or a.shape[0] != len(records):
        raise ValueError("Expected aligned N by D FP32 vectors")
    if a.shape[1] == 0 or isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("Positive dimension and integer block size required")
    ids = [r.get("record_id") for r in records]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Unique nonempty record IDs required")
    if any(not isinstance(r.get("text"), str) or not r["text"].strip() for r in records):
        raise ValueError("Natural record text required for blinded assignments")
    required = {"dataset_id", "encoder_id", "encoder_revision", "population_scope", "evidence_role"}
    if not isinstance(provenance, dict) or not required <= provenance.keys():
        raise ValueError("Incomplete representation and population provenance")
    if any(not isinstance(provenance[k], str) or not provenance[k].strip() for k in required):
        raise ValueError("Provenance declarations must be nonempty strings")
    if provenance["evidence_role"] not in {"engineering_nonconfirmatory", "confirmatory_calibration"}:
        raise ValueError("Explicit evidence role required")
    if provenance["evidence_role"] == "confirmatory_calibration":
        if any(r.get("partition") != "calibration" for r in records):
            raise ValueError("Confirmatory calibration requires calibration-only records")
        if a.shape[1] != 768:
            raise ValueError("Primary semantic calibration requires pinned 768-dimensional vectors")
    vh = hashlib.sha256()
    vh.update(_canonical({"dtype": "float32-little-endian", "shape": list(a.shape)}))
    for start in range(0, len(records), block_size):
        _scoring.graph_normalize(a[start:start + block_size])
        vh.update(np.asarray(a[start:start + block_size], dtype="<f4", order="C").tobytes())
    return {
        "vector_sha256": vh.hexdigest(), "shape": list(a.shape),
        "records_sha256": digest([{k: r[k] for k in ("record_id", "text")} for r in records]),
        "provenance": provenance,
        "scorer": {"id": "fp32-promote-fp64-coordinate-ordered-normalize-and-dot-v1",
                   "block_size": block_size, "numpy": np.__version__,
                   "python": platform.python_version(), "score_predicate": "strict_greater_than",
                   "clamp": "none_for_scores; bin_assignment_saturates_at_first_and_last_bin",
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "shared_scorer_sha256": hashlib.sha256(Path(_scoring.__file__).read_bytes()).hexdigest()},
    }


def iter_pair_scores(vectors, block_size=256, audit=None):
    """Exhaustive i<j scores; O(BD+B²) working memory beyond caller input.

    Shared coordinate-ordered normalization and summation match the panel/guard
    reference independently of tile size. No claim about real-arithmetic cosine
    is made. Raw FP64 scores are not clipped for threshold predicates.
    """
    a = np.asanyarray(vectors)
    n = len(a)
    candidates = [float(x) for x in GRID]
    if audit is not None:
        audit.update({"band": 1e-12, "recomputed_pairs": 0,
                      "maximum_absolute_score_difference": 0.0,
                      "threshold_decision_disagreements": 0, "examples": [],
                      "independent_accumulation": "math.fsum of FP64 component products after shared FP64 normalization",
                      "out_of_range_raw_scores": 0,
                      "authority": "shared coordinate-ordered FP64 reference decides, including any disagreement",
                      "real_arithmetic_certificate": False})
    for i0 in range(0, n, block_size):
        left = _scoring.graph_normalize(a[i0:i0 + block_size])
        for j0 in range(i0, n, block_size):
            if j0 == i0:
                right = left
            else:
                right = _scoring.graph_normalize(a[j0:j0 + block_size])
            scores = _scoring.reference_cosines(left, right)
            for ii in range(len(left)):
                start = ii + 1 if j0 == i0 else 0
                for jj in range(start, len(right)):
                    score = float(scores[ii, jj])
                    if audit is not None:
                        audit["out_of_range_raw_scores"] += int(not -1 <= score <= 1)
                        pos = bisect.bisect_left(candidates, score)
                        near = candidates[max(0, pos - 1):min(len(candidates), pos + 1)]
                        if any(abs(score - value) <= audit["band"] for value in near):
                            check = math.fsum(float(x) * float(y)
                                for x, y in zip(left[ii], right[jj]))
                            difference = abs(score - check)
                            disagrees = sum((score > t) != (check > t) for t in candidates)
                            audit["recomputed_pairs"] += 1
                            audit["maximum_absolute_score_difference"] = max(
                                audit["maximum_absolute_score_difference"], difference)
                            audit["threshold_decision_disagreements"] += disagrees
                            if disagrees and len(audit["examples"]) < 20:
                                audit["examples"].append({"i": i0 + ii, "j": j0 + jj,
                                    "reference_score_hex": score.hex(), "check_score_hex": check.hex()})
                    yield i0 + ii, j0 + jj, score


def _bin(score):
    # Fixed nearest-FP64 decimal edges, [a,b), and +1 in the closed final bin.
    return min(19, max(0, bisect.bisect_right(BIN_EDGES, score) - 1))


def _reservoir_insert(reservoir, item, seen, capacity, rng):
    if len(reservoir) < capacity:
        reservoir.append(item)
    else:
        position = rng.randrange(seen)
        if position < capacity:
            reservoir[position] = item


def _sample_pairs(raw, records, role, context, probabilities):
    pairs = []
    for i, j, score, bin_id in sorted(raw, key=lambda p: (p[0], p[1])):
        natural_id = digest(sorted([records[i]["record_id"], records[j]["record_id"]]))
        pair_id = digest(["blind-pair", context, role, natural_id])
        probability = probabilities[bin_id]
        pairs.append({"pair_id": pair_id, "natural_pair_id": natural_id,
                      "left_id": records[i]["record_id"], "right_id": records[j]["record_id"],
                      "role": role, "score": score, "score_hex": score.hex(), "bin": bin_id,
                      "inclusion_probability": float(probability),
                      "inclusion_probability_exact": _fjson(probability),
                      "assignment_ids": [digest(["blind-assignment", pair_id, k]) for k in range(3)]})
    return pairs


def prepare_selection(vectors, records, provenance, seed=20271003, block_size=256):
    """Return sealed private selection manifest; never create or fill human labels."""
    frame = _check_inputs(vectors, records, provenance, block_size)
    rngs = [_rng(seed, ["selection", b]) for b in range(20)]
    reservoirs, populations = [[] for _ in range(20)], [0] * 20
    numerical_audit = {}
    for i, j, score in iter_pair_scores(vectors, block_size, numerical_audit):
        b = _bin(score)
        populations[b] += 1
        _reservoir_insert(reservoirs[b], (i, j, score, b), populations[b], 30, rngs[b])
    probabilities = {b: Fraction(min(30, count), count) for b, count in enumerate(populations) if count}
    context = digest([frame, "selection", seed])
    return _seal({"schema": "ccu-calibration-sample-1", "protocol_id": PROTOCOL,
                  "role": "threshold_selection", "frame": frame, "seed": seed,
                  "pair_population": len(records) * (len(records) - 1) // 2,
                  "bin_populations": populations, "bin_sample_counts": [len(x) for x in reservoirs],
                  "bins": "20 intervals at nearest-FP64 decimal edges; left closed, right open, final endpoint closed",
                  "bin_edges_hex": [x.hex() for x in BIN_EDGES],
                  "boundary_score_audit": numerical_audit,
                  "sampling": "independent reservoir SRS without replacement per bin; census below 30",
                  "pairs": _sample_pairs([p for r in reservoirs for p in r], records,
                                          "threshold_selection", context, probabilities)})


def write_blinded_pack(manifest, records, directory):
    """Write assignment texts and an EMPTY response bundle. Never distribute it."""
    _verify(manifest)
    by_id = {r["record_id"]: r for r in records}
    if digest([{k: r[k] for k in ("record_id", "text")} for r in records]) != manifest["frame"]["records_sha256"]:
        raise ValueError("Assignment records differ from sampling frame")
    assignments, responses = [], []
    for pair in manifest["pairs"]:
        for assignment_id in pair["assignment_ids"]:
            # Pair orientation is independently blinded; no IDs/scores/roles/labels included.
            reverse = int(assignment_id[:2], 16) % 2 == 1
            left, right = pair["left_id"], pair["right_id"]
            if reverse:
                left, right = right, left
            assignments.append({"assignment_id": assignment_id, "pair_id": pair["pair_id"],
                                "text_a": by_id[left]["text"], "text_b": by_id[right]["text"]})
            responses.append({"response_id": "", "assignment_id": assignment_id,
                              "pair_id": pair["pair_id"], "annotator_id": "", "category": "",
                              "label": "", "human_completed": False, "completed_at": ""})
    assignments.sort(key=lambda a: a["assignment_id"])
    response_bundle = {"manifest_sha256": manifest["sha256"],
                       "collection_declaration": {"protocol_id": PROTOCOL, "human_only": False,
                           "independent": False, "blinded": False, "responsible_collector": ""},
                       "responses": responses}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in [("private_sampling_manifest.json", manifest),
                        ("blinded_assignments.json", assignments),
                        ("responses.template.json", response_bundle)]:
        path = directory / name
        if path.exists():
            raise FileExistsError("Refusing to overwrite frozen annotation pack: " + str(path))
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return {"assignment_count": len(assignments), "pair_count": len(manifest["pairs"]),
            "responses_completed": 0, "directory": str(directory)}


def _majority_labels(manifest, bundle):
    _verify(manifest)
    if not isinstance(bundle, dict) or bundle.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("Response bundle is not bound to the frozen sampling manifest")
    declaration = bundle.get("collection_declaration", {})
    if (declaration.get("protocol_id") != PROTOCOL
            or any(declaration.get(k) is not True for k in ("human_only", "independent", "blinded"))
            or not isinstance(declaration.get("responsible_collector"), str)
            or not declaration["responsible_collector"].strip()):
        raise ValueError("Actual independent blinded human collection declaration is missing")
    required = {a: pair["pair_id"] for pair in manifest["pairs"] for a in pair["assignment_ids"]}
    responses = bundle.get("responses")
    if not isinstance(responses, list) or len(responses) != len(required):
        raise ValueError("Exactly three completed assignments per pair are required")
    grouped = {pair["pair_id"]: [] for pair in manifest["pairs"]}
    seen_assignment, seen_response = set(), set()
    for response in responses:
        assignment = response.get("assignment_id")
        if assignment not in required or assignment in seen_assignment:
            raise ValueError("Unknown or duplicated annotation assignment")
        response_id = response.get("response_id")
        if not isinstance(response_id, str) or not response_id.strip() or response_id in seen_response:
            raise ValueError("Unique nonempty response IDs are required")
        if response.get("pair_id") != required[assignment]:
            raise ValueError("Response pair does not match its assignment")
        annotator = response.get("annotator_id")
        if not isinstance(annotator, str) or not annotator.strip():
            raise ValueError("Human annotator ID is missing")
        timestamp = response.get("completed_at")
        if response.get("human_completed") is not True or not isinstance(timestamp, str):
            raise ValueError("Blank or uncompleted human responses cannot pass")
        try:
            completed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Completion timestamp must be ISO8601 with timezone") from exc
        if completed.tzinfo is None:
            raise ValueError("Completion timestamp timezone is required")
        label, category = response.get("label"), response.get("category")
        if category not in CATEGORIES or label != CATEGORIES[category]:
            raise ValueError("Six-category judgment must agree with the locked binary mapping")
        grouped[required[assignment]].append((annotator, label))
        seen_assignment.add(assignment)
        seen_response.add(response_id)
    labels, uncertain_count, agreement = {}, 0, 0
    for pair_id, judgments in grouped.items():
        if len(judgments) != 3 or len({j[0] for j in judgments}) != 3:
            raise ValueError("Three distinct independent human annotators per pair required")
        labels[pair_id] = int(sum(label == "duplicate" for _, label in judgments) >= 2)
        uncertain_count += sum(label == "unsure" for _, label in judgments)
        agreement += int(len({label for _, label in judgments}) == 1)
    return labels, {"individual_unsure_count": uncertain_count, "unanimous_pairs": agreement,
                    "individual_judgments": len(responses), "response_bundle_sha256": digest(bundle),
                    "human_authenticity": "collector attestation; software cannot authenticate that humans supplied labels"}


def select_threshold(selection_manifest, response_bundle):
    """Freeze the least strict qualifying grid point, or a failed diagnostic lock."""
    _verify(selection_manifest)
    if selection_manifest["role"] != "threshold_selection":
        raise ValueError("Selection requires the selection-stage manifest")
    labels, annotation_report = _majority_labels(selection_manifest, response_bundle)
    ledger = []
    selected = None
    for threshold in GRID:
        weights, positive = [], Fraction(0)
        for pair in selection_manifest["pairs"]:
            if float.fromhex(pair["score_hex"]) > float(threshold):
                probability = pair["inclusion_probability_exact"]
                weight = Fraction(probability["denominator"], probability["numerator"])
                weights.append(weight)
                positive += weight * labels[pair["pair_id"]]
        total = sum(weights, Fraction(0))
        precision = positive / total if total else None
        effective = total * total / sum((w * w for w in weights), Fraction(0)) if weights else None
        passes = precision is not None and precision >= Fraction(19, 20)
        ledger.append({"threshold": float(threshold), "threshold_exact": _fjson(threshold),
                       "raw_support": len(weights), "estimated_population_support": _fjson(total),
                       "effective_support": _fjson(effective) if effective is not None else None,
                       "weighted_ratio_precision": _fjson(precision) if precision is not None else None,
                       "qualifies": passes})
        if passes and selected is None:
            selected = threshold
    qualified = selected is not None
    selected = selected if qualified else GRID[-1]
    return _seal({"schema": "ccu-calibration-selection-lock-1", "protocol_id": PROTOCOL,
                  "selection_manifest_sha256": selection_manifest["sha256"],
                  "frame": selection_manifest["frame"], "selection_seed": selection_manifest["seed"],
                  "selection_natural_pair_ids": [p["natural_pair_id"] for p in selection_manifest["pairs"]],
                  "selection_pair_ids": [p["pair_id"] for p in selection_manifest["pairs"]],
                  "selection_status": "qualified_development_only" if qualified else "failed_diagnostic",
                  "threshold": float(selected), "threshold_exact": _fjson(selected),
                  "estimator": "ratio of HT weighted positives to HT weighted sampled above-threshold pairs",
                  "annotation_report": annotation_report, "grid_ledger": ledger,
                  "post_validation_retuning_allowed": False})


def prepare_validation(vectors, records, selection_lock, seed=20271004, block_size=256):
    """Independent SRS from ALL above-threshold pairs; overlap is never excluded."""
    _verify(selection_lock)
    if selection_lock.get("schema") != "ccu-calibration-selection-lock-1":
        raise ValueError("Frozen completed selection lock required before validation")
    frame = _check_inputs(vectors, records, selection_lock["frame"]["provenance"], block_size)
    if frame != selection_lock["frame"]:
        raise ValueError("Validation frame/scoring differs from frozen selection")
    rng, reservoir, population = _rng(seed, ["validation", selection_lock["sha256"]]), [], 0
    threshold = selection_lock["threshold"]
    numerical_audit = {}
    for i, j, score in iter_pair_scores(vectors, block_size, numerical_audit):
        if score > threshold:
            population += 1
            _reservoir_insert(reservoir, (i, j, score, 0), population, 200, rng)
    probabilities = {0: Fraction(min(200, population), population)} if population else {}
    context = digest([frame, "validation", seed, selection_lock["sha256"]])
    pairs = _sample_pairs(reservoir, records, "quality_validation", context, probabilities)
    previous = set(selection_lock["selection_natural_pair_ids"])
    return _seal({"schema": "ccu-calibration-sample-1", "protocol_id": PROTOCOL,
                  "role": "quality_validation", "frame": frame, "seed": seed,
                  "selection_lock_sha256": selection_lock["sha256"],
                  "threshold": threshold, "pair_population": population,
                  "sample_count": len(pairs), "census": len(pairs) == population,
                  "boundary_score_audit": numerical_audit,
                  "sampling": "independent reservoir SRS from entire above-threshold population",
                  "overlap_natural_pair_count": sum(p["natural_pair_id"] in previous for p in pairs),
                  "fresh_blinded_assignments": True, "pairs": pairs})


def _hypergeom_tail_numerator(N, n, K, x):
    """Integer numerator over comb(N,n); no floating-point probability arithmetic."""
    return sum(math.comb(K, j) * math.comb(N - K, n - j)
               for j in range(max(x, 0, n - (N - K)), min(n, K) + 1))


def finite_population_lower_bound(N, n, x, alpha=Fraction(1, 20)):
    """Exact one-sided lower count L=min{K:P_K[X>=x]>alpha}.

    Inclusive rejection at tail<=alpha yields conservative design-based coverage.
    The only randomness is SRS without replacement from fixed binary pair outcomes.
    Census returns its known proportion; no pairs means precision undefined.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (N, n, x)):
        raise ValueError("Population, sample, and success counts must be integers")
    if not 0 <= x <= n <= N:
        raise ValueError("Invalid finite population counts")
    alpha = _fraction(alpha)
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be strictly between zero and one")
    if N == 0 or n == 0:
        return {"defined": False, "lower_success_count": None, "lower_proportion": None,
                "N": N, "n": n, "x": x, "alpha": _fjson(alpha)}
    if n == N:
        lower = x
    elif x == 0:
        lower = 0
    else:
        denominator = math.comb(N, n)
        lo, hi = 0, N
        while lo < hi:
            mid = (lo + hi) // 2
            tail = _hypergeom_tail_numerator(N, n, mid, x)
            if tail * alpha.denominator > denominator * alpha.numerator:
                hi = mid
            else:
                lo = mid + 1
        lower = lo
    return {"defined": True, "lower_success_count": lower,
            "lower_proportion": _fjson(Fraction(lower, N)), "N": N, "n": n, "x": x,
            "alpha": _fjson(alpha), "confidence": _fjson(1 - alpha),
            "method": "exact finite population hypergeometric tail inversion; census exact",
            "tail_rejection": "less_than_or_equal_alpha"}


def quality_gate(validation_manifest, response_bundle, selection_lock):
    """Pure fail-closed gate: missing responses raise, never become pseudo-labels."""
    _verify(validation_manifest)
    _verify(selection_lock)
    if (validation_manifest.get("role") != "quality_validation"
            or validation_manifest.get("selection_lock_sha256") != selection_lock["sha256"]
            or validation_manifest.get("threshold") != selection_lock["threshold"]):
        raise ValueError("Validation must bind to the frozen selection lock")
    labels, annotation_report = _majority_labels(validation_manifest, response_bundle)
    N, n = validation_manifest["pair_population"], len(validation_manifest["pairs"])
    bound = finite_population_lower_bound(N, n, sum(labels.values()))
    quality_pass = bound["defined"] and 10 * bound["lower_success_count"] >= 9 * N
    selection_pass = selection_lock["selection_status"] == "qualified_development_only"
    primary_pass = bool(quality_pass and selection_pass)
    return _seal({"schema": "ccu-calibration-quality-report-1", "protocol_id": PROTOCOL,
                  "validation_manifest_sha256": validation_manifest["sha256"],
                  "selection_lock_sha256": selection_lock["sha256"],
                  "threshold": selection_lock["threshold"], "bound": bound,
                  "statistical_quality_gate_pass": bool(quality_pass),
                  "development_selection_pass": selection_pass,
                  "primary_quality_gate_pass": primary_pass,
                  "statistical_and_declared_scope_eligible": primary_pass and validation_manifest["frame"]["provenance"]["evidence_role"] == "confirmatory_calibration",
                  "external_provenance_verified_by_this_module": False,
                  "confirmatory_study_ready": False,
                  "annotation_report": annotation_report,
                  "statistical_target": "finite above-threshold pair population under the fixed independent three-human-rater majority protocol",
                  "interpretation": "conditional design-based SRS coverage for fixed or potential protocol outcomes and rater assignment independent of pair selection; no ontological semantic-truth guarantee and no coverage of annotator variation"})
