"""Locked descriptive analysis for counterfactual-curation development runs.

The resampling unit is an independently reset deletion trajectory.  Methods
are paired *within* its row; independently sampled request arms are never
paired by their ordinal names.  Percentile intervals describe trajectory
variability conditional on the fixed corpus, partition, labels and features.
They are not population confidence intervals, tests, or confirmation claims.

No rows/checkpoints are selected by their observed effects.  Undefined ratios
remain undefined; a zero denominator is never replaced by an arbitrary floor.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_BOOTSTRAP_SEED = 20271004
DEFAULT_METRICS = (
    "addition_count", "removed_selected_count", "selected_count",
    "frozen_mse", "oracle_mse", "uncurated_mse", "baseline_mse",
    "predelete_mse", "mse_effect", "relative_mse_effect",
    "prediction_rms", "predelete_score_sd", "normalized_prediction_rms",
)


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not all(array.shape):
        raise ValueError(f"{name} must be a nonempty vector or matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def prediction_metrics(
    y: Any,
    predictions: Mapping[str, Any],
    *,
    predelete_predictions: Any,
    predelete_mean: Any,
) -> dict[str, Any]:
    """Compute paired fixed-test-set losses and prediction effects.

    ``predictions`` must include ``frozen`` and ``oracle`` and can include
    ``uncurated``.  Additional names receive their own ``<name>_mse`` field.
    ``predelete_mean`` is a training-side mean frozen before deletion, NOT a
    mean estimated from the evaluation labels.  This function cannot verify
    its provenance: the runner must record that contract and its value.

    For multiple responses, MSE/RMS average over records and responses.  The
    fixed score SD is sqrt(mean of per-response population variances), with
    ddof=0.  For one response this is the ordinary population score SD.
    Positive signed loss effect means the frozen-selection method is worse.
    """
    labels = _matrix(y, "y")
    before = _matrix(predelete_predictions, "predelete_predictions")
    if before.shape != labels.shape:
        raise ValueError("predelete predictions must match label shape")
    if not {"frozen", "oracle"}.issubset(predictions):
        raise ValueError("predictions must include frozen and oracle")
    arrays: dict[str, np.ndarray] = {}
    reserved = {"baseline", "predelete", "mse_effect", "relative"}
    for name, values in predictions.items():
        if not isinstance(name, str) or not name.isidentifier() or name in reserved:
            raise ValueError("prediction names must be unreserved identifiers")
        candidate = _matrix(values, f"predictions[{name!r}]")
        if candidate.shape != labels.shape:
            raise ValueError(f"prediction shape differs for {name}")
        arrays[name] = candidate
    mean = np.asarray(predelete_mean, dtype=np.float64)
    if mean.ndim == 0 and labels.shape[1] == 1:
        mean = mean.reshape(1)
    if mean.shape != (labels.shape[1],) or not np.all(np.isfinite(mean)):
        raise ValueError("predelete_mean must be one finite value per response")

    def mse(predicted: np.ndarray) -> float:
        result = float(np.mean(np.square(predicted - labels)))
        if not math.isfinite(result):
            raise ValueError("MSE overflow: inputs exceed the float analysis range")
        return result

    result: dict[str, Any] = {f"{name}_mse": mse(pred) for name, pred in arrays.items()}
    result["baseline_mse"] = mse(np.broadcast_to(mean, labels.shape))
    result["predelete_mse"] = mse(before)
    result["mse_effect"] = result["frozen_mse"] - result["oracle_mse"]
    result["prediction_rms"] = float(np.sqrt(np.mean(
        np.square(arrays["frozen"] - arrays["oracle"]))))
    # Translate by an observed value before centering.  Repeated identical
    # scores must have exactly zero SD: direct mean subtraction can otherwise
    # invent a tiny nonzero denominator through summation roundoff.
    shifted_before = before - before[:1]
    result["predelete_score_sd"] = float(np.sqrt(np.mean(np.square(
        shifted_before - np.mean(shifted_before, axis=0, keepdims=True)))))
    if not math.isfinite(result["prediction_rms"]) or not math.isfinite(result["predelete_score_sd"]):
        raise ValueError("Prediction effect overflow")
    undefined: dict[str, str] = {}
    if result["predelete_mse"] == 0:
        result["relative_mse_effect"] = None
        undefined["relative_mse_effect"] = "predelete_mse_is_zero"
    else:
        result["relative_mse_effect"] = result["mse_effect"] / result["predelete_mse"]
    if result["predelete_score_sd"] == 0:
        result["normalized_prediction_rms"] = None
        undefined["normalized_prediction_rms"] = "predelete_score_sd_is_zero"
    else:
        result["normalized_prediction_rms"] = result["prediction_rms"] / result["predelete_score_sd"]
    for key in ("relative_mse_effect", "normalized_prediction_rms"):
        if result[key] is not None and not math.isfinite(result[key]):
            raise ValueError(f"{key} overflow")
    result["undefined_reasons"] = undefined
    result["evaluation_records"] = labels.shape[0]
    result["response_dimensions"] = labels.shape[1]
    result["fixed_predelete_mean"] = mean.tolist()
    return result


def _number(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field}: booleans are not numeric measurements")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field}: expected a numeric value or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}: nonfinite values are forbidden; use None for undefined ratios")
    return result


def _metric_summary(values: np.ndarray, means: np.ndarray) -> dict[str, Any]:
    valid = values[np.isfinite(values)]
    boot = means[np.isfinite(means)]
    result: dict[str, Any] = {
        "n_total": len(values), "n_defined": len(valid),
        "n_undefined": int(len(values) - len(valid)),
        "mean": float(np.mean(valid)) if len(valid) else None,
        "standard_deviation": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
        "minimum": float(np.min(valid)) if len(valid) else None,
        "median": float(np.median(valid)) if len(valid) else None,
        "maximum": float(np.max(valid)) if len(valid) else None,
        "bootstrap_percentile_interval_95": (
            np.quantile(boot, [0.025, 0.975], method="linear").tolist() if len(boot) else None),
        "bootstrap_defined_replicates": len(boot),
        "bootstrap_undefined_replicates": int(len(means) - len(boot)),
    }
    return result


def summarize_trajectories(
    records: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """Summarize every supplied (arm, checkpoint) without effect selection.

    There must be exactly one row per (trajectory_id, arm, checkpoint), and
    every row must contain a nonnegative integer ``addition_count``.  Its
    zero values enter both the unconditional magnitude and frequency.
    Missing/None metric entries are counted explicitly.  Conditional active
    magnitudes resample complete trajectories before restricting to active
    ones; all-inactive bootstrap draws are undefined and counted.

    Each group draws its own reproducible trajectory index matrix.  All
    methods, contrasts and metric columns within a group reuse those indices.
    No confidence interval pools checkpoints or pairs independently sampled
    arms.  The caller must supply the complete locked checkpoint manifest.
    """
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if len(set(metrics)) != len(metrics) or any(not isinstance(x, str) for x in metrics):
        raise ValueError("metrics must be unique strings")
    derived_names = {"activation_frequency", "addition_count_conditional_active"}
    if derived_names.intersection(metrics):
        raise ValueError("derived activation metrics must not be listed in metrics")
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, str]] = set()
    for row in records:
        arm, checkpoint, trajectory_id = (row.get(name) for name in ("arm", "checkpoint", "trajectory_id"))
        if not isinstance(arm, str) or not arm:
            raise ValueError("arm must be a nonempty string")
        if isinstance(checkpoint, bool) or not isinstance(checkpoint, int) or checkpoint < 0:
            raise ValueError("checkpoint must be a nonnegative integer")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("trajectory_id must be a nonempty string")
        key = (arm, checkpoint, trajectory_id)
        if key in seen:
            raise ValueError(f"duplicate trajectory checkpoint: {key}")
        seen.add(key)
        count = row.get("addition_count")
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("addition_count must be a nonnegative integer")
        groups[(arm, checkpoint)].append(row)
    if not groups:
        raise ValueError("records must not be empty")

    output: list[dict[str, Any]] = []
    for (arm, checkpoint), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["trajectory_id"])
        names = [name for name in metrics if any(name in row for row in rows)]
        names.extend(["activation_frequency", "addition_count_conditional_active"])
        values = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
        for index, row in enumerate(rows):
            for column, name in enumerate(names[:-2]):
                value = row.get(name)
                if value is not None:
                    values[index, column] = _number(value, name)
            count = int(row["addition_count"])
            values[index, -2] = float(count > 0)
            if count > 0:
                values[index, -1] = float(count)
        digest = hashlib.sha256(f"{seed}\0{arm}\0{checkpoint}".encode()).digest()
        group_seed = int.from_bytes(digest[:16], "big")
        rng = np.random.default_rng(group_seed)
        means = np.full((bootstrap_samples, len(names)), np.nan, dtype=np.float64)
        # Bound temporary array size while using a single shared row draw for
        # every column.  Chunking does not change RNG order or the estimate.
        chunk = max(1, min(256, 1_000_000 // max(1, len(rows) * len(names))))
        for start in range(0, bootstrap_samples, chunk):
            stop = min(bootstrap_samples, start + chunk)
            indices = rng.integers(0, len(rows), size=(stop - start, len(rows)))
            sampled = values[indices]
            counts = np.sum(np.isfinite(sampled), axis=1)
            totals = np.nansum(sampled, axis=1)
            np.divide(totals, counts, out=means[start:stop], where=counts > 0)
        output.append({
            "arm": arm, "checkpoint": checkpoint, "trajectory_count": len(rows),
            "trajectory_ids": [row["trajectory_id"] for row in rows],
            "structural_zero_addition_trajectories": int(np.sum(values[:, -2] == 0)),
            "active_trajectories": int(np.sum(values[:, -2] == 1)),
            "bootstrap_group_seed": str(group_seed),
            "metrics": {name: _metric_summary(values[:, column], means[:, column])
                        for column, name in enumerate(names)},
        })
    return {
        "schema": "ccu-trajectory-descriptive-analysis-v1",
        "status": "nonconfirmatory_development_analysis",
        "resampling_unit": "independently_reset_trajectory",
        "conditioning": "fixed_corpus_partition_features_labels_curator_and_request_arm",
        "interval_semantics": "descriptive_percentile_trajectory_bootstrap_not_dataset_population_confidence_interval",
        "method_pairing": "same_trajectory_row_and_same_resampling_indices_within_each_arm_checkpoint",
        "arms_paired": False,
        "checkpoint_pooling": False,
        "structural_zeros_included": True,
        "undefined_ratios": "reported_as_null_with_counts_no_denominator_floor",
        "active_magnitude_interval": "complete_trajectory_resampling_then_restrict_to_active; all_inactive_draws_counted_undefined",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "groups": output,
    }
