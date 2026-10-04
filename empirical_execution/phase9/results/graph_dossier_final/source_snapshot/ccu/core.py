"""Reference curation and eligible-payload ridge maintenance.

The curator is *earlier raw-neighbor suppression*, not greedy independent-set
selection.  Scores are FP64 dot products of independently normalized document
vectors and an edge exists iff score > threshold.  Learner values are stored
FP32, promoted to FP64 without another normalization before moment products.

This module makes no claim that floating-point accumulated moments, Python heap
layout, or serialized bytes are canonical across deletion histories.  Its
residual estimates are numerical diagnostics, not outward-rounded certificates.

Development verification: a three-record mathematical chain unit fixture tested
60 sequential record/source states, finite horizons, batch deletion, empty
targets, and atomic budget rejection. This is a software fixture, not a
synthetic empirical dataset or empirical result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _unique_ids(values: Sequence[str], n: int, name: str) -> tuple[str, ...]:
    ids = tuple(values)
    if len(ids) != n or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError(f"{name} must contain {n} nonempty strings")
    if len(set(ids)) != n:
        raise ValueError(f"{name} must be unique")
    return ids


def canonical_features(features: Any) -> np.ndarray:
    """Own an immutable C-contiguous FP32 copy; never normalize learner values."""
    z = np.array(features, dtype=np.float32, order="C", copy=True)
    if z.ndim != 2 or z.shape[1] == 0 or not np.all(np.isfinite(z)):
        raise ValueError("features must be a finite [N,d] array with d > 0")
    z.flags.writeable = False
    return z


def canonical_targets(targets: Any, n: int) -> np.ndarray:
    """Own a finite FP64 [N,C] target array, preserving fractional labels."""
    y = np.array(targets, dtype=np.float64, order="C", copy=True)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != n or y.shape[1] == 0:
        raise ValueError("targets must have shape [N] or [N,C], C > 0")
    if not np.all(np.isfinite(y)):
        raise ValueError("targets must be finite")
    y.flags.writeable = False
    return y


def normalized_graph_features(features: Any) -> np.ndarray:
    """Own FP64 unit vectors. Zero vectors are rejected, never assigned edges."""
    e = np.array(features, dtype=np.float64, order="C", copy=True)
    if e.ndim != 2 or e.shape[1] == 0 or not np.all(np.isfinite(e)):
        raise ValueError("graph features must be finite [N,d], d > 0")
    lengths = np.linalg.norm(e, axis=1)
    if np.any(lengths == 0) or not np.all(np.isfinite(lengths)):
        raise ValueError("graph features must have nonzero finite norms")
    e /= lengths[:, None]
    return e


def stable_priority(record_ids: Sequence[str], salt: str = "ccu-priority-v1") -> tuple[str, ...]:
    """SHA-256 order with record-ID tie break; independent of labels/features."""
    ids = _unique_ids(record_ids, len(record_ids), "record_ids")
    def key(value: str) -> tuple[bytes, str]:
        return hashlib.sha256((salt + "\0" + value).encode("utf-8")).digest(), value
    return tuple(sorted(ids, key=key))


def _order(record_ids: tuple[str, ...], priority: Sequence[str] | None) -> np.ndarray:
    if priority is None:
        priority = stable_priority(record_ids)
    p = _unique_ids(priority, len(record_ids), "priority")
    if set(p) != set(record_ids):
        raise ValueError("priority must be a full permutation of record_ids")
    lookup = {name: i for i, name in enumerate(record_ids)}
    return np.asarray([lookup[name] for name in p], dtype=np.int64)


def _threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [-1,1]")
    return threshold


@dataclass(frozen=True)
class BlockerGraph:
    """CSR blockers indexed by input record position; arrays are read-only.

    No vectors or pair-score matrix are retained. source_ids is a complete
    disjoint ownership partition, including caller-declared singleton fallbacks.
    """
    record_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    priority_indices: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    threshold: float

    @property
    def blockers(self) -> tuple[np.ndarray, ...]:
        return tuple(self.indices[self.indptr[i]:self.indptr[i + 1]]
                     for i in range(len(self.record_ids)))

    @property
    def blocker_counts(self) -> np.ndarray:
        return np.diff(self.indptr)

    @property
    def edge_count(self) -> int:
        return int(self.indices.size)

    @property
    def priority(self) -> tuple[str, ...]:
        return tuple(self.record_ids[int(i)] for i in self.priority_indices)

    def selected_indices(self, deleted_records: Iterable[str] = ()) -> np.ndarray:
        """Graph-based audit only; direct_oracle independently rebuilds scores."""
        requested = set(deleted_records)
        if not requested <= set(self.record_ids):
            raise ValueError("unknown deleted record ID")
        alive = np.asarray([r not in requested for r in self.record_ids], dtype=bool)
        selected = [i for i in self.priority_indices if alive[i] and
                    not np.any(alive[self.indices[self.indptr[i]:self.indptr[i + 1]]])]
        return np.asarray(selected, dtype=np.int64)

    def accounting(self) -> dict[str, int]:
        return {"records": len(self.record_ids), "edges": self.edge_count,
                "array_bytes": self.indptr.nbytes + self.indices.nbytes + self.priority_indices.nbytes,
                "id_utf8_bytes": sum(len(x.encode()) for x in self.record_ids) +
                                 sum(len(x.encode()) for x in self.source_ids)}


def build_blocker_graph(features: Any, record_ids: Sequence[str], threshold: float,
                        priority: Sequence[str] | None = None,
                        source_ids: Sequence[str] | None = None,
                        block_size: int = 512) -> BlockerGraph:
    """Exhaustively score every unordered pair, retaining all strict edges.

    No ANN or top-k candidate truncation occurs. Peak score workspace is at most
    block_size squared FP64 values; the resulting graph can still be quadratic.
    Numerical edge semantics depend on the pinned FP64 BLAS implementation.
    """
    e = normalized_graph_features(features)
    n = e.shape[0]
    ids = _unique_ids(record_ids, n, "record_ids")
    if source_ids is None:
        sources = ids
    else:
        sources = tuple(source_ids)
        if len(sources) != n or any(not isinstance(s, str) or not s for s in sources):
            raise ValueError("source_ids must contain one nonempty string per record")
    threshold = _threshold(threshold)
    if isinstance(block_size, bool) or int(block_size) != block_size or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)
    order = _order(ids, priority)
    ordered = e[order]
    lists: list[list[int]] = [[] for _ in range(n)]
    for lo in range(0, n, block_size):
        hi = min(n, lo + block_size)
        for early in range(0, hi, block_size):
            end = min(n, early + block_size)
            scores = ordered[lo:hi] @ ordered[early:end].T
            later_rows, earlier_cols = np.nonzero(scores > threshold)
            valid = (early + earlier_cols) < (lo + later_rows)
            for row, col in zip(later_rows[valid], earlier_cols[valid]):
                lists[int(order[lo + row])].append(int(order[early + col]))
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i, blockers in enumerate(lists):
        blockers.sort()  # canonical record-index order, independent of score chunks
        indptr[i + 1] = indptr[i] + len(blockers)
    indices = np.fromiter((v for group in lists for v in group), dtype=np.int64,
                          count=int(indptr[-1]))
    for array in (order, indptr, indices):
        array.flags.writeable = False
    return BlockerGraph(ids, sources, order, indptr, indices, threshold)


@dataclass
class RidgeMoments:
    """Unnormalized sufficient statistics for average-loss penalized ridge."""
    gram: np.ndarray
    cross: np.ndarray
    count: int

    def copy(self) -> "RidgeMoments":
        return RidgeMoments(self.gram.copy(), self.cross.copy(), self.count)

    @property
    def dimension(self) -> int:
        return int(self.gram.shape[0])

    @property
    def outputs(self) -> int:
        return int(self.cross.shape[1])

    def validate(self) -> None:
        d = self.gram.shape[0]
        if self.gram.ndim != 2 or self.gram.shape != (d, d):
            raise ValueError("gram must be square")
        if self.cross.ndim != 2 or self.cross.shape[0] != d:
            raise ValueError("cross must have shape [d,C]")
        if isinstance(self.count, bool) or int(self.count) != self.count or self.count < 0:
            raise ValueError("count must be a nonnegative integer")
        if not np.all(np.isfinite(self.gram)) or not np.all(np.isfinite(self.cross)):
            raise ValueError("moments must be finite")
        if not np.allclose(self.gram, self.gram.T, rtol=1e-13, atol=1e-14):
            raise ValueError("Gram matrix is not numerically symmetric")


def ridge_moments(features: Any, targets: Any) -> RidgeMoments:
    """Bulk FP64 moments of the exact FP32 learner values."""
    z = canonical_features(features).astype(np.float64)
    y = canonical_targets(targets, len(z))
    return RidgeMoments(z.T @ z, z.T @ y, len(z))


@dataclass(frozen=True)
class RidgeSolution:
    weights: np.ndarray
    normal_equation_residual_fro: float
    parameter_error_bound_diagnostic: float
    average_gradient_residual_fro: float
    lambda_reg: float
    count: int
    solver: str = "scipy.linalg.cho_factor/cho_solve (lower, FP64)"


def solve_ridge(moments: RidgeMoments, lambda_reg: float) -> RidgeSolution:
    """Solve (Gram + lambda*n*I) W = cross; all coordinates penalized.

    Empty training data returns the zero head. Residual/(lambda*n) is a
    real-arithmetic perturbation bound estimated with ordinary floating point;
    neither residual arithmetic nor moments have rigorous interval enclosures.
    The shared decoder uses FP64 Cholesky and propagates factorization failures.
    """
    moments.validate()
    lam = float(lambda_reg)
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError("lambda_reg must be positive and finite")
    if moments.count == 0:
        if np.any(moments.gram) or np.any(moments.cross):
            raise ValueError("empty moments must be exactly zero")
        w = np.zeros_like(moments.cross, dtype=np.float64)
        return RidgeSolution(w, 0.0, 0.0, 0.0, lam, 0)
    a = np.array(moments.gram, dtype=np.float64, copy=True)
    a.flat[::a.shape[0] + 1] += lam * moments.count
    # No PSD projection, jitter, precision downgrade, or silent fallback.
    # An SPD/factorization failure must be reported by the experiment runner.
    factor = cho_factor(a, lower=True, overwrite_a=False, check_finite=True)
    w = cho_solve(factor, moments.cross, overwrite_b=False, check_finite=True)
    residual = float(np.linalg.norm(a @ w - moments.cross))
    return RidgeSolution(w, residual, residual / (lam * moments.count),
                         residual / moments.count, lam, moments.count)


@dataclass(frozen=True)
class OracleResult:
    selected_ids: tuple[str, ...]
    selected_indices: np.ndarray
    moments: RidgeMoments
    solution: RidgeSolution


def direct_oracle(features: Any, targets: Any, record_ids: Sequence[str],
                  threshold: float, lambda_reg: float,
                  deleted_records: Iterable[str] = (),
                  priority: Sequence[str] | None = None,
                  graph_features: Any | None = None,
                  score_chunk_size: int = 4096) -> OracleResult:
    """Independent retained-data oracle, with no stored graph/update code.

    For each retained record, directly test *all* earlier retained raw records,
    including records that were themselves suppressed. Rebuild moments in one
    bulk matrix product. graph_features may differ from learner FP32 features,
    but must be exactly the input values supplied to build_blocker_graph.
    """
    z = canonical_features(features)
    y = canonical_targets(targets, len(z))
    e = normalized_graph_features(z if graph_features is None else graph_features)
    if e.shape[0] != len(z):
        raise ValueError("graph and learner records differ")
    ids = _unique_ids(record_ids, len(z), "record_ids")
    threshold = _threshold(threshold)
    requested = set(deleted_records)
    if not requested <= set(ids):
        raise ValueError("unknown deleted record ID")
    order = _order(ids, priority)
    if isinstance(score_chunk_size, bool) or int(score_chunk_size) != score_chunk_size or score_chunk_size < 1:
        raise ValueError("score_chunk_size must be a positive integer")
    alive_order = [int(i) for i in order if ids[int(i)] not in requested]
    selected = []
    for position, i in enumerate(alive_order):
        blocked = False
        for start in range(0, position, int(score_chunk_size)):
            earlier = alive_order[start:min(position, start + int(score_chunk_size))]
            if np.any(e[earlier] @ e[i] > threshold):
                blocked = True
                break
        if not blocked:
            selected.append(i)
    selected_array = np.asarray(selected, dtype=np.int64)
    moments = ridge_moments(z[selected_array], y[selected_array])
    return OracleResult(tuple(ids[i] for i in selected), selected_array,
                        moments, solve_ridge(moments, lambda_reg))


@dataclass(frozen=True)
class DeletionResult:
    requested_units: tuple[str, ...]
    removed_selected_ids: tuple[str, ...]
    admitted_ids: tuple[str, ...]
    discarded_payload_records: int
    touched_records: int
    remaining_horizon: int
    selected_count: int


class EligiblePayloadState:
    """B-E: blocker indexes + eligible FP32 features/FP64 labels + moments.

    Construction copies retained payload; the original graph and full arrays
    are not referenced thereafter. Eligible metadata is pruned after every
    request to the decreasing remaining horizon. Sources form a partition.
    Units absent from eligible metadata still exist in the service registry,
    consume budget when withdrawn, and are recognized as fresh identifiers.
    """
    def __init__(self, graph: BlockerGraph, features: Any, targets: Any,
                 horizon: int, mode: str = "record") -> None:
        z = canonical_features(features)
        y = canonical_targets(targets, len(z))
        if len(graph.record_ids) != len(z):
            raise ValueError("graph and payload record count differ")
        if mode not in {"record", "source"}:
            raise ValueError("mode must be 'record' or 'source'")
        owners = graph.record_ids if mode == "record" else graph.source_ids
        all_units = set(owners)
        if isinstance(horizon, bool) or int(horizon) != horizon or not 0 <= horizon <= len(all_units):
            raise ValueError("horizon must be an integer in [0, number of service units]")
        self.mode = mode
        self.horizon = int(horizon)
        self.dimension = z.shape[1]
        self.outputs = y.shape[1]
        self.alive_units = all_units
        self._owner: dict[str, str] = {}
        self._blockers: dict[str, set[str]] = {}
        self._features: dict[str, np.ndarray] = {}
        self._targets: dict[str, np.ndarray] = {}
        self._by_owner: dict[str, set[str]] = {}
        self._reverse: dict[str, set[str]] = {}
        self._by_count: dict[int, set[str]] = {}
        self._selected: set[str] = set()
        for i, record in enumerate(graph.record_ids):
            owner = owners[i]
            blockers = {owners[int(v)] for v in graph.indices[graph.indptr[i]:graph.indptr[i + 1]]}
            # Same-source blocker prevents activation while own source survives.
            if owner in blockers or len(blockers) > self.horizon:
                continue
            self._owner[record] = owner
            self._blockers[record] = blockers
            self._features[record] = z[i].copy()
            self._targets[record] = y[i].copy()
            self._by_owner.setdefault(owner, set()).add(record)
            self._by_count.setdefault(len(blockers), set()).add(record)
            for unit in blockers:
                self._reverse.setdefault(unit, set()).add(record)
            if not blockers:
                self._selected.add(record)
        self._moments = RidgeMoments(np.zeros((self.dimension, self.dimension), dtype=np.float64),
                                    np.zeros((self.dimension, self.outputs), dtype=np.float64), 0)
        self._add_signed(sorted(self._selected), +1)

    def _add_signed(self, records: Sequence[str], sign: int) -> None:
        if not records:
            return
        # Bounded row blocks avoid a giant temporary for whole-source requests.
        for lo in range(0, len(records), 1024):
            chunk = records[lo:lo + 1024]
            z = np.stack([self._features[r] for r in chunk]).astype(np.float64)
            y = np.stack([self._targets[r] for r in chunk])
            self._moments.gram += sign * (z.T @ z)
            self._moments.cross += sign * (z.T @ y)
            self._moments.count += sign * len(chunk)

    @staticmethod
    def _discard_index(index: dict[Any, set[str]], key: Any, record: str) -> None:
        group = index[key]
        group.remove(record)
        if not group:
            del index[key]

    def _drop_payload(self, record: str) -> None:
        owner = self._owner.pop(record)
        blockers = self._blockers.pop(record)
        self._discard_index(self._by_owner, owner, record)
        self._discard_index(self._by_count, len(blockers), record)
        for unit in blockers:
            self._discard_index(self._reverse, unit, record)
        del self._features[record]
        del self._targets[record]

    def delete(self, unit_ids: Iterable[str]) -> DeletionResult:
        """Atomically validate a fresh batch, then remove it and consume horizon.

        Repeated IDs within or across requests are rejected. An empty batch is
        a no-op. Retry deduplication belongs to a separately charged service
        layer; no unaccounted deletion log is retained here.
        """
        request = tuple(unit_ids)
        if any(not isinstance(unit, str) for unit in request):
            raise ValueError("unit IDs must be strings")
        if len(set(request)) != len(request):
            raise ValueError("duplicate unit IDs within request")
        if not set(request) <= self.alive_units:
            raise ValueError("request contains unknown or already deleted units")
        if len(request) > self.horizon:
            raise ValueError("request exceeds remaining cumulative horizon")
        request = tuple(sorted(request))
        removed_records = set().union(*(self._by_owner.get(unit, set()) for unit in request)) if request else set()
        removed_selected = sorted(removed_records & self._selected)
        self._add_signed(removed_selected, -1)
        self._selected.difference_update(removed_selected)
        for record in sorted(removed_records):
            self._drop_payload(record)
        touched: set[str] = set()
        for unit in request:
            for record in tuple(self._reverse.get(unit, ())):
                blockers = self._blockers[record]
                self._discard_index(self._by_count, len(blockers), record)
                blockers.remove(unit)
                self._by_count.setdefault(len(blockers), set()).add(record)
                touched.add(record)
            self._reverse.pop(unit, None)
        admitted = sorted(record for record in touched
                          if not self._blockers[record] and record not in self._selected)
        self._selected.update(admitted)
        self._add_signed(admitted, +1)
        self.alive_units.difference_update(request)
        self.horizon -= len(request)
        # A record becoming eligible now was necessarily eligible initially;
        # no original payload access is needed. Conversely newly unreachable
        # candidates are erased immediately, including all reciprocal indexes.
        prune = [record for count, group in self._by_count.items()
                 if count > self.horizon for record in group]
        for record in sorted(prune):
            self._drop_payload(record)
        if self._moments.count != len(self._selected):
            raise RuntimeError("internal selected-count invariant failed")
        if self._moments.count == 0:
            # Mathematical zero is known independently of cancellation error.
            self._moments.gram.fill(0)
            self._moments.cross.fill(0)
        return DeletionResult(request, tuple(removed_selected), tuple(admitted),
                              len(removed_records) + len(prune), len(touched),
                              self.horizon, self._moments.count)

    def selected_ids(self) -> tuple[str, ...]:
        """Internal audit interface, not the common head-only release contract."""
        return tuple(sorted(self._selected))

    def moments(self) -> RidgeMoments:
        """Copy statistics; caller mutation cannot corrupt maintained state."""
        return self._moments.copy()

    def statistics(self) -> tuple[np.ndarray, np.ndarray, int]:
        value = self.moments()
        return value.gram, value.cross, value.count

    def decode(self, lambda_reg: float) -> RidgeSolution:
        return solve_ridge(self._moments, lambda_reg)

    def solve_ridge(self, lambda_reg: float) -> np.ndarray:
        return self.decode(lambda_reg).weights

    def check_invariants(self, check_moments: bool = False) -> None:
        """Internal consistency only; does not replace an independent oracle."""
        keys = set(self._owner)
        assert keys == set(self._blockers) == set(self._features) == set(self._targets)
        assert self._selected == {r for r in keys if not self._blockers[r]}
        assert self._moments.count == len(self._selected)
        for r in keys:
            b, own = self._blockers[r], self._owner[r]
            assert own in self.alive_units and own not in b and b <= self.alive_units
            assert len(b) <= self.horizon
            assert r in self._by_owner[own] and r in self._by_count[len(b)]
            assert self._features[r].dtype == np.float32
            assert all(r in self._reverse[u] for u in b)
        assert (set().union(*self._by_owner.values()) if self._by_owner else set()) == keys
        assert (set().union(*self._by_count.values()) if self._by_count else set()) == keys
        for own, group in self._by_owner.items():
            assert group and all(self._owner[r] == own for r in group)
        for count, group in self._by_count.items():
            assert group and all(len(self._blockers[r]) == count for r in group)
        for unit, group in self._reverse.items():
            assert group and all(unit in self._blockers[r] for r in group)
        if check_moments:
            records = sorted(self._selected)
            if records:
                z = np.stack([self._features[r] for r in records])
                y = np.stack([self._targets[r] for r in records])
                expected = ridge_moments(z, y)
                np.testing.assert_allclose(self._moments.gram, expected.gram, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(self._moments.cross, expected.cross, rtol=1e-10, atol=1e-10)
            else:
                assert not np.any(self._moments.gram) and not np.any(self._moments.cross)

    def state_digest(self) -> str:
        """Hash actual logical state including float bytes, not a privacy proof.

        Moment summation roundoff can make different histories hash differently.
        This digest is for deterministic replay/resume auditing only.
        """
        digest = hashlib.sha256()
        def add(value: Any) -> None:
            raw = str(value).encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
        add((self.mode, self.horizon, self.dimension, self.outputs))
        for unit in sorted(self.alive_units):
            add(unit)
        for record in sorted(self._owner):
            add(record); add(self._owner[record]); add(tuple(sorted(self._blockers[record])))
            digest.update(self._features[record].tobytes())
            digest.update(self._targets[record].tobytes())
        digest.update(self._moments.gram.tobytes()); digest.update(self._moments.cross.tobytes())
        add(self._moments.count)
        return digest.hexdigest()

    def accounting(self) -> dict[str, int | str]:
        """Report live accounted contents, excluding caller/oracle allocations.

        Python-owned byte estimate recursively counts each object once; it is
        neither RSS/PSS nor peak memory. Record those in isolated runner workers.
        No serialized-byte claim is made without writing an actual artifact.
        """
        arrays = list(self._features.values()) + list(self._targets.values()) + [self._moments.gram, self._moments.cross]
        return {"mode": self.mode, "remaining_horizon": self.horizon,
                "alive_service_units": len(self.alive_units),
                "eligible_payload_records": len(self._features),
                "selected_records": len(self._selected),
                "blocker_incidences": sum(map(len, self._blockers.values())),
                "feature_array_bytes": sum(a.nbytes for a in self._features.values()),
                "target_array_bytes": sum(a.nbytes for a in self._targets.values()),
                "moment_array_bytes": self._moments.gram.nbytes + self._moments.cross.nbytes,
                "all_array_bytes": sum(a.nbytes for a in arrays),
                "python_owned_bytes_estimate": _owned_size(self.__dict__)}


def _owned_size(obj: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    identity = id(obj)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(obj)
    if isinstance(obj, np.ndarray):
        # getsizeof includes owned array data; views are not used for payload.
        return size
    if isinstance(obj, dict):
        return size + sum(_owned_size(k, seen) + _owned_size(v, seen) for k, v in obj.items())
    if isinstance(obj, (tuple, list, set, frozenset)):
        return size + sum(_owned_size(v, seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return size + _owned_size(obj.__dict__, seen)
    return size
