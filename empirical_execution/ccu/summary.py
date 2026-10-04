"""Finite-horizon, indexed, head-only ridge repair.

This is a NumPy/Python research implementation, not an optimized systems claim.
It implements the coefficient map from theory specification v3.  For record v,
with owner o(v) and distinct blocker-owner set B(v), its contribution is

    t(v) * (x_B(v) - x_(B(v) union {o(v)})),

truncated to the declared cumulative deletion horizon.  A record whose owner
also blocks it contributes identically zero.  Substitution after each deletion
is followed by the *decreased* horizon cutoff.  No feature/target rows, original
blocker graph, original coefficient map, or historical model is retained.

The numeric implementation uses FP32-canonical feature values promoted to FP64
for products and packed FP64 coefficients.  Algebraic exactness is an
exact-arithmetic statement: floating-point values need comparison with an
independent rebuild and residual checks.  Exact-zero cancellation only is used;
there is no near-zero pruning or claim of bitwise history independence.

Public IDs are integer positions in a fixed universe.  An outer service may
map public string IDs to these positions.  The state returns moments/a head,
NOT selected record IDs.  Privacy of allocator remnants/backups is out of scope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterable, Sequence
import sys

import numpy as np


@dataclass(frozen=True)
class SummaryPreflight:
    """Structural forecast, before any dense coefficient vectors are allocated."""

    record_count: int
    feature_dimension: int
    response_dimension: int
    universe_size: int
    horizon: int
    eligible_records: int
    same_source_blocked_records: int
    structural_keys: int
    structural_incidences: int
    linear_query_rank: int
    packed_dimension: int
    coefficient_bytes_upper_bound: int
    statistic_workspace_bytes: int
    dense_decoder_workspace_bytes_lower_estimate: int
    canonical_feature_bytes_external: int
    target_bytes_external: int

    def to_dict(self) -> dict:
        return asdict(self)


class SummaryMemoryError(MemoryError):
    """A declared coefficient allocation cap was exceeded before allocation."""

    def __init__(self, forecast: SummaryPreflight, cap: int):
        self.forecast, self.cap = forecast, cap
        super().__init__(
            f"Coefficient forecast {forecast.coefficient_bytes_upper_bound:,} "
            f"bytes exceeds declared cap {cap:,}; no coefficient arrays allocated. "
            "This cap excludes metadata, caller-held rows, and solver workspace."
        )


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _metadata(
    blockers: Sequence[Iterable[int]], owners: Sequence[int] | None,
    horizon: int, universe_size: int | None,
):
    """Validate and materialize only currently eligible structural endpoints."""
    n = len(blockers)
    h = _integer(horizon, "horizon")
    if h < 0:
        raise ValueError("horizon must be nonnegative")
    if owners is None:
        owner_ids = tuple(range(n))
        universe = n if universe_size is None else _integer(universe_size, "universe_size")
        if universe != n:
            raise ValueError("record mode requires universe_size == number of records")
    else:
        if len(owners) != n:
            raise ValueError("owners must have one entry per record")
        owner_ids = tuple(_integer(u, "owner") for u in owners)
        inferred = max(owner_ids, default=-1) + 1
        universe = inferred if universe_size is None else _integer(universe_size, "universe_size")
    if universe < 0 or h > universe:
        raise ValueError("horizon must be at most the nonnegative public universe size")
    if any(u < 0 or u >= universe for u in owner_ids):
        raise ValueError("owner outside the public universe")
    records, keys, same_source = [], set(), 0
    for v, raw_blockers in enumerate(blockers):
        b = set()
        for raw in raw_blockers:
            u = _integer(raw, "blocker record index")
            if not 0 <= u < n or u == v:
                raise ValueError("blocker must be another valid record index")
            # Graph order is an external fixed-curator contract.  Its priority
            # order need not coincide with row-number order.
            b.add(owner_ids[u])
        owner = owner_ids[v]
        if owner in b:
            same_source += 1
            continue
        if len(b) > h:
            continue
        tail = tuple(sorted(b))
        head = tuple(sorted((*tail, owner))) if len(tail) < h else None
        records.append((v, tail, head))
        keys.add(tail)
        if head is not None:
            keys.add(head)
    return universe, h, records, keys, same_source


def _incidence_rank(records, keys) -> int:
    """Incidence rank p-c0, with a separate ground node for truncated heads."""
    node_id = {key: i for i, key in enumerate(keys)}
    ground = len(node_id)
    parent, sizes = list(range(ground + 1)), [1] * (ground + 1)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, tail, head in records:
        a, b = root(node_id[tail]), root(ground if head is None else node_id[head])
        if a != b:
            if sizes[a] < sizes[b]:
                a, b = b, a
            parent[b] = a
            sizes[a] += sizes[b]
    components = {root(i) for i in range(ground)}
    c0 = len(components - {root(ground)})
    return len(keys) - c0


def preflight(
    blockers: Sequence[Iterable[int]], feature_dimension: int,
    response_dimension: int, horizon: int, owners: Sequence[int] | None = None,
    universe_size: int | None = None,
) -> SummaryPreflight:
    """Count actual structural keys and predict dense bytes at native dimension.

    This consumes each blocker iterable once; reusable arrays/tuples are needed
    if the caller subsequently invokes ``build`` with the same blockers.
    Metadata and Python allocator overhead are deliberately not called free.
    They are measured separately by ``accounting`` once constructed.
    """
    d, c = _integer(feature_dimension, "feature_dimension"), _integer(response_dimension, "response_dimension")
    if d <= 0 or c <= 0:
        raise ValueError("feature and response dimensions must be positive")
    universe, h, records, keys, same_source = _metadata(blockers, owners, horizon, universe_size)
    s = d * (d + 1) // 2 + d * c + 1
    return SummaryPreflight(
        len(blockers), d, c, universe, h, len(records), same_source,
        len(keys), sum(map(len, keys)), _incidence_rank(records, keys), s,
        len(keys) * s * 8,
        # Packed t plus a d-by-d outer product and per-row promoted operands.
        8 * (s + d * d + d + c),
        # M, regularized A, factor/solver workspace, response and solution.
        # LAPACK can allocate more: this is explicitly a lower estimate.
        8 * (3 * d * d + 3 * d * c),
        len(blockers) * d * 4, len(blockers) * c * 8,
    )


class _Entry:
    __slots__ = ("key", "value")

    def __init__(self, key: tuple[int, ...], value: np.ndarray):
        self.key, self.value = key, value


class IndexedRidgeSummary:
    """Only the current indexed coefficient map and public membership persist."""

    __slots__ = (
        "universe_size", "horizon", "dimension", "response_dimension",
        "packed_dimension", "alive", "_coeff", "_inverse", "_degrees",
    )

    @classmethod
    def build(
        cls, features, targets, blockers: Sequence[Iterable[int]], horizon: int,
        owners: Sequence[int] | None = None, universe_size: int | None = None,
        max_coefficient_bytes: int | None = None,
    ) -> "IndexedRidgeSummary":
        """Construct using fixed features; input rows are never retained.

        Features are canonicalized to FP32 before moment products in FP64.
        Targets are represented in FP64.  Callers must use precisely these same
        values in every baseline and oracle.  No intercept is silently appended.
        ``max_coefficient_bytes`` is a pre-allocation numeric-coefficient cap,
        not a guarantee about process peak RSS.
        """
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) != len(blockers):
            raise ValueError("features, targets, and blockers must have matching record dimensions")
        if x.shape[1] < 1 or y.shape[1] < 1:
            raise ValueError("feature and response dimensions must be positive")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("nonfinite feature or target")
        if len(x) > 2**52:
            raise ValueError("FP64 exact-count guard exceeded")
        universe, h, records, keys, same_source = _metadata(blockers, owners, horizon, universe_size)
        d, c = x.shape[1], y.shape[1]
        s = d * (d + 1) // 2 + d * c + 1
        forecast = SummaryPreflight(
            len(x), d, c, universe, h, len(records), same_source,
            len(keys), sum(map(len, keys)), _incidence_rank(records, keys), s,
            len(keys) * s * 8, 8 * (s + d * d + d + c),
            8 * (3 * d * d + 3 * d * c), len(x) * d * 4, y.nbytes,
        )
        if max_coefficient_bytes is not None:
            cap = _integer(max_coefficient_bytes, "max_coefficient_bytes")
            if cap < 0:
                raise ValueError("max_coefficient_bytes must be nonnegative")
            if forecast.coefficient_bytes_upper_bound > cap:
                raise SummaryMemoryError(forecast, cap)
        # Discard structural preflight key set before dense construction.
        del keys, forecast
        state = cls.__new__(cls)
        state.universe_size, state.horizon = universe, h
        state.dimension, state.response_dimension, state.packed_dimension = d, c, s
        state.alive = set(range(universe))
        state._coeff, state._inverse, state._degrees = {}, {}, {}
        tri = np.triu_indices(d)
        q = d * (d + 1) // 2
        for v, tail, head in records:
            z = x[v].astype(np.float64)
            stat = np.empty(s, dtype=np.float64)
            stat[:q] = np.outer(z, z)[tri]
            stat[q:-1] = np.outer(z, y[v]).ravel()
            stat[-1] = 1.0
            for key, sign in ((tail, 1), (head, -1)):
                if key is None:
                    continue
                target = state._coeff.get(key)
                if target is None:
                    # A private owned array: no alias with caller or another key.
                    value = stat.copy() if sign == 1 else -stat
                    state._coeff[key] = _Entry(key, value)
                elif sign == 1:
                    target.value += stat
                else:
                    target.value -= stat
        # Build live indices only after construction and exact-zero pruning.
        for key in list(state._coeff):
            entry = state._coeff[key]
            if not np.any(entry.value != 0):
                del state._coeff[key]
                continue
            state._degrees.setdefault(len(key), set()).add(entry)
            for u in key:
                state._inverse.setdefault(u, set()).add(entry)
        return state

    @property
    def coefficient_count(self) -> int:
        return len(self._coeff)

    def _remove_degree(self, entry: _Entry, degree: int):
        bucket = self._degrees[degree]
        bucket.discard(entry)
        if not bucket:
            del self._degrees[degree]

    def _remove_incidences(self, entry: _Entry, meter: dict):
        for u in entry.key:
            bucket = self._inverse[u]
            bucket.remove(entry)
            meter["inverse_incidence_removals"] += 1
            if not bucket:
                del self._inverse[u]

    def _drop(self, entry: _Entry, meter: dict):
        del self._coeff[entry.key]
        self._remove_degree(entry, len(entry.key))
        self._remove_incidences(entry, meter)

    def _delete_one(self, u: int, meter: dict):
        old_horizon = self.horizon
        while u in self._inverse:
            bucket = self._inverse[u]
            entry = bucket.pop()
            meter["inverse_incidence_removals"] += 1
            if not bucket:
                del self._inverse[u]
            old_key = entry.key
            del self._coeff[old_key]
            self._remove_degree(entry, len(old_key))
            entry.key = tuple(v for v in old_key if v != u)
            meter["key_shrink_events"] += 1
            meter["tuple_cells_copied"] += len(old_key)
            target = self._coeff.get(entry.key)
            if target is None:
                self._coeff[entry.key] = entry
                self._degrees.setdefault(len(entry.key), set()).add(entry)
                # Surviving inverse references already point to this entry.
            else:
                self._remove_incidences(entry, meter)
                target.value += entry.value
                meter["coefficient_collisions"] += 1
                meter["scalar_additions"] += self.packed_dimension
                if not np.any(target.value != 0):
                    self._drop(target, meter)
                    meter["exact_cancellations"] += 1
        # Only untouched old-degree-h keys exceed the new horizon.
        while old_horizon in self._degrees:
            # pop avoids repeatedly scanning tombstones from a new set iterator.
            # _drop/_remove_degree safely discard the already-popped handle.
            entry = self._degrees[old_horizon].pop()
            self._drop(entry, meter)
            meter["horizon_pruned_keys"] += 1
        self.alive.remove(u)
        self.horizon -= 1

    def delete(self, identifiers: Iterable[int]) -> dict:
        """Atomically validate an IDs-only request, then apply fresh deletions.

        Retries and duplicate IDs are rejected atomically, consuming no horizon,
        matching the baseline kernel. IDs outside the fixed public universe and
        out-of-budget requests likewise raise before mutation.  The returned counters belong to the evaluator, not state.
        """
        requested = [_integer(u, "request identifier") for u in identifiers]
        if any(u < 0 or u >= self.universe_size for u in requested):
            raise ValueError("request identifier outside public universe")
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate unit IDs within request")
        if not set(requested) <= self.alive:
            raise ValueError("request contains already deleted units")
        fresh = sorted(requested)
        if len(fresh) > self.horizon:
            raise ValueError("request exceeds remaining cumulative deletion horizon")
        meter = {
            "input_identifiers": len(requested), "fresh_deletions": len(fresh),
            "key_shrink_events": 0, "tuple_cells_copied": 0,
            "coefficient_collisions": 0, "scalar_additions": 0,
            "exact_cancellations": 0, "horizon_pruned_keys": 0,
            "inverse_incidence_removals": 0,
        }
        for u in fresh:
            self._delete_one(u, meter)
        return meter

    def statistics(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Return copied current Gram, cross moment, and exact integer count."""
        d, c = self.dimension, self.response_dimension
        entry = self._coeff.get(())
        if entry is None:
            return np.zeros((d, d)), np.zeros((d, c)), 0
        value = entry.value
        if not np.isfinite(value).all():
            raise FloatingPointError("nonfinite current moments")
        count_float = float(value[-1])
        if count_float < 0 or not count_float.is_integer():
            raise FloatingPointError("current count is not an exact nonnegative integer")
        n = int(count_float)
        tri = np.triu_indices(d)
        q = d * (d + 1) // 2
        gram = np.zeros((d, d), dtype=np.float64)
        gram[tri] = value[:q]
        gram[(tri[1], tri[0])] = value[:q]
        cross = value[q:-1].reshape(d, c).copy()
        # Empty retained selection has the explicitly prescribed zero head.
        # Residual moment drift is still exposed by statistics(), never erased.
        return gram, cross, n

    def solve_ridge(self, lambda_reg: float) -> np.ndarray:
        """Use the identical FP64 decoder used by the incremental baseline."""
        return self.decode(lambda_reg).weights

    def moments(self):
        """Common runner representation; no row reread is involved."""
        from .core import RidgeMoments
        return RidgeMoments(*self.statistics())

    def decode(self, lambda_reg: float):
        """Common Cholesky solver, including its residual diagnostics.

        At exactly zero count, the target is the declared zero head.  Floating
        cancellation residue remains visible in ``statistics()`` for the audit,
        but is not passed as a nonempty objective to the common decoder.  No
        nonempty-state coefficient or moment is rounded/thresholded here.
        """
        from .core import RidgeMoments, solve_ridge
        moments = self.moments()
        if moments.count == 0:
            moments = RidgeMoments(np.zeros_like(moments.gram),
                                   np.zeros_like(moments.cross), 0)
        return solve_ridge(moments, lambda_reg)

    def coefficients(self) -> dict[tuple[int, ...], np.ndarray]:
        """Expensive audit-only copy; do not call during measured repair."""
        return {key: entry.value.copy() for key, entry in self._coeff.items()}

    def check_invariants(self) -> None:
        """Full structural scan intended for audits outside timed updates."""
        assert 0 <= self.horizon <= len(self.alive)
        expected_inverse, expected_degrees = {}, {}
        for key, entry in self._coeff.items():
            assert entry.key == key == tuple(sorted(set(key)))
            assert len(key) <= self.horizon and set(key) <= self.alive
            assert entry.value.shape == (self.packed_dimension,)
            assert entry.value.dtype == np.float64 and entry.value.flags.owndata
            assert np.isfinite(entry.value).all() and np.any(entry.value != 0)
            expected_degrees.setdefault(len(key), set()).add(entry)
            for u in key:
                expected_inverse.setdefault(u, set()).add(entry)
        assert expected_inverse == self._inverse
        assert expected_degrees == self._degrees

    def accounting(self) -> dict:
        """Live bytes and work potentials; process RSS must be measured outside.

        Python bytes are a recursive, de-duplicated ``sys.getsizeof`` estimate,
        including arrays' owned buffers.  They exclude allocator slack and
        external shared libraries.  No old build forecast is kept in state.
        """
        numeric = sum(entry.value.nbytes for entry in self._coeff.values())
        seen = set()

        def size(obj):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            total = sys.getsizeof(obj)
            if isinstance(obj, dict):
                total += sum(size(k) + size(v) for k, v in obj.items())
            elif isinstance(obj, (set, tuple, list)):
                total += sum(map(size, obj))
            elif isinstance(obj, _Entry):
                total += size(obj.key) + size(obj.value)
            return total

        total = sys.getsizeof(self)
        total += sum(size(getattr(self, name)) for name in self.__slots__)
        degrees = [len(key) for key in self._coeff]
        return {
            "method": "P-index-python-numpy",
            "state_contract": "current coefficients and head; no selected-ID recovery",
            "coefficient_count": self.coefficient_count,
            "packed_dimension": self.packed_dimension,
            "numeric_coefficient_bytes": numeric,
            "python_live_bytes_estimate": total,
            "python_metadata_bytes_estimate": total - numeric,
            "alive_public_identifiers": len(self.alive),
            "remaining_horizon": self.horizon,
            "key_incidences": sum(degrees),
            "quadratic_key_potential": sum(k * (k + 1) // 2 for k in degrees),
            "retained_feature_rows": 0,
            "retained_target_rows": 0,
            "retained_original_graph": False,
            "numeric_guarantee": "FP64 diagnostic fidelity; not certified roundoff",
        }

    def save(self, path: str | Path) -> int:
        """Write the current state only; charge this full export separately.

        Sorted keys and IDs give a canonical *ordering*, not bitwise equality
        across floating accumulation histories.  Uncompressed NPZ avoids pickle.
        Serialization allocates a second dense coefficient matrix temporarily;
        its peak memory must be counted.  This is intentionally explicit.
        """
        keys = sorted(self._coeff)
        indptr = np.zeros(len(keys) + 1, dtype=np.int64)
        for i, key in enumerate(keys):
            indptr[i + 1] = indptr[i] + len(key)
        indices = np.fromiter((u for key in keys for u in key), dtype=np.int64, count=int(indptr[-1]))
        values = np.stack([self._coeff[key].value for key in keys]) if keys else np.empty((0, self.packed_dimension))
        meta = np.array([self.universe_size, self.horizon, self.dimension, self.response_dimension], dtype=np.int64)
        destination = Path(path)
        with destination.open("wb") as stream:
            np.savez(stream, format_version=np.array(1, dtype=np.int64), metadata=meta,
                     alive=np.array(sorted(self.alive), dtype=np.int64),
                     indptr=indptr, indices=indices, values=values)
        return destination.stat().st_size

    @classmethod
    def load(cls, path: str | Path) -> "IndexedRidgeSummary":
        """Load a checkpoint produced by save; never load Python pickles."""
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["format_version"]) != 1:
                raise ValueError("unsupported summary format")
            meta = data["metadata"]
            if meta.shape != (4,):
                raise ValueError("invalid summary metadata")
            universe, h, d, c = map(int, meta)
            if min(universe, h) < 0 or d <= 0 or c <= 0 or h > universe:
                raise ValueError("invalid summary dimensions or horizon")
            state = cls.__new__(cls)
            state.universe_size, state.horizon = universe, h
            state.dimension, state.response_dimension = d, c
            state.packed_dimension = d * (d + 1) // 2 + d * c + 1
            alive_array = data["alive"]
            state.alive = set(map(int, alive_array))
            if alive_array.ndim != 1 or len(state.alive) != len(alive_array) or any(u < 0 or u >= universe for u in state.alive):
                raise ValueError("invalid public membership")
            ptr, indices, values = data["indptr"], data["indices"], data["values"]
            if ptr.ndim != 1 or len(ptr) == 0 or ptr[0] != 0 or np.any(np.diff(ptr) < 0) or ptr[-1] != len(indices):
                raise ValueError("invalid coefficient index")
            if values.shape != (len(ptr) - 1, state.packed_dimension) or values.dtype != np.float64:
                raise ValueError("invalid coefficient values")
            state._coeff, state._inverse, state._degrees = {}, {}, {}
            for i in range(len(values)):
                key = tuple(map(int, indices[ptr[i]:ptr[i+1]]))
                if key in state._coeff:
                    raise ValueError("duplicate coefficient key")
                entry = _Entry(key, values[i].copy())
                state._coeff[key] = entry
                state._degrees.setdefault(len(key), set()).add(entry)
                for u in key:
                    state._inverse.setdefault(u, set()).add(entry)
        state.check_invariants()
        return state


# Explicit alias for the construction preflight, convenient for runners.
IndexedRidgeSummary.preflight = staticmethod(preflight)
