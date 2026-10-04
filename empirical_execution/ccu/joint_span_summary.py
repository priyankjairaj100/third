"""Indexed ridge coefficients compressed in their joint Gram/cross span.

For each monomial, persist only (U, A, B, n), with G=U A U.T and H=U B,
or the cheaper packed dense (G,H,n).  U spans the current aggregate's joint
range; signed Gram cancellation must NOT discard a surviving cross moment.
No contributors, feature/target rows, original graph, or earlier states persist.

The algebraic equivalence and rank bounds are exact-arithmetic statements.
QR/SVD arithmetic is FP64 here.  Singular values are discarded ONLY when
exactly zero, never using a numerical-rank tolerance.  Consequently roundoff
can inflate rank, and numerical fidelity needs independent-oracle checks.
Basis signs are normalized but degenerate subspaces are not byte canonical.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import sys
from pathlib import Path
import zipfile
from typing import Iterable, Sequence

import numpy as np

from .summary import IndexedRidgeSummary, SummaryMemoryError, _integer, _metadata


def _dense_size(d: int, c: int) -> int:
    return d * (d + 1) // 2 + d * c


def _compact_size(d: int, c: int, rank: int) -> int:
    return d * rank + rank * (rank + 1) // 2 + rank * c


def _unpack(packed: np.ndarray, d: int) -> np.ndarray:
    out = np.zeros((d, d), dtype=np.float64)
    tri = np.triu_indices(d)
    out[tri] = packed
    out[(tri[1], tri[0])] = packed
    return out


def _signed_columns(q: np.ndarray, r: np.ndarray):
    """Normalize each Q column's largest-absolute entry to nonnegative."""
    if q.shape[1]:
        sign = np.sign(q[np.argmax(np.abs(q), axis=0), np.arange(q.shape[1])])
        sign[sign == 0] = 1
        q *= sign
        r *= sign[:, None]
    return q, r


def _canonical_range_coordinates(u: np.ndarray):
    """Canonical projector-column basis, without forming the d-by-d projector.

    In exact arithmetic, choose the largest residual projector diagonal, with
    smallest ambient-coordinate index breaking ties; Gram--Schmidt the chosen
    projector columns with positive pivot. If U is replaced by U R for any
    orthogonal R, this produces the same ambient basis U O. Thus the factors
    depend only on the current joint range, not contributor/history axes.
    Floating QR/SVD and pivot ties still preclude byte-canonical claims.
    """
    d, r = u.shape
    original = u.T
    residual = original.copy()
    rotation = np.zeros((r, r), dtype=np.float64)
    for j in range(r):
        squared_norms = np.einsum("ij,ij->j", residual, residual)
        pivot = int(np.argmax(squared_norms))  # first index breaks exact ties
        direction = original[:, pivot].copy()
        if j:
            previous = rotation[:, :j]
            # Reorthogonalization adds no tolerance or rank dropping.
            direction -= previous @ (previous.T @ direction)
            direction -= previous @ (previous.T @ direction)
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length == 0.0:
            raise FloatingPointError("canonical joint-span pivot lost rank")
        direction /= length
        rotation[:, j] = direction
        residual -= direction[:, None] * (direction @ residual)[None, :]
        residual -= direction[:, None] * (direction @ residual)[None, :]
    return rotation


@dataclass(frozen=True)
class JointSpanPreflight:
    record_count: int
    feature_dimension: int
    response_dimension: int
    universe_size: int
    horizon: int
    eligible_records: int
    structural_keys: int
    coefficient_bytes_upper_bound: int
    dense_coefficient_bytes: int
    maximum_contributors_per_key: int
    count_scalars: int
    note: str = "Numeric-array allocation envelope only; Python counts, metadata, caller rows and temporary workspace excluded."

    def to_dict(self):
        return asdict(self)


def preflight(blockers, feature_dimension, response_dimension, horizon,
              owners=None, universe_size=None) -> JointSpanPreflight:
    d = _integer(feature_dimension, "feature_dimension")
    c = _integer(response_dimension, "response_dimension")
    if d < 1 or c < 1:
        raise ValueError("feature and response dimensions must be positive")
    universe, h, records, keys, _ = _metadata(blockers, owners, horizon, universe_size)
    contributions = {key: 0 for key in keys}
    for _, tail, head in records:
        contributions[tail] += 1
        if head is not None:
            contributions[head] += 1
    return _forecast(len(blockers), d, c, universe, h, records, contributions)


def _forecast(n, d, c, universe, h, records, contributions):
    return JointSpanPreflight(
        n, d, c, universe, h, len(records), len(contributions),
        8 * sum(min(_dense_size(d, c), _compact_size(d, c, min(d, m)))
                for m in contributions.values()),
        8 * len(contributions) * _dense_size(d, c),
        max(contributions.values(), default=0), len(contributions))


class _Value:
    __slots__ = ("d", "c", "count", "u", "a", "b", "dense")

    def __init__(self, d, c, count):
        self.d, self.c, self.count = d, c, int(count)
        self.u = self.a = self.b = self.dense = None

    @classmethod
    def from_dense(cls, gram, cross, count):
        d, c = cross.shape
        out = cls(d, c, count)
        out.dense = np.concatenate((gram[np.triu_indices(d)], cross.ravel()))
        return out

    @classmethod
    def from_core(cls, u, a, b, count):
        d, r = u.shape
        c = b.shape[1]
        # Current aggregate range([G,H]), not range(G) alone.  An SVD of this
        # r-by-(r+c) core avoids any d-by-d allocation in the compact path.
        if r:
            left, singular, _ = np.linalg.svd(np.concatenate((a, b), axis=1),
                                            full_matrices=False)
            keep = singular != 0.0
            if not np.all(keep):
                basis = left[:, keep]
                u = u @ basis
                a, b = basis.T @ a @ basis, basis.T @ b
                r = u.shape[1]
        if _compact_size(d, c, r) >= _dense_size(d, c):
            return cls.from_dense(u @ a @ u.T, u @ b, count)
        out = cls(d, c, count)
        # Remove contributor/history coordinate rotations in exact arithmetic.
        # The pivot is positive by construction for each projector column.
        if r:
            rotation = _canonical_range_coordinates(u)
            u = u @ rotation
            a = rotation.T @ a @ rotation
            b = rotation.T @ b
        # Averaging paired entries removes only anti-symmetric roundoff, which
        # has no symmetric Gram interpretation. No eigenvalue clipping occurs.
        a = (a + a.T) * 0.5
        out.u = np.array(u, dtype=np.float64, order="C", copy=True)
        out.a = np.array(a[np.triu_indices(r)], dtype=np.float64, copy=True)
        out.b = np.array(b, dtype=np.float64, order="C", copy=True)
        return out

    @classmethod
    def from_rows(cls, features, targets, indices, signs):
        d, c = features.shape[1], targets.shape[1]
        count = sum(signs)
        idx = np.asarray(indices, dtype=np.int64)
        s = np.asarray(signs, dtype=np.float64)
        rmax = min(d, len(idx))
        if _compact_size(d, c, rmax) >= _dense_size(d, c):
            # Bounded gather: a large all-selected constant key must not copy
            # its entire corpus of d-dimensional rows as hidden workspace.
            gram, cross = np.zeros((d, d)), np.zeros((d, c))
            for start in range(0, len(idx), 256):
                part = idx[start:start + 256]
                weights = s[start:start + 256, None]
                x = np.asarray(features[part], dtype=np.float64)
                gram += x.T @ (weights * x)
                cross += x.T @ (weights * targets[part])
            return cls.from_dense(gram, cross, count)
        x = np.asarray(features[idx], dtype=np.float64)
        y = targets[idx]
        q, r = _signed_columns(*np.linalg.qr(x.T, mode="reduced"))
        return cls.from_core(q, (r * s) @ r.T, r @ (s[:, None] * y), count)

    @property
    def rank(self):
        return self.d if self.dense is not None else self.u.shape[1]

    def arrays(self):
        return (self.dense,) if self.dense is not None else (self.u, self.a, self.b)

    def nbytes(self):
        return sum(a.nbytes for a in self.arrays())

    def zero(self):
        if self.count != 0:
            return False
        if self.dense is not None:
            return not np.any(self.dense != 0)
        return not np.any(self.a != 0) and not np.any(self.b != 0)

    def moments(self):
        q = self.d * (self.d + 1) // 2
        if self.dense is not None:
            return _unpack(self.dense[:q], self.d), self.dense[q:].reshape(self.d, self.c).copy(), self.count
        a = _unpack(self.a, self.rank)
        return self.u @ a @ self.u.T, self.u @ self.b, self.count

    def packed(self):
        if self.dense is not None:
            return np.concatenate((self.dense, [float(self.count)]))
        g, h, n = self.moments()
        return np.concatenate((g[np.triu_indices(self.d)], h.ravel(), [float(n)]))

    def add(self, other):
        """Create current aggregate only; contributors/history are discarded."""
        assert (self.d, self.c) == (other.d, other.c)
        count = self.count + other.count
        # Dense states stay dense. This avoids a hidden O(d^3) rank test on
        # every dense collision; consequently current-eligibility bounds are
        # not claimed for this implementation.
        if (self.dense is not None or other.dense is not None or
                _compact_size(self.d, self.c, min(self.d, self.rank + other.rank))
                >= _dense_size(self.d, self.c)):
            g1, h1, _ = self.moments()
            g2, h2, _ = other.moments()
            return _Value.from_dense(g1 + g2, h1 + h2, count)
        # QR removes redundancy of coordinate systems; SVD of the resulting
        # joint aggregate, inside from_core, removes exactly zero directions.
        joined = np.concatenate((self.u, other.u), axis=1)
        q, r = _signed_columns(*np.linalg.qr(joined, mode="reduced"))
        r1, r2 = r[:, :self.rank], r[:, self.rank:]
        a = r1 @ _unpack(self.a, self.rank) @ r1.T
        a += r2 @ _unpack(other.a, other.rank) @ r2.T
        b = r1 @ self.b + r2 @ other.b
        return _Value.from_core(q, a, b, count)


class _Entry:
    __slots__ = ("key", "value")

    def __init__(self, key, value):
        self.key, self.value = key, value


class JointSpanRidgeSummary(IndexedRidgeSummary):
    """Same indexed substitution protocol, smaller aggregate payloads."""
    __slots__ = ()

    @classmethod
    def build(cls, features, targets, blockers: Sequence[Iterable[int]], horizon,
              owners=None, universe_size=None, max_state_bytes=None,
              max_coefficient_bytes=None):
        x, y = np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) != len(blockers):
            raise ValueError("features, targets, and blockers must have matching dimensions")
        if x.shape[1] < 1 or y.shape[1] < 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("positive dimensions and finite feature/target values required")
        if len(x) > 2**52:
            raise ValueError("FP64 audit exact-count guard exceeded")
        universe, h, records, keys, _ = _metadata(blockers, owners, horizon, universe_size)
        grouped = {key: ([], []) for key in keys}
        for row, tail, head in records:
            for key, sign in ((tail, 1), (head, -1)):
                if key is not None:
                    grouped[key][0].append(row)
                    grouped[key][1].append(sign)
        d, c = x.shape[1], y.shape[1]
        forecast = _forecast(len(x), d, c, universe, h, records,
                             {key: len(vals[0]) for key, vals in grouped.items()})
        if max_state_bytes is not None and max_coefficient_bytes is not None:
            raise ValueError("supply only one numeric-array allocation cap")
        cap = max_state_bytes if max_state_bytes is not None else max_coefficient_bytes
        if cap is not None:
            cap = _integer(cap, "max_state_bytes")
            if cap < 0:
                raise ValueError("max_state_bytes must be nonnegative")
            if forecast.coefficient_bytes_upper_bound > cap:
                raise SummaryMemoryError(forecast, cap)
        state = cls.__new__(cls)
        state.universe_size, state.horizon = universe, h
        state.dimension, state.response_dimension = d, c
        state.packed_dimension = _dense_size(d, c) + 1
        state.alive = set(range(universe))
        state._coeff, state._inverse, state._degrees = {}, {}, {}
        for key in sorted(grouped):
            value = _Value.from_rows(x, y, *grouped[key])
            if value.zero():
                continue
            entry = _Entry(key, value)
            state._coeff[key] = entry
            state._degrees.setdefault(len(key), set()).add(entry)
            for u in key:
                state._inverse.setdefault(u, set()).add(entry)
        return state

    def _delete_one(self, u, meter):
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
            else:
                self._remove_incidences(entry, meter)
                target.value = target.value.add(entry.value)
                meter["coefficient_collisions"] += 1
                # Dense-equivalent operation count, not an actual FLOP claim.
                meter["scalar_additions"] += self.packed_dimension
                if target.value.zero():
                    self._drop(target, meter)
                    meter["exact_cancellations"] += 1
        while old_horizon in self._degrees:
            self._drop(self._degrees[old_horizon].pop(), meter)
            meter["horizon_pruned_keys"] += 1
        self.alive.remove(u)
        self.horizon -= 1

    def statistics(self):
        entry = self._coeff.get(())
        if entry is None:
            return (np.zeros((self.dimension, self.dimension)),
                    np.zeros((self.dimension, self.response_dimension)), 0)
        gram, cross, count = entry.value.moments()
        if count < 0 or not np.isfinite(gram).all() or not np.isfinite(cross).all():
            raise FloatingPointError("invalid current moments/count")
        return gram, cross, count

    def coefficient(self, key):
        """Audit-only packed dense copy of one coefficient, or None."""
        entry = self._coeff.get(tuple(key))
        return None if entry is None else entry.value.packed()

    def decode_compact(self, lambda_reg):
        """Decode through the current joint span without materializing d-by-d G.

        This is an optional solver, audited separately from common decode().
        Exact arithmetic gives W = U (A + lambda*n I)^(-1) B. The residual is
        evaluated with the actual factor operator, including measured loss of
        U orthogonality; it remains a diagnostic, not a roundoff certificate.
        Dense fallback coefficients use the common dense Cholesky decoder.
        """
        from scipy.linalg import cho_factor, cho_solve
        from .core import RidgeSolution
        lam = float(lambda_reg)
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError("lambda_reg must be positive and finite")
        entry = self._coeff.get(())
        if entry is None or entry.value.count == 0:
            return RidgeSolution(np.zeros((self.dimension, self.response_dimension)),
                                 0.0, 0.0, 0.0, lam, 0,
                                 "joint-span Cholesky, FP64; empty target")
        value = entry.value
        if value.count < 0:
            raise FloatingPointError("negative current count")
        if value.dense is not None:
            return self.decode(lam)
        a = _unpack(value.a, value.rank)
        alpha = lam * value.count
        regularized = a.copy()
        regularized.flat[::value.rank + 1] += alpha
        if value.rank:
            factor = cho_factor(regularized, lower=True, check_finite=True)
            small = cho_solve(factor, value.b, check_finite=True)
            weights = value.u @ small
        else:
            weights = np.zeros((self.dimension, self.response_dimension))
        residual_matrix = (value.u @ (a @ (value.u.T @ weights) - value.b)
                           + alpha * weights)
        residual = float(np.linalg.norm(residual_matrix))
        return RidgeSolution(weights, residual, residual / alpha,
                             residual / value.count, lam, value.count,
                             "joint-span Cholesky, FP64; factor residual diagnostic")

    def coefficient_keys(self):
        return tuple(sorted(self._coeff))

    def iter_coefficients(self):
        """Materialize at most one packed coefficient at a time for audits."""
        for key in sorted(self._coeff):
            yield key, self._coeff[key].value.packed()

    def coefficients(self):
        """Expensive compatibility audit; prefer iter_coefficients at large d."""
        return dict(self.iter_coefficients())

    def check_invariants(self):
        assert 0 <= self.horizon <= len(self.alive)
        inverse, degrees = {}, {}
        for key, entry in self._coeff.items():
            assert entry.key == key == tuple(sorted(set(key)))
            assert len(key) <= self.horizon and set(key) <= self.alive
            value = entry.value
            assert (value.d, value.c) == (self.dimension, self.response_dimension)
            assert isinstance(value.count, int) and not value.zero()
            assert all(a.dtype == np.float64 and a.flags.owndata and np.isfinite(a).all()
                       for a in value.arrays())
            if value.dense is not None:
                assert value.dense.shape == (_dense_size(value.d, value.c),)
            else:
                rank = value.rank
                assert value.u.shape == (value.d, rank)
                assert value.a.shape == (rank * (rank + 1) // 2,)
                assert value.b.shape == (rank, value.c)
                assert value.nbytes() < 8 * _dense_size(value.d, value.c)
            degrees.setdefault(len(key), set()).add(entry)
            for u in key:
                inverse.setdefault(u, set()).add(entry)
        assert inverse == self._inverse and degrees == self._degrees

    def accounting(self):
        numeric = sum(entry.value.nbytes() for entry in self._coeff.values())
        seen = set()
        def size(obj):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            out = sys.getsizeof(obj)
            if isinstance(obj, dict):
                out += sum(size(k) + size(v) for k, v in obj.items())
            elif isinstance(obj, (set, tuple, list)):
                out += sum(size(v) for v in obj)
            elif isinstance(obj, (_Value, _Entry)):
                out += sum(size(getattr(obj, slot)) for slot in obj.__slots__)
            return out
        total = sys.getsizeof(self) + sum(size(getattr(self, name))
                                         for name in IndexedRidgeSummary.__slots__)
        values = [entry.value for entry in self._coeff.values()]
        compact = [v for v in values if v.dense is None]
        degrees = [len(k) for k in self._coeff]
        return {
            "method": "P-joint-span-python-numpy",
            "coefficient_count": len(values), "compact_coefficients": len(compact),
            "dense_coefficients": len(values) - len(compact),
            "compact_rank_sum": sum(v.rank for v in compact),
            "maximum_compact_rank": max((v.rank for v in compact), default=0),
            "numeric_coefficient_bytes": numeric, "all_array_bytes": numeric,
            "numeric_total_bytes": numeric,
            "integer_count_scalars": len(values),
            "integer_counts_included_in_array_bytes": False,
            "python_live_bytes_estimate": total,
            "python_metadata_bytes_estimate": total - numeric,
            "alive_public_identifiers": len(self.alive),
            "remaining_horizon": self.horizon,
            "key_incidences": sum(degrees),
            "quadratic_key_potential": sum(k * (k + 1) // 2 for k in degrees),
            "retained_feature_rows": 0, "retained_target_rows": 0,
            "retained_original_graph": False,
            "state_contract": "Current aggregate coefficient factors; no selected-ID recovery",
            "numeric_guarantee": "FP64 diagnostic fidelity; no numerical rank truncation or certified roundoff",
            "rank_envelope": "Initial eligible contribution envelope; dense keys are not recompressed",
        }

    def save(self, path):
        """Stream current factors to uncompressed NPZ without stacked payloads.

        Non-pickle arrays only. Sorted keys canonicalize ordering, not QR bases
        or floating accumulation. Export uses O(key incidences) metadata and
        bounded NumPy writer buffering instead of duplicating the whole state.
        """
        keys = sorted(self._coeff)
        pointers = np.zeros(len(keys) + 1, dtype=np.int64)
        for i, key in enumerate(keys):
            pointers[i + 1] = pointers[i] + len(key)
        indices = np.fromiter((u for key in keys for u in key), dtype=np.int64,
                              count=int(pointers[-1]))
        modes = np.array([self._coeff[k].value.dense is not None for k in keys],
                         dtype=np.uint8)
        counts = np.array([self._coeff[k].value.count for k in keys], dtype=np.int64)
        destination = Path(path)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            def write(name, value):
                with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
            write("format_version", np.array(1, dtype=np.int64))
            write("metadata", np.array([self.universe_size, self.horizon,
                                         self.dimension, self.response_dimension], dtype=np.int64))
            write("alive", np.array(sorted(self.alive), dtype=np.int64))
            write("indptr", pointers)
            write("indices", indices)
            write("modes", modes)
            write("counts", counts)
            for i, key in enumerate(keys):
                value = self._coeff[key].value
                prefix = f"coefficient_{i:08d}_"
                if value.dense is not None:
                    write(prefix + "dense", value.dense)
                else:
                    write(prefix + "u", value.u)
                    write(prefix + "a", value.a)
                    write(prefix + "b", value.b)
        return destination.stat().st_size

    @classmethod
    def load(cls, path):
        """Read this class's non-pickle current-state checkpoint."""
        with np.load(Path(path), allow_pickle=False) as archive:
            if int(archive["format_version"]) != 1:
                raise ValueError("unsupported joint-span checkpoint version")
            metadata = archive["metadata"]
            if metadata.shape != (4,) or metadata.dtype != np.int64:
                raise ValueError("invalid metadata")
            universe, h, d, c = map(int, metadata)
            if min(universe, h) < 0 or h > universe or d < 1 or c < 1:
                raise ValueError("invalid dimensions/horizon")
            alive, ptr, indices = archive["alive"], archive["indptr"], archive["indices"]
            modes, counts = archive["modes"], archive["counts"]
            for name, array in (("alive", alive), ("indptr", ptr), ("indices", indices), ("counts", counts)):
                if array.dtype != np.int64 or array.ndim != 1:
                    raise ValueError(f"invalid {name}")
            if (len(ptr) < 1 or ptr[0] != 0 or np.any(np.diff(ptr) < 0)
                    or ptr[-1] != len(indices) or modes.shape != counts.shape
                    or len(modes) + 1 != len(ptr) or np.any(modes > 1)):
                raise ValueError("invalid coefficient index")
            state = cls.__new__(cls)
            state.universe_size, state.horizon = universe, h
            state.dimension, state.response_dimension = d, c
            state.packed_dimension = _dense_size(d, c) + 1
            state.alive = set(map(int, alive))
            if len(state.alive) != len(alive) or any(u < 0 or u >= universe for u in state.alive):
                raise ValueError("invalid public membership")
            state._coeff, state._inverse, state._degrees = {}, {}, {}
            for i, mode in enumerate(modes):
                key = tuple(map(int, indices[ptr[i]:ptr[i + 1]]))
                if key in state._coeff:
                    raise ValueError("duplicate coefficient key")
                value = _Value(d, c, int(counts[i]))
                prefix = f"coefficient_{i:08d}_"
                if mode:
                    value.dense = archive[prefix + "dense"].copy()
                else:
                    value.u = archive[prefix + "u"].copy()
                    value.a = archive[prefix + "a"].copy()
                    value.b = archive[prefix + "b"].copy()
                entry = _Entry(key, value)
                state._coeff[key] = entry
                state._degrees.setdefault(len(key), set()).add(entry)
                for u in key:
                    state._inverse.setdefault(u, set()).add(entry)
        state.check_invariants()
        return state


JointSpanRidgeSummary.preflight = staticmethod(preflight)
