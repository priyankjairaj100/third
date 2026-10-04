"""Adaptive signed-Gram coefficient storage for exact finite-horizon repair.

Each *current* polynomial coefficient is (G, H, n), with G either a packed
symmetric matrix or F diag(signs) F.T, H a separate dense cross moment, and n
an exact Python integer. There is no separate feature/label table, contributor
ID table, original graph, history, or shared original-data basis; repair reads
only current coefficient factors. Rank-one factors can equal input features
or reveal values inherent in their coefficients. This is not a privacy mechanism.

In exact arithmetic QR/eigendecomposition is lossless. The implementation is
FP64: every nonzero computed eigenvalue is retained, without tolerance-based
rank truncation. Numerical comparisons with a fresh oracle remain necessary;
there is no bitwise canonicality or certified roundoff claim. Dense fallback
avoids storing a factor whose arrays cost as much as a packed Gram. Building
high-degree keys and merging wide factors choose dense *before* concatenating.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import sys
import numpy as np

from .summary import _integer, _metadata, _incidence_rank, SummaryPreflight, SummaryMemoryError


@dataclass(frozen=True)
class FactoredPreflight(SummaryPreflight):
    dense_coefficient_bytes_upper_bound: int
    factor_columns_before_compression_upper_bound: int
    storage_rule: str = "per key: min(packed Gram bytes, signed factor bytes) + dense cross"


def _forecast(blockers, d, c, horizon, owners=None, universe_size=None):
    d, c = _integer(d, "feature_dimension"), _integer(c, "response_dimension")
    if d <= 0 or c <= 0:
        raise ValueError("feature and response dimensions must be positive")
    universe, h, records, keys, same_source = _metadata(blockers, owners, horizon, universe_size)
    degree = {key: 0 for key in keys}
    for _, tail, head in records:
        degree[tail] += 1
        if head is not None:
            degree[head] += 1
    q = d * (d + 1) // 2
    compact = sum(min(q * 8, k * (d * 8 + 1)) + d * c * 8 for k in degree.values())
    # Counts are Python integers, included in heap accounting rather than array bytes.
    result = FactoredPreflight(
        len(blockers), d, c, universe, h, len(records), same_source,
        len(keys), sum(map(len, keys)), _incidence_rank(records, keys), q+d*c+1,
        compact, 8*(3*d*d+3*d*c), 8*(3*d*d+3*d*c),
        len(blockers)*d*4, len(blockers)*c*8,
        len(keys)*(q+d*c)*8, sum(degree.values()),
    )
    return result, records, degree


def preflight(blockers, feature_dimension, response_dimension, horizon,
              owners=None, universe_size=None):
    """Conservative live-array construction bound; excludes heap and workspace.

    Per-key contribution counts bound retained factor widths. QR/eigh may keep
    spurious tiny floating eigenvalues, but cannot exceed the input width. The
    cap is a construction bound, not a peak-RSS or all-future-state guarantee.
    Blocker iterables are consumed once.
    """
    return _forecast(blockers, feature_dimension, response_dimension, horizon,
                     owners, universe_size)[0]


class _Coefficient:
    __slots__ = ("factor", "signs", "packed", "cross", "count")

    def __init__(self, factor, signs, packed, cross, count):
        self.factor, self.signs, self.packed = factor, signs, packed
        self.cross, self.count = cross, int(count)

    @property
    def width(self):
        return 0 if self.factor is None else self.factor.shape[1]

    def arrays(self):
        return [a for a in (self.factor, self.signs, self.packed, self.cross) if a is not None]

    def gram_packed(self, dimension):
        if self.packed is not None:
            return self.packed.copy()
        tri = np.triu_indices(dimension)
        out = np.zeros(len(tri[0]), dtype=np.float64)
        for i in range(self.width):
            z = self.factor[:, i]
            out += int(self.signs[i]) * z[tri[0]] * z[tri[1]]
        return out

    def is_zero(self):
        gram_zero = not np.any(self.packed != 0) if self.packed is not None else self.width == 0
        return self.count == 0 and gram_zero and not np.any(self.cross != 0)


def _compress(factor, signs):
    """Lossless spectral refactorization in exact arithmetic; owned FP64 arrays."""
    d, width = factor.shape
    if width == 0:
        return np.empty((d, 0), dtype=np.float64), np.empty(0, dtype=np.int8)
    # Rank-one construction is already an aggregate moment factor. Avoid a
    # needless eigensolve and its rounding, including exact-zero columns.
    if width == 1:
        if not np.any(factor[:, 0] != 0):
            return np.empty((d, 0), dtype=np.float64), np.empty(0, dtype=np.int8)
        out = np.array(factor, dtype=np.float64, order="C", copy=True)
        pivot = int(np.argmax(np.abs(out[:, 0])))
        if out[pivot, 0] < 0:
            out[:, 0] *= -1
        return out, np.array(signs, dtype=np.int8, copy=True)
    q, r = np.linalg.qr(factor, mode="reduced")
    small = (r * signs[None, :]) @ r.T
    # Only remove asymmetry caused by multiplication roundoff, not eigenvalues.
    small = (small + small.T) * 0.5
    values, vectors = np.linalg.eigh(small)
    keep = values != 0.0
    out = (q @ vectors[:, keep]) * np.sqrt(np.abs(values[keep]))[None, :]
    # Fix each column sign deterministically; repeated eigenspaces still have
    # no byte-canonical basis and are explicitly outside that claim.
    for j in range(out.shape[1]):
        pivot = int(np.argmax(np.abs(out[:, j])))
        if out[pivot, j] < 0:
            out[:, j] *= -1
    return np.array(out, dtype=np.float64, order="C", copy=True), np.array(np.sign(values[keep]), dtype=np.int8, copy=True)


class _Entry:
    __slots__ = ("key", "value")
    def __init__(self, key, value):
        self.key, self.value = key, value


class FactoredRidgeSummary:
    """Current coefficients, indexed keys, public membership; no row cache.

    Moment factors may inherently reveal input values already determined by
    their coefficients. This representation is not a privacy guarantee.
    """
    __slots__ = ("universe_size", "horizon", "dimension", "response_dimension",
                 "packed_dimension", "alive", "_coeff", "_inverse", "_degrees")

    @classmethod
    def build(cls, features, targets, blockers: Sequence[Iterable[int]], horizon: int,
              owners=None, universe_size=None, max_coefficient_bytes=None):
        x, y = np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) != len(blockers):
            raise ValueError("features, targets, and blockers must have matching record dimensions")
        if x.shape[1] <= 0 or y.shape[1] <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("invalid feature or target values/dimensions")
        d, c = x.shape[1], y.shape[1]
        forecast, records, degree = _forecast(blockers, d, c, horizon, owners, universe_size)
        if max_coefficient_bytes is not None:
            cap = _integer(max_coefficient_bytes, "max_coefficient_bytes")
            if cap < 0:
                raise ValueError("max_coefficient_bytes must be nonnegative")
            if forecast.coefficient_bytes_upper_bound > cap:
                raise SummaryMemoryError(forecast, cap)
        state = cls.__new__(cls)
        state.universe_size, state.horizon = forecast.universe_size, forecast.horizon
        state.dimension, state.response_dimension = d, c
        state.packed_dimension = d*(d+1)//2+d*c+1
        state.alive = set(range(state.universe_size))
        state._coeff, state._inverse, state._degrees = {}, {}, {}
        # Structural contributor lists are temporary construction workspace.
        contributors = {key: [] for key in degree}
        for v, tail, head in records:
            contributors[tail].append((v, 1))
            if head is not None:
                contributors[head].append((v, -1))
        del records
        packed_bytes = d*(d+1)//2*8
        tri = np.triu_indices(d)
        for key, rows in contributors.items():
            width = len(rows)
            cross = np.zeros((d, c), dtype=np.float64)
            count = sum(sign for _, sign in rows)
            if width*(d*8+1) >= packed_bytes:
                packed = np.zeros(d*(d+1)//2, dtype=np.float64)
                for v, sign in rows:
                    z = x[v].astype(np.float64)
                    packed += sign*z[tri[0]]*z[tri[1]]
                    cross += sign*np.outer(z, y[v])
                value = _Coefficient(None, None, packed, cross, count)
            else:
                factor = np.empty((d, width), dtype=np.float64)
                signs = np.empty(width, dtype=np.int8)
                for i, (v, sign) in enumerate(rows):
                    factor[:, i], signs[i] = x[v], sign
                    cross += sign*np.outer(factor[:, i], y[v])
                factor, signs = _compress(factor, signs)
                value = _Coefficient(factor, signs, None, cross, count)
            if not value.is_zero():
                state._insert(_Entry(key, value))
        return state

    def _insert(self, entry):
        self._coeff[entry.key] = entry
        self._degrees.setdefault(len(entry.key), set()).add(entry)
        for u in entry.key:
            self._inverse.setdefault(u, set()).add(entry)

    @property
    def coefficient_count(self):
        return len(self._coeff)

    def _merge(self, left, right, meter):
        d = self.dimension
        width = left.width+right.width
        dense = left.packed is not None or right.packed is not None or width*(d*8+1) >= d*(d+1)//2*8
        if dense:
            # Materialize one packed accumulator, never the combined factor.
            out = left.gram_packed(d)
            if right.packed is not None:
                out += right.packed
            else:
                tri = np.triu_indices(d)
                for i in range(right.width):
                    z = right.factor[:, i]
                    out += int(right.signs[i])*z[tri[0]]*z[tri[1]]
            if left.packed is None:
                meter["dense_promotions"] += 1
            factor = signs = None
            packed = out
        else:
            factor = np.concatenate((left.factor, right.factor), axis=1)
            signs = np.concatenate((left.signs, right.signs))
            factor, signs = _compress(factor, signs)
            packed = None
            meter["factor_recompressions"] += 1
        meter["factor_columns_merged"] += width
        return _Coefficient(factor, signs, packed, left.cross+right.cross, left.count+right.count)

    def _remove_degree(self, entry, degree):
        bucket = self._degrees[degree]
        bucket.discard(entry)
        if not bucket:
            del self._degrees[degree]

    def _remove_incidences(self, entry, meter):
        for u in entry.key:
            self._inverse[u].remove(entry)
            meter["inverse_incidence_removals"] += 1
            if not self._inverse[u]:
                del self._inverse[u]

    def _drop(self, entry, meter):
        del self._coeff[entry.key]
        self._remove_degree(entry, len(entry.key))
        self._remove_incidences(entry, meter)

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
                target.value = self._merge(target.value, entry.value, meter)
                meter["coefficient_collisions"] += 1
                meter["scalar_additions"] += self.dimension*self.response_dimension+1
                if target.value.is_zero():
                    self._drop(target, meter)
                    meter["exact_cancellations"] += 1
        while old_horizon in self._degrees:
            entry = self._degrees[old_horizon].pop()
            self._drop(entry, meter)
            meter["horizon_pruned_keys"] += 1
        self.alive.remove(u)
        self.horizon -= 1

    def delete(self, identifiers):
        """Validate all IDs and budget before mutation; retries are rejected."""
        requested = [_integer(u, "request identifier") for u in identifiers]
        if any(u < 0 or u >= self.universe_size for u in requested):
            raise ValueError("request identifier outside public universe")
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate unit IDs within request")
        if not set(requested) <= self.alive:
            raise ValueError("request contains already deleted units")
        if len(requested) > self.horizon:
            raise ValueError("request exceeds remaining cumulative deletion horizon")
        meter = {name: 0 for name in ("key_shrink_events", "tuple_cells_copied", "coefficient_collisions",
                  "scalar_additions", "exact_cancellations", "horizon_pruned_keys", "inverse_incidence_removals",
                  "dense_promotions", "factor_recompressions", "factor_columns_merged")}
        meter.update(input_identifiers=len(requested), fresh_deletions=len(requested))
        for u in sorted(requested):
            self._delete_one(u, meter)
        return meter

    def statistics(self):
        d, c = self.dimension, self.response_dimension
        entry = self._coeff.get(())
        if entry is None:
            return np.zeros((d, d)), np.zeros((d, c)), 0
        value = entry.value
        if value.count < 0:
            raise FloatingPointError("negative current count")
        if value.packed is None:
            gram = (value.factor*value.signs[None, :]) @ value.factor.T
            gram = (gram+gram.T)*0.5
        else:
            tri = np.triu_indices(d)
            gram = np.zeros((d, d), dtype=np.float64)
            gram[tri] = value.packed
            gram[(tri[1], tri[0])] = value.packed
        if not np.isfinite(gram).all() or not np.isfinite(value.cross).all():
            raise FloatingPointError("nonfinite current moments")
        return gram, value.cross.copy(), value.count

    def moments(self):
        from .core import RidgeMoments
        return RidgeMoments(*self.statistics())

    def decode(self, lambda_reg):
        from .core import RidgeMoments, solve_ridge
        moments = self.moments()
        if moments.count == 0:
            # The empty target has a prescribed zero head. Statistics expose
            # the raw FP cancellation residue for independent auditing.
            moments = RidgeMoments(np.zeros_like(moments.gram), np.zeros_like(moments.cross), 0)
        return solve_ridge(moments, lambda_reg)

    def solve_ridge(self, lambda_reg):
        return self.decode(lambda_reg).weights

    def coefficient(self, key):
        """One-key dense audit copy; absent coefficients raise KeyError."""
        value = self._coeff[tuple(key)].value
        out = np.empty(self.packed_dimension, dtype=np.float64)
        q = self.dimension*(self.dimension+1)//2
        out[:q] = value.gram_packed(self.dimension)
        out[q:-1] = value.cross.ravel()
        out[-1] = value.count
        return out

    def coefficient_keys(self):
        """Current structural keys only, for streaming independent audits."""
        return tuple(self._coeff)

    def dense_coefficients(self):
        """Audit-only dense copies matching the original packed statistic API."""
        result = {}
        for key, entry in self._coeff.items():
            value = entry.value
            out = np.empty(self.packed_dimension, dtype=np.float64)
            q = self.dimension*(self.dimension+1)//2
            out[:q] = value.gram_packed(self.dimension)
            out[q:-1] = value.cross.ravel()
            out[-1] = value.count
            result[key] = out
        return result

    coefficients = dense_coefficients

    def check_invariants(self):
        assert 0 <= self.horizon <= len(self.alive)
        expected_inverse, expected_degrees = {}, {}
        q = self.dimension*(self.dimension+1)//2
        for key, entry in self._coeff.items():
            assert entry.key == key == tuple(sorted(set(key)))
            assert len(key) <= self.horizon and set(key) <= self.alive
            value = entry.value
            assert isinstance(value.count, int)
            assert value.cross.shape == (self.dimension, self.response_dimension)
            assert value.cross.dtype == np.float64
            if value.packed is None:
                assert value.factor.shape == (self.dimension, value.width)
                assert value.signs.shape == (value.width,)
                assert value.signs.dtype == np.int8 and np.all(np.abs(value.signs) == 1)
                assert value.factor.dtype == np.float64
                assert value.factor.nbytes+value.signs.nbytes < q*8
            else:
                assert value.factor is None and value.signs is None
                assert value.packed.shape == (q,) and value.packed.dtype == np.float64
            assert all(a.flags.owndata and np.isfinite(a).all() for a in value.arrays())
            assert not value.is_zero()
            expected_degrees.setdefault(len(key), set()).add(entry)
            for u in key:
                expected_inverse.setdefault(u, set()).add(entry)
        assert expected_inverse == self._inverse and expected_degrees == self._degrees

    def accounting(self):
        numeric = sum(a.nbytes for e in self._coeff.values() for a in e.value.arrays())
        seen = set()
        def size(obj):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            total = sys.getsizeof(obj)
            if isinstance(obj, dict):
                total += sum(size(k)+size(v) for k, v in obj.items())
            elif isinstance(obj, (set, tuple, list)):
                total += sum(map(size, obj))
            elif isinstance(obj, (_Entry, _Coefficient)):
                total += sum(size(getattr(obj, slot)) for slot in obj.__slots__)
            return total
        total = sys.getsizeof(self)+sum(size(getattr(self, name)) for name in self.__slots__)
        coeffs = [entry.value for entry in self._coeff.values()]
        degrees = [len(key) for key in self._coeff]
        gram_bytes = sum(sum(a.nbytes for a in (v.factor, v.signs, v.packed) if a is not None) for v in coeffs)
        return {
            "method": "P-index-adaptive-signed-Gram-python-numpy",
            "state_contract": "current aggregated coefficients and public membership; no selected-ID recovery",
            "coefficient_count": len(coeffs), "packed_dimension": self.packed_dimension,
            "numeric_coefficient_bytes": numeric, "numeric_total_bytes": numeric, "numeric_gram_bytes": gram_bytes,
            "numeric_cross_bytes": sum(v.cross.nbytes for v in coeffs),
            "numeric_sign_bytes": sum(v.signs.nbytes for v in coeffs if v.signs is not None),
            "exact_count_python_bytes": sum(sys.getsizeof(v.count) for v in coeffs),
            "python_live_bytes_estimate": total, "python_metadata_bytes_estimate": total-numeric,
            "factored_coefficients": sum(v.packed is None for v in coeffs),
            "dense_coefficients": sum(v.packed is not None for v in coeffs),
            "stored_factor_columns": sum(v.width for v in coeffs),
            "alive_public_identifiers": len(self.alive), "remaining_horizon": self.horizon,
            "key_incidences": sum(degrees),
            "quadratic_key_potential": sum(k*(k+1)//2 for k in degrees),
            "retained_separate_feature_rows": 0, "retained_separate_target_rows": 0,
            "retained_row_table": False, "repair_time_payload_reads": False,
            "retained_original_graph": False, "retained_contributor_ids": False,
            "numeric_guarantee": "FP64 diagnostic fidelity; exact-arithmetic factor identity; no rank truncation",
        }


    def save(self, path: str | Path):
        """Serialize current state only, no pickle and no dense factor expansion.

        The NPZ exporter creates metadata arrays and Python references to the
        existing owned buffers. Archive-writing temporary memory is not free.
        Sorting makes archive ordering stable, not floating values canonical.
        """
        keys = sorted(self._coeff)
        arrays = {
            "format_version": np.array(1, dtype=np.int64),
            "metadata": np.array([self.universe_size, self.horizon, self.dimension,
                                  self.response_dimension, len(keys)], dtype=np.int64),
            "alive": np.array(sorted(self.alive), dtype=np.int64),
        }
        for i, key in enumerate(keys):
            value = self._coeff[key].value
            prefix = f"k{i}_"
            arrays[prefix+"key"] = np.array(key, dtype=np.int64)
            arrays[prefix+"count"] = np.array(str(value.count))
            arrays[prefix+"cross"] = value.cross
            if value.packed is None:
                arrays[prefix+"factor"] = value.factor
                arrays[prefix+"signs"] = value.signs
            else:
                arrays[prefix+"packed"] = value.packed
        destination = Path(path)
        with destination.open("wb") as stream:
            np.savez(stream, **arrays)
        return destination.stat().st_size

    @classmethod
    def load(cls, path: str | Path):
        """Load owned current-state buffers from a trusted-shape NPZ checkpoint."""
        with np.load(Path(path), allow_pickle=False) as data:
            if data["format_version"].shape != () or int(data["format_version"]) != 1:
                raise ValueError("unsupported summary format")
            meta = data["metadata"]
            if meta.shape != (5,) or meta.dtype != np.int64:
                raise ValueError("invalid summary metadata")
            universe, h, d, c, length = map(int, meta)
            if min(universe, h, length) < 0 or min(d, c) <= 0 or h > universe:
                raise ValueError("invalid summary dimensions or horizon")
            state = cls.__new__(cls)
            state.universe_size, state.horizon = universe, h
            state.dimension, state.response_dimension = d, c
            state.packed_dimension = d*(d+1)//2+d*c+1
            alive_array = data["alive"]
            if alive_array.ndim != 1 or alive_array.dtype != np.int64:
                raise ValueError("invalid public membership")
            state.alive = set(map(int, alive_array))
            if len(state.alive) != len(alive_array) or any(u < 0 or u >= universe for u in state.alive):
                raise ValueError("invalid public membership")
            state._coeff, state._inverse, state._degrees = {}, {}, {}
            for i in range(length):
                prefix = f"k{i}_"
                raw_key = data[prefix+"key"]
                if raw_key.ndim != 1 or raw_key.dtype != np.int64:
                    raise ValueError("invalid coefficient key")
                key = tuple(map(int, raw_key))
                if key in state._coeff:
                    raise ValueError("duplicate coefficient key")
                raw_count = data[prefix+"count"]
                if raw_count.shape != () or raw_count.dtype.kind != "U":
                    raise ValueError("invalid exact count")
                count = int(str(raw_count))
                cross = data[prefix+"cross"].copy()
                if prefix+"packed" in data.files:
                    if prefix+"factor" in data.files or prefix+"signs" in data.files:
                        raise ValueError("ambiguous Gram representation")
                    value = _Coefficient(None, None, data[prefix+"packed"].copy(), cross, count)
                else:
                    value = _Coefficient(data[prefix+"factor"].copy(), data[prefix+"signs"].copy(), None, cross, count)
                state._insert(_Entry(key, value))
        try:
            state.check_invariants()
        except (AssertionError, TypeError, ValueError) as exc:
            raise ValueError("invalid coefficient state") from exc
        return state


FactoredRidgeSummary.preflight = staticmethod(preflight)
# A descriptive alias for experiments using explicit adaptive nomenclature.
AdaptiveFactoredRidgeSummary = FactoredRidgeSummary
