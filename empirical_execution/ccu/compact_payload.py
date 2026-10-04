"""Memory-conscious eligible-payload comparator with no persistent Gram.

The state intentionally retains eligible FP32 feature rows and FP64 targets.
It is a fair payload-access baseline, not a payload-free summary.  Releases
rebuild only the selected payload matrix and solve in the smaller of the
sample and feature dimensions.  Decode allocations are temporary and must be
included separately in measured peak memory and time.

Identifiers are integer positions, matching IndexedRidgeSummary.  External
public-ID adapters are excluded equally from both methods' state accounting.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral
import sys
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .core import RidgeMoments, RidgeSolution, canonical_features, canonical_targets


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


class CompactEligiblePayloadState:
    """Indexed eligible payload and selected membership; no cached moments."""

    __slots__ = (
        "universe_size", "horizon", "dimension", "response_dimension", "alive",
        "_owner", "_blockers", "_features", "_targets", "_by_owner",
        "_reverse", "_by_count", "_selected",
    )

    @classmethod
    def build(cls, features, targets, blockers: Sequence[Iterable[int]], horizon: int,
              owners: Sequence[int] | None = None, universe_size: int | None = None):
        x = canonical_features(features)
        y = canonical_targets(targets, len(x))
        n, d = x.shape
        if len(blockers) != n:
            raise ValueError("features, targets and blockers must match")
        h = _integer(horizon, "horizon")
        if owners is None:
            owner_ids = tuple(range(n))
            universe = n if universe_size is None else _integer(universe_size, "universe_size")
            if universe != n:
                raise ValueError("record mode requires universe_size == record count")
        else:
            if len(owners) != n:
                raise ValueError("owners must match record count")
            owner_ids = tuple(_integer(u, "owner") for u in owners)
            universe = max(owner_ids, default=-1) + 1 if universe_size is None else _integer(universe_size, "universe_size")
        if universe < 0 or not 0 <= h <= universe:
            raise ValueError("horizon must fit the service universe")
        if any(u < 0 or u >= universe for u in owner_ids):
            raise ValueError("owner outside service universe")
        state = cls.__new__(cls)
        state.universe_size, state.horizon = universe, h
        state.dimension, state.response_dimension = d, y.shape[1]
        state.alive = set(range(universe))
        state._owner, state._blockers, state._features, state._targets = {}, {}, {}, {}
        state._by_owner, state._reverse, state._by_count = {}, {}, {}
        state._selected = set()
        for record, raw in enumerate(blockers):
            b = set()
            for value in raw:
                v = _integer(value, "blocker index")
                if not 0 <= v < n or v == record:
                    raise ValueError("blocker must be another valid record index")
                b.add(owner_ids[v])
            own = owner_ids[record]
            if own in b or len(b) > h:
                continue
            state._owner[record] = own
            state._blockers[record] = b
            state._features[record] = x[record].copy()
            state._targets[record] = y[record].copy()
            state._by_owner.setdefault(own, set()).add(record)
            state._by_count.setdefault(len(b), set()).add(record)
            for u in b:
                state._reverse.setdefault(u, set()).add(record)
            if not b:
                state._selected.add(record)
        return state

    @staticmethod
    def _discard(index, key, record):
        bucket = index[key]
        bucket.remove(record)
        if not bucket:
            del index[key]

    def _drop_payload(self, record):
        own, b = self._owner.pop(record), self._blockers.pop(record)
        self._discard(self._by_owner, own, record)
        self._discard(self._by_count, len(b), record)
        for u in b:
            self._discard(self._reverse, u, record)
        del self._features[record], self._targets[record]

    def delete(self, identifiers: Iterable[int]) -> dict:
        request = [_integer(u, "request identifier") for u in identifiers]
        if any(u < 0 or u >= self.universe_size for u in request):
            raise ValueError("request identifier outside public universe")
        if len(set(request)) != len(request):
            raise ValueError("duplicate unit IDs within request")
        if not set(request) <= self.alive:
            raise ValueError("request contains already deleted units")
        if len(request) > self.horizon:
            raise ValueError("request exceeds remaining cumulative deletion horizon")
        request.sort()
        removed = set().union(*(self._by_owner.get(u, set()) for u in request)) if request else set()
        removed_selected = removed & self._selected
        self._selected.difference_update(removed_selected)
        for record in sorted(removed):
            self._drop_payload(record)
        touched = set()
        for u in request:
            for record in tuple(self._reverse.get(u, ())):
                b = self._blockers[record]
                self._discard(self._by_count, len(b), record)
                b.remove(u)
                self._by_count.setdefault(len(b), set()).add(record)
                touched.add(record)
            self._reverse.pop(u, None)
        admitted = {r for r in touched if not self._blockers[r] and r not in self._selected}
        self._selected.update(admitted)
        self.alive.difference_update(request)
        self.horizon -= len(request)
        prune = [r for count, records in self._by_count.items() if count > self.horizon for r in records]
        for r in sorted(prune):
            self._drop_payload(r)
        return {
            "input_identifiers": len(request), "fresh_deletions": len(request),
            "removed_selected_ids": sorted(removed_selected), "admitted_ids": sorted(admitted),
            "discarded_payload_records": len(removed) + len(prune),
            "touched_records": len(touched), "remaining_horizon": self.horizon,
            "selected_count": len(self._selected),
        }

    def selected_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._selected))

    def _selected_arrays(self):
        records = self.selected_ids()
        if not records:
            return np.empty((0, self.dimension), dtype=np.float64), np.empty((0, self.response_dimension), dtype=np.float64)
        # Direct FP64 allocation avoids a full intermediate stacked FP32 copy.
        z = np.empty((len(records), self.dimension), dtype=np.float64)
        y = np.empty((len(records), self.response_dimension), dtype=np.float64)
        for j, r in enumerate(records):
            z[j], y[j] = self._features[r], self._targets[r]
        return z, y

    def moments(self) -> RidgeMoments:
        """Audit helper; returned moments are never retained by this state."""
        z, y = self._selected_arrays()
        return RidgeMoments(z.T @ z, z.T @ y, len(z))

    def statistics(self):
        value = self.moments()
        return value.gram, value.cross, value.count

    def decode(self, lambda_reg: float) -> RidgeSolution:
        """Exact real-arithmetic primal/dual ridge identity; FP64 computation.

        The dual uses alpha=lambda*n, preserving average-loss regularization.
        No model, kernel, Gram or solver factors survive this method call.
        Residuals use X^T(XW-Y)+alpha*W, avoiding a d-by-d diagnostic Gram.
        """
        lam = float(lambda_reg)
        if not math.isfinite(lam) or lam <= 0:
            raise ValueError("lambda_reg must be positive and finite")
        z, y = self._selected_arrays()
        n, d = z.shape
        if n == 0:
            w = np.zeros((d, self.response_dimension), dtype=np.float64)
            return RidgeSolution(w, 0.0, 0.0, 0.0, lam, 0, "empty target; no persistent moments")
        alpha = lam * n
        if n < d:
            a = z @ z.T
            a.flat[::n + 1] += alpha
            factor = cho_factor(a, lower=True, overwrite_a=False, check_finite=True)
            beta = cho_solve(factor, y, overwrite_b=False, check_finite=True)
            w = z.T @ beta
            solver = "FP64 dual ridge Cholesky; rebuilt selected payload; no persistent Gram"
        else:
            a = z.T @ z
            a.flat[::d + 1] += alpha
            cross = z.T @ y
            factor = cho_factor(a, lower=True, overwrite_a=False, check_finite=True)
            w = cho_solve(factor, cross, overwrite_b=False, check_finite=True)
            solver = "FP64 primal ridge Cholesky; rebuilt selected payload; no persistent Gram"
        residual = float(np.linalg.norm(z.T @ (z @ w - y) + alpha * w))
        return RidgeSolution(w, residual, residual / alpha, residual / n, lam, n, solver)

    def solve_ridge(self, lambda_reg):
        return self.decode(lambda_reg).weights

    def check_invariants(self):
        keys = set(self._owner)
        assert keys == set(self._blockers) == set(self._features) == set(self._targets)
        assert self._selected == {r for r in keys if not self._blockers[r]}
        for r in keys:
            b, own = self._blockers[r], self._owner[r]
            assert own in self.alive and own not in b and b <= self.alive
            assert len(b) <= self.horizon
            assert r in self._by_owner[own] and r in self._by_count[len(b)]
            assert self._features[r].dtype == np.float32
            assert self._targets[r].dtype == np.float64
            assert self._features[r].flags.owndata and self._targets[r].flags.owndata
            assert all(r in self._reverse[u] for u in b)
        assert (set().union(*self._by_owner.values()) if self._by_owner else set()) == keys
        assert (set().union(*self._by_count.values()) if self._by_count else set()) == keys
        for own, group in self._by_owner.items():
            assert group and all(self._owner[r] == own for r in group)
        for count, group in self._by_count.items():
            assert group and all(len(self._blockers[r]) == count for r in group)
        for unit, group in self._reverse.items():
            assert group and all(unit in self._blockers[r] for r in group)

    def state_digest(self):
        digest = hashlib.sha256()
        for value in (self.universe_size, self.horizon, self.dimension, self.response_dimension, tuple(sorted(self.alive))):
            digest.update(repr(value).encode())
            digest.update(b"\0")
        for r in sorted(self._owner):
            digest.update(repr((r, self._owner[r], tuple(sorted(self._blockers[r])))).encode())
            digest.update(self._features[r].tobytes())
            digest.update(self._targets[r].tobytes())
        return digest.hexdigest()

    def accounting(self):
        arrays = list(self._features.values()) + list(self._targets.values())
        numeric = sum(a.nbytes for a in arrays)
        seen = set()
        def owned_size(obj):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            total = sys.getsizeof(obj)
            if isinstance(obj, dict):
                total += sum(owned_size(k) + owned_size(v) for k, v in obj.items())
            elif isinstance(obj, (tuple, list, set, frozenset)):
                total += sum(map(owned_size, obj))
            return total
        total = sys.getsizeof(self) + sum(owned_size(getattr(self, key)) for key in self.__slots__)
        n, d, c = len(self._selected), self.dimension, self.response_dimension
        return {
            "method": "B-E-compact-primal-dual", "state_contract": "eligible FP32 payload retained; no persistent moments",
            "remaining_horizon": self.horizon, "alive_service_units": len(self.alive),
            "eligible_payload_records": len(self._features), "selected_records": n,
            "blocker_incidences": sum(map(len, self._blockers.values())),
            "feature_array_bytes": sum(a.nbytes for a in self._features.values()),
            "target_array_bytes": sum(a.nbytes for a in self._targets.values()),
            "moment_array_bytes": 0, "kernel_array_bytes": 0, "all_array_bytes": numeric,
            "python_live_bytes_estimate": total, "python_metadata_bytes_estimate": total - numeric,
            "release_solver": "empty" if not n else ("dual" if n < d else "primal"),
            "release_system_dimension": min(n, d),
            "release_selected_copy_bytes": 8 * n * (d + c),
            "release_system_and_factor_bytes_estimate": 16 * min(n, d) ** 2,
            "workspace_estimate_note": "Partial named arrays only; excludes response/output/residual and BLAS workspaces; measure process peak separately",
        }
