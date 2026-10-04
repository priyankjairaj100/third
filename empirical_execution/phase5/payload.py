"""Packed finite-horizon eligible-payload antijoin (B-E) and full-payload B-A.

The relational count/antijoin and signed ridge-statistic mechanisms are established
baselines; this is a versioned implementation, not a novelty claim. FP64 moments
are numerical approximations, not the exact rational canonical-state contract.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from numbers import Integral
from typing import Iterable

import numpy as np
from ccu.core import RidgeMoments, solve_ridge, _owned_size

SCHEMA = 'ccu-packed-payload-1'
_ARRAYS = ('x', 'y', 'owner', 'counts', 'live', 'selected', 'blocker_ptr',
           'blocker_idx', 'reverse_ptr', 'reverse_idx', 'owner_ptr', 'owner_idx',
           'bucket_head', 'bucket_prev', 'bucket_next', 'unit_alive', 'gram', 'cross')


def _int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _csr(groups, columns):
    """Build compact int32 values/int64 offsets from integer rows."""
    ptr = np.zeros(len(groups) + 1, dtype=np.int64)
    for i, group in enumerate(groups):
        ptr[i + 1] = ptr[i] + len(group)
    idx = np.fromiter((v for group in groups for v in group), dtype=np.int32,
                      count=int(ptr[-1]))
    if idx.size and (idx.min() < 0 or idx.max() >= columns):
        raise ValueError('CSR incidence out of range')
    return ptr, idx


def _inverse(ptr, idx, universe):
    """Integer stable transpose, retaining no Python per-incidence sets."""
    lengths = np.bincount(idx, minlength=universe)
    iptr = np.empty(universe + 1, dtype=np.int64)
    iptr[0] = 0
    np.cumsum(lengths, out=iptr[1:])
    rows = np.repeat(np.arange(len(ptr) - 1, dtype=np.int32), np.diff(ptr))
    return iptr, rows[np.argsort(idx, kind='stable')]


class PayloadState:
    """Owns only stored payload/indexes, never references constructor arrays.

    Requests are atomic for malformed/unknown/over-budget batches. Repeated IDs
    within or across calls are no-ops and consume no new horizon. Allocation or
    unexpected numerical failures are fatal worker failures, not transactional
    rollback promises. B-A uses the same code while retaining all live payload.
    """
    def __init__(self, graph, learner_fp32, targets_fp64, horizon,
                 unit='record', retain_all=False, compaction_fraction=.5,
                 batch_rows=1024):
        x, y = np.asarray(learner_fp32), np.asarray(targets_fp64)
        n = len(graph.record_ids)
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[0] != n or x.shape[1] < 1 or not np.isfinite(x).all():
            raise ValueError('finite aligned FP32 learner matrix required')
        if y.ndim == 1:
            y = y[:, None]
        if y.dtype != np.float64 or y.ndim != 2 or y.shape[0] != n or y.shape[1] < 1 or not np.isfinite(y).all():
            raise ValueError('finite aligned FP64 target matrix required')
        ids = tuple(graph.record_ids)
        if len(set(ids)) != n or any(not isinstance(r, str) or not r for r in ids):
            raise ValueError('unique nonempty record IDs required')
        if unit not in {'record', 'source'} or not isinstance(retain_all, bool):
            raise ValueError('unit must be record/source; retain_all must be bool')
        owners = ids if unit == 'record' else tuple(graph.source_ids)
        if len(owners) != n or any(not isinstance(r, str) or not r for r in owners):
            raise ValueError('complete disjoint source ownership required')
        self.unit_ids = tuple(sorted(set(owners)))
        if n > np.iinfo(np.int32).max or len(self.unit_ids) > np.iinfo(np.int32).max:
            raise ValueError('int32 index capacity exceeded')
        self._unit_lookup = {r: i for i, r in enumerate(self.unit_ids)}
        h = _int(horizon, 'horizon')
        if not 0 <= h <= len(self.unit_ids):
            raise ValueError('horizon outside service universe')
        self.batch_rows = _int(batch_rows, 'batch_rows')
        if self.batch_rows < 1:
            raise ValueError('batch_rows must be positive')
        if isinstance(compaction_fraction, bool) or not math.isfinite(compaction_fraction) or not 0 < compaction_fraction <= 1:
            raise ValueError('compaction_fraction must lie in (0,1]')
        self.compaction_fraction = float(compaction_fraction)
        self.unit, self.retain_all = unit, retain_all
        self.horizon = self.initial_horizon = h
        self.dimension, self.outputs = x.shape[1], y.shape[1]
        self.unit_alive = np.ones(len(self.unit_ids), dtype=bool)
        owner_codes = np.asarray([self._unit_lookup[u] for u in owners], dtype=np.int32)
        ptr, idx = np.asarray(graph.indptr), np.asarray(graph.indices)
        order = np.asarray(graph.priority_indices)
        if ptr.dtype.kind not in 'iu' or idx.dtype.kind not in 'iu' or ptr.shape != (n + 1,) or ptr[0] != 0 or ptr[-1] != len(idx) or np.any(np.diff(ptr) < 0):
            raise ValueError('invalid blocker CSR')
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise ValueError('invalid blocker record index')
        if order.shape != (n,) or order.dtype.kind not in 'iu' or set(map(int, order)) != set(range(n)):
            raise ValueError('invalid graph priority permutation')
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n)
        groups, keep = [], []
        for i in range(n):
            blockers = idx[ptr[i]:ptr[i + 1]]
            if len(set(map(int, blockers))) != len(blockers) or np.any(ranks[blockers] >= ranks[i]):
                raise ValueError('blockers must be unique earlier raw records')
            b = np.unique(owner_codes[blockers])
            if retain_all or (len(b) <= h and owner_codes[i] not in b):
                groups.append(b)
                keep.append(i)
        keep = np.asarray(keep, dtype=np.int64)
        self.record_ids = tuple(ids[i] for i in keep)
        self.x = np.array(x[keep], dtype=np.float32, order='C', copy=True)
        self.y = np.array(y[keep], dtype=np.float64, order='C', copy=True)
        self.owner = owner_codes[keep].copy()
        self.blocker_ptr, self.blocker_idx = _csr(groups, len(self.unit_ids))
        self.counts = np.diff(self.blocker_ptr).astype(np.int32)
        self.live = np.ones(len(keep), dtype=bool)
        self.selected = self.counts == 0
        self._live_rows = len(keep)
        self._active_blockers = len(self.blocker_idx)
        self.gram = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        self.cross = np.zeros((self.dimension, self.outputs), dtype=np.float64)
        self.count = 0
        self.cumulative = {'requests': 0, 'fresh_units': 0, 'incidence_touches': 0,
                           'statistic_coordinate_additions': 0, 'blas_batches': 0,
                           'expired_rows': 0, 'compactions': 0,
                           'compaction_rows_scanned': 0, 'compaction_incidences_scanned': 0,
                           'payload_rows_read': 0, 'peak_blas_workspace_array_bytes': 0}
        self._rebuild_indices()
        self._signed(np.flatnonzero(self.selected), +1)

    def _rebuild_indices(self):
        self.reverse_ptr, self.reverse_idx = _inverse(self.blocker_ptr, self.blocker_idx, len(self.unit_ids))
        ptr = np.arange(len(self.owner) + 1, dtype=np.int64)
        self.owner_ptr, self.owner_idx = _inverse(ptr, self.owner, len(self.unit_ids))
        width = len(self.unit_ids) + 1 if self.retain_all else self.horizon + 1
        self.bucket_head = np.full(width, -1, dtype=np.int32)
        self.bucket_prev = np.full(len(self.owner), -1, dtype=np.int32)
        self.bucket_next = np.full(len(self.owner), -1, dtype=np.int32)
        for row in np.flatnonzero(self.live):
            self._bucket_add(int(row))

    def _bucket_add(self, row):
        degree = int(self.counts[row])
        front = int(self.bucket_head[degree])
        self.bucket_next[row] = front
        self.bucket_prev[row] = -1
        if front >= 0:
            self.bucket_prev[front] = row
        self.bucket_head[degree] = row

    def _bucket_remove(self, row):
        previous, nxt = int(self.bucket_prev[row]), int(self.bucket_next[row])
        if previous < 0:
            self.bucket_head[self.counts[row]] = nxt
        else:
            self.bucket_next[previous] = nxt
        if nxt >= 0:
            self.bucket_prev[nxt] = previous
        self.bucket_prev[row] = self.bucket_next[row] = -1

    def _drop(self, rows):
        for row in rows:
            row = int(row)
            self._bucket_remove(row)
            self._active_blockers -= int(self.counts[row])
        self.live[rows] = False
        self.selected[rows] = False
        self._live_rows -= len(rows)

    def _signed(self, rows, sign):
        """Bounded batched BLAS; no per-record outer products are retained."""
        for lo in range(0, len(rows), self.batch_rows):
            block = rows[lo:lo + self.batch_rows]
            gathered = self.x[block]
            z = gathered.astype(np.float64)
            y = self.y[block]
            g, h = z.T @ z, z.T @ y
            if sign > 0:
                self.gram += g
                self.cross += h
            else:
                self.gram -= g
                self.cross -= h
            self.count += sign * len(block)
            workspace = gathered.nbytes + z.nbytes + y.nbytes + g.nbytes + h.nbytes
            self.cumulative['peak_blas_workspace_array_bytes'] = max(self.cumulative['peak_blas_workspace_array_bytes'], workspace)
            self.cumulative['blas_batches'] += 1
            self.cumulative['payload_rows_read'] += len(block)
            self.cumulative['statistic_coordinate_additions'] += g.size + h.size

    def _compact_if_due(self):
        row_fraction = 1 - self._live_rows / len(self.live) if len(self.live) else 0
        edge_fraction = 1 - self._active_blockers / len(self.blocker_idx) if len(self.blocker_idx) else 0
        if max(row_fraction, edge_fraction) < self.compaction_fraction:
            return False
        self.cumulative['compactions'] += 1
        self.cumulative['compaction_rows_scanned'] += len(self.live)
        self.cumulative['compaction_incidences_scanned'] += len(self.blocker_idx)
        rows = np.flatnonzero(self.live)
        groups = []
        for row in rows:
            b = self.blocker_idx[self.blocker_ptr[row]:self.blocker_ptr[row + 1]]
            groups.append(b[self.unit_alive[b]])
        self.record_ids = tuple(self.record_ids[r] for r in rows)
        self.x, self.y = self.x[rows].copy(), self.y[rows].copy()
        self.owner, self.counts = self.owner[rows].copy(), self.counts[rows].copy()
        self.live = np.ones(len(rows), dtype=bool)
        self.selected = self.selected[rows].copy()
        self.blocker_ptr, self.blocker_idx = _csr(groups, len(self.unit_ids))
        self._rebuild_indices()
        return True

    def delete(self, units: Iterable[str]):
        request = tuple(units)
        if any(not isinstance(u, str) or u not in self._unit_lookup for u in request):
            raise ValueError('request contains an unknown/non-string service unit')
        codes = np.asarray(sorted({self._unit_lookup[u] for u in request}), dtype=np.int32)
        fresh = codes[self.unit_alive[codes]]
        if len(fresh) > self.horizon:
            raise ValueError('request exceeds remaining cumulative horizon')
        before = self.cumulative.copy()
        removed_parts = [self.owner_idx[self.owner_ptr[u]:self.owner_ptr[u + 1]] for u in fresh]
        removed = np.concatenate(removed_parts) if removed_parts else np.empty(0, dtype=np.int32)
        removed = removed[self.live[removed]]
        removed_selected = removed[self.selected[removed]]
        removed_ids = sorted(self.record_ids[r] for r in removed_selected)
        self._signed(removed_selected, -1)
        self._drop(removed)
        touched_parts = [self.reverse_idx[self.reverse_ptr[u]:self.reverse_ptr[u + 1]] for u in fresh]
        touched_raw = np.concatenate(touched_parts) if touched_parts else np.empty(0, dtype=np.int32)
        self.cumulative['incidence_touches'] += len(touched_raw) + sum(len(part) for part in removed_parts)
        touched, decrement = np.unique(touched_raw[self.live[touched_raw]], return_counts=True)
        for row, amount in zip(touched, decrement):
            row = int(row)
            self._bucket_remove(row)
            self.counts[row] -= amount
            self._bucket_add(row)
        self._active_blockers -= int(decrement.sum())
        admitted = touched[self.counts[touched] == 0]
        admitted_ids = sorted(self.record_ids[r] for r in admitted)
        self.selected[admitted] = True
        self._signed(admitted, +1)
        self.unit_alive[fresh] = False
        previous_horizon = self.horizon
        self.horizon -= len(fresh)
        expired = []
        if not self.retain_all:
            for degree in range(self.horizon + 1, previous_horizon + 1):
                row = int(self.bucket_head[degree])
                while row >= 0:
                    expired.append(row)
                    row = int(self.bucket_next[row])
            self._drop(np.asarray(expired, dtype=np.int32))
        self.cumulative['expired_rows'] += len(expired)
        self.cumulative['requests'] += 1
        self.cumulative['fresh_units'] += len(fresh)
        if self.count == 0:
            self.gram.fill(0)
            self.cross.fill(0)
        compacted = self._compact_if_due()
        return {'requested_units': sorted(set(request)), 'input_identifiers': len(request),
                'fresh_deletions': len(fresh), 'removed_selected_ids': removed_ids,
                'admitted_ids': admitted_ids, 'discarded_payload_records': len(removed) + len(expired),
                'touched_records': len(touched), 'remaining_horizon': self.horizon,
                'selected_count': self.count, 'compacted': compacted,
                'operations': {k: self.cumulative[k] - before[k] for k in before}}

    def selected_ids(self):
        """Verification-only membership view, outside head-only service output."""
        return tuple(sorted(self.record_ids[r] for r in np.flatnonzero(self.selected)))

    def moments(self):
        return RidgeMoments(self.gram.copy(), self.cross.copy(), self.count)

    def statistics(self):
        return self.gram.copy(), self.cross.copy(), self.count

    def decode(self, lambda_reg):
        """Common decoder uses lambda * current_count, including empty zero."""
        return solve_ridge(self.moments(), lambda_reg)

    def logical_membership(self):
        """Fresh-state audit; float moment bytes may depend on update history."""
        return {self.record_ids[r]: {'owner': self.unit_ids[int(self.owner[r])],
                'blockers': tuple(self.unit_ids[int(u)] for u in
                    self.blocker_idx[self.blocker_ptr[r]:self.blocker_ptr[r + 1]]
                    if self.unit_alive[u])}
                for r in np.flatnonzero(self.live)}

    def accounting(self):
        arrays = {name: getattr(self, name).nbytes for name in _ARRAYS}
        payload_per_row = self.dimension * 4 + self.outputs * 8
        stale_rows = len(self.live) - self._live_rows
        stale_blockers = len(self.blocker_idx) - self._active_blockers
        return {'schema': SCHEMA, 'method': 'B-A' if self.retain_all else 'B-E',
                'access_capability': 'all live learner payload' if self.retain_all else 'horizon-eligible learner payload',
                'remaining_horizon': self.horizon, 'selected_count': self.count,
                'allocated_rows': len(self.live), 'live_payload_rows': self._live_rows,
                'stale_payload_rows': stale_rows, 'live_blocker_incidences': self._active_blockers,
                'stale_blocker_incidences': stale_blockers,
                'allocated_array_bytes': sum(arrays.values()), 'array_bytes_by_field': arrays,
                'live_payload_bytes': self._live_rows * payload_per_row,
                'stale_payload_bytes': stale_rows * payload_per_row,
                'stale_csr_value_bytes': stale_blockers * 8 + stale_rows * 4,
                'stale_row_slot_metadata_bytes': stale_rows * (4 + 4 + 1 + 1 + 4 + 4),
                'registry_utf8_bytes': sum(len(r.encode()) for r in self.unit_ids) + sum(len(r.encode()) for r in self.record_ids),
                'python_owned_bytes_estimate': _owned_size(self.__dict__),
                'physical_erasure_claim': False, 'peak_rss_measured_here': False,
                'compaction_fraction': self.compaction_fraction,
                'cumulative_operations': dict(self.cumulative)}

    def snapshot_bytes(self):
        """Serialize allocated state without filesystem access; charge buffer RAM."""
        meta = {k: getattr(self, k) for k in ('unit_ids', 'record_ids', 'unit', 'retain_all',
                'horizon', 'initial_horizon', 'dimension', 'outputs', 'count',
                '_live_rows', '_active_blockers', 'batch_rows', 'compaction_fraction', 'cumulative')}
        meta['schema'] = SCHEMA
        raw = json.dumps(meta, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        arrays = {name: getattr(self, name) for name in _ARRAYS}
        handle = io.BytesIO()
        np.savez(handle, metadata=np.frombuffer(raw, dtype=np.uint8), **arrays)
        return handle.getvalue()

    def snapshot(self, path):
        """Save actual allocated state, including charged tombstones, no pickle."""
        raw = self.snapshot_bytes()
        path = Path(path)
        path.write_bytes(raw)
        return {'path': str(path), 'serialized_bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest(),
                'compression': 'none', 'includes_allocated_tombstones': True}

    @classmethod
    def load(cls, path):
        """Trusted research checkpoint restore; no original graph/payload input."""
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {*_ARRAYS, 'metadata'}:
                raise ValueError('snapshot members do not match schema')
            meta = json.loads(archive['metadata'].tobytes())
            if meta.pop('schema', None) != SCHEMA:
                raise ValueError('unsupported payload snapshot schema')
            obj = cls.__new__(cls)
            for key, value in meta.items():
                setattr(obj, key, value)
            obj.unit_ids, obj.record_ids = tuple(obj.unit_ids), tuple(obj.record_ids)
            obj._unit_lookup = {r: i for i, r in enumerate(obj.unit_ids)}
            for name in _ARRAYS:
                setattr(obj, name, np.array(archive[name], copy=True))
        obj.check_invariants(check_moments=True)
        return obj

    def check_invariants(self, check_moments=False):
        """Internal consistency check, not a scientific provenance certificate."""
        m, u = len(self.record_ids), len(self.unit_ids)
        if len(set(self.unit_ids)) != u or len(set(self.record_ids)) != m:
            raise ValueError('duplicate state IDs')
        expected_dtypes = {name: np.dtype('int32') for name in ('owner', 'counts', 'blocker_idx', 'reverse_idx', 'owner_idx', 'bucket_head', 'bucket_prev', 'bucket_next')}
        expected_dtypes.update({name: np.dtype('int64') for name in ('blocker_ptr', 'reverse_ptr', 'owner_ptr')})
        expected_dtypes.update({name: np.dtype('bool') for name in ('live', 'selected', 'unit_alive')})
        expected_dtypes.update({'x': np.dtype('float32'), 'y': np.dtype('float64'), 'gram': np.dtype('float64'), 'cross': np.dtype('float64')})
        if any(getattr(self, name).dtype != dtype for name, dtype in expected_dtypes.items()):
            raise ValueError('snapshot dtype mismatch')
        if self.x.shape != (m, self.dimension) or self.y.shape != (m, self.outputs) or self.gram.shape != (self.dimension, self.dimension) or self.cross.shape != (self.dimension, self.outputs):
            raise ValueError('snapshot array shape mismatch')
        if any(not np.isfinite(getattr(self, name)).all() for name in ('x', 'y', 'gram', 'cross')):
            raise ValueError('nonfinite payload/statistics')
        if any(getattr(self, name).shape != (m,) for name in ('live', 'selected', 'owner', 'counts', 'bucket_prev', 'bucket_next')) or self.unit_alive.shape != (u,):
            raise ValueError('snapshot row/unit shape mismatch')
        if not 0 <= self.horizon <= self.initial_horizon <= u or self.initial_horizon - self.horizon != np.count_nonzero(~self.unit_alive):
            raise ValueError('snapshot budget mismatch')
        if self._live_rows != np.count_nonzero(self.live) or self.count != np.count_nonzero(self.selected) or np.any(self.selected & ~self.live):
            raise ValueError('snapshot membership counts mismatch')
        for ptr, idx, rows, cols in ((self.blocker_ptr, self.blocker_idx, m, u), (self.reverse_ptr, self.reverse_idx, u, m), (self.owner_ptr, self.owner_idx, u, m)):
            if ptr.shape != (rows + 1,) or ptr[0] != 0 or ptr[-1] != len(idx) or np.any(np.diff(ptr) < 0) or (len(idx) and (idx.min() < 0 or idx.max() >= cols)):
                raise ValueError('snapshot CSR malformed')
        if len(self.owner) and (self.owner.min() < 0 or self.owner.max() >= u):
            raise ValueError('snapshot owner outside universe')
        ip, ii = _inverse(self.blocker_ptr, self.blocker_idx, u)
        op, oi = _inverse(np.arange(m + 1, dtype=np.int64), self.owner, u)
        if not (np.array_equal(ip, self.reverse_ptr) and np.array_equal(ii, self.reverse_idx) and np.array_equal(op, self.owner_ptr) and np.array_equal(oi, self.owner_idx)):
            raise ValueError('snapshot reciprocal CSR mismatch')
        seen, active = set(), 0
        for degree, first in enumerate(self.bucket_head):
            previous, row = -1, int(first)
            while row >= 0:
                if row >= m or row in seen or not self.live[row] or self.counts[row] != degree or self.bucket_prev[row] != previous:
                    raise ValueError('snapshot degree bucket mismatch')
                seen.add(row)
                previous, row = row, int(self.bucket_next[row])
        if seen != set(map(int, np.flatnonzero(self.live))):
            raise ValueError('snapshot degree buckets omit live records')
        for r in np.flatnonzero(self.live):
            b = self.blocker_idx[self.blocker_ptr[r]:self.blocker_ptr[r + 1]]
            if len(np.unique(b)) != len(b):
                raise ValueError('duplicate blocker-source incidence')
            b = b[self.unit_alive[b]]
            active += len(b)
            if not self.unit_alive[self.owner[r]] or self.counts[r] != len(b) or self.selected[r] != (len(b) == 0):
                raise ValueError('snapshot live ownership/count mismatch')
            if not self.retain_all and (len(b) > self.horizon or self.owner[r] in b):
                raise ValueError('snapshot retains ineligible B-E payload')
        if active != self._active_blockers:
            raise ValueError('snapshot active incidence count mismatch')
        if self.count == 0 and (np.any(self.gram) or np.any(self.cross)):
            raise ValueError('empty state requires exact zero moments')
        if check_moments:
            rows = np.flatnonzero(self.selected)
            z, y = self.x[rows].astype(np.float64), self.y[rows]
            np.testing.assert_allclose(self.gram, z.T @ z, atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(self.cross, z.T @ y, atol=1e-9, rtol=1e-10)
        return True
