"""Canonical signed graph values for frozen human-audit dossier consumers.

This adapter supplies serialization and a graph interface, not graph/source
acceptance or human evidence. Frozen human_job still compares the graph with
the graph freshly rebuilt from its accepted source/cache bundle.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
from ccu.core import BlockerGraph
from phase4 import requests as rq
from phase5 import task_program as task
from phase6 import recipes

SCHEMA = "ccu-phase9-canonical-blocker-graph-1"
FIELDS = frozenset({"schema", "record_ids", "source_ids", "priority_indices",
                    "indptr", "indices", "threshold_hex"})
INT64_MAX = (1 << 63) - 1


def _integers(value, name):
    if not isinstance(value, (list, tuple)) or any(type(x) is not int or not 0 <= x <= INT64_MAX for x in value):
        raise ValueError(name + " requires nonnegative int64 JSON integers, excluding booleans")
    return tuple(value)


def _strings(value, name):
    if not isinstance(value, (list, tuple)) or any(type(x) is not str or not x for x in value):
        raise ValueError(name + " requires nonempty text identifiers")
    return tuple(value)


def _immutable_int64(values):
    # A readonly flag alone can be reversed on an owning ndarray. Immutable
    # bytes are the backing buffer, so callers cannot re-enable writes.
    return np.frombuffer(np.asarray(values, dtype=np.int64).tobytes(), dtype=np.int64)


def _validate(payload):
    if not isinstance(payload, dict) or set(payload) != FIELDS or payload.get("schema") != SCHEMA:
        raise ValueError("Exact canonical graph schema and fields required")
    ids = _strings(payload["record_ids"], "record_ids")
    sources = _strings(payload["source_ids"], "source_ids")
    n = len(ids)
    if len(set(ids)) != n or len(sources) != n:
        raise ValueError("Unique records and aligned complete source ownership required")
    order = _integers(payload["priority_indices"], "priority_indices")
    ptr = _integers(payload["indptr"], "indptr")
    ix = _integers(payload["indices"], "indices")
    if len(order) != n or set(order) != set(range(n)):
        raise ValueError("Priority indices must be a complete record permutation")
    if len(ptr) != n + 1 or not ptr or ptr[0] != 0 or ptr[-1] != len(ix) or any(a > b for a, b in zip(ptr, ptr[1:])):
        raise ValueError("Invalid canonical CSR offsets")
    if any(i >= n for i in ix):
        raise ValueError("Blocker index outside the record population")
    ranks = {v: j for j, v in enumerate(order)}
    for v in range(n):
        row = ix[ptr[v]:ptr[v + 1]]
        if any(a >= b for a, b in zip(row, row[1:])) or any(ranks[u] >= ranks[v] for u in row):
            raise ValueError("Canonical blockers must be sorted distinct earlier raw neighbors")
    hex_value = payload["threshold_hex"]
    if type(hex_value) is not str:
        raise ValueError("Canonical float.hex threshold required")
    try:
        threshold = float.fromhex(hex_value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Invalid graph threshold") from exc
    if not math.isfinite(threshold) or not -1 <= threshold <= 1 or threshold.hex() != hex_value:
        raise ValueError("Finite canonical float.hex threshold in [-1,1] required")
    return {"schema": SCHEMA, "record_ids": ids, "source_ids": sources,
            "priority_indices": order, "indptr": ptr, "indices": ix, "threshold_hex": hex_value}


def _graph(payload):
    p = _validate(payload)
    result = BlockerGraph(p["record_ids"], p["source_ids"],
                          _immutable_int64(p["priority_indices"]),
                          _immutable_int64(p["indptr"]), _immutable_int64(p["indices"]),
                          float.fromhex(p["threshold_hex"]))
    rq._graph_state(result)
    return result


class CanonicalGraph(dict):
    """Fingerprint-compatible immutable value with the frozen graph interface.

    Runtime graph values are always derived from signed dictionary fields.
    There is no private cached graph that could disagree with those fields.
    Graph-array/method access allocates/validates O(N+E) metadata; scalar and
    tuple properties directly read constructor-validated immutable values.
    No constant-memory or repair-speed claim is made. This is not a
    hostile-Python-object sandbox.
    """
    __slots__ = ()

    def __init__(self, payload):
        if self:
            raise TypeError("Canonical graph is already initialized")
        canonical = _validate(payload)
        _graph(canonical)
        dict.__init__(self, canonical)

    def _immutable(self, *args, **kwargs):
        raise TypeError("Canonical graph fields are immutable; construct and review a new descriptor")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __copy__(self):
        return self

    def copy(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @property
    def record_ids(self):
        return self["record_ids"]

    @property
    def source_ids(self):
        return self["source_ids"]

    @property
    def priority_indices(self):
        return _graph(self).priority_indices

    @property
    def indptr(self):
        return _graph(self).indptr

    @property
    def indices(self):
        return _graph(self).indices

    @property
    def threshold(self):
        return float.fromhex(self["threshold_hex"])

    @property
    def priority(self):
        return _graph(self).priority

    @property
    def blockers(self):
        return _graph(self).blockers

    @property
    def blocker_counts(self):
        return _graph(self).blocker_counts

    @property
    def edge_count(self):
        return len(self["indices"])

    def selected_indices(self, deleted_records=()):
        return _graph(self).selected_indices(deleted_records)

    def accounting(self):
        return _graph(self).accounting()


def graph_payload(graph):
    """Export an existing graph as plain canonical JSON, retaining full identity.

    Input record order and source ownership are preserved. CSR rows are sorted
    by input index; the frozen graph/source binding must remain exactly equal.
    This does not certify that the graph was built from authentic features.
    """
    original_binding = rq.graph_binding(graph)
    order = np.asarray(graph.priority_indices)
    ptr = np.asarray(graph.indptr)
    indices = np.asarray(graph.indices)
    if order.ndim != 1 or order.dtype.kind not in "iu" or ptr.dtype.kind not in "iu" or indices.dtype.kind not in "iu":
        raise ValueError("Graph priority and CSR arrays must have integer dtype")
    if isinstance(graph.threshold, (bool, np.bool_)):
        raise ValueError("Boolean threshold is not a graph threshold")
    canonical_indices = [int(v) for i in range(len(graph.record_ids))
                         for v in sorted(indices[int(ptr[i]):int(ptr[i + 1])])]
    payload = {"schema": SCHEMA, "record_ids": list(graph.record_ids),
               "source_ids": list(graph.source_ids), "priority_indices": [int(x) for x in order],
               "indptr": [int(x) for x in ptr], "indices": canonical_indices,
               "threshold_hex": float(graph.threshold).hex()}
    adapted = CanonicalGraph(payload)
    if rq.graph_binding(adapted) != original_binding:
        raise ValueError("Graph serialization changed the frozen graph/source binding")
    return task.fingerprint(adapted)


def hydrate_human_graphs(value):
    """Hydrate only corpus_inputs[*].graph in an already bound-loaded dossier.

    The frozen fingerprint remains identical. Features, rows, provenance,
    requests, reviews, and response values are not transformed or invented.
    Ordinary non-human bundles are returned unchanged.
    """
    if not isinstance(value, dict) or "corpus_inputs" not in value:
        return value
    frames = value["corpus_inputs"]
    if not isinstance(frames, dict):
        raise ValueError("Human corpus_inputs must be an object")
    hydrated = {}
    for corpus, inputs in frames.items():
        if type(corpus) is not str or not corpus or not isinstance(inputs, dict) or "graph" not in inputs:
            raise ValueError("Complete named human corpus frame with graph descriptor required")
        descriptor = inputs["graph"]
        graph = CanonicalGraph(descriptor)
        if task.fingerprint(graph) != task.fingerprint(descriptor):
            raise ValueError("Graph hydration changed the signed descriptor fingerprint")
        hydrated[corpus] = {**inputs, "graph": graph}
    return {**value, "corpus_inputs": hydrated}


def reviewed_dossier_sha256(dossier):
    """Compute the exact frozen narrow-review digest; create no testimony.

    A graph object must first be exported with graph_payload. Hashing and
    hydration validate structural identity, never provenance or human origin.
    """
    if not isinstance(dossier, dict):
        raise ValueError("Stage dossier must be an object")
    hydrated = hydrate_human_graphs(dossier)
    return recipes.digest(task.fingerprint({k: v for k, v in hydrated.items()
                                           if k != "external_evidence_review"}))
