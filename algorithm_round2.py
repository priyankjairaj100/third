#!/usr/bin/env python3
"""Indexed canonical coefficient repair, using exact sparse tuple keys.

Python standard library only. The deployed state contains a nonzero coefficient
map, current identifiers/horizon, and live inverse/degree indices. Stable Entry
objects let an index retain its handle while the entry's subset key shrinks.
Vectors are immutable integer tuples and are transferred without copying; only
coefficient collisions perform vector arithmetic. No original records, graph,
deleted-ID log, initial coefficient copy, or historical counter is in the state.

Instrumentation and full-rebuild oracles below are external test-harness state.
They must not be mistaken for part of the canonical learner. The guarantee is
logical state equality, modulo renaming internal handles; this is not secure
physical erasure of Python's allocator or constant-cost full serialization.

Run: python algorithm_round2.py --output round2_algorithm_verification.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations, permutations
import json
from math import comb
from pathlib import Path
import random
import time


@dataclass
class Meter:
    """An external observer; deliberately not retained by IndexedState."""

    initial_keys: int = 0
    initial_incidences: int = 0
    initial_quadratic_potential: int = 0
    input_identifiers: int = 0
    fresh_deletions: int = 0
    key_shrink_events: int = 0
    tuple_cells_copied: int = 0
    coefficient_collisions: int = 0
    scalar_additions: int = 0
    cancellations: int = 0
    horizon_pruned_keys: int = 0
    inverse_incidence_removals: int = 0
    full_scan_keys_for_same_atomic_sequence: int = 0


class Entry:
    # Object identity is a stable internal handle, not a record or a secret ID.
    __slots__ = ("key", "value")

    def __init__(self, key, value):
        self.key = key
        self.value = value


class IndexedState:
    __slots__ = ("universe_size", "horizon", "dimension", "alive", "coeff",
                 "inverse", "degrees")

    def __init__(self, universe_size, horizon, dimension, alive, coefficients):
        self.universe_size = universe_size  # public ID universe 0,...,N-1
        self.horizon = horizon
        self.dimension = dimension
        self.alive = set(alive)
        self.coeff = {}
        self.inverse = {}
        self.degrees = {}
        for key, value in coefficients.items():
            key, value = tuple(sorted(key)), tuple(value)
            assert len(key) <= horizon and len(value) == dimension
            assert set(key) <= self.alive
            if not any(value):
                continue
            entry = Entry(key, value)
            self.coeff[key] = entry
            self.degrees.setdefault(len(key), set()).add(entry)
            for u in key:
                self.inverse.setdefault(u, set()).add(entry)

    def meter(self):
        sizes = [len(key) for key in self.coeff]
        return Meter(initial_keys=len(sizes), initial_incidences=sum(sizes),
                     initial_quadratic_potential=sum(d*(d+1)//2 for d in sizes))

    def _remove_degree_handle(self, entry, degree):
        bucket = self.degrees[degree]
        bucket.discard(entry)
        if not bucket:
            del self.degrees[degree]

    def _remove_remaining_inverse_handles(self, entry, meter):
        for u in entry.key:
            index = self.inverse[u]
            index.remove(entry)
            meter.inverse_incidence_removals += 1
            if not index:
                del self.inverse[u]

    def _drop_live_entry(self, entry, meter):
        """Permanently remove an entry, including every live index reference."""
        del self.coeff[entry.key]
        self._remove_degree_handle(entry, len(entry.key))
        self._remove_remaining_inverse_handles(entry, meter)

    def _delete_current_one(self, u, meter):
        assert u in self.alive and self.horizon > 0
        old_horizon = self.horizon
        meter.full_scan_keys_for_same_atomic_sequence += len(self.coeff)
        meter.fresh_deletions += 1

        # Only these entries contain u. New keys never contain u, so no entry
        # can be revisited in this atomic operation. pop avoids rescanning a
        # sparsely occupied Python set from its beginning on every iteration.
        while u in self.inverse:
            index = self.inverse[u]
            entry = index.pop()
            meter.inverse_incidence_removals += 1
            if not index:
                del self.inverse[u]
            old_key = entry.key
            del self.coeff[old_key]
            self._remove_degree_handle(entry, len(old_key))
            meter.key_shrink_events += 1
            meter.tuple_cells_copied += len(old_key)
            entry.key = tuple(v for v in old_key if v != u)
            # All remaining inverse lists already reference this same object.
            # No reindexing of surviving incidences is needed.
            target = self.coeff.get(entry.key)
            if target is None:
                self.coeff[entry.key] = entry
                self.degrees.setdefault(len(entry.key), set()).add(entry)
                # entry.value is transferred by reference, with no vector copy.
            else:
                self._remove_remaining_inverse_handles(entry, meter)
                meter.coefficient_collisions += 1
                meter.scalar_additions += self.dimension
                merged = tuple(a+b for a, b in zip(target.value, entry.value))
                if any(merged):
                    target.value = merged
                else:
                    meter.cancellations += 1
                    self._drop_live_entry(target, meter)

        # Every touched key lost one variable and already meets the new cutoff.
        # Remaining old-degree-h keys are untouched and must be discarded.
        while old_horizon in self.degrees:
            entry = self.degrees[old_horizon].pop()
            self._drop_live_entry(entry, meter)
            meter.horizon_pruned_keys += 1

        self.alive.remove(u)
        self.horizon -= 1

    def delete(self, identifiers, meter):
        """Idempotent batch delete; retries consume no additional horizon.

        Validate the full request before mutating. Identifiers outside the
        public universe are errors; identifiers in that universe but absent
        from the current set are harmless retries. No historical ID log is
        needed. Batch requests are decomposed into singleton canonical updates.
        """
        requested = list(identifiers)
        meter.input_identifiers += len(requested)
        if any(not isinstance(u, int) or not 0 <= u < self.universe_size
               for u in requested):
            raise ValueError("Identifier outside the public universe")
        fresh = set(requested) & self.alive
        if len(fresh) > self.horizon:
            raise ValueError("Deletion request exceeds remaining horizon")
        for u in fresh:
            self._delete_current_one(u, meter)
        return len(fresh)

    def coefficients(self):
        return {key: entry.value for key, entry in self.coeff.items()}

    def constant(self):
        entry = self.coeff.get(())
        return (0,)*self.dimension if entry is None else entry.value

    def snapshot(self):
        """Expensive deterministic serialization for testing, not repair."""
        return {"universe_size": self.universe_size, "horizon": self.horizon,
                "dimension": self.dimension, "alive": tuple(sorted(self.alive)),
                "coefficients": tuple(sorted(self.coefficients().items())),
                "inverse": tuple(sorted((u, tuple(sorted(e.key for e in index)))
                                        for u, index in self.inverse.items())),
                "degrees": tuple(sorted((d, tuple(sorted(e.key for e in bucket)))
                                        for d, bucket in self.degrees.items()))}

    def check_invariants(self):
        """Full audit deliberately outside update-time measurements."""
        expected_inverse = {}
        expected_degrees = {}
        assert 0 <= self.horizon <= len(self.alive)
        for key, entry in self.coeff.items():
            assert key == entry.key == tuple(sorted(set(key)))
            assert len(key) <= self.horizon
            assert len(entry.value) == self.dimension and any(entry.value)
            assert set(key) <= self.alive
            expected_degrees.setdefault(len(key), set()).add(entry)
            for u in key:
                expected_inverse.setdefault(u, set()).add(entry)
        assert expected_inverse == self.inverse
        assert expected_degrees == self.degrees


def add_to_map(result, key, value):
    if key not in result:
        new_value = tuple(value)
    else:
        new_value = tuple(a+b for a, b in zip(result[key], value))
    if any(new_value):
        result[key] = new_value
    else:
        result.pop(key, None)


def rebuild(blockers, records, alive, horizon):
    """Independent direct rebuild from the retained graph and record values."""
    result = {}
    for v in alive:
        b = tuple(sorted(set(blockers[v]) & alive))
        if len(b) <= horizon:
            add_to_map(result, b, records[v])
        if len(b) < horizon:
            add_to_map(result, tuple(sorted((*b, v))), tuple(-x for x in records[v]))
    return result


def direct_selected_aggregate(blockers, records, alive):
    dimension = len(records[0]) if records else 3
    result = [0]*dimension
    for v in alive:
        if not (set(blockers[v]) & alive):
            result = [a+b for a, b in zip(result, records[v])]
    return tuple(result)


def assert_bounds(state, meter):
    assert meter.key_shrink_events <= meter.initial_incidences
    assert meter.inverse_incidence_removals <= meter.initial_incidences
    assert meter.coefficient_collisions <= meter.initial_keys
    assert meter.tuple_cells_copied <= meter.initial_quadratic_potential
    assert len(state.coeff) <= meter.initial_keys
    assert meter.scalar_additions == state.dimension * meter.coefficient_collisions


def compare_to_rebuild(state, blockers, records, meter):
    target_map = rebuild(blockers, records, state.alive, state.horizon)
    assert state.coefficients() == target_map
    oracle = IndexedState(state.universe_size, state.horizon, state.dimension,
                          state.alive, target_map)
    assert state.snapshot() == oracle.snapshot()
    assert state.constant() == direct_selected_aggregate(blockers, records, state.alive)
    state.check_invariants()
    assert_bounds(state, meter)


def make_state(blockers, records, horizon):
    n = len(blockers)
    dimension = len(records[0]) if records else 3
    alive = set(range(n))
    return IndexedState(n, horizon, dimension, alive,
                        rebuild(blockers, records, alive, horizon))


def exhaustive_checks(max_n=4):
    counts = {"graphs": 0, "ordered_sequences": 0, "canonical_rebuild_checks": 0,
              "retry_idempotence_checks": 0, "batch_vs_sequence_checks": 0}
    for n in range(max_n+1):
        edges = list(combinations(range(n), 2))
        records = [(v-2, (-1)**v*(v+1), v % 3 - 1) for v in range(n)]
        for graph_code in range(1 << comb(n, 2)):
            counts["graphs"] += 1
            blockers = [set() for _ in range(n)]
            for bit, (u, v) in enumerate(edges):
                if (graph_code >> bit) & 1:
                    blockers[v].add(u)
            for horizon in range(n+1):
                for order in permutations(range(n), horizon):
                    state = make_state(blockers, records, horizon)
                    meter = state.meter()
                    for u in order:
                        state.delete([u], meter)
                        compare_to_rebuild(state, blockers, records, meter)
                        counts["canonical_rebuild_checks"] += 1
                        previous = state.snapshot()
                        state.delete([u, u], meter)
                        assert state.snapshot() == previous
                        counts["retry_idempotence_checks"] += 1
                    batched = make_state(blockers, records, horizon)
                    batch_meter = batched.meter()
                    batched.delete(list(order)+list(order), batch_meter)
                    assert batched.snapshot() == state.snapshot()
                    assert_bounds(batched, batch_meter)
                    counts["batch_vs_sequence_checks"] += 1
                    counts["ordered_sequences"] += 1
    return counts


def run_case(name, blockers, records, horizon, requests):
    state = make_state(blockers, records, horizon)
    meter = state.meter()
    checks = 0
    for request in requests:
        state.delete(request, meter)
        compare_to_rebuild(state, blockers, records, meter)
        checks += 1
    return {"name": name, "vertices": len(blockers), "initial_horizon": horizon,
            "final_horizon": state.horizon, "remaining_keys": len(state.coeff),
            "canonical_rebuild_checks": checks, "work": asdict(meter)}


def adversarial_checks():
    cases = []
    n, k = 2048, 128
    records = [(v+1, (v % 7)-3, 1) for v in range(n)]
    cases.append(run_case("edgeless_sparse_touches_then_terminal_pruning",
                          [set() for _ in range(n)], records, k,
                          [[u, u] for u in range(k)]))
    n, k = 256, 128
    records = [(v+1, 1, v % 5) for v in range(n)]
    cases.append(run_case("clique_nested_keys_worst_shrink_work",
                          [set(range(v)) for v in range(n)], records, k,
                          [[u] for u in range(k)]))
    n = 1024
    star = [set()] + [{0} for _ in range(n-1)]
    records = [(v+1, 1, 2-v) for v in range(n)]
    cases.append(run_case("star_horizon_one", star, records, 1, [[0], [0, 0]]))
    cases.append(run_case("star_horizon_two", star, records, 2,
                          [[0], [0, 1], [0, 1, 1]]))
    n, k = 128, 64
    records = [(1 if v % 2 == 0 else -1,)*3 for v in range(n)]
    cases.append(run_case("alternating_exact_cancellations", [set() for _ in range(n)],
                          records, k, [[u] for u in range(k)]))
    cases.append(run_case("identically_zero_polynomial", [set(range(v)) for v in range(n)],
                          [(0, 0, 0) for _ in range(n)], k,
                          [list(range(start, start+8)) for start in range(0, k, 8)]))
    return cases


def random_checks(seed=20261003, trials=200):
    rng = random.Random(seed)
    totals = {"graphs": 0, "canonical_rebuild_checks": 0,
              "invalid_request_atomicity_checks": 0, "batch_order_checks": 0}
    for trial in range(trials):
        n = rng.randint(8, 120)
        k = rng.randint(1, min(n, 25))
        density = rng.choice((0.02, 0.05, 0.1, 0.3, 0.7))
        blockers = [{u for u in range(v) if rng.random() < density} for v in range(n)]
        records = [tuple(rng.randint(-5, 5) for _ in range(5)) for _ in range(n)]
        state = make_state(blockers, records, k)
        meter = state.meter()
        order = rng.sample(range(n), k)
        prefix = []
        cursor = 0
        while cursor < k:
            batch = order[cursor:cursor+rng.randint(1, 5)]
            request = batch+batch[:1]+rng.sample(prefix, min(len(prefix), 3))
            before = state.snapshot()
            try:
                state.delete([n], meter)
            except ValueError:
                assert state.snapshot() == before
            else:
                raise AssertionError("Invalid request was accepted")
            totals["invalid_request_atomicity_checks"] += 1
            state.delete(request, meter)
            compare_to_rebuild(state, blockers, records, meter)
            totals["canonical_rebuild_checks"] += 1
            prefix.extend(batch)
            cursor += len(batch)
        alternate = make_state(blockers, records, k)
        alternate_meter = alternate.meter()
        alternate.delete(reversed(order), alternate_meter)
        assert alternate.snapshot() == state.snapshot()
        assert_bounds(alternate, alternate_meter)
        totals["batch_order_checks"] += 1
        # Budget exhaustion must reject a fresh retained ID before mutation.
        if state.alive:
            before = state.snapshot()
            try:
                state.delete([next(iter(state.alive))], meter)
            except ValueError:
                assert before == state.snapshot()
            else:
                raise AssertionError("Exceeded horizon without rejection")
            totals["invalid_request_atomicity_checks"] += 1
        totals["graphs"] += 1
    return {"seed": seed, "counts": totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("round2_algorithm_verification.json"))
    parser.add_argument("--random-trials", type=int, default=200)
    args = parser.parse_args()
    started = time.monotonic()
    result = {"status": "passed", "arithmetic": "exact Python integers",
              "key_representation": "sorted sparse identifier tuples",
              "exhaustive": exhaustive_checks(),
              "adversarial": adversarial_checks(),
              "random": random_checks(trials=args.random_trials)}
    result["elapsed_seconds_including_oracle_rebuilds"] = time.monotonic()-started
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
