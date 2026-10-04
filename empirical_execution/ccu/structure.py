"""Label-blind structural audits for counterfactual curation.

All functions operate on an already constructed natural-data ``BlockerGraph``.
No texts, embeddings, labels, duplicate records, or experimental graphs are
created here. A structural coefficient-key count is *not* a measured count of
nonzero vector coefficients: statistic-dependent cancellation may remove keys.
Rank is the exact incidence-matrix rank for arbitrary independent statistics;
it is not a total-memory lower bound for structured ridge moments.

Random workloads are simulated withdrawals, not observed withdrawal histories.
Uniform samplers use Python's seeded ``random.sample`` on sorted stable IDs;
this is a partial permutation, with the same distribution as full-permutation
prefixes. Record and source service universes are never narrowed to a sampling
frame when computing eligibility, key counts, or rank.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
import random
import time
from typing import Any, Iterable, Literal, Sequence

from .core import BlockerGraph

Unit = Literal["record", "source"]


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(value: int, name: str) -> int:
    value = _nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _universe(graph: BlockerGraph, unit: Unit) -> tuple[str, ...]:
    if unit == "record":
        return tuple(sorted(graph.record_ids))
    if unit == "source":
        return tuple(sorted(set(graph.source_ids)))
    raise ValueError("unit must be 'record' or 'source'")


def _frame(graph: BlockerGraph, unit: Unit,
           sampling_frame: Iterable[str] | None) -> tuple[str, ...]:
    universe = _universe(graph, unit)
    if sampling_frame is None:
        return universe
    result = tuple(sampling_frame)
    if any(not isinstance(x, str) for x in result):
        raise ValueError("sampling frame must contain string IDs")
    if len(result) != len(set(result)):
        raise ValueError("sampling frame contains repeated IDs")
    if not set(result) <= set(universe):
        raise ValueError("sampling frame contains IDs outside the service universe")
    return tuple(sorted(result))


def _terms(graph: BlockerGraph, unit: Unit) -> tuple[
        tuple[str, ...], list[tuple[int, tuple[int, ...]]]]:
    """Return variable universe and (owner, distinct-blocker-variable) terms.

    Source variables range over the entire ownership partition, including
    singleton fallback sources. Own-source blocker terms are retained here so
    callers can explicitly count and omit their identically zero columns.
    """
    universe = _universe(graph, unit)
    lookup = {name: i for i, name in enumerate(universe)}
    names = graph.record_ids if unit == "record" else graph.source_ids
    owner = [lookup[x] for x in names]
    terms = []
    for v in range(len(graph.record_ids)):
        blockers = graph.indices[graph.indptr[v]:graph.indptr[v + 1]]
        variables = tuple(sorted({owner[int(u)] for u in blockers}))
        terms.append((owner[v], variables))
    return universe, terms


class _DSU:
    def __init__(self) -> None:
        # Node zero is ground and is present even if no edge reaches it.
        self.parent = [0]
        self.size = [1]

    def add(self) -> int:
        node = len(self.parent)
        self.parent.append(node)
        self.size.append(1)
        return node

    def find(self, node: int) -> int:
        while node != self.parent[node]:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _incidence_census(terms: Sequence[tuple[int, tuple[int, ...]]],
                      horizon: int) -> dict[str, int]:
    dsu = _DSU()
    nodes: dict[tuple[int, ...], int] = {}
    count_coefficients: Counter[tuple[int, ...]] = Counter()
    eligible = zero_columns = interior = boundary = 0

    def node(key: tuple[int, ...]) -> int:
        if key not in nodes:
            nodes[key] = dsu.add()
        return nodes[key]

    for owner, blockers in terms:
        if owner in blockers:
            zero_columns += 1
            continue
        degree = len(blockers)
        if degree > horizon:
            continue
        eligible += 1
        tail = node(blockers)
        count_coefficients[blockers] += 1
        if degree < horizon:
            interior += 1
            head_key = tuple(sorted((*blockers, owner)))
            head = node(head_key)
            count_coefficients[head_key] -= 1
        else:
            boundary += 1
            head = 0
        dsu.union(tail, head)

    ground_root = dsu.find(0)
    roots = {dsu.find(v) for v in nodes.values()}
    c0 = len(roots - {ground_root})
    p = len(nodes)
    count_nonzero = sum(value != 0 for value in count_coefficients.values())
    return {
        "horizon": horizon,
        "eligible_records": eligible,
        "identically_zero_owner_blocker_records": zero_columns,
        "interior_columns": interior,
        "boundary_columns": boundary,
        "structural_coefficient_keys": p,
        "ungrounded_components": c0,
        "query_rank": p - c0,
        "nonzero_count_coordinate_keys": count_nonzero,
        "key_incidence_entries": sum(len(key) for key in nodes),
        "key_tuple_work_potential": sum(len(key) * (len(key) + 1) // 2 for key in nodes),
    }


def incidence_census(graph: BlockerGraph, horizon: int,
                     unit: Unit = "record") -> dict[str, int]:
    """Exact DSU rank p-c0, structural keys, and full-service eligibility.

    ``structural_coefficient_keys`` is the vector-map upper bound p. The count
    coordinate certifies at least ``nonzero_count_coordinate_keys`` nonzero
    vectors when each record statistic has final coordinate one. Actual M
    requires the numerical summary. Neither bound includes metadata bytes.
    """
    horizon = _nonnegative_integer(horizon, "horizon")
    universe, terms = _terms(graph, unit)
    if horizon > len(universe):
        raise ValueError("horizon exceeds service-universe size")
    return _incidence_census(terms, horizon)


def record_rank_profile(graph: BlockerGraph) -> list[dict[str, int]]:
    """Compute the complete record rank curve by reduced blocker signatures.

    Returns k=0 through min(N, max blocker count + 1). Later admissible horizons
    all have rank N. This implementation is independent of the DSU calculation
    and does not construct any request matrix.
    """
    n = len(graph.record_ids)
    if n == 0:
        return [{"horizon": 0, "query_rank": 0, "eligible_records": 0}]
    roots_by_head: dict[tuple[int, ...], int] = {}
    root_ids: dict[tuple[int, ...], int] = {}
    boundary_roots: dict[int, set[int]] = {}
    counts: Counter[int] = Counter()
    next_root = 0
    for value in graph.priority_indices:
        v = int(value)
        blockers = tuple(int(x) for x in graph.indices[graph.indptr[v]:graph.indptr[v + 1]])
        root = roots_by_head.get(blockers)
        if root is None:
            if blockers not in root_ids:
                root_ids[blockers] = next_root
                next_root += 1
            root = root_ids[blockers]
        roots_by_head[tuple(sorted((*blockers, v)))] = root
        degree = len(blockers)
        counts[degree] += 1
        boundary_roots.setdefault(degree, set()).add(root)
    profile = []
    interior = 0
    for horizon in range(min(n, max(counts) + 1) + 1):
        profile.append({"horizon": horizon,
                        "query_rank": interior + len(boundary_roots.get(horizon, ())),
                        "eligible_records": interior + counts[horizon]})
        interior += counts[horizon]
    return profile


def log_comb(n: int, k: int) -> float:
    """Log binomial coefficient, with invalid combinations represented by -inf."""
    if n < 0 or k < 0 or k > n:
        return -math.inf
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def activation_probability(population: int, required: int,
                           owner_in_frame: bool, request_size: int) -> float:
    """Uniform k-subset probability: all required units removed, owner retained.

    The caller must first verify that every blocker lies in the sampling frame
    and that the owner is not a required blocker. Those disqualifications cannot
    be inferred from counts alone. The formula handles owners outside the frame.
    """
    g = _nonnegative_integer(population, "population")
    b = _nonnegative_integer(required, "required")
    k = _nonnegative_integer(request_size, "request_size")
    if not isinstance(owner_in_frame, bool):
        raise ValueError("owner_in_frame must be Boolean")
    if k > g:
        raise ValueError("request_size exceeds population")
    a = int(owner_in_frame)
    numerator_n, numerator_k = g - b - a, k - b
    if numerator_n < 0 or numerator_k < 0 or numerator_k > numerator_n:
        return 0.0
    # Exact integer binomials avoid log-gamma cancellation for the protocol's
    # small horizons. Python computes a bounded float ratio of large integers.
    if min(k, g - k) <= 256:
        return math.comb(numerator_n, numerator_k) / math.comb(g, k)
    value = math.exp(log_comb(numerator_n, numerator_k) - log_comb(g, k))
    return min(1.0, max(0.0, value))


def expected_admissions(graph: BlockerGraph, request_size: int,
                        unit: Unit = "record",
                        sampling_frame: Iterable[str] | None = None) -> dict[str, Any]:
    """Exact finite-population mean formula evaluated in floating point.

    Admissions exclude initially selected records. This returns a mean count,
    never a probability that any admission occurs. A restricted frame affects
    this workload expectation only, not the service's storage eligibility.
    """
    k = _nonnegative_integer(request_size, "request_size")
    universe, terms = _terms(graph, unit)
    frame = _frame(graph, unit, sampling_frame)
    if k > len(frame):
        raise ValueError("request_size exceeds sampling-frame size")
    frame_names = set(frame)
    frame_indices = {i for i, name in enumerate(universe) if name in frame_names}
    groups: Counter[tuple[int, bool]] = Counter()
    outside = own_blocker = excluded = 0
    for owner, blockers in terms:
        if not blockers:
            continue
        excluded += 1
        if owner in blockers:
            own_blocker += 1
            continue
        if not set(blockers) <= frame_indices:
            outside += 1
            continue
        groups[(len(blockers), owner in frame_indices)] += 1
    expectation = math.fsum(count * activation_probability(len(frame), b, a, k)
                            for (b, a), count in groups.items())
    return {
        "unit": unit, "request_size": k,
        "service_universe_size": len(universe), "sampling_frame_size": len(frame),
        "initially_excluded_records": excluded,
        "own_source_blocker_records": own_blocker,
        "blockers_outside_sampling_frame_records": outside,
        "expected_admissions": expectation,
        "interpretation": "mean number, not probability of any admission",
    }


def ridge_storage_forecast(eligible_records: int, structural_keys: int,
                           query_rank: int, dimension: int, outputs: int,
                           feature_bytes: int = 4, target_bytes: int = 8,
                           moment_bytes: int = 8) -> dict[str, int | str]:
    """Arithmetic-only dense moments and eligible-payload forecasts.

    Excludes IDs, graph, key indices, runtime overhead, working space, serializer
    duplication and shared assets. Upper-bound dense coefficient storage may be
    reduced by actual vector cancellation. The generic rank forecast is not a
    lower bound for constrained ridge moments or a realizable zero-metadata
    implementation. Actual physical memory must be measured separately.
    """
    e = _nonnegative_integer(eligible_records, "eligible_records")
    p = _nonnegative_integer(structural_keys, "structural_keys")
    r = _nonnegative_integer(query_rank, "query_rank")
    d = _positive_integer(dimension, "dimension")
    c = _positive_integer(outputs, "outputs")
    fb = _positive_integer(feature_bytes, "feature_bytes")
    tb = _positive_integer(target_bytes, "target_bytes")
    mb = _positive_integer(moment_bytes, "moment_bytes")
    if r > p:
        raise ValueError("query_rank cannot exceed structural_keys")
    s = d * (d + 1) // 2 + d * c + 1
    vector_bytes = s * mb
    payload_bytes = e * (d * fb + c * tb)
    return {
        "dimension": d, "outputs": c, "packed_statistic_coordinates": s,
        "packed_vector_bytes": vector_bytes,
        "dense_structural_key_values_bytes_upper_bound": p * vector_bytes,
        "generic_rank_basis_values_bytes": r * vector_bytes,
        "eligible_feature_and_target_bytes": payload_bytes,
        "eligible_payload_plus_current_packed_moments_bytes": payload_bytes + vector_bytes,
        "feature_bytes_per_coordinate": fb, "target_bytes_per_coordinate": tb,
        "moment_bytes_per_coordinate": mb,
        "status": "arithmetic forecast; metadata, workspace and actual measurements excluded",
    }


def structural_census(graph: BlockerGraph, horizons: Sequence[int],
                      unit: Unit = "record",
                      sampling_frame: Iterable[str] | None = None,
                      dimensions: Sequence[int] = (768,),
                      outputs: Sequence[int] = (1, 20)) -> dict[str, Any]:
    """JSON-ready natural-graph census, full-service ranks and byte forecasts."""
    universe, terms = _terms(graph, unit)
    frame = _frame(graph, unit, sampling_frame)
    levels = sorted({_nonnegative_integer(h, "horizon") for h in horizons})
    if levels and levels[-1] > len(universe):
        raise ValueError("horizon exceeds service-universe size")
    hist = Counter(len(blockers) for _, blockers in terms)
    own_blocker = sum(owner in blockers for owner, blockers in terms)
    signatures = Counter(blockers for owner, blockers in terms if owner not in blockers)
    profile = record_rank_profile(graph) if unit == "record" else None
    profile_lookup = {row["horizon"]: row["query_rank"] for row in profile or ()}
    rows = []
    for horizon in levels:
        row: dict[str, Any] = _incidence_census(terms, horizon)
        if unit == "record":
            reduced_rank = profile_lookup.get(horizon, len(graph.record_ids))
            if row["query_rank"] != reduced_rank:
                raise AssertionError("DSU rank disagrees with reduced-signature rank")
            row["independent_reduced_signature_rank"] = reduced_rank
        row["storage_forecasts"] = [
            ridge_storage_forecast(row["eligible_records"], row["structural_coefficient_keys"],
                                   row["query_rank"], d, c)
            for d in dimensions for c in outputs]
        if horizon <= len(frame):
            row["workload_expectation"] = expected_admissions(graph, horizon, unit, frame)
        else:
            row["workload_expectation"] = None
        rows.append(row)
    source_sizes = Counter(graph.source_ids)
    return {
        "unit": unit, "records": len(graph.record_ids), "edges": int(graph.edge_count),
        "initially_selected_records": sum(len(b) == 0 for _, b in terms),
        "service_universe_size": len(universe), "sampling_frame_size": len(frame),
        "source_size_histogram": {str(k): v for k, v in sorted(Counter(source_sizes.values()).items())},
        "blocker_degree_histogram": {str(k): v for k, v in sorted(hist.items())},
        "identically_zero_owner_blocker_records": own_blocker,
        "distinct_nonzero_blocker_signatures": len(signatures),
        "blocker_signature_multiplicity_histogram": {
            str(k): v for k, v in sorted(Counter(signatures.values()).items())},
        "rank_profile": profile,
        "horizons": rows,
        "scope": "ranks and eligibility use complete service universe; expectations use explicit sampling frame",
    }


def derive_seed(master_seed: int, namespace: str, *parts: str) -> int:
    """Stable independent stream key without Python's process-randomized hash."""
    master_seed = _nonnegative_integer(master_seed, "master_seed")
    if any(not isinstance(x, str) for x in (namespace, *parts)):
        raise ValueError("seed namespace and parts must be strings")
    digest = hashlib.sha256()
    for value in (str(master_seed), namespace, *parts):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:16], "big")


@dataclass(frozen=True)
class RequestTrajectory:
    arm: str
    unit: Unit
    seed: int
    requested_horizon: int
    initial_horizon: int
    sampling_frame_size: int
    deletion_order: tuple[str, ...]
    checkpoints: tuple[int, ...]
    status: str = "ready"
    selected_candidate: str | None = None
    stress_candidate_pool: int = 0
    stress_candidates_scored: int = 0
    stress_unpadded_admissions: int | None = None
    stress_search_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prefix(self, checkpoint: int) -> tuple[str, ...]:
        if checkpoint not in self.checkpoints:
            raise ValueError("checkpoint is not declared for this trajectory")
        return self.deletion_order[:checkpoint]

    def batches(self) -> tuple[tuple[str, ...], ...]:
        """Incremental requests; cumulative horizon is never replenished."""
        previous = 0
        result = []
        for checkpoint in self.checkpoints:
            result.append(self.deletion_order[previous:checkpoint])
            previous = checkpoint
        return tuple(result)


def _checkpoints(checkpoints: Sequence[int], horizon: int) -> tuple[int, ...]:
    checked = [_positive_integer(x, "checkpoint") for x in checkpoints]
    if horizon == 0:
        return ()
    return tuple(sorted({min(x, horizon) for x in checked}))


def _admission_count(graph: BlockerGraph, deletion_indices: set[int]) -> int:
    count = 0
    for v in range(len(graph.record_ids)):
        blockers = graph.indices[graph.indptr[v]:graph.indptr[v + 1]]
        if len(blockers) and v not in deletion_indices and all(int(b) in deletion_indices for b in blockers):
            count += 1
    return count


def generate_request_trajectory(graph: BlockerGraph, arm: str, seed: int,
                                horizon: int, checkpoints: Sequence[int],
                                genuine_sources: Iterable[str] | None = None,
                                excluded_blockers_only: bool = False,
                                stress_candidates: int = 1024) -> RequestTrajectory:
    """Generate one R/S/U/A trajectory, without accessing labels or test results.

    S requires an explicit genuine-source frame; fallback singleton source IDs
    remain in the service universe. A chooses the sampled candidate's *unpadded*
    blocker set with most admissions, ties by stable candidate ID, then pads with
    a random order. A failure to find a candidate is explicitly recorded and
    yields the prescribed random padding, never a silently resampled success.
    """
    seed = _nonnegative_integer(seed, "seed")
    horizon = _nonnegative_integer(horizon, "horizon")
    stress_candidates = _positive_integer(stress_candidates, "stress_candidates")
    if arm not in {"R", "S", "U", "A"}:
        raise ValueError("arm must be R, S, U, or A")
    if excluded_blockers_only and arm != "A":
        raise ValueError("excluded_blockers_only applies only to arm A")
    rng = random.Random(seed)
    unit: Unit = "source" if arm == "S" else "record"
    all_records = tuple(sorted(graph.record_ids))
    excluded = tuple(sorted(graph.record_ids[i] for i in range(len(graph.record_ids))
                            if graph.indptr[i + 1] > graph.indptr[i]))
    if arm == "S":
        if genuine_sources is None:
            raise ValueError("arm S requires an explicit genuine_sources sampling frame")
        frame = _frame(graph, "source", genuine_sources)
    elif arm == "U":
        frame = excluded
    else:
        frame = all_records
    limit = min(horizon, len(frame))
    levels = _checkpoints(checkpoints, limit)
    status = "structural_zero_empty_sampling_frame" if not frame else "ready"
    metadata: dict[str, Any] = {}
    if arm != "A":
        order = tuple(rng.sample(frame, limit))
    else:
        started = time.perf_counter()
        lookup = {name: i for i, name in enumerate(graph.record_ids)}
        excluded_indices = {lookup[name] for name in excluded}
        pool = []
        for candidate in excluded:
            v = lookup[candidate]
            blockers = graph.indices[graph.indptr[v]:graph.indptr[v + 1]]
            if len(blockers) <= limit and (not excluded_blockers_only or
                    all(int(b) in excluded_indices for b in blockers)):
                pool.append(candidate)
        sampled = rng.sample(pool, min(len(pool), stress_candidates))
        best_candidate = None
        best_blockers: set[int] = set()
        best_score = -1
        for candidate in sorted(sampled):
            v = lookup[candidate]
            blockers = {int(b) for b in graph.indices[graph.indptr[v]:graph.indptr[v + 1]]}
            score = _admission_count(graph, blockers)
            if score > best_score:
                best_candidate, best_blockers, best_score = candidate, blockers, score
        prefix = tuple(sorted(graph.record_ids[b] for b in best_blockers))
        prefix_set = set(prefix)
        remaining = [name for name in frame if name not in prefix_set]
        order = prefix + tuple(rng.sample(remaining, limit - len(prefix)))
        if best_candidate is None:
            status = "no_stress_candidate_random_padding_only" if frame else status
        metadata = {"selected_candidate": best_candidate, "stress_candidate_pool": len(pool),
                    "stress_candidates_scored": len(sampled),
                    "stress_unpadded_admissions": best_score if best_candidate is not None else None,
                    "stress_search_seconds": time.perf_counter() - started}
    return RequestTrajectory(arm="A-excluded-blockers" if excluded_blockers_only else arm,
                             unit=unit, seed=seed, requested_horizon=horizon,
                             initial_horizon=limit, sampling_frame_size=len(frame),
                             deletion_order=order, checkpoints=levels, status=status, **metadata)


def trajectory_observations(graph: BlockerGraph,
                            trajectory: RequestTrajectory) -> list[dict[str, Any]]:
    """Graph-only measurements at cumulative checkpoints; includes every zero."""
    known = set(_universe(graph, trajectory.unit))
    if (len(set(trajectory.deletion_order)) != len(trajectory.deletion_order)
            or not set(trajectory.deletion_order) <= known):
        raise ValueError("trajectory has repeated or unknown deletion units")
    if len(trajectory.deletion_order) > trajectory.initial_horizon:
        raise ValueError("trajectory exceeds its initial horizon")
    if tuple(sorted(set(trajectory.checkpoints))) != trajectory.checkpoints:
        raise ValueError("checkpoints must be distinct and increasing")
    if any(k <= 0 or k > len(trajectory.deletion_order) for k in trajectory.checkpoints):
        raise ValueError("invalid checkpoint")
    original_selected = {i for i in range(len(graph.record_ids)) if graph.indptr[i] == graph.indptr[i + 1]}
    names = graph.record_ids if trajectory.unit == "record" else graph.source_ids
    rows = []
    for checkpoint in trajectory.checkpoints:
        units = set(trajectory.prefix(checkpoint))
        deleted = {i for i, name in enumerate(names) if name in units}
        selected = {
            i for i in range(len(graph.record_ids)) if i not in deleted and
            all(int(b) in deleted for b in graph.indices[graph.indptr[i]:graph.indptr[i + 1]])}
        additions = selected - original_selected
        rows.append({
            "arm": trajectory.arm, "seed": trajectory.seed, "unit": trajectory.unit,
            "cumulative_deleted_units": checkpoint,
            "cumulative_deleted_records": len(deleted),
            "originally_selected_deleted": len(original_selected & deleted),
            "retained_records": len(graph.record_ids) - len(deleted),
            "current_selected_records": len(selected), "admissions": len(additions),
            "has_admission": bool(additions),
            "remaining_horizon": trajectory.initial_horizon - checkpoint,
            "admission_fraction_of_initially_excluded": len(additions) / (
                len(graph.record_ids) - len(original_selected)) if len(graph.record_ids) > len(original_selected) else 0.0,
        })
    return rows


def match_source_trajectory_record_volume(graph: BlockerGraph,
                                          source_trajectory: RequestTrajectory,
                                          seed: int) -> RequestTrajectory:
    """Separate-horizon R control using one permutation at S's raw-count prefixes."""
    if source_trajectory.unit != "source":
        raise ValueError("requires a source trajectory")
    seed = _nonnegative_integer(seed, "seed")
    counts = tuple(row["cumulative_deleted_records"]
                   for row in trajectory_observations(graph, source_trajectory))
    levels = tuple(sorted(set(counts)))
    horizon = max(levels, default=0)
    frame = tuple(sorted(graph.record_ids))
    order = tuple(random.Random(seed).sample(frame, horizon))
    return RequestTrajectory(arm="R-volume-matched-to-S", unit="record", seed=seed,
                             requested_horizon=horizon, initial_horizon=horizon,
                             sampling_frame_size=len(frame), deletion_order=order,
                             checkpoints=levels,
                             status="ready" if levels else "no_source_checkpoints")
