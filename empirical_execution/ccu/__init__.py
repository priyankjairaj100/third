"""Counterfactual curation execution primitives; no experiment runs on import."""
from .core import (
    BlockerGraph, DeletionResult, EligiblePayloadState, OracleResult,
    RidgeMoments, RidgeSolution, build_blocker_graph, canonical_features,
    canonical_targets, direct_oracle, normalized_graph_features,
    ridge_moments, solve_ridge, stable_priority,
)

__all__ = [
    "BlockerGraph", "DeletionResult", "EligiblePayloadState", "OracleResult",
    "RidgeMoments", "RidgeSolution", "build_blocker_graph", "canonical_features",
    "canonical_targets", "direct_oracle", "normalized_graph_features",
    "ridge_moments", "solve_ridge", "stable_priority",
]
