#!/usr/bin/env python3
"""Audit compact payload comparator on unchanged acquired Civil records."""
from pathlib import Path
import json
import numpy as np

from ccu.data import read_natural_jsonl, lexical_engineering_features, sha256_file
from ccu.core import build_blocker_graph, direct_oracle
from ccu.compact_payload import CompactEligiblePayloadState

ROOT = Path(__file__).resolve().parent

def main():
    rows = read_natural_jsonl(ROOT / "data/civil_comments_engineering_preview.jsonl", require_labels=True)
    ids = [r["record_id"] for r in rows]
    y = np.array([r["label"] for r in rows])
    e, _ = lexical_engineering_features(rows, 128)
    graph = build_blocker_graph(e, ids, .6)
    records, checks, initial = [], [], {}
    for d in (64, 128, 768):
        z, _ = lexical_engineering_features(rows, d)
        s = CompactEligiblePayloadState.build(z, y, graph.blockers, 8)
        initial[str(d)] = s.accounting()
        for deleted in range(9):
            if deleted:
                s.delete([deleted - 1])
            s.check_invariants()
            oracle = direct_oracle(z, y, ids, .6, .01, ids[:deleted], graph.priority, graph_features=e)
            solution = s.decode(.01)
            moments = s.moments()
            head_error = float(np.max(np.abs(solution.weights - oracle.solution.weights)))
            moment_error = max(float(np.max(np.abs(moments.gram - oracle.moments.gram))),
                               float(np.max(np.abs(moments.cross - oracle.moments.cross))))
            assert set(s.selected_ids()) == set(map(int, oracle.selected_indices))
            assert moments.count == oracle.moments.count
            assert head_error < 1e-9 and moment_error < 1e-9
            records.append({"dimension": d, "deleted": deleted, "head_error": head_error,
                            "moment_error": moment_error, "solver": solution.solver})
    z, _ = lexical_engineering_features(rows, 64)
    s = CompactEligiblePayloadState.build(z, y, graph.blockers, len(rows))
    for deleted in range(1, len(rows) + 1):
        s.delete([deleted - 1])
        s.check_invariants()
        oracle = direct_oracle(z, y, ids, .6, .01, ids[:deleted], graph.priority, graph_features=e)
        error = float(np.max(np.abs(s.solve_ridge(.01) - oracle.solution.weights)))
        assert error < 1e-9 and set(s.selected_ids()) == set(map(int, oracle.selected_indices))
        records.append({"dimension": 64, "full_deletion_prefix": deleted, "head_error": error})
    assert s.accounting()["all_array_bytes"] == 0 and not np.any(s.solve_ridge(.01))
    s = CompactEligiblePayloadState.build(z, y, graph.blockers, 2)
    def reject(values, name):
        before = s.state_digest()
        try:
            s.delete(values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid request accepted: " + name)
        assert before == s.state_digest()
        checks.append(name)
    reject([0, 0], "duplicate_atomic")
    reject([len(rows)], "unknown_atomic")
    reject([0, 1, 2], "budget_atomic")
    reject([True], "bool_id_atomic")
    reject([0.0], "float_id_atomic")
    s.delete([0])
    reject([0], "retry_atomic")
    s.delete([1])
    reject([2], "exhausted_atomic")
    reports = [CompactEligiblePayloadState.build(z, y, graph.blockers, 8) for _ in range(3)]
    for i in range(8):
        reports[0].delete([i])
    reports[1].delete(range(8))
    for i in reversed(range(8)):
        reports[2].delete([i])
    assert len({s.state_digest() for s in reports}) == 1
    checks.append("batch_singleton_reversed_identical_logical_state")
    report = {
        "status": "passed", "confirmatory": False,
        "dataset_sha256": sha256_file(ROOT / "data/civil_comments_engineering_preview.jsonl"),
        "feature_scope": "fixed128d lexical curator; unchanged natural text and original labels",
        "checked_model_states": len(records), "boundary_checks": checks,
        "max_abs_head_error": max(r["head_error"] for r in records),
        "initial_accounting": initial, "checked_states": records,
        "scope_limit": "FP64 software checks; no formal floating certificate, semantic utility or performance claim",
    }
    destination = ROOT / "results/compact_payload_audit.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "checked_model_states", "boundary_checks", "max_abs_head_error")}, indent=2))

if __name__ == "__main__":
    main()
