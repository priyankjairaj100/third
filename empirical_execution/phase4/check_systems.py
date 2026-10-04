"""Software command tests only; no empirical method timing or paper systems run."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

from empirical_execution.phase4 import systems

ROOT = Path(__file__).resolve().parents[2]
CHECKS = []


def check(name, passed):
    if not passed:
        raise AssertionError(name)
    CHECKS.append(name)


def rejects(name, fn):
    try:
        fn()
    except ValueError:
        check(name, True)
    else:
        raise AssertionError(name + ": accepted")


def command(mid, code, inputs=None):
    return {"method_id": mid, "argv": [sys.executable, "-c", code, "{output_dir}"], "input_files": inputs or []}


def spec(methods, **kwargs):
    return {"evidence_role": "software_harness_validation", "methods": methods, "timeout_seconds": 3,
            "threads": 1, "order_seed": 20261004, **kwargs}


def main():
    with tempfile.TemporaryDirectory(prefix="systems-software-check-", dir=ROOT) as folder:
        workspace = Path(folder)
        payload = workspace / "software_fixture_input.txt"
        payload.write_bytes(b"software harness fixture, not empirical data\n")
        descriptor = {"path": payload.name, "sha256": systems.file_hash(payload)}
        code = "import json,os,pathlib,sys; p=pathlib.Path(sys.argv[1]); (p/'payload.bin').write_bytes(bytes(257)); (p/'runtime.json').write_text(json.dumps({'pid':os.getpid(),'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED']}})); print('ok')"
        success = systems.run_benchmark(spec([command("alpha", code, [descriptor]), command("beta", code, [descriptor])]), workspace, workspace / "success")
        check("default_exactly_five_repetitions_per_method", len(success["runs"]) == 10 and all(sum(r["method_id"] == m for r in success["runs"]) == 5 for m in ("alpha", "beta")))
        check("all_success_measurements_complete", success["successful_complete_runs"] == 10 and success["failed_or_incomplete_runs"] == 0)
        check("randomized_order_prespecified_and_reproducible", success["planned_order"] == systems.planned_order(success["spec"]) and [{k:r[k] for k in ("repetition","position","method_id")} for r in success["runs"]] == success["planned_order"])
        check("each_repetition_contains_both_methods_once", all(sorted(r["method_id"] for r in success["runs"] if r["repetition"] == repeat) == ["alpha", "beta"] for repeat in range(5)))
        pids = []
        for row in success["runs"]:
            runtime = json.loads((Path(row["output_directory"]) / "runtime.json").read_text())
            pids.append(runtime["pid"])
            check("fixed_threads_and_hashseed_" + str(len(pids)), runtime["threads"] == {"OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "PYTHONHASHSEED":"0"})
            actual_bytes = sum(p.stat().st_size for p in Path(row["run_directory"]).rglob("*") if p.is_file())
            check("complete_regular_output_byte_ledger_" + str(len(pids)), actual_bytes == row["persistent_output_logical_bytes"])
            check("command_and_input_hashes_" + str(len(pids)), row["argv_canonical_utf8_sha256"] == systems.digest(row["argv"]) and row["inputs_before"][0]["matches"] and row["declared_inputs_unchanged"])
        check("all_runs_fresh_distinct_processes", len(set(pids)) == 10 and os.getpid() not in pids)
        check("positive_wait4_RSS_and_charged_end_to_end", all(r["child_peak_rss_bytes"] > 0 and r["charged_end_to_end_elapsed_seconds"] >= r["spawn_to_reap_elapsed_seconds"] > 0 for r in success["runs"]))
        check("ledger_persistence_receipt_matches", json.loads((workspace / "success/persistence_receipt.json").read_text())["benchmark_json_sha256"] == systems.file_hash(workspace / "success/benchmark.json"))
        rejects("old_results_never_overwritten", lambda: systems.run_benchmark(success["spec"], workspace, workspace / "success"))
        rejects("more_than_five_repetitions_rejected", lambda: systems.run_benchmark(spec([command("x", code)], repetitions=6), workspace, workspace / "six"))
        rejects("zero_timeout_rejected", lambda: systems.run_benchmark(spec([command("x", code)], timeout_seconds=0), workspace, workspace / "zero"))
        rejects("unstructured_shell_string_rejected", lambda: systems.run_benchmark(spec([{"method_id":"x", "argv":"echo hello", "input_files":[]}]), workspace, workspace / "shell"))
        failures = systems.run_benchmark(spec([
            command("exit7", "import sys; print('retained failure'); sys.exit(7)"),
            command("signal", "import os,signal; os.kill(os.getpid(),signal.SIGTERM)"),
            command("timeout", "import time; time.sleep(2)"),
            {"method_id":"missing_executable", "argv":[str(workspace / "no-such-executable")], "input_files":[]},
            command("bad_hash", "raise AssertionError('must not run')", [{"path":payload.name,"sha256":"0"*64}]),
            command("output_symlink", "import pathlib,sys; (pathlib.Path(sys.argv[1])/'unmeasured-link').symlink_to('../stdout.log')"),
        ], repetitions=1, timeout_seconds=.15, termination_grace_seconds=.1), workspace, workspace / "failures")
        byid = {r["method_id"]:r for r in failures["runs"]}
        check("nonzero_exit_retained", byid["exit7"]["returncode"] == 7 and byid["exit7"]["status"] == "nonzero_exit")
        check("signal_retained", byid["signal"]["signal"] == signal.SIGTERM and byid["signal"]["status"] == "signal")
        check("timeout_retained", byid["timeout"]["timed_out"] and byid["timeout"]["status"] == "timeout")
        check("launch_failure_retained", byid["missing_executable"]["status"] == "launch_or_io_failure")
        check("wrong_input_hash_prevents_execution", byid["bad_hash"]["returncode"] is None and not byid["bad_hash"]["declared_inputs_unchanged"])
        check("output_symlink_incomplete_not_success_evidence", byid["output_symlink"]["status"] == "success" and not byid["output_symlink"]["eligible_for_success_timing_summary"] and byid["output_symlink"]["output_inspection_errors"])
        check("all_failed_incomplete_trials_remain_in_registry", len(failures["runs"]) == 6 and failures["failed_or_incomplete_runs"] == 6)
        mutate = systems.run_benchmark(spec([command("mutate", "import pathlib; pathlib.Path('software_fixture_input.txt').write_text('changed')", [descriptor])], repetitions=1), workspace, workspace / "mutate")
        check("declared_input_mutation_retained_as_failure", mutate["runs"][0]["status"] == "input_changed_or_mismatched")
        check("scope_never_promotes_software_to_paper_evidence", all(r["paper_systems_result"] is False and r["confirmatory_study_ready"] is False for r in (success, failures, mutate)))
        report = {"passed": True, "check_count": len(CHECKS), "checks": CHECKS,
                  "scope": "software subprocess/measurement checks only; no corpus, model, empirical method comparison, or paper systems run",
                  "temporary_test_outputs_removed_after_check": True,
                  "retained_raw_software_command_reports": [success, failures, mutate],
                  "module_sha256": systems.file_hash(systems.__file__), "paper_systems_result": False,
                  "confirmatory_study_ready": False}
        destination = ROOT / "empirical_execution/phase4/results/systems_checks.json"
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "checks": len(CHECKS), "paper_systems_result": False}))


if __name__ == "__main__":
    main()
