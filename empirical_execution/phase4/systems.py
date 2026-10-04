"""Sequential fresh-process systems measurement, not a filesystem sandbox.

Commands are trusted local programs. Inputs and output roots are declared and
checked, but subprocesses retain the caller's permissions. This harness cannot
prove absence of undeclared filesystem/network I/O or whole-machine interference.
wait4 peak RSS is an OS child high-water statistic, NOT summed process-tree RAM.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import time

VERSION = "ccu-fresh-process-systems-v1"
FIXED_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_spec(spec, workspace):
    if not hasattr(os, "wait4") or os.name != "posix":
        raise ValueError("This reference harness requires POSIX wait4; unsupported platforms fail explicitly")
    if not isinstance(spec, dict) or spec.get("evidence_role") not in {"software_harness_validation", "development_systems"}:
        raise ValueError("Explicit software_harness_validation/development_systems evidence role required")
    reps, threads = spec.get("repetitions", 5), spec.get("threads", 1)
    if isinstance(reps, bool) or not isinstance(reps, int) or not 1 <= reps <= 5:
        raise ValueError("Repetitions must be an integer in 1..5 (default five)")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("Positive explicit thread count required")
    if not _positive(spec.get("timeout_seconds")):
        raise ValueError("Positive finite per-child timeout_seconds required")
    grace = spec.get("termination_grace_seconds", 0.5)
    if not _positive(grace) or grace > 5:
        raise ValueError("Termination grace must be positive and at most five seconds")
    if isinstance(spec.get("order_seed"), bool) or not isinstance(spec.get("order_seed"), int):
        raise ValueError("Fixed integer order_seed required")
    if not isinstance(spec.get("methods"), list) or not spec["methods"]:
        raise ValueError("At least one separately declared method command required")
    ids = []
    for method in spec["methods"]:
        mid = method.get("method_id")
        if not isinstance(mid, str) or not mid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in mid):
            raise ValueError("Method IDs must be nonempty ASCII alphanumeric/_/-")
        ids.append(mid)
        argv = method.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a or "\0" in a for a in argv):
            raise ValueError("argv must be a nonempty array of nonempty NUL-free strings; no shell string")
        if any("{output_dir}" in a for a in argv[:1]):
            raise ValueError("Executable cannot be created in the output directory")
        if not isinstance(method.get("input_files"), list):
            raise ValueError("Explicit input_files list required (may be empty for software command tests)")
        seen = set()
        for item in method["input_files"]:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("Input descriptors require path and SHA256")
            path = (workspace / item["path"]).resolve()
            if not path.is_relative_to(workspace) or not path.is_file() or path in seen:
                raise ValueError("Each declared input must be a distinct existing file in the workspace")
            seen.add(path)
            sha = item.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("Every input requires a lowercase SHA256")
    if len(set(ids)) != len(ids):
        raise ValueError("Method IDs must be unique")
    return {**spec, "repetitions": reps, "threads": threads, "termination_grace_seconds": grace}


def planned_order(spec):
    """Prespecify all sequential method orders before any command is executed."""
    rng = random.Random(spec["order_seed"])
    order = []
    for repeat in range(spec["repetitions"]):
        methods = sorted(m["method_id"] for m in spec["methods"])
        rng.shuffle(methods)
        order.extend({"repetition": repeat, "position": pos, "method_id": mid} for pos, mid in enumerate(methods))
    return order


def _inputs(method, workspace):
    ledger = []
    for item in method["input_files"]:
        path = (workspace / item["path"]).resolve()
        if not path.is_relative_to(workspace):
            raise ValueError("Declared input now resolves outside workspace")
        actual = file_hash(path)
        ledger.append({"path": item["path"], "declared_sha256": item["sha256"], "actual_sha256": actual,
                       "logical_bytes": path.stat().st_size, "matches": actual == item["sha256"]})
    return ledger


def _persisted_outputs(run_root):
    """Inspect and fsync regular run outputs. Do not follow output symlinks."""
    entries, errors = [], []
    for root, dirs, files in os.walk(run_root, followlinks=False):
        dirs.sort()
        root = Path(root)
        for name in list(dirs):
            p = root / name
            if p.is_symlink():
                errors.append({"path": str(p.relative_to(run_root)), "error": "output_directory_symlink_not_traversed"})
                dirs.remove(name)
        for name in sorted(files):
            p = root / name
            relative = str(p.relative_to(run_root))
            if p.is_symlink() or not p.is_file():
                errors.append({"path": relative, "error": "nonregular_output_not_measured"})
                continue
            try:
                with p.open("rb") as stream:
                    os.fsync(stream.fileno())
                info = p.stat()
                entries.append({"path": relative, "logical_bytes": info.st_size,
                                "allocated_bytes": getattr(info, "st_blocks", 0) * 512,
                                "sha256": file_hash(p)})
            except OSError as err:
                errors.append({"path": relative, "error": type(err).__name__ + ": " + str(err)})
        try:
            fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as err:
            errors.append({"path": str(root.relative_to(run_root)), "error": "directory_fsync: " + str(err)})
    return entries, errors


def _terminate_group(pid, sig):
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False


def _run_one(method, identity, spec, workspace, run_root, environment):
    started = time.monotonic()
    run_root.mkdir()
    out = run_root / "artifacts"
    out.mkdir()
    argv = [a.replace("{output_dir}", str(out)) for a in method["argv"]]
    row = {**identity, "argv": argv, "argv_canonical_utf8_sha256": digest(argv),
           "argv_canonical_utf8_bytes": len(_canonical(argv)), "argv_template": method["argv"],
           "run_directory": str(run_root), "output_directory": str(out),
           "status": "not_started", "returncode": None, "signal": None, "timed_out": False,
           "child_peak_rss_bytes": None, "child_peak_rss_raw": None,
           "spawn_to_reap_elapsed_seconds": None, "child_user_cpu_seconds": None, "child_system_cpu_seconds": None,
           "remaining_process_group_detected": False, "failure_details": []}
    proc = None
    try:
        row["inputs_before"] = _inputs(method, workspace)
        if not all(x["matches"] for x in row["inputs_before"]):
            row["status"] = "input_hash_mismatch"
        else:
            executable = (argv[0] if os.path.isabs(argv[0]) else
                          str(workspace / argv[0]) if "/" in argv[0] else
                          shutil.which(argv[0], path=environment.get("PATH")))
            if executable and Path(executable).is_file():
                row["resolved_executable"] = str(Path(executable).resolve())
                row["executable_sha256"] = file_hash(executable)
            spawned = time.monotonic()
            with (run_root / "stdout.log").open("wb") as stdout, (run_root / "stderr.log").open("wb") as stderr:
                proc = subprocess.Popen(argv, cwd=workspace, env=environment, stdout=stdout, stderr=stderr,
                                        stdin=subprocess.DEVNULL, shell=False, start_new_session=True)
                termination_at = None
                while True:
                    pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
                    now = time.monotonic()
                    if pid:
                        proc.returncode = os.waitstatus_to_exitcode(status)
                        row.update(returncode=proc.returncode,
                                   signal=-proc.returncode if proc.returncode < 0 else None,
                                   spawn_to_reap_elapsed_seconds=now - spawned,
                                   child_peak_rss_raw=usage.ru_maxrss,
                                   child_peak_rss_bytes=int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
                                   child_user_cpu_seconds=usage.ru_utime, child_system_cpu_seconds=usage.ru_stime)
                        break
                    if termination_at is None and now - spawned >= spec["timeout_seconds"]:
                        row["timed_out"] = True
                        _terminate_group(proc.pid, signal.SIGTERM)
                        termination_at = now
                    elif termination_at is not None and now - termination_at >= spec["termination_grace_seconds"]:
                        _terminate_group(proc.pid, signal.SIGKILL)
                    time.sleep(0.005)
                # Successful leader exit must not leave active children writing
                # after measurement. Treat any remaining group as a failed run.
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    row["remaining_process_group_detected"] = True
                    _terminate_group(proc.pid, signal.SIGKILL)
            row["status"] = "timeout" if row["timed_out"] else ("signal" if row["signal"] else ("success" if row["returncode"] == 0 else "nonzero_exit"))
            if row["remaining_process_group_detected"]:
                row["status"] = "unreaped_process_group"
    except (OSError, ValueError) as err:
        row["status"] = "launch_or_io_failure"
        row["failure_details"].append(type(err).__name__ + ": " + str(err))
        if proc is not None and proc.returncode is None:
            _terminate_group(proc.pid, signal.SIGKILL)
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            row["returncode"] = proc.returncode
    try:
        row["inputs_after"] = _inputs(method, workspace)
        row["declared_inputs_unchanged"] = all(x["matches"] for x in row["inputs_after"])
    except (OSError, ValueError) as err:
        row["declared_inputs_unchanged"] = False
        row["failure_details"].append("post_run_input_check: " + str(err))
    if not row["declared_inputs_unchanged"]:
        row["status_before_input_check"] = row["status"]
        row["status"] = "input_changed_or_mismatched"
    entries, errors = _persisted_outputs(run_root)
    row["output_file_ledger"] = entries
    row["output_inspection_errors"] = errors
    row["persistent_output_logical_bytes"] = sum(x["logical_bytes"] for x in entries)
    row["persistent_output_allocated_bytes"] = sum(x["allocated_bytes"] for x in entries)
    row["method_artifact_logical_bytes"] = sum(x["logical_bytes"] for x in entries if x["path"].startswith("artifacts/"))
    row["charged_end_to_end_elapsed_seconds"] = time.monotonic() - started
    row["measurement_complete"] = not errors and not row["remaining_process_group_detected"] and row["declared_inputs_unchanged"]
    row["eligible_for_success_timing_summary"] = row["status"] == "success" and row["measurement_complete"]
    return row


def run_benchmark(spec, workspace, destination):
    workspace, destination = Path(workspace).resolve(), Path(destination).resolve()
    if not workspace.is_dir() or not destination.is_relative_to(workspace):
        raise ValueError("Existing workspace and new destination inside it required")
    spec = validate_spec(spec, workspace)
    if destination.exists():
        raise ValueError("Destination already exists; old runs are never overwritten")
    destination.mkdir(parents=True)
    benchmark_start = time.monotonic()
    fixed = {key: str(spec["threads"]) for key in FIXED_THREADS}
    fixed.update(PYTHONHASHSEED="0", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    environment = dict(os.environ)
    environment.update(fixed)
    order = planned_order(spec)
    report = {"schema": VERSION, "started_utc": datetime.now(timezone.utc).isoformat(), "spec": spec,
              "spec_sha256": digest(spec), "planned_order": order, "workspace": str(workspace),
              "fixed_thread_environment": fixed, "software": {"python": sys.version, "platform": platform.platform()},
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "harness_code_sha256": file_hash(__file__), "runs": [],
              "limits": {"repetitions_max": 5, "per_child_timeout_seconds": spec["timeout_seconds"],
                         "termination_grace_seconds": spec["termination_grace_seconds"],
                         "memory_limit_enforced": False, "disk_limit_enforced": False,
                         "filesystem_or_network_security_sandbox": False},
              "measurement_contract": {
                  "charged_elapsed": "per_run_from_before_input_hashing_and_directory_setup_through_spawn_wait_output_fsync_hashing_and_post_input_verification",
                  "peak_RSS": "POSIX_wait4_ru_maxrss_child_high_water_mark_not_sum_or_simultaneous_process_tree_peak",
                  "output_bytes": "regular_files_in_declared_run_directory_including_stdout_stderr; ledger_files_charged_separately; undeclared_writes_not_measurable",
                  "process_state": "fresh_process_and_output_directory_every_repetition; OS_page_cache_not_flushed; no_competing_job_exclusion_enforced",
                  "input_access": "all_supplied_input_files_hashed_before_and_after; commands_still_retain_normal_caller_permissions"},
              "method_code_pinning_completeness_verified": False,
              "per_method_comparative_isolation_verified": False,
              "confirmatory_study_ready": False, "paper_systems_result": False}
    ledger = destination / "benchmark.json"
    # Freeze planned order before collecting timings; update atomically per run.
    def checkpoint():
        payload = {k: v for k, v in report.items() if k != "sha256"}
        payload["sha256"] = digest(payload)
        temporary = destination / "benchmark.json.tmp"
        with temporary.open("wb") as handle:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(ledger)
        return payload
    checkpoint()
    methods = {m["method_id"]: m for m in spec["methods"]}
    for index, identity in enumerate(order):
        root = destination / f"run-{index:03d}-{identity['method_id']}-repeat-{identity['repetition']}"
        report["runs"].append(_run_one(methods[identity["method_id"]], identity, spec, workspace, root, environment))
        checkpoint()
    report["total_benchmark_elapsed_seconds_before_final_checkpoint"] = time.monotonic() - benchmark_start
    report["successful_complete_runs"] = sum(r["eligible_for_success_timing_summary"] for r in report["runs"])
    report["failed_or_incomplete_runs"] = len(report["runs"]) - report["successful_complete_runs"]
    result = checkpoint()
    # Separate ledger-cost receipt avoids a self-referential JSON byte count.
    receipt = {"benchmark_json_bytes": ledger.stat().st_size, "benchmark_json_sha256": file_hash(ledger),
               "run_output_logical_bytes": sum(r["persistent_output_logical_bytes"] for r in report["runs"]),
               "receipt_self_bytes_excluded": True,
               "total_elapsed_through_final_checkpoint_seconds": time.monotonic() - benchmark_start}
    (destination / "persistence_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark(json.loads(args.spec.read_text()), args.workspace, args.destination)
    print(json.dumps({"successful_complete_runs": result["successful_complete_runs"],
                      "failed_or_incomplete_runs": result["failed_or_incomplete_runs"],
                      "paper_systems_result": False, "destination": str(args.destination)}))


if __name__ == "__main__":
    main()
