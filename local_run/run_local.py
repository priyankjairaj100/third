#!/usr/bin/env python3
"""Thin local launcher for frozen CCU entry points. Never grants input approval."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO / "local_workspace"
SCHEMA = "ccu-local-launch-1"
# route: script, subcommand, required input arguments, new output, fixed arguments
ROUTES = {
    "registry": ("phase6/recipes.py", None, (), "out", {}),
    "calibration-dossier": ("phase8/dossiers.py", "calibration", ("source-spec", "encoder"), "out", {}),
    "calibration-pack": ("phase7/prepare_calibration.py", None, ("dataset", "encoder", "dossier"), "out", {}),
    "candidate": ("phase8/dossiers.py", "candidate", ("config",), "out", {}),
    "prepare-resource": ("phase9/study_assembly.py", "prepare-resource", ("candidate", "attachment"), None, {}),
    "measure-resource": ("phase9/resource_measurement.py", None, ("request",), "out", {}),
    "qualify-resource": ("phase9/resource_profiles.py", None, ("request",), "out", {}),
    "prepare-review": ("phase9/study_assembly.py", "prepare-review", ("candidate", "attachment"), None, {}),
    "finalize-review": ("phase9/study_assembly.py", "finalize-review", ("candidate", "review"), None, {}),
    "assemble": ("phase9/study_assembly.py", "assemble", ("config",), "out", {}),
    "execute": ("phase9/study_assembly.py", "execute", ("plan",), "out", {}),
    "engineering-dispatch": ("phase7/run_dispatch.py", None, ("registry", "jobs", "bundles", "policy"), "out", {"mode": "natural_text_engineering"}),
}
SCALARS = {"encoder", "dataset"}
ROLES = {"preparation", "engineering", "development", "confirmatory", "analysis"}
FIXED_STAGE = {
    "prepare-resource": "phase9_resource_stage",
    "prepare-review": "phase9_review_stage",
    "finalize-review": "phase9_final_stage",
}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def private_path(value, *, exists=False):
    if not isinstance(value, str) or not value:
        raise ValueError("Paths must be explicit nonempty strings")
    path = Path(value)
    path = (path if path.is_absolute() else REPO / path).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()) or path == WORKSPACE.resolve():
        raise ValueError("All run inputs and outputs must remain beneath local_workspace")
    if exists and not path.is_file():
        raise ValueError("Required input file is missing: " + str(path))
    return path


def git_read(*args):
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError("Git inspection failed: " + " ".join(args))
    return result.stdout.strip()


def init_workspace():
    WORKSPACE.mkdir(exist_ok=True)
    guard = WORKSPACE / ".gitignore"
    if guard.exists() and guard.read_text() != "*\n":
        raise ValueError("Existing workspace guard differs; inspect it before continuing")
    if not guard.exists():
        guard.write_text("*\n")
    for name in ("inputs", "models", "prepared", "calibration", "development", "candidates", "plans", "runs", "reviews", "reports", "runner_logs", "launch_specs"):
        (WORKSPACE / name).mkdir(exist_ok=True)
    return {"workspace": str(WORKSPACE), "created_missing_directories_only": True,
            "scientific_inputs_created": False, "primary_execution_allowed": False}


def dependency_binding(entry):
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "pointer", "equals"}:
        raise ValueError("Each dependency needs exactly path, sha256, pointer, and equals")
    path = private_path(entry["path"], exists=True)
    actual = digest(path)
    if entry["sha256"] != actual:
        raise ValueError("Dependency hash differs: " + str(path))
    value = strict_json(path.read_text())
    pointer = entry["pointer"]
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Dependency pointer must be an explicit JSON pointer")
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    if type(value) is not type(entry["equals"]) or value != entry["equals"]:
        raise ValueError("Dependency condition failed: " + str(path) + pointer)
    return {"path": str(path), "sha256": actual, "pointer": pointer, "equals": value}


def plan(spec_path):
    spec_path = private_path(str(spec_path), exists=True)
    spec = strict_json(spec_path.read_text())
    if not isinstance(spec, dict) or set(spec) != {"schema", "run_id", "evidence_role", "route", "arguments", "dependencies"}:
        raise ValueError("Launch specification has unknown or missing fields")
    if spec["schema"] != SCHEMA:
        raise ValueError("Unknown launch specification schema")
    if not isinstance(spec["run_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", spec["run_id"]):
        raise ValueError("run_id must be a short file-safe unique identifier")
    role = spec["evidence_role"]
    if role not in ROLES:
        raise ValueError("Declare preparation, engineering, development, confirmatory, or analysis")
    route = spec["route"]
    if route not in ROUTES:
        raise ValueError("Unknown allowlisted route")
    if route == "engineering-dispatch" and role != "engineering":
        raise ValueError("Engineering dispatch cannot produce confirmatory evidence")
    if route in {"measure-resource", "qualify-resource"} and role != "development":
        raise ValueError("Resource collection/qualification must carry the development role")
    if route == "execute" and role not in {"confirmatory", "analysis"}:
        raise ValueError("Frozen Phase 9 execution needs confirmatory or analysis accounting")
    script_name, command, required, output_arg, fixed = ROUTES[route]
    arguments = spec["arguments"]
    expected = set(required) | ({output_arg} if output_arg else set())
    if not isinstance(arguments, dict) or set(arguments) != expected:
        raise ValueError("This route requires exactly these arguments: " + ", ".join(sorted(expected)))
    paths = {}; input_hashes = {}
    for key in required:
        if key in SCALARS:
            if not isinstance(arguments[key], str) or not arguments[key]:
                raise ValueError("Explicit scalar value required: " + key)
            paths[key] = arguments[key]
        else:
            paths[key] = str(private_path(arguments[key], exists=True))
            input_hashes[paths[key]] = digest(paths[key])
    if output_arg:
        output = private_path(arguments[output_arg])
        if output.exists() or output.is_symlink():
            raise ValueError("Preserve existing output; choose a new run: " + str(output))
        paths[output_arg] = str(output)
    else:
        output = Path(paths["candidate"]).parent / FIXED_STAGE[route]
        if output.exists() or output.is_symlink():
            raise ValueError("Candidate stage already exists; preserve its receipt and build a new candidate")
    if not isinstance(spec["dependencies"], list):
        raise ValueError("Explicit dependencies list required")
    dependencies = [dependency_binding(item) for item in spec["dependencies"]]
    script = REPO / "empirical_execution" / script_name
    if not script.is_file():
        raise ValueError("Frozen entry point is missing")
    argv = [sys.executable, str(script)]
    if command:
        argv.append(command)
    if route == "registry":
        argv.append(paths["out"])
    else:
        for key, value in paths.items():
            argv.extend(["--" + key, value])
        for key, value in fixed.items():
            argv.extend(["--" + key, value])
    log_dir = WORKSPACE / "runner_logs" / spec["run_id"]
    if log_dir.exists() or log_dir.is_symlink():
        raise ValueError("This run_id already has a log; never overwrite or resume it")
    return {"schema": SCHEMA, "run_id": spec["run_id"], "evidence_role": role,
            "route": route, "argv": argv, "cwd": str(REPO), "output_directory": str(output),
            "log_directory": str(log_dir), "specification": str(spec_path), "specification_sha256": digest(spec_path),
            "top_level_input_sha256": input_hashes, "dependencies": dependencies,
            "entrypoint_sha256": digest(script), "launcher_sha256": digest(__file__),
            "role_is_user_declaration_not_scientific_approval": True,
            "nested_inputs_verified_by_frozen_executor": True}


def inspect_directory(path):
    directory = private_path(str(path))
    result = {"directory": str(directory), "exists": directory.is_dir(), "reports": {}}
    keys = ("schema", "status", "planned_jobs", "selected_jobs", "missing_scope_jobs", "observed_jobs", "attempted_jobs",
            "every_planned_job_retained", "primary_outputs_accepted", "inputs_and_sources_unchanged", "status_counts",
            "primary_execution_allowed", "machine_acceptance_passed", "selection_pack_created", "execution_allowed",
            "primary_inputs_accepted", "returncode", "unchanged_launcher_inputs", "evidence_role")
    for filename in ("receipt.json", "summary.json", "qualification.json", "measurement.json", "result.json", "registry.json"):
        candidate = directory / filename
        if candidate.is_file():
            value = strict_json(candidate.read_text())
            result["reports"][filename] = {key: value[key] for key in keys if key in value}
            result["reports"][filename]["sha256"] = digest(candidate)
    result["interpretation"] = "Process completion and completed_dispatch do not mean scientific success. Read the full receipts and final ledger."
    return result


def run(spec_path):
    conf = plan(spec_path)
    frozen_roots = ["empirical_execution/phase" + str(i) for i in range(3, 11)]
    dirty = git_read("status", "--porcelain", "--", *frozen_roots, "output/empirical_program")
    if dirty:
        raise ValueError("Frozen source or design paths have local changes; inspect and version the amendment first")
    # Preserve logs before invoking code. The launcher never executes a shell.
    log_dir = Path(conf["log_directory"])
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / ".gitignore").write_text("*\n")
    conf.update(git_head=git_read("rev-parse", "HEAD"), python_version=sys.version,
                platform=sys.platform, interpreter_sha256=digest(sys.executable), start_time_unix=time.time())
    write_new(log_dir / "invocation.json", conf)
    began = time.monotonic()
    returncode = None
    interruption = None
    try:
        with (log_dir / "stdout.txt").open("x") as stdout, (log_dir / "stderr.txt").open("x") as stderr:
            process = subprocess.run(conf["argv"], cwd=REPO, stdout=stdout, stderr=stderr, check=False)
            returncode = process.returncode
    except (OSError, KeyboardInterrupt) as error:
        interruption = type(error).__name__
    checks = dict(conf["top_level_input_sha256"])
    checks[conf["specification"]] = conf["specification_sha256"]
    checks[conf["argv"][1]] = conf["entrypoint_sha256"]
    checks[str(Path(__file__).resolve())] = conf["launcher_sha256"]
    checks.update({dep["path"]: dep["sha256"] for dep in conf["dependencies"]})
    unchanged = all(Path(path).is_file() and digest(path) == sha for path, sha in checks.items())
    try:
        output_report = inspect_directory(conf["output_directory"])
    except (ValueError, OSError, KeyError, TypeError) as error:
        output_report = {"directory": conf["output_directory"], "inspection_failed": type(error).__name__,
                         "instruction": "Inspect raw output files and retained subprocess logs; no acceptance is inferred."}
    report = {"schema": SCHEMA, "run_id": conf["run_id"], "route": conf["route"],
              "evidence_role": conf["evidence_role"], "returncode": returncode,
              "status": "process_completed" if returncode == 0 and unchanged else "process_failed_or_inputs_changed",
              "unchanged_launcher_inputs": unchanged, "interruption": interruption,
              "wall_seconds": time.monotonic() - began,
              "scientific_acceptance": "Only frozen executor receipts and their final bound ledger determine acceptance.",
              "output": output_report}
    write_new(log_dir / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("routes")
    for command in ("plan", "run"):
        p = sub.add_parser(command); p.add_argument("--spec", required=True)
    p = sub.add_parser("inspect"); p.add_argument("--directory", required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            result = init_workspace()
        elif args.command == "routes":
            result = {key: {"script": value[0], "subcommand": value[1],
                            "arguments": list(value[2]) + ([value[3]] if value[3] else []),
                            "fixed_arguments": value[4]} for key, value in ROUTES.items()}
        elif args.command == "plan":
            result = plan(args.spec)
        elif args.command == "run":
            result = run(args.spec)
        else:
            result = inspect_directory(args.directory)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get("status") == "process_failed_or_inputs_changed" else 0
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.exit(2, "BLOCKED: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
