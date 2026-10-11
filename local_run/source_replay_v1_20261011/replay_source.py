#!/usr/bin/env python3
"""Replay AskUbuntu adapter/preparation in private scratch; no model execution."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import tempfile
import time


class Refusal(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise Refusal(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def equal(left, right, code):
    require(canonical(left) == canonical(right), code)


def binding(path):
    h, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
            size += len(block)
    return {"sha256": h.hexdigest(), "bytes": size}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    def bad_number(_):
        raise Refusal("nonfinite_json_number")
    try:
        return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=bad_number)
    except (UnicodeError, json.JSONDecodeError):
        raise Refusal("invalid_utf8_or_json") from None


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


class Watch:
    def __init__(self):
        self.files = {}

    def add(self, path, declared=None):
        path = Path(path).resolve(strict=True)
        actual = binding(path)
        if path in self.files:
            equal(actual, self.files[path], "input_changed_between_stages")
        if declared is not None:
            require(all(actual.get(key) == value for key, value in declared.items()), "input_binding_mismatch")
        self.files[path] = actual
        return actual

    def verify(self):
        for path, wanted in self.files.items():
            equal(binding(path), wanted, "watched_input_changed_during_replay")


def within(path, roots):
    path = Path(path).resolve(strict=True)
    require(any(path.is_relative_to(Path(root).resolve()) for root in roots), "input_path_outside_allowed_roots")
    require(path.is_file(), "input_file_required")
    return path


def verify_package(folder, expected_sha, watch):
    manifest_path = folder / "package_manifest.json"
    watch.add(manifest_path, {"sha256": expected_sha})
    manifest = read_json(manifest_path)
    require(isinstance(manifest.get("files"), dict) and manifest["files"], "empty_package_manifest")
    for name, wanted in manifest["files"].items():
        require(Path(name).name == name and name not in (".", ".."), "unsafe_package_member")
        watch.add(within(folder / name, (folder,)), wanted)
    return manifest


def validate_lineage(report):
    require(report.get("schema") == "third-askubuntu-source-lineage-audit-v1"
            and report.get("passed") is True and report.get("stage") == "complete", "lineage_not_complete_and_passed")
    stages = report.get("completed_stages", {})
    require(set(stages) == {"input_inventory", "archive_members", "tag_transform", "adapter_bindings",
                            "physical_jsonl_transport", "population_bindings", "immutability"}, "lineage_stage_set_mismatch")
    require(stages["immutability"].get("all_watched_input_final_bytes_equal_initial") is True,
            "lineage_immutability_not_passed")
    require(stages["archive_members"].get("local_archive_member_byte_equality") is True
            and stages["tag_transform"].get("complete_expected_transformed_bytes_equal") is True
            and stages["physical_jsonl_transport"].get("complete_object_and_order_equality") is True,
            "lineage_required_check_not_passed")
    return stages


def prepare_gate(args, repo, workspace, watch):
    lineage_dir = repo / "local_run/source_lineage_audit_v1_20261011"
    package = verify_package(lineage_dir, args.lineage_package_sha256, watch)
    require({"audit_source.py", "expectations.json"}.issubset(package["files"]), "lineage_package_missing_driver")
    result_path = within(args.lineage_result, (workspace,))
    watch.add(result_path, {"sha256": args.lineage_result_sha256})
    report = read_json(result_path)
    stages = validate_lineage(report)
    inventory = stages["input_inventory"]
    for name, wanted in inventory.items():
        watch.add(within(name, (repo, workspace)), wanted)
    expected = read_json(lineage_dir / "expectations.json")
    required = [workspace / rel for rel in expected["files"]]
    required += [repo / rel for rel in expected["code"]]
    required += [workspace / "inputs/askubuntu/askubuntu.com.7z",
                 repo / "output/empirical_program/study_design.json",
                 lineage_dir / "audit_source.py", lineage_dir / "expectations.json"]
    require(all(str(path.resolve()) in inventory for path in required), "lineage_inventory_missing_required_input")
    for name in ("audit_source.py", "expectations.json"):
        equal(inventory[str((lineage_dir / name).resolve())], package["files"][name], "lineage_driver_package_binding_mismatch")
    # The older lineage report binds these two executed files, not its whole
    # package. Current full-package verification is recorded separately above.
    for name, value in stages.items():
        path = within(result_path.parent / (name + ".json"), (workspace,))
        watch.add(path)
        equal(read_json(path), value, "lineage_stage_receipt_mismatch")
    spec = importlib.util.spec_from_file_location("third_lineage_bound", lineage_dir / "audit_source.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return expected, helper, report


def metadata_binding(item, path, watch):
    require(Path(item["path"]).resolve() == path.resolve(), "transport_metadata_path_mismatch")
    watch.add(path, {key: item[key] for key in ("bytes", "sha256")})


def verify_view(view, population, repo, workspace, helper, watch):
    original_records, original_audit_path = population / "records.jsonl", population / "audit.json"
    records, audit_path, receipt_path = (view / name for name in ("records.jsonl", "audit.json", "receipt.json"))
    for path in (records, audit_path, receipt_path, original_records, original_audit_path):
        watch.add(path)
    original, audit, receipt = (read_json(path) for path in (original_audit_path, audit_path, receipt_path))
    helper.verify_seal(original)
    helper.verify_seal(audit)
    require(receipt.get("schema") == "third-lossless-jsonl-transport-receipt-v1"
            and receipt.get("dataset") == "askubuntu" and receipt.get("passed") is True
            and receipt.get("complete_decoded_row_equality_and_order_verified") is True
            and receipt.get("unchanged_frozen_load_prepared_records_passed") is True,
            "transport_receipt_not_passed")
    for key, path in (("original_records", original_records), ("original_audit_file", original_audit_path),
                      ("derived_records", records), ("derived_audit_file", audit_path)):
        metadata_binding(receipt[key], path, watch)
    require(receipt.get("original_audit_seal_sha256") == original["sha256"]
            and receipt.get("derived_audit_seal_sha256") == audit["sha256"], "transport_seal_binding_mismatch")
    converter = within(receipt["converter"]["path"], (repo, workspace))
    metadata_binding(receipt["converter"], converter, watch)
    reference_converter = repo / "local_run/engineering_v20261011/source_transport_v1.py"
    equal(watch.add(reference_converter), watch.files[converter.resolve()], "transport_converter_code_changed")
    lineage = audit.get("transport_lineage", {})
    require(lineage.get("schema") == "third-lossless-jsonl-transport-v1"
            and lineage.get("original_audit_seal_sha256") == original["sha256"], "transport_audit_lineage_mismatch")
    require(Path(lineage["original_preparation_directory"]).resolve() == population.resolve(),
            "transport_original_directory_mismatch")
    for key in ("original_records", "original_audit_file", "converter"):
        equal(lineage.get(key), receipt[key], "transport_lineage_receipt_mismatch")
    core = {k: v for k, v in audit.items() if k not in ("sha256", "transport_lineage", "prepared_records_file_sha256")}
    equal(core, {k: v for k, v in original.items() if k not in ("sha256", "prepared_records_file_sha256")},
          "transport_changed_original_audit_semantics")
    require(audit.get("prepared_records_file_sha256") == watch.files[records.resolve()]["sha256"], "transport_record_hash_mismatch")
    bridge = helper.ascii_transport(original_records, records)
    return original, audit, {"rows": bridge["rows"], "complete_object_order_and_ascii_bytes_rechecked": True,
                              "original_audit_seal": original["sha256"], "derived_audit_seal": audit["sha256"]}


def compare_adapter(original, rebuilt, supplied, replayed):
    # All adapter outputs, including the SQLite identity index, are exact-byte
    # checks here. A runtime-dependent SQLite difference refuses this version;
    # it is never quietly downgraded to a weaker comparison.
    equal(sorted(supplied), sorted(replayed), "adapter_audit_key_set_mismatch")
    equal(set_as_list(supplied["outputs"]), set_as_list(replayed["outputs"]), "adapter_output_inventory_mismatch")
    for key in set(supplied) | set(replayed):
        if key != "outputs":
            equal(supplied.get(key), replayed.get(key), "adapter_metadata_mismatch:" + key)
    for name, expected in supplied["outputs"].items():
        require(Path(name).name == name, "unsafe_adapter_output")
        equal(binding(original / name), expected, "original_adapter_output_binding_mismatch")
        equal(binding(rebuilt / name), expected, "replayed_adapter_output_bytes_mismatch:" + name)


def set_as_list(value):
    return sorted(value)


def compare_population(rebuilt, panel_ids, replay_audit, original_records, original_audit, original_ids, helper):
    sentinel, count = object(), 0
    for wanted, actual in itertools.zip_longest(rebuilt, helper.jsonl(original_records), fillvalue=sentinel):
        require(wanted is not sentinel and actual is not sentinel, "prepared_population_row_count_mismatch")
        equal(wanted, actual, "prepared_full_row_or_order_mismatch")
        count += 1
    equal(panel_ids, original_ids, "panel_ids_mismatch")
    # This checks every key produced by prepare_population, including all
    # retained indices, panels, split, fixed guards and unchanged pending gates.
    for key, value in replay_audit.items():
        require(key in original_audit, "original_population_audit_field_missing")
        equal(value, original_audit[key], "population_audit_mismatch:" + key)
    return {"rows": count, "all_prepared_objects_and_order_equal": True,
            "panel_ids_equal": True, "all_rebuilt_population_audit_fields_equal": True,
            "panel_id_sha256": hashlib.sha256(canonical(panel_ids)).hexdigest()}


def run(args, report):
    repo, workspace, out, view = (Path(getattr(args, k)).resolve() for k in ("repo", "workspace", "out", "view"))
    require(os.environ.get("SLURM_JOB_ID"), "slurm_allocation_required")
    require(platform.system() == "Linux" and socket.gethostname().split(".")[:2] != ["csis", "mn1"], "linux_compute_node_required")
    require(out.is_relative_to(workspace / "qualification") and not out.exists(), "new_private_qualification_output_required")
    require(view.is_dir() and view.is_relative_to(workspace), "explicit_existing_private_ascii_view_required")
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=False)
    (out / ".gitignore").write_text("*\n")
    report["out"] = str(out)
    watch = Watch()
    verify_package(Path(__file__).resolve().parent, args.replay_package_sha256, watch)
    frozen = read_json(Path(__file__).with_name("frozen_code.json"))
    for relative, wanted in frozen["files"].items():
        watch.add(within(repo / relative, (repo,)), wanted)
    def stage(name, value):
        write_json(out / (name + ".json"), value)
        report["completed_stages"][name] = value
    report["stage"] = "lineage_gate"
    expected, helper, lineage_report = prepare_gate(args, repo, workspace, watch)
    stage("lineage_gate", {"lineage_result_sha256": args.lineage_result_sha256,
                           "lineage_package_sha256": args.lineage_package_sha256,
                           "all_recorded_inputs_rebound": True, "prior_stage_reports_equal": True})
    population = workspace / "prepared/askubuntu/a02/population"
    adapter = workspace / "prepared/askubuntu/a02/adapter"
    panel_path, adapter_audit_path = population / "panel_ids.json", adapter / "audit.json"
    for path in (panel_path, adapter_audit_path):
        watch.add(within(path, (workspace,)))
    supplied = read_json(adapter_audit_path)
    for name, value in supplied["outputs"].items():
        require(Path(name).name == name, "unsafe_adapter_output")
        watch.add(within(adapter / name, (workspace,)), value)
    report["stage"] = "prepared_ascii_view"
    for name in ("records.jsonl", "audit.json", "receipt.json"):
        within(view / name, (workspace,))
    original_audit, view_audit, transport = verify_view(view, population, repo, workspace, helper, watch)
    require(transport["rows"] == expected["population_rows"], "transport_population_count_mismatch")
    stage("prepared_ascii_view", transport)
    # Frozen modules are imported only after their lineage/code bindings pass.
    sys.path.insert(0, str(repo / "empirical_execution"))
    from phase3 import panels
    from phase4 import adapters, embeddings
    report["runtime"] = {"python": platform.python_version(), "platform": platform.platform(),
                         "sqlite": adapters.sqlite3.sqlite_version,
                         "numpy": panels.np.__version__, "idna": adapters.idna.__version__}
    for module in (panels, adapters, embeddings):
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(repo), "unexpected_frozen_module_path")
        watch.add(path)
    design_path = repo / "output/empirical_program/study_design.json"
    design = read_json(design_path)
    require(original_audit.get("study_design_sha256") == watch.files[design_path.resolve()]["sha256"], "design_binding_mismatch")
    report["stage"] = "frozen_adapter_replay"
    with tempfile.TemporaryDirectory(prefix="private-replay-", dir=out) as scratch:
        rebuilt_dir = Path(scratch) / "adapter"
        settings = supplied["settings"]
        replayed = adapters.adapt_stack(workspace / "inputs/askubuntu/Posts.angle.xml",
            workspace / "inputs/askubuntu/PostLinks.xml", rebuilt_dir, dataset_id="askubuntu",
            archive_sha256=settings["archive_sha256_declared_unverified"],
            archive_release=settings["archive_release_declared"], evidence_scope=settings["evidence_scope"])
        compare_adapter(adapter, rebuilt_dir, supplied, replayed)
        verified = adapters.verify_adapter_output(rebuilt_dir)
        require(verified["verified_rows"] == expected["adapter_rows"], "replayed_adapter_count_mismatch")
        stage("frozen_adapter_replay", {"rows": verified["verified_rows"], "all_output_bytes_equal": True,
                                       "all_settings_counts_and_audit_metadata_equal": True,
                                       "replayed_output_inventory": replayed["outputs"]})
        # This is the sole transport boundary replacing the defective caller's
        # str.splitlines reader. The frozen parser and preparation never change.
        raw_rows = list(helper.jsonl(rebuilt_dir / "records.jsonl"))
        links = read_json(rebuilt_dir / "duplicate_links.json")
    report["stage"] = "frozen_preparation_replay"
    prepared, ids, replay_audit = panels.prepare_population(raw_rows, "askubuntu", design,
        duplicate_links=links, target=10000, allow_preview_sources=False)
    del raw_rows, links
    gc.collect()
    compared = compare_population(prepared, ids, replay_audit, population / "records.jsonl",
                                  original_audit, read_json(panel_path), helper)
    require(compared["rows"] == expected["population_rows"], "prepared_count_mismatch")
    # The existing derived view must also pass the unchanged frozen loader now.
    loaded, loaded_audit = embeddings.load_prepared_records(view / "records.jsonl", view / "audit.json")
    require(len(loaded) == len(prepared), "frozen_loader_row_count_mismatch")
    for left, right in zip(loaded, prepared):
        equal(left, right, "frozen_loader_rebuilt_population_mismatch")
    equal(loaded_audit, view_audit, "frozen_loader_audit_mismatch")
    del loaded, prepared, ids
    gc.collect()
    stage("frozen_preparation_replay", {**compared, "unchanged_frozen_loader_passed": True})
    report["stage"] = "final_immutability"
    watch.verify()
    stage("final_immutability", {"all_watched_files_equal_initial_bytes": True,
                                "bindings": {str(path): value for path, value in watch.files.items()}})
    report["passed"], report["stage"] = True, "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "workspace", "view", "out", "lineage-result", "lineage-result-sha256",
                "lineage-package-sha256", "replay-package-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    report = {"schema": "third-askubuntu-source-replay-v1", "passed": False, "stage": "preflight",
              "evidence_role": "prospective_frozen_adapter_and_preparation_replay",
              "completed_stages": {}, "source_authenticity_verified": False,
              "primary_input_acceptance": False, "full_population_cache_acceptance": False,
              "confirmatory_study_ready": False, "model_execution_performed": False,
              "deferred": ["external_archive_authenticity_completeness_and_attribution",
                           "full_frozen_CPU_E5_encoding_and_fresh_process_cache_replay",
                           "semantic_and_human_or_prospectively_versioned_study_gates"]}
    try:
        run(args, report)
    except Exception as error:
        report["error"] = str(error) if isinstance(error, Refusal) else type(error).__name__
    report["seconds"] = time.monotonic() - start
    report["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if "out" in report:
        write_json(Path(report["out"]) / "result.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "stage", "seconds")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
