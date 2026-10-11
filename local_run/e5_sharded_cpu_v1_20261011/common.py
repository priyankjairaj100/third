"""Small shared binding layer for the prospective CPU shard contract."""
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys

BUDGET_PACKAGE_SHA = "89e11b8bbbf725f70dc61245e02ae32233b5362d90ca3062926d8d297fe1adb2"
prior = Path(__file__).resolve().parent.parent / "e5_forward_budget_v1_20261011"
raw = (prior / "package_manifest.json").read_bytes()
if hashlib.sha256(raw).hexdigest() != BUDGET_PACKAGE_SHA:
    raise RuntimeError("sealed_budget_package_changed")
for name, expected in json.loads(raw)["files"].items():
    if Path(name).name != name:
        raise RuntimeError("invalid_sealed_package_member")
    data = (prior / name).read_bytes()
    if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise RuntimeError("sealed_budget_member_changed")
spec = importlib.util.spec_from_file_location("shard_budget_helpers", prior / "budget.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
c = b.c
require, binding, write_json = c.require, c.binding, c.write_json


def add_arguments(parser):
    for name in ("repo", "workspace", "view", "replay-result", "census-dir", "budget-dir", "package-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--qualification-sha256", help="Independently reviewed qualification result SHA; required for full work")


def common_arguments(args):
    answer = []
    for name in ("repo", "workspace", "view", "replay-result", "census-dir", "budget-dir", "package-sha256"):
        answer += ["--" + name, str(getattr(args, name.replace("-", "_")))]
    return answer


def seal(value):
    require("sha256" not in value, "already_sealed_object")
    return {**value, "sha256": hashlib.sha256(c.canonical(value)).hexdigest()}


def verify_seal(value):
    require(value.get("sha256") == hashlib.sha256(c.canonical({k: v for k, v in value.items() if k != "sha256"})).hexdigest(), "invalid_seal")


def new_output(ctx, path):
    path = Path(path).resolve()
    require(path.is_relative_to(ctx["workspace"] / "qualification") and not path.exists(), "new_private_output_required")
    os.umask(0o077)
    path.mkdir(parents=True, exist_ok=False)
    return path


def atomic_completion(path, value):
    path = Path(path)
    require(not path.exists(), "completion_already_exists")
    temporary = path.with_name(path.name + ".pending")
    write_json(temporary, value)
    # Same-directory hard-link publication is atomic and refuses an existing
    # destination; os.replace would allow a last-moment overwrite race.
    os.link(temporary, path)
    temporary.unlink()
    descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def cpu_identity():
    first = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
    fields = dict(line.split(":", 1) for line in first.splitlines() if ":" in line)
    fields = {key.strip(): value.strip() for key, value in fields.items()}
    return {key: fields.get(key) for key in ("vendor_id", "cpu family", "model", "model name", "stepping", "microcode", "flags")}


def context(args):
    require(platform.system() == "Linux" and os.environ.get("SLURM_JOB_ID"), "linux_slurm_allocation_required")
    require(socket.gethostname().split(".")[:2] != ["csis", "mn1"], "login_node_prohibited")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ctx = {name: Path(getattr(args, name)).resolve() for name in ("repo", "workspace", "view", "census_dir", "budget_dir")}
    folder, watch = Path(__file__).resolve().parent, c.Watch()
    ctx.update({"folder": folder, "watch": watch, "package_sha256": args.package_sha256,
                "qualification_sha256": getattr(args, "qualification_sha256", None)})
    c.verify_package(folder, args.package_sha256, watch)
    c.verify_package(prior, BUDGET_PACKAGE_SHA, watch)
    c.verify_package(b.helper_path.parent, b.HELPER_PACKAGE_SHA, watch)
    pins = c.read_json(folder / "expectations.json")
    ctx["pins"] = pins
    for name, expected in c.read_json(folder / "frozen_code.json")["files"].items():
        watch.add(c.inside(ctx["repo"] / name, ctx["repo"]), expected)
    replay_path = c.inside(args.replay_result, ctx["workspace"])
    watch.add(replay_path, {"sha256": pins["replay_result_sha256"]})
    stages = c.validate_replay(c.read_json(replay_path))
    for key, dirname in (("census_files", "census_dir"), ("budget_files", "budget_dir")):
        for name, expected in pins[key].items():
            watch.add(c.inside(ctx[dirname] / name, ctx["workspace"]), expected)
    census = c.read_json(ctx["census_dir"] / "result.json")
    aggregate = c.read_json(ctx["census_dir"] / "aggregate.json")
    b.validate_census(census, aggregate, pins)
    ctx["aggregate"] = aggregate
    budget = c.read_json(ctx["budget_dir"] / "result.json")
    require(budget.get("passed") is True and budget.get("stage") == "complete"
            and budget.get("initial_final_input_code_assets_equal") is True
            and budget.get("exact_vector_byte_comparisons") == {"repeat": True, "reverse": True}
            and budget.get("census_result_sha256") == pins["census_files"]["result.json"]["sha256"], "passed_exact_budget_required")
    ctx["projection"] = c.read_json(ctx["budget_dir"] / "budget_projection.json")
    require(budget["artifacts"]["budget_projection.json"] == pins["budget_files"]["budget_projection.json"], "budget_projection_receipt_mismatch")
    source_bound = c.read_json(ctx["census_dir"] / "input_bindings.json")["files"]
    replay_bound = stages["final_immutability"]["bindings"]
    for name, expected in pins["view"].items():
        path = c.inside(ctx["view"] / name, ctx["workspace"])
        require(source_bound.get(str(path)) == expected and replay_bound.get(str(path)) == expected, "source_view_binding_mismatch")
        watch.add(path, expected)
    sys.path.insert(0, str(ctx["repo"] / "empirical_execution"))
    from phase4 import embeddings as e
    from phase3 import calibration as cal
    import numpy as np
    require(platform.python_version() == pins["python"] and e.installed_versions() == pins["library_versions"], "pinned_runtime_mismatch")
    assets_path = c.inside(ctx["workspace"] / "models/e5/assets.json", ctx["workspace"])
    watch.add(assets_path, pins["assets_file"])
    assets = c.read_json(assets_path)
    require(assets.get("sha256") == pins["asset_seal_sha256"] and assets.get("encoder_id") == e.E5
            and assets.get("encoder_revision") == pins["revision"] and assets.get("tokenizer_revision") == pins["revision"], "asset_identity_mismatch")
    model_dir, tokenizer_dir = ctx["workspace"] / "models/e5/model", ctx["workspace"] / "models/e5/tokenizer"
    e.verify_assets(assets, model_dir, tokenizer_dir, verify_runtime=True)
    require([x for x in assets["model_files"] if x["path"] == "model.safetensors"] == [pins["model_weight"]], "model_weight_binding_mismatch")
    audit = c.read_json(ctx["view"] / "audit.json"); cal._verify(audit)
    require(audit["dataset_id"] == "askubuntu" and audit["sha256"] == stages["prepared_ascii_view"]["derived_audit_seal"]
            and audit["prepared_rows_sha256"] == aggregate["full_prepared_rows_sha256"] and audit["text_binding"] == aggregate["text_binding"], "source_audit_mismatch")
    runtime = {"python": platform.python_version(), "libraries": e.installed_versions(), "platform": platform.platform(),
        "machine": platform.machine(), "cpu": cpu_identity(), "threads": 1, "interop_threads": 1,
        "device": "cpu", "dtype": "float32", "attention": "eager", "mkldnn": False,
        "batch_size": 8, "padding": 261, "deterministic_algorithms": True, "seed": 0}
    identity = {"runtime": runtime, "package_manifest_sha256": args.package_sha256,
        "expectations": binding(folder / "expectations.json"), "frozen_code": binding(folder / "frozen_code.json")}
    ctx.update({"e": e, "cal": cal, "np": np, "assets": assets, "assets_path": assets_path,
        "model_dir": model_dir, "tokenizer_dir": tokenizer_dir, "audit": audit, "runtime": runtime,
        "fingerprint": hashlib.sha256(c.canonical(identity)).hexdigest()})
    return ctx


def verify_final(ctx):
    ctx["e"].verify_assets(ctx["assets"], ctx["model_dir"], ctx["tokenizer_dir"], verify_runtime=True)
    require(ctx["e"].installed_versions() == ctx["pins"]["library_versions"] and cpu_identity() == ctx["runtime"]["cpu"], "runtime_changed")
    ctx["watch"].verify()


def validate_plan(plan, fingerprint, full_rows):
    verify_seal(plan)
    require(plan.get("schema") == "third-e5-cpu-shard-plan-v1" and plan.get("fingerprint") == fingerprint, "plan_identity_mismatch")
    indices, shards = plan["frame_row_indices"], plan["shards"]
    require(indices and all(type(i) is int and 0 <= i < full_rows for i in indices)
            and indices == sorted(set(indices)), "invalid_frame_row_indices")
    expected_count = 32 if plan["kind"] == "full" else 4 if plan["kind"] == "qualification" else None
    require(expected_count is not None and len(shards) == expected_count, "invalid_plan_kind_or_shards")
    if plan["kind"] == "full":
        require(indices == list(range(full_rows)), "full_plan_incomplete_population")
    else:
        require(len(indices) == 19, "qualification_requires_19_rows")
    start = 0
    for index, shard in enumerate(shards):
        require(type(shard["index"]) is int and type(shard["start"]) is int
                and shard["index"] == index and shard["start"] == start and type(shard["stop"]) is int
                and start < shard["stop"] <= len(indices), "shard_gap_overlap_empty_or_out_of_order")
        require(math.isfinite(shard["modeled_forward_seconds"]) and shard["modeled_forward_seconds"] > 0, "invalid_shard_weight")
        start = shard["stop"]
    require(start == len(indices), "shards_do_not_cover_frame")


def load_plan(ctx, path):
    path = c.inside(path, ctx["workspace"])
    ctx["watch"].add(path)
    plan = c.read_json(path)
    validate_plan(plan, ctx["fingerprint"], ctx["pins"]["rows"])
    require(plan.get("full_source_frame_sha256") == ctx["aggregate"]["full_prepared_rows_sha256"]
            and plan.get("census_ledger") == ctx["pins"]["census_files"]["row_token_chunk_ledger.private.jsonl"], "plan_source_or_census_mismatch")
    if plan["kind"] == "full":
        require(plan.get("qualification_result", {}).get("sha256") == ctx.get("qualification_sha256")
                and ctx.get("qualification_sha256") is not None, "full_plan_qualification_pin_mismatch")
    return plan


def rows_for_shard(ctx, plan, index):
    require(type(index) is int and 0 <= index < len(plan["shards"]), "invalid_shard_index")
    shard = plan["shards"][index]
    return rows_for_indices(ctx, plan["frame_row_indices"][shard["start"]:shard["stop"]])


def rows_for_indices(ctx, indices):
    wanted = set(indices)
    require(len(wanted) == len(indices), "repeated_selected_index")
    expected = {}
    for index, row in enumerate(c.physical_rows(ctx["census_dir"] / "row_token_chunk_ledger.private.jsonl")):
        if index in wanted:
            require(row["row_index"] == index, "census_row_index_mismatch")
            expected[index] = row
    require(set(expected) == wanted, "missing_census_rows")
    whole, text_digest, ids = c.ListDigest(), c.ListDigest(), c.ListDigest()
    seen, parts, selected = set(), Counter(), {}
    for index, row in enumerate(c.physical_rows(ctx["view"] / "records.jsonl")):
        rid, text = row.get("record_id"), row.get("text")
        require(isinstance(rid, str) and rid and rid not in seen and isinstance(text, str) and text, "invalid_source_row")
        seen.add(rid); whole.add(row); text_digest.add({"record_id": rid, "text": text}); ids.add(rid)
        parts[row["partition"]] += 1
        if index in wanted:
            entry = expected[index]
            require(rid == entry["record_id"] and row["partition"] == entry["partition"]
                    and hashlib.sha256(text.encode()).hexdigest() == entry["text_sha256"], "selected_source_census_mismatch")
            selected[index] = row
    aggregate = ctx["aggregate"]
    require(len(seen) == aggregate["totals"]["rows"] and dict(parts) == aggregate["partition_counts"]
            and whole.finish() == aggregate["full_prepared_rows_sha256"] and text_digest.finish() == aggregate["text_binding"]
            and ids.finish() == aggregate["ordered_record_ids_sha256"], "complete_source_frame_mismatch")
    require(set(selected) == wanted, "missing_source_rows")
    return [selected[i] for i in indices], [expected[i] for i in indices]


def require_qualification(ctx, path):
    require(path is not None, "passed_qualification_result_required")
    path = c.inside(path, ctx["workspace"])
    expected_sha = ctx.get("qualification_sha256")
    require(isinstance(expected_sha, str) and len(expected_sha) == 64, "independently_reviewed_qualification_SHA_required")
    ctx["watch"].add(path, {"sha256": expected_sha})
    report = c.read_json(path)
    verify_seal(report)
    require(report.get("schema") == "third-e5-sharded-cpu-qualification-v1" and report.get("passed") is True
            and report.get("fingerprint") == ctx["fingerprint"]
            and report.get("golden_all_vector_bytes_equal") is True
            and report.get("golden_all_chunk_log_rows_equal") is True
            and report.get("all_shards_fresh_process_replayed") is True, "qualification_gate_not_passed_for_current_pins")
    required = {"plan.json", "assembled/manifest.json", "assembled/vectors.npy", "assembled/chunk_log.jsonl",
        "golden/cache_manifest.json", "golden/vectors.npy", "golden/chunk_log.jsonl",
        "engineering_subset/records.jsonl", "engineering_subset/audit.json"}
    required.update(f"{kind}-{i}/{name}" for kind in ("encode", "replay") for i in range(4)
                    for name in ("completion.json", "vectors.npy", "chunk_log.jsonl"))
    required.add("golden_process/process.json")
    required.update(f"{kind}-{i}-process/process.json" for kind in ("encode", "replay") for i in range(4))
    require(required <= set(report.get("evidence_files", {})), "qualification_evidence_inventory_incomplete")
    require(all(type(report.get(key)) is int and report[key] == value for key, value in {
        "qualification_rows": 19, "chunks_per_pass": 149, "logical_passes": 3,
        "document_encodings": 57, "chunks_processed": 447,
        "forward_batches_per_pass": 28, "model_forward_batches": 84}.items())
        and report.get("phase6_acceptance_import_preflight_passed") is True, "qualification_work_counts_or_preflight_missing")
    require(all(report.get(key) is False for key in ("source_authenticity_verified", "primary_input_acceptance",
        "original_monolithic_full_writer_executed", "original_monolithic_full_replay_executed",
        "full_population_cache_acceptance", "confirmatory_study_ready")), "qualification_claim_boundaries_changed")
    for name, expected in report["evidence_files"].items():
        candidate = c.inside(path.parent / name, path.parent)
        ctx["watch"].add(candidate, expected)
    process_paths = [("golden_process/process.json", None, None)] + [(f"{kind}-{i}-process/process.json", kind, i)
        for kind in ("encode", "replay") for i in range(4)]
    for name, kind, index in process_paths:
        process = c.read_json(path.parent / name)
        require(type(process.get("returncode")) is int and process["returncode"] == 0
                and type(process.get("child_pid")) is int and type(process.get("launcher_pid")) is int
                and process["child_pid"] > 0 and process["child_pid"] != process["launcher_pid"], "qualification_fresh_child_receipt_missing")
        command = process.get("command", [])
        require(isinstance(command, list) and len(command) > 2 and command[0] == sys.executable, "qualification_child_interpreter_mismatch")
        def flag(name):
            require(command.count(name) == 1 and command.index(name) + 1 < len(command), "qualification_child_argument_missing")
            return command[command.index(name) + 1]
        if kind is None:
            require(command[1:3] == [str(ctx["repo"] / "empirical_execution/phase4/embeddings.py"), "encode"]
                    and flag("--records") == str(path.parent / "engineering_subset/records.jsonl")
                    and flag("--out") == str(path.parent / "golden") and flag("--batch-size") == "8"
                    and flag("--threads") == "1", "qualification_golden_process_mismatch")
        else:
            require(command[1] == str(ctx["folder"] / "worker.py") and flag("--mode") == kind
                    and flag("--shard-index") == str(index) and flag("--plan") == str(path.parent / "plan.json")
                    and flag("--out") == str(path.parent / f"{kind}-{index}")
                    and flag("--package-sha256") == ctx["package_sha256"], "qualification_worker_process_mismatch")
    plan = c.read_json(path.parent / "plan.json")
    validate_plan(plan, ctx["fingerprint"], ctx["pins"]["rows"])
    require(plan["kind"] == "qualification" and plan["sha256"] == report.get("plan_sha256")
            and plan["chunks_per_pass"] == 149, "qualification_plan_mismatch")
    # Actual golden and shard evidence is checked again; boolean declarations
    # alone cannot unlock full work. No additional model execution occurs here.
    from driver import compare_cache_files, compare_shard_slice, make_plan
    require(plan == make_plan(ctx, "qualification"), "qualification_plan_not_prospectively_selected")
    compare_cache_files(path.parent / "golden", path.parent / "assembled")
    assembled = c.read_json(path.parent / "assembled/manifest.json"); verify_seal(assembled)
    require(assembled.get("schema") == "third-e5-sharded-cpu-cache-v1" and assembled.get("passed") is True
            and assembled.get("plan_sha256") == plan["sha256"] and assembled.get("fingerprint") == ctx["fingerprint"]
            and assembled.get("rows") == 19, "qualification_assembled_manifest_mismatch")
    require(set(assembled.get("files", {})) == {"vectors.npy", "chunk_log.jsonl"}, "qualification_assembled_file_inventory_mismatch")
    for name, expected in assembled["files"].items():
        ctx["watch"].add(path.parent / "assembled" / name, expected)
    rows, expected_rows = rows_for_indices(ctx, plan["frame_row_indices"])
    from worker import validate_encoded
    from driver import validate_replay_receipt
    for i, shard in enumerate(plan["shards"]):
        start, stop = shard["start"], shard["stop"]
        encoded_dir, replay_dir = path.parent / f"encode-{i}", path.parent / f"replay-{i}"
        encoded = validate_encoded(ctx, plan, i, encoded_dir, rows=rows[start:stop], expected=expected_rows[start:stop])
        validate_replay_receipt(ctx, plan, i, replay_dir, encoded_dir, encoded)
        compare_shard_slice(path.parent / "assembled", encoded_dir, start, stop)
    ctx["qualified_result_binding"] = binding(path)
    return report
