#!/usr/bin/env python3
"""Offline, immutable, engineering-only model-annotation pilot. No human labels."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
import traceback

MAX_NEW_TOKENS = 512
TOTAL_TOKEN_BUDGET = 32768
LMFE_VERSION = "0.11.3"
SCHEDULING = "full_input_tokens_desc_then_original_index"
CONSTRAINT_ALGORITHM = "lmfe-jsonschema-prefix-token-filter-v4-replay-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_json(stream, value):
    stream.write(canonical(value).decode() + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def strict_json(raw):
    def bad_constant(value):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=bad_constant)


def validate_schema(value, schema, path="$"):
    """Strict subset of JSON Schema used by the pinned response schemas."""
    if "const" in schema:
        require(type(value) is type(schema["const"]) and value == schema["const"], path + ": const")
    if "enum" in schema:
        require(any(type(value) is type(x) and value == x for x in schema["enum"]), path + ": enum")
    kind = schema.get("type")
    if kind:
        predicates = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
                      "string": lambda x: isinstance(x, str), "null": lambda x: x is None,
                      "boolean": lambda x: isinstance(x, bool),
                      "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
                      "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)}
        require(kind in predicates and predicates[kind](value), path + ": type")
    if isinstance(value, dict):
        require(set(schema.get("required", ())).issubset(value), path + ": required keys")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            require(set(value).issubset(properties), path + ": extra keys")
        for key, child in properties.items():
            if key in value:
                validate_schema(value[key], child, path + "." + key)
    if isinstance(value, str):
        require(len(value) >= schema.get("minLength", 0), path + ": minLength")
        require(len(value) <= schema.get("maxLength", float("inf")), path + ": maxLength")
        if "pattern" in schema:
            require(re.search(schema["pattern"], value) is not None, path + ": pattern")
        if schema.get("format") == "date-time":
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            require(parsed.tzinfo is not None and "T" in value, path + ": date-time timezone")
    if isinstance(value, list):
        require(schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")), path + ": array length")
        if schema.get("uniqueItems"):
            require(len({canonical(x) for x in value}) == len(value), path + ": uniqueItems")
        for index, item in enumerate(value):
            if "items" in schema:
                validate_schema(item, schema["items"], f"{path}[{index}]")
    if "oneOf" in schema:
        successes = 0
        for branch in schema["oneOf"]:
            try:
                validate_schema(value, branch, path)
                successes += 1
            except ValueError:
                pass
        require(successes == 1, path + ": oneOf")
    for branch in schema.get("allOf", ()):
        validate_schema(value, branch, path)
    if "if" in schema:
        try:
            validate_schema(value, schema["if"], path)
            condition = True
        except ValueError:
            condition = False
        branch = schema.get("then" if condition else "else")
        if branch is not None:
            validate_schema(value, branch, path)
    return value


def validate_judgment(value, schema):
    validate_schema(value, schema)
    require(len(value["reason"].split()) <= 80, "Reason exceeds the pinned 80-word prompt limit")
    return value


def validate_response(value, schema):
    return validate_schema(value, schema)


def parse_judgment(raw, schema):
    return validate_judgment(strict_json(raw), schema)


def constraint_config(alphabet):
    from bounded_reason import exact_config
    return exact_config(alphabet)


def constraint_description(schema, alphabet=None):
    return {"algorithm_revision": CONSTRAINT_ALGORITHM, "package": "lm-format-enforcer",
            "package_version": LMFE_VERSION, "schema_sha256": digest_bytes(canonical(schema)),
            "max_consecutive_whitespaces": 12, "force_json_field_order": False,
            "max_json_array_length": 8,
            "alphabet_sha256": digest_bytes(alphabet.encode()) if alphabet is not None else None,
            "alphabet_length": len(alphabet) if alphabet is not None else None,
            "alphabet_source": "effective_constraint_config.json; exact tokenizer-derived alphabet",
            "effective_config_verified": alphabet is not None,
            "guard_source_sha256": file_hash(Path(__file__).with_name("bounded_reason.py")),
            "reason_word_limit": 80, "word_count_semantics": "Python str.split on decoded reason value",
            "base_proposal_shortcut_allowed": True, "guard_shortcut_allowed": False,
            "candidate_selection": "Descending finite proposal score, then lowest token ID; first passing guarded transition",
            "non_prefix_stable_token_transition": "reject",
            "logit_argmax_evidence": "runtime-attested; no saved full logits",
            "fresh_parser_per_call": True, "fresh_prefix_filter_per_call": True,
            "generated_token_and_character_replay": True,
            "grammar_is_partial_schema_validation": True,
            "strict_final_validation_including_unique_items_and_80_words": True}


def build_constraint(schema, tokenizer_data):
    from bounded_reason import make_base_enforcer
    enforcer = make_base_enforcer(schema, tokenizer_data)
    def prefix_allowed_tokens_fn(batch_id, sent):
        return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens
    prefix_allowed_tokens_fn.token_enforcer = enforcer
    return enforcer.root_parser, prefix_allowed_tokens_fn


def replay_constraint(raw, schema, alphabet):
    """Replay with the same effective alphabet and decoded word guard."""
    from bounded_reason import replay
    replay(raw, schema, alphabet)
    return True


def replay_tokens(input_ids, output_ids, schema, tokenizer_data):
    """Replay native proposals and exact guarded transitions; no logits claim."""
    from bounded_reason import replay_guard_tokens
    replay_guard_tokens(input_ids, output_ids, schema, tokenizer_data)
    return True


def pair_seed(plan_sha256, pair_id, model_id):
    return int(hashlib.sha256(f"{plan_sha256}|{pair_id}|{model_id}".encode()).hexdigest(), 16) % (2**63)


def load_verified_inputs(path, expected_sha256, max_pairs):
    require(file_hash(path) == expected_sha256, "Input file SHA256 mismatch")
    rows, canonical_rows, swapped_rows = [], {}, {}
    with Path(path).open("rb") as stream:
        for index, physical in enumerate(stream, 1):
            require(bool(physical.strip()), f"Blank input physical line {index}")
            row = strict_json(physical.decode("utf-8"))
            require(isinstance(row, dict) and set(row) == {"pair_id", "text_a", "text_b", "orientation"},
                    "Input fields must be exactly pair_id,text_a,text_b,orientation")
            require(all(isinstance(row[k], str) and row[k] for k in row), "Input fields must be nonempty strings")
            require(row["orientation"] in ("canonical", "swapped_diagnostic"), "Unknown input orientation")
            target = canonical_rows if row["orientation"] == "canonical" else swapped_rows
            require(row["pair_id"] not in target, "Repeated pair/orientation")
            target[row["pair_id"]] = row
            rows.append(row)
    require(1 <= len(canonical_rows) <= max_pairs, "Canonical pair count outside pilot bound")
    require(len(swapped_rows) <= len(canonical_rows) // 10, "Swapped diagnostics exceed 10% of canonical pairs")
    for pair_id, row in swapped_rows.items():
        require(pair_id in canonical_rows, "Orphan swapped diagnostic")
        original = canonical_rows[pair_id]
        require(row["text_a"] == original["text_b"] and row["text_b"] == original["text_a"],
                "Swapped diagnostic must contain the identical full texts in reverse order")
    require(file_hash(path) == expected_sha256, "Input file changed during validation")
    return rows


def safe_relative(name):
    require(isinstance(name, str) and bool(name), "Relative path must be nonempty text")
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "Unsafe relative asset/package path")
    return p


def verify_weights(manifest, model_path):
    items = manifest.get("files")
    require(isinstance(items, list) and items, "Weight manifest needs complete file inventory")
    seen = set()
    for item in items:
        name = item["relative_path"]
        relative = safe_relative(name)
        require(name not in seen, "Duplicate model inventory entry")
        seen.add(name)
        path = model_path / relative
        require(path.is_file() and path.stat().st_size == item["size_bytes"], "Model asset missing/size mismatch: " + name)
        require(file_hash(path) == item["sha256"], "Model asset hash mismatch: " + name)
        require(path.suffix not in (".py", ".pyc"), "Custom model code not allowed")
    actual = {str(p.relative_to(model_path)) for p in model_path.rglob("*") if p.is_file()}
    # These publisher metadata files are never loaded by the inference backend.
    # All executable/config/tokenizer/generation/weight files remain bound.
    require(actual - seen <= {"README.md", "LICENSE", ".gitattributes"}, "Unbound inference-capable snapshot file")
    require({"config.json", "tokenizer_config.json", "tokenizer.json"}.issubset(seen), "Incomplete model/tokenizer assets")
    require(any(name.endswith(".safetensors") for name in seen), "Safetensors weights required")
    for name in seen:
        if name.endswith(".safetensors.index.json"):
            index = strict_json((model_path / name).read_text())
            require(set(index["weight_map"].values()).issubset(seen), "Unbound model shard")
    for name in ("config.json", "tokenizer_config.json"):
        require(not strict_json((model_path / name).read_text()).get("auto_map"), "Custom auto_map forbidden")


def prepare_context(manifest_path, out, state):
    manifest = strict_json(manifest_path.read_text())
    required = {"plan_file", "plan_sha256", "inputs_file", "inputs_sha256", "weights_manifest_file",
                "weights_manifest_sha256", "model_path", "model_id", "model_revision", "source_role", "pilot_id", "seed",
                "driver_sha256", "expected_pairs", "expected_calls", "preparation_sha256",
                "runtime_plan_file", "runtime_plan_sha256", "engineering_frame_census_file",
                "engineering_frame_census_sha256"}
    require(required.issubset(manifest), "Incomplete launch manifest")
    require(manifest["source_role"] == "engineering_only", "Runner is engineering-only")
    require(file_hash(__file__) == manifest["driver_sha256"], "Runner source differs from launch manifest")
    require(isinstance(manifest["seed"], int) and not isinstance(manifest["seed"], bool), "Manifest seed must be integer")
    for name in ("plan_file", "inputs_file", "weights_manifest_file", "model_path", "runtime_plan_file", "engineering_frame_census_file"):
        require(Path(manifest[name]).is_absolute(), name + " must be absolute")
    require(Path(manifest["model_path"]).name == manifest["model_revision"], "Snapshot directory revision mismatch")
    plan_path = Path(manifest["plan_file"])
    require(file_hash(plan_path) == manifest["plan_sha256"], "Plan SHA256 mismatch")
    plan = strict_json(plan_path.read_text())
    require(plan["technical_pilot"]["source_role"] == "engineering_only" and plan["technical_pilot"]["human_reviews"] == 0,
            "Unexpected pilot scope")
    require(manifest["model_id"] == plan["technical_pilot"]["judge"], "Wrong pilot judge")
    require(any(m["model_id"] == manifest["model_id"] and m["revision"] == manifest["model_revision"]
                for m in plan["models"]), "Unpinned model/revision")
    runtime_plan_path = Path(manifest["runtime_plan_file"])
    require(file_hash(runtime_plan_path) == manifest["runtime_plan_sha256"], "Runtime amendment SHA256 mismatch")
    runtime_plan = strict_json(runtime_plan_path.read_text())
    required_runtime = {"runtime_plan_schema": "third-judge-runtime-amendment-v4",
        "lm_format_enforcer_version": LMFE_VERSION, "max_new_tokens": MAX_NEW_TOKENS,
        "total_token_budget": TOTAL_TOKEN_BUDGET, "format_retry_limit": 0,
        "attention_implementation": "sdpa", "sdpa_backend": "FLASH_ATTENTION",
        "scheduling": SCHEDULING, "seed_plan_sha256": manifest["plan_sha256"]}
    for key, expected in required_runtime.items():
        require(runtime_plan.get(key) == expected, "Runtime amendment mismatch: " + key)
    require(runtime_plan.get("lmfe_config") == {"max_consecutive_whitespaces": 12,
            "force_json_field_order": False, "max_json_array_length": 8}, "Runtime grammar config mismatch")
    require(runtime_plan.get("effective_constraint_policy") == {
        "reason_word_limit": 80, "word_count_semantics": "Python str.split on decoded reason value",
        "tokenizer_alphabet": True, "base_proposal_shortcut_allowed": True, "guard_shortcut_allowed": False,
        "effective_config_verified_after_enforcer_creation": True,
        "selection": "Descending finite native-proposal score, then lowest token ID; first candidate whose exact tokenizer-state transition passes full schema and decoded-word guard",
        "non_prefix_stable_token_transition": "rejected with explicit evidence", "sampling_allowed": False,
        "beam_count": 1, "logit_argmax_evidence": "Runtime code, assertions and candidate-count ledger; no independent full-logit replay"},
        "Effective constraint policy mismatch")
    require(importlib.metadata.version("lm-format-enforcer") == LMFE_VERSION, "Unpinned LMFE version")
    repo = Path(manifest.get("repository_root", str(plan_path.parent.parent.parent)))
    bound = {str(manifest_path): file_hash(manifest_path), str(plan_path): manifest["plan_sha256"],
             str(runtime_plan_path): manifest["runtime_plan_sha256"]}
    require(isinstance(runtime_plan.get("package_files"), dict) and
            {"run_pilot.py", "bounded_reason.py", "prepare_pilot.py", "requirements.txt", "pilot.sbatch"}.issubset(runtime_plan["package_files"]),
            "Runtime amendment must bind driver/preparation/requirements/Slurm package")
    for name, expected in runtime_plan["package_files"].items():
        path = runtime_plan_path.parent / safe_relative(name)
        require(file_hash(path) == expected, "Runtime package hash mismatch: " + name)
        bound[str(path)] = expected
    for version in ("v2", "v3"):
        inventory = runtime_plan.get("original_" + version + "_folder_files", {})
        require(inventory, "Preserved " + version + " inventory required")
        for name, expected in inventory.items():
            path = runtime_plan_path.parent.parent / ("model_annotation_runtime_" + version + "_20261011") / safe_relative(name)
            require(file_hash(path) == expected, "Preserved " + version + " file hash mismatch: " + name)
            bound[str(path)] = expected
    census_path = manifest["engineering_frame_census_file"]
    require(file_hash(census_path) == manifest["engineering_frame_census_sha256"], "Engineering frame census SHA mismatch")
    bound[census_path] = manifest["engineering_frame_census_sha256"]
    for name, expected in plan["package_files"].items():
        path = plan_path.parent / safe_relative(name)
        require(file_hash(path) == expected, "Plan package hash mismatch: " + name)
        bound[str(path)] = expected
    for name, expected in plan["protocol_bindings"].items():
        path = repo / safe_relative(name)
        require(file_hash(path) == expected, "Frozen protocol hash mismatch: " + name)
        bound[str(path)] = expected
    rows = load_verified_inputs(manifest["inputs_file"], manifest["inputs_sha256"], plan["technical_pilot"]["max_pairs"])
    require(manifest["inputs_sha256"] == runtime_plan["expected_input_sha256"], "v4 must use byte-identical pilot inputs")
    require(sum(r["orientation"] == "canonical" for r in rows) == manifest["expected_pairs"] == 40,
            "Expected exactly 40 sealed canonical pilot pairs")
    require(len(rows) == manifest["expected_calls"] == 44, "Expected exactly 44 sealed input calls")
    preparation_path = manifest_path.parent / "preparation.json"
    require(file_hash(preparation_path) == manifest["preparation_sha256"], "Preparation receipt hash mismatch")
    preparation = strict_json(preparation_path.read_text())
    require(preparation["evidence_role"] == "engineering_only" and preparation["new_human_responses"] == 0 and
            preparation["selected_unique_pairs"] == 40 and preparation["scheduled_calls"] == 44,
            "Unexpected preparation receipt scope/counts")
    mapping_path = manifest_path.parent / "private_source_mapping.json"
    require(file_hash(mapping_path) == preparation["source_mapping_sha256"], "Private source mapping hash mismatch")
    source_records = preparation["full_text_source"]
    require(file_hash(source_records) == preparation["source_records_sha256"], "Prepared full-text source hash mismatch")
    bound.update({str(preparation_path): manifest["preparation_sha256"],
                  str(mapping_path): preparation["source_mapping_sha256"], source_records: preparation["source_records_sha256"]})
    state.update(manifest=manifest, plan=plan, runtime_plan=runtime_plan, rows=rows, bound_files=bound)
    state["judgment_schema"] = strict_json((plan_path.parent / "model_judgment.schema.json").read_text())
    state["response_schema"] = strict_json((plan_path.parent / "model_response.schema.json").read_text())
    state["system_prompt"] = (plan_path.parent / "judge_system_prompt.txt").read_text()
    state["task_definition"] = (plan_path.parent / "judge_task_definition.txt").read_text()
    prompt = {"system_prompt": state["system_prompt"], "task_definition": state["task_definition"],
              "user_message_renderer": "json object with task_definition and documents A/text then B/text; ensure_ascii=False",
              "format_only_retry_limit": 0, "pair_ids_and_provenance_shown_to_model": False}
    write_json(out / "prompt_manifest.json", prompt)
    state["prompt_sha256"] = file_hash(out / "prompt_manifest.json")
    decoding = {"do_sample": False, "num_beams": 1, "batch_size": 1, "max_new_tokens": MAX_NEW_TOKENS,
                "total_token_budget_including_output_reserve": TOTAL_TOKEN_BUDGET, "truncation": False,
                "dtype": "bfloat16", "attention_implementation": "sdpa", "use_cache": True,
                "sdpa_backend": "FLASH_ATTENTION", "sdpa_math_mem_efficient_cudnn_fallback": False,
                "output_attentions": False, "scheduling": SCHEDULING,
                "repetition_penalty": 1.0, "format_only_retry_limit": 0,
                "per_pair_seed_formula": "int(SHA256(plan_sha256|pair_id|model_id),16) mod 2^63",
                "seed_plan_sha256": manifest["plan_sha256"], "swap_seed": "same as canonical pair",
                "runtime_plan_sha256": manifest["runtime_plan_sha256"],
                "constraint": constraint_description(state["judgment_schema"]),
                "manifest_seed_role": "input-selection provenance; no extra sampling during inference"}
    write_json(out / "decoding_config.json", decoding)
    state["decoding_config_sha256"] = file_hash(out / "decoding_config.json")
    weight_path = Path(manifest["weights_manifest_file"])
    require(file_hash(weight_path) == manifest["weights_manifest_sha256"], "Weight manifest SHA256 mismatch")
    weights = strict_json(weight_path.read_text())
    if "model_id" in weights:
        require(weights["model_id"] == manifest["model_id"], "Weights model identity mismatch")
    if "model_revision" in weights:
        require(weights["model_revision"] == manifest["model_revision"], "Weights revision mismatch")
    if "revision" in weights:
        require(weights["revision"] == manifest["model_revision"], "Weights revision mismatch")
    verify_weights(weights, Path(manifest["model_path"]))
    state["weights"] = weights
    bound[str(weight_path)] = manifest["weights_manifest_sha256"]
    bound[manifest["inputs_file"]] = manifest["inputs_sha256"]
    bound[str(Path(__file__).resolve())] = file_hash(__file__)
    write_json(out / "launch_manifest.private.json", manifest)
    write_json(out / "validated_input_binding.json", {"inputs_sha256": manifest["inputs_sha256"],
        "rows": len(rows), "canonical_pairs": sum(r["orientation"] == "canonical" for r in rows),
        "full_texts_preserved": True, "source_role": "engineering_only", "human_responses_created": 0})


def base_runtime(state):
    versions = {}
    names = set(("torch", "transformers", "tokenizers", "safetensors", "numpy", "accelerate", "lm-format-enforcer", "interegular"))
    names.update(state.get("runtime_plan", {}).get("required_packages", {}))
    for name in sorted(names):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"schema": "third-judge-runtime-v4", "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "hostname": platform.node(), "packages": versions,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "sys_path": sys.path,
            "runtime_dependency_source": state.get("manifest", {}).get("runtime_dependency_source"),
            "source_role": "engineering_only", "human_completed": False,
            "process_address_space_limits": list(resource.getrlimit(resource.RLIMIT_AS))}


def load_model(state, out):
    require(platform.system() == "Linux" and os.environ.get("SLURM_JOB_ID"), "Use a Linux Slurm GPU allocation")
    pins = state["runtime_plan"].get("required_packages", {})
    require({"torch", "transformers", "tokenizers", "lm-format-enforcer"}.issubset(pins), "Incomplete runtime package pins")
    for name, expected in pins.items():
        require(importlib.metadata.version(name) == expected, "Runtime package version mismatch: " + name)
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
        os.environ[key] = value
    import torch
    import transformers
    import lmformatenforcer
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    state["torch"] = torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one visible CUDA GPU required")
    require(torch.cuda.is_bf16_supported(), "CUDA BF16 support required")
    device = torch.cuda.get_device_properties(0)
    require("A100" in device.name and device.total_memory >= 75 * 1024**3, "Frozen pilot requires A100 80GB")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    def flash_flags():
        return {"flash": torch.backends.cuda.flash_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
                "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled()}
    state["flash_context"] = lambda: sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    state["flash_flags"] = flash_flags
    with state["flash_context"]():
        selected_flags = flash_flags()
        require(selected_flags == {"flash": True, "math": False, "mem_efficient": False, "cudnn": False},
                "FLASH_ATTENTION must be the only enabled SDPA backend")
    model_path = state["manifest"]["model_path"]
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_fast=True)
    require(bool(tokenizer.chat_template), "Pinned tokenizer needs a chat template")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=False,
        device_map=None)
    model.to(device="cuda:0", dtype=torch.bfloat16)
    model.eval()
    require(all(p.device.type == "cuda" and p.dtype == torch.bfloat16 for p in model.parameters()), "Model dtype/device mismatch")
    require(getattr(model.config, "_attn_implementation", None) == "sdpa", "Expected SDPA attention")
    model.config.output_attentions = False
    require(model.config.model_type == "qwen2", "Expected the pinned native Qwen2 implementation")
    require(model.config.max_position_embeddings >= TOTAL_TOKEN_BUDGET, "Model position capacity is too small")
    require(not getattr(model.config, "rope_scaling", None), "Native context only; no RoPE scaling amendment")
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True, repetition_penalty=1.0, eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id, output_attentions=False,
        forced_bos_token_id=None, forced_eos_token_id=None, remove_invalid_values=False,
        renormalize_logits=False, num_return_sequences=1)
    write_json(out / "generation_config.json", generation.to_dict())
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    verified_parser, _ = build_constraint(state["judgment_schema"], tokenizer_data)
    require(verified_parser.config == constraint_config(tokenizer_data.tokenizer_alphabet), "Effective constraint config differs")
    from bounded_reason import make_parser
    verified_guard = make_parser(state["judgment_schema"], tokenizer_data.tokenizer_alphabet)
    require(verified_guard.config == verified_parser.config and verified_guard.shortcut_key() is None,
            "Guard config must match native proposal config with guard shortcut disabled")
    effective_constraint = constraint_description(state["judgment_schema"], tokenizer_data.tokenizer_alphabet)
    write_json(out / "effective_constraint_config.json", {"schema": "third-effective-constraint-config-v4",
        "alphabet": tokenizer_data.tokenizer_alphabet, "configuration": effective_constraint})
    implementation_files = {str(Path(inspect.getfile(model.__class__)).resolve()),
        str(Path(inspect.getfile(model.model.layers[0].self_attn.__class__)).resolve())}
    from transformers.integrations import sdpa_attention
    from transformers.generation import utils as generation_utils, logits_process
    implementation_files.update(str(Path(module.__file__).resolve()) for module in
                                (sdpa_attention, generation_utils, logits_process))
    implementation_files.update(str(p.resolve()) for p in Path(lmformatenforcer.__file__).parent.rglob("*.py"))
    implementation_bindings = {p: file_hash(p) for p in sorted(implementation_files)}
    state["bound_files"].update(implementation_bindings)
    runtime = base_runtime(state)
    runtime.update({"status": "model_loaded_before_first_inference", "torch_module": torch.__file__,
                    "transformers_module": transformers.__file__, "torch_cuda_version": torch.version.cuda,
                    "gpu": device.name, "gpu_total_bytes": device.total_memory, "visible_gpus": 1,
                    "cuda_capability": list(torch.cuda.get_device_capability(0)), "dtype": "bfloat16",
                    "attention_implementation": "sdpa", "sdpa_backend": "FLASH_ATTENTION",
                    "sdpa_flags_inside_generation_context": selected_flags,
                    "output_attentions": False, "deterministic_algorithms": True,
                    "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"], "allow_tf32": False,
                    "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                    "chat_template_sha256": digest_bytes(canonical(tokenizer.chat_template)),
                    "tokenizer_class": type(tokenizer).__module__ + "." + type(tokenizer).__name__,
                    "tokenizer_length": len(tokenizer), "tokenizer_vocab_size": tokenizer.vocab_size,
                    "tokenizer_backend_json_sha256": digest_bytes(tokenizer.backend_tokenizer.to_str().encode()),
                    "tokenizer_special_ids": tokenizer.all_special_ids,
                    "model_class": type(model).__module__ + "." + type(model).__name__,
                    "attention_class": type(model.model.layers[0].self_attn).__name__,
                    "model_native_position_limit": model.config.max_position_embeddings,
                    "implementation_file_bindings": implementation_bindings,
                    "constraint": effective_constraint,
                    "effective_constraint_config_sha256": file_hash(out / "effective_constraint_config.json"),
                    "runtime_plan_sha256": state["manifest"]["runtime_plan_sha256"],
                    "generation_config_sha256": file_hash(out / "generation_config.json"),
                    "model_load_seconds": time.perf_counter() - started,
                    "model_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                    "no_network_or_tools_available_to_judge": True})
    write_json(out / "runtime_manifest.json", runtime)
    state.update(model=model, tokenizer=tokenizer, tokenizer_data=tokenizer_data, generation=generation, runtime=runtime,
                 constraint_alphabet=tokenizer_data.tokenizer_alphabet,
                 effective_constraint_config_sha256=file_hash(out / "effective_constraint_config.json"),
                 runtime_manifest_sha256=file_hash(out / "runtime_manifest.json"))


def envelope(state, row, call_id, status, judgment=None, raw_hash=None):
    m = state["manifest"]
    result = {"schema": "third-model-response-v2", "annotator_type": "model", "human_completed": False,
              "pair_id": row["pair_id"], "call_id": call_id, "model_id": m["model_id"],
              "model_revision": m["model_revision"], "weights_manifest_sha256": m["weights_manifest_sha256"],
              "prompt_sha256": state["prompt_sha256"], "input_sha256": digest_bytes(canonical(row)),
              "runtime_manifest_sha256": state["runtime_manifest_sha256"],
              "decoding_config_sha256": state["decoding_config_sha256"], "orientation": row["orientation"],
              "status": status, "judgment": judgment, "raw_output_sha256": raw_hash, "created_at": utcnow()}
    return validate_response(result, state["response_schema"])


def tokenize_call(state, row):
    messages = [{"role": "system", "content": state["system_prompt"]}, {"role": "user", "content": json.dumps(
        {"task_definition": state["task_definition"], "documents": [{"label": "A", "text": row["text_a"]},
           {"label": "B", "text": row["text_b"]}]}, ensure_ascii=False)}]
    rendered = state["tokenizer"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = state["tokenizer"](rendered, add_special_tokens=False, truncation=False, return_tensors="pt")
    return {"messages": messages, "rendered": rendered, "encoded": encoded,
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "rendered_prompt_sha256": digest_bytes(rendered.encode("utf-8"))}


def prepare_schedule(state, out):
    """Census all sealed rows before inference; return original indexes longest-first."""
    prepared = {index: tokenize_call(state, row) for index, row in enumerate(state["rows"])}
    schedule = sorted(prepared, key=lambda i: (-prepared[i]["input_tokens"], i))
    census = [{"input_index": i, "input_sha256": digest_bytes(canonical(state["rows"][i])),
        "orientation": state["rows"][i]["orientation"], "input_tokens": p["input_tokens"],
        "rendered_prompt_sha256": p["rendered_prompt_sha256"],
        "context_fit_with_output_reserve": p["input_tokens"] + MAX_NEW_TOKENS <= TOTAL_TOKEN_BUDGET}
        for i, p in prepared.items()]
    write_json(out / "schedule.json", {"schema": "third-pilot-input-schedule-v4",
        "source_role": "engineering_only", "policy": SCHEDULING,
        "inputs_sha256": state["manifest"]["inputs_sha256"], "execution_order_original_indexes": schedule,
        "input_census": census, "all_inputs_preserved": True, "outcome_based_selection": False,
        "total_token_budget": TOTAL_TOKEN_BUDGET, "max_new_tokens": MAX_NEW_TOKENS})
    state.update(prepared_calls=prepared, schedule=schedule, schedule_sha256=file_hash(out / "schedule.json"))
    return schedule


def run_call(state, row, index, out, attempts_stream):
    torch, tokenizer, model = state["torch"], state["tokenizer"], state["model"]
    call_id = digest_bytes(canonical([state["manifest"]["pilot_id"], index, row["pair_id"], row["orientation"]]))
    call_dir = out / "calls" / f"call-{index:04d}"
    call_dir.mkdir(parents=True, exist_ok=False)
    seed = pair_seed(state["manifest"]["plan_sha256"], row["pair_id"], state["manifest"]["model_id"])
    final_status, final_judgment, raw_hash = "runtime_error", None, None
    attempt_started = time.perf_counter()
    metrics = {"call_id": call_id, "input_index": index, "pair_id": row["pair_id"],
        "orientation": row["orientation"], "attempt": 0, "seed": seed,
        "format_only_retry": False, "format_only_retry_limit": 0,
        "constraint_algorithm_revision": CONSTRAINT_ALGORITHM,
        "effective_constraint_config_sha256": state["effective_constraint_config_sha256"],
        "created_at": utcnow(), "human_completed": False}
    prefix = call_dir / "attempt-0"
    encoded = generated = None
    greedy_guard = None
    gpu_metrics_started = False
    inference_started = None
    try:
        prepared = state.get("prepared_calls", {}).get(index)
        if prepared is None:
            prepared = tokenize_call(state, row)
        messages, rendered, input_tokens = prepared["messages"], prepared["rendered"], prepared["input_tokens"]
        encoded = prepared["encoded"]
        input_ids = encoded["input_ids"][0].tolist()
        write_json(str(prefix) + ".messages.private.json", messages)
        write_json(str(prefix) + ".input_tokens.private.json", input_ids)
        metrics.update(input_tokens=input_tokens, full_input_tokenization=True, truncation_applied=False,
                       rendered_prompt_sha256=prepared["rendered_prompt_sha256"])
        if input_tokens + MAX_NEW_TOKENS > TOTAL_TOKEN_BUDGET:
            final_status = "context_overflow"
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats(0)
            gpu_metrics_started = True
            encoded = {k: v.to("cuda:0") for k, v in encoded.items()}
            _, prefix_allowed = build_constraint(state["judgment_schema"], state["tokenizer_data"])
            from bounded_reason import GreedyGuardProcessor
            greedy_guard = GreedyGuardProcessor(state["judgment_schema"], state["tokenizer_data"],
                                                prefix_allowed.token_enforcer, input_ids)
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            with torch.inference_mode(), state["flash_context"]():
                flags = state["flash_flags"]()
                require(flags == {"flash": True, "math": False, "mem_efficient": False, "cudnn": False},
                        "FLASH_ATTENTION must be the only enabled SDPA backend during generation")
                metrics["sdpa_flags_during_generation"] = flags
                generated = model.generate(**encoded, generation_config=state["generation"],
                    prefix_allowed_tokens_fn=prefix_allowed, logits_processor=[greedy_guard], output_attentions=False)
            torch.cuda.synchronize()
            metrics["generation_seconds"] = time.perf_counter() - inference_started
            output_ids = generated[0, input_tokens:].detach().cpu().tolist()
            raw = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            raw_bytes = raw.encode("utf-8")
            raw_hash = digest_bytes(raw_bytes)
            with Path(str(prefix) + ".raw.private.txt").open("xb") as stream:
                stream.write(raw_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            write_json(str(prefix) + ".output_tokens.private.json", output_ids)
            metrics.update(output_tokens=len(output_ids), raw_output_sha256=raw_hash,
                           reached_output_token_cap=len(output_ids) >= MAX_NEW_TOKENS)
            try:
                replay_tokens(input_ids, output_ids, state["judgment_schema"], state["tokenizer_data"])
                metrics["generated_token_filter_replay_passed"] = True
                replay_constraint(raw, state["judgment_schema"], state["constraint_alphabet"])
                metrics["decoded_character_grammar_replay_passed"] = True
                final_judgment = parse_judgment(raw, state["judgment_schema"])
                final_status = "valid"
            except (ValueError, TypeError) as error:
                final_status, final_judgment = "invalid_json", None
                metrics["format_error"] = str(error)
    except Exception as error:
        final_status, final_judgment = "runtime_error", None
        metrics.update(exception_type=type(error).__name__, exception_message=str(error))
    finally:
        metrics["status"] = final_status
        if inference_started is not None and "generation_seconds" not in metrics:
            metrics["generation_seconds"] = time.perf_counter() - inference_started
            metrics["generation_timing_scope"] = "failed generation attempt elapsed wall time"
        metrics["seconds"] = time.perf_counter() - attempt_started
        metrics["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if gpu_metrics_started:
            # Retain peaks even after an OOM/backend error. No fallback is used.
            metrics["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            metrics["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        encoded = generated = None
        if greedy_guard is not None:
            ledger_path = Path(str(prefix) + ".guard_steps.private.json")
            write_json(ledger_path, greedy_guard.steps)
            rejected = Counter()
            for step in greedy_guard.steps:
                rejected.update(step["rejected_candidates"])
            metrics.update(guard_steps_sha256=file_hash(ledger_path), guard_steps=len(greedy_guard.steps),
                guard_candidates_considered=sum(step["candidates_considered"] for step in greedy_guard.steps),
                guard_rejected_candidates=dict(rejected),
                guard_selection_seconds=sum(step["selection_seconds"] for step in greedy_guard.steps))
        append_json(attempts_stream, metrics)
        write_json(str(prefix) + ".metrics.json", metrics)
    result = envelope(state, row, call_id, final_status, final_judgment, raw_hash)
    write_json(call_dir / "response.json", result)
    return result


def summarize(state, responses, started, fatal=None):
    canonical_responses = [r for r in responses if r["orientation"] == "canonical"]
    canonical_map = {r["pair_id"]: r for r in canonical_responses}
    diagnostics = Counter()
    for row in responses:
        if row["orientation"] != "swapped_diagnostic":
            continue
        original = canonical_map.get(row["pair_id"])
        if not original or original["status"] != "valid" or row["status"] != "valid":
            diagnostics["unavailable_comparison"] += 1
        else:
            a, b = original["judgment"]["category"], row["judgment"]["category"]
            diagnostics["valid_comparison"] += 1
            diagnostics["category_changed"] += a != b
            positive = {"exact_copy", "substantially_same_meaning"}
            diagnostics["binary_positive_changed"] += (a in positive) != (b in positive)
    return {"schema": "third-model-annotation-technical-pilot-summary-v4", "completed": fatal is None,
            "source_role": "engineering_only", "pilot_id": state.get("manifest", {}).get("pilot_id"),
            "planned_input_rows": len(state.get("rows", [])), "recorded_input_rows": len(responses),
            "canonical_pairs": len(canonical_responses),
            "status_counts_all_calls": dict(Counter(r["status"] for r in responses)),
            "status_counts_canonical": dict(Counter(r["status"] for r in canonical_responses)),
            "canonical_category_counts": dict(Counter(r["judgment"]["category"] for r in canonical_responses if r["status"] == "valid")),
            "swapped_order_diagnostics": dict(diagnostics), "diagnostic_calls_are_extra_votes": False,
            "human_responses_created": 0, "human_completed": False, "v1_human_gate_satisfied": False,
            "threshold_selected": False, "model_panel_calibration_passed": False,
            "confirmatory_study_ready": False, "seconds": time.perf_counter() - started,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_scope": "Linux process high-water mark including model loading and prior pairs",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "fatal_error": fatal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    (out / ".gitignore").write_text("*\n")
    state, responses, fatal = {}, [], None
    completed_indexes = set()
    started = time.perf_counter()
    with (out / "responses.jsonl").open("x", encoding="utf-8") as records, (out / "attempts.jsonl").open("x", encoding="utf-8") as attempts:
        try:
            prepare_context(Path(args.manifest).resolve(), out, state)
            load_model(state, out)
            for index in prepare_schedule(state, out):
                row = state["rows"][index]
                response = run_call(state, row, index, out, attempts)
                append_json(records, response)
                responses.append(response)
                completed_indexes.add(index)
                print(json.dumps({"completed_calls": len(responses), "planned_calls": len(state["rows"]),
                                  "status": response["status"], "orientation": row["orientation"]}), flush=True)
            for path, expected in state["bound_files"].items():
                require(file_hash(path) == expected, "Bound source changed during execution: " + path)
            verify_weights(state["weights"], Path(state["manifest"]["model_path"]))
        except Exception as error:
            fatal = {"exception_type": type(error).__name__, "message": str(error)}
            with (out / "traceback.private.txt").open("x") as stream:
                traceback.print_exc(file=stream)
            # Validated planned rows never disappear after startup or top-level
            # failure. Missing provenance means inputs could not be trusted yet.
            if state.get("rows") and "decoding_config_sha256" in state:
                if "runtime_manifest_sha256" not in state:
                    runtime = base_runtime(state)
                    runtime.update(status="startup_failed_before_inference", failure=fatal)
                    write_json(out / "runtime_manifest.json", runtime)
                    state["runtime_manifest_sha256"] = file_hash(out / "runtime_manifest.json")
                for index, row in enumerate(state["rows"]):
                    if index in completed_indexes:
                        continue
                    call_id = digest_bytes(canonical([state["manifest"]["pilot_id"], index, row["pair_id"], row["orientation"]]))
                    response = envelope(state, row, call_id, "runtime_error")
                    append_json(records, response)
                    responses.append(response)
            write_json(out / "FAILED.json", fatal)
    summary = summarize(state, responses, started, fatal)
    summary["schedule_sha256"] = state.get("schedule_sha256")
    summary["format_retry_limit"] = 0
    summary["runtime_plan_sha256"] = state.get("manifest", {}).get("runtime_plan_sha256")
    attempt_rows = [strict_json(line) for line in (out / "attempts.jsonl").read_text().splitlines() if line.strip()]
    summary["attempt_accounting"] = {
        "attempts": len(attempt_rows), "format_retry_attempts": sum(r["format_only_retry"] for r in attempt_rows),
        "total_attempt_seconds": sum(r["seconds"] for r in attempt_rows),
        "generation_seconds": sum(r.get("generation_seconds", 0.0) for r in attempt_rows),
        "full_input_tokens_across_attempts": sum(r.get("input_tokens", 0) for r in attempt_rows),
        "output_tokens_across_attempts": sum(r.get("output_tokens", 0) for r in attempt_rows),
        "peak_gpu_allocated_bytes": max((r.get("peak_gpu_allocated_bytes", 0) for r in attempt_rows), default=0),
        "peak_gpu_reserved_bytes": max((r.get("peak_gpu_reserved_bytes", 0) for r in attempt_rows), default=0),
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if fatal is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
