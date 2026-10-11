"""Independent Mistral-v1 provenance audit; no model inference or changed judgments.

Requires jsonschema and lm-format-enforcer. --tokenizer-path additionally uses
Transformers on CPU to independently reconstruct prompts and replay token filters.
The source workspace is required; copied reports alone cannot prove source texts.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from jsonschema import Draft202012Validator

FLAGS = {"flash": True, "math": False, "mem_efficient": False, "cudnn": False}
ALGORITHM = "lmfe-jsonschema-prefix-token-filter-v5-replay-v1"
SCHEDULE = "full_input_tokens_desc_then_original_index"
MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"
MODEL_REVISION = "04d8a90549d23fc6bd7f642064003592df51e9b3"
MISTRAL_CONFIG = {"max_position_embeddings": 131072, "head_dim": 128,
    "hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 32,
    "num_key_value_heads": 8, "sliding_window": None, "vocab_size": 131072,
    "bos_token_id": 1, "eos_token_id": 2, "rope_theta": 1000000.0}
TOKEN_POLICY = {
    'schema': 'third-standalone-token-policy-v5',
    'standalone_decode': {'skip_special_tokens': False, 'clean_up_tokenization_spaces': False},
    'regular_token_must_decode_nonempty': True,
    'regular_token_must_not_contain': 'U+FFFD',
    'eos_policy': 'Separate EOS IDs; permitted only after complete guarded JSON',
    'non_eos_emitted_transition_must_be_nonempty': True,
    'full_output_prefix_decode_must_equal_guarded_text': True,
    'scope': 'Narrower token vocabulary; valid split-codepoint tokens and literal U+FFFD are excluded',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value, ascii=True):
    return json.dumps(value, sort_keys=True, ensure_ascii=ascii, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value, ascii=True):
    return hashlib.sha256(canonical(value, ascii)).hexdigest()


def strict(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "Duplicate JSON object key")
            value[key] = item
        return value
    def nonfinite(_value):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def check_schema(value, validator, message):
    require(not list(validator.iter_errors(value)), message)


def verify_token_eligibility(artifact, regular_ids, eos_ids, raw_decoder=None):
    """Independent inventory audit; raw_decoder must not strip replacement characters."""
    require(set(artifact) == {'schema', 'policy', 'policy_sha256', 'eligible_token_ids',
            'excluded_token_reasons', 'eos_token_ids', 'regular_token_count', 'eligible_token_count',
            'excluded_token_count', 'tokenizer_length', 'special_token_ids', 'standalone_decode_stream_sha256', 'standalone_decode_stream_format'},
            "Token eligibility artifact fields differ")
    require(artifact['schema'] == 'third-token-eligibility-v5' and artifact['policy'] == TOKEN_POLICY
            and artifact['policy_sha256'] == digest(TOKEN_POLICY), "Token eligibility policy differs")
    eligible, excluded = artifact['eligible_token_ids'], artifact['excluded_token_reasons']
    require(isinstance(eligible, list) and all(type(x) is int and x >= 0 for x in eligible)
            and eligible == sorted(set(eligible)), "Invalid eligible token IDs")
    require(isinstance(excluded, dict) and all(isinstance(k, str) and k.isdecimal() and str(int(k)) == k
            and v in {'empty_standalone_decode', 'replacement_character_in_standalone_decode'} for k, v in excluded.items()),
            "Invalid excluded token inventory")
    excluded_ids = {int(x) for x in excluded}
    regular_ids, eos_ids = sorted(regular_ids), sorted(eos_ids)
    require(type(artifact['tokenizer_length']) is int and artifact['tokenizer_length'] > 0
            and isinstance(artifact['special_token_ids'], list)
            and all(type(x) is int and 0 <= x < artifact['tokenizer_length'] for x in artifact['special_token_ids'])
            and artifact['special_token_ids'] == sorted(set(artifact['special_token_ids']))
            and set(regular_ids) == set(range(artifact['tokenizer_length'])) - set(artifact['special_token_ids'])
            and set(eos_ids).issubset(artifact['special_token_ids']), "Incomplete tokenizer vocabulary coverage")
    require(len(set(regular_ids)) == len(regular_ids) and not set(eligible).intersection(excluded_ids)
            and sorted(set(eligible).union(excluded_ids)) == regular_ids
            and not set(eos_ids).intersection(regular_ids) and artifact['eos_token_ids'] == eos_ids,
            "Incomplete or overlapping token inventory")
    require(artifact['regular_token_count'] == len(regular_ids) and artifact['eligible_token_count'] == len(eligible)
            and artifact['excluded_token_count'] == len(excluded), "Token inventory counts differ")
    require(artifact['standalone_decode_stream_format'] == 'Sorted regular IDs; canonical JSON [token_id,decoded_text] plus LF',
            "Token decode hash format differs")
    decode_sha = artifact['standalone_decode_stream_sha256']
    require(isinstance(decode_sha, str) and len(decode_sha) == 64 and all(c in '0123456789abcdef' for c in decode_sha),
            "Invalid token decode stream hash")
    if raw_decoder is not None:
        observed_eligible, observed_excluded, stream_hash = [], {}, hashlib.sha256()
        for token in regular_ids:
            raw = raw_decoder([token])
            require(isinstance(raw, str), "Raw standalone decoder returned non-text")
            stream_hash.update(canonical([token, raw]) + b'\n')
            if not raw:
                observed_excluded[str(token)] = 'empty_standalone_decode'
            elif '\ufffd' in raw:
                observed_excluded[str(token)] = 'replacement_character_in_standalone_decode'
            else:
                observed_eligible.append(token)
        require(eligible == observed_eligible and excluded == observed_excluded
                and decode_sha == stream_hash.hexdigest(), "Independently decoded token inventory differs")
    return frozenset(eligible), frozenset(eos_ids)


def replay_raw_token_prefixes(output_ids, data, eligible_ids, eos_ids, raw_decoder):
    """Independently cross-check LMFE emissions against unmodified tokenizer decoding."""
    regular_output, current_word, emitted, ended = [], [], '', False
    for token in output_ids:
        require(not ended, "Token appears after EOS")
        if token in eos_ids:
            require(raw_decoder(regular_output) == emitted, "Full decode differs at EOS")
            ended = True
            continue
        require(token in eligible_ids, "Non-EOS token excluded by independently checked inventory")
        require(token in data.tokenizer_tree.tokens_to_strs, "Non-EOS token missing from regular vocabulary")
        if token in data.tokenizer_tree.new_word_tokens:
            current_word, addition = [token], data.tokenizer_tree.tokens_to_strs[token]
        else:
            before = data.decoder(current_word)
            current_word = current_word + [token]
            after = data.decoder(current_word)
            require(after.startswith(before), "LMFE transition is not prefix stable")
            addition = after[len(before):]
        require(bool(addition), "Non-EOS transition emits no characters")
        regular_output.append(token)
        emitted += addition
        require(raw_decoder(regular_output) == emitted, "Raw full-prefix decoding differs from LMFE emissions")
    return emitted


def verify_effective_constraint(receipt, configuration, judgment_schema, expected_guard_hash):
    """Check a hash-bound receipt without trusting runner helper functions."""
    require(set(receipt) == {"schema", "alphabet", "configuration"}
            and receipt["schema"] == "third-effective-constraint-config-mistral-v1"
            and receipt["configuration"] == configuration, "Effective constraint receipt differs")
    alphabet = receipt["alphabet"]
    require(isinstance(alphabet, str) and bool(alphabet), "Missing effective tokenizer alphabet")
    require(configuration["alphabet_sha256"] == hashlib.sha256(alphabet.encode("utf-8")).hexdigest()
            and configuration["alphabet_length"] == len(alphabet), "Effective alphabet binding differs")
    require(configuration["schema_sha256"] == digest(judgment_schema)
            and configuration["guard_source_sha256"] == expected_guard_hash, "Constraint schema/implementation differs")
    require(configuration["max_consecutive_whitespaces"] == 12 and configuration["force_json_field_order"] is False
            and configuration["max_json_array_length"] == 8 and configuration["reason_word_limit"] == 80
            and configuration["effective_config_verified"] is True and configuration["guard_shortcut_allowed"] is False
            and configuration["base_proposal_shortcut_allowed"] is True
            and configuration["word_count_semantics"] == "Python str.split on decoded reason value"
            and configuration["candidate_selection"] == "Descending finite proposal score, then lowest token ID; first passing guarded transition"
            and configuration["non_prefix_stable_token_transition"] == "reject"
            and configuration["logit_argmax_evidence"] == "runtime-attested; no saved full logits"
            and configuration["standalone_token_policy"] == TOKEN_POLICY
            and configuration["standalone_token_policy_sha256"] == digest(TOKEN_POLICY),
            "Effective parser limits differ")
    return alphabet


def verify_mistral_runtime(runtime):
    """Check native model capacity separately from the smaller execution cap."""
    require(runtime["schema"] == "third-judge-runtime-mistral-v1", "Wrong Mistral runtime schema")
    require(runtime["model_native_position_limit"] == 131072
            and runtime["execution_total_token_budget"] == 32768, "Native/execution context limits differ")
    require(runtime["model_configuration_checked"] == MISTRAL_CONFIG, "Pinned Mistral model configuration differs")
    require(runtime["model_class"] == "transformers.models.mistral.modeling_mistral.MistralForCausalLM"
            and runtime["tokenizer_class"] == "transformers.tokenization_utils_fast.PreTrainedTokenizerFast"
            and runtime["model_parameter_bytes"] == 24495564800, "Pinned Mistral model/tokenizer/parameter bytes differ")


def audit(root, repo, source_workspace=None, tokenizer_path=None):
    require(importlib.metadata.version("lm-format-enforcer") == "0.11.3", "Audit requires pinned LMFE 0.11.3")
    bindings = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        observed = sha(path)
        require(expected is None or observed == expected, "Artifact SHA256 mismatch: " + path.name)
        prior = bindings.get(str(path))
        require(prior is None or prior == observed, "Artifact changed while auditing: " + path.name)
        bindings[str(path)] = observed
        return observed
    def read(path, expected=None):
        bind(path, expected)
        return strict(Path(path).read_bytes().decode("utf-8"))
    def lines(path, expected=None):
        bind(path, expected)
        result = []
        with Path(path).open("rb") as stream:
            for line in stream:
                require(bool(line.strip()), "Blank JSONL physical line: " + Path(path).name)
                result.append(strict(line.decode("utf-8")))
        return result
    def package_path(base, name):
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe package path")
        return base / relative
    def grammar_accepts(raw):
        try:
            bounded_reason.replay(raw, judgment_schema, effective_alphabet)
            return True
        except ValueError:
            return False
    pre, out = root / "preparation", root / "inference"
    m = read(pre / "manifest.json")
    require(m["schema"] == "third-judge-pilot-launch-mistral-v1" and m["source_role"] == "engineering_only", "Wrong launch scope/version")
    require(m["expected_pairs"] == 40 and m["expected_calls"] == 44, "Wrong planned denominator")
    require(m["model_id"] == MODEL_ID and m["model_revision"] == MODEL_REVISION, "Wrong pinned second judge")
    require(read(out / "launch_manifest.private.json") == m, "Launch manifests differ")
    summary = read(out / "summary.json")
    require(summary["completed"] is True and summary["fatal_error"] is None, "Run did not complete without a fatal error")
    require(not (out / "FAILED.json").exists(), "Completed run contains FAILED marker")
    require(summary["schema"] == "third-model-annotation-technical-pilot-summary-mistral-v1", "Wrong summary version")
    runtime_dir = repo / "local_run/model_annotation_runtime_mistral_v1_20261011"
    amendment = read(runtime_dir / "runtime_plan.json", m["runtime_plan_sha256"])
    require(amendment["expected_input_sha256"] == m["inputs_sha256"], "Amendment input binding differs")
    require(amendment["runtime_plan_schema"] == "third-judge-runtime-amendment-mistral-v1"
            and amendment["model_id"] == MODEL_ID and amendment["model_revision"] == MODEL_REVISION
            and amendment["model_native_position_limit"] == 131072
            and amendment["technical_role"] == "second_cross_family_engineering_judge", "Wrong second-judge amendment")
    require(amendment["seed_plan_sha256"] == m["plan_sha256"], "Pair seed plan binding differs")
    require(amendment["max_new_tokens"] == 512 and amendment["total_token_budget"] == 32768 and amendment["format_retry_limit"] == 0, "Unexpected amendment limits")
    require(amendment["scheduling"] == SCHEDULE and amendment["attention_fallback_allowed"] is False, "Unexpected amendment execution policy")
    for name, expected in amendment["package_files"].items():
        bind(package_path(runtime_dir, name), expected)
    require({"run_pilot.py", "prepare_pilot.py", "requirements.txt", "pilot.sbatch", "bounded_reason.py", "audit_pilot_artifacts.py"}.issubset(amendment["package_files"]), "Incomplete package bindings")
    for name, expected in amendment["original_v2_folder_files"].items():
        bind(package_path(runtime_dir.parent / "model_annotation_runtime_v2_20261011", name), expected)
    require({"run_pilot.py", "prepare_pilot.py", "test_runner.py", "runtime_plan.json", "audit_pilot_artifacts.py"}.issubset(amendment["original_v3_folder_files"]), "Incomplete preserved v3 bindings")
    for name, expected in amendment["original_v3_folder_files"].items():
        bind(package_path(runtime_dir.parent / "model_annotation_runtime_v3_20261011", name), expected)
    require({"run_pilot.py", "prepare_pilot.py", "test_runner.py", "runtime_plan.json", "bounded_reason.py", "audit_pilot_artifacts.py"}.issubset(amendment["original_v4_folder_files"]), "Incomplete preserved v4 bindings")
    for name, expected in amendment["original_v4_folder_files"].items():
        bind(package_path(runtime_dir.parent / "model_annotation_runtime_v4_20261011", name), expected)
    require({"run_pilot.py", "prepare_pilot.py", "test_runner.py", "runtime_plan.json", "bounded_reason.py", "audit_pilot_artifacts.py"}.issubset(amendment["original_v5_folder_files"]), "Incomplete preserved v5 bindings")
    for name, expected in amendment["original_v5_folder_files"].items():
        bind(package_path(runtime_dir.parent / "model_annotation_runtime_v5_20261011", name), expected)
    require(amendment["package_files"]["bounded_reason.py"] == amendment["original_v5_folder_files"]["bounded_reason.py"], "Reused v5 guard differs")
    bind(Path(__file__).resolve().with_name("bounded_reason.py"), amendment["package_files"]["bounded_reason.py"])
    import bounded_reason
    bind(runtime_dir / "run_pilot.py", m["driver_sha256"])
    plan_path = repo / "local_run/model_annotation_v2_20261011/PLAN.json"
    plan = read(plan_path, m["plan_sha256"])
    require(any(model["model_id"] == m["model_id"] and model["revision"] == m["model_revision"] for model in plan["models"]), "Unpinned model identity")
    for name, expected in plan["package_files"].items():
        bind(package_path(plan_path.parent, name), expected)
    for name, expected in plan["protocol_bindings"].items():
        bind(package_path(repo, name), expected)
    judgment_schema = read(plan_path.parent / "model_judgment.schema.json")
    judgment_validator = Draft202012Validator(judgment_schema)
    response_validator = Draft202012Validator(read(plan_path.parent / "model_response.schema.json"))
    inputs = lines(pre / "inputs.jsonl", m["inputs_sha256"])
    responses = lines(out / "responses.jsonl")
    attempts = lines(out / "attempts.jsonl")
    require(len(inputs) == len(responses) == len(attempts) == 44, "Expected all 44 one-attempt outcomes")
    require(all(set(row) == {"pair_id", "text_a", "text_b", "orientation"} and all(isinstance(v, str) and v for v in row.values()) for row in inputs), "Unexpected model input fields/types")
    canonical_inputs = {r["pair_id"]: r for r in inputs if r["orientation"] == "canonical"}
    swaps = [r for r in inputs if r["orientation"] == "swapped_diagnostic"]
    require(len(canonical_inputs) == 40 and len(swaps) == 4 and len({r["pair_id"] for r in swaps}) == 4, "Expected 40 unique canonical pairs and four unique swaps")
    require(len({(r["pair_id"], r["orientation"]) for r in inputs}) == 44, "Repeated pair/orientation")
    for row in swaps:
        original = canonical_inputs[row["pair_id"]]
        require(row["text_a"] == original["text_b"] and row["text_b"] == original["text_a"], "Swapped full texts differ")
    preparation = read(pre / "preparation.json", m["preparation_sha256"])
    require(preparation["source_pack_seal_verified"] is True and preparation["new_human_responses"] == 0, "Unexpected preparation gates")
    mapping = read(pre / "private_source_mapping.json", preparation["source_mapping_sha256"])
    require(len(mapping) == 40, "Wrong source mapping count")
    recorded_source = Path(preparation["full_text_source"])
    workspace = Path(source_workspace).resolve() if source_workspace else recorded_source.parents[4]
    records_path = workspace / "prepared/askubuntu/a02/population/records.jsonl"
    source_path = workspace / "calibration/askubuntu/e5/mean_removed_a01/selection/private_sampling_manifest.json"
    source = read(source_path, preparation["source_pack_sha256"])
    require(source["sha256"] == digest({k: v for k, v in source.items() if k != "sha256"}, False), "Source pack canonical seal differs")
    records = lines(records_path, preparation["source_records_sha256"])
    frame = [{key: row[key] for key in ("record_id", "text")} for row in records if row["partition"] == "calibration"]
    require(digest(frame, False) == source["frame"]["records_sha256"] == preparation["ordered_calibration_frame_sha256"], "Full ordered source frame differs")
    texts = {row["record_id"]: row["text"] for row in frame}
    require(len(texts) == len(frame) and len(frame) * (len(frame) - 1) // 2 == source["pair_population"], "Source frame identity/population mismatch")
    pairs = source["pairs"]
    require(len(pairs) == 420 and len({p["pair_id"] for p in pairs}) == 420, "Wrong engineering source pack")
    native_pairs = [tuple(sorted([p["left_id"], p["right_id"]])) for p in pairs]
    require(len(set(native_pairs)) == 420 and all(a != b for a, b in native_pairs), "Repeated/self engineering pair")
    ranked = sorted(pairs, key=lambda p: (digest(["pilot-pair-v2", m["seed"], preparation["source_pack_sha256"], sorted([p["left_id"], p["right_id"]])]), p["pair_id"]))[:40]
    swap_ids = {p["pair_id"] for p in sorted(ranked, key=lambda p: digest(["pilot-swap-v2", m["seed"], preparation["source_pack_sha256"], p["pair_id"]]))[:4]}
    for index, (pair, item) in enumerate(zip(ranked, mapping)):
        native = sorted([pair["left_id"], pair["right_id"]])
        shown = native[::-1] if int(digest(["orientation-v2", m["plan_sha256"], native]), 16) % 2 else native
        opaque = digest(["pilot-opaque-v2", m["seed"], preparation["source_pack_sha256"], native])
        expected = {"pair_id": opaque, "source_pair_id": pair["pair_id"], "native_pair": native,
                    "shown_ids": shown, "source_bin": pair["bin"], "swapped_diagnostic": pair["pair_id"] in swap_ids}
        require(item == expected, "Source selection/mapping mismatch")
        require(inputs[index] == {"pair_id": opaque, "text_a": texts[shown[0]], "text_b": texts[shown[1]], "orientation": "canonical"}, "Pilot full text differs from source records")
    require({r["pair_id"] for r in swaps} == {i["pair_id"] for i in mapping if i["swapped_diagnostic"]}, "Wrong swapped sample")
    census = lines(pre / "engineering_frame_token_census.private.jsonl", m["engineering_frame_census_sha256"])
    census_summary = read(pre / "engineering_frame_token_census_summary.json")
    require(census_summary == preparation["engineering_frame_token_census"], "Engineering census receipts differ")
    require(len(census) == 840 and census_summary["rows"] == 840 and census_summary["census_sha256"] == m["engineering_frame_census_sha256"], "Wrong engineering census coverage")
    require(census_summary["execution_total_budget"] == 32768 and census_summary["model_native_position_limit"] == 131072
            and census_summary["output_reserve"] == 512, "Wrong census token limits")
    expected_census = [(p["pair_id"], o) for p in pairs for o in ("canonical", "swapped_diagnostic")]
    require([(x["source_pair_id"], x["orientation"]) for x in census] == expected_census, "Engineering census ordering/identity mismatch")
    for row in census:
        require(type(row["input_tokens"]) is int and row["input_tokens"] > 0, "Invalid census token count")
        require(row["fits_32768_with_512_output"] is (row["input_tokens"] + 512 <= 32768), "Incorrect census context flag")
    require(census_summary["max_input_tokens"] == max(x["input_tokens"] for x in census), "Wrong maximum census tokens")
    require(census_summary["over_execution_budget"] == sum(not x["fits_32768_with_512_output"] for x in census), "Wrong census overflow count")
    weights = read(pre / "weights_manifest.json", m["weights_manifest_sha256"])
    require(weights["model_id"] == m["model_id"] and weights["revision"] == m["model_revision"], "Weight manifest model identity differs")
    input_binding = read(out / "validated_input_binding.json")
    require(input_binding == {"inputs_sha256": m["inputs_sha256"], "rows": 44, "canonical_pairs": 40,
            "full_texts_preserved": True, "source_role": "engineering_only", "human_responses_created": 0}, "Validated input receipt differs")
    runtime = read(out / "runtime_manifest.json")
    runtime_sha = sha(out / "runtime_manifest.json")
    verify_mistral_runtime(runtime)
    require(runtime["status"] == "model_loaded_before_first_inference" and "A100" in runtime["gpu"] and runtime["visible_gpus"] == 1 and runtime["gpu_total_bytes"] >= 75 * 1024**3, "Unexpected GPU runtime")
    require(runtime["dtype"] == "bfloat16" and runtime["attention_implementation"] == "sdpa" and runtime["sdpa_backend"] == "FLASH_ATTENTION", "Unexpected attention/dtype")
    require(runtime["sdpa_flags_inside_generation_context"] == FLAGS and runtime["output_attentions"] is False, "Runtime permits attention fallback")
    require(runtime["deterministic_algorithms"] is True and runtime["allow_tf32"] is False, "Runtime determinism controls differ")
    require(runtime["model_native_position_limit"] >= 32768 and runtime["runtime_plan_sha256"] == m["runtime_plan_sha256"], "Native context/amendment mismatch")
    for name, version in amendment["required_packages"].items():
        require(runtime["packages"][name] == version, "Recorded package version mismatch: " + name)
    require(runtime["constraint"]["algorithm_revision"] == ALGORITHM and runtime["constraint"]["package_version"] == "0.11.3", "Unexpected grammar runtime")
    effective = read(out / "effective_constraint_config.json", runtime["effective_constraint_config_sha256"])
    constraint = runtime["constraint"]
    effective_alphabet = verify_effective_constraint(effective, constraint, judgment_schema,
        amendment["package_files"]["bounded_reason.py"])
    token_inventory = read(out / "token_eligibility.json", runtime["token_eligibility_sha256"])
    require(token_inventory['tokenizer_length'] == runtime['tokenizer_length']
            and token_inventory['special_token_ids'] == sorted(runtime['tokenizer_special_ids']), "Recorded token vocabulary metadata differs")
    require(token_inventory["eos_token_ids"] == [2], "Wrong Mistral EOS token policy")
    regular_ids = sorted(set(range(runtime['tokenizer_length'])) - set(runtime['tokenizer_special_ids']))
    eligible_ids, eos_ids = verify_token_eligibility(token_inventory, regular_ids, token_inventory['eos_token_ids'])
    require(runtime['eligible_regular_token_count'] == len(eligible_ids)
            and runtime['excluded_regular_token_count'] == token_inventory['excluded_token_count'], "Runtime token eligibility counts differ")
    implementation_files_checked = 0
    for path, expected in runtime["implementation_file_bindings"].items():
        if Path(path).is_file():
            bind(path, expected)
            implementation_files_checked += 1
    generation = read(out / "generation_config.json", runtime["generation_config_sha256"])
    require(generation["do_sample"] is False and generation["num_beams"] == 1 and generation["max_new_tokens"] == 512
            and generation["use_cache"] is True and generation["repetition_penalty"] == 1.0
            and generation["output_attentions"] is False and generation["forced_bos_token_id"] is None
            and generation["forced_eos_token_id"] is None and generation["remove_invalid_values"] is False
            and generation["renormalize_logits"] is False and generation["num_return_sequences"] == 1,
            "Generation config differs from fixed runtime")
    require(generation["bos_token_id"] == 1 and generation["eos_token_id"] == 2, "Mistral generation BOS/EOS differs")
    decoding = read(out / "decoding_config.json")
    require(decoding["max_new_tokens"] == 512 and decoding["total_token_budget_including_output_reserve"] == 32768 and decoding["format_only_retry_limit"] == 0, "Wrong decoding limits")
    planned_constraint = dict(constraint)
    planned_constraint.update(alphabet_sha256=None, alphabet_length=None, effective_config_verified=False)
    require(decoding["truncation"] is False and decoding["sdpa_math_mem_efficient_cudnn_fallback"] is False
            and decoding["constraint"] == planned_constraint, "Planned/effective decoding constraints differ")
    require(decoding["do_sample"] is False and decoding["num_beams"] == 1 and decoding["batch_size"] == 1
            and decoding["dtype"] == "bfloat16" and decoding["attention_implementation"] == "sdpa"
            and decoding["sdpa_backend"] == "FLASH_ATTENTION" and decoding["output_attentions"] is False,
            "Decoding backend or sampling configuration differs")
    prompt = read(out / "prompt_manifest.json")
    system = (plan_path.parent / "judge_system_prompt.txt").read_text()
    task = (plan_path.parent / "judge_task_definition.txt").read_text()
    require(prompt["system_prompt"] == system and prompt["task_definition"] == task and prompt["format_only_retry_limit"] == 0 and prompt["pair_ids_and_provenance_shown_to_model"] is False, "Prompt manifest differs")
    schedule = read(out / "schedule.json", summary["schedule_sha256"])
    require(schedule["schema"] == "third-pilot-input-schedule-mistral-v1", "Wrong second-judge schedule version")
    require(schedule["policy"] == SCHEDULE and schedule["inputs_sha256"] == m["inputs_sha256"] and schedule["all_inputs_preserved"] is True and schedule["outcome_based_selection"] is False, "Wrong scheduling contract")
    sc = schedule["input_census"]
    require(len(sc) == 44 and [x["input_index"] for x in sc] == list(range(44)), "Wrong scheduled input census")
    order = sorted(range(44), key=lambda i: (-sc[i]["input_tokens"], i))
    require(schedule["execution_order_original_indexes"] == order, "Schedule is not longest-first with fixed ties")
    require([a["input_index"] for a in attempts] == order, "Attempts do not follow sealed schedule")
    call_ids = {digest([m["pilot_id"], i, r["pair_id"], r["orientation"]]): i for i, r in enumerate(inputs)}
    require(len(call_ids) == 44 and {r["call_id"] for r in responses} == set(call_ids), "Missing/duplicate response call identity")
    require([call_ids[r["call_id"]] for r in responses] == order, "Responses do not follow scheduled original indexes")
    require({p.name for p in (out / "calls").iterdir()} == {"call-%04d" % i for i in range(44)}, "Missing or extra call artifacts")
    tokenizer = tokenizer_data = token_policy = None
    independently_checked_assets = 0
    if tokenizer_path is not None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        for asset in weights["files"]:
            path = package_path(Path(tokenizer_path), asset["relative_path"])
            if path.is_file() and not asset["relative_path"].endswith(".safetensors"):
                require(path.stat().st_size == asset["size_bytes"], "Cached tokenizer/config size mismatch")
                bind(path, asset["sha256"])
                independently_checked_assets += 1
        required = {"tokenizer.json", "tokenizer_config.json", "config.json"}
        require(required.issubset({x["relative_path"] for x in weights["files"]}), "Tokenizer files not bound")
        for name in required:
            require((Path(tokenizer_path) / name).is_file(), "Required tokenizer asset missing")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False, use_fast=True)
        require(type(tokenizer).__name__ == "PreTrainedTokenizerFast" and tokenizer.bos_token_id == 1
                and tokenizer.eos_token_id == 2 and tokenizer.clean_up_tokenization_spaces is False, "Pinned Mistral tokenizer differs")
        require(hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() == runtime["tokenizer_backend_json_sha256"], "Tokenizer backend differs")
        require(hashlib.sha256(canonical(tokenizer.chat_template)).hexdigest() == runtime["chat_template_sha256"], "Chat template differs")
        tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
        require(tokenizer_data.tokenizer_alphabet == effective_alphabet, "Independently reconstructed tokenizer alphabet differs")
        require(len(tokenizer) == runtime['tokenizer_length'] and sorted(tokenizer.all_special_ids) == token_inventory['special_token_ids']
                and sorted(tokenizer_data.tokenizer_tree.tokens_to_strs) == regular_ids, "Independent tokenizer vocabulary differs")
        independently_decoded_eos = tokenizer_data.eos_token_id if isinstance(tokenizer_data.eos_token_id, list) else [tokenizer_data.eos_token_id]
        def raw_decoder(tokens):
            return tokenizer.decode(list(tokens), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        eligible_ids, eos_ids = verify_token_eligibility(token_inventory, regular_ids, independently_decoded_eos, raw_decoder)
        token_policy = bounded_reason.StandaloneTokenPolicy(eligible_ids, eos_ids, raw_decoder)
    def messages(a, b):
        return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps({"task_definition": task,
            "documents": [{"label": "A", "text": a}, {"label": "B", "text": b}]}, ensure_ascii=False)}]
    if tokenizer is not None:
        for pair_index, pair in enumerate(pairs):
            native = sorted([pair["left_id"], pair["right_id"]])
            if int(digest(["orientation-v2", m["plan_sha256"], native]), 16) % 2:
                native.reverse()
            for offset, shown in enumerate((native, native[::-1])):
                rendered = tokenizer.apply_chat_template(messages(texts[shown[0]], texts[shown[1]]), tokenize=False, add_generation_prompt=True)
                tokens = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
                require(len(tokens) == census[2 * pair_index + offset]["input_tokens"], "Independent engineering token census differs")
    raw_diagnostics, canonical_responses = Counter(), {}
    for response, metric in zip(responses, attempts):
        index = call_ids[response["call_id"]]
        row, census_row = inputs[index], sc[index]
        check_schema(response, response_validator, "Response schema invalid")
        require(response["pair_id"] == row["pair_id"] and response["orientation"] == row["orientation"], "Response/input identity differs")
        require(response["model_id"] == m["model_id"] and response["model_revision"] == m["model_revision"] and response["weights_manifest_sha256"] == m["weights_manifest_sha256"], "Response model bindings differ")
        require(response["input_sha256"] == digest(row) == census_row["input_sha256"], "Response input hash differs")
        for field, file in (("runtime_manifest_sha256", "runtime_manifest.json"), ("prompt_sha256", "prompt_manifest.json"), ("decoding_config_sha256", "decoding_config.json")):
            require(response[field] == sha(out / file), "Response provenance hash differs: " + field)
        require(metric["input_index"] == index and metric["call_id"] == response["call_id"] and metric["pair_id"] == row["pair_id"] and metric["orientation"] == row["orientation"], "Attempt identity differs")
        require(metric["attempt"] == 0 and metric["format_only_retry"] is False and metric["format_only_retry_limit"] == 0, "Unexpected retry")
        require(metric["human_completed"] is False and metric["status"] == response["status"] and metric["constraint_algorithm_revision"] == ALGORITHM, "Attempt status/provenance differs")
        require(metric["effective_constraint_config_sha256"] == runtime["effective_constraint_config_sha256"], "Attempt effective constraint binding differs")
        require(metric["token_eligibility_sha256"] == runtime["token_eligibility_sha256"], "Attempt token eligibility binding differs")
        require(response["status"] in {"valid", "invalid_json", "context_overflow", "runtime_error"}, "Status is not produced by the pinned runner")
        seed = int(hashlib.sha256(f'{m["plan_sha256"]}|{row["pair_id"]}|{m["model_id"]}'.encode()).hexdigest(), 16) % (2**63)
        require(metric["seed"] == seed, "Incorrect per-pair seed")
        call = out / "calls" / ("call-%04d" % index)
        expected_files = {"response.json", "attempt-0.metrics.json", "attempt-0.messages.private.json", "attempt-0.input_tokens.private.json"}
        if "guard_steps_sha256" in metric:
            expected_files.add("attempt-0.guard_steps.private.json")
        if response["raw_output_sha256"] is not None:
            expected_files |= {"attempt-0.raw.private.txt", "attempt-0.output_tokens.private.json"}
        require({p.name for p in call.iterdir()} == expected_files and all(p.is_file() for p in call.iterdir()), "Missing or extra per-call artifacts")
        require(read(call / "response.json") == response and read(call / "attempt-0.metrics.json") == metric, "Per-call records differ")
        guard_steps = None
        if "guard_steps_sha256" in metric:
            guard_steps = read(call / "attempt-0.guard_steps.private.json", metric["guard_steps_sha256"])
            require(isinstance(guard_steps, list) and len(guard_steps) == metric["guard_steps"] <= 512, "Guard step accounting differs")
            rejected = Counter()
            for step_index, step in enumerate(guard_steps):
                require(set(step) == {"step", "accepted_token_id", "finite_native_candidates", "candidates_considered", "rejected_candidates", "prefix_mask_verified", "selection_seconds"}, "Guard ledger fields differ")
                failed_selection = step["accepted_token_id"] is None
                if failed_selection:
                    require(response["status"] == "runtime_error" and step_index == len(guard_steps) - 1
                            and metric.get("exception_type") == "CandidateExhaustedError"
                            and step["candidates_considered"] == step["finite_native_candidates"], "Invalid failed guard selection state")
                else:
                    require(type(step["accepted_token_id"]) is int and step["accepted_token_id"] >= 0, "Invalid selected token ID")
                require(step["step"] == step_index and type(step["finite_native_candidates"]) is int and type(step["candidates_considered"]) is int
                        and (0 if failed_selection else 1) <= step["candidates_considered"] <= step["finite_native_candidates"]
                        and step["prefix_mask_verified"] is True, "Invalid guard candidate/selection state")
                require(isinstance(step["rejected_candidates"], dict) and all(isinstance(k, str) and k and type(v) is int and v > 0 for k, v in step["rejected_candidates"].items()), "Invalid guard rejection counts")
                require(sum(step["rejected_candidates"].values()) == step["candidates_considered"] - (0 if failed_selection else 1), "Guard rejected candidate denominator differs")
                require(type(step["selection_seconds"]) in (int, float) and math.isfinite(step["selection_seconds"]) and step["selection_seconds"] >= 0, "Invalid guard elapsed time")
                rejected.update(step["rejected_candidates"])
            require(metric["guard_candidates_considered"] == sum(x["candidates_considered"] for x in guard_steps)
                    and metric["guard_rejected_candidates"] == dict(rejected)
                    and metric["guard_selection_seconds"] == sum(x["selection_seconds"] for x in guard_steps), "Guard aggregate accounting differs")
        expected_messages = messages(row["text_a"], row["text_b"])
        require(read(call / "attempt-0.messages.private.json") == expected_messages, "Full model input messages differ")
        input_ids = read(call / "attempt-0.input_tokens.private.json")
        require(all(type(x) is int and x >= 0 for x in input_ids), "Invalid input token IDs")
        require(len(input_ids) == metric["input_tokens"] == census_row["input_tokens"], "Input token counts differ")
        require(metric["full_input_tokenization"] is True and metric["truncation_applied"] is False, "Truncated/unmeasured model input")
        require(metric["rendered_prompt_sha256"] == census_row["rendered_prompt_sha256"], "Rendered prompt hash differs")
        require(census_row["orientation"] == row["orientation"] and census_row["context_fit_with_output_reserve"] is (len(input_ids) + 512 <= 32768), "Schedule input metadata differs")
        if tokenizer is not None:
            rendered = tokenizer.apply_chat_template(expected_messages, tokenize=False, add_generation_prompt=True)
            require(hashlib.sha256(rendered.encode()).hexdigest() == metric["rendered_prompt_sha256"], "Independent prompt rendering differs")
            require(tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"] == input_ids, "Independent full input tokens differ")
        if "sdpa_flags_during_generation" in metric:
            require(metric["sdpa_flags_during_generation"] == FLAGS, "Attention backend fallback recorded")
        if "generation_seconds" in metric:
            require(guard_steps is not None, "Generation attempt lacks bound guard ledger")
        if response["status"] == "context_overflow":
            require(len(input_ids) + 512 > 32768 and "output_tokens" not in metric, "Incorrect overflow state")
            require(response["raw_output_sha256"] is None and not {"generation_seconds", "sdpa_flags_during_generation", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"}.intersection(metric), "Overflow state contains generation evidence")
            require(guard_steps is None, "Overflow state contains guard selection evidence")
        if response["status"] == "invalid_json":
            require(response["raw_output_sha256"] is not None and isinstance(metric.get("format_error"), str) and bool(metric["format_error"]), "Invalid JSON lacks raw artifacts or failure evidence")
        if response["status"] == "runtime_error":
            require(isinstance(metric.get("exception_type"), str) and bool(metric["exception_type"]) and isinstance(metric.get("exception_message"), str), "Runtime failure lacks exception evidence")
        raw_path, tokens_path = call / "attempt-0.raw.private.txt", call / "attempt-0.output_tokens.private.json"
        if response["raw_output_sha256"] is not None:
            require(len(input_ids) + 512 <= 32768, "Generated after context overflow")
            require(metric.get("sdpa_flags_during_generation") == FLAGS, "Generated without flash-only evidence")
            bind(raw_path, response["raw_output_sha256"])
            require(metric["raw_output_sha256"] == response["raw_output_sha256"], "Raw hash differs from metrics")
            raw = raw_path.read_bytes().decode("utf-8")
            output_ids = read(tokens_path)
            require(all(type(x) is int and x >= 0 for x in output_ids) and len(output_ids) == metric["output_tokens"] <= 512, "Output token accounting differs")
            require(guard_steps is not None and [x["accepted_token_id"] for x in guard_steps] == output_ids, "Output tokens differ from guarded selections")
            require(metric["reached_output_token_cap"] is (len(output_ids) >= 512), "Incorrect token-cap flag")
            inventory_ok = all(token in eligible_ids or token in eos_ids for token in output_ids)
            raw_diagnostics['bound_token_inventory_pass' if inventory_ok else 'bound_token_inventory_fail'] += 1
            token_ok = None if inventory_ok else False
            if tokenizer is not None:
                require(tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) == raw, "Independent token decoding differs from raw bytes")
                try:
                    independently_emitted = replay_raw_token_prefixes(output_ids, tokenizer_data, eligible_ids, eos_ids, raw_decoder)
                    require(independently_emitted == raw, "Independent cumulative raw output decode differs")
                    replayed_state = bounded_reason.replay_guard_tokens(input_ids, output_ids, judgment_schema, tokenizer_data, token_policy)
                    require(replayed_state.emitted_text == raw, "Independent guarded/full raw decode mismatch")
                    token_ok = True
                except ValueError:
                    token_ok = False
                raw_diagnostics["independent_token_filter_pass" if token_ok else "independent_token_filter_fail"] += 1
            grammar_ok = grammar_accepts(raw)
            raw_diagnostics["independent_character_grammar_pass" if grammar_ok else "independent_character_grammar_fail"] += 1
            try:
                parsed = strict(raw)
                schema_ok = not list(judgment_validator.iter_errors(parsed)) and len(parsed["reason"].split()) <= 80
            except (ValueError, TypeError, KeyError):
                parsed, schema_ok = None, False
            raw_diagnostics["independent_final_schema_pass" if schema_ok else "independent_final_schema_fail"] += 1
            if response["status"] == "valid":
                require(grammar_ok and schema_ok and token_ok is not False, "Recorded valid output fails independent checks")
                require(metric.get("generated_token_filter_replay_passed") is True and metric.get("decoded_character_grammar_replay_passed") is True, "Valid output lacks replay evidence")
                require(metric.get("full_output_prefix_decode_consistency_passed") is True, "Valid output lacks full-prefix decode consistency evidence")
                require(parsed == response["judgment"], "Response judgment differs from raw JSON")
            elif response["status"] == "invalid_json":
                require(isinstance(metric.get("format_error"), str) and bool(metric["format_error"]), "Invalid output lacks failure reason")
                all_pass = schema_ok and grammar_ok and (token_ok if token_ok is not None else metric.get("generated_token_filter_replay_passed") is True)
                require(not all_pass, "Invalid status conflicts with every independent/attested validation")
        else:
            require(response["status"] != "valid", "Valid response lacks raw output")
            require(not raw_path.exists() and not tokens_path.exists() and "output_tokens" not in metric, "Unbound raw/token output exists")
        if response["orientation"] == "canonical":
            canonical_responses[row["pair_id"]] = response
    diagnostics = Counter()
    for response in responses:
        if response["orientation"] != "swapped_diagnostic": continue
        original = canonical_responses[response["pair_id"]]
        if original["status"] != "valid" or response["status"] != "valid":
            diagnostics["unavailable_comparison"] += 1
        else:
            a, b = original["judgment"]["category"], response["judgment"]["category"]
            positive = {"exact_copy", "substantially_same_meaning"}
            diagnostics["valid_comparison"] += 1
            diagnostics["category_changed"] += a != b
            diagnostics["binary_positive_changed"] += (a in positive) != (b in positive)
    require(dict(diagnostics) == summary["swapped_order_diagnostics"], "Swapped diagnostics differ")
    counts = dict(Counter(x["status"] for x in responses))
    canonical_counts = dict(Counter(x["status"] for x in canonical_responses.values()))
    require(counts == summary["status_counts_all_calls"] and canonical_counts == summary["status_counts_canonical"], "Status counts differ")
    require(dict(Counter(x["judgment"]["category"] for x in canonical_responses.values() if x["status"] == "valid")) == summary["canonical_category_counts"], "Category counts differ")
    require(summary["planned_input_rows"] == summary["recorded_input_rows"] == 44 and summary["canonical_pairs"] == 40 and summary["format_retry_limit"] == 0, "Summary denominator/retry mismatch")
    require(summary["runtime_plan_sha256"] == m["runtime_plan_sha256"], "Summary amendment binding differs")
    for key in ("diagnostic_calls_are_extra_votes", "human_completed", "v1_human_gate_satisfied", "threshold_selected", "model_panel_calibration_passed", "confirmatory_study_ready"):
        require(summary[key] is False, "Scientific/human gate incorrectly promoted")
    require(summary["human_responses_created"] == 0, "Human response count changed")
    accounting = {"attempts": 44, "format_retry_attempts": 0,
        "total_attempt_seconds": sum(x.get("seconds", 0.0) for x in attempts),
        "generation_seconds": sum(x.get("generation_seconds", 0.0) for x in attempts),
        "full_input_tokens_across_attempts": sum(x.get("input_tokens", 0) for x in attempts),
        "output_tokens_across_attempts": sum(x.get("output_tokens", 0) for x in attempts),
        "peak_gpu_allocated_bytes": max(x.get("peak_gpu_allocated_bytes", 0) for x in attempts),
        "peak_gpu_reserved_bytes": max(x.get("peak_gpu_reserved_bytes", 0) for x in attempts)}
    require(summary["attempt_accounting"] == accounting, "Attempt aggregate accounting differs")
    for path, expected in bindings.items():
        require(sha(path) == expected, "Bound file changed during audit: " + Path(path).name)
    return {"schema": "third-pilot-independent-artifact-audit-mistral-v1", "passed": True,
        "scope": "Completed-run provenance and full source reconstruction; failed model outputs remain failures; not semantic correctness or human validity",
        "created_at": datetime.now(timezone.utc).isoformat(), "pairs": 40, "scheduled_calls": 44,
        "attempts": 44, "status_counts": counts, "canonical_status_counts": canonical_counts,
        "swapped_order_diagnostics": dict(diagnostics), "engineering_census_rows": 840,
        "engineering_census_max_input_tokens": census_summary["max_input_tokens"],
        "engineering_census_over_execution_budget": census_summary["over_execution_budget"],
        "model_native_position_limit": 131072, "execution_total_token_budget": 32768,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "independent_full_source_text_reconstruction": True,
        "independent_character_grammar_replay": True, "independent_tokenizer_and_token_filter_replay": tokenizer is not None,
        "independent_raw_standalone_vocabulary_reconstruction": tokenizer is not None,
        "independent_nonempty_full_decode_transition_checks": tokenizer is not None,
        "eligible_regular_token_count": len(eligible_ids), "excluded_regular_token_count": token_inventory['excluded_token_count'],
        "token_policy_scope": "Restricted standalone-decoding vocabulary; split-codepoint tokens and any token containing U+FFFD (including literal replacement characters) are excluded",
        "character_replay_alphabet_evidence": "independently reconstructed from bound tokenizer" if tokenizer is not None else "hash-bound runtime alphabet receipt; tokenizer derivation not independently checked",
        "grammar_replay_implementation": "frozen bounded_reason module replayed independently of runner; raw strict JSON/schema/decoded word count checked separately",
        "guard_selection_evidence": "Selected token ledger checked; native-proposal/guard eligibility independently replayed only when tokenizer supplied; score argmax is runtime-attested because full logits are not retained",
        "token_filter_evidence_without_optional_tokenizer": "runner-attested replay fields only; no independent token-filter claim",
        "independently_checked_tokenizer_config_assets": independently_checked_assets,
        "runtime_implementation_files_independently_checked": implementation_files_checked,
        "runtime_implementation_files_recorded": len(runtime["implementation_file_bindings"]),
        "weights_evidence": "Runner-verified weight manifest and hashes are bound; this audit rehashes tokenizer/config assets when supplied, not full weight shards",
        "raw_validation_diagnostics": dict(raw_diagnostics), "attempt_accounting": accounting,
        "max_input_tokens": max(x["input_tokens"] for x in attempts),
        "max_generated_input_tokens": max((x["input_tokens"] for x in attempts if "output_tokens" in x), default=0),
        "all_44_outputs_valid": counts == {"valid": 44}, "labels_changed": False,
        "new_inference_calls": 0, "human_responses_created": 0, "confirmatory_study_ready": False,
        "audit_script_sha256": sha(__file__), "source_bindings": bindings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-workspace", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.root / "independent_artifact_audit.json"
    require(not destination.exists(), "Audit report already exists; choose a new --output")
    try:
        result = audit(args.root.resolve(), args.repo.resolve(), args.source_workspace, args.tokenizer_path)
    except Exception as error:
        result = {"schema": "third-pilot-independent-artifact-audit-mistral-v1", "passed": False,
            "error_type": type(error).__name__, "error": str(error), "labels_changed": False,
            "new_inference_calls": 0, "created_at": datetime.now(timezone.utc).isoformat()}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    destination.chmod(0o600)
    print(json.dumps({k: v for k, v in result.items() if k != "source_bindings"}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
