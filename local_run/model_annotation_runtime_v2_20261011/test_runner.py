"""Synthetic software fixtures for the engineering runner; no empirical labels/GPU."""
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

import run_pilot as runner


PACKAGE = Path(__file__).resolve().parent.parent / "model_annotation_v2_20261011"
JUDGMENT_SCHEMA = json.loads((PACKAGE / "model_judgment.schema.json").read_text())
RESPONSE_SCHEMA = json.loads((PACKAGE / "model_response.schema.json").read_text())
JUDGMENT = {"category": "unrelated", "distinction_types": ["assertion"],
            "reason": "Synthetic fixture texts express different assertions."}


def response():
    return {"schema": "third-model-response-v2", "annotator_type": "model", "human_completed": False,
            "pair_id": "software-fixture", "call_id": "software-call", "model_id": "software-model",
            "model_revision": "a" * 40, "weights_manifest_sha256": "b" * 64,
            "prompt_sha256": "c" * 64, "input_sha256": "d" * 64,
            "runtime_manifest_sha256": "e" * 64, "decoding_config_sha256": "f" * 64,
            "orientation": "canonical", "status": "valid", "judgment": copy.deepcopy(JUDGMENT),
            "raw_output_sha256": "0" * 64, "created_at": "2026-10-11T12:00:00+00:00"}


def rows(count=10, swaps=1):
    result = [{"pair_id": "fixture-" + str(i), "text_a": "Synthetic A " + str(i),
               "text_b": "Synthetic B " + str(i), "orientation": "canonical"} for i in range(count)]
    result += [{**r, "text_a": r["text_b"], "text_b": r["text_a"],
                "orientation": "swapped_diagnostic"} for r in result[:swaps]]
    return result


class InputTests(unittest.TestCase):
    def load(self, values, expected=None, max_pairs=40):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return runner.load_verified_inputs(path, expected or digest, max_pairs)

    def test_preserves_full_unicode_text_and_44_rows(self):
        values = rows(40, 4)
        text = "Full text\u2028not a physical line\u2029\nnext line café \U0001f600" + " tail" * 1000
        values[0]["text_a"] = text
        values[40]["text_b"] = text
        self.assertEqual(self.load(values), values)

    def test_hash_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load(rows(), "0" * 64)

    def test_input_changed_during_validation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows()))
            digest = runner.file_hash(path)
            with unittest.mock.patch.object(runner, "file_hash", side_effect=[digest, "0" * 64]):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    runner.load_verified_inputs(path, digest, 40)

    def test_repeated_pair_orientation_and_orphan_swaps_rejected(self):
        for values in (rows() + [rows()[0]], rows() + [rows()[-1]],
                       rows(10, 0) + [{**rows()[-1], "pair_id": "orphan"}]):
            with self.subTest(values=len(values)), self.assertRaises(ValueError):
                self.load(values)

    def test_wrong_swapped_text_and_too_many_swaps_rejected(self):
        incorrect = rows()
        incorrect[-1]["text_a"] += " altered"
        for values in (incorrect, rows(10, 2)):
            with self.subTest(values=len(values)), self.assertRaises(ValueError):
                self.load(values)

    def test_metadata_leak_fields_and_empty_text_rejected(self):
        for changes in ({"previous_judgment": "unrelated"}, {"score": 0.9}, {"text_a": ""},
                        {"orientation": "human"}, {"text_b": None}):
            values = rows()
            values[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load(values)

    def test_canonical_count_bound(self):
        for values in ([], rows(41, 0)):
            with self.subTest(count=len(values)), self.assertRaises(ValueError):
                self.load(values)


class JudgmentTests(unittest.TestCase):
    def test_valid_judgment_exact_preservation(self):
        self.assertEqual(runner.parse_judgment(json.dumps(JUDGMENT), JUDGMENT_SCHEMA), JUDGMENT)

    def test_strict_json_rejects_fences_trailing_duplicate_and_nonfinite(self):
        valid = json.dumps(JUDGMENT)
        invalid = ["```json\n" + valid + "\n```", valid + " trailing", valid + valid,
                   '{"category":"unrelated","category":"exact_copy","distinction_types":["none"],"reason":"x"}',
                   '{"category":"unrelated","distinction_types":["none"],"reason":NaN}',
                   '{"category":"unrelated","distinction_types":["none"],"reason":Infinity}']
        for raw in invalid:
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                runner.parse_judgment(raw, JUDGMENT_SCHEMA)

    def test_schema_rejects_extras_unknown_labels_duplicates_and_reason_limits(self):
        for changes in ({"confidence": 0.9}, {"human_completed": True}, {"category": "human_validated"},
                        {"distinction_types": ["none", "none"]}, {"distinction_types": []},
                        {"reason": "x" * 601}, {"reason": "x " * 81}, {"reason": ""}):
            value = {**JUDGMENT, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.validate_judgment(value, JUDGMENT_SCHEMA)


class ProvenanceTests(unittest.TestCase):
    def test_valid_model_envelope_and_missingness(self):
        self.assertEqual(runner.validate_response(response(), RESPONSE_SCHEMA), response())
        for status in ("invalid_json", "refusal", "context_overflow", "runtime_error"):
            value = response()
            value.update(status=status, judgment=None, raw_output_sha256=None)
            runner.validate_response(value, RESPONSE_SCHEMA)

    def test_human_status_provenance_and_conditional_judgment_enforced(self):
        for changes in ({"annotator_type": "human"}, {"human_completed": True}, {"human_completed": 0},
                        {"model_revision": "main"}, {"prompt_sha256": "not-a-hash"},
                        {"created_at": "2026-10-11T12:00:00"}, {"reviewer_id": "author"},
                        {"judgment": None}, {"raw_output_sha256": None}, {"status": "context_overflow"}):
            value = {**response(), **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.validate_response(value, RESPONSE_SCHEMA)

    def test_seed_determinism_and_binding(self):
        original = runner.pair_seed("a" * 64, "pair", "model")
        self.assertEqual(original, runner.pair_seed("a" * 64, "pair", "model"))
        self.assertTrue(0 <= original < 2**63)
        for args in (("b" * 64, "pair", "model"), ("a" * 64, "other", "model"),
                     ("a" * 64, "pair", "other")):
            self.assertNotEqual(original, runner.pair_seed(*args))

    def test_summary_keeps_swaps_separate_and_gates_false(self):
        canonical = response()
        swapped = {**response(), "orientation": "swapped_diagnostic"}
        swapped["judgment"]["category"] = "exact_copy"
        result = runner.summarize({"rows": [1, 2]}, [canonical, swapped], time.perf_counter())
        self.assertEqual(result["canonical_pairs"], 1)
        self.assertEqual(result["recorded_input_rows"], 2)
        self.assertEqual(result["swapped_order_diagnostics"]["binary_positive_changed"], 1)
        for field in ("diagnostic_calls_are_extra_votes", "human_completed", "v1_human_gate_satisfied",
                      "threshold_selected", "model_panel_calibration_passed", "confirmatory_study_ready"):
            self.assertIs(result[field], False)


class FakeTensor:
    def __init__(self, count):
        self.count = count
        self.shape = (1, count)

    def __getitem__(self, key):
        return self

    def tolist(self):
        return [1] * self.count

    def to(self, *_args):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self


class ExecutionContractTests(unittest.TestCase):
    def state(self, outputs, input_lengths):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **_kwargs: json.dumps(messages)
        tokenizer.side_effect = lambda *_args, **_kwargs: {"input_ids": FakeTensor(next(input_lengths))}
        tokenizer.decode.side_effect = outputs
        model = Mock()
        model.generate.return_value = FakeTensor(10)
        torch = Mock()
        torch.inference_mode.side_effect = nullcontext
        torch.cuda.max_memory_allocated.return_value = 0
        torch.cuda.max_memory_reserved.return_value = 0
        return {"manifest": {"pilot_id": "software-fixture", "plan_sha256": "a" * 64,
                "model_id": "software-model", "model_revision": "b" * 40,
                "weights_manifest_sha256": "c" * 64}, "system_prompt": "Synthetic fixture rubric",
                "task_definition": "Synthetic fixture only", "prompt_sha256": "d" * 64,
                "runtime_manifest_sha256": "e" * 64, "decoding_config_sha256": "f" * 64,
                "judgment_schema": JUDGMENT_SCHEMA, "response_schema": RESPONSE_SCHEMA,
                "torch": torch, "tokenizer": tokenizer, "model": model, "generation": object()}

    def execute(self, state):
        with tempfile.TemporaryDirectory() as temporary:
            with (Path(temporary) / "attempts.jsonl").open("w", encoding="utf-8") as attempts:
                result = runner.run_call(state, rows(1, 0)[0], 0, Path(temporary), attempts)
            saved = {p.name: json.loads(p.read_text()) for p in Path(temporary).rglob("*.messages.private.json")}
            return result, saved

    def test_context_overflow_never_generates_or_truncates(self):
        state = self.state([], iter([runner.TOTAL_TOKEN_BUDGET]))
        result, saved = self.execute(state)
        self.assertEqual(result["status"], "context_overflow")
        self.assertIsNone(result["judgment"])
        self.assertEqual(state["model"].generate.call_count, 0)
        self.assertEqual(len(saved), 1)
        self.assertIs(state["tokenizer"].call_args.kwargs["truncation"], False)

    def test_valid_first_attempt_does_not_retry(self):
        state = self.state([json.dumps(JUDGMENT)], iter([100]))
        result, saved = self.execute(state)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["judgment"], JUDGMENT)
        self.assertEqual(state["model"].generate.call_count, 1)
        self.assertEqual(len(saved), 1)

    def test_format_retry_preserves_raw_first_output_and_is_bounded(self):
        for last, expected in ((json.dumps(JUDGMENT), "valid"), ("invalid again", "invalid_json")):
            with self.subTest(expected=expected):
                state = self.state(["invalid first", last], iter([100, 120]))
                result, saved = self.execute(state)
                self.assertEqual(result["status"], expected)
                self.assertEqual(state["model"].generate.call_count, 2)
                second = saved["attempt-1.messages.private.json"]
                self.assertEqual(second[-2], {"role": "assistant", "content": "invalid first"})
                self.assertEqual(second[-1]["content"], runner.FORMAT_RETRY)

    def test_runtime_error_does_not_trigger_format_retry(self):
        state = self.state([], iter([100]))
        state["model"].generate.side_effect = RuntimeError("Synthetic software failure")
        result, saved = self.execute(state)
        self.assertEqual(result["status"], "runtime_error")
        self.assertIsNone(result["judgment"])
        self.assertEqual(state["model"].generate.call_count, 1)
        self.assertEqual(len(saved), 1)

    def test_format_retry_cannot_change_recoverable_category(self):
        for flaw in ({"extra_field": "synthetic"}, {"reason": "x " * 81}):
            first = json.dumps({**JUDGMENT, **flaw})
            for category, expected in ((JUDGMENT["category"], "valid"), ("exact_copy", "invalid_json")):
                with self.subTest(flaw=tuple(flaw), category=category):
                    state = self.state([first, json.dumps({**JUDGMENT, "category": category})], iter([100, 120]))
                    result, _saved = self.execute(state)
                    self.assertEqual(result["status"], expected)
                    self.assertEqual(state["model"].generate.call_count, 2)
                    if expected == "invalid_json":
                        self.assertIsNone(result["judgment"])


if __name__ == "__main__":
    unittest.main()
