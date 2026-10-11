"""Synthetic v4 software fixtures; real CPU grammar, mocked generation, no empirical labels/GPU."""
from contextlib import nullcontext, redirect_stdout
import copy
import hashlib
import io
import json
import itertools
import random
import math
from types import SimpleNamespace
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


class ActualGrammarTests(unittest.TestCase):
    """Exercise installed LMFE, with no tokenizer/model downloads or model calls."""
    def accepts(self, raw):
        from bounded_reason import replay
        try:
            replay(raw, copy.deepcopy(JUDGMENT_SCHEMA), GuardTests.ALPHABET)
            return True
        except ValueError:
            return False

    def test_all_categories_and_field_orders_representable(self):
        for category in JUDGMENT_SCHEMA["properties"]["category"]["enum"]:
            for reverse in (False, True):
                value = {**JUDGMENT, "category": category}
                if reverse:
                    value = dict(reversed(list(value.items())))
                with self.subTest(category=category, reverse=reverse):
                    self.assertTrue(self.accepts(json.dumps(value)))

    def test_actual_grammar_blocks_fences_duplicate_braces_unknown_categories_and_keys(self):
        valid = json.dumps(JUDGMENT)
        invalid = ["```json\n" + valid + "\n```", "{" + valid,
                   json.dumps({**JUDGMENT, "category": "invalid_category"}),
                   json.dumps({**JUDGMENT, "extra_field": "synthetic"}),
                   '{"category":"unrelated","category":"exact_copy","distinction_types":["none"],"reason":"x"}',
                   json.dumps({"category": "unrelated", "reason": "missing field"}), valid + valid]
        for raw in invalid:
            with self.subTest(raw=raw[:40]):
                self.assertFalse(self.accepts(raw))

    def test_grammar_length_bounds_and_incomplete_output(self):
        for changes in ({"reason": "x" * 601}, {"reason": ""}, {"distinction_types": []},
                        {"distinction_types": ["none"] * 9}):
            with self.subTest(changes=tuple(changes)):
                self.assertFalse(self.accepts(json.dumps({**JUDGMENT, **changes})))
        self.assertFalse(self.accepts(json.dumps(JUDGMENT)[:-1]))

    def test_final_validator_catches_constraints_not_enforced_by_grammar(self):
        # LMFE 0.11.3 permits duplicate array values. The independent final
        # validator remains necessary even with the prospective word guard.
        for changes in ({"distinction_types": ["none", "none"]},):
            raw = json.dumps({**JUDGMENT, **changes})
            with self.subTest(changes=tuple(changes)):
                self.assertTrue(self.accepts(raw))
                with self.assertRaises(ValueError):
                    runner.parse_judgment(raw, JUDGMENT_SCHEMA)
        self.assertFalse(self.accepts(json.dumps({**JUDGMENT, "reason": "x " * 81})))

    def test_actual_token_filter_rejects_multichar_structural_tokens_and_defers_eos(self):
        from lmformatenforcer import TokenEnforcerTokenizerData
        from bounded_reason import make_token_enforcer
        tokens = ["\n", "\r", "\t"] + [chr(n) for n in range(32, 127)]
        tokens += ["{{", "```json", '{"category":"wrong"', '{"category":"unrelated",']
        lookup = {text: index for index, text in enumerate(tokens)}
        eos, prompt = len(tokens), len(tokens) + 1
        data = TokenEnforcerTokenizerData([(i, text, False) for i, text in enumerate(tokens)],
            lambda ids: "".join(tokens[i] for i in ids), eos, False, len(tokens) + 2)
        enforcer = make_token_enforcer(copy.deepcopy(JUDGMENT_SCHEMA), data)
        sequence = [prompt]
        initial = enforcer.get_allowed_tokens(sequence).allowed_tokens
        self.assertIn(lookup["{"], initial)
        self.assertIn(lookup['{"category":"unrelated",'], initial)
        for forbidden in (eos, lookup["{{"], lookup["```json"], lookup['{"category":"wrong"']):
            self.assertNotIn(forbidden, initial)
        for character in json.dumps(JUDGMENT):
            allowed = enforcer.get_allowed_tokens(sequence).allowed_tokens
            self.assertIn(lookup[character], allowed)
            self.assertNotIn(eos, allowed)
            sequence.append(lookup[character])
        self.assertIn(eos, enforcer.get_allowed_tokens(sequence).allowed_tokens)

    def test_runner_character_replay_is_strict_and_complete(self):
        self.assertTrue(runner.replay_constraint(json.dumps(JUDGMENT), JUDGMENT_SCHEMA, GuardTests.ALPHABET))
        for raw in ("{" + json.dumps(JUDGMENT), "```json\n" + json.dumps(JUDGMENT) + "\n```",
                    json.dumps(JUDGMENT)[:-1]):
            with self.subTest(raw=raw[:20]), self.assertRaises(ValueError):
                runner.replay_constraint(raw, JUDGMENT_SCHEMA, GuardTests.ALPHABET)


class IndependentAuditTests(unittest.TestCase):
    def test_effective_receipt_alphabet_and_limit_tampering_rejected(self):
        import audit_pilot_artifacts as audit
        alphabet = 'abc 中虽😀'
        config = runner.constraint_description(JUDGMENT_SCHEMA, alphabet)
        receipt = {'schema': 'third-effective-constraint-config-v4', 'alphabet': alphabet, 'configuration': config}
        guard_hash = runner.file_hash(Path(runner.__file__).with_name('bounded_reason.py'))
        self.assertEqual(audit.verify_effective_constraint(receipt, config, JUDGMENT_SCHEMA, guard_hash), alphabet)
        for changes in ({'alphabet_sha256': '0' * 64}, {'alphabet_length': 1},
                {'effective_config_verified': False}, {'reason_word_limit': 81},
                {'guard_shortcut_allowed': True}, {'base_proposal_shortcut_allowed': False}, {'max_consecutive_whitespaces': 99},
                {'force_json_field_order': True}, {'max_json_array_length': 20},
                {'schema_sha256': '0' * 64}, {'guard_source_sha256': '0' * 64}):
            wrong = {**config, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                audit.verify_effective_constraint({**receipt, 'configuration': wrong}, wrong, JUDGMENT_SCHEMA, guard_hash)
        with self.assertRaises(ValueError):
            audit.verify_effective_constraint({**receipt, 'alphabet': alphabet + 'x'}, config, JUDGMENT_SCHEMA, guard_hash)


class GuardTests(unittest.TestCase):
    """CPU-only independent boundary fixtures; no original judgments or models."""
    SPACES = [chr(i) for i in range(0x10000) if chr(i).isspace()]
    ALPHABET = ''.join(dict.fromkeys(''.join(chr(i) for i in range(32, 127)) +
        '\n\r\t' + ''.join(SPACES) + '中虽😀\u200b\ufeff\u2060'))

    def test_matches_base_schema_intersection_with_decoded_word_bound(self):
        from bounded_reason import make_parser, exact_config
        from lmformatenforcer import JsonSchemaParser
        def accepts(raw, guarded):
            parser = make_parser(JUDGMENT_SCHEMA, self.ALPHABET) if guarded else JsonSchemaParser(
                JUDGMENT_SCHEMA, config=exact_config(self.ALPHABET))
            try:
                for c in raw:
                    if c not in parser.get_allowed_characters():
                        return False
                    parser = parser.add_character(c)
                return parser.can_end()
            except ValueError:
                return False
        # Escaped JSON whitespace must use decoded str.split semantics. Symbols
        # such as backspace, zero-width space and literal backslash+n are not separators.
        for separator, n, escaped, reverse in itertools.product(
                self.SPACES + ['\\n', '\b', '\u200b', '\ufeff', '\u2060', '😀', '"', '\\'],
                (79, 80, 81), (True, False), (True, False)):
            reason = separator.join(['a'] * n)
            value = {**JUDGMENT, 'reason': reason}
            if reverse:
                value = dict(reversed(list(value.items())))
            raw = json.dumps(value, ensure_ascii=escaped)
            with self.subTest(separator=repr(separator), n=n, escaped=escaped, reverse=reverse):
                self.assertEqual(accepts(raw, True), accepts(raw, False) and len(reason.split()) <= 80)

    def test_random_lexical_count_equals_decoded_python_split(self):
        from bounded_reason import LexicalState
        rng = random.Random(47)
        pool = ['a', 'b', '中', '😀', '"', '\\', '/', '\b', '\u200b', '\ufeff', '\u2060'] + self.SPACES
        for i in range(1000):
            reason = ''.join(rng.choice(pool) for _ in range(rng.randrange(1, 500)))
            raw = json.dumps({**JUDGMENT, 'reason': reason}, ensure_ascii=bool(i % 2))
            words, state, raised = len(reason.split()), LexicalState(), False
            try:
                for character in raw:
                    state = state.step(character)
            except ValueError:
                raised = True
            self.assertEqual(raised, words > 80, i)
            if not raised:
                self.assertEqual(state.words, words, i)

    def test_effective_tokenizer_config_and_multichar_boundary_filter(self):
        from bounded_reason import make_token_enforcer, exact_config
        from lmformatenforcer import TokenEnforcerTokenizerData
        from lmformatenforcer.consts import (CONFIG_ENV_VAR_MAX_CONSECUTIVE_WHITESPACES,
            CONFIG_ENV_VAR_STRICT_JSON_FIELD_ORDER, CONFIG_ENV_VAR_MAX_JSON_ARRAY_LENGTH)
        tokens = list(dict.fromkeys(['\n', '\r', '\t'] + [chr(n) for n in range(32, 127)] +
            ['中', '虽', '😀', '\u00a0', '\u2028', '\u200b', 'a b', 'a ', ' b',
             '\\u0020', '\\u0061', '\\uFFF', '\\n', '\\b']))
        lookup = {s: i for i, s in enumerate(tokens)}
        eos, prompt = len(tokens), len(tokens) + 1
        data = TokenEnforcerTokenizerData([(i, s, False) for i, s in enumerate(tokens)],
            lambda ids: ''.join(tokens[i] for i in ids), eos, False, len(tokens) + 2)
        with unittest.mock.patch.dict('os.environ', {CONFIG_ENV_VAR_MAX_CONSECUTIVE_WHITESPACES: '1',
                CONFIG_ENV_VAR_STRICT_JSON_FIELD_ORDER: 'true', CONFIG_ENV_VAR_MAX_JSON_ARRAY_LENGTH: '99'}):
            enforcer = make_token_enforcer(JUDGMENT_SCHEMA, data)
        self.assertEqual(enforcer.root_parser.config, exact_config(data.tokenizer_alphabet))
        self.assertEqual(enforcer.root_parser.inner.config, exact_config(data.tokenizer_alphabet))
        self.assertIsNone(enforcer.root_parser.shortcut_key())
        self.assertIsNone(enforcer.root_parser.cache_key())
        sequence = [prompt]
        raw = '{"category":"unrelated","distinction_types":["none"],"reason":"' + 'a ' * 80
        for c in raw:
            self.assertIn(lookup[c], enforcer.get_allowed_tokens(sequence).allowed_tokens)
            sequence.append(lookup[c])
        allowed = enforcer.get_allowed_tokens(sequence).allowed_tokens
        for legal in [' ', '"', '\\u0020', '\\n', '\u00a0', '\u2028']:
            self.assertIn(lookup[legal], allowed, repr(legal))
        for illegal in ['a', '中', '虽', '😀', '\u200b', 'a b', 'a ', ' b', '\\u0061', '\\uFFF', '\\b']:
            self.assertNotIn(lookup[illegal], allowed, repr(illegal))
        self.assertNotIn(eos, allowed)
        for c in '"}':
            self.assertIn(lookup[c], enforcer.get_allowed_tokens(sequence).allowed_tokens)
            sequence.append(lookup[c])
        self.assertIn(eos, enforcer.get_allowed_tokens(sequence).allowed_tokens)

    def test_unicode_prefixes_remain_completable_and_immutable(self):
        from bounded_reason import make_parser, WHITESPACE_HEX, exact_config
        parser = make_parser(JUDGMENT_SCHEMA, self.ALPHABET)
        for c in '{"category":"unrelated","distinction_types":["none"],"reason":"' + 'a ' * 80:
            parser = parser.add_character(c)
        for code in WHITESPACE_HEX:
            child = parser
            for c in '\\u' + code.upper():
                self.assertIn(c, child.get_allowed_characters())
                child = child.add_character(c)
            self.assertEqual(child.lexical.words, 80)
            self.assertFalse(child.lexical.inside_word)
        for c in ('a', '中', '😀'):
            self.assertNotIn(c, parser.get_allowed_characters())
        self.assertEqual(parser.lexical.words, 80)
        self.assertEqual(parser.lexical.role, 'reason')
        self.assertEqual(parser.add_character('"').lexical.role, '')
        with self.assertRaises(ValueError):
            parser.config = exact_config(self.ALPHABET + 'different')


class GreedyGuardTests(unittest.TestCase):
    def data(self):
        from lmformatenforcer import TokenEnforcerTokenizerData
        tokens = list(dict.fromkeys(['\n', '\r', '\t'] + [chr(i) for i in range(32, 127)] +
            ['中', '😀', 'a b', 'a ', ' b', '\\u0020', '\\u0061', '\\n', '\\b']))
        data = TokenEnforcerTokenizerData([(i, s, False) for i, s in enumerate(tokens)],
            lambda ids: ''.join(tokens[i] for i in ids), len(tokens), False, len(tokens) + 2)
        return tokens, {s: i for i, s in enumerate(tokens)}, data

    def test_production_ranking_ties_and_nonfinite_scores(self):
        from bounded_reason import rank_finite_scores
        self.assertEqual(rank_finite_scores([0, 2, 2, -math.inf, -5]), [1, 2, 0, 4])
        self.assertEqual(rank_finite_scores([-math.inf] * 3), [])
        for scores in ([math.nan], [math.inf], [0, math.nan], [-math.inf, math.inf]):
            with self.assertRaises(ValueError):
                rank_finite_scores(scores)

    def test_transition_matches_full_trie_for_prefix_stable_fixture_candidates(self):
        from bounded_reason import make_token_enforcer, make_parser
        from bounded_reason import GuardTokenState, choose_ranked
        tokens, lookup, data = self.data()
        eos, prompt = len(tokens), len(tokens) + 1
        for n, tail in ((79, 'a'), (79, ''), (80, ''), (1, '')):
            raw = '{"category":"unrelated","distinction_types":["none"],"reason":"' + 'a ' * n + tail
            enforcer = make_token_enforcer(JUDGMENT_SCHEMA, data)
            state = GuardTokenState(make_parser(JUDGMENT_SCHEMA, data.tokenizer_alphabet))
            sequence = [prompt]
            for c in raw:
                enforcer.get_allowed_tokens(sequence)
                state = state.transition(lookup[c], data)
                sequence.append(lookup[c])
            allowed = set(enforcer.get_allowed_tokens(sequence).allowed_tokens)
            for token in range(eos + 1):
                try:
                    state.transition(token, data)
                    accepted = True
                except ValueError:
                    accepted = False
                self.assertEqual(accepted, token in allowed, (n, token))
            if n == 80:
                selected, next_state, count, rejected = choose_ranked([lookup['a'], lookup['"']], state, data)
                self.assertEqual(selected, lookup['"'])
                self.assertEqual(count, 2)
                self.assertEqual(sum(rejected.values()), 1)
                self.assertEqual(next_state.parser.lexical.role, '')
                from bounded_reason import CandidateExhaustedError
                with self.assertRaises(CandidateExhaustedError) as exhausted:
                    choose_ranked([lookup['a']], state, data)
                self.assertEqual(exhausted.exception.candidates, 1)
                self.assertEqual(sum(exhausted.exception.rejected.values()), 1)
                with self.assertRaises(CandidateExhaustedError) as empty:
                    choose_ranked([], state, data)
                self.assertEqual(empty.exception.candidates, 0)
                self.assertEqual(empty.exception.rejected, {})

    def test_partial_unicode_prefix_stability_and_eos_policy(self):
        from bounded_reason import make_parser
        from bounded_reason import GuardTokenState
        tokens, lookup, data = self.data()
        state = GuardTokenState(make_parser(JUDGMENT_SCHEMA, data.tokenizer_alphabet))
        with self.assertRaisesRegex(ValueError, 'Premature EOS'):
            state.transition(data.eos_token_id, data)
        for c in '{"category":"unrelated","distinction_types":["none"],"reason":"':
            state = state.transition(lookup[c], data)
        tree = SimpleNamespace(tokens_to_strs={0: '', 1: '中'}, new_word_tokens=set())
        partial = SimpleNamespace(tokenizer_tree=tree, eos_token_id=99,
            decoder=lambda ids: '' if not ids or ids == [0] else '中')
        next_state = GuardTokenState(state.parser).transition(0, partial).transition(1, partial)
        self.assertEqual(next_state.parser.lexical.words, 1)
        unstable = SimpleNamespace(tokenizer_tree=tree, eos_token_id=99,
            decoder=lambda ids: 'x' if ids == [0] else '中')
        with self.assertRaisesRegex(ValueError, 'Non-prefix-stable'):
            GuardTokenState(state.parser, (0,)).transition(1, unstable)
        for c in 'x"}':
            state = state.transition(lookup[c], data)
        ended = state.transition(data.eos_token_id, data)
        self.assertTrue(ended.ended)
        with self.assertRaisesRegex(ValueError, 'Token after EOS'):
            ended.transition(lookup[' '], data)

    def test_real_torch_processor_ties_mask_order_and_history(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Actual torch tensor integration requires the pinned runtime environment; no package installation')
        from bounded_reason import GreedyGuardProcessor, make_base_enforcer
        tokens, lookup, data = self.data()
        prompt = len(tokens) + 1
        processor = GreedyGuardProcessor(JUDGMENT_SCHEMA, data, make_base_enforcer(JUDGMENT_SCHEMA, data), [prompt])
        ids = torch.tensor([[prompt]], dtype=torch.long)
        scores = torch.full((1, len(tokens) + 2), -math.inf)
        scores[0, lookup['{']] = scores[0, lookup[' ']] = 7
        result = processor(ids, scores)
        selected = min(lookup['{'], lookup[' '])
        self.assertEqual(torch.nonzero(torch.isfinite(result[0])).flatten().tolist(), [selected])
        self.assertEqual(processor.steps[0]['accepted_token_id'], selected)
        self.assertTrue(processor.steps[0]['prefix_mask_verified'])
        with self.assertRaisesRegex(ValueError, 'departed'):
            processor(ids, scores)
        processor = GreedyGuardProcessor(JUDGMENT_SCHEMA, data, make_base_enforcer(JUDGMENT_SCHEMA, data), [prompt])
        scores[0, lookup['a']] = 8
        with self.assertRaisesRegex(ValueError, 'Native prefix mask'):
            processor(ids, scores)
        from bounded_reason import CandidateExhaustedError
        processor = GreedyGuardProcessor(JUDGMENT_SCHEMA, data, make_base_enforcer(JUDGMENT_SCHEMA, data), [prompt])
        with self.assertRaises(CandidateExhaustedError):
            processor(ids, torch.full_like(scores, -math.inf))
        self.assertIsNone(processor.steps[0]['accepted_token_id'])
        self.assertEqual(processor.steps[0]['candidates_considered'], 0)

    def test_real_transformers_prefix_then_custom_smoke(self):
        try:
            import torch
            from transformers.generation.logits_process import LogitsProcessorList, PrefixConstrainedLogitsProcessor
        except ImportError:
            self.skipTest('Actual torch/Transformers processor integration requires the pinned runtime environment')
        from bounded_reason import GreedyGuardProcessor, make_base_enforcer
        tokens, lookup, data = self.data()
        prompt = len(tokens) + 1
        ids = torch.tensor([[prompt]], dtype=torch.long)
        def pipeline(reverse=False):
            base = make_base_enforcer(JUDGMENT_SCHEMA, data)
            prefix = PrefixConstrainedLogitsProcessor(lambda batch, sent: base.get_allowed_tokens(sent.tolist()).allowed_tokens, 1)
            custom = GreedyGuardProcessor(JUDGMENT_SCHEMA, data, base, [prompt])
            return LogitsProcessorList([custom, prefix] if reverse else [prefix, custom]), custom
        scores = torch.full((1, len(tokens) + 2), -math.inf)
        scores[0, lookup['{']] = scores[0, lookup[' ']] = 7
        scores[0, lookup['a']] = 100  # Native mask must remove this first.
        processors, custom = pipeline()
        result = processors(ids, scores)
        expected = min(lookup['{'], lookup[' '])
        self.assertEqual(torch.argmax(result, dim=-1).tolist(), [expected])
        self.assertEqual(torch.nonzero(torch.isfinite(result[0])).flatten().tolist(), [expected])
        self.assertEqual(custom.steps[0]['candidates_considered'], 1)
        with self.assertRaisesRegex(ValueError, 'Native prefix mask'):
            pipeline(reverse=True)[0](ids, scores)
        for value in (math.nan, math.inf):
            invalid = scores.clone()
            invalid[0, lookup['{']] = value
            with self.assertRaisesRegex(ValueError, 'NaN or positive infinity'):
                pipeline()[0](ids, invalid)


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
    def state(self, output, input_length):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **_kwargs: json.dumps(messages)
        tokenizer.return_value = {"input_ids": FakeTensor(input_length)}
        tokenizer.decode.return_value = output
        model = Mock()
        model.generate.return_value = FakeTensor(10)
        torch = Mock()
        torch.inference_mode.side_effect = nullcontext
        torch.cuda.max_memory_allocated.return_value = 0
        torch.cuda.max_memory_reserved.return_value = 0
        return {"manifest": {"pilot_id": "software-fixture-v4", "plan_sha256": "a" * 64,
                "runtime_plan_sha256": "0" * 64, "inputs_sha256": "1" * 64,
                "model_id": "software-model", "model_revision": "b" * 40,
                "weights_manifest_sha256": "c" * 64}, "system_prompt": "Synthetic fixture rubric",
                "task_definition": "Synthetic fixture only", "prompt_sha256": "d" * 64,
                "runtime_manifest_sha256": "e" * 64, "decoding_config_sha256": "f" * 64,
                "judgment_schema": JUDGMENT_SCHEMA, "response_schema": RESPONSE_SCHEMA,
                "constraint_alphabet": GuardTests.ALPHABET,
                "effective_constraint_config_sha256": "2" * 64,
                "torch": torch, "tokenizer": tokenizer, "model": model, "generation": object(),
                "tokenizer_data": object(), "flash_context": Mock(side_effect=nullcontext),
                "flash_flags": lambda: {"flash": True, "math": False, "mem_efficient": False, "cudnn": False}}

    def execute(self, state):
        with tempfile.TemporaryDirectory() as temporary:
            with (Path(temporary) / "attempts.jsonl").open("w", encoding="utf-8") as attempts:
                with unittest.mock.patch.object(runner, "build_constraint", return_value=(object(), Mock())):
                    with (unittest.mock.patch.object(runner, "replay_tokens", return_value=True),
                          unittest.mock.patch('bounded_reason.GreedyGuardProcessor', return_value=SimpleNamespace(steps=[]))):
                        result = runner.run_call(state, rows(1, 0)[0], 0, Path(temporary), attempts)
            saved = {p.name: json.loads(p.read_text()) for p in Path(temporary).rglob("*.messages.private.json")}
            metrics = [json.loads(line) for line in (Path(temporary) / "attempts.jsonl").read_text().splitlines()]
            return result, saved, metrics

    def test_v4_constants_are_locked(self):
        self.assertEqual(runner.MAX_NEW_TOKENS, 512)
        self.assertEqual(runner.TOTAL_TOKEN_BUDGET, 32768)

    def test_context_overflow_never_generates_or_truncates(self):
        state = self.state(json.dumps(JUDGMENT), 32768 - 512 + 1)
        result, saved, metrics = self.execute(state)
        self.assertEqual(result["status"], "context_overflow")
        self.assertIsNone(result["judgment"])
        self.assertEqual(state["model"].generate.call_count, 0)
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(metrics), 1)
        self.assertIs(state["tokenizer"].call_args.kwargs["truncation"], False)

    def test_context_boundary_generates_once_with_constraint_and_flash_context(self):
        state = self.state(json.dumps(JUDGMENT), 32768 - 512)
        result, saved, metrics = self.execute(state)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(state["model"].generate.call_count, 1)
        self.assertIn("prefix_allowed_tokens_fn", state["model"].generate.call_args.kwargs)
        self.assertEqual(state["flash_context"].call_count, 1)
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(metrics), 1)

    def test_invalid_format_schema_and_word_limit_never_retry(self):
        for output in ("```json\n" + json.dumps(JUDGMENT) + "\n```", "{" + json.dumps(JUDGMENT),
                       json.dumps({**JUDGMENT, "distinction_types": ["none", "none"]}),
                       json.dumps({**JUDGMENT, "reason": "x " * 81})):
            state = self.state(output, 100)
            with self.subTest(output=output[:30]):
                result, saved, metrics = self.execute(state)
                self.assertEqual(result["status"], "invalid_json")
                self.assertIsNone(result["judgment"])
                self.assertEqual(state["model"].generate.call_count, 1)
                self.assertEqual(len(saved), 1)
                self.assertEqual(len(metrics), 1)

    def test_runtime_error_never_retries(self):
        state = self.state("unused", 100)
        state["model"].generate.side_effect = RuntimeError("Synthetic software failure")
        result, saved, metrics = self.execute(state)
        self.assertEqual(result["status"], "runtime_error")
        self.assertIsNone(result["judgment"])
        self.assertEqual(state["model"].generate.call_count, 1)
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(metrics), 1)

    def test_attention_fallback_refused_before_generation(self):
        state = self.state(json.dumps(JUDGMENT), 100)
        state["flash_flags"] = lambda: {"flash": True, "math": True, "mem_efficient": False, "cudnn": False}
        result, _saved, metrics = self.execute(state)
        self.assertEqual(result["status"], "runtime_error")
        self.assertEqual(state["model"].generate.call_count, 0)
        self.assertEqual(len(metrics), 1)

    def test_schedule_preserves_all_44_rows_longest_first_with_stable_ties(self):
        state = self.state(json.dumps(JUDGMENT), 100)
        values = rows(40, 4)
        values[0]["text_a"] = "Synthetic full text\u2028including Unicode and\nnewlines " + "tail " * 100
        values[40]["text_b"] = values[0]["text_a"]
        state["rows"] = values
        lengths = [10 + i % 7 for i in range(44)]
        lengths[31] = 20737
        state["tokenizer"].side_effect = [{"input_ids": FakeTensor(length)} for length in lengths]
        with tempfile.TemporaryDirectory() as temporary:
            schedule = runner.prepare_schedule(state, Path(temporary))
            self.assertEqual(schedule, sorted(range(44), key=lambda i: (-lengths[i], i)))
            self.assertEqual(sorted(schedule), list(range(44)))
            self.assertEqual(state["tokenizer"].call_count, 44)
            self.assertEqual(state["model"].generate.call_count, 0)
            census = json.loads((Path(temporary) / "schedule.json").read_text())
            self.assertEqual(len(census["input_census"]), 44)
            self.assertEqual(state["schedule_sha256"], runner.file_hash(Path(temporary) / "schedule.json"))
            for index, item in enumerate(census["input_census"]):
                self.assertEqual(item["input_index"], index)
                self.assertEqual(item["input_sha256"], runner.digest_bytes(runner.canonical(values[index])))
                self.assertNotIn("text_a", item)
                self.assertNotIn("text_b", item)
            shown = json.loads(state["prepared_calls"][0]["messages"][1]["content"])
            self.assertEqual(shown["documents"][0]["text"], values[0]["text_a"])
            for call in state["tokenizer"].call_args_list:
                self.assertIs(call.kwargs["truncation"], False)

    def test_fatal_after_reordered_call_preserves_exact_40_plus_4_denominator(self):
        fixture = self.state(json.dumps(JUDGMENT), 100)
        fixture["rows"] = rows(40, 4)
        fixture["bound_files"] = {}

        def prepare(_manifest, _out, state):
            state.update(fixture)

        def call(state, row, index, _out, _attempts):
            if index == 5:
                raise RuntimeError("Synthetic outer failure after a reordered completed call")
            call_id = runner.digest_bytes(runner.canonical([state["manifest"]["pilot_id"], index,
                                                          row["pair_id"], row["orientation"]]))
            return runner.envelope(state, row, call_id, "valid", copy.deepcopy(JUDGMENT), "0" * 64)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-output"
            with (unittest.mock.patch.object(runner, "prepare_context", side_effect=prepare),
                  unittest.mock.patch.object(runner, "load_model"),
                  unittest.mock.patch.object(runner, "prepare_schedule", return_value=[43, 5] + [i for i in range(44) if i not in (43, 5)]),
                  unittest.mock.patch.object(runner, "run_call", side_effect=call),
                  unittest.mock.patch("sys.argv", ["run_pilot.py", "--manifest", str(Path(temporary) / "fixture.json"), "--output-dir", str(output)]),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(), 1)
            responses = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            self.assertEqual(len(responses), 44)
            self.assertEqual(len({r["call_id"] for r in responses}), 44)
            self.assertEqual({(r["pair_id"], r["orientation"]) for r in responses},
                             {(r["pair_id"], r["orientation"]) for r in fixture["rows"]})
            self.assertEqual(responses[0]["pair_id"], fixture["rows"][43]["pair_id"])
            self.assertEqual(responses[0]["status"], "valid")
            self.assertTrue(all(r["status"] == "runtime_error" for r in responses[1:]))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["canonical_pairs"], 40)
            self.assertEqual(summary["recorded_input_rows"], 44)
            self.assertEqual(summary["status_counts_canonical"], {"runtime_error": 40})
            self.assertIs(summary["completed"], False)
            self.assertIs(summary["human_completed"], False)
            self.assertIs(summary["confirmatory_study_ready"], False)


if __name__ == "__main__":
    unittest.main()
