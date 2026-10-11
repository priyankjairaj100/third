"""Tiny software fixtures/fake tokenizer only, never semantic research evidence."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import census

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "empirical_execution"))
from phase4 import embeddings
from phase3 import calibration


class FakeTokenizer:
    """Software token-count fixture, not an actual semantic tokenizer."""
    model_max_length = 512
    is_fast = True

    def encode(self, text, *, add_special_tokens=False, truncation=False):
        if add_special_tokens or truncation:
            raise AssertionError("Hidden specials or truncation requested")
        return [900, 901, 902] if text == "query: " else list(range(int(text.removeprefix("tokens:"))))

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def prepare_for_model(self, content, **kwargs):
        if kwargs != {"add_special_tokens": True, "padding": False, "truncation": False,
                      "return_attention_mask": True, "return_token_type_ids": False}:
            raise AssertionError("Frozen chunk options changed")
        ids = [101] + content + [102]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def rows_for(lengths):
    return [{"record_id": "software-fixture-" + str(i), "text": "tokens:" + str(length),
             "partition": ("train", "calibration", "test")[i % 3],
             "original_fields": {"software_fixture": "a\u2028b\u2029c"}}
            for i, length in enumerate(lengths)]


class CensusTests(unittest.TestCase):
    def scan(self, rows, ledger=None):
        return census.census_rows(rows, FakeTokenizer(), embeddings.chunk_inputs, embeddings.E5,
                                  512, 261, ledger if ledger is not None else io.StringIO())

    def test_exact_256_boundaries_and_per_record_batches(self):
        rows = rows_for([1, 256, 257, 2048, 2049])
        ledger = io.StringIO()
        result = self.scan(rows, ledger)
        self.assertEqual(result["totals"]["chunks"], 21)
        self.assertEqual(result["totals"]["batch_count"], 6)
        self.assertEqual(result["totals"]["full_batches_of_8"], 2)
        self.assertEqual(result["tail_batch_size_histogram_excluding_zero"], {1: 3, 2: 1})
        self.assertEqual(result["content_token_bins_256_upper_inclusive"], {256: 2, 512: 1, 2048: 1, 2304: 1})
        self.assertEqual(result["totals"]["content_tokens"], 4611)
        self.assertEqual(result["totals"]["padded_token_slots"], 5481)
        self.assertEqual(result["totals"]["attended_token_slots"], 4716)
        self.assertEqual(result["totals"]["padding_token_slots"], 765)
        self.assertEqual(result["maximum_batch_padded_token_slots"], 2088)
        self.assertEqual(result["maximum_record_batch_count"], 2)
        self.assertEqual(result["maximum_record_chunks"], 9)
        self.assertIsNone(result["encoding_walltime_projection"])
        decoded = [json.loads(line) for line in ledger.getvalue().splitlines()]
        self.assertEqual([r["row_index"] for r in decoded], list(range(5)))
        self.assertEqual([r["record_id"] for r in decoded], [r["record_id"] for r in rows])
        self.assertTrue(all("text" not in r and "input_ids" not in r for r in decoded))
        self.assertTrue(all("content_token_ids_sha256" in r["detail"] for r in decoded))

    def test_frame_digests_order_and_private_source_unchanged(self):
        rows = rows_for([3, 301, 8])
        before = copy.deepcopy(rows)
        result = self.scan(rows)
        self.assertEqual(result["full_prepared_rows_sha256"], calibration.digest(rows))
        self.assertEqual(result["text_binding"], calibration.digest([{k: r[k] for k in ("record_id", "text")} for r in rows]))
        self.assertEqual(result["ordered_record_ids_sha256"], calibration.digest([r["record_id"] for r in rows]))
        self.assertEqual(rows, before)
        self.assertNotEqual(self.scan(list(reversed(rows)))["ordered_record_ids_sha256"], result["ordered_record_ids_sha256"])

    def test_physical_lf_unicode_separators_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            rows = rows_for([3, 4])
            path.write_bytes(b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows))
            raw = path.read_bytes()
            self.assertGreater(len(raw.decode().splitlines()), 2)
            self.assertEqual(list(census.physical_rows(path)), rows)
            self.assertEqual(self.scan(census.physical_rows(path))["totals"]["rows"], 2)
            self.assertEqual(path.read_bytes(), raw)

    def test_missing_repeated_and_empty_rows_refused(self):
        rows = rows_for([1])
        with self.assertRaisesRegex(census.Refusal, "duplicate_or_missing"):
            self.scan(rows + rows)
        with self.assertRaisesRegex(census.Refusal, "empty_census"):
            self.scan([])
        for field, value in (("record_id", ""), ("text", ""), ("partition", "invalid")):
            damaged = copy.deepcopy(rows)
            damaged[0][field] = value
            with self.assertRaises(census.Refusal):
                self.scan(damaged)

    def test_chunk_coverage_and_mask_tamper_refused(self):
        chunks, detail = embeddings.chunk_inputs(FakeTokenizer(), "tokens:257", embeddings.E5, 512)
        for change in ("drop", "offset", "mask", "content_count"):
            altered, log = copy.deepcopy(chunks), copy.deepcopy(detail)
            if change == "drop": altered.pop()
            if change == "offset": altered[1]["content_start"] = 255
            if change == "mask": altered[0]["attention_mask"][0] = 0
            if change == "content_count": log["chunk_content_tokens"] = [255, 2]
            with self.assertRaises(census.Refusal):
                census.row_cost(altered, log, 261)
        with self.assertRaisesRegex(census.Refusal, "fixed_padding"):
            census.row_cost(chunks, detail, 260)

    def test_strict_json_no_duplicate_keys_or_nonfinite_numbers(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'\xff'):
            with self.assertRaises(census.Refusal):
                census.strict_parse(raw)

    def test_source_replay_failure_partial_and_nonboolean_refused(self):
        good = {"schema": "third-askubuntu-source-replay-v1", "passed": True, "stage": "complete",
            "completed_stages": {
                "final_immutability": {"all_watched_files_equal_initial_bytes": True},
                "prepared_ascii_view": {"complete_object_order_and_ascii_bytes_rechecked": True},
                "frozen_preparation_replay": {key: True for key in ("all_prepared_objects_and_order_equal", "panel_ids_equal", "all_rebuilt_population_audit_fields_equal", "unchanged_frozen_loader_passed")}}}
        census.validate_replay(good)
        for changed in ({**good, "passed": False}, {**good, "passed": 1}, {**good, "stage": "final_immutability"}):
            with self.assertRaises(census.Refusal):
                census.validate_replay(changed)
        changed = copy.deepcopy(good)
        changed["completed_stages"]["frozen_preparation_replay"]["panel_ids_equal"] = False
        with self.assertRaises(census.Refusal):
            census.validate_replay(changed)

    def test_input_and_package_mutation_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = folder / "fixture.py"
            data.write_text("# tiny software fixture\n")
            census.write_json(folder / "package_manifest.json", {"files": {data.name: census.binding(data)}})
            digest = census.binding(folder / "package_manifest.json")["sha256"]
            watched = census.Watch()
            census.verify_package(folder, digest, watched)
            watched.verify()
            with self.assertRaises(census.Refusal):
                census.verify_package(folder, "f" * 64, census.Watch())
            data.write_text("# changed fixture\n")
            with self.assertRaises(census.Refusal):
                watched.verify()

    def test_frozen_original_chunker_and_dependencies_match(self):
        frozen = census.read_json(Path(__file__).with_name("frozen_code.json"))
        for relative, expected in frozen["files"].items():
            self.assertEqual(census.binding(REPO / relative), expected)

    def test_first_window_and_first_chunk_costs_are_distinct(self):
        result = self.scan(rows_for([256, 257, 507, 508]))
        self.assertEqual(result["totals"]["first_chunk_truncated_records"], 3)
        self.assertEqual(result["totals"]["first_window_truncated_records"], 1)


if __name__ == "__main__":
    unittest.main()
