"""Tiny software fixtures only; never empirical source or timing evidence."""
from collections import Counter
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("bounded_e5_budget", Path(__file__).with_name("budget.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
c = b.c


def fixture():
    source, ledger = [], []
    # Three candidates per bucket permit selection invariance checks. Train and
    # >8-chunk records exercise the exclusions and full-record batch accounting.
    for k, part in [(k, "calibration") for k in range(1, 9) for _ in range(3)] + [(17, "train"), (8, "train")]:
        index = len(source)
        rid = f"software-only-{index}"
        text = f"software fixture {index}\nUnicode café \u2028 and \u2029"
        row = {"record_id": rid, "partition": part, "text": text, "source_unit_id": f"fixture-{index}"}
        source.append(row)
        n = k * 256 - 3
        ledger.append({"row_index": index, "record_id": rid, "partition": part,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "detail": {"content_tokens": n, "chunks": k, "chunk_content_tokens": [256] * (k - 1) + [253],
                "prefix_tokens_per_chunk": 3, "special_tokens_per_chunk": 2,
                "content_token_ids_sha256": "software-only-token-digest"},
            "batch_count": (k + 7) // 8, "full_batches_of_8": k // 8, "tail_batch_size": k % 8,
            "padded_token_slots": k * 261, "attended_token_slots": n + 5 * k,
            "padding_token_slots": 3, "maximum_batch_size": min(k, 8),
            "maximum_batch_padded_token_slots": min(k, 8) * 261})
    total = {key: sum(row[key] for row in ledger) for key in ("batch_count", "full_batches_of_8", "padded_token_slots", "attended_token_slots", "padding_token_slots")}
    total.update({"rows": len(ledger), "content_tokens": sum(row["detail"]["content_tokens"] for row in ledger), "chunks": sum(row["detail"]["chunks"] for row in ledger)})
    aggregate = {"totals": total,
        "partition_counts": dict(Counter(row["partition"] for row in ledger)),
        "chunks_per_record_histogram": dict(Counter(str(row["detail"]["chunks"]) for row in ledger)),
        "tail_batch_size_histogram_excluding_zero": dict(Counter(str(row["tail_batch_size"]) for row in ledger if row["tail_batch_size"])),
        "ordered_record_ids_sha256": hashlib.sha256(c.canonical([row["record_id"] for row in source])).hexdigest(),
        "full_prepared_rows_sha256": hashlib.sha256(c.canonical(source)).hexdigest(),
        "text_binding": hashlib.sha256(c.canonical([{"record_id": row["record_id"], "text": row["text"]} for row in source])).hexdigest()}
    return source, ledger, aggregate


class BudgetTests(unittest.TestCase):
    def test_selection_is_calibration_hash_order_and_bounded(self):
        source, ledger, aggregate = fixture()
        selected, candidates, batches = b.select_ledger(iter(ledger), aggregate)
        self.assertEqual(candidates, {k: 3 for k in range(1, 9)})
        self.assertEqual(len(selected), 16)
        self.assertEqual(sum(row["detail"]["chunks"] for row in selected) * 3, 216)
        self.assertEqual(sum(batches.values()), aggregate["totals"]["batch_count"])
        for k in range(1, 9):
            expected = sorted([r for r in ledger if r["partition"] == "calibration" and r["detail"]["chunks"] == k], key=lambda r: b.selection_key(r["record_id"]))[:2]
            self.assertEqual(selected[(k-1)*2:k*2], expected)

    def test_duplicate_and_missing_census_records_refused(self):
        _, rows, aggregate = fixture()
        rows[1]["record_id"] = rows[0]["record_id"]
        with self.assertRaisesRegex(c.Refusal, "duplicate"):
            b.select_ledger(rows, aggregate)
        _, rows, aggregate = fixture()
        with self.assertRaises(c.Refusal):
            b.select_ledger(rows[:-1], aggregate)

    def test_row_order_and_cost_mutation_refused(self):
        for field, value in (("row_index", 99), ("padded_token_slots", 0), ("batch_count", 0)):
            _, rows, aggregate = fixture()
            rows[0][field] = value
            with self.assertRaises(c.Refusal):
                b.select_ledger(rows, aggregate)

    def test_missing_calibration_bucket_refused(self):
        _, rows, aggregate = fixture()
        # Rebind aggregate partition counts: still no permission to replace a
        # missing natural calibration bucket with a train record.
        for row in rows:
            if row["detail"]["chunks"] == 8:
                row["partition"] = "train"
        aggregate["partition_counts"] = dict(Counter(r["partition"] for r in rows))
        with self.assertRaisesRegex(c.Refusal, "two_calibration"):
            b.select_ledger(rows, aggregate)

    def test_full_source_binding_preserves_unicode_and_lf(self):
        source, ledger, aggregate = fixture()
        selected, _, _ = b.select_ledger(ledger, aggregate)
        actual = b.bind_source(iter(source), selected, aggregate)
        self.assertEqual([r["record_id"] for r in actual], [r["record_id"] for r in selected])
        self.assertTrue(all("\u2028" in r["text"] and "\n" in r["text"] for r in actual))

    def test_unselected_full_frame_mutation_refused(self):
        source, ledger, aggregate = fixture()
        selected, _, _ = b.select_ledger(ledger, aggregate)
        source[-1]["text"] += "changed train text"
        with self.assertRaisesRegex(c.Refusal, "complete_source_frame"):
            b.bind_source(source, selected, aggregate)

    def test_selected_text_and_token_id_mutation_refused(self):
        source, ledger, aggregate = fixture()
        selected, _, _ = b.select_ledger(ledger, aggregate)
        source[selected[0]["row_index"]]["text"] += "changed"
        with self.assertRaisesRegex(c.Refusal, "selected_source"):
            b.bind_source(source, selected, aggregate)
        detail = {**selected[0]["detail"], "batch_count": 1}
        b.validate_detail(detail, selected[0])
        detail["content_token_ids_sha256"] = "changed"
        with self.assertRaisesRegex(c.Refusal, "token_ids"):
            b.validate_detail(detail, selected[0])

    def test_failed_census_refused(self):
        with self.assertRaisesRegex(c.Refusal, "passed_immutable"):
            b.validate_census({"passed": False}, {}, {})

    def test_exact_shape_weighted_projection_accounting(self):
        _, _, aggregate = fixture()
        obs = [{"shape": [k, 261], "model_forward_seconds": float(k)} for k in range(1, 9) for _ in range(6)]
        docs = [{"chunks": k, "encode_document_seconds": 2.0 * k, "model_forward_seconds": float(k)} for k in range(1, 9) for _ in range(6)]
        result = b.projections(obs, docs, aggregate, 10, 20)
        expected = aggregate["totals"]["chunks"] * 2 + 30
        for scenario in result["scenarios"].values():
            self.assertEqual(scenario["one_full_pass_modeled_seconds"], expected)
            self.assertEqual(scenario["two_separate_full_passes_modeled_seconds"], 2 * expected)
        self.assertIs(result["guaranteed_walltime_bound"], False)
        self.assertIs(result["full_encoding_launch_authorized_by_this_report"], False)

    def test_missing_shapes_and_invalid_timings_refused(self):
        _, _, aggregate = fixture()
        obs = [{"shape": [k, 261], "model_forward_seconds": float(k)} for k in range(1, 9) for _ in range(6)]
        docs = [{"chunks": k, "encode_document_seconds": 2.0 * k, "model_forward_seconds": float(k)} for k in range(1, 9) for _ in range(6)]
        with self.assertRaisesRegex(c.Refusal, "six_forward"):
            b.projections(obs[:-1], docs, aggregate, 10, 20)
        obs[0]["model_forward_seconds"] = float("nan")
        with self.assertRaisesRegex(c.Refusal, "invalid_forward"):
            b.projections(obs, docs, aggregate, 10, 20)


if __name__ == "__main__":
    unittest.main()
