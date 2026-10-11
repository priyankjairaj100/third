"""Tiny software fixtures only; no corpus acquisition or scientific evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import replay_source as replay

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "empirical_execution"))
from phase3 import panels, calibration as cal
from phase4 import adapters

spec = importlib.util.spec_from_file_location("software_fixture_lineage", REPO / "local_run/source_lineage_audit_v1_20261011/audit_source.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def write_rows(path, rows, ascii_only=False):
    path.write_bytes(b"".join(json.dumps(row, ensure_ascii=ascii_only, allow_nan=False).encode() + b"\n" for row in rows))


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.posts, cls.links = cls.root / "Posts.angle.xml", cls.root / "PostLinks.xml"
        rows = ['<?xml version="1.0" encoding="utf-8"?>', '<posts>']
        for i in range(1, 33):
            rows.append(f'<row Id="{i}" PostTypeId="1" OwnerUserId="{i}" CreationDate="2019-01-01T00:00:00" Title="Software fixture {i}" Body="&lt;p&gt;tiny\u2028fixture {i}&lt;/p&gt;" Tags="&lt;fixture&gt;" />')
        rows.append('</posts>')
        cls.posts.write_text('\n'.join(rows) + '\n', encoding='utf-8')
        cls.links.write_text('<postlinks>\n<row Id="1" PostId="1" RelatedPostId="2" LinkTypeId="3" />\n</postlinks>\n')
        cls.original, cls.rebuilt = cls.root / "original", cls.root / "rebuilt"
        kwargs = dict(dataset_id="askubuntu", archive_sha256="0" * 64, archive_release="software-fixture-only")
        cls.supplied = adapters.adapt_stack(cls.posts, cls.links, cls.original, **kwargs)
        cls.replayed = adapters.adapt_stack(cls.posts, cls.links, cls.rebuilt, **kwargs)
        cls.raw = list(helper.jsonl(cls.rebuilt / "records.jsonl"))
        cls.design = replay.read_json(REPO / "output/empirical_program/study_design.json")
        cls.prepared, cls.ids, cls.audit = panels.prepare_population(cls.raw, "askubuntu", cls.design,
            duplicate_links=replay.read_json(cls.rebuilt / "duplicate_links.json"), target=1, allow_preview_sources=False)
        cls.population = cls.root / "population"
        cls.population.mkdir()
        write_rows(cls.population / "records.jsonl", cls.prepared)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_frozen_code_bindings_match(self):
        frozen = replay.read_json(Path(__file__).with_name("frozen_code.json"))
        for name, expected in frozen["files"].items():
            self.assertEqual(replay.binding(REPO / name), expected)

    def test_frozen_adapter_replay_all_outputs_and_settings(self):
        replay.compare_adapter(self.original, self.rebuilt, self.supplied, self.replayed)
        self.assertEqual(adapters.verify_adapter_output(self.rebuilt)["verified_rows"], 32)
        changed = copy.deepcopy(self.replayed)
        changed["settings"]["question_only"] = False
        with self.assertRaisesRegex(replay.Refusal, "adapter_metadata_mismatch:settings"):
            replay.compare_adapter(self.original, self.rebuilt, self.supplied, changed)

    def test_adapter_byte_tamper_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            damaged = Path(directory) / "adapter"
            shutil.copytree(self.rebuilt, damaged)
            path = damaged / "records.jsonl"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(replay.Refusal, "replayed_adapter_output_bytes_mismatch"):
                replay.compare_adapter(self.original, damaged, self.supplied, self.replayed)

    def test_physical_lf_preserves_unicode_and_full_preparation(self):
        physical_rows = list(helper.jsonl(self.rebuilt / "records.jsonl"))
        self.assertEqual(len(physical_rows), 32)
        self.assertGreater(len((self.rebuilt / "records.jsonl").read_text().splitlines()), 32)
        checked = replay.compare_population(self.prepared, self.ids, self.audit,
            self.population / "records.jsonl", self.audit, self.ids, helper)
        self.assertEqual(checked["rows"], len(self.prepared))
        self.assertTrue(checked["panel_ids_equal"])

    def test_reordered_prepared_rows_refused(self):
        with self.assertRaisesRegex(replay.Refusal, "prepared_full_row_or_order_mismatch"):
            replay.compare_population(list(reversed(self.prepared)), self.ids, self.audit,
                self.population / "records.jsonl", self.audit, self.ids, helper)

    def test_changed_panel_ids_refused(self):
        changed = copy.deepcopy(self.ids)
        changed["calibration"].append("invented-software-fixture-id")
        with self.assertRaisesRegex(replay.Refusal, "panel_ids_mismatch"):
            replay.compare_population(self.prepared, self.ids, self.audit,
                self.population / "records.jsonl", self.audit, changed, helper)

    def test_changed_panel_or_guard_audit_refused(self):
        for key in ("panels", "fixed_guard", "retained_original_indices"):
            changed = copy.deepcopy(self.audit)
            changed[key] = {}
            with self.assertRaisesRegex(replay.Refusal, "population_audit_mismatch"):
                replay.compare_population(self.prepared, self.ids, self.audit,
                    self.population / "records.jsonl", changed, self.ids, helper)

    def test_lineage_failure_partial_and_false_boolean_refused(self):
        stages = {name: {} for name in ("input_inventory", "archive_members", "tag_transform", "adapter_bindings",
                                        "physical_jsonl_transport", "population_bindings", "immutability")}
        stages["immutability"]["all_watched_input_final_bytes_equal_initial"] = True
        stages["archive_members"]["local_archive_member_byte_equality"] = True
        stages["tag_transform"]["complete_expected_transformed_bytes_equal"] = True
        stages["physical_jsonl_transport"]["complete_object_and_order_equality"] = True
        good = {"schema": "third-askubuntu-source-lineage-audit-v1", "passed": True,
                "stage": "complete", "completed_stages": stages}
        replay.validate_lineage(good)
        for changed in ({**good, "passed": False}, {**good, "passed": 1}, {**good, "stage": "population_bindings"}):
            with self.assertRaises(replay.Refusal):
                replay.validate_lineage(changed)
        changed = copy.deepcopy(good)
        del changed["completed_stages"]["immutability"]
        with self.assertRaises(replay.Refusal):
            replay.validate_lineage(changed)

    def test_watch_detects_mutation_and_wrong_expected_package(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = folder / "data.py"
            data.write_text("# tiny software fixture\n")
            replay.write_json(folder / "package_manifest.json", {"files": {data.name: replay.binding(data)}})
            sha = replay.binding(folder / "package_manifest.json")["sha256"]
            watched = replay.Watch()
            replay.verify_package(folder, sha, watched)
            watched.verify()
            with self.assertRaises(replay.Refusal):
                replay.verify_package(folder, "f" * 64, replay.Watch())
            data.write_text("# changed fixture\n")
            with self.assertRaises(replay.Refusal):
                watched.verify()

    def test_ascii_view_receipt_semantics_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            population, view = workspace / "population", workspace / "view"
            population.mkdir()
            view.mkdir()
            write_rows(population / "records.jsonl", self.prepared)
            write_rows(view / "records.jsonl", self.prepared, ascii_only=True)
            original = cal._seal({**self.audit, "prepared_records_file_sha256": replay.binding(population / "records.jsonl")["sha256"]})
            replay.write_json(population / "audit.json", original)
            def metadata(path):
                return {"path": str(path), **replay.binding(path)}
            converter = REPO / "local_run/engineering_v20261011/source_transport_v1.py"
            lineage = {"schema": "third-lossless-jsonl-transport-v1", "original_preparation_directory": str(population),
                       "original_audit_seal_sha256": original["sha256"], "original_records": metadata(population / "records.jsonl"),
                       "original_audit_file": metadata(population / "audit.json"), "converter": metadata(converter)}
            derived = cal._seal({**{k: v for k, v in original.items() if k != "sha256"},
                "prepared_records_file_sha256": replay.binding(view / "records.jsonl")["sha256"], "transport_lineage": lineage})
            replay.write_json(view / "audit.json", derived)
            receipt = {"schema": "third-lossless-jsonl-transport-receipt-v1", "dataset": "askubuntu", "passed": True,
                       "complete_decoded_row_equality_and_order_verified": True, "unchanged_frozen_load_prepared_records_passed": True,
                       **{k: lineage[k] for k in ("original_records", "original_audit_file", "converter", "original_audit_seal_sha256")},
                       "derived_records": metadata(view / "records.jsonl"), "derived_audit_file": metadata(view / "audit.json"),
                       "derived_audit_seal_sha256": derived["sha256"]}
            replay.write_json(view / "receipt.json", receipt)
            _, _, result = replay.verify_view(view, population, REPO, workspace, helper, replay.Watch())
            self.assertEqual(result["rows"], len(self.prepared))
            changed = copy.deepcopy(derived)
            changed["panels"] = {}
            changed = cal._seal({k: v for k, v in changed.items() if k != "sha256"})
            (view / "audit.json").write_text(json.dumps(changed))
            receipt["derived_audit_file"] = metadata(view / "audit.json")
            receipt["derived_audit_seal_sha256"] = changed["sha256"]
            (view / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(replay.Refusal, "transport_changed_original_audit_semantics"):
                replay.verify_view(view, population, REPO, workspace, helper, replay.Watch())


if __name__ == "__main__":
    unittest.main()
