"""Tiny software fixtures only: no natural records or research evidence."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import audit_source as a


class AuditTests(unittest.TestCase):
    def test_expectations_are_well_formed_and_frozen_code_matches(self):
        expected = a.read_json(Path(__file__).with_name("expectations.json"))
        repo = Path(__file__).resolve().parents[2]
        for rel, sha in expected["code"].items():
            self.assertEqual(a.binding(repo / rel)["sha256"], sha)
        for spec in expected["files"].values():
            if "sha256" in spec:
                self.assertRegex(spec["sha256"], "^[0-9a-f]{64}$")

    def test_stream_hash_and_mismatch(self):
        got = a.hash_stream(io.BytesIO(b"fixture"))
        self.assertEqual(got, {"bytes": 7, "sha256": hashlib.sha256(b"fixture").hexdigest()})
        with self.assertRaises(a.AuditError):
            a.match(got, {"bytes": 8}, "changed")

    def test_seal_binds_all_fields_and_strict_json(self):
        value = {"a": 1}
        value["sha256"] = a.digest(value)
        a.verify_seal(value)
        value["a"] = True
        with self.assertRaises(a.AuditError):
            a.verify_seal(value)
        for raw in (b'{"a":1,"a":1}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(a.AuditError):
                a.parse(raw)

    def test_tag_only_transform_preserves_non_tags_and_newlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, target = Path(tmp) / "original", Path(tmp) / "target"
            original.write_bytes(b'<row Body="same" Tags="|a|b|" />\r\n<row Tags="&lt;c&gt;" />\n')
            correct = b'<row Body="same" Tags="&lt;a&gt;&lt;b&gt;" />\r\n<row Tags="&lt;c&gt;" />\n'
            target.write_bytes(correct)
            result = a.tag_bridge(original, target)
            self.assertEqual(result["tags_converted"], 1)
            self.assertEqual(result["tags_unchanged"], 1)
            target.write_bytes(correct.replace(b'same', b'edit'))
            with self.assertRaises(a.AuditError):
                a.tag_bridge(original, target)
            self.assertIn(b'Body="same"', original.read_bytes())

    def test_physical_transport_unicode_order_and_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "target"
            rows = [{"record_id": "fixture1", "text": "a\u2028b\u2029c", "v": 1},
                    {"record_id": "fixture2", "text": "second"}]
            raw = b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in rows)
            source.write_bytes(raw)
            def write(items):
                target.write_bytes(b"".join(json.dumps(r, ensure_ascii=True).encode() + b"\n" for r in items))
            write(rows)
            self.assertEqual(len(raw.decode().splitlines()), 4)
            self.assertEqual(a.ascii_transport(source, target)["rows"], 2)
            write(list(reversed(rows)))
            with self.assertRaises(a.AuditError):
                a.ascii_transport(source, target)
            write([{**rows[0], "v": True}, rows[1]])
            with self.assertRaises(a.AuditError):
                a.ascii_transport(source, target)
            write(rows[:1])
            with self.assertRaises(a.AuditError):
                a.ascii_transport(source, target)
            self.assertEqual(source.read_bytes(), raw)

    def test_archive_stdout_hash_and_nonzero_refusal(self):
        process = mock.Mock(stdout=io.BytesIO(b"archive member fixture"))
        process.wait.return_value = 0
        process.poll.return_value = 0
        with mock.patch.object(a.subprocess, "Popen", return_value=process) as popen:
            actual = a.member_binding("7z", Path("fixture.7z"), "Posts.xml")
            self.assertEqual(actual["bytes"], 22)
            self.assertEqual(popen.call_args.args[0], ["7z", "x", "-so", "-y", "fixture.7z", "Posts.xml"])
        process.stdout = io.BytesIO(b"partial")
        process.wait.return_value = 2
        with mock.patch.object(a.subprocess, "Popen", return_value=process):
            with self.assertRaises(a.AuditError):
                a.member_binding("7z", Path("fixture.7z"), "PostLinks.xml")

    def test_population_full_frame_binding_and_independent_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records"
            salt, source = "fixture-salt", 'account:["askubuntu.com",7]'
            v = int.from_bytes(hashlib.sha256((salt + "\0" + source).encode()).digest(), "big")
            part = "train" if v * 10 < 2**256 * 7 else "calibration" if v * 20 < 2**256 * 17 else "test"
            row = {"record_id": "fixture1", "text": "tiny software fixture", "original_fields": {"site": "askubuntu.com", "OwnerUserId": 7}, "source_unit_id": source, "partition": part}
            audit = {"prepared_rows_sha256": a.digest([row]), "text_binding": a.digest([{k: row[k] for k in ("record_id", "text")}])}
            path.write_text(json.dumps(row) + "\n")
            design = {"seeds": {"source_split_salt": salt}}
            self.assertEqual(a.population_scan(path, audit, design)["rows"], 1)
            row["partition"] = "wrong"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(a.AuditError):
                a.population_scan(path, audit, design)

    def test_paths_cannot_escape_private_source_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            outside = Path(tmp) / "outside"
            outside.write_text("fixture")
            (base / "link").symlink_to(outside)
            with self.assertRaises(a.AuditError):
                a.safe_input(base, "link")


if __name__ == "__main__":
    unittest.main()
