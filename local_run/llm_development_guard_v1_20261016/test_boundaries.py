"""Transport-free immutable output and child-process boundary fixtures."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import common as c
import launch
import runner


class Boundaries(unittest.TestCase):
    def test_unbound_package_file_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); (root/'fixture.py').write_text('# fixture\n')
            c.publish(root/'package_manifest.json', {'files': {'fixture.py': c.binding(root/'fixture.py')}})
            pin = c.binding(root/'package_manifest.json')['sha256']
            with patch.object(c, 'HERE', root):
                c.verify_package(pin)
                (root/'unbound.py').write_text('# unexpected\n')
                with self.assertRaisesRegex(ValueError, 'exact_package_inventory'): c.verify_package(pin)

    def test_active_project_and_malformed_queue_rejected(self):
        launch.empty_project_queue('12|unrelated|RUNNING\n')
        for raw in ('12|third-selection|RUNNING\n','12|third-development-guard|PENDING\n','malformed'):
            with self.assertRaises(ValueError): launch.empty_project_queue(raw)

    def test_duplicate_json_and_nonfinite_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError): c.strict(raw)

    def test_publication_never_overwrites_and_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); path = root/'result.json'
            c.write_bytes(path, b'old')
            with self.assertRaises(FileExistsError): c.write_bytes(path, b'new')
            self.assertEqual(path.read_bytes(), b'old')
            with self.assertRaisesRegex(ValueError, 'aggregate_output_cap'):
                c.write_bytes(root/'too-big', b'xx', cap=1)

    def test_failed_child_preserves_own_receipt_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'SLURM_JOB_ID':'999'}):
            root = Path(temporary).resolve()
            with patch.object(c, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'child_failed'):
                    runner.child('fixture', [sys.executable, '-c', 'print("fixture-only"); raise SystemExit(7)'], 'synthetic-pin')
                result = c.strict((root/'fixture_process.json').read_bytes())
                self.assertEqual(result['returncode'], 7)
                self.assertGreater(result['child_peak_rss_kib_linux'], 0)
                with patch.object(runner.subprocess, 'Popen') as start:
                    with self.assertRaises(FileExistsError): runner.child('fixture', ['never-run'], 'synthetic-pin')
                    start.assert_not_called()

    def test_child_log_cap_kills_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'SLURM_JOB_ID':'999'}):
            root = Path(temporary).resolve()
            with patch.object(c, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'child_failed'):
                    runner.child('cap', [sys.executable, '-c', 'import os; os.write(1, b"x"*(5<<20))'], 'synthetic-pin')
                result = c.strict((root/'cap_process.json').read_bytes())
                self.assertTrue(result['console_cap_exceeded'])
                self.assertLessEqual(result['console_bytes'], 4 << 20)


if __name__ == '__main__': unittest.main()
