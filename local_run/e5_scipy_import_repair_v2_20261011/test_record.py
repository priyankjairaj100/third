"""Only the version2 RECORD ownership regression; never install packages."""
import importlib.metadata as metadata
from pathlib import Path
import tempfile
import unittest

import repair


class RecordOwnershipTests(unittest.TestCase):
    def test_vendored_RECORDs_do_not_ambiguate_owning_distribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            own = root / 'setuptools-78.1.0.dist-info'; own.mkdir()
            (own / 'METADATA').write_bytes(b'Metadata-Version: 2.1\nName: setuptools\nVersion: 78.1.0\n')
            vendored = root / 'setuptools/_vendor/example-1.0.dist-info'
            vendored.mkdir(parents=True)
            (vendored / 'RECORD').write_bytes(b'vendored/data.py,,1\n')
            other_vendor = root / 'setuptools/_vendor/other-2.0.dist-info'
            other_vendor.mkdir(parents=True)
            (other_vendor / 'RECORD').write_bytes(b'other/data.py,,1\n')
            (own / 'RECORD').write_bytes(b'setuptools-78.1.0.dist-info/RECORD,,\nsetuptools/_vendor/example-1.0.dist-info/RECORD,,\nsetuptools/_vendor/other-2.0.dist-info/RECORD,,\n')
            distribution = metadata.PathDistribution(own)
            self.assertEqual(len([p for p in distribution.files if str(p).endswith('.dist-info/RECORD')]), 3)
            result = repair.owning_record(distribution, 'setuptools', '78.1.0')
            self.assertEqual(result, {'record_path': str((own / 'RECORD').resolve()), 'record': repair.binding(own / 'RECORD')})
            (vendored / 'RECORD').write_bytes(b'changed-vendored-record\n')
            self.assertEqual(repair.owning_record(distribution, 'setuptools', '78.1.0'), result)
            (own / 'RECORD').write_bytes(b'changed-own-record\n')
            self.assertNotEqual(repair.owning_record(distribution, 'setuptools', '78.1.0')['record'], result['record'])

    def test_metadata_mismatch_refused_and_missing_owner_never_uses_vendored_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            own = Path(temporary) / 'example-1.0.dist-info'; own.mkdir()
            (own / 'METADATA').write_bytes(b'Name: example\nVersion: 1.0\n')
            (own / 'RECORD').write_bytes(b'example.py,,1\n')
            distribution = metadata.PathDistribution(own)
            for name, version in [('other', '1.0'), ('example', '2.0')]:
                with self.subTest(name=name, version=version), self.assertRaisesRegex(ValueError, 'metadata_identity_mismatch'):
                    repair.owning_record(distribution, name, version)
            (own / 'RECORD').unlink()
            vendored = Path(temporary) / 'example/_vendor/other.dist-info'
            vendored.mkdir(parents=True); (vendored / 'RECORD').write_bytes(b'vendored.py,,1\n')
            self.assertEqual(repair.owning_record(distribution, 'example', '1.0'),
                             {'record_path': None, 'record': None})


if __name__ == '__main__':
    unittest.main()
