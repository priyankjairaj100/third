"""Tiny software archives only; no natural records or source-authenticity claims."""
import contextlib
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import py7zr
import archive_reader as reader


class ReaderTests(unittest.TestCase):
    def make_archive(self, directory, items):
        path = Path(directory) / 'fixture.7z'
        with py7zr.SevenZipFile(path, 'w', filters=[{'id': py7zr.FILTER_COPY}]) as archive:
            for name, data in items:
                archive.writestr(data, name)
        return path

    def command(self, directory, args):
        return subprocess.run([sys.executable, str(Path(reader.__file__).resolve()), *args],
                              cwd=directory, capture_output=True, check=False,
                              env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})

    def test_actual_archive_exact_stdout_both_members_no_extraction_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = [('Posts.xml', b'<fixture>\x00\xff\n\r\n</fixture>'),
                     ('PostLinks.xml', b'<fixture links="1"/>'), ('unselected.txt', b'never output')]
            path = self.make_archive(tmp, items)
            original = path.read_bytes()
            for member, data in items[:2]:
                result = self.command(tmp, ['x', '-so', '-y', str(path), member])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, data)
                self.assertEqual(result.stderr, b'')
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual({p.name for p in Path(tmp).iterdir()}, {'fixture.7z'})

    def test_actual_empty_archive_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_archive(tmp, [('Posts.xml', b'')])
            result = self.command(tmp, ['x', '-so', '-y', str(path), 'Posts.xml'])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, b'')
        # An archive may mark an empty member as an empty stream, skipping the
        # worker's CRC-stage seek; MemIO.close then supplies the only marker.
        writer = reader.StdoutWriter(io.BytesIO(), 0)
        writer.seek(0)
        writer.close()

    def test_actual_compressed_solid_archive_streams_selected_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'compressed.7z'
            payload = bytes(range(256)) * 4096
            with py7zr.SevenZipFile(path, 'w') as archive:
                archive.writestr(b'unselected fixture prefix', 'other.txt')
                archive.writestr(payload, 'Posts.xml')
                archive.writestr(b'<links/>', 'PostLinks.xml')
            before = path.read_bytes()
            result = self.command(tmp, ['x', '-so', '-y', str(path), 'Posts.xml'])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, payload)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual({p.name for p in Path(tmp).iterdir()}, {'compressed.7z'})

    def test_exact_command_allowlist_and_missing_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_archive(tmp, [('Posts.xml', b'fixture')])
            for args in [[], ['l', '-so', '-y', str(path), 'Posts.xml'],
                         ['x', '-so', '-y', str(path), '../Posts.xml'],
                         ['x', '-so', '-y', str(path), 'PostLinks.xml'],
                         ['x', '-so', '-y', str(path), 'Posts.xml', 'extra']]:
                result = self.command(tmp, args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b'')

    def test_actual_duplicate_member_refused_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_archive(tmp, [('Posts.xml', b'one'), ('Posts.xml', b'two')])
            result = self.command(tmp, ['x', '-so', '-y', str(path), 'Posts.xml'])
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, b'')
            self.assertEqual(result.stderr.strip(), b'missing_or_duplicate_member')

    def test_actual_crc_corruption_and_truncation_are_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = b'UNIQUE_TINY_CRC_FIXTURE_12345'
            path = self.make_archive(tmp, [('Posts.xml', payload)])
            original = path.read_bytes()
            self.assertEqual(original.count(payload), 1)
            for broken in [original.replace(payload, payload[:-1] + b'X'), original[:25]]:
                path.write_bytes(broken)
                result = self.command(tmp, ['x', '-so', '-y', str(path), 'Posts.xml'])
                self.assertEqual(result.returncode, 2)
                self.assertNotEqual(result.stderr, b'')
                self.assertEqual(path.read_bytes(), broken)
                self.assertEqual({p.name for p in Path(tmp).iterdir()}, {'fixture.7z'})

    def test_writer_terminal_marker_never_rewinds_or_reopens(self):
        out = io.BytesIO()
        writer = reader.StdoutWriter(out, 3)
        with self.assertRaises(reader.ReaderError): writer.seek(0)
        writer.write(b'abc')
        with self.assertRaises(reader.ReaderError): writer.seek(1)
        with self.assertRaises(reader.ReaderError): writer.seek(-1, 1)
        writer.seek(0)
        with self.assertRaises(reader.ReaderError): writer.write(b'')
        with self.assertRaises(reader.ReaderError): writer.read()
        writer.seek(0)
        writer.close()
        self.assertEqual(out.getvalue(), b'abc')
        self.assertEqual(writer.size(), 3)
        with self.assertRaises(reader.ReaderError): writer.seek(0)
        with self.assertRaises(reader.ReaderError): writer.close()

    def test_length_checks_and_factory_single_exact_stream(self):
        writer = reader.StdoutWriter(io.BytesIO(), 2)
        with self.assertRaises(reader.ReaderError): writer.write(b'abc')
        writer.write(b'a')
        with self.assertRaises(reader.ReaderError): writer.close()
        factory = reader.SingleMemberFactory('Posts.xml', 2, io.BytesIO())
        with self.assertRaises(reader.ReaderError): factory.create('PostLinks.xml')
        with self.assertRaises(reader.ReaderError): factory.create('./Posts.xml')
        factory.create('Posts.xml')
        with self.assertRaises(reader.ReaderError): factory.create('Posts.xml')

    def test_short_stdout_writes_are_completed(self):
        class ShortWriter(io.BytesIO):
            def write(self, value): return super().write(value[:2])
        out = ShortWriter()
        writer = reader.StdoutWriter(out, 7)
        self.assertEqual(writer.write(b'1234567'), 7)
        self.assertEqual(out.getvalue(), b'1234567')

    def test_python_diagnostics_do_not_enter_member_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_archive(tmp, [('Posts.xml', b'fixture')])
            original_extract = py7zr.SevenZipFile.extract
            def noisy_extract(archive, *args, **kwargs):
                print('software diagnostic')
                return original_extract(archive, *args, **kwargs)
            output, errors = io.BytesIO(), io.StringIO()
            with mock.patch.object(py7zr.SevenZipFile, 'extract', noisy_extract), contextlib.redirect_stderr(errors):
                reader.stream_member(path, 'Posts.xml', output)
            self.assertEqual(output.getvalue(), b'fixture')
            self.assertEqual(errors.getvalue(), 'software diagnostic\n')

    def test_version_and_runtime_receipt_bind_actual_modules(self):
        receipt = reader.runtime_binding()
        self.assertEqual(receipt['py7zr_version'], '1.1.3')
        self.assertEqual(receipt['adapter']['sha256'], hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest())
        self.assertTrue(any(x['path'].endswith('/py7zr/io.py') for x in receipt['module_and_distribution_files']))
        for item in receipt['module_and_distribution_files']:
            self.assertEqual(item['sha256'], hashlib.sha256(Path(item['path']).read_bytes()).hexdigest())
        with mock.patch.object(reader.importlib.metadata, 'version', return_value='wrong'):
            with self.assertRaises(reader.ReaderError): reader.check_version()


if __name__ == '__main__':
    unittest.main()
