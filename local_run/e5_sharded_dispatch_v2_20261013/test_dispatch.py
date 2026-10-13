"""Small launcher/receipt fixtures only: no models, scheduler or network calls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import dispatch as d


def arguments(mode='plan'):
    values = ['/repo', '/workspace', '/workspace/view', '/workspace/replay.json',
              '/workspace/census', '/workspace/budget', d.CORE_SHA,
              '/workspace/qualification/result.json', 'a' * 64, '/workspace/qualification/full']
    if mode != 'plan':
        values += ['/workspace/qualification/full/plan-run/output/plan.json', 'b' * 64]
    values += ['c' * 64]
    return [mode, *values]


class DispatchTests(unittest.TestCase):
    def test_exact_positional_contract(self):
        for mode, count in [('plan', 11), ('shard', 13), ('assembly', 13)]:
            parsed = d.parse_args(arguments(mode))
            self.assertEqual(len(arguments(mode)) - 1, count)
            self.assertEqual(parsed.dispatch_package_sha, 'c' * 64)
            self.assertEqual(parsed.qualification_sha, 'a' * 64)
            if mode != 'plan': self.assertEqual(parsed.full_plan_sha, 'b' * 64)
            with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                d.parse_args(arguments(mode)[:-1])

    def test_task_ids_and_complete_array_contract(self):
        env = {'SLURM_ARRAY_TASK_COUNT': '32', 'SLURM_ARRAY_TASK_MIN': '0', 'SLURM_ARRAY_TASK_MAX': '31'}
        for index in range(32):
            self.assertEqual(d.task_index({**env, 'SLURM_ARRAY_TASK_ID': str(index)}), index)
        for value in ('', '-1', '32', '0.0', '00', ' 1', 'true'):
            with self.subTest(task=value), self.assertRaises(ValueError):
                d.task_index({**env, 'SLURM_ARRAY_TASK_ID': value})
        for key in env:
            with self.subTest(field=key), self.assertRaises(ValueError):
                d.task_index({**env, key: '1', 'SLURM_ARRAY_TASK_ID': '0'})

    def test_exact_commands_and_ordered_32_shard_assembly(self):
        for mode in ('plan', 'shard', 'assembly'):
            args = d.parse_args(arguments(mode))
            for name in ('repo', 'workspace', 'qualification_result', 'dispatch_root'):
                setattr(args, name, Path(getattr(args, name)))
            stage = args.dispatch_root / 'stage'
            commands = d.commands(args, stage, 3 if mode == 'shard' else None)
            self.assertEqual(len(commands), 2 if mode == 'shard' else 1)
            for name, command, artifact in commands:
                self.assertEqual(command[command.index('--qualification-sha256') + 1], 'a' * 64)
                self.assertEqual(command[command.index('--package-sha256') + 1], d.CORE_SHA)
                self.assertEqual(command[command.index('--qualification-result') + 1], str(args.qualification_result))
            if mode == 'shard':
                self.assertEqual([item[0] for item in commands], ['encode', 'replay'])
                self.assertNotEqual(commands[0][2], commands[1][2])
                command = commands[1][1]
                self.assertEqual(command[command.index('--encoded-dir') + 1], str(stage / 'encode'))
            if mode == 'assembly':
                command = commands[0][1]
                start, stop = command.index('--encoded-dirs') + 1, command.index('--replay-dirs')
                encoded, replayed = command[start:stop], command[stop + 1:]
                self.assertEqual(encoded, [str(args.dispatch_root / 'shards' / f'shard-{i:04d}' / 'encode') for i in range(32)])
                self.assertEqual(replayed, [str(args.dispatch_root / 'shards' / f'shard-{i:04d}' / 'replay') for i in range(32)])

    def test_manifest_member_tampering_traversal_and_symlinks_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = root / 'file.py'; item.write_text('software fixture\n')
            manifest = root / 'package_manifest.json'
            manifest.write_text(json.dumps({'files': {'file.py': d.binding(item)}}))
            digest = d.binding(manifest)['sha256']
            d.verify_package(root, digest)
            item.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'package_member_mismatch'):
                d.verify_package(root, digest)
            manifest.write_text(json.dumps({'files': {'../outside': d.binding(item)}}))
            with self.assertRaisesRegex(ValueError, 'unsafe_package_member'):
                d.verify_package(root, d.binding(manifest)['sha256'])
            alias = root / 'alias'; alias.symlink_to(item)
            manifest.write_text(json.dumps({'files': {'alias': d.binding(item)}}))
            with self.assertRaisesRegex(ValueError, 'package_member_mismatch'):
                d.verify_package(root, d.binding(manifest)['sha256'])

    def test_pinned_qualification_or_plan_hash_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pinned.json'
            path.write_text('{"passed":true}\n')
            expected = d.binding(path)['sha256']
            self.assertEqual(d.verify_file(path, expected), d.binding(path))
            path.write_text('{"passed":false}\n')
            with self.assertRaisesRegex(ValueError, 'pinned_file_SHA256_mismatch'):
                d.verify_file(path, expected)

    def test_whole_child_receipt_preserves_nonzero_exit_and_written_stdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            command = [sys.executable, '-c', 'import sys; print("software fixture"); sys.exit(7)']
            with self.assertRaisesRegex(ValueError, 'child_failed_no_retry_or_following_stage'):
                d.run_child(command, stage, 'encode')
            report = json.loads((stage / 'encode.process.json').read_text())
            self.assertEqual(report['returncode'], 7)
            self.assertNotEqual(report['child_pid'], report['launcher_pid'])
            self.assertGreater(report['child_wall_seconds_through_exit'], 0)
            self.assertTrue(report['timer_includes_child_artifact_publication_and_exit'])
            self.assertEqual((stage / 'encode.stdout.txt').read_text().strip(), 'software fixture')

    def run_fixture(self, root, *, failed=False, missing=False, changed=None):
        args = SimpleNamespace(mode='shard', dispatch_root=root)
        evidence = {key: {'sha256': chr(97 + i) * 64, 'bytes': i + 1}
                    for i, key in enumerate(('dispatch_package', 'core_package', 'qualification_result', 'full_plan'))}
        after = copy.deepcopy(evidence)
        if changed: after[changed]['sha256'] = 'f' * 64
        stage = root / 'shards' / 'shard-0000'
        def fake_commands(unused_args, actual_stage, index):
            self.assertEqual(actual_stage, stage)
            entries = []
            for name in ('encode', 'replay'):
                expected = stage / (name + '.artifact.json')
                if failed and name == 'encode': code = 'raise SystemExit(3)'
                elif missing and name == 'encode': code = 'pass'
                else: code = f'from pathlib import Path; Path({str(expected)!r}).write_text(\'{{"passed":true}}\')'
                entries.append((name, [sys.executable, '-c', code], expected))
            return entries
        with mock.patch.object(d, 'preflight', side_effect=[(0, evidence), (0, after), (0, after)]), \
                mock.patch.object(d, 'commands', side_effect=fake_commands), \
                mock.patch.dict(os.environ, {'SLURM_JOB_ID': '1'}):
            report = d.run(args)
        return args, stage, report

    def test_failure_or_missing_artifact_does_not_run_replay(self):
        for choice in ('failed', 'missing'):
            with self.subTest(kind=choice), tempfile.TemporaryDirectory() as temporary:
                _, stage, report = self.run_fixture(Path(temporary), **{choice: True})
                self.assertFalse(report['passed'])
                self.assertTrue((stage / 'encode.process.json').is_file())
                self.assertFalse((stage / 'replay.process.json').exists())

    def test_any_changed_launch_binding_stops_before_replay(self):
        for key in ('dispatch_package', 'core_package', 'qualification_result', 'full_plan'):
            with self.subTest(binding=key), tempfile.TemporaryDirectory() as temporary:
                _, stage, report = self.run_fixture(Path(temporary), changed=key)
                self.assertFalse(report['passed'])
                self.assertEqual(report['error'], 'launch_bindings_changed_during_child')
                self.assertFalse((stage / 'replay.process.json').exists())

    def test_successful_pair_is_durable_and_existing_stage_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, stage, report = self.run_fixture(Path(temporary))
            self.assertTrue(report['passed'])
            self.assertEqual([r['name'] for r in report['children']], ['encode', 'replay'])
            original = (stage / 'dispatch_result.json').read_bytes()
            with mock.patch.object(d, 'preflight', return_value=(0, {})), self.assertRaisesRegex(ValueError, 'new_stage_directory_required'):
                d.run(args)
            self.assertEqual((stage / 'dispatch_result.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
