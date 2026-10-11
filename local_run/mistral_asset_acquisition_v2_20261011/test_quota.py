"""Synthetic quota-controller fixtures only; no cluster, network, or models."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import quota_grants as q


class QuotaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / 'mistral-nemo-hub-v2'
        self.cache.mkdir()
        self.files = [{'relative_path': 'config.json', 'size_bytes': 622}]
        self.package_sha = 'a' * 64
        self.request = {'schema': 'third-quota-request-v2', 'job_id': '13300', 'nonce': 'b' * 32,
            'created_unix': 1000, 'cache': str(self.cache.resolve()), 'package_sha256': self.package_sha,
            'expectations_sha256': q.EXPECTATIONS_SHA, 'reserve_bytes': q.RESERVE_BYTES,
            **q.request_spec(0, self.files)}
        self.request_sha = hashlib.sha256(q.canonical(self.request) + b'\n').hexdigest()
        self.quota = {'used_kib': 100, 'soft_kib': 100000000, 'hard_kib': 100000001,
                      'available_soft_bytes': (100000000 - 100) * 1024}
        self.grant = {'schema': 'third-quota-grant-v2', 'request_sha256': self.request_sha,
            'request': copy.deepcopy(self.request), 'observed_unix': 1001, 'published_unix': 1002,
            'quota': self.quota, 'quota_raw_sha256': 'c' * 64, 'quota_raw_file': '01.quota.json',
            'login_mount': q.LOGIN_MOUNT, 'package_sha256': self.package_sha}

    def test_request_initial_total_and_file_identity(self):
        q.validate_request(self.request, self.files, '13300', self.cache, self.package_sha, 1002)
        initial = {**self.request, **q.request_spec(-1, self.files)}
        q.validate_request(initial, self.files, '13300', self.cache, self.package_sha, 1002)
        self.assertEqual(initial['size_bytes'], q.TOTAL_BYTES)
        for field, value in [('size_bytes', 621), ('index', True), ('stage', 'initial_total'),
                             ('relative_path', 'other.json'), ('job_id', '13301'), ('nonce', 'bad'),
                             ('package_sha256', 'd' * 64), ('expectations_sha256', 'd' * 64)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                q.validate_request({**self.request, field: value}, self.files, '13300', self.cache, self.package_sha, 1002)

    def test_stale_future_requests_rejected(self):
        for now in (1301, 994):
            with self.assertRaises(ValueError):
                q.validate_request(self.request, self.files, '13300', self.cache, self.package_sha, now)

    def test_grant_bound_to_exact_request_and_job(self):
        q.validate_grant(self.grant, self.request, self.request_sha, 1003)
        for field, value in [('request_sha256', 'd' * 64), ('package_sha256', 'd' * 64),
                             ('login_mount', q.WORKER_MOUNT), ('request', {**self.request, 'nonce': 'e' * 32})]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                q.validate_grant({**self.grant, field: value}, self.request, self.request_sha, 1003)

    def test_grant_age_120_second_boundary_and_future(self):
        q.validate_grant(self.grant, self.request, self.request_sha, 1121)
        for now in (1121.001, 995):
            with self.assertRaises(ValueError):
                q.validate_grant(self.grant, self.request, self.request_sha, now)
        for field, value in [('observed_unix', float('nan')), ('published_unix', float('inf'))]:
            with self.assertRaises(ValueError):
                q.validate_grant({**self.grant, field: value}, self.request, self.request_sha, 1003)

    def test_quota_units_reserve_and_arithmetic(self):
        raw = 'Disk quotas for user fixture:\n Filesystem blocks quota limit grace files quota limit grace\n/dev/sdb1 100 100000000 100000001 10 0 0\n'
        self.assertEqual(q.parse_login_quota(raw), self.quota)
        minimum = (q.RESERVE_BYTES + self.request['size_bytes'] + 1023) // 1024
        quota = {'used_kib': 0, 'soft_kib': minimum, 'hard_kib': minimum,
                 'available_soft_bytes': minimum * 1024}
        q.validate_grant({**self.grant, 'quota': quota}, self.request, self.request_sha, 1003)
        for bad in ({**quota, 'available_soft_bytes': quota['available_soft_bytes'] + 1},
                    {'used_kib': 0, 'soft_kib': minimum - 1, 'hard_kib': minimum,
                     'available_soft_bytes': (minimum - 1) * 1024}):
            with self.assertRaises(ValueError):
                q.validate_grant({**self.grant, 'quota': bad}, self.request, self.request_sha, 1003)
        for bad in (raw.replace('/dev/sdb1', 'csis.mn1:/nfs_home'), raw + raw, raw.replace('100000000', '-1')):
            with self.assertRaises(ValueError):
                q.parse_login_quota(bad)

    def test_atomic_publication_never_overwrites(self):
        path = self.root / 'receipt.json'
        q.atomic_new(path, self.request)
        self.assertEqual(q.strict(path.read_bytes()), self.request)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            q.atomic_new(path, {'different': True})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob('.*.tmp')), [])

    def test_exact_mount_mapping(self):
        with patch.object(q.subprocess, 'check_output', return_value=json.dumps({'filesystems': [q.WORKER_MOUNT]})):
            self.assertEqual(q.mount_identity(self.cache), q.WORKER_MOUNT)
        with patch.object(q.subprocess, 'check_output', return_value=json.dumps({'filesystems': [q.WORKER_MOUNT, q.LOGIN_MOUNT]})):
            with self.assertRaises(ValueError):
                q.mount_identity(self.cache)

    def test_wait_timeout_is_bounded_without_download(self):
        (self.root / 'quota_requests').mkdir()
        with patch.object(q.time, 'time', return_value=1000), patch.object(q.time, 'monotonic', side_effect=[0, 301]), patch.object(q.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'timed out'):
                q.wait_for_grant(self.root, 0, self.files, self.cache, self.package_sha, '13300')
            sleep.assert_not_called()
        self.assertEqual(len(list((self.root / 'quota_requests').glob('*.request.json'))), 1)

    def test_strict_json_rejects_duplicates_and_nonfinite(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                q.strict(raw)

    def test_worker_accepts_exact_atomic_grant_and_checks_raw_quota(self):
        (self.root / 'quota_requests').mkdir()
        raw = '/dev/sdb1 100 100000000 100000001 10 0 0\n'
        def publish(_delay):
            path = self.root / 'quota_requests/01.request.json'
            request = q.strict(path.read_bytes())
            evidence = path.with_name('01.quota.json')
            q.atomic_new(evidence, {'quota_stdout': raw, 'observed_unix': 1001, 'request_sha256': q.sha(path),
                                   'login_mount': q.LOGIN_MOUNT, 'quota_units': 'KiB from quota -v -w'})
            grant = {**self.grant, 'request': request, 'request_sha256': q.sha(path),
                     'observed_unix': 1001, 'published_unix': 1001, 'quota_raw_sha256': q.sha(evidence)}
            q.atomic_new(path.with_name('01.grant.json'), grant)
        with patch.object(q.time, 'time', return_value=1001), patch.object(q.time, 'sleep', side_effect=publish), redirect_stdout(io.StringIO()):
            grant, digest = q.wait_for_grant(self.root, 0, self.files, self.cache, self.package_sha, '13300')
        self.assertEqual(grant['quota'], self.quota)
        self.assertEqual(digest, q.sha(self.root/'quota_requests/01.grant.json'))

    def test_controller_one_shot_grant_then_idle_preserves_bytes(self):
        import grant_quota
        import package_check
        (self.root / 'quota_requests').mkdir()
        path = self.root / 'quota_requests/01.request.json'
        q.atomic_new(path, self.request)
        expectations = Path(__file__).resolve().parent.parent / 'model_annotation_runtime_mistral_v1_20261011/asset_expectations.json'
        argv = ['grant_quota.py', '--out', str(self.root), '--cache', str(self.cache), '--expectations', str(expectations),
                '--package-sha256', self.package_sha, '--job-id', '13300']
        with patch('sys.argv', argv), patch.object(package_check, 'verify'), patch.object(grant_quota.platform, 'system', return_value='Linux'), patch.object(grant_quota.socket, 'gethostname', return_value='csis.mn1'), patch.object(q, 'mount_identity', return_value=q.LOGIN_MOUNT), patch.object(q.time, 'time', return_value=1002), patch.object(q.subprocess, 'check_output', return_value='/dev/sdb1 100 100000000 100000001 10 0 0\n'), redirect_stdout(io.StringIO()):
            self.assertEqual(grant_quota.main(), 0)
            before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
            self.assertEqual(grant_quota.main(), 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in path.parent.iterdir()})
        grant = q.strict(path.with_name('01.grant.json').read_bytes())
        q.validate_grant(grant, self.request, q.sha(path), 1003)

    def test_raw_quota_provenance_must_match_grant(self):
        evidence = {'quota_stdout': '/dev/sdb1 100 100000000 100000001 10 0 0\n',
            'observed_unix': 1001, 'request_sha256': self.request_sha,
            'login_mount': q.LOGIN_MOUNT, 'quota_units': 'KiB from quota -v -w'}
        q.validate_raw_evidence(evidence, self.grant)
        for field, value in [('observed_unix', 1000), ('request_sha256', 'd' * 64),
                             ('login_mount', q.WORKER_MOUNT), ('quota_units', 'bytes'),
                             ('quota_stdout', '/dev/sdb1 101 100000000 100000001 10 0 0\n')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                q.validate_raw_evidence({**evidence, field: value}, self.grant)
        with self.assertRaises(ValueError):
            q.validate_raw_evidence({**evidence, 'extra': True}, self.grant)


if __name__ == '__main__':
    unittest.main()
