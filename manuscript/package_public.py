#!/usr/bin/env python3
"""Package only the reviewed public source-check closure; never launch research."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def binding(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    if not args.version.isascii() or not args.version.replace('_', '').isalnum():
        raise ValueError('simple version name required')
    archive = ROOT / 'manuscript' / ('release_' + args.version + '.zip')
    manifest_path = ROOT / 'manuscript' / ('release_manifest_' + args.version + '.json')
    if archive.exists() or manifest_path.exists():
        raise ValueError('version already occupied; no overwrite')
    evidence = read('manuscript/evidence_20261016.json')
    author = read('manuscript/claims_review_20261016.json')
    compute = read('manuscript/compute_accounting_20261016.json')
    build = read('manuscript/BUILD_STATUS.json')
    assert build['compile_verified'] and build['source_binding'] == binding(ROOT / 'manuscript/third_working_draft.tex')
    bound = evidence['source_bindings'] + [evidence['manuscript'], evidence['checker']]
    bound += author['historical_source_bindings_reverified'] + author['additional_sources']
    bound += [author['historical_independent_review']] + compute['source_bindings']
    included = set()
    for entry in bound:
        path = entry['path']
        p = PurePosixPath(path)
        assert not p.is_absolute() and '..' not in p.parts and 'local_workspace' not in p.parts
        assert binding(ROOT / path) == {k: entry[k] for k in ['bytes', 'sha256']}, path
        included.add(path)
    included.update(['manuscript/evidence_20261016.json', 'manuscript/BUILD_STATUS.json',
                     'manuscript/README.txt', 'manuscript/REPRODUCIBILITY.txt',
                     'manuscript/package_public.py',
                     'local_run/llm_v4_negative_analysis_v1_20261016/README.txt'])
    # These are explicit source/report roots, not a recursive export of the repo.
    allowed_roots = ('manuscript/', 'empirical_execution/', 'local_run/', 'output/pdf/')
    for name in included:
        path = ROOT / name
        assert name.startswith(allowed_roots) and path.is_file() and not path.is_symlink()
        assert path.suffix in {'.py', '.json', '.md', '.tex', '.txt'}
        assert path.name not in {'TODO.json', 'ANALYSIS_LOCK.json', 'ANALYSIS_LOCK_EXECUTION.json'}
    manifest = {'schema': 'third-public-source-check-package-v1',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Public manuscript and author source-check closure; not full empirical replay.',
                'version': args.version,
                'historical_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'files': {name: binding(ROOT / name) for name in sorted(included)},
                'private_source_text_or_pairs_included': False,
                'model_weights_included': False, 'compiled_pdf_included': False,
                'primary_empirical_replay_available': False, 'submission_ready': False}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    manifest_member = manifest_path.relative_to(ROOT).as_posix()
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(included | {manifest_member}):
            raw = (ROOT / name).read_bytes()
            info = zipfile.ZipInfo(name, (2026, 10, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, raw)
    with tempfile.TemporaryDirectory(prefix='third-public-package-check-') as temp:
        temp = Path(temp)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert set(z.namelist()) == included | {manifest_member}
            z.extractall(temp)
        for name, expected in manifest['files'].items():
            assert binding(temp / name) == expected, name
        result = subprocess.run([sys.executable, '-B', 'manuscript/build_evidence.py'],
                                cwd=temp, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr
        replay = json.loads((temp / 'manuscript/evidence_20261016.json').read_text())
        original = dict(evidence)
        for report in [replay, original]:
            report.pop('created_utc')
        assert replay == original, 'public replay changed evidence beyond creation time'
    status = {'schema': 'third-public-package-build-status-v1',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'archive': {'path': archive.relative_to(ROOT).as_posix(), **binding(archive)},
              'manifest': {'path': manifest_member, **binding(manifest_path)},
              'payload_members': len(included), 'archive_members': len(included) + 1,
              'archive_crc_and_exact_member_hashes_passed': True,
              'public_source_check_in_clean_extraction_passed': True,
              'reproduced_evidence_equal_except_creation_time': True,
              'source_check_summary': json.loads(result.stdout),
              'native_compile_previously_verified_for_packaged_source': True,
              'visual_review_verified': False, 'full_empirical_replay_tested': False,
              'submission_ready': False, 'builder': binding(Path(__file__))}
    (ROOT / 'manuscript/RELEASE_BUILD_STATUS.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
