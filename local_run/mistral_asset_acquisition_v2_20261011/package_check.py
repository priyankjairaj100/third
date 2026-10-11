"""Minimal standard-library package verifier used before acquisition helpers."""
import hashlib
import json
from pathlib import Path


def verify(folder, expected):
    folder = Path(folder)
    raw = (folder / 'package_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Acquisition package manifest mismatch')
    files = json.loads(raw)['files']
    if not {'acquire.py', 'grant_quota.py', 'quota_grants.py', 'package_check.py', 'acquire.sbatch'}.issubset(files):
        raise ValueError('Incomplete acquisition package manifest')
    for name, item in files.items():
        if Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('Unsafe package member')
        data = (folder / name).read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Acquisition package member mismatch: ' + name)
