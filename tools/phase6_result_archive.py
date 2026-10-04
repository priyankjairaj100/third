#!/usr/bin/env python3
"""Pack or restore a complete Phase 6 result directory without losing files.

Packing never removes the source directory. Existing archives are immutable.
Restoring refuses differing files, links, and paths outside the result directory.
"""
from pathlib import Path, PurePosixPath
import argparse
import gzip
import hashlib
import io
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'empirical_execution/phase6/results'
ARCHIVES = RESULTS / 'archives'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def paths(name):
    if not name or Path(name).name != name or name in {'.', '..', 'archives'}:
        raise ValueError('Use one result directory name')
    return RESULTS / name, ARCHIVES / (name + '.tar.gz'), ARCHIVES / (name + '.json')


def pack(name):
    source, archive, manifest = paths(name)
    if archive.exists() or manifest.exists():
        raise FileExistsError('Preserve the existing archive')
    if source.is_symlink() or not source.is_dir():
        raise ValueError('A regular result directory is required')
    files = []
    for item in sorted(source.rglob('*')):
        if item.is_symlink():
            raise ValueError('Symbolic links are not supported')
        if item.is_file():
            files.append(item)
        elif not item.is_dir():
            raise ValueError('Only regular files are supported')
    if not files:
        raise ValueError('No result files found')
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    entries = []
    with archive.open('xb') as handle:
        with gzip.GzipFile(fileobj=handle, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for item in files:
                    raw = item.read_bytes()
                    relative = item.relative_to(RESULTS).as_posix()
                    info = tarfile.TarInfo(relative)
                    info.size = len(raw)
                    info.mode = 0o644
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(raw))
                    entries.append({'path': relative, 'bytes': len(raw), 'sha256': sha(raw)})
    report = {'schema': 'ccu-phase6-result-archive-1', 'directory': name,
              'archive': archive.name, 'archive_bytes': archive.stat().st_size,
              'archive_sha256': sha(archive.read_bytes()), 'file_count': len(entries),
              'uncompressed_file_bytes': sum(row['bytes'] for row in entries),
              'files': entries, 'results_omitted': False,
              'archive_tool_sha256': sha(Path(__file__).read_bytes())}
    manifest.write_text(json.dumps(report, indent=2) + '\n')
    return verify(name)


def verify(name, restore=False):
    source, archive, manifest = paths(name)
    report = json.loads(manifest.read_text())
    raw = archive.read_bytes()
    if report['directory'] != name or sha(raw) != report['archive_sha256']:
        raise ValueError('Archive identity or checksum differs')
    if len(raw) != report['archive_bytes']:
        raise ValueError('Archive byte count differs')
    expected = {row['path']: row for row in report['files']}
    if len(expected) != report['file_count']:
        raise ValueError('Duplicate manifest paths or inconsistent file count')
    seen = set()
    verified = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or not path.parts or path.parts[0] != name
                    or member.name in seen or member.name not in expected):
                raise ValueError('Unsafe or unexpected member')
            data = tar.extractfile(member).read()
            entry = expected[member.name]
            if sha(data) != entry['sha256'] or len(data) != entry['bytes']:
                raise ValueError('Member bytes differ')
            target = RESULTS / member.name
            if any(parent.is_symlink() for parent in (target, *target.parents)):
                raise ValueError('A destination component is a symbolic link')
            if target.exists() and (not target.is_file() or sha(target.read_bytes()) != entry['sha256']):
                raise ValueError('Existing result differs; preserve both versions')
            verified.append((target, data))
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Missing archive members')
    if restore:
        for target, data in verified:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as handle:
                    handle.write(data)
    return {'directory': name, 'verified_files': len(seen), 'all_bytes_verified': True,
            'archive_bytes': len(raw), 'restore_requested': restore}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['pack', 'verify', 'restore'])
    parser.add_argument('directory')
    args = parser.parse_args()
    result = pack(args.directory) if args.action == 'pack' else verify(args.directory, args.action == 'restore')
    print(json.dumps(result, indent=2))
