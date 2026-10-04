#!/usr/bin/env python3
"""Package measured code/results without redistributing CC-News article text."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent / 'output/counterfactual_curation_execution_01.zip'


def eligible(path):
    rel = path.relative_to(ROOT)
    if not path.is_file() or '__pycache__' in rel.parts:
        return False
    if path.name in {'SHA256SUMS.txt', 'PACKAGE_MANIFEST.json'}:
        return False
    if rel.parts[0] == 'data' and not path.name.startswith('civil_comments_'):
        return False
    return path.suffix in {'.py', '.txt', '.json', '.jsonl', '.png', '.svg'}


def main():
    required = [ROOT / 'README.txt', ROOT / 'requirements.txt', ROOT / 'STATUS.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    paths = sorted(path for path in ROOT.rglob('*') if eligible(path))
    for path in paths:
        if path.suffix == '.py':
            compile(path.read_text(), str(path), 'exec')
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    manifest = {
        'archive_scope': 'nonconfirmatory engineering implementation and measured results',
        'files': hashes,
        'omitted': ['CC-News article text and raw connector responses', 'Python caches'],
        'omitted_news_data_available_in_current_workspace': True,
        'civil_only_replay_supported': True,
        'all_computation_performed_locally': True,
    }
    (ROOT / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (ROOT / 'SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for name, digest in hashes.items()))
    paths += [ROOT / 'PACKAGE_MANIFEST.json', ROOT / 'SHA256SUMS.txt']
    with zipfile.ZipFile(DEST, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            archive.write(path, 'empirical_execution/' + str(path.relative_to(ROOT)))
    with zipfile.ZipFile(DEST) as archive:
        assert archive.testzip() is None
        assert not any('/data/cc_news' in name for name in archive.namelist())
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read('empirical_execution/' + name)).hexdigest() == digest
    print(json.dumps({'archive': str(DEST), 'files': len(paths), 'bytes': DEST.stat().st_size,
                      'sha256': hashlib.sha256(DEST.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
