#!/usr/bin/env python3
"""Scan new release files, including compressed members, for accidental disclosure.

This version reuses the frozen Phase 5 scanner and excluded News fingerprints.
It covers Phase 6 plus every other changed or new publishable project file.
It does not establish corpus authenticity or guarantee the absence of secrets.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from empirical_execution.phase5 import audit_publication as previous

OUTPUT = ROOT / 'empirical_execution/phase6/results/publication_content_audit.json'
EXCLUDED_GENERATED = {OUTPUT, ROOT / 'BACKUP_MANIFEST.json'}


def candidates():
    commands = [
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard',
         '--', 'empirical_execution/phase6'],
        ['git', 'diff', '--name-only', '-z', 'origin/main'],
        ['git', 'ls-files', '-z', '--others', '--exclude-standard'],
    ]
    names = set()
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
        names.update(p.decode() for p in result.stdout.split(b'\0') if p)
    return sorted(ROOT / name for name in names
                  if (ROOT / name).is_file() and ROOT / name not in EXCLUDED_GENERATED)


def main():
    previous.OUTPUT = OUTPUT
    previous.candidates = candidates
    previous.main()
    report = json.loads(OUTPUT.read_text())
    report['schema'] = 'ccu-publication-content-audit-3'
    report['scope'] = (
        'All publishable Phase 6 files and other changed or new project files. '
        'Includes recursively decompressed gzip, tar, zip, and NPZ members. '
        'This is a targeted scan, not an absence guarantee.'
    )
    report['audit_dependencies'] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(previous.__file__))
    }
    report['candidate_selection'] = 'Git changes against origin/main, new files, and all Phase 6 files'
    report['generated_exclusions'] = {
        'publication_content_audit.json': 'The report cannot include its own hash.',
        'BACKUP_MANIFEST.json': 'Generated file hashes are verified separately to avoid a circular binding.'
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
