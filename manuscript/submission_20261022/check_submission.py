#!/usr/bin/env python3
"""Check the current source and anonymous V5 package without private data."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile

if sys.version_info < (3, 10) or not __debug__:
    raise SystemExit('Use Python 3.10+ without -O.')
here = Path(__file__).resolve().parent
source = here.parent / 'third_working_draft.tex'
receipt = json.loads((here/'package_receipt.json').read_text())
data = source.read_bytes()
assert hashlib.sha256(data).hexdigest() == receipt['source']['sha256']
archive = here / 'third_tmlr_supplement_v5.zip'
assert archive.stat().st_size == receipt['supplement']['bytes']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['supplement']['sha256']
with tempfile.TemporaryDirectory(prefix='third-submission-check-') as tmp:
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.read('manuscript.tex') == data
        # This pinned six-file wrapper has no paths or symlinks.
        assert set(z.namelist()) == {'README.txt', 'SOURCE_CHANGE.json', 'MANIFEST.json',
                                    'manuscript.tex', 'check_all.py', 'evidence_v4.zip'}
        z.extractall(tmp)
    p = subprocess.run([sys.executable, '-B', str(Path(tmp)/'check_all.py')],
                       cwd=tmp, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
review = json.loads((here/'independent_submission_review.json').read_text())
assert review['passed'] is True
assert review['current_source_sha256'] == receipt['source']['sha256']
assert review['supplement_sha256'] == receipt['supplement']['sha256']
assert review['pdf_sha256'] == receipt['pdf']['sha256']
print(json.dumps({'passed': True, 'source_matches_upload_package': True,
                  'nested_aggregate_replay_passed': True,
                  'independent_submission_review_passed': True,
                  'author_declarations_verified': False,
                  'submission_occurred': False}, indent=2))
