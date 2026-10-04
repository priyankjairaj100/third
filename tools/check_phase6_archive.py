#!/usr/bin/env python3
"""Restore a real result archive and check refusal of a changed destination."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import phase6_result_archive as archive


def main():
    name = 'dispatch_candidate1'
    source_archive = archive.ARCHIVES / (name + '.tar.gz')
    source_manifest = archive.ARCHIVES / (name + '.json')
    original_results = archive.RESULTS
    with tempfile.TemporaryDirectory(prefix='phase6-restore-check-') as temporary:
        archive.RESULTS = Path(temporary) / 'results'
        archive.ARCHIVES = archive.RESULTS / 'archives'
        archive.ARCHIVES.mkdir(parents=True)
        shutil.copyfile(source_archive, archive.ARCHIVES / source_archive.name)
        shutil.copyfile(source_manifest, archive.ARCHIVES / source_manifest.name)
        restored = archive.verify(name, restore=True)
        verified = archive.verify(name)
        manifest = json.loads(source_manifest.read_text())
        changed = archive.RESULTS / manifest['files'][0]['path']
        changed.write_bytes(b'changed destination for refusal check')
        refused = False
        try:
            archive.verify(name, restore=True)
        except ValueError:
            refused = True
        assert refused
    result = {'status': 'passed', 'real_archive_restored': name,
              'restored_files': restored['verified_files'],
              'all_restored_bytes_reverified': verified['all_bytes_verified'],
              'differing_destination_refused': refused,
              'original_results_modified': False,
              'archive_sha256': hashlib.sha256(source_archive.read_bytes()).hexdigest(),
              'sources': {str(p.relative_to(archive.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(archive.__file__))}}
    output = original_results / 'archive_restore_check.json'
    if output.exists():
        raise FileExistsError('Preserve the existing restoration check')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
