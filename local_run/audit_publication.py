#!/usr/bin/env python3
"""Scan handoff changes without claiming unavailable News fingerprint replay."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from empirical_execution.phase5.audit_publication import COMPILED

BASELINE = 'ab64c10ef559451bc3f737064bd690fc295b31e5'
OUTPUT = ROOT / 'local_run/review/publication_content_audit.json'


def main():
    names = set()
    for args in (['git','diff','--name-only','-z',BASELINE],
                 ['git','ls-files','-z','--others','--exclude-standard']):
        names.update(p.decode() for p in subprocess.check_output(args,cwd=ROOT).split(b'\0') if p)
    excluded = {'BACKUP_MANIFEST.json', str(OUTPUT.relative_to(ROOT))}
    inventory = []; findings = []
    for name in sorted(names - excluded):
        path = ROOT / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        # No new archive/data export belongs to this textual handoff.
        try:
            raw.decode('utf-8')
        except UnicodeDecodeError:
            findings.append({'path':name,'kind':'unexpected_nontext_change'})
        for key, pattern in COMPILED.items():
            count = sum(1 for _ in pattern.finditer(raw))
            if count:
                findings.append({'path':name,'kind':key,'match_count':count})
        inventory.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    report = {'schema':'ccu-local-publication-audit-1',
        'status':'passed_scoped_text_scan' if not findings else 'needs_review',
        'baseline_commit':BASELINE,'files_checked':len(inventory),'files':inventory,
        'pattern_names':list(COMPILED),'findings':findings,
        'news_fingerprint_scan':'not_performed; intentionally excluded source bodies absent from restored public clone',
        'scope':'Changed and new Git-visible text artifacts, excluding this report and separately verified manifest',
        'raw_corpus_or_human_collection_added':False,
        'no_secrets_guarantee':False,'scientific_readiness_claim':False}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('status','files_checked','findings','news_fingerprint_scan')},indent=2))
    return 2 if findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
