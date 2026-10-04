#!/usr/bin/env python3
"""Apply the frozen disclosure scanner to this checkpoint's changed files."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from empirical_execution.phase5 import audit_publication as scanner

OUTPUT = ROOT / "empirical_execution/phase8/results/publication_content_audit.json"


def candidates():
    names = set()
    commands = [
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "empirical_execution/phase8"],
        ["git", "diff", "--name-only", "-z", "origin/main"],
        ["git", "ls-files", "-z", "--others", "--exclude-standard"],
    ]
    for command in commands:
        raw = subprocess.check_output(command, cwd=ROOT)
        names.update(x.decode() for x in raw.split(b"\0") if x)
    excluded = {OUTPUT, ROOT / "BACKUP_MANIFEST.json"}
    return sorted(ROOT / name for name in names if (ROOT / name).is_file() and ROOT / name not in excluded)


if __name__ == "__main__":
    scanner.OUTPUT = OUTPUT
    scanner.candidates = candidates
    scanner.main()
    report = json.loads(OUTPUT.read_text())
    report["schema"] = "ccu-publication-content-audit-phase8-1"
    report["scope"] = "All publishable Phase8 files and other changed or new project files; compressed members included"
    report["wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["generated_exclusions"] = ["This report", "BACKUP_MANIFEST.json; verified separately"]
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
