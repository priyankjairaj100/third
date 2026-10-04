"""Verify the preserved checkpoint files without altering research outputs."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "BACKUP_MANIFEST.json").read_text())
    failures = []
    seen = set()
    for entry in manifest["files"]:
        name = entry["path"]
        path = ROOT / name
        if name in seen or Path(name).is_absolute() or ".." in Path(name).parts:
            failures.append({"path": name, "reason": "invalid or duplicate manifest path"})
            continue
        seen.add(name)
        if not path.is_file():
            failures.append({"path": name, "reason": "missing"})
            continue
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            failures.append({"path": name, "reason": "bytes/hash changed from checkpoint"})
    if (ROOT / ".git").exists():
        raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        tracked = set(filter(None, raw.decode().split("\0"))) - {"BACKUP_MANIFEST.json"}
        if tracked != seen:
            failures.append({"reason": "tracked file set differs from manifest",
                "unlisted": sorted(tracked - seen), "not_tracked": sorted(seen - tracked)})
    result = {"passed": not failures, "files_checked": len(seen), "failures": failures,
              "semantic_study_complete": False,
              "note": "Checks backup consistency, not scientific correctness or corpus/human authenticity"}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
