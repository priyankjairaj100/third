"""Refresh the initial/project checkpoint inventory after staging intended files."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "BACKUP_MANIFEST.json"


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    paths = sorted(p for p in raw.decode().split("\0") if p and p != TARGET.name)
    files = [{"path": p, "bytes": (ROOT / p).stat().st_size,
              "sha256": checksum(ROOT / p)} for p in paths]
    excluded = []
    for path in sorted((ROOT / "empirical_execution/data").glob("cc_news*")):
        if path.is_file():
            excluded.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": checksum(path), "reason": "News body redistribution excluded by project packaging policy"})
    current_status = json.loads((ROOT / "empirical_execution/CURRENT_STATUS.json").read_text())
    result = {"schema": "ccu-project-backup-1", "checkpoint_date": current_status["date"],
        "repository": "https://github.com/priyankjairaj100/third", "branch": "main",
        "scientific_status": current_status["stage"],
        "files": files, "file_count_excluding_manifest": len(files),
        "total_bytes_excluding_manifest": sum(x["bytes"] for x in files),
        "excluded_input_files": excluded,
        "other_exclusions": ["git internal state", "Python bytecode/cache directories",
            "temporary renders and redundant extraction folders", "TeX build intermediates"],
        "rescued_temporary_authored_sources": "archive/RESCUED_SOURCES.json",
        "manifest_self_hash": "omitted to avoid self-reference; Git commit identifies the manifest",
        "scope": "All preserved authored research/code/report/result/history files, not a full chat transcript or original corpus backup"}
    TARGET.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("file_count_excluding_manifest", "total_bytes_excluding_manifest")}, indent=2))


if __name__ == "__main__":
    main()
