"""Restore the byte-exact large Phase 4 trace from its portable gzip archive."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
meta = json.loads((ROOT / "empirical_execution/phase4/results/large_result_archive.json").read_text())
source = ROOT / meta["archived_path"]
compressed = source.read_bytes()
if hashlib.sha256(compressed).hexdigest() != meta["archive_sha256"]:
    raise ValueError("Archive checksum mismatch")
data = gzip.decompress(compressed)
if len(data) != meta["original_bytes"] or hashlib.sha256(data).hexdigest() != meta["original_sha256"]:
    raise ValueError("Restored result checksum mismatch")
target = ROOT / meta["original_relative_path"]
if target.exists():
    if target.read_bytes() != data:
        raise FileExistsError("Existing trace differs; preserve it before restoring this checkpoint")
else:
    with target.open("xb") as stream:
        stream.write(data)
print(json.dumps({"status": "verified", "path": str(target), "bytes": len(data), "sha256": meta["original_sha256"]}))
