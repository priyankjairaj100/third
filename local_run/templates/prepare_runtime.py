#!/usr/bin/env python3
"""Record actual installed packages; do not assert semantic compatibility."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

REQUIRED = ("numpy", "torch", "transformers", "tokenizers", "safetensors")
OTHER = ("scipy", "scikit-learn", "joblib", "threadpoolctl", "idna", "sentencepiece", "faiss-cpu", "faiss-gpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New private directory; existing directories are refused.")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in REQUIRED + OTHER:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    missing = [name for name in REQUIRED if versions[name] is None]
    report = {
        "schema": "ccu-local-runtime-observation-1",
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "missing_encoder_packages": missing,
        "semantic_runtime_qualified": False,
        "model_execution_performed": False,
        "note": "Installed metadata does not prove imports, compatibility, numerical replay, or model provenance.",
    }
    (out / ".gitignore").write_text("*\n", encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not missing:
        (out / "encoder_versions.json").write_text(
            json.dumps({name: versions[name] for name in REQUIRED}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "missing_encoder_packages": missing, "semantic_runtime_qualified": False}))
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
