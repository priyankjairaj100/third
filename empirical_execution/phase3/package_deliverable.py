"""Package this preparation stage without News article bodies or stale packs."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]
files = []
for p in sorted((ROOT / "empirical_execution/phase3").rglob("*")):
    if not p.is_file() or "__pycache__" in p.parts or "calibration_engineering_pack_pre_scope_fix" in p.parts:
        continue
    files.append(p)
for rel in [
    "empirical_execution/ccu/__init__.py", "empirical_execution/ccu/core.py", "empirical_execution/ccu/data.py",
    "empirical_execution/data/civil_comments_engineering_preview.jsonl",
    "empirical_execution/DATA_ACCESS_NOTE.txt", "empirical_execution/preview_acquisition_manifest.json",
    "empirical_execution/CURRENT_STATUS.json", "empirical_execution/requirements.txt",
    "empirical_execution/phase2/intake_validate.py", "empirical_execution/phase2/assets_requirements.txt",
    "output/empirical_program/study_design.json",
    "output/empirical_program/counterfactual_curation_empirical_protocol.tex",
    "output/pdf/counterfactual_curation_empirical_preparation.pdf",
]:
    files.append(ROOT / rel)
entries = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
manifest = {"date": "2026-10-04", "scope": "phase3 preparation; primary semantic study blocked",
    "files": entries, "news_article_bodies_included": False,
    "source_models_or_embeddings_included": False, "completed_human_ratings_included": False,
    "notes": "Civil100 is an engineering preview. See acquisition note and README. Manifest excludes its own bytes."}
target = ROOT / "output/counterfactual_curation_empirical_preparation.zip"
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, str(path.relative_to(ROOT)))
    archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    archive.writestr("START_HERE.txt", "Read empirical_execution/phase3/README.txt and output/pdf/counterfactual_curation_empirical_preparation.pdf.\nPrimary semantic results are not present.\n")
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    for entry in entries:
        assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
print(json.dumps({"file": str(target), "bytes": target.stat().st_size,
                  "files": len(files) + 2, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}, indent=2))
