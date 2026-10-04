"""Integration checks on the real Civil100 fixture, with lexical diagnostics only."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "empirical_execution"))
from ccu.data import lexical_engineering_features
from phase3.run_preparation import read_rows, text_binding, file_hash
from phase3.calibration import digest


def main():
    (ROOT / "tmp").mkdir(exist_ok=True)
    cli = ROOT / "empirical_execution/phase3/run_preparation.py"
    calls = []
    def run(*args, expected=0):
        result = subprocess.run([sys.executable, str(cli), *map(str, args)],
                                capture_output=True, text=True)
        assert result.returncode == expected, (result.returncode, result.stdout, result.stderr)
        body = json.loads(result.stdout if expected == 0 else result.stderr)
        calls.append({"command": args[0], "expected_exit": expected, "result": body})
        return body
    with tempfile.TemporaryDirectory(prefix="phase3-workflow-", dir=ROOT / "tmp") as temp:
        temp = Path(temp)
        data = ROOT / "empirical_execution/data/civil_comments_engineering_preview.jsonl"
        prepared = temp / "prepared"
        run("prepare", "--records", data, "--dataset", "civil_comments", "--out", prepared)
        records = prepared / "records.jsonl"
        rows = read_rows(records)
        assert len(rows) == 99
        x, _ = lexical_engineering_features(rows, 128)
        vectors = temp / "vectors.npy"
        np.save(vectors, x)
        provenance = {"dataset_id": "civil_comments_engineering_preview",
            "encoder_id": "lexical_engineering_only", "encoder_revision": "ccu.data.lexical_engineering_features",
            "population_scope": "reused_preview_source_hash_calibration_partition",
            "evidence_role": "engineering_nonconfirmatory"}
        manifest = {"row_ids": [r["record_id"] for r in rows],
            "normalized_records_sha256": text_binding(rows),
            "prepared_rows_sha256": digest(rows),
            "cache_file_sha256": file_hash(vectors), "provenance": provenance}
        cache = temp / "cache_manifest.json"
        cache.write_text(json.dumps(manifest))
        common = ["--records", records, "--vectors", vectors, "--cache-manifest", cache,
                  "--preparation-audit", prepared / "audit.json"]
        pack = temp / "selection"
        body = run("selection", *common, "--block-size", 17, "--out", pack)
        assert body["completed_human_ratings"] == 0
        sample = json.loads((pack / "private_sampling_manifest.json").read_text())
        assert sample["frame"]["shape"] == [16, 128] and sample["pair_population"] == 120
        run("selection", *common, "--out", pack, expected=2)
        before = file_hash(pack / "private_sampling_manifest.json")
        threshold = temp / "threshold.json"
        run("threshold", "--manifest", pack / "private_sampling_manifest.json",
            "--responses", pack / "responses.template.json", "--out", threshold, expected=2)
        assert not threshold.exists()
        assert before == file_hash(pack / "private_sampling_manifest.json")
        # Actual inputs are unchanged; wrong declarations must fail before any output.
        bad = dict(manifest, row_ids=list(reversed(manifest["row_ids"])))
        cache.write_text(json.dumps(bad))
        run("selection", *common, "--out", temp / "wrong_order", expected=2)
        bad = dict(manifest, normalized_records_sha256="0" * 64)
        cache.write_text(json.dumps(bad))
        run("selection", *common, "--out", temp / "wrong_text", expected=2)
        bad = dict(manifest, cache_file_sha256="0" * 64)
        cache.write_text(json.dumps(bad))
        run("selection", *common, "--out", temp / "wrong_cache", expected=2)
        bad = dict(manifest, provenance=dict(provenance, evidence_role="confirmatory_calibration"))
        cache.write_text(json.dumps(bad))
        run("selection", *common, "--out", temp / "lexical_primary", expected=2)
        cache.write_text(json.dumps(manifest))
        changed = [dict(row) for row in rows]
        changed[0]["partition"] = "test" if changed[0]["partition"] != "test" else "train"
        records.write_text("".join(json.dumps(row) + "\n" for row in changed))
        run("selection", *common, "--out", temp / "changed_partition", expected=2)
    output = {"status": "passed", "cli_calls_checked": len(calls), "calls": calls,
        "natural_records": 100, "after_fixed_guard": 99, "calibration_records": 16,
        "calibration_pairs_scored": 120, "human_responses_created": 0,
        "primary_threshold_selected": False, "primary_study_ready": False,
        "scope": "natural-text lexical integration; no synthetic empirical records",
        "code_sha256": {p.name: file_hash(p) for p in [Path(__file__), cli]},
        "data_sha256": file_hash(data)}
    target = ROOT / "empirical_execution/phase3/results/workflow_checks.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "calls"}, indent=2))


if __name__ == "__main__":
    main()
