"""Refusal checks for the provisional dossier review; no human data fabricated."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "empirical_execution"))
from phase3.calibration import _seal
from phase4.study_lock import empty_template, validate_execution_lock


def main():
    names = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        names.append(name)
    template = empty_template()
    for value in (None, [], {}, template, _seal(template)):
        report = validate_execution_lock(value, {})
        check("missing dossier refused " + str(len(names)), not report["execution_allowed"]
              and not report["mechanical_integrity_passed"] and len(report["blockers"]) > 10)
    with tempfile.TemporaryDirectory(prefix="ccu-lock-software-check-") as directory:
        base = Path(directory)
        resources = {"threads": 1, "per_method_memory_cap_bytes": 100,
                     "common_timeout_seconds": 10, "timing_repetitions": 5,
                     "fixed_before_confirmation": True, "execution_location": "current_workspace"}
        for field, invalid in (("threads", 1.5), ("timing_repetitions", True),
                               ("common_timeout_seconds", float("inf")),
                               ("per_method_memory_cap_bytes", 0)):
            path = base / "resources.json"
            path.write_text(json.dumps({**resources, field: invalid}))
            lock = empty_template()
            lock["artifacts"]["resource_policy"] = {"path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            report = validate_execution_lock(_seal(lock), {}, base_dir=base)
            check("invalid resource " + field, "resource." + field in report["blockers"])
        lock = empty_template()
        lock["artifacts"]["original_corpus"] = {"path": "../outside", "sha256": "0" * 64}
        report = validate_execution_lock(_seal(lock), {}, base_dir=base)
        check("path escape refused", "artifact.original_corpus" in report["blockers"])
    report = validate_execution_lock(_seal(empty_template()), {})
    check("all lineage limitations explicit", len(report["unimplemented_validations"]) == 3
          and not report["external_provenance_verified_by_software"]
          and not report["confirmatory_claims_established"])
    output = {"status": "passed", "check_count": len(names), "checks": names,
        "scope": "software refusal paths only; no completed calibration or human ratings",
        "study_lock_code_sha256": hashlib.sha256((Path(__file__).parent / "study_lock.py").read_bytes()).hexdigest()}
    target = Path(__file__).parent / "results/study_lock_checks.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
