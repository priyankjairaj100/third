"""Calendar/cohort software verification; fixture evidence is never real coverage."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "empirical_execution"))
from phase4 import adapters as a, news_calendar as n
from phase4 import embeddings
from phase3 import calibration as cal
from phase3.panels import digest_partition, hash_integer


def main():
    checks = []
    def check(name, truth):
        assert truth, name
        checks.append(name)
    def refuse(name, fn):
        try:
            fn()
        except (ValueError, FileExistsError):
            checks.append(name)
        else:
            raise AssertionError(name)
    complete = {"months": {month: {"status": "complete", "basis": "SOFTWARE metadata fixture only"} for month in n.MONTHS}}
    result = n.calendar_decision({"2018-01": 9000, "2018-02": 12000, "2018-03": 15000}, complete)
    check("Earliest qualifying whole month precedes prefix rule", result["selected_months"] == ["2018-02"] and result["overshoot"] == 2000)
    result = n.calendar_decision({"2018-01": 4500, "2018-02": 6000}, complete)
    check("Otherwise complete chronological prefix includes full boundary month", result["selected_months"] == ["2018-01", "2018-02"] and result["selected_count"] == 10500)
    gap = json.loads(json.dumps(complete)); gap["months"]["2018-02"]["status"] = "incomplete"
    result = n.calendar_decision({"2018-01": 4500, "2018-02": 6000, "2018-03": 7000}, gap)
    check("Prefix stops at completeness gap", result["selected_months"] == ["2018-01"] and result["prefix_stopped_at_coverage_gap"] == "2018-02" and not result["target_met"])
    result = n.calendar_decision({"2018-01": 4500, "2018-02": 20000, "2018-03": 15000}, gap)
    check("Incomplete month cannot qualify by high count", result["selected_months"] == ["2018-03"])
    result = n.calendar_decision({}, complete)
    check("Zero population reports all-month shortfall", result["selected_count"] == 0 and result["shortfall"] == 10000 and len(result["selected_months"]) == 24)
    refuse("Negative count refused", lambda: n.calendar_decision({"2018-01": -1}, complete))
    refuse("Incomplete coverage map refused", lambda: n.calendar_decision({}, {"months": {}}))
    with tempfile.TemporaryDirectory(prefix="ccu-news-calendar-software-") as folder:
        root = Path(folder)
        psl = root / "SOFTWARE_PSL.dat"
        psl.write_text("// ===BEGIN ICANN DOMAINS===\ncom\n// ===END ICANN DOMAINS===\n// ===BEGIN PRIVATE DOMAINS===\n// ===END PRIVATE DOMAINS===\n")
        groups = {}
        for i in range(100):
            domain = f"software{i}.com"
            part = digest_partition(hash_integer("ccu-v1-source-split", "domain:" + domain))
            groups.setdefault(part, domain)
            if len(groups) == 3:
                break
        rows = []
        for part in ("train", "calibration", "test"):
            for year in (2017, 2018):
                rows.append({"title": "SOFTWARE FIXTURE ONLY", "text": "same test text" if part != "test" else "reserved test",
                    "domain": groups[part], "url": "https://" + groups[part] + "/" + str(year), "date": f"{year}-01-01 00:00:00"})
        # A second analysis record survives the exact calibration-text guard.
        rows.append(dict(rows[1], title="SOFTWARE DIFFERENT QUESTION", text="different test text", date="2018-02-01 00:00:00"))
        raw = root / "SOFTWARE_NEWS_SCHEMA.jsonl"
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        a.adapt_news(raw, root / "adapted", psl_path=psl, psl_sha256=a.file_sha256(psl))
        evidence = root / "SOFTWARE_EVIDENCE.txt"
        evidence.write_text("SOFTWARE FIXTURE ONLY. These coverage declarations are not acquisition evidence for any corpus.")
        coverage = dict(complete, schema="ccu-news-calendar-coverage-1", input_file_sha256=a.file_sha256(raw),
                        completeness_unit="all_records_in_pinned_extraction_for_recorded_month",
                        evidence=[{"path": evidence.name, "sha256": a.file_sha256(evidence)}])
        coverage_path = root / "SOFTWARE_COVERAGE.json"
        coverage_path.write_text(json.dumps(coverage))
        result = n.prepare_news_cohorts(root / "adapted", coverage_path, root / "cohorts")
        prepared = [json.loads(line) for line in (root / "cohorts/records.jsonl").read_text().splitlines()]
        check("Source-date cohort preserves exact allocations", result["counts"]["calibration_rows"] == 1 and result["counts"]["analysis_rows"] == 2 and result["counts"]["excluded:test_domain_reserved"] == 2)
        check("Cross-year same source never crosses cohort boundary", result["cross_cohort_source_overlap"] == 0)
        check("Exact-text guard retains calibration removes training", len(prepared) == 2 and result["counts"]["exact_guard_removed_training_rows"] == 1 and prepared[0]["partition"] == "calibration")
        check("Streaming full-row digest matches canonical list digest", result["prepared_rows_sha256"] == cal.digest(prepared))
        check("Streaming text digest matches canonical list digest", result["normalized_records_sha256"] == cal.digest([{k: row[k] for k in ("record_id", "text")} for row in prepared]))
        cal._verify(result)
        encoded_rows, encoding_audit = embeddings.load_prepared_records(root / "cohorts/records.jsonl", root / "cohorts/audit.json")
        check("News cohort accepted by explicit embedding bridge", encoded_rows == prepared and encoding_audit["schema"] == "ccu-news-preparation-1")
        check("Calendar remains presemantic provisional", result["provisional_calendar_counts"]["status"] == "provisional_after_fixed_guard_before_semantic_guard_not_locked_window" and not result["confirmatory_study_ready"])
        refuse("Coverage cannot bind a different snapshot", lambda: n.load_coverage(coverage_path, "0"*64))
        missing = dict(coverage, evidence=[])
        missing_path = root / "missing.json"; missing_path.write_text(json.dumps(missing))
        refuse("Coverage needs evidence not counts alone", lambda: n.load_coverage(missing_path, a.file_sha256(raw)))
        evidence.write_text("CHANGED SOFTWARE EVIDENCE")
        refuse("Changed acquisition evidence refused", lambda: n.load_coverage(coverage_path, a.file_sha256(raw)))
        fail_audit = root / "failed_audit.json"
        fail_audit.write_text(json.dumps(cal._seal({"stage": "fixed_guards_done_semantic_guard_pending"})))
        fail_quality = root / "failed_quality.json"
        fail_quality.write_text(json.dumps(cal._seal({"schema": "ccu-calibration-quality-report-1", "statistical_and_declared_scope_eligible": False})))
        refuse("No final calendar before semantic guard and quality pass", lambda: n.select_post_guard_window(root / "cohorts/records.jsonl", coverage_path, a.file_sha256(raw), fail_audit, fail_quality, root / "forbidden_window"))
    result = {"schema": "ccu-news-calendar-software-audit-1", "passed": len(checks), "checks": checks,
              "code_sha256": a.file_sha256(n.__file__), "adapter_code_sha256": a.file_sha256(a.__file__),
              "software_fixtures_only": True, "real_calendar_coverage_verified": False,
              "synthetic_empirical_datasets_used": False, "confirmatory_study_ready": False}
    output = ROOT / "empirical_execution/phase4/results/news_calendar_checks.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks_passed": len(checks), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
