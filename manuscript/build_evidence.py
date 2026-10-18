#!/usr/bin/env python3
"""Bind this working manuscript to existing evidence; never launch research.

This is an author source/claim check, not an independent scientific review.
Only public reports and theory sources are read. No source texts or labels.
"""
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from fractions import Fraction
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REPORT = "local_run/local_llm_execution_20261014/"


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def binding(relative):
    data = (ROOT / relative).read_bytes()
    return {"path": relative, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main():
    tex = (HERE / "third_working_draft.tex").read_text()
    citations = {key for group in re.findall(r"\\cite\{([^}]+)\}", tex)
                 for key in group.split(",")}
    bib = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex)
    assert len(bib) == len(set(bib)), "duplicate bibliography key"
    assert len(labels) == len(set(labels)), "duplicate label"
    assert citations <= set(bib), "undefined citation"
    assert set(refs) <= set(labels), "undefined reference"
    assert set(bib) <= citations, "unused bibliography entry"
    assert "\\input{" not in tex and "\\include{" not in tex
    assert "\u2014" not in tex and "---" not in tex.replace("$a\\;---\\;b\\;---\\;c$", "")

    v3 = read_json(REPORT + "llm_selection_v3_terminal_result.json")
    intake = read_json(REPORT + "llm_development_intake_completed_20261016.json")
    pilot = read_json(REPORT + "llm_guard_benchmark_completed_20261016.json")
    locks = read_json(REPORT + "llm_v4_qwen_audit_and_task_locks_20261016.json")
    review = read_json(REPORT + "task_and_audit_independent_review_20261016.json")
    plan = read_json("local_run/model_annotation_v4_20261014/PLAN.json")
    v4 = read_json(REPORT + "llm_selection_v4_terminal_result.json")
    negative = read_json(REPORT + "llm_v4_negative_diagnostics_20261016.json")
    assert v4["execution_complete"] and v4["all51_exact_weighted_cuts_independently_recomputed"]
    assert (v4["selection_pairs"], v4["conservative_positives_of360"], v4["model_requests_attempted"]) == (360, 27, 792)
    assert v4["selection_passed"] is False and v4["selected_tau"] is None
    assert v4["independent_validation_performed"] is False and v4["no_retuning"]
    assert not v4["threshold_dependent_downstream_authorized"]
    assert negative["separate_author_arithmetic_replay_passed"]
    assert negative["all51_full_cut_records_exactly_recomputed"]
    assert negative["maximum_weighted_joint_positive_proportion"] == {"numerator": 2, "denominator": 5}
    assert negative["maximum_missing_uncertain_sensitivity"] == {"numerator": 8, "denominator": 15}
    assert negative["both_valid_pairs"] == 355 and negative["valid_six_way_disagreements"] == 308
    assert negative["models"]["qwen"]["canonical_positive_count"] == 32
    assert negative["models"]["mistral"]["canonical_positive_count"] == 151
    assert negative["swap_diagnostics"]["qwen"]["status_or_category_changes"] == 3
    assert negative["swap_diagnostics"]["qwen"]["binary_positive_changes"] == 1
    assert negative["swap_diagnostics"]["mistral"]["status_or_category_changes"] == 14
    assert negative["swap_diagnostics"]["mistral"]["binary_positive_changes"] == 6
    # Verify the actual displayed rounded table cells against the exact ledger.
    cuts = {Fraction(r["tau"]["numerator"], r["tau"]["denominator"]): r for r in negative["grid_ledger"]}
    for display in [".90", ".95", ".98", ".99", ".995"]:
        row = cuts[Fraction(display)]
        values = [100 * Fraction(row[k]["numerator"], row[k]["denominator"])
                  for k in ["conservative_precision", "missing_uncertain_sensitivity_upper"]]
        cell = f'{display} & {row["sample_support"]} & {float(values[0]):.2f} & {float(values[1]):.2f}'
        assert cell in tex, "calibration table mismatch"
    assert v3["execution_complete"] and v3["all51_exact_weighted_cuts_independently_recomputed"]
    assert (v3["selection_pairs"], v3["conservative_positives_of90"], v3["model_requests_attempted"]) == (90, 0, 198)
    assert v3["selection_passed"] is False and v3["independent_validation_performed"] is False
    assert v3["selected_tau"] is None and v3["model_retries"] == 0
    allocation = intake["result"]["allocation"]
    expected = {"development_train": 10314, "development_test": 2138,
                "evaluation_train": 87873, "evaluation_test": 18929, "calibration": 20996}
    assert {k: v["rows"] for k, v in allocation.items()} == expected
    assert sum(expected.values()) == 140250
    assert sum(v["source_units"] for v in allocation.values()) == 88344
    assert sum(v["unknown_singleton_rows"] for v in allocation.values()) == 1522
    assert intake["independent_review"]["passed"]
    assert intake["result"]["all_reserved_sources_excluded_from_evaluation"]
    assert intake["result"]["effect_experiments_started"] is False
    assert intake["result"]["development_threshold_accepted"] is False
    assert pilot["result"]["pairs"] == 262144
    assert pilot["result"]["full_development_pairs"] == 10314 * (20996 + 21067) == 433837782
    assert pilot["result"]["full_development_guard_executed"] is False
    assert round(pilot["result"]["core_wall_seconds"], 4) == 0.7783
    assert all(x["exact_witness_score_keep_parity"] for x in pilot["result"]["exact_original_guard_parity"])
    assert [x["threshold"] for x in pilot["result"]["exact_original_guard_parity"]] == [0.5, 0.99, 0.995]
    assert locks["Qwen_audit"]["status_counts"] == {"valid": 391, "invalid_json": 5}
    assert locks["Qwen_audit"]["model_calls_repeated"] == 0
    assert locks["native_task_configuration"]["selected_ridge_lambda"] == 1e-5
    assert locks["native_task_configuration"]["native_tag_outputs"] == 20
    assert locks["native_task_configuration"]["test_labels_used"] is False
    assert review["passed"] and review["verified"]["source_group_folds"] == 5
    assert review["verified"]["lambda_candidates"] == 6
    assert len(plan["selection"]["threshold_grid"]) == 51
    assert plan["selection"]["precision_min"] == "19/20"
    assert plan["validation"]["validation_lower_bound_min"] == 0.9
    assert plan["validation"]["validation_pair_max"] == 200
    assert plan["resources"]["human_annotation_calls"] == 0
    assert "zero human annotation and zero human review" in tex
    assert "No completed natural-text repair/utility/cost experiment" in tex
    literature = read_json("manuscript/literature_contract_review_20261016.json")
    citation_sources = read_json("manuscript/related_work_sources_20261016.json")
    assert {entry["key"] for entry in citation_sources["sources"]} == set(bib)
    assert set(literature["primary_artifacts"]) == {"arcane", "ticketed", "space"}
    assert all(item["downloaded"] for item in literature["primary_artifacts"].values())
    assert literature["novelty_clearance"] is False
    assert all(item["gap_closed"] is False for item in literature["unresolved_routes"])
    author_review = read_json("manuscript/claims_review_20261016.json")
    compute = read_json("manuscript/compute_accounting_20261016.json")
    assert author_review["review_passed_in_stated_scope"] and not author_review["new_independent_proof_review"]
    assert author_review["manuscript"] == binding("manuscript/third_working_draft.tex")
    assert len(author_review["resolved_findings"]) == 4
    assert len(author_review["historical_source_bindings_reverified"]) == 18
    for saved in author_review["historical_source_bindings_reverified"] + author_review["additional_sources"]:
        assert saved == binding(saved["path"]), "review source binding drift"
    assert compute["gpu_allocation_seconds"] == 19045
    assert compute["cpu_job_elapsed_seconds_sum"] == 1469
    assert compute["project_total_compute_complete"] is False
    assert compute["failed_audit_cost_retained"] and not compute["batch_step_times_added"]
    for saved in compute["source_bindings"]:
        assert saved == binding(saved["path"]), "compute source binding drift"
    attribution = read_json("manuscript/attribution_review_20261016.json")
    for saved in attribution["source_bindings"]:
        assert saved == binding(saved["path"]), "attribution source binding drift"
    models = {m["model_id"]: m for m in attribution["models"]}
    e5 = models["intfloat/multilingual-e5-base"]
    assert e5["revision"] == read_json("local_run/e5_sharded_cpu_v2_20261013/expectations.json")["revision"]
    assert e5["publisher_declared_license"] == "mit"
    for judge in plan["inherits_unchanged"]["models"]:
        assert models[judge["model_id"]]["revision"] == judge["revision"]
        assert models[judge["model_id"]]["publisher_declared_license"] == "apache-2.0"
    assert len(models) == 3 and all(not m["weights_redistributed"] for m in models.values())
    assert not attribution["source_policy"]["raw_text_release_authorized_by_this_review"]
    qualification = read_json("local_run/source_acceptance_v2_20261011/qualification.json")
    for gate in ["archive_attribution_review_completed", "primary_input_acceptance",
                 "publisher_bytes_cryptographically_authenticated", "confirmatory_study_ready"]:
        assert attribution["preserved_gates"][gate] is False
        assert qualification["flags"][gate] is False
    assert not attribution["adapter_review"]["full_record_metadata_coverage_verified_in_this_review"]
    assert not attribution["replay_availability"]["public_checker_checks_private_labels"]
    assert not attribution["replay_availability"]["threshold_dependent_relaunch_authorized"]
    assert "intfloat/multilingual-e5-base" in tex
    history = read_json("manuscript/compute_history_20261016.json")
    for saved in history["source_bindings"] + [history["builder"]]:
        assert saved == binding(saved["path"]), "historical compute source binding drift"
    checked = subprocess.run([sys.executable, "-B", str(HERE / "build_compute_history.py"), "--check"],
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stderr
    assert history["job_count"] == 72 and history["state_counts"] == {"COMPLETED": 68, "FAILED": 4}
    assert history["existing_v4_jobs_included_once"] and not history["project_total_compute_complete"]
    for number in [history["six_model_job_gpu_allocation_seconds"],
                   history["embedding_array_elapsed_seconds"],
                   history["groups"]["other_stage"]["elapsed_seconds_sum"]]:
        assert f"{number:,}" in tex, "historical compute manuscript count differs"
    assert "266.39 requested-CPU hours" in tex
    unified = read_json("manuscript/unified_review_20261016.json")
    independent = read_json("manuscript/independent_review_20261018.json")
    assert independent["reviewer_kind"] == "independent AI scientific manuscript review"
    assert independent["human_review"] is False and independent["followup_completed"]
    assert independent["current_manuscript"] == binding("manuscript/third_working_draft.tex")
    assert {f["id"] for f in independent["findings"]} == {"F1", "F2"}
    assert all(f["independent_followup_resolved"] for f in independent["findings"])
    assert independent["submission_ready"] is False and independent["novelty_clearance"] is False
    # The author review and first independent report remain bound to their OLD
    # manuscript. Verify that basis from the immutable prior public package,
    # rather than rebinding a historical review to a revised document.
    assert unified["manuscript"] == independent["reviewed_manuscript"]
    archive = independent["historical_input_archive"]
    assert archive == binding(archive["path"])
    assert len(independent["historical_input_bindings"]) == 20
    with zipfile.ZipFile(ROOT / archive["path"]) as prior:
        for saved in independent["historical_input_bindings"]:
            raw = prior.read(saved["path"])
            assert len(raw) == saved["bytes"] and hashlib.sha256(raw).hexdigest() == saved["sha256"]
    for saved in independent["source_bindings"] + independent["review_artifacts"]:
        assert saved == binding(saved["path"]), "independent review source/artifact drift"
    followup = read_json("manuscript/independent_review_followup_20261018.json")
    assert followup["manuscript"] == independent["current_manuscript"]
    assert {f["id"] for f in followup["findings"]} == {"F1", "F2"}
    assert all(f["status"] == "resolved" for f in followup["findings"])
    assert followup["regressions_found"] == [] and not followup["scientific_claim_scope_changed"]
    assert not followup["author_checker_and_packaging_reviewed"]
    assert {k: followup["report"][k] for k in ["bytes", "sha256"]} == {
        k: binding("manuscript/independent_review_followup_20261018.txt")[k]
        for k in ["bytes", "sha256"]}
    for saved in followup["frozen_implementation_sources"]:
        assert saved == binding(saved["path"]), "follow-up implementation binding drift"
    arithmetic_review = read_json("manuscript/independent_review_checks_20261018.json")
    assert [{k: saved[k] for k in ["path", "bytes", "sha256"]}
            for saved in arithmetic_review["pinned_sources"]] == independent["historical_input_bindings"]
    assert unified["review_passed_in_stated_scope"] and not unified["new_independent_review"]
    assert unified["novelty_clearance"] is False
    for saved in unified["source_bindings"]:
        assert saved == binding(saved["path"]), "unified review source binding drift"
    assert unified["primary_artifact"]["key"] == "ivmunlearning"
    assert all(not gap["gap_closed"] for gap in unified["remaining_gaps"])

    sources = [
        "output/pdf/counterfactual_curation_theory.tex",
        "empirical_execution/canonical_theory_addendum.tex",
        "empirical_execution/phase10/finite_word_theory.tex",
        "empirical_execution/phase10/FINITE_WORD_THEORY.md",
        "empirical_execution/phase10/SCORER_TRANSFER.md",
        "empirical_execution/phase10/FINITE_WORD_REVIEW.md",
        "empirical_execution/phase10/CODEC.md",
        "empirical_execution/phase10/CODEC_REVIEW.md",
        "empirical_execution/phase10/CONTRIBUTION_POSITION.md",
        "empirical_execution/phase10/LITERATURE_FOLLOWUP.md",
        "local_run/EMPIRICAL_PROGRAM.md",
        "local_run/model_annotation_v4_20261014/PLAN.json",
        "local_run/source_acceptance_v2_20261011/qualification.json",
        REPORT + "llm_selection_v3_terminal_result.json",
        REPORT + "llm_v4_qwen_audit_and_task_locks_20261016.json",
        REPORT + "task_and_audit_independent_review_20261016.json",
        REPORT + "llm_development_intake_completed_20261016.json",
        REPORT + "llm_guard_benchmark_completed_20261016.json",
        REPORT + "e5_calibration_frame_result.json",
        "manuscript/literature_contract_review_20261016.json",
        "manuscript/related_work_sources_20261016.json",
        REPORT + "llm_selection_v4_terminal_result.json",
        REPORT + "llm_v4_negative_diagnostics_20261016.json",
        "local_run/llm_v4_negative_analysis_v1_20261016/analyze.py",
        "manuscript/claims_review_20261016.json",
        "manuscript/compute_accounting_20261016.json",
        "manuscript/build_review.py",
        "manuscript/attribution_review_20261016.json",
        "empirical_execution/phase4/adapters.py",
        "empirical_execution/phase4/ADAPTERS_README.txt",
        "empirical_execution/phase4/embeddings.py",
        "local_run/e5_sharded_cpu_v2_20261013/expectations.json",
        "manuscript/compute_history_20261016.json",
        "manuscript/compute_receipts_extract_20261016.json",
        "manuscript/build_compute_history.py",
        "manuscript/unified_review_20261016.json",
        "manuscript/independent_review_20261018.json",
    ]
    sources += [entry["path"] for entry in independent["review_artifacts"]]
    sources += [archive["path"]]
    sources += [entry["path"] for entry in independent["source_bindings"]
                if entry["path"] not in sources]
    claims = [
        {"sections": ["sec:setup", "sec:method", "app:work"],
         "claim": "Fixed ordered curator, finite-horizon polynomial repair, indexed work, ridge and partition-source extension",
         "sources": sources[:2], "limit": "Exact algebraic designated state; construction, solves, export and bit complexity additional. No full-refit guarantee."},
        {"sections": ["sec:memory", "app:scorer", "app:information", "app:certificate"],
         "claim": "Stored FP32 family, frozen FP64 scorer, exact and approximate private-information bounds, canonical finite-family codec",
         "sources": sources[2:9], "limit": "Constructed rows and rational head; no natural semantic, total-byte, physical erasure or uncertified floating-release claim."},
        {"sections": ["sec:study", "sec:design"],
         "claim": "Actual source/cache/task preparation and source-disjoint reservation",
         "sources": [sources[12], sources[14], sources[15], sources[16], sources[18]],
         "limit": "Separate fully LLM development; pre-guard upper bounds; historical human and primary acceptance remain unmet."},
        {"sections": ["sec:judges"], "claim": "Frozen v4 two-model selection and fresh validation design",
         "sources": [sources[11]], "limit": "Prospective model-defined criterion, not semantic truth or independent model votes; failure stops configuration."},
        {"sections": ["sec:evidence", "tab:calibration"], "claim": "Completed v3 and v4 negatives, audited model outputs, exact selection replay, order diagnostics, and guard resource pilot",
         "sources": [sources[13], sources[14], sources[15], sources[17], sources[21], sources[22], sources[23]],
         "limit": "All51 v4 cuts failed; no fresh validation or full guard/effect study. Sensitivity is imputation, not a confidence bound; raw stratified sample counts are not population prevalence."},
        {"sections": ["sec:evidence", "sec:limits"], "claim": "Earlier lexical total-byte comparison favored B-E",
         "sources": [sources[8]], "limit": "Historical lexical development remains distinct from the current natural semantic study."},
        {"sections": ["sec:related", "app:repro"], "claim": "Source-located comparison of ARCANE selection, ticketed access and bounded-deletion memory",
         "sources": [sources[19], sources[20]],
         "limit": "Primary-text contract comparison only; two full-text gaps remain. No independent audit of prior proofs or exhaustive novelty clearance."},
        {"sections": ["sec:method", "sec:memory", "app:work", "app:scorer", "app:repro"],
         "claim": "Author claim comparison, including corrected kernel/service distinction, historical review source checks, and partial v4 compute accounting",
         "sources": [sources[24], sources[25], sources[26]],
         "limit": "No new independent proof review, visual layout verification, complete project compute/energy budget or empirical effect evidence."},
        {"sections": ["app:repro"],
         "claim": "Pinned model identity and publisher license declarations, supplied metadata retention and public/private replay boundaries",
         "sources": sources[27:32] + [sources[11], sources[12]],
         "limit": "Documentary and code review only; no complete record-level attribution census, legal clearance, source/model redistribution, bitwise inference portability or changed acceptance gate."},
        {"sections": ["app:repro"],
         "claim": "Expanded72-job historical accounting with six model jobs,32 CPU embedding tasks,34 other stages and four preserved failures",
         "sources": sources[32:35],
         "limit": "Explicit saved subset, not complete project cost. Array and model launch excerpts require private originals for extraction replay; public arithmetic is reproducible. Requested CPU times elapsed is not utilization or billed core time."},
        {"sections": ["sec:related", "sec:evidence", "sec:limits"],
         "claim": "Unified author read-through, completed negative-calibration framing and explicit credit for preprocessing-aware unlearning as view maintenance",
         "sources": [sources[35], sources[20], sources[13], sources[21]],
         "limit": "Bounded author consistency review, not independent proof review or exhaustive novelty clearance. Two close-prior full texts and visual layout remain unverified."},
        {"sections": ["sec:method", "app:work"],
         "claim": "Independent AI manuscript review and verified correction of retry semantics and service request overhead",
         "sources": sources[36:],
         "limit": "Manual AI review of specified claims and public aggregates, not formal proof, private-evidence authentication, complete literature clearance or rendered-layout review. Original report and old manuscript basis are preserved."},
    ]
    report = {
        "schema": "third-working-manuscript-evidence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manuscript": binding("manuscript/third_working_draft.tex"),
        "checker": binding("manuscript/build_evidence.py"),
        "source_claim_check_passed": True,
        "reviewer_kind": "author AI assistant source and arithmetic check; not independent manuscript review",
        "standalone_source": True,
        "bibliography_entries": len(bib), "references_checked": len(refs),
        "source_bindings": [binding(s) for s in sources],
        "claims": claims,
        "fixed_cutoff": "16 October 2026; terminal v4 negative at09:52:01UTC; downstream work blocked under this configuration",
        "human_annotation_calls": 0, "human_review_required": False,
        "historical_human_gates_unchanged": True,
        "primary_semantic_study_started": False,
        "submission_ready": False,
        "not_verified_here": ["Native compilation and visual layout are tracked separately in BUILD_STATUS.json",
                              "The checker binds the separate independent AI review; it does not perform or reproduce that reasoning", "No native solver certificate",
                              "No model agreement validation, effect experiment, or held-out utility",
                              "No full literature novelty clearance", "No permission for source text redistribution"],
    }
    (HERE / "evidence_20261016.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"source_claim_check_passed": True, "source_bindings": len(sources),
                      "bibliography_entries": len(bib), "manuscript_sha256": report["manuscript"]["sha256"]}))


if __name__ == "__main__":
    main()
