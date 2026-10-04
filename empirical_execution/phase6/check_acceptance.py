"""Natural-preview replay and adversarial software checks. No semantic results."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase3 import calibration as cal, panels, run_preparation as prep
from phase4 import adapters, embeddings as emb
from phase6 import acceptance as a


def run(out):
    outcomes = []

    def check(name, truth):
        if not truth:
            raise AssertionError(name)
        outcomes.append({"check": name, "passed": True})

    def refused(name, fn):
        try:
            fn()
        except (ValueError, KeyError, RuntimeError, FileNotFoundError, TypeError):
            check(name, True)
        else:
            raise AssertionError("Expected refusal: " + name)

    natural = ROOT / "data/civil_comments_engineering_preview.jsonl"
    design = emb.read_json(ROOT.parent / "output/empirical_program/study_design.json")
    bindings = a.code_bindings()
    with tempfile.TemporaryDirectory(prefix="phase6-acceptance-") as temporary:
        t = Path(temporary)
        directory = t / "adapter"
        adapters.adapt_civil(natural, directory, input_schema="engineering_preview")
        spec = {"dataset_id": "civil_comments", "adapter": {"directory": str(directory),
                "inputs": {"records": str(natural)}, "kwargs": {"input_schema": "engineering_preview"}}}
        rows, links, report = a.replay_adapter(spec)
        check("all_100_natural_rows_replayed", len(rows) == 100 and report["all_derived_rows_replayed"])
        check("no_original_sources_invented", all(r["source_kind"] == "unknown_singleton" for r in rows))
        check("no_duplicate_links_invented", links == [])
        check("original_input_digest_bound", report["input_bindings"]["records"]["sha256"] == emb.file_hash(natural))
        check("original_fractional_targets_preserved", [r["labels"][0] for r in rows] ==
              [r["fields"]["toxicity"] for r in prep.read_rows(natural)])
        group = {"group_id": "software-natural-preview", "corpus": "civil_comments"}
        response = a.accept_source_cache(group, {"source_acceptance": spec}, design)
        check("preview_cannot_activate_primary", not response["execution_allowed"] and not response["machine_acceptance_passed"])
        check("preview_has_explicit_blocker", response["blockers"] == ["engineering_preview_is_not_an_original_source_rich_archive"])
        check("preview_records_successful_byte_replay", response["checks"] == [{"check": "original_archive_replay", "passed": True}])
        check("preview_binding_present", len(response["replay_binding_sha256"]) == 64)
        check("no_corpus_authenticity_assertion", not response["source_authenticity_verified"])
        check("no_human_authenticity_assertion", not response["human_authenticity_verified"])
        prepared_rows, panel_ids, audit = panels.prepare_population(rows, "civil_comments", design, target=5)
        audit.update({"input_records_file_sha256": emb.file_hash(directory / "records.jsonl"),
                      "study_design_sha256": emb.file_hash(ROOT.parent / "output/empirical_program/study_design.json"),
                      "stage": "fixed_guards_done_semantic_guard_pending"})
        prepared_dir = t / "prepared"
        prep.write_population(prepared_dir, prepared_rows, panel_ids, audit)
        spec["prepared"] = {"records": str(prepared_dir / "records.jsonl"),
                            "audit": str(prepared_dir / "audit.json"), "target": 5}
        replayed, source_report = a.replay_preparation(spec, design, rows, links)
        check("complete_fixed_preparation_replayed", replayed == prepared_rows)
        check("source_hash_partitions_recomputed", source_report["unknown_source_units"] == len(prepared_rows))
        check("genuine_source_count_remains_zero", source_report["genuine_source_units"] == 0)
        for dataset in ("civil_comments", "askubuntu", "cc_news", "english_stackexchange"):
            for target in (10000, 25000, 100000, 200000):
                contract = a.panel_contract({"panel": "primary-" + str(target)}, {"dataset_id": dataset})
                check("registered_target_" + dataset + "_" + str(target),
                      contract == {"target": target, "pool": "primary", "calendar": False})
        contract = a.panel_contract({"panel": "second-disjoint-10000"}, {"dataset_id": "civil_comments"})
        check("replication_pool_is_explicit", contract["pool"] == "replication" and contract["target"] == 10000)
        contract = a.panel_contract({"panel": "complete-calendar-window"}, {"dataset_id": "cc_news"})
        check("calendar_uses_complete_window", contract == {"target": 10000, "pool": "calendar", "calendar": True})
        refused("tiny_registered_target_refused", lambda: a.panel_contract({"panel": "primary-5"}, {"dataset_id": "civil_comments"}))
        refused("calendar_on_labeled_corpus_refused", lambda: a.panel_contract({"panel": "complete-calendar-window"}, {"dataset_id": "civil_comments"}))
        refused("wcep_source_panel_refused", lambda: a.panel_contract({"panel": "primary-25000"}, {"dataset_id": "wcep100"}))
        contract = a.panel_contract({"panel": "whole-events-25000"}, {"dataset_id": "wcep100"})
        check("wcep_registered_target", contract["target"] == 25000 and contract["pool"] == "whole_events")
        contract = a.panel_contract({"panel": "separate-refit-5000"}, {"dataset_id": "civil_comments"})
        check("refit_excludes_largest_primary_and_second_panel", contract["exclude_primary_target"] == 200000 and contract["exclude_replication_target"] == 10000)
        refit_ids, refit_report = a.separate_refit_panel(prepared_rows, design, contract)
        check("natural_preview_refit_shortfall_preserved", refit_report["shortfall"] == 5000 and refit_ids["separate_refit"] == [])
        # Abstract set-membership fixtures are software tests, never corpus data.
        abstract_rows = [{"record_id": f"unit-{i}-{j}", "source_unit_id": f"source-{i}", "partition": "train"}
                         for i in range(40) for j in range(3)]
        abstract_contract = {**contract, "target": 10, "exclude_primary_target": 3, "exclude_replication_target": 3}
        refit_ids, refit_report = a.separate_refit_panel(abstract_rows, design, abstract_contract)
        check("refit_whole_source_boundary_software", len(refit_ids["separate_refit"]) == 12 and refit_report["overshoot"] == 2)
        check("refit_positive_source_disjoint_software", refit_report["no_selected_primary_or_replication_source_reused"])
        check("refit_source_units_complete_software", len(refit_report["selected_source_units"]) == 4)
        a._equal({"features": np.eye(2, dtype=np.float32)}, {"features": np.eye(2, dtype=np.float32)}, "Separate array dictionaries")
        check("independent_array_dictionary_loads_compare_safely", True)
        # Vocabulary identity fixtures contain no records or human judgments.
        vocabulary = {"sha256": "software-vocabulary-a"}
        threshold = {"vocabulary_lock_sha256": vocabulary["sha256"], "lambda_lock_sha256": "software-lambda-a"}
        learner = {"calibration": {"vocabulary_lock": vocabulary},
                   "lambda_lock": {"vocabulary_lock_sha256": vocabulary["sha256"]},
                   "tag_threshold_lock": threshold}
        with patch.object(a.model_selection, "select_tag_decision_thresholds", return_value=threshold):
            a.bind_stack_vocabulary({"learners": {"e5": copy.deepcopy(learner)}}, vocabulary)
            check("positive_shared_stack_output_binding_software", True)
            changed = copy.deepcopy(learner); changed["calibration"]["vocabulary_lock"] = {"sha256": "different"}
            refused("different_learner_vocabulary_refused", lambda: a.bind_stack_vocabulary({"learners": {"e5": changed}}, vocabulary))
            changed = copy.deepcopy(learner); changed["lambda_lock"]["vocabulary_lock_sha256"] = "different"
            refused("different_lambda_vocabulary_refused", lambda: a.bind_stack_vocabulary({"learners": {"e5": changed}}, vocabulary))
            changed = copy.deepcopy(learner); changed["tag_threshold_lock"] = {**threshold, "lambda_lock_sha256": "different"}
            refused("different_tag_decision_lock_refused", lambda: a.bind_stack_vocabulary({"learners": {"e5": changed}}, vocabulary))
            changed = copy.deepcopy(learner); changed["projection_parameters"] = {"64": copy.deepcopy(learner)}
            changed["projection_parameters"]["64"]["calibration"]["vocabulary_lock"] = {"sha256": "different"}
            refused("different_projected_vocabulary_refused", lambda: a.bind_stack_vocabulary({"learners": {"e5": changed}}, vocabulary))
            changed = copy.deepcopy(learner); changed["logistic_threshold_lock"] = {"vocabulary_lock_sha256": "different"}
            refused("different_logistic_vocabulary_refused", lambda: a.bind_stack_vocabulary({"learners": {"e5": changed}}, vocabulary))
        corrupted = copy.deepcopy(prepared_rows)
        corrupted[0]["source_unit_id"] = "account:imputed"
        refused("source_imputation_refused", lambda: a._independent_sources(corrupted, "civil_comments", design))
        corrupted = copy.deepcopy(prepared_rows)
        corrupted[0]["partition"] = "test" if corrupted[0]["partition"] != "test" else "train"
        refused("partition_reassignment_refused", lambda: a._independent_sources(corrupted, "civil_comments", design))
        source_bytes = (directory / "records.jsonl").read_bytes()
        changed = copy.deepcopy(rows)
        changed[0]["labels"] = [1 - changed[0]["labels"][0]]
        (directory / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in changed))
        refused("changed_labels_refused", lambda: a.replay_adapter(spec))
        (directory / "records.jsonl").write_bytes(source_bytes)
        old_audit = (directory / "audit.json").read_bytes()
        changed_audit = json.loads(old_audit)
        changed_audit["settings"]["input_schema"] = "tfds_1_2_4"
        (directory / "audit.json").write_text(json.dumps(changed_audit))
        refused("schema_relabel_refused", lambda: a.replay_adapter(spec))
        (directory / "audit.json").write_bytes(old_audit)
        relabelled = copy.deepcopy(spec)
        relabelled["adapter"]["kwargs"]["input_schema"] = "tfds_1_2_4"
        refusal = a.accept_source_cache(group, {"source_acceptance": relabelled}, design)
        check("preview_cannot_parse_as_original", not refusal["machine_acceptance_passed"])
        check("missing_native_metadata_reported", "Missing Civil columns" in refusal["blockers"][0])
        check("missing_source_spec_fails_closed", not a.accept_source_cache(group, {}, design)["machine_acceptance_passed"])
        missing = copy.deepcopy(spec)
        missing["adapter"]["inputs"]["records"] = str(t / "missing-original.jsonl")
        refusal = a.accept_source_cache(group, {"source_acceptance": missing}, design)
        check("missing_actual_original_bytes_refused", not refusal["machine_acceptance_passed"] and not refusal["execution_allowed"])
        refused("absent_real_cache_refused", lambda: a.replay_cache({"directory": str(t / "missing-model-cache")}, prepared_rows, spec["prepared"]))
        failcache = t / "failedcache"
        failcache.mkdir()
        (failcache / "FAILED.json").write_text("{}")
        (failcache / "cache_manifest.json").write_text("{}")
        (failcache / "assets.json").write_text("{}")
        refused("failed_cache_refused", lambda: a.replay_cache({"directory": str(failcache)}, prepared_rows, spec["prepared"]))
        refused("encoder_fp32_drift_refused", lambda: a._array(np.array([1], np.float32), np.array([np.nextafter(np.float32(1), np.float32(2))], np.float32), "cache"))
        refused("encoder_dtype_drift_refused", lambda: a._array(np.array([1], np.float32), np.array([1], np.float64), "cache"))
        evidence = t / "evidence.json"
        evidence.write_text('{"approved":true}')
        bound = a._external_evidence({"external_evidence": [{"path": str(evidence), "sha256": emb.file_hash(evidence), "purpose": "software trust test"}]}, ".")
        check("approval_boolean_not_authenticated", not bound[0]["authenticated_by_software"])
        refused("changed_external_evidence_refused", lambda: a._external_evidence({"external_evidence": [{"path": str(evidence), "sha256": "0" * 64, "purpose": "software trust test"}]}, "."))
        final_replay, _, _ = a.replay_adapter(spec)
        check("source_input_unchanged_after_checks", final_replay == rows)
        # Positive algorithm-path test. Only the transformer and its absent
        # asset loader are replaced. All cache rows, logs, and summaries replay.
        # These temporary vectors are software fixtures, never semantic outputs.
        class FixtureTokenizer:
            def encode(self, text, **kwargs):
                return [ord(c) + 1 for c in text]
            def num_special_tokens_to_add(self, pair=False):
                return 2
            def prepare_for_model(self, ids, **kwargs):
                value = [1] + ids + [2]
                return {"input_ids": value, "attention_mask": [1] * len(value)}

        class FixtureBackend:
            tokenizer = FixtureTokenizer()
            max_length = 512
            model_bytes = 0
            model_buffer_bytes = 0
            def __call__(self, chunks):
                answer = np.zeros((len(chunks), 768), np.float32)
                for index, chunk in enumerate(chunks):
                    answer[index, 0] = 1
                    answer[index, 1] = sum(chunk["input_ids"]) % 17 + 1
                return answer

        fixture = t / "mock_encoder_software_only"
        fixture.mkdir()
        fixture_rows = prepared_rows[:3]
        fixture_prepared_dir = t / "mock_prepared_software_only"
        prep.write_population(fixture_prepared_dir, fixture_rows, {},
            {"stage": "fixed_guards_done_semantic_guard_pending", "dataset_id": "civil_comments",
             "scope": "temporary_subset_for_mock_backend_software_test"})
        fixture_prepared = {"records": str(fixture_prepared_dir / "records.jsonl"),
                            "audit": str(fixture_prepared_dir / "audit.json")}
        backend = FixtureBackend()
        vectors, details = [], []
        logs = []
        for index, row in enumerate(fixture_rows):
            vector, detail = emb.encode_document(backend.tokenizer, backend, row["text"], emb.E5, 512, batch_size=8)
            vectors.append(vector); details.append(detail)
            logs.append({"row_index": index, "record_id": row["record_id"],
                         "text_sha256": a.hashlib.sha256(row["text"].encode()).hexdigest(), **detail})
        np.save(fixture / "vectors.npy", np.asarray(vectors, np.float32), allow_pickle=False)
        emb.write_json(fixture / "row_ids.json", [r["record_id"] for r in fixture_rows])
        (fixture / "chunk_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
        summary = {"rows": len(fixture_rows), "dimension": 768, "model_parameter_bytes": 0, "model_buffer_bytes": 0,
            "output_npy_bytes": (fixture / "vectors.npy").stat().st_size,
            "content_tokens": sum(d["content_tokens"] for d in details), "chunks": sum(d["chunks"] for d in details),
            "maximum_record_content_tokens": max(d["content_tokens"] for d in details),
            "maximum_record_chunks": max(d["chunks"] for d in details),
            "first_chunk_baseline_truncated_records": sum(d["first_chunk_baseline_omitted_tokens"] > 0 for d in details),
            "first_chunk_baseline_truncated_fraction": sum(d["first_chunk_baseline_omitted_tokens"] > 0 for d in details) / len(fixture_rows),
            "model_first_window_truncated_records": sum(d["model_first_window_omitted_tokens"] > 0 for d in details),
            "model_first_window_truncated_fraction": sum(d["model_first_window_omitted_tokens"] > 0 for d in details) / len(fixture_rows)}
        emb.write_json(fixture / "encoding_summary.json", summary)
        assets = cal._seal({"encoder_revision": "0" * 40, "tokenizer_revision": "0" * 40,
                           "scope": "temporary_mock_software_fixture_not_a_valid_asset_manifest"})
        manifest = cal._seal({"provenance": {"encoder_id": emb.E5}, "row_ids": [r["record_id"] for r in fixture_rows],
                             "implementation_lock": a.replication_encoding.implementation(8, 1),
                             "cache_file_sha256": emb.file_hash(fixture / "vectors.npy"),
                             "scope": "temporary_mock_software_fixture_not_a_valid_cache_manifest"})
        emb.write_json(fixture / "assets.json", assets)
        emb.write_json(fixture / "cache_manifest.json", manifest)
        fspec = {"directory": str(fixture), "model_dir": str(fixture), "tokenizer_dir": str(fixture)}
        with patch.object(a.calibrate_cache, "load_aligned_cache", return_value=([], None, None)), \
                patch.object(a.emb, "verify_assets", return_value=assets), \
                patch.object(a.emb, "LocalTransformer", return_value=backend):
            _, mock_report, _ = a._replay_cache_in_process(fspec, fixture_rows, fixture_prepared)
            check("positive_full_row_replay_with_explicit_mock_backend", mock_report["rows_replayed"] == 3 and mock_report["exact_FP32_match"])
            check("positive_complete_token_summary_replay_with_mock", mock_report["recomputed_encoding_summary"] == summary)
            changed_summary = dict(summary); changed_summary["chunks"] += 1
            (fixture / "encoding_summary.json").write_text(json.dumps(changed_summary))
            refused("false_summary_count_refused_with_mock", lambda: a._replay_cache_in_process(fspec, fixture_rows, fixture_prepared))
            (fixture / "encoding_summary.json").write_text(json.dumps(summary))
            changed_vectors = np.asarray(vectors, np.float32)
            changed_vectors[1, 2] = np.nextafter(np.float32(0), np.float32(1))
            np.save(fixture / "vectors.npy", changed_vectors, allow_pickle=False)
            refused("one_cached_coordinate_refused_with_mock", lambda: a._replay_cache_in_process(fspec, fixture_rows, fixture_prepared))
        refused("mock_fixture_refused_by_real_production_loader", lambda: a.replay_cache(fspec, fixture_rows, fixture_prepared))
        # Positive whole acceptance control path. Dependencies are explicitly
        # replaced for this unit test. No resulting acceptance report is saved.
        # The real parser, encoder, target-size and human gates remain separate.
        full_vectors = np.zeros((len(prepared_rows), 768), np.float32)
        full_vectors[:, 0] = 1
        cr = [r for r in prepared_rows if r["partition"] == "calibration"]
        positions = {r["record_id"]: i for i, r in enumerate(prepared_rows)}
        cf = full_vectors[[positions[r["record_id"]] for r in cr]]
        cp = {"dataset_id": "civil_comments", "encoder_id": emb.E5,
              "scope": "explicit_software_dependency_mock_not_semantic_provenance"}
        frame = {"scorer": {"block_size": 2}, "scope": "software_mock_not_a_calibration_frame"}
        lock = cal._seal({"threshold": 1.0, "frame": frame, "scope": "software_mock_not_a_human_threshold"})
        quality = cal._seal({"threshold": 1.0, "selection_lock_sha256": lock["sha256"],
            "statistical_and_declared_scope_eligible": True, "scope": "software_mock_not_human_evidence"})
        retained, retained_indices, semantic = panels.semantic_guard(prepared_rows, full_vectors, 1.0, block_size=2)
        guard_dir = t / "mock_guard_software_only"
        prep.write_population(guard_dir, retained, {}, {"semantic_guard": semantic,
            "retained_original_row_indices_sha256": cal.digest(retained_indices),
            "scope": "software_dependency_mock_not_primary_acceptance"})
        pids, paudit = panels.select_panels(retained, design, target=5)
        training_ids, evaluation_ids = pids["primary"], pids["test"]
        training_rows = [prepared_rows[positions[i]] for i in training_ids]
        evaluation_rows = [prepared_rows[positions[i]] for i in evaluation_ids]
        source_ids = [r["source_unit_id"] for r in training_rows]
        spec_mock = {**spec, "caches": {"e5": {"scope": "explicit_dependency_mock"}}, "guarded": {
            "records": str(guard_dir / "records.jsonl"), "audit": str(guard_dir / "audit.json"),
            "panel_ids": pids, "selection_lock": lock, "quality_report": quality,
            "target": 10000, "pool": "primary", "block_size": 2}}
        bundle_mock = {"source_acceptance": spec_mock, "train_ids": training_ids,
            "train_records": training_rows,
            "evaluation_ids": evaluation_ids, "source_ids": source_ids,
            "source_kinds": {s: "unknown_singleton" for s in source_ids},
            "evaluation_source_ids": [r["source_unit_id"] for r in evaluation_rows],
            "train_y": np.asarray([[r["original_fields"]["toxicity"]] for r in training_rows], np.float64),
            "evaluation_y": np.asarray([[r["original_fields"]["toxicity"]] for r in evaluation_rows], np.float64),
            "curators": {"e5": {"encoder_id": emb.E5, "train_ids": training_ids,
                "train": full_vectors[[positions[i] for i in training_ids]], "threshold_lock": lock,
                "calibration": {"records": cr, "features": cf, "provenance": cp, "quality_report": quality}}},
            "learners": {}}
        group_mock = {"group_id": "explicit-software-mock", "corpus": "civil_comments", "panel": "primary-10000"}
        source_mock = {**report, "engineering_preview": False, "scope": "mocked_parser_result_not_archive_acceptance"}
        panel_mock = copy.deepcopy(paudit); panel_mock["primary"]["shortfall"] = 0
        with patch.object(a, "replay_adapter", return_value=(rows, links, source_mock)), \
                patch.object(a, "replay_preparation", return_value=(prepared_rows, source_report)), \
                patch.object(a, "replay_cache", return_value=(full_vectors, {"encoder_id": emb.E5,
                    "scope": "mocked_cache_result_not_encoder_acceptance"}, (cr, cf, cp))), \
                patch.object(a.cal, "_check_inputs", return_value=frame), \
                patch.object(a.panels, "select_panels", return_value=(pids, panel_mock)):
            accepted_mock = a.accept_source_cache(group_mock, bundle_mock, design)
            check("positive_machine_control_path_with_explicit_dependency_mocks", accepted_mock["machine_acceptance_passed"])
            check("machine_control_path_does_not_authorize_execution", not accepted_mock["execution_allowed"])
            check("machine_control_path_preserves_external_trust_boundary", not accepted_mock["source_authenticity_verified"])
            check("heldout_source_kinds_recomputed_from_original_rows", accepted_mock["validated_bindings"]["evaluation_source_kinds"] ==
                  {r["source_unit_id"]: "unknown_singleton" for r in evaluation_rows})
            changed = copy.deepcopy(bundle_mock); changed["evaluation_source_ids"][0] = "incorrect-source"
            check("wrong_evaluation_source_refused_at_acceptance", not a.accept_source_cache(group_mock, changed, design)["machine_acceptance_passed"])
            changed = copy.deepcopy(bundle_mock); changed["source_acceptance"]["guarded"]["target"] = 5
            check("caller_cannot_lower_registered_target", not a.accept_source_cache(group_mock, changed, design)["machine_acceptance_passed"])
            changed = copy.deepcopy(bundle_mock); changed["evaluation_y"][0, 0] = 1 - changed["evaluation_y"][0, 0]
            check("changed_original_evaluation_target_refused", not a.accept_source_cache(group_mock, changed, design)["machine_acceptance_passed"])
            changed = copy.deepcopy(bundle_mock); changed["source_kinds"][source_ids[0]] = "genuine_native"
            check("unknown_source_promotion_refused_at_acceptance", not a.accept_source_cache(group_mock, changed, design)["machine_acceptance_passed"])
            changed = copy.deepcopy(bundle_mock); changed["train_records"][0]["text"] += " software-tamper"
            check("changed_training_text_refused_at_acceptance", not a.accept_source_cache(group_mock, changed, design)["machine_acceptance_passed"])
            changed = copy.deepcopy(bundle_mock); changed["train_records"][0]["original_fields"]["created_date"] = "1900-01-01"
            check("changed_native_training_date_refused_at_acceptance", not a.accept_source_cache(group_mock, changed, design)["machine_acceptance_passed"])
            calibration_mock = {"source_acceptance": spec_mock, "records": cr, "features": cf, "provenance": cp}
            calibration_result = a.accept_calibration_inputs("civil_comments", "e5", calibration_mock, design=design)
            check("positive_prethreshold_control_path_with_explicit_mocks", calibration_result["machine_acceptance_passed"])
            check("prethreshold_route_requires_no_response_or_lock", not calibration_result["calibration_binding"]["threshold_required"]
                  and not calibration_result["calibration_binding"]["human_responses_required"])
            check("prethreshold_route_does_not_authorize_primary", not calibration_result["execution_allowed"])
            changed = copy.deepcopy(calibration_mock); changed["records"][0]["partition"] = "test"
            check("test_record_refused_by_prethreshold_route", not a.accept_calibration_inputs("civil_comments", "e5", changed, design=design)["machine_acceptance_passed"])
            changed = copy.deepcopy(calibration_mock); changed["features"][0, 1] = np.float32(0.01)
            check("altered_calibration_feature_refused_before_threshold", not a.accept_calibration_inputs("civil_comments", "e5", changed, design=design)["machine_acceptance_passed"])
            check("wcep_cannot_select_own_threshold", not a.accept_calibration_inputs("wcep100", "e5", calibration_mock, design=design)["machine_acceptance_passed"])
        actual_preview_calibration = a.accept_calibration_inputs("civil_comments", "e5", {"source_acceptance": spec}, design=design)
        check("actual_preview_cannot_supply_primary_calibration", not actual_preview_calibration["machine_acceptance_passed"])
        del accepted_mock
    result = {"schema": "ccu-phase6-source-acceptance-checks-1", "scope": "software_and_reused_natural_Civil100_preview_only",
        "checks": len(outcomes), "passed": True, "outcomes": outcomes, "code_bindings": bindings,
        "natural_input_sha256": emb.file_hash(natural), "original_source_rich_archive_accepted": False,
        "actual_transformer_cache_replayed": False, "primary_execution_allowed": False,
        "actual_human_ratings": 0,
        "positive_mock_backend_scope": "Temporary software fixtures only; invalid production manifests; no primary acceptance artifact retained",
        "positive_machine_path_scope": "Explicit dependency mocks test control flow only; actual parser/cache/panel/human gates are not discharged",
        "untested_actual_backend_paths": ["full_local_E5_MPNet_inference", "authentic_News_calendar", "authentic_Stack_links", "authentic_WCEP_release"]}
    emb.write_json(out, result)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    value = run(args.out)
    print(json.dumps({"checks": value["checks"], "passed": value["passed"], "semantic_results": False}))
