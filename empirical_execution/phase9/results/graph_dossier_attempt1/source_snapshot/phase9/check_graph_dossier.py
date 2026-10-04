"""Focused natural-Civil graph serialization and blank-human-frame checks.

No semantic embeddings, source identities, ratings, or review testimony are
created. The existing Civil40 lexical fixture retains its engineering scope.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
from phase9 import graph_dossier as gd
from phase6.check_extensions import natural_bundle
from phase6 import dispatch, extensions, recipes
from phase5 import task_program as task
from phase8 import dossiers
from phase3.reference_graph import build_reference_graph
from phase4 import requests as rq
from phase4.admission_audit import prepare_admission_audit, write_admission_pack
from phase5.context_audit import prepare_context_audits, write_context_pack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(gd.__file__), ROOT/"phase8/dossiers.py",
               ROOT/"phase5/task_program.py", ROOT/"phase6/check_extensions.py",
               ROOT/"phase6/extensions.py", ROOT/"phase6/dispatch.py", ROOT/"phase4/requests.py",
               ROOT/"phase4/admission_audit.py", ROOT/"phase5/context_audit.py",
               ROOT/"phase3/reference_graph.py", ROOT/"ccu/core.py"]
    bound = {str(p.relative_to(ROOT)): task.file_hash(p) for p in sources}
    for source in sources:
        target = out/"source_snapshot"/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    checks = []
    report = {"schema": "ccu-phase9-graph-dossier-checks-1", "source_sha256": bound,
              "scope": "existing_natural_Civil40_lexical_graph_and_blank_human_forms_only",
              "created_human_judgments": 0, "created_external_review_testimony": False,
              "genuine_primary_source_gate_mocked": False, "primary_execution": False}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def refuse(name, action, message=None):
        try:
            action()
        except (ValueError, TypeError, KeyError, OverflowError, AttributeError) as exc:
            check(name, message is None or message in str(exc))
        else:
            raise AssertionError("Expected refusal: " + name)

    try:
        bundle = natural_bundle()
        features = bundle["curators"]["lexical32"]["train"]
        graph = build_reference_graph(features, bundle["train_ids"], .6, bundle["source_ids"])
        descriptor = gd.graph_payload(graph)
        adapted = gd.CanonicalGraph(descriptor)
        binding = rq.graph_binding(graph)
        check("plain_canonical_JSON_descriptor", type(descriptor) is dict and json.loads(json.dumps(descriptor)) == descriptor)
        check("frozen_graph_source_binding_identical", rq.graph_binding(adapted) == binding)
        check("input_record_order_identical", adapted.record_ids == graph.record_ids)
        check("source_ownership_identical", adapted.source_ids == graph.source_ids)
        check("priority_identical", adapted.priority == graph.priority)
        check("threshold_hex_identical", adapted.threshold.hex() == graph.threshold.hex())
        check("frozen_fingerprint_matches_primitive_descriptor", task.fingerprint(adapted) == task.fingerprint(descriptor))
        check("JSON_signature_bytes_unchanged", json.dumps(task.fingerprint(adapted), sort_keys=True) == json.dumps(task.fingerprint(descriptor), sort_keys=True))
        check("immutable_copy_contract", copy.copy(adapted) is adapted and copy.deepcopy(adapted) is adapted and adapted.copy() is adapted)
        check("no_instance_hidden_graph_storage", not hasattr(adapted, "__dict__"))
        for name in ("priority_indices", "indptr", "indices"):
            arr = getattr(adapted, name)
            check(name + "_equals_graph", np.array_equal(arr, getattr(graph, name)))
            refuse(name + "_cannot_enable_writes", lambda a=arr: a.setflags(write=True))
        refuse("mapping_assignment_refused", lambda: adapted.__setitem__("threshold_hex", 0.5.hex()))
        refuse("mapping_update_refused", lambda: adapted.update({"threshold_hex": 0.5.hex()}))
        refuse("mapping_clear_refused", lambda: adapted.clear())
        refuse("attribute_replacement_refused", lambda: setattr(adapted, "indices", np.array([], dtype=np.int64)))
        check("selected_indices_initial_identical", np.array_equal(adapted.selected_indices(), graph.selected_indices()))
        check("selected_indices_deleted_identical", all(np.array_equal(adapted.selected_indices([rid]), graph.selected_indices([rid])) for rid in graph.record_ids[:8]))
        refuse("unknown_deleted_identifier_refused", lambda: adapted.selected_indices(["not-an-existing-record"]))

        bad = []
        p = copy.deepcopy(descriptor); p.pop("schema"); bad.append(("missing_schema", p))
        p = copy.deepcopy(descriptor); p["unsigned_extra"] = True; bad.append(("extra_field", p))
        p = copy.deepcopy(descriptor); p["record_ids"][1] = p["record_ids"][0]; bad.append(("duplicate_record_id", p))
        p = copy.deepcopy(descriptor); p["source_ids"].pop(); bad.append(("unaligned_source_ids", p))
        p = copy.deepcopy(descriptor); p["source_ids"][0] = ""; bad.append(("empty_source_id", p))
        p = copy.deepcopy(descriptor); p["priority_indices"][1] = p["priority_indices"][0]; bad.append(("duplicate_priority", p))
        p = copy.deepcopy(descriptor); p["priority_indices"][0] = True; bad.append(("boolean_index", p))
        p = copy.deepcopy(descriptor); p["indices"][0] = 2**63; bad.append(("overflow_index", p))
        p = copy.deepcopy(descriptor); p["indices"][0] = -1; bad.append(("negative_index", p))
        p = copy.deepcopy(descriptor); p["indptr"][-1] += 1; bad.append(("CSR_end_mismatch", p))
        p = copy.deepcopy(descriptor); p["threshold_hex"] = "0.6"; bad.append(("noncanonical_threshold", p))
        p = copy.deepcopy(descriptor); p["threshold_hex"] = "inf"; bad.append(("nonfinite_threshold", p))
        p = copy.deepcopy(descriptor); p["threshold_hex"] = 0.6; bad.append(("nonstring_threshold", p))
        row = next(i for i in range(len(graph.record_ids)) if descriptor["indptr"][i] < descriptor["indptr"][i+1])
        p = copy.deepcopy(descriptor); p["indices"][p["indptr"][row]] = row; bad.append(("self_blocker", p))
        row2 = next(i for i in range(len(graph.record_ids)) if descriptor["indptr"][i+1] - descriptor["indptr"][i] >= 2)
        p = copy.deepcopy(descriptor); lo = p["indptr"][row2]; p["indices"][lo+1] = p["indices"][lo]; bad.append(("duplicate_blocker", p))
        p = copy.deepcopy(descriptor); lo = p["indptr"][row2]; p["indices"][lo], p["indices"][lo+1] = p["indices"][lo+1], p["indices"][lo]; bad.append(("unsorted_blocker", p))
        for name, value in bad:
            refuse("invalid_" + name, lambda x=value: gd.CanonicalGraph(x))

        frame = {"civil_comments": {"records": bundle["train_records"], "features": features,
                 "graph": graph, "request_manifest": bundle["requests"],
                 "existing_labels": {r["record_id"]: r["label"] for r in bundle["train_records"]},
                 "provenance": {"corpus_snapshot": "actual_existing_Civil_preview40",
                    "representation_id": "lexical32", "representation_revision": "frozen_workspace_code",
                    "evidence_role": "engineering_nonconfirmatory"}}}
        report["natural_input_sha256"] = bundle["natural_input_sha256"]
        report["natural_graph_binding"] = binding
        context = {"dispatch_module": dispatch, "mode": dispatch.ENGINEERING_MODE,
                   "design": json.loads(recipes.legacy.DESIGN.read_text()), "base_dir": ROOT.parent}
        check("ordinary_bundle_not_transformed", gd.hydrate_human_graphs(bundle) is bundle)
        routes = []
        with tempfile.TemporaryDirectory(prefix="phase9-graph-dossier-") as temp:
            temp = Path(temp)
            for name, prepare, writer, key in [
                ("human_admission_main", prepare_admission_audit, write_admission_pack, "human_admission"),
                ("human_admission_context", prepare_context_audits, write_context_pack, "human_context")]:
                manifest = prepare(frame)
                pack = temp/(name + "_blank_pack")
                writer(manifest, {"civil_comments": bundle["train_records"]}, pack)
                responses = json.loads((pack/"responses.template.json").read_text())
                serialized_frame = {"civil_comments": {**frame["civil_comments"], "graph": descriptor}}
                dossier = {"corpus_inputs": serialized_frame, "manifest": manifest, "responses": responses,
                           "evidence_role": dispatch.ENGINEERING_MODE}
                source_dir = temp/name; source_dir.mkdir()
                dossiers.dump_value(dossier, source_dir/"dossier.json")
                plain_loaded = dossiers.load_value(source_dir/"dossier.json")
                loaded = gd.hydrate_human_graphs(plain_loaded)
                check(name + "_bound_loader_fingerprint_parity", task.fingerprint(dossier) == task.fingerprint(loaded))
                check(name + "_exact_JSON_signing_parity", json.dumps(task.fingerprint(plain_loaded), sort_keys=True) == json.dumps(task.fingerprint(loaded), sort_keys=True))
                check(name + "_review_digest_matches_frozen_expression", gd.reviewed_dossier_sha256(plain_loaded) == recipes.digest(task.fingerprint(loaded)))
                check(name + "_review_field_exclusion", gd.reviewed_dossier_sha256({**loaded, "external_evidence_review": None}) == gd.reviewed_dossier_sha256(loaded))
                check(name + "_frozen_frame_replay_identical", prepare(loaded["corpus_inputs"]) == manifest)
                check(name + "_responses_untouched_and_blank", loaded["responses"] == responses and all(r["human_completed"] is False for r in responses["responses"]))
                job = {"recipe_id": name, "job_id": name + "/blank-software-check"}
                destination = temp/(name + "_analysis"); destination.mkdir()
                result = extensions.human_job(job, {key: loaded, "civil_comments/primary-10000": bundle}, destination, context)
                check(name + "_frozen_human_job_waits_for_actual_responses", result["status"] == "awaiting_real_human_responses")
                check(name + "_no_primary_acceptance", result["primary_output_accepted"] is False)
                routes.append({"recipe_id": name, "status": result["status"], "blank_assignments": len(responses["responses"])})

                changed = copy.deepcopy(descriptor)
                changed["source_ids"][0] = changed["source_ids"][1]
                source_changed = {**plain_loaded, "corpus_inputs": {"civil_comments": {
                    **plain_loaded["corpus_inputs"]["civil_comments"], "graph": changed}}}
                check(name + "_source_change_alters_signing_digest", gd.reviewed_dossier_sha256(source_changed) != gd.reviewed_dossier_sha256(plain_loaded))
                mismatch = gd.hydrate_human_graphs(source_changed)
                refuse(name + "_changed_source_rejected_by_frozen_accepted_graph", lambda: extensions.human_job(job, {key: mismatch, "civil_comments/primary-10000": bundle}, destination, context), "Human frame graph differs")
                changed = copy.deepcopy(descriptor)
                order = changed["priority_indices"]
                index = next(j for j in range(len(order)-1)
                             if order[j] not in set(map(int, graph.blockers[order[j+1]])))
                order[index], order[index+1] = order[index+1], order[index]
                check(name + "_different_valid_priority_changes_binding", rq.graph_binding(gd.CanonicalGraph(changed)) != binding)
                priority_changed = {**plain_loaded, "corpus_inputs": {"civil_comments": {
                    **plain_loaded["corpus_inputs"]["civil_comments"], "graph": changed}}}
                mismatch = gd.hydrate_human_graphs(priority_changed)
                refuse(name + "_changed_priority_rejected_by_frozen_accepted_graph", lambda: extensions.human_job(job, {key: mismatch, "civil_comments/primary-10000": bundle}, destination, context), "Human frame graph differs")
                # Existing hashed file loader still owns on-disk artifact checks.
                raw = json.loads((source_dir/"dossier.json").read_text())
                feature_descriptor = raw["corpus_inputs"]["civil_comments"]["features"]
                feature_file = source_dir/feature_descriptor["path"]
                with feature_file.open("ab") as stream:
                    stream.write(b"altered")
                refuse(name + "_changed_bound_feature_file_rejected", lambda: dossiers.load_value(source_dir/"dossier.json"), "checksum mismatch")
        report["routes"] = routes
        # Explicit low-level bypass changes both runtime and fingerprint; no
        # stale hidden cache is used even outside the ordinary immutable API.
        bypass = gd.CanonicalGraph(descriptor)
        altered_sources = tuple([descriptor["source_ids"][1]] + descriptor["source_ids"][1:])
        dict.__setitem__(bypass, "source_ids", altered_sources)
        check("no_hidden_graph_disagreement_after_low_level_dict_bypass", bypass.source_ids == altered_sources and rq.graph_binding(bypass) != binding and task.fingerprint(bypass) != task.fingerprint(descriptor))
        check("frozen_dependencies_unchanged", all(task.file_hash(ROOT/p) == h for p, h in bound.items()))
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["checks"] = checks
        report["passed_checks"] = len(checks)
        (out/"checks.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "checks": len(checks), "out": str(out)}))


if __name__ == "__main__":
    main()
