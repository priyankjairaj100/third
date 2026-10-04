"""Software format checks only. No genuine original archive is available here."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from empirical_execution.phase8 import archive_export as ae
from empirical_execution.phase4 import adapters


def civil_row(text, **changes):
    value = {"id": "software-format-1", "comment_text": text, "parent_text": "", "publication_id": "software-publication",
             "created_date": "2016-01-01T00:00:00Z", "parent_id": "", "article_id": "1", "split": "train",
             **{name: "0.1" for name in ae.CIVIL_LABELS}, "extra_original_column": "software-format-only"}
    value.update(changes)
    return value


def make_civil(path, rows, *, extra_members=(), header=None):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header or list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("civil_comments.csv", stream.getvalue().encode("utf-8"))
        for name, data in extra_members:
            z.writestr(name, data)


def news_row(text, **changes):
    value = {"title": " software format ", "maintext": " " + text + "\n", "source_domain": " example.test ",
             "date_publish": " 2017-01-01 ", "description": None, "url": " https://example.test/article ",
             "image_url": None, "extra_original_column": {"retained": True}}
    value.update(changes)
    return value


def make_news(path, items):
    """Items are (name, bytes/dict/None); None creates a directory."""
    with tarfile.open(path, "w:gz") as tar:
        for name, value in items:
            info = tarfile.TarInfo(name)
            if value is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                content = json.dumps(value, ensure_ascii=False).encode("utf-8") if isinstance(value, dict) else value
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def run_checks():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def refuses(name, call, exceptions=(ValueError, FileExistsError, FileNotFoundError, KeyError)):
        try:
            call()
        except exceptions:
            checks.append(name)
        else:
            raise AssertionError("Expected refusal: " + name)

    natural = json.loads((ROOT / "empirical_execution/data/civil_comments_engineering_preview.jsonl").read_text().splitlines()[0])["text"]
    with tempfile.TemporaryDirectory(prefix="ccu-archive-format-only-") as temporary:
        work = Path(temporary)
        cpath = work / "format-only-civil.zip"
        source_rows = [civil_row(natural), civil_row(natural, id="software-format-2", split="test_public", parent_id="7.0"),
                       civil_row(natural, id="software-format-3", split="test_private"),
                       civil_row(natural, id="software-format-4", severe_toxicity=""),
                       civil_row(natural, id="software-format-5", split="not-selected", article_id="unparsed")]
        make_civil(cpath, source_rows, extra_members=[("README.txt", b"Software format fixture only")])
        cdir = work / "civil-format"; cdir.mkdir()
        civil = ae._render_archive("civil_comments", cpath, cdir, ae.file_binding(cpath))
        export, ledger, inventory = [rows(cdir / name) for name in ae.DATA_FILES]
        check("civil_three_official_split_rows", len(export) == 3)
        check("civil_official_splits_are_lineage", [r["_archive_lineage"]["official_split"] for r in export] == ["train", "validation", "test"])
        check("civil_own_text_without_parent_concatenation", export[0]["text"] == natural)
        check("civil_full_source_metadata_present", all(k in export[0] for k in ("id", "publication_id", "article_id", "parent_id", "created_date")))
        check("civil_empty_parent_zero_matches_upstream", export[0]["parent_id"] == 0)
        check("civil_float_parent_conversion", export[1]["parent_id"] == 7)
        check("civil_fp32_feature_conversion", export[0]["toxicity"] == struct.unpack("<f", struct.pack("<f", 0.1))[0] != 0.1)
        check("civil_all_seven_labels_preserved", all(name in export[0] for name in ae.CIVIL_LABELS))
        check("civil_every_raw_row_preserved", [r["raw_values"] for r in ledger] == source_rows)
        check("civil_original_fraction_lexeme", ledger[0]["raw_values"]["toxicity"] == "0.1")
        check("civil_missing_label_omission", ledger[3]["omission_reason"] == "missing_base_label:severe_toxicity")
        check("civil_unknown_split_omission_before_common_parse", ledger[4]["omission_reason"] == "not_in_base_official_splits")
        check("civil_all_members_accounted", len(inventory) == 2 and inventory[1]["disposition"] == "not_used_by_upstream_builder")
        check("civil_member_content_hash_bound", all(r["_archive_lineage"]["member_sha256"] == inventory[0]["sha256"] for r in export))
        check("civil_raw_row_hash_bound", all(r["location"]["raw_values_sha256"] == ae._digest(r["raw_values"]) for r in ledger))
        check("civil_parse_not_archive_acceptance", not (cdir / "manifest.json").exists())
        cagain = work / "civil-reconstruction"; cagain.mkdir()
        reconstructed = ae._render_archive("civil_comments", cpath, cagain, ae.file_binding(cpath))
        ae._compare_reconstruction(civil, reconstructed)
        check("civil_exact_deterministic_reconstruction", civil == reconstructed)
        csinks = {name: ae._HashSink() for name in ae.DATA_FILES}
        check("civil_hash_sink_replay_matches_every_file_byte_binding", ae._hash_reconstruction("civil_comments", cpath, ae.file_binding(cpath), csinks) == civil)
        altered = copy.deepcopy(civil); altered["counts"]["rows_seen"] += 1
        refuses("reconstruction_rejects_changed_counts", lambda: ae._compare_reconstruction(altered, reconstructed))
        altered = copy.deepcopy(civil); altered["outputs"]["lineage.jsonl"]["sha256"] = "0" * 64
        refuses("reconstruction_rejects_changed_lineage", lambda: ae._compare_reconstruction(altered, reconstructed))
        refuses("civil_fixture_population_not_release", lambda: ae._check_population("civil_comments", civil["counts"]))
        adapted = work / "civil-adapter"
        adapters.adapt_civil(cdir / "records.jsonl", adapted)
        adapted_rows = rows(adapted / "records.jsonl")
        check("frozen_adapter_consumes_export_schema", len(adapted_rows) == 3)
        check("frozen_adapter_retains_archive_lineage", adapted_rows[0]["original_fields"]["_archive_lineage"] == export[0]["_archive_lineage"])
        check("frozen_adapter_verifies_construction_only", adapters.verify_adapter_output(adapted)["input_authenticity_verified"] is False)

        missing = civil_row(natural, toxicity="", article_id="2147483648")
        check("missing_label_precedes_feature_dtype_check", ae._civil_value(missing) == (None, "missing_base_label:toxicity"))
        refuses("common_parser_precedes_missing_label", lambda: ae._civil_value(civil_row(natural, article_id="invalid", toxicity="")))
        refuses("earlier_label_parse_precedes_later_omission", lambda: ae._civil_value(civil_row(natural, toxicity="invalid", severe_toxicity="")))
        refuses("civil_int32_overflow_refused", lambda: ae._civil_value(civil_row(natural, article_id="2147483648")))
        refuses("civil_nonfinite_fraction_refused", lambda: ae._civil_value(civil_row(natural, toxicity="nan")))
        refuses("civil_out_of_range_fraction_refused", lambda: ae._civil_value(civil_row(natural, toxicity="1.1")))

        malformed = work / "civil-missing-header.zip"
        reduced = {k: v for k, v in civil_row(natural).items() if k != "publication_id"}
        make_civil(malformed, [reduced])
        target = work / "civil-missing-header"; target.mkdir()
        refuses("civil_missing_original_source_column_refused", lambda: ae._render_archive("civil_comments", malformed, target, ae.file_binding(malformed)))
        duplicate = work / "civil-duplicate-member.zip"
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            make_civil(duplicate, [civil_row(natural)], extra_members=[("civil_comments.csv", b"")])
        target2 = work / "civil-duplicate-member"; target2.mkdir()
        refuses("civil_duplicate_target_member_refused", lambda: ae._render_archive("civil_comments", duplicate, target2, ae.file_binding(duplicate)))
        multiline = work / "civil-multiline.zip"
        make_civil(multiline, [civil_row(natural + "\n" + natural)])
        mdir = work / "civil-multiline"; mdir.mkdir()
        ae._render_archive("civil_comments", multiline, mdir, ae.file_binding(multiline))
        check("civil_multiline_csv_row_lineage", rows(mdir / "lineage.jsonl")[0]["location"]["csv_physical_end_line"] == 3)
        for index, ending in enumerate(("\r\n", "\r")):
            line_path = work / ("civil-newlines-" + str(index) + ".zip")
            make_civil(line_path, [civil_row(natural + ending + natural)])
            line_dir = work / ("civil-newlines-" + str(index)); line_dir.mkdir()
            ae._render_archive("civil_comments", line_path, line_dir, ae.file_binding(line_path))
            check("civil_universal_newline_translation_" + str(index), rows(line_dir / "records.jsonl")[0]["text"] == natural + "\n" + natural)
            check("civil_original_newline_preserved_in_lineage_" + str(index), rows(line_dir / "lineage.jsonl")[0]["raw_values"]["comment_text"] == natural + ending + natural)

        npath = work / "format-only-news.tar.gz"
        article = news_row(natural)
        all_null = {name: None for name in ae.NEWS_FIELDS.values()}
        make_news(npath, [("articles/", None), ("articles/same.json", article), ("articles/same.json", article),
                          ("articles/null.json", all_null), ("README.txt", b"format-only nonarticle"), ("articles/UPPER.JSON", b"not selected")])
        ndir = work / "news-format"; ndir.mkdir()
        news = ae._render_archive("cc_news", npath, ndir, ae.file_binding(npath))
        nrows, nledger, ninventory = [rows(ndir / name) for name in ae.DATA_FILES]
        check("news_original_field_mapping", nrows[0]["text"] == natural and nrows[0]["domain"] == "example.test" and nrows[0]["date"] == "2017-01-01")
        check("news_null_to_empty_only", nrows[0]["description"] == nrows[0]["image_url"] == "")
        check("news_all_null_article_not_export_filtered", all(nrows[2][name] == "" for name in ae.NEWS_FIELDS))
        check("news_three_article_instances_preserved", len(nrows) == 3)
        check("news_duplicate_member_name_preserved", nrows[0]["_archive_lineage"]["member_name"] == nrows[1]["_archive_lineage"]["member_name"])
        check("news_duplicate_instances_have_distinct_ordinals", nrows[0]["_archive_lineage"]["member_ordinal"] != nrows[1]["_archive_lineage"]["member_ordinal"])
        check("news_upstream_indices_zero_based", [r["_archive_lineage"]["upstream_example_index"] for r in nrows] == [0, 1, 2])
        check("news_raw_unstripped_fields_preserved", nledger[0]["raw_values"] == article)
        check("news_all_logical_members_accounted", len(ninventory) == 6)
        check("news_case_sensitive_json_filter", news["counts"]["nonarticle_files_omitted"] == 2)
        check("news_does_not_invent_labels_or_native_ids", all("labels" not in r and "id" not in r for r in nrows))
        nagain = work / "news-reconstruction"; nagain.mkdir()
        nr = ae._render_archive("cc_news", npath, nagain, ae.file_binding(npath))
        ae._compare_reconstruction(news, nr)
        check("news_exact_deterministic_reconstruction", news == nr)
        nsinks = {name: ae._HashSink() for name in ae.DATA_FILES}
        check("news_hash_sink_replay_matches_every_file_byte_binding", ae._hash_reconstruction("cc_news", npath, ae.file_binding(npath), nsinks) == news)
        check("hash_sink_retains_no_corpus_buffer", set(vars(nsinks["records.jsonl"])) == {"_hash", "_bytes"})
        psl = work / "software-only-psl.txt"
        psl.write_text("// ===BEGIN ICANN DOMAINS===\ntest\n// ===END ICANN DOMAINS===\n// ===BEGIN PRIVATE DOMAINS===\n// ===END PRIVATE DOMAINS===\n")
        news_adapted = work / "news-adapter"
        adapters.adapt_news(ndir / "records.jsonl", news_adapted, psl_path=psl, psl_sha256=ae.file_binding(psl)["sha256"])
        news_adapted_rows = rows(news_adapted / "records.jsonl")
        check("news_frozen_adapter_consumes_export_schema", len(news_adapted_rows) == 2)
        check("news_frozen_adapter_retains_member_lineage", news_adapted_rows[0]["original_fields"]["_archive_lineage"] == nrows[0]["_archive_lineage"])
        check("news_null_date_exclusion_is_adapter_responsibility", rows(news_adapted / "exclusions.jsonl")[0]["reason"] == "missing_or_invalid_date")
        refuses("news_fixture_population_not_release", lambda: ae._check_population("cc_news", news["counts"]))
        refuses("news_missing_metadata_not_filled", lambda: ae._news_value({"title": ""}))
        refuses("news_nonstring_field_not_stringified", lambda: ae._news_value(news_row(natural, source_domain=123)))
        bad = work / "bad-news-json.tar.gz"; make_news(bad, [("article.json", b"{invalid")])
        bdir = work / "bad-news-json"; bdir.mkdir()
        refuses("news_malformed_json_refused", lambda: ae._render_archive("cc_news", bad, bdir, ae.file_binding(bad)))
        unsafe = work / "unsafe-news.tar.gz"; make_news(unsafe, [("../article.json", article)])
        udir = work / "unsafe-news"; udir.mkdir()
        refuses("news_path_traversal_refused_without_extraction", lambda: ae._render_archive("cc_news", unsafe, udir, ae.file_binding(unsafe)))
        symlink = work / "linked-news.tar.gz"
        with tarfile.open(symlink, "w:gz") as tar:
            info = tarfile.TarInfo("article.json"); info.type = tarfile.SYMTYPE; info.linkname = "/outside"
            tar.addfile(info)
        sdir = work / "linked-news"; sdir.mkdir()
        refuses("news_links_refused_without_extraction", lambda: ae._render_archive("cc_news", symlink, sdir, ae.file_binding(symlink)))
        trailing = work / "trailing-news.tar.gz"
        make_news(trailing, [("article.json/", article), ("article.json/.", article)])
        tdir = work / "trailing-news"; tdir.mkdir()
        trailing_report = ae._render_archive("cc_news", trailing, tdir, ae.file_binding(trailing))
        check("news_selection_matches_posix_basename_without_normalization", trailing_report["counts"].get("rows_exported", 0) == 0)

        for dataset, path in (("civil_comments", cpath), ("cc_news", npath)):
            expected = ae.expected_archive(dataset)
            refuses(dataset + "_wrong_sha_refused_at_published_size", lambda: ae._check_identity(dataset, {"bytes": expected["bytes"], "sha256": "0" * 64}))
            # These calls genuinely execute the production entry point and fail
            # its original identity gate. They never patch that gate.
            dest = work / (dataset + "-production-refusal")
            refuses(dataset + "_fixture_rejected_by_production", lambda: ae.export_archive(dataset, path, dest))
            failure, _ = ae._read_json(dest / "FAILED.json")
            check(dataset + "_failed_attempt_preserved", failure["machine_acceptance_passed"] is False and (dest / "attempt.json").exists())
            check(dataset + "_no_record_export_before_identity_gate", not (dest / "records.jsonl").exists())
            before = ae.file_binding(dest / "FAILED.json")
            refuses(dataset + "_existing_directory_refused", lambda: ae.export_archive(dataset, path, dest))
            check(dataset + "_failed_attempt_not_overwritten", before == ae.file_binding(dest / "FAILED.json"))
        missing_out = work / "missing-archive"
        refuses("missing_original_file_retains_failure", lambda: ae.export_archive("civil_comments", work / "absent.zip", missing_out))
        check("missing_file_failure_marker", (missing_out / "FAILED.json").exists())
        fixture_manifest = {"schema": ae.SCHEMA, "status": "completed", "dataset_id": "civil_comments",
                            "evidence_scope": "software_format_fixture", "archive": ae.file_binding(cpath), **civil}
        ae._write_json(cdir / "manifest.json", fixture_manifest)
        rd = work / "fixture-replay-refusal"
        refuses("fixture_receipt_cannot_pass_production_replay", lambda: ae.replay_export(cdir, cpath, rd))
        check("replay_failure_preserved", (rd / "FAILED.json").exists())
        spoof = dict(fixture_manifest, evidence_scope="published_original_archive_bytes", expected_archive=ae.expected_archive("civil_comments"))
        refuses("self_declared_original_scope_cannot_bypass_identity", lambda: ae._validate_manifest(spoof))
        refuses("replay_output_must_be_new", lambda: ae.replay_export(cdir, cpath, rd))
        with contextlib.redirect_stderr(io.StringIO()):
            refuses("production_cli_refuses_identity_override", lambda: ae.main(["export", "civil_comments", str(cpath), "--out", str(work / "override"), "--expected-sha256", "0" * 64]), exceptions=(SystemExit,))
        loaded_hash = ae._LOADED_SOURCE_SHA256
        try:
            ae._LOADED_SOURCE_SHA256 = "0" * 64
            refuses("changed_source_binding_refuses_production_start", lambda: ae.export_archive("civil_comments", cpath, work / "source-changed"))
            check("changed_source_failure_preserved", (work / "source-changed" / "FAILED.json").exists())
        finally:
            ae._LOADED_SOURCE_SHA256 = loaded_hash

    return {"schema": "ccu-archive-export-software-checks-1", "status": "passed", "passed": len(checks), "checks": checks,
            "code_sha256": ae.file_binding(ae.__file__)["sha256"], "checker_sha256": ae.file_binding(__file__)["sha256"],
            "scope": "software_format_fixtures_and_production_refusals_only",
            "fixture_provenance": "Existing Civil preview text; explicitly invented format metadata, never empirical provenance or human judgments.",
            "original_archives_available": False, "genuine_archive_export_executed": False,
            "genuine_archive_replay_executed": False, "primary_study_started": False,
            "earlier_reports": "Intermediate check reports preserve hashes/outcomes, but their exact source snapshots were not retained; do not claim exact replay of those versions."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    target = Path(args.out)
    if target.exists():
        raise FileExistsError("Never overwrite earlier checks")
    target.parent.mkdir(parents=True, exist_ok=True)
    report = run_checks()
    ae._write_json(target, report)
    print(json.dumps({"status": report["status"], "passed": report["passed"], "scope": report["scope"]}))


if __name__ == "__main__":
    main()
