"""Schema/software checks plus existing natural previews; never semantic evidence.

Tiny handcrafted XML/CSV/PSL examples below are software fixtures only. All
outputs containing News text are temporary and removed before this script exits.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from xml.sax.saxutils import quoteattr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "empirical_execution"))
from phase4 import adapters as a
from phase3.panels import fixed_guard


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    checks = []
    natural = {}
    def check(name, condition):
        assert condition, name
        checks.append(name)
    def refusal(name, fn):
        try:
            fn()
        except (ValueError, FileExistsError):
            checks.append(name)
        else:
            raise AssertionError("Expected refusal: " + name)
    with tempfile.TemporaryDirectory(prefix="ccu-adapter-software-check-") as folder:
        root = Path(folder)
        psl = root / "SOFTWARE_FIXTURE_NOT_OFFICIAL_PSL.dat"
        psl.write_text("// SOFTWARE FIXTURE ONLY\n// ===BEGIN ICANN DOMAINS===\ncom\norg\nuk\nco.uk\njp\n*.kawasaki.jp\n!city.kawasaki.jp\n// ===END ICANN DOMAINS===\n// ===BEGIN PRIVATE DOMAINS===\nblogspot.com\n// ===END PRIVATE DOMAINS===\n")
        psl_sha = a.file_sha256(psl)
        domains = a.OfflinePSL(psl, psl_sha)
        check("PSL exact multipart", domains.registrable("news.example.co.uk") == "example.co.uk")
        check("PSL private included", domains.registrable("x.writer.blogspot.com") == "writer.blogspot.com")
        check("PSL private excluded", a.OfflinePSL(psl, psl_sha, include_private=False).registrable("writer.blogspot.com") == "blogspot.com")
        check("PSL wildcard suffix only", domains.registrable("a.kawasaki.jp") is None)
        check("PSL wildcard deeper", domains.registrable("b.a.kawasaki.jp") == "b.a.kawasaki.jp")
        check("PSL exception", domains.registrable("a.city.kawasaki.jp") == "city.kawasaki.jp")
        check("PSL unknown singleton", domains.registrable("example.not-a-tld") is None)
        check("host IDNA and trailing dot", a.canonical_host("BÜCHER.com.") == "xn--bcher-kva.com")
        refusal("PSL hash mismatch refused", lambda: a.OfflinePSL(psl, "0" * 64))
        refusal("IP source refused", lambda: a.canonical_host("127.0.0.1"))
        check("News exact title line removal", a.news_text("A Title", "\n A Title\nNot the same.") == ("A Title Not the same.", True))
        check("News title prefix retained", a.news_text("A Title", "A Title is longer.\nOther.") == ("A Title A Title is longer. Other.", False))
        check("News lowercase different retained", a.news_text("A Title", "a title\nOther.")[1] is False)
        civil_row = {"id": "13", "article_id": 9, "publication_id": "pub", "parent_id": 12,
                     "created_date": "2017-01-01T00:00:00Z", "text": "No\tchange café.", "toxicity": 0.25}
        civil = root / "civil_schema_fixture.csv"
        with civil.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(civil_row)); writer.writeheader(); writer.writerow(civil_row)
        audit = a.adapt_civil(civil, root / "civil", input_format="csv")
        row = read_rows(root / "civil/records.jsonl")[0]
        check("Civil CSV label is fraction", row["labels"] == [.25] and row["label"] == .25)
        check("Civil own text preserved and normalized", row["text"] == "No change café.")
        check("Civil parent ID preserved", row["original_fields"]["parent_id"] == 12 and row["record_id"] == "13" and row["adapter_input"]["raw_lexemes_for_typed_fields"]["parent_id"] == "12")
        child = dict(row, partition="train")
        parent = dict(row, record_id="12", text="A different parent", partition="test", original_fields={})
        retained, guard = fixed_guard([child, parent])
        check("Civil native parent resolves in fixed guard", [x["record_id"] for x in retained] == ["12"])
        check("Civil record verifier", a.verify_adapter_output(root / "civil")["verified_rows"] == 1)
        civil_json = root / "civil_schema_fixture.jsonl"
        civil_json.write_text(json.dumps(civil_row) + "\n")
        a.adapt_civil(civil_json, root / "civil_json")
        check("Civil CSV JSON source type invariant", read_rows(root / "civil_json/records.jsonl")[0]["source_unit_id"] == row["source_unit_id"])
        odd_id = root / "civil_bad_id.jsonl"
        odd_id.write_text(json.dumps(dict(civil_row, id="0013")) + "\n")
        refusal("Civil noncanonical numeric ID cannot break parent guard", lambda: a.adapt_civil(odd_id, root / "civil_bad_id"))
        refusal("Frozen output overwrite refused", lambda: a.adapt_civil(civil, root / "civil", input_format="csv"))
        duplicate = root / "duplicate.jsonl"
        duplicate.write_text((json.dumps(civil_row) + "\n") * 2)
        refusal("Duplicate native IDs refused", lambda: a.adapt_civil(duplicate, root / "duplicates"))
        check("Failed output marked incomplete", json.loads((root / "duplicates/audit.json").read_text())["status"] == "failed_incomplete_output")
        posts = root / "Posts.xml"
        links = root / "PostLinks.xml"
        records = [
            {"Id": "1", "PostTypeId": "1", "CreationDate": "2018-05-02T00:00:00.000", "OwnerUserId": "7", "Title": "Question", "Body": '<p>No <em>change</em>! <a href="https://example.com/a">link</a></p><blockquote>quote</blockquote><pre><code>x &lt; 4 &amp;&amp; y</code></pre><div class="question-status">Platform notice</div>', "Tags": "<python><linux>", "LastEditDate": "2024-03-01T00:00:00.000", "ContentLicense": "CC BY-SA 4.0"},
            {"Id": "2", "PostTypeId": "1", "CreationDate": "2023-12-31T23:59:59.999", "OwnerUserId": "-1", "Title": "Other", "Body": "<p>Possible Duplicate: author-written quotation</p>", "Tags": "<linux>"},
            {"Id": "3", "PostTypeId": "1", "CreationDate": "2018-05-01T23:59:59.999", "Title": "Old", "Body": "<p>old</p>", "Tags": "<linux>"},
            {"Id": "4", "PostTypeId": "1", "CreationDate": "2024-01-01T00:00:00", "Title": "New", "Body": "<p>new</p>", "Tags": "<linux>"},
            {"Id": "5", "PostTypeId": "2", "CreationDate": "2020-01-01T00:00:00", "Body": "answer"},
            {"Id": "6", "PostTypeId": "1", "CreationDate": "2020-01-01T00:00:00", "Title": "", "Body": "<p> </p>", "Tags": "<linux>"},
            {"Id": "7", "PostTypeId": "1", "CreationDate": "2020-01-01T00:00:00", "Title": "Missing owner", "Body": "<p>Unattributed text</p>", "Tags": "<linux>"},
            {"Id": "8", "PostTypeId": "1", "CreationDate": "2020-01-01T00:00:00", "OwnerUserId": "abc", "Title": "Bad owner", "Body": "<p>Bad schema only</p>", "Tags": "<linux>"},
        ]
        posts.write_text('<posts>\n' + '\n'.join('<row ' + ' '.join(k + '=' + quoteattr(v) for k, v in row.items()) + '/>' for row in records) + '\n</posts>')
        links.write_text('<postlinks><row Id="1" PostId="1" RelatedPostId="2" LinkTypeId="3"/><row Id="2" PostId="1" RelatedPostId="999" LinkTypeId="3"/><row Id="3" PostId="1" RelatedPostId="2" LinkTypeId="1"/></postlinks>')
        audit = a.adapt_stack(posts, links, root / "stack", evidence_scope="tiny_software_schema_fixture")
        rows = read_rows(root / "stack/records.jsonl")
        check("Stack exact date boundaries and question filter", [r["record_id"] for r in rows] == ["1", "2", "7"])
        check("Stack code quotation negation and link retained", all(x in rows[0]["text"] for x in ['No change!', 'https://example.com/a', 'quote', 'x < 4 && y']))
        check("Stack explicit notice removed", "Platform notice" not in rows[0]["text"])
        check("Stack ambiguous legacy notice preserved flagged", rows[1]["parse_audit"]["ambiguous_legacy_notice_text"] and "Possible Duplicate" in rows[1]["text"])
        check("Stack missing community owners distinct singletons", rows[1]["source_unit_id"] == "unknown:2" and rows[2]["source_unit_id"] == "unknown:7")
        check("Malformed owner never mints native source", audit["counts"]["excluded:malformed_owner_integer"] == 1)
        check("Stack original attribution revision retained", rows[0]["original_fields"]["LastEditDate"] == records[0]["LastEditDate"] and rows[0]["original_fields"]["ContentLicense"] == records[0]["ContentLicense"])
        check("Stack original tags retained", rows[0]["tags"] == ["python", "linux"])
        check("Stack duplicate links only and outside retained preserved", json.loads((root / "stack/duplicate_links.json").read_text()) == [["1", "2"], ["1", "999"]])
        check("Stack record verifier", a.verify_adapter_output(root / "stack")["verified_rows"] == 3)
        badxml = root / "bad.xml"
        badxml.write_text('<!DOCTYPE posts [<!ENTITY x "X">]><posts></posts>')
        refusal("XML entity declarations refused", lambda: list(a._xml_rows(badxml, "posts")))
        news = root / "news_schema_fixture.jsonl"
        news_rows = [
            {"title": "Headline", "text": "Headline\nContent", "date": "2017-01-01 00:00:00", "url": "https://www.example.com/a", "domain": "www.example.com"},
            {"title": "Headline", "text": "Headline\nContent", "date": "2019-12-31 23:59:59", "url": "https://www.example.com/a", "domain": "www.example.com"},
            {"title": "Conflict", "text": "Retain but unknown source", "date": "2018-06-01 00:00:00", "url": "https://a.example.com/x", "domain": "different.com"},
            {"title": "Undated", "text": "Unavailable date", "date": "", "url": "https://www.example.com/b", "domain": "www.example.com"},
        ]
        news.write_text("".join(json.dumps(row) + "\n" for row in news_rows))
        audit = a.adapt_news(news, root / "news", psl_path=psl, psl_sha256=psl_sha)
        rows = read_rows(root / "news/records.jsonl")
        check("News repeated URL/text instances preserved", len(rows) == 3 and rows[0]["record_id"] != rows[1]["record_id"] and rows[0]["text"] == rows[1]["text"])
        check("News calendar roles", rows[0]["role_by_date"] == "calibration_candidate" and rows[1]["role_by_date"] == "analysis_candidate")
        check("News URL/domain conflict retained unknown", rows[2]["source_kind"] == "unknown_singleton")
        check("News missing date explicit exclusion", audit["counts"]["excluded:missing_or_invalid_date"] == 1)
        check("News title-body verifier fixes old intake equality", a.verify_adapter_output(root / "news")["verified_rows"] == 3)
        preview = ROOT / "empirical_execution/data/civil_comments_engineering_preview.jsonl"
        audit = a.adapt_civil(preview, root / "civil_preview", input_schema="engineering_preview")
        natural["civil100"] = {"input_sha256": a.file_sha256(preview), "counts": audit["counts"], "scope": "Existing real connector preview, not original corpus bytes", "verification": a.verify_adapter_output(root / "civil_preview")}
        check("Natural Civil100 source remains unknown", audit["counts"]["retained"] == 100 and audit["counts"]["source_kind:unknown_singleton"] == 100)
        refusal("Preview cannot masquerade as source-rich Civil", lambda: a.adapt_civil(preview, root / "false_civil"))
        preview = ROOT / "empirical_execution/data/cc_news_engineering_preview.jsonl"
        if preview.exists():
            audit = a.adapt_news(preview, root / "news_preview", input_schema="engineering_preview", psl_path=psl, psl_sha256=psl_sha)
            natural["news90"] = {"input_sha256": a.file_sha256(preview), "counts": audit["counts"], "scope": "Existing real connector preview; tiny software PSL, never native-source evidence", "verification": a.verify_adapter_output(root / "news_preview")}
            check("Natural News90 record plumbing", audit["counts"]["input_rows"] == 90 and audit["counts"]["retained"] + sum(v for k,v in audit["counts"].items() if k.startswith("excluded:")) == 90)
        else:
            natural["news90"] = {"available": False, "reason": "News bodies intentionally not redistributed in Git backup"}
    result = {"schema": "ccu-adapter-software-audit-1", "passed": len(checks), "checks": checks,
              "adapter_code_sha256": a.file_sha256(a.__file__), "natural_preview_checks": natural,
              "synthetic_empirical_datasets_used": False, "tiny_handcrafted_software_fixtures_used": True,
              "original_archives_validated": False, "official_full_PSL_validated": False,
              "confirmatory_study_ready": False}
    target = ROOT / "empirical_execution/phase4/results/adapter_checks.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks_passed": len(checks), "natural_inputs": list(natural), "result": str(target)}, indent=2))


if __name__ == "__main__":
    main()
