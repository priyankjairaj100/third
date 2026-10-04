"""Extract explicitly nonconfirmatory engineering rows from saved connector Markdown.

This does not claim that Markdown is a byte-faithful export of the original corpus.
The untouched connector response is retained and hashed for every extracted row.
"""
from pathlib import Path
import hashlib
import json
import re

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unescape(value):
    # Only presentation escapes observed in the connector's Markdown table.
    return re.sub(r"\\([~|*_#>])", r"\1", value.strip())


def extract(path, repo, config, columns, prefix):
    obj = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join(x.get("text", "") for x in obj["content"] if x["type"] == "text")
    if "### Rows\n" not in text:
        return []
    assert "preview output was truncated" not in text.lower(), path
    expected = re.search(r"- Rows: `(\d+)-(\d+)`", text)
    assert expected, path
    first, last = map(int, expected.groups())
    rows = []
    for line in text.split("### Rows\n", 1)[1].strip().splitlines()[2:]:
        if not line.startswith("|"):
            continue
        parts = re.split(r"(?<!\\)\|", line)[1:-1]
        assert len(parts) == len(columns) + 1, (path, len(parts))
        index = int(parts[0].strip())
        fields = {name: unescape(value) for name, value in zip(columns, parts[1:])}
        rendered = {name: value.strip() for name, value in zip(columns, parts[1:])}
        if repo == "google/civil_comments":
            for name in columns[1:]:
                fields[name] = float(fields[name])
                assert 0 <= fields[name] <= 1
        out = {
            "record_id": f"{prefix}:train:row:{index}",
            "text": fields["text"],
            "source_id": fields.get("domain"),
            "label": fields.get("toxicity"),
            "fields": fields,
            "connector_rendered_fields": rendered,
            "provenance": {
                "repository": repo,
                "config": config,
                "split": "train",
                "row_index": index,
                "page_offset": first,
                "page_last_row": last,
                "tool": "hugging_face.hub_repo_details",
                "operation": "dataset_preview",
                "repository_url": f"https://huggingface.co/datasets/{repo}",
                "hub_revision": None,
                "hub_revision_verified": False,
                "fetched_utc_date": "2026-10-04",
                "response_file": path.name,
                "response_sha256": digest(path),
                "text_fidelity": "connector-rendered Markdown; known presentation escapes removed; original whitespace and bytes unverified",
                "use": "nonconfirmatory software engineering pilot only",
            },
        }
        if repo == "vblagoje/cc_news":
            out["source_kind"] = "host field from connector; not registrable-domain canonicalized or corporate ownership"
        else:
            out["source_kind"] = "unavailable; original source and article IDs absent"
        rows.append(out)
    assert [row["provenance"]["row_index"] for row in rows] == list(range(first, last + 1)), path
    return rows


def main():
    civil_cols = ["text", "toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
    news_cols = ["title", "text", "domain", "date", "description", "url", "image_url"]
    civil = extract(DATA / "civil_comments_preview_raw.json", "google/civil_comments", "default", civil_cols, "civil_comments_connector_preview")
    news = []
    failures = []
    for path in sorted(DATA.glob("cc_news_preview_*_raw.json")):
        rows = extract(path, "vblagoje/cc_news", "plain_text", news_cols, "cc_news_connector_preview")
        if not rows:
            failures.append(path.name)
        news.extend(rows)
    assert len({row["record_id"] for row in civil}) == len(civil)
    assert len({row["record_id"] for row in news}) == len(news)
    for name, rows in [("civil_comments_engineering_preview.jsonl", civil), ("cc_news_engineering_preview.jsonl", news)]:
        (DATA / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "status": "partial_preview_acquisition_only",
        "confirmatory_dataset_acquired": False,
        "civil_comments_rows": len(civil),
        "cc_news_rows": len(news),
        "cc_news_distinct_hosts": sorted(set(row["source_id"] for row in news)),
        "cc_news_unavailable_indices": sorted(set(range(100)) - {row["provenance"]["row_index"] for row in news}),
        "failed_preview_responses": failures,
        "outputs": {p.name: digest(p) for p in [DATA / "civil_comments_engineering_preview.jsonl", DATA / "cc_news_engineering_preview.jsonl"]},
        "limitations": [
            "Civil preview lacks original record IDs, publication IDs, article IDs, timestamps and parent metadata; record-only tests are possible.",
            "CC-News has no supervised response label; label remains null.",
            "Row-index record IDs are engineering identifiers tied to the saved response hashes, not immutable source IDs.",
            "This is a first-rows convenience slice, not a random or complete natural population.",
            "No immutable Hub commit revision was exposed by the functioning connector.",
            "The connector emits Markdown; escaped presentation is normalized but source bytes and whitespace are not verified.",
            "CC-News offsets 30 and 85 returned 502 then 429; no missing rows were manufactured.",
            "Source-level CC-News tests refer only to host groups present in this incomplete engineering slice.",
        ],
    }
    (BASE / "preview_acquisition_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
