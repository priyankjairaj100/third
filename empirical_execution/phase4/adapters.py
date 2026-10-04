"""Streaming, offline corpus adapters. This module verifies consistency, not authenticity.

Raw archives, licenses and extraction provenance still require independent review.
No input text or label is synthesized, and no network calls are made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import ipaddress
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import idna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3.panels import normalized_text, source_unit

VERSION = "ccu-corpus-adapters-1"
SCHEMA_SOURCES = {
    "civil": "https://www.tensorflow.org/datasets/catalog/civil_comments",
    "stack": "https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede",
    "news": "https://huggingface.co/datasets/vblagoje/cc_news",
    "psl": "https://github.com/publicsuffix/list/wiki/Format",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _dump(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _rows(path, input_format="jsonl"):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        if input_format == "csv":
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("CSV needs unique named columns")
            for ordinal, row in enumerate(reader, 1):
                if None in row:
                    raise ValueError(f"CSV extra fields at row {ordinal}")
                yield ordinal, row
        elif input_format == "jsonl":
            for line, text in enumerate(stream, 1):
                if not text.strip():
                    raise ValueError(f"Blank JSONL physical line {line}; explicit row identity required")
                row = json.loads(text)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line} is not an object")
                yield line, row
        else:
            raise ValueError("Only csv/jsonl are supported; no hidden archive extraction")


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?", value):
        raise ValueError("Expected ISO date or timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Calendar membership uses the recorded date, not an invented timezone.
    return parsed.date()


def _native_id(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Missing/non-scalar record identifier")
    value = str(value)
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError("Empty/noncanonical record identifier")
    if positive and (not value.isascii() or not value.isdecimal() or int(value) <= 0):
        raise ValueError("Expected positive integer identifier")
    return value


class Writer:
    """One-row text working set; duplicate-ID index is disk-backed and charged."""
    def __init__(self, directory, dataset_id, inputs, settings):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records = (self.directory / "records.jsonl").open("x", encoding="utf-8")
        self.exclusions = (self.directory / "exclusions.jsonl").open("x", encoding="utf-8")
        self.db_path = self.directory / "identity_index.sqlite3"
        self.db = sqlite3.connect(self.db_path)
        self.db.execute("PRAGMA cache_size=-2048")
        self.db.execute("CREATE TABLE ids(id TEXT PRIMARY KEY)")
        self.dataset_id = dataset_id
        self.counts = Counter()
        self.audit = {"schema": VERSION, "dataset_id": dataset_id, "inputs": inputs,
                      "settings": settings, "code_sha256": file_sha256(__file__),
                      "source_helper_sha256": file_sha256(Path(__file__).resolve().parents[1] / "phase3/panels.py"),
                      "primary_schema_sources": SCHEMA_SOURCES, "schema_checked_date": "2026-10-04",
                      "input_authenticity_verified": False, "confirmatory_study_ready": False,
                      "memory_contract": "Streaming input and output; current record/HTML tree plus disk-backed unique-ID index; PSL rules resident if used. No all-record text list.",
                      "identity_rule": "Native ID where present; otherwise immutable file SHA256 plus physical row ordinal. No URL/text deduplication."}

    def exclude(self, ordinal, reason):
        self.counts["excluded:" + reason] += 1
        _dump(self.exclusions, {"input_ordinal": ordinal, "reason": reason})

    def write(self, row):
        rid = row["record_id"]
        try:
            self.db.execute("INSERT INTO ids VALUES (?)", (rid,))
        except sqlite3.IntegrityError as exc:
            raise ValueError("Duplicate record identity; refusing to silently drop a natural instance: " + rid) from exc
        row["source_unit_id"], row["source_kind"] = source_unit(row, self.dataset_id)
        row["text_sha256"] = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        _dump(self.records, row)
        self.counts["retained"] += 1
        self.counts["source_kind:" + row["source_kind"]] += 1

    def finish(self, error=None):
        self.db.commit()
        self.audit["identity_index_bytes"] = self.db_path.stat().st_size
        self.db.close()
        self.records.close()
        self.exclusions.close()
        self.audit["counts"] = dict(sorted(self.counts.items()))
        self.audit["status"] = "failed_incomplete_output" if error else "parsed_pending_external_verification"
        if error:
            self.audit["error"] = str(error)
        self.audit["outputs"] = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                                 for p in self.directory.iterdir() if p.is_file() and p.name != "audit.json"}
        (self.directory / "audit.json").write_text(json.dumps(self.audit, indent=2, ensure_ascii=False) + "\n")
        return self.audit


def _input(path):
    return {"name": Path(path).name, "sha256": file_sha256(path), "bytes": Path(path).stat().st_size}


def _original(raw, schema):
    if schema == "engineering_preview":
        if not isinstance(raw.get("fields"), dict):
            raise ValueError("Engineering preview requires fields object")
        return raw["fields"]
    return raw


def adapt_civil(input_path, output_dir, *, input_format="jsonl", input_schema="tfds_1_2_4"):
    if input_schema not in ("tfds_1_2_4", "engineering_preview"):
        raise ValueError("Use explicit TFDS 1.2.4 export schema; Kaggle aliases are not silently guessed")
    info = _input(input_path)
    writer = Writer(output_dir, "civil_comments", [info], {"input_schema": input_schema, "input_format": input_format,
        "text_rule": "Own comment text only, NFC and whitespace collapse; no parent concatenation", "date_filter": None,
        "label_rule": "Original toxicity fraction as one output, without thresholding", "scope": "engineering_nonconfirmatory" if input_schema == "engineering_preview" else "original_export_pending_verification"})
    try:
        for ordinal, raw in _rows(input_path, input_format):
            writer.counts["input_rows"] += 1
            fields = _original(raw, input_schema)
            required = {"text", "toxicity"} if input_schema == "engineering_preview" else {"id", "publication_id", "article_id", "parent_id", "created_date", "text", "toxicity"}
            if not required.issubset(fields):
                raise ValueError(f"Missing Civil columns: {sorted(required - fields.keys())}")
            if not isinstance(fields["text"], str) or not normalized_text(fields["text"]):
                writer.exclude(ordinal, "blank_or_nonstring_comment"); continue
            rid = _native_id(raw["record_id"] if input_schema == "engineering_preview" else fields["id"])
            if input_schema == "tfds_1_2_4" and re.fullmatch(r"[+-]?[0-9]+", rid) and str(int(rid)) != rid:
                raise ValueError("Civil numeric record ID must use canonical spelling to align parent lookup")
            if isinstance(fields["toxicity"], bool):
                raise ValueError("Boolean toxicity is not an original fraction")
            label = float(fields["toxicity"])
            if not math.isfinite(label) or not 0 <= label <= 1:
                raise ValueError("Invalid original toxicity fraction")
            original = dict(fields)
            typed_lexemes = {}
            if input_schema == "tfds_1_2_4":
                if original["publication_id"] is not None and not isinstance(original["publication_id"], str):
                    raise ValueError("TFDS publication_id must be a string or absent")
                # CSV has no types. Normalize schema numeric fields identically
                # across CSV/JSONL, retaining every changed raw lexical value.
                for key in ("article_id", "parent_id"):
                    value = original[key]
                    if value not in (None, ""):
                        if isinstance(value, bool) or not isinstance(value, (int, str)) or not re.fullmatch(r"[+-]?[0-9]+", str(value)):
                            raise ValueError("Civil integer metadata malformed: " + key)
                        parsed = int(value)
                        if value != parsed:
                            typed_lexemes[key] = value
                        original[key] = parsed
                if original["article_id"] in ("", None) or original["article_id"] <= 0:
                    typed_lexemes["article_id"] = fields["article_id"]
                    original["article_id"] = None
                if original["toxicity"] != label:
                    typed_lexemes["toxicity"] = original["toxicity"]
                original["toxicity"] = label
            if fields.get("created_date"):
                _date(fields["created_date"])
            else:
                writer.counts["missing_created_date"] += 1
            row = {"record_id": rid, "text": normalized_text(fields["text"]), "original_fields": original,
                   "labels": [label], "label": label, "adapter_input": {"sha256": info["sha256"], "ordinal": ordinal,
                       "raw_lexemes_for_typed_fields": typed_lexemes}}
            if input_schema == "engineering_preview":
                row["provenance"] = raw.get("provenance", {})
            writer.write(row)
    except Exception as exc:
        writer.finish(exc); raise
    return writer.finish()


class BodyParser(HTMLParser):
    """Rendered-HTML to text with block boundaries and author link destinations.

    Only explicitly tagged platform notice containers are removed. Ambiguous
    old 'Possible Duplicate' prose remains visible and is flagged for review.
    """
    BLOCKS = {"p", "div", "blockquote", "pre", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr", "td", "br", "hr"}
    VOIDS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    NOTICES = {"question-status", "question-status-container", "post-notice", "special-status"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.fragments = []
        self.stack = []
        self.removed_notices = 0
        self.markup_repairs = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        parent_skip = self.stack[-1][1] if self.stack else False
        notice = bool(set(attrs.get("class", "").split()) & self.NOTICES)
        skip = parent_skip or notice or tag in {"script", "style"}
        if notice and not parent_skip:
            self.removed_notices += 1
        if not skip and tag in self.BLOCKS:
            self.fragments.append(" ")
        if not skip and tag == "img" and attrs.get("alt"):
            self.fragments.extend([" ", attrs["alt"], " "])
        if tag not in self.VOIDS:
            self.stack.append((tag, skip, attrs.get("href") if tag == "a" else None))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOIDS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        positions = [i for i, value in enumerate(self.stack) if value[0] == tag]
        if not positions:
            self.markup_repairs += 1
            return
        index = positions[-1]
        if index != len(self.stack) - 1:
            self.markup_repairs += 1
        for _, skip, href in reversed(self.stack[index:]):
            if not skip and href:
                self.fragments.extend([" (", href, ") "])
        skip = self.stack[index][1]
        del self.stack[index:]
        if not skip and tag in self.BLOCKS:
            self.fragments.append(" ")

    def handle_data(self, data):
        if not self.stack or not self.stack[-1][1]:
            self.fragments.append(data)


def stack_text(title, body):
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError("Question Title and Body must be strings")
    parser = BodyParser()
    parser.feed(body); parser.close()
    if parser.stack:
        parser.markup_repairs += 1
    body_text = normalized_text("".join(parser.fragments))
    text = normalized_text(title + " " + body_text)
    audit = {"removed_tagged_platform_notices": parser.removed_notices,
             "html_repairs": parser.markup_repairs,
             "ambiguous_legacy_notice_text": bool(re.search(r"\b(?:Possible Duplicate|This question already has an answer|This question is closed)\b", body_text))}
    return text, audit


def _xml_rows(path, expected_root):
    # Do not load the XML tree: clear each completed row and then its parent.
    # Reject custom entity/DTD declarations before ElementTree can expand them.
    with open(path, "rb") as stream:
        tail = b""
        for block in iter(lambda: stream.read(1 << 20), b""):
            scan = (tail + block).upper()
            if b"<!DOCTYPE" in scan or b"<!ENTITY" in scan:
                raise ValueError("Data dump XML must not contain DTD/entity declarations")
            tail = scan[-16:]
    iterator = ET.iterparse(path, events=("start", "end"))
    event, root = next(iterator)
    if event != "start" or root.tag != expected_root:
        raise ValueError("Unexpected XML root: " + str(root.tag))
    ordinal = 0
    for event, element in iterator:
        if event == "end" and element.tag == "row":
            ordinal += 1
            if list(element) or (element.text and element.text.strip()):
                raise ValueError("Dump rows must use attributes only")
            yield ordinal, dict(element.attrib)
            element.clear(); root.clear()


def adapt_stack(posts_path, links_path, output_dir, *, dataset_id="askubuntu", archive_sha256=None, archive_release="April 2024 corrected", evidence_scope="schema_only_pending_archive"):
    if dataset_id not in ("askubuntu", "english_stackexchange"):
        raise ValueError("Unsupported Stack site")
    site = "askubuntu.com" if dataset_id == "askubuntu" else "english.stackexchange.com"
    if archive_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", archive_sha256):
        raise ValueError("archive_sha256 must be an actual lowercase digest or absent")
    info = _input(posts_path)
    link_info = _input(links_path)
    writer = Writer(output_dir, dataset_id, [info, link_info], {"archive_release_declared": archive_release,
        "archive_sha256_declared_unverified": archive_sha256, "evidence_scope": evidence_scope,
        "date_filter_inclusive": ["2018-05-02", "2023-12-31"], "question_only": True,
        "missing_owner": "Record singleton; never a shared source", "native_duplicate_link_type": 3,
        "boilerplate_rule": "Remove explicitly marked platform notice containers only; flag ambiguous unmarked legacy text", "no_answers_concatenated": True})
    try:
        for ordinal, fields in _xml_rows(posts_path, "posts"):
            writer.counts["input_rows"] += 1
            if fields.get("PostTypeId") != "1":
                writer.exclude(ordinal, "not_question"); continue
            if not {"Id", "CreationDate", "Title", "Body", "Tags"}.issubset(fields):
                writer.exclude(ordinal, "missing_question_fields"); continue
            owner = fields.get("OwnerUserId")
            if owner not in (None, "") and not re.fullmatch(r"[+-]?[0-9]+", owner):
                writer.exclude(ordinal, "malformed_owner_integer"); continue
            rid = _native_id(fields["Id"], positive=True)
            try:
                date = _date(fields["CreationDate"])
            except ValueError:
                writer.exclude(ordinal, "invalid_creation_date"); continue
            if not "2018-05-02" <= date.isoformat() <= "2023-12-31":
                writer.exclude(ordinal, "outside_creation_date_window"); continue
            raw_tags = fields["Tags"]
            tags = re.findall(r"<([^<>]+)>", raw_tags)
            if "".join("<" + tag + ">" for tag in tags) != raw_tags or len(set(tags)) != len(tags):
                writer.exclude(ordinal, "malformed_question_tags"); continue
            text, parse_audit = stack_text(fields["Title"], fields["Body"])
            if not text:
                writer.exclude(ordinal, "blank_parsed_question"); continue
            for key, value in parse_audit.items():
                writer.counts[key] += int(value)
            # site is a declared archive route, not a field invented inside the XML.
            original = dict(fields)
            original["site"] = site
            original["OwnerUserId"] = int(owner) if owner not in (None, "") else None
            row = {"record_id": rid, "text": text, "original_fields": original, "tags": tags,
                   "adapter_input": {"sha256": info["sha256"], "ordinal": ordinal, "added_context_fields": ["site"], "absent_native_fields": ["OwnerUserId"] if "OwnerUserId" not in fields else [], "raw_lexemes_for_typed_fields": {"OwnerUserId": owner}},
                   "parse_audit": parse_audit}
            writer.write(row)
        with (writer.directory / "duplicate_links.jsonl").open("x", encoding="utf-8") as pairs, (writer.directory / "duplicate_links.json").open("x", encoding="utf-8") as pair_array, (writer.directory / "duplicate_link_metadata.jsonl").open("x", encoding="utf-8") as metadata:
            pair_array.write("[")
            first_pair = True
            for ordinal, fields in _xml_rows(links_path, "postlinks"):
                writer.counts["input_link_rows"] += 1
                if fields.get("LinkTypeId") != "3":
                    writer.counts["nonduplicate_links_ignored"] += 1; continue
                left = _native_id(fields.get("PostId"), positive=True)
                right = _native_id(fields.get("RelatedPostId"), positive=True)
                _dump(pairs, [left, right])
                if not first_pair:
                    pair_array.write(",")
                pair_array.write(json.dumps([left, right]))
                first_pair = False
                _dump(metadata, {"original_fields": fields, "input_sha256": link_info["sha256"], "ordinal": ordinal})
                writer.counts["duplicate_links_preserved"] += 1
                for rid in (left, right):
                    if not writer.db.execute("SELECT 1 FROM ids WHERE id=?", (rid,)).fetchone():
                        writer.counts["duplicate_link_endpoints_outside_retained_records"] += 1
            pair_array.write("]\n")
        writer.audit["archive_and_attribution_review_required"] = True
        writer.audit["legacy_notice_review_required"] = writer.counts["ambiguous_legacy_notice_text"] > 0
    except Exception as exc:
        writer.finish(exc); raise
    return writer.finish()


def canonical_host(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Missing or malformed hostname")
    value = value.removesuffix(".")
    if any(c in value for c in "/:@[]\\?#"):
        raise ValueError("Expected hostname, not URL/port/address")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        raise ValueError("IP addresses are not registrable publisher domains")
    return idna.encode(value, uts46=True, std3_rules=True).decode("ascii").lower()


class OfflinePSL:
    def __init__(self, path, sha256, *, include_private=True):
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256) or file_sha256(path) != sha256:
            raise ValueError("A pinned matching PSL SHA256 is required")
        self.exact, self.wildcard, self.exceptions = set(), set(), set()
        active = False
        markers = set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "===BEGIN ICANN DOMAINS===" in line:
                active = True; markers.add("icann")
            elif "===BEGIN PRIVATE DOMAINS===" in line:
                active = include_private; markers.add("private")
            elif "===END" in line:
                active = False
            elif active and line.strip() and not line.lstrip().startswith("//"):
                rule = line.split()[0]
                if rule.startswith("! "):
                    raise ValueError("Invalid PSL rule")
                target = self.exceptions if rule.startswith("!") else self.wildcard if rule.startswith("*.") else self.exact
                rule = rule[1:] if rule.startswith("!") else rule[2:] if rule.startswith("*.") else rule
                target.add(canonical_host(rule))
        if markers != {"icann", "private"} or not self.exact:
            raise ValueError("Pinned full PSL sections required; hand-built lists are only software fixtures")
        self.binding = {"sha256": sha256, "include_private": include_private, "idna_version": idna.__version__,
                        "rule_counts": {"exact": len(self.exact), "wildcard": len(self.wildcard), "exception": len(self.exceptions)},
                        "unknown_suffix_policy": "Unknown singleton rather than PSL default-star inferred source"}

    def registrable(self, host):
        host = canonical_host(host)
        labels = host.split(".")
        matches, exceptions = [], []
        for i in range(len(labels)):
            suffix = ".".join(labels[i:])
            if suffix in self.exceptions:
                exceptions.append(len(labels) - i - 1)
            if suffix in self.exact:
                matches.append(len(labels) - i)
            if i > 0 and suffix in self.wildcard:
                matches.append(len(labels) - i + 1)
        depth = max(exceptions) if exceptions else max(matches, default=0)
        if depth == 0 or len(labels) <= depth:
            return None
        return ".".join(labels[-depth-1:])


def news_text(title, body):
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError("News title/body must be strings")
    # Remove only one exact normalized first nonempty body line equal to title.
    title = normalized_text(title)
    lines = body.splitlines()
    first = next((i for i, line in enumerate(lines) if normalized_text(line)), None)
    removed = bool(title and first is not None and normalized_text(lines[first]) == title)
    if removed:
        lines.pop(first)
    return normalized_text(title + " " + "\n".join(lines)), removed


def adapt_news(input_path, output_dir, *, psl_path, psl_sha256, input_format="jsonl", input_schema="raw", include_private=True):
    if input_schema not in ("raw", "engineering_preview"):
        raise ValueError("News schema must be raw or engineering_preview")
    psl = OfflinePSL(psl_path, psl_sha256, include_private=include_private)
    info = _input(input_path)
    writer = Writer(output_dir, "cc_news", [info, _input(psl_path)], {"input_schema": input_schema,
        "input_format": input_format, "psl": psl.binding, "date_window": ["2017-01-01", "2019-12-31"],
        "role_by_date": "2017 calibration candidate; 2018-2019 analysis candidate; source disjointness still required",
        "text_rule": "Title then body; remove one exact normalized first nonempty body line only if equal to title",
        "scope": "engineering_nonconfirmatory" if input_schema == "engineering_preview" else "original_export_pending_verification",
        "no_news_text_redistribution": True})
    try:
        for ordinal, raw in _rows(input_path, input_format):
            writer.counts["input_rows"] += 1
            fields = _original(raw, input_schema)
            if not {"title", "text", "domain", "date", "url"}.issubset(fields):
                raise ValueError("News input requires title, text, domain, date, url")
            try:
                date = _date(fields["date"])
            except ValueError:
                writer.exclude(ordinal, "missing_or_invalid_date"); continue
            if not "2017-01-01" <= date.isoformat() <= "2019-12-31":
                writer.exclude(ordinal, "outside_news_date_window"); continue
            try:
                text, removed = news_text(fields["title"], fields["text"])
            except ValueError:
                writer.exclude(ordinal, "nonstring_title_or_body"); continue
            if not text:
                writer.exclude(ordinal, "blank_article"); continue
            rid = _native_id(raw["record_id"]) if input_schema == "engineering_preview" else "cc_news:" + info["sha256"] + ":row:" + str(ordinal)
            domain = None
            host = None
            domain_reason = None
            try:
                if not isinstance(fields["url"], str):
                    raise ValueError("Article URL must be a string")
                url = urlsplit(fields["url"])
                if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
                    raise ValueError("Malformed article URL")
                host = canonical_host(url.hostname)
                declared = canonical_host(fields["domain"])
                if declared != host:
                    raise ValueError("URL and declared domain differ")
                domain = psl.registrable(host)
                if domain is None:
                    domain_reason = "unknown_or_public_suffix_only"
            except (ValueError, idna.IDNAError, TypeError):
                domain_reason = "invalid_or_conflicting_domain"
            row = {"record_id": rid, "text": text, "original_fields": fields, "labels": [],
                   "registrable_domain": domain, "canonical_host": host, "domain_status": domain_reason or "psl_derived_pending_provenance",
                   "date_recorded": date.isoformat(), "role_by_date": "calibration_candidate" if date.year == 2017 else "analysis_candidate",
                   "adapter_input": {"sha256": info["sha256"], "ordinal": ordinal},
                   "parse_audit": {"exact_first_title_line_removed": removed}}
            if input_schema == "engineering_preview":
                row["provenance"] = raw.get("provenance", {})
            writer.counts["title_lines_removed"] += int(removed)
            writer.counts["role_by_date:" + row["role_by_date"]] += 1
            writer.counts["month:" + date.isoformat()[:7]] += 1
            if domain_reason:
                writer.counts["domain:" + domain_reason] += 1
            writer.write(row)
    except Exception as exc:
        writer.finish(exc); raise
    return writer.finish()


def verify_adapter_output(directory):
    """Versioned record-construction check, including corrected News title/body.

    This is deliberately separate from frozen phase3 intake_v2: that module's
    News equality check predates the protocol title+body implementation.
    """
    directory = Path(directory)
    audit = json.loads((directory / "audit.json").read_text())
    if audit.get("schema") != VERSION or audit.get("status") != "parsed_pending_external_verification":
        raise ValueError("Adapter run incomplete or wrong version")
    if audit.get("code_sha256") != file_sha256(__file__):
        raise ValueError("Adapter version binding does not match")
    for name, spec in audit["outputs"].items():
        if Path(name).name != name or file_sha256(directory / name) != spec["sha256"]:
            raise ValueError("Adapter output checksum mismatch")
    count = 0
    for _, row in _rows(directory / "records.jsonl"):
        count += 1
        fields = row["original_fields"]
        if audit["dataset_id"] == "civil_comments":
            expected = normalized_text(fields["text"])
            if row["labels"] != [float(fields["toxicity"])]:
                raise ValueError("Civil label does not equal original fraction")
        elif audit["dataset_id"] == "cc_news":
            expected, removed = news_text(fields["title"], fields["text"])
            if row["labels"] != [] or row["parse_audit"]["exact_first_title_line_removed"] != removed:
                raise ValueError("News parsing/empty-label consistency failure")
        else:
            expected, parse_audit = stack_text(fields["Title"], fields["Body"])
            if parse_audit != row["parse_audit"] or re.findall(r"<([^<>]+)>", fields["Tags"]) != row["tags"]:
                raise ValueError("Stack parse/tag consistency failure")
        if row["text"] != expected or row["source_unit_id"] != source_unit(row, audit["dataset_id"])[0]:
            raise ValueError("Text/source derivation mismatch")
    if count != audit["counts"].get("retained", 0):
        raise ValueError("Retained row count mismatch")
    return {"schema": "ccu-adapter-record-verifier-1", "verified_rows": count,
            "construction_consistency": True, "input_authenticity_verified": False,
            "confirmatory_study_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    civil = sub.add_parser("civil")
    news = sub.add_parser("news")
    for command in (civil, news):
        command.add_argument("input_path"); command.add_argument("output_dir")
        command.add_argument("--input-format", choices=("jsonl", "csv"), default="jsonl")
    civil.add_argument("--input-schema", choices=("tfds_1_2_4", "engineering_preview"), default="tfds_1_2_4")
    news.add_argument("--input-schema", choices=("raw", "engineering_preview"), default="raw")
    news.add_argument("--psl-path", required=True); news.add_argument("--psl-sha256", required=True)
    news.add_argument("--icann-only", action="store_true")
    stack = sub.add_parser("stack")
    stack.add_argument("posts_path"); stack.add_argument("links_path"); stack.add_argument("output_dir")
    stack.add_argument("--dataset-id", choices=("askubuntu", "english_stackexchange"), required=True)
    stack.add_argument("--archive-sha256"); stack.add_argument("--archive-release", default="April 2024 corrected")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "news":
        args["include_private"] = not args.pop("icann_only")
    audit = {"civil": adapt_civil, "stack": adapt_stack, "news": adapt_news}[command](**args)
    print(json.dumps({k: audit[k] for k in ("status", "counts", "confirmatory_study_ready")}, indent=2))


if __name__ == "__main__":
    main()
