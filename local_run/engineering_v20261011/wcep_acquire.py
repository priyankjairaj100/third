#!/usr/bin/env python3
"""Bounded, private WCEP author-file intake; no scientific acceptance implied.

Only Python's standard library is used. Ordinary Google Drive confirmation
forms are followed with their supplied hidden fields. Access failures stop the
attempt; no mirror, authentication, retry, or reconstructed metadata is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import gzip
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

AUTHOR_COMMIT = "8aa8418d55a4e6710277bed18bfc5349b50a5d96"
AUTHOR_NOTEBOOK = ("https://github.com/complementizer/wcep-mds-dataset/blob/"
                   + AUTHOR_COMMIT + "/wcep_getting_started.ipynb")
SOURCES = [
    ("test", "1qsd5pOCpeSXsaqNobXCrcAzhcjtG1wA1"),
    ("val", "1_kHTZ32jazTbXaFRg0vBeIsVcpI7CTmy"),
    ("train", "1kUjSRXzKnTYdJ732BkKVLg3CFxDKo25u"),
]
GIB = 1024 ** 3
FILE_CAP = 4 * GIB
TOTAL_CAP = 6 * GIB
HTML_CAP = 256 * 1024
LINE_CAP = 64 * 1024 ** 2
EXPANDED_CAP = 64 * GIB
CHUNK = 1024 * 1024


class IntakeError(RuntimeError):
    pass


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True,
                  ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def allowed_url(url):
    p = urllib.parse.urlsplit(url)
    host = (p.hostname or "").lower()
    allowed = host in {"drive.google.com", "drive.usercontent.google.com"}
    allowed = allowed or bool(re.fullmatch(r"doc-[a-z0-9-]+-docs\.googleusercontent\.com", host))
    if (p.scheme != "https" or not allowed or p.username or p.password
            or p.port not in (None, 443) or p.fragment):
        raise IntakeError("Redirect/form URL outside the explicit HTTPS Drive host allowlist")
    return url


def public_location(url):
    # Signed query strings and confirmation tokens never enter reports.
    p = urllib.parse.urlsplit(url)
    return p.scheme + "://" + p.netloc + p.path


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    max_repeats = 1
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        allowed_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Confirmation(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms = []
        self.form = None
        self.text = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style"}:
            self.skip += 1
        if tag == "form":
            if self.form is not None:
                raise IntakeError("Nested download form refused")
            self.form = {"attrs": attrs, "hidden": [], "other_inputs": []}
            self.forms.append(self.form)
        elif tag == "input" and self.form is not None:
            kind = attrs.get("type", "text").lower()
            if kind == "hidden":
                self.form["hidden"].append((attrs.get("name", ""), attrs.get("value", "")))
            elif kind not in {"submit", "button"}:
                self.form["other_inputs"].append(kind)

    def handle_endtag(self, tag):
        if tag == "form":
            self.form = None
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, value):
        if not self.skip and value.strip():
            self.text.append(value.strip())

    def target(self, base_url, expected_id, filename):
        visible = " ".join(self.text)
        lower = visible.lower()
        if any(s in lower for s in ("access denied", "request access", "you need access",
                                   "sign in", "download quota", "too many users",
                                   "violat", "abuse")):
            raise IntakeError("Drive returned an access, authentication, quota, or abuse page")
        if ("virus scan warning" not in lower and "can't scan this file for viruses" not in lower
                and "cannot scan this file for viruses" not in lower):
            raise IntakeError("HTML response is not the normal virus-scan confirmation")
        if filename not in visible:
            raise IntakeError("Confirmation page does not name the expected author file")
        forms = [f for f in self.forms if f["attrs"].get("id") == "download-form"]
        if len(forms) != 1:
            raise IntakeError("Exactly one ordinary Drive download form is required")
        form = forms[0]
        if form["other_inputs"] or form["attrs"].get("method", "get").lower() != "get":
            raise IntakeError("Authentication/non-GET form refused")
        action = allowed_url(urllib.parse.urljoin(base_url, form["attrs"].get("action", "")))
        p = urllib.parse.urlsplit(action)
        if p.hostname != "drive.usercontent.google.com" or p.path != "/download":
            raise IntakeError("Unexpected confirmation action refused")
        pairs = urllib.parse.parse_qsl(p.query, keep_blank_values=True) + form["hidden"]
        if any(not k for k, _ in pairs) or len({k for k, _ in pairs}) != len(pairs):
            raise IntakeError("Ambiguous confirmation form fields")
        fields = dict(pairs)
        if fields.get("id") != expected_id or fields.get("export") != "download" or not fields.get("confirm"):
            raise IntakeError("Confirmation must identify the expected file and supplied download token")
        if not set(fields).issubset({"id", "export", "confirm", "uuid"}):
            raise IntakeError("Unexpected confirmation field refused")
        # Use exactly the form's fields; do not invent confirm tokens or credentials.
        target = urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, urllib.parse.urlencode(pairs), ""))
        size = re.search(re.escape(filename) + r"\s*\(([^)]+)\)", visible)
        return allowed_url(target), size.group(1) if size else None


def download(split, fid, out, opener, total_bytes, args):
    filename = split + ".jsonl.gz"
    url = "https://drive.google.com/uc?export=download&id=" + fid
    receipt = {"split": split, "file": filename, "source_id": fid,
               "author_url": "https://drive.google.com/uc?id=" + fid,
               "started_utc": utcnow(), "status": "started", "requests": [],
               "input_authenticity_verified": False, "complete_author_release_verified": False}
    part = out / (filename + ".part")
    deadline = time.monotonic() + args.file_timeout
    seen = set()
    try:
        for stage in range(3):
            allowed_url(url)
            if url in seen:
                raise IntakeError("Repeated confirmation target refused; no retry")
            seen.add(url)
            req = urllib.request.Request(url, headers={"User-Agent": "WCEP-research-intake/1.0",
                                                       "Accept-Encoding": "identity"})
            with opener.open(req, timeout=args.request_timeout) as response:
                allowed_url(response.geturl())
                if response.status != 200:
                    raise IntakeError("Archive response must have HTTP status 200")
                ctype = response.headers.get("Content-Type", "").lower()
                length = response.headers.get("Content-Length")
                declared = int(length) if length is not None else None
                if declared is not None and declared < 0:
                    raise IntakeError("Negative Content-Length refused")
                receipt["requests"].append({"stage": stage, "status": response.status,
                    "response_location_without_query": public_location(response.geturl()),
                    "content_type": ctype, "content_length": declared})
                encoding = response.headers.get("Content-Encoding", "identity").lower()
                if encoding not in {"identity", ""}:
                    raise IntakeError("Unexpected HTTP content encoding refused")
                if "html" in ctype:
                    html = response.read(HTML_CAP + 1)
                    if len(html) > HTML_CAP or time.monotonic() > deadline:
                        raise IntakeError("Confirmation response exceeds byte/time bound")
                    page = Confirmation()
                    page.feed(html.decode("utf-8", "strict"))
                    url, hint = page.target(response.geturl(), fid, filename)
                    receipt["advertised_size_text"] = hint
                    continue
                remaining = min(FILE_CAP, TOTAL_CAP - total_bytes)
                if remaining <= 0 or (declared is not None and declared > remaining):
                    raise IntakeError("Advertised archive exceeds compressed transfer budget")
                digest = hashlib.sha256()
                count = 0
                with part.open("xb") as stream:
                    while True:
                        if time.monotonic() > deadline:
                            raise IntakeError("Per-file wall-time bound exceeded")
                        block = response.read(min(CHUNK, remaining - count + 1))
                        if not block:
                            break
                        if count == 0 and not block.startswith(b"\x1f\x8b"):
                            raise IntakeError("Archive response lacks gzip magic; HTML/errors refused")
                        if count + len(block) > remaining:
                            raise IntakeError("Compressed transfer byte budget exceeded")
                        stream.write(block)
                        digest.update(block)
                        count += len(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                if count == 0 or (declared is not None and count != declared):
                    raise IntakeError("Empty or incomplete archive response")
                target = out / filename
                if target.exists():
                    raise IntakeError("Refusing to replace an existing archive")
                part.rename(target)
                receipt.update(status="downloaded_pending_validation", compressed_bytes=count,
                               compressed_sha256=digest.hexdigest(), completed_utc=utcnow())
                dump_new(out / (split + ".download.json"), receipt)
                return target, receipt
        raise IntakeError("Confirmation stages exhausted without an archive; no retry")
    except Exception as exc:
        receipt.update(status="failed", failed_utc=utcnow(), error_type=type(exc).__name__,
                       error=safe_error(exc), partial_bytes=part.stat().st_size if part.exists() else 0)
        dump_new(out / (split + ".download_failed.json"), receipt)
        raise


def reject_constant(_):
    raise IntakeError("Non-finite JSON number refused")


def validate_archive(path, split, global_ids):
    counts = Counter()
    article_sizes = []
    expanded_hash = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while True:
            line = stream.readline(LINE_CAP + 1)
            if not line:
                break  # gzip consumes and verifies its trailer/CRC before EOF.
            counts["physical_lines"] += 1
            counts["uncompressed_bytes"] += len(line)
            if len(line) > LINE_CAP or counts["uncompressed_bytes"] > EXPANDED_CAP:
                raise IntakeError("Expanded archive or JSONL line exceeds inspection bound")
            expanded_hash.update(line)
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            try:
                event = json.loads(line.decode("utf-8", "strict"), parse_constant=reject_constant)
                if not isinstance(event, dict):
                    raise IntakeError("Event is not an object")
                for field in ("id", "date", "collection", "articles"):
                    counts["events_with_" + field + "_field"] += field in event
                eid = event.get("id")
                if isinstance(eid, bool) or not isinstance(eid, (str, int)) or not str(eid).strip() or "\0" in str(eid):
                    raise IntakeError("Missing or invalid original native event ID")
                if str(eid) in global_ids:
                    raise IntakeError("Repeated native event ID across supplied archive population")
                global_ids.add(str(eid))
                if not isinstance(event.get("date"), str):
                    raise IntakeError("Original event date string missing")
                date.fromisoformat(event["date"])
                if event.get("collection") != split:
                    raise IntakeError("Original collection field absent or differs from author split filename")
                articles = event.get("articles")
                if not isinstance(articles, list):
                    raise IntakeError("Original article array missing")
                for article in articles:
                    if not isinstance(article, dict) or not isinstance(article.get("title"), str) or not isinstance(article.get("text"), str):
                        raise IntakeError("Original article title/text strings missing")
                    for field in ("title", "text", "url"):
                        counts["articles_with_" + field + "_field"] += field in article
                    counts["articles"] += 1
                    counts["articles_with_nonempty_url"] += isinstance(article.get("url"), str) and bool(article["url"].strip())
                counts["events"] += 1
                counts["events_with_summary_field"] += "summary" in event
                article_sizes.append(len(articles))
            except Exception as exc:
                # Never include article text, IDs, or parser excerpts in report errors.
                raise IntakeError("JSONL schema failure at physical line " + str(counts["physical_lines"])
                                  + ": " + safe_error(exc)) from None
    if not counts["events"] or not counts["articles"]:
        raise IntakeError("Empty event/article population refused")
    return {"status": "gzip_crc_and_all_jsonl_rows_checked", "split": split,
            "counts": dict(counts), "uncompressed_sha256": expanded_hash.hexdigest(),
            "min_articles_per_event": min(article_sizes), "max_articles_per_event": max(article_sizes),
            "no_event_truncation": True, "no_summary_used_as_input": True,
            "input_authenticity_verified": False, "complete_author_release_verified": False,
            "confirmatory_study_ready": False}


def safe_error(exc):
    if isinstance(exc, IntakeError):
        return str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        return "HTTP " + str(exc.code) + "; no retry or alternate host attempted"
    # Library exceptions can contain signed URLs, HTML, or original field values.
    return type(exc).__name__ + "; details deliberately omitted from sanitized receipt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path, help="New private attempt directory")
    p.add_argument("--request-timeout", type=int, default=30)
    p.add_argument("--file-timeout", type=int, default=1800)
    args = p.parse_args()
    if min(args.request_timeout, args.file_timeout) < 1:
        p.error("Timeouts must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chmod(out, 0o700)
    (out / ".gitignore").write_text("*\n", encoding="ascii")
    started = utcnow()
    manifest = {"schema": "wcep-author-acquisition-v1", "created_utc": started,
        "author_notebook": AUTHOR_NOTEBOOK, "author_code_commit": AUTHOR_COMMIT,
        "driver_sha256": sha_file(__file__), "acquisition_order": [s for s, _ in SOURCES],
        "sources": [{"split": s, "file": s + ".jsonl.gz", "author_url": "https://drive.google.com/uc?id=" + fid}
                    for s, fid in SOURCES], "known_published_compressed_sha256": None,
        "limits": {"per_file_compressed_bytes": FILE_CAP, "total_compressed_bytes": TOTAL_CAP,
                   "request_socket_timeout_seconds": args.request_timeout,
                   "per_file_wall_timeout_seconds": args.file_timeout,
                   "max_jsonl_line_bytes": LINE_CAP, "max_expanded_bytes_per_file": EXPANDED_CAP},
        "scope": "Author-linked bytes acquired now; historical byte identity and complete release remain unverified",
        "input_authenticity_verified": False, "complete_author_release_verified": False,
        "confirmatory_study_ready": False, "retries": 0, "contacted_people": False}
    dump_new(out / "acquisition_manifest.json", manifest)
    os.chmod(out / "acquisition_manifest.json", 0o400)
    opener = urllib.request.build_opener(SafeRedirect(), urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    # Normal TLS verification stays enabled; cookies remain transient in memory.
    total = 0
    receipts = []
    global_ids = set()
    try:
        for split, fid in SOURCES:
            path, receipt = download(split, fid, out, opener, total, args)
            total += receipt["compressed_bytes"]
            validation = validate_archive(path, split, global_ids)
            if sha_file(path) != receipt["compressed_sha256"]:
                raise IntakeError("Original compressed archive changed during schema validation")
            dump_new(out / (split + ".validation.json"), validation)
            receipts.append({"split": split, "compressed_bytes": receipt["compressed_bytes"],
                             "compressed_sha256": receipt["compressed_sha256"],
                             "schema_counts": validation["counts"]})
            print(json.dumps({"split": split, "status": "downloaded_and_schema_checked",
                              "compressed_bytes": receipt["compressed_bytes"],
                              "events": validation["counts"]["events"],
                              "articles": validation["counts"]["articles"]}), flush=True)
        binding = hashlib.sha256(json.dumps(receipts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        snapshot = ("author-linked-live-acquisition-" + started[:10] + "-notebook-" + AUTHOR_COMMIT
                    + "-file-bindings-" + binding + "-historical-authenticity-unverified")
        summary = {"schema": "wcep-author-acquisition-result-v1", "completed_utc": utcnow(),
            "status": "three_author_linked_archives_downloaded_and_schema_checked", "files": receipts,
            "compressed_bytes_total": total, "file_bindings_sha256": binding,
            "descriptive_snapshot_declaration": snapshot,
            "input_authenticity_verified": False, "complete_author_release_verified": False,
            "confirmatory_study_ready": False,
            "adapter_executed": False, "adapter_next_step": "Review source receipts and real quota before invoking unchanged phase5/replication.py on all three archives"}
        dump_new(out / "acquisition_result.json", summary)
        print(json.dumps({"status": "completed_with_acceptance_gates_unmet", "out": str(out),
                          "compressed_bytes_total": total, "confirmatory_study_ready": False}), flush=True)
        return 0
    except Exception as exc:
        dump_new(out / "FAILED.json", {"failed_utc": utcnow(), "status": "failed_preserved_no_retry",
            "error_type": type(exc).__name__, "error": safe_error(exc),
            "completed_files": receipts, "completed_compressed_bytes": total,
            "input_authenticity_verified": False, "confirmatory_study_ready": False})
        print(json.dumps({"status": "failed_preserved_no_retry", "error": safe_error(exc), "out": str(out)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
