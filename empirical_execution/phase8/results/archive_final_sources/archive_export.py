"""Pinned original-archive exports and full byte replay; offline and fail closed.

No public API accepts alternative archive identities. Format tests exercise the
private renderer only; they cannot produce accepted original-archive manifests.
Outputs contain private corpus text and must never be published.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import stat
import struct
import sys
import tarfile
import zipfile

SCHEMA = "ccu-original-archive-export-1"
REPLAY_SCHEMA = "ccu-original-archive-replay-1"
DATA_FILES = ("records.jsonl", "lineage.jsonl", "members.jsonl")
CIVIL_LABELS = ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit")
CIVIL_SPLITS = {"train": "train", "test_public": "validation", "test_private": "test"}
CIVIL_COUNTS = {"train": 1804874, "validation": 97320, "test": 97320}
NEWS_FIELDS = {"title": "title", "text": "maintext", "domain": "source_domain", "date": "date_publish",
               "description": "description", "url": "url", "image_url": "image_url"}
TFDS_REVISION = "ff070f916bdbc7f5f833e2bf9e48336946d6fb56"
NEWS_REVISION = "0ae17dffa45f621cb8061a8393e2dd225d0876a0"
_LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def expected_archive(dataset_id):
    """Return a fresh fixed binding; callers cannot configure the identity gate."""
    if dataset_id == "civil_comments":
        return {"name": "civil_comments_v1.2.zip", "bytes": 448174578,
                "sha256": "89a60573acbbfaf189b2c77278115eefcbc1654dbd8235dd8fbecc1b83f1585c",
                "url": "https://storage.googleapis.com/jigsaw-unintended-bias-in-toxicity-classification/civil_comments_v1.2.zip",
                "upstream_revision": TFDS_REVISION, "configuration": "civil_comments/CivilComments/1.2.4"}
    if dataset_id == "cc_news":
        return {"name": "cc_news.tar.gz", "bytes": 845131146,
                "sha256": "1aaf8e5af33e3a73472b58afba48c6a839ebc2dd190c4e0754fc00f8899a9cec",
                "url": "https://huggingface.co/datasets/vblagoje/cc_news/resolve/" + NEWS_REVISION + "/data/cc_news.tar.gz",
                "upstream_revision": NEWS_REVISION, "configuration": "cc_news/plain_text/1.0.0"}
    raise ValueError("Unsupported original corpus")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path, value):
    with Path(path).open("xb") as f:
        f.write(_json_bytes(value))


def _line(stream, value):
    stream.write(_json_bytes(value))


def _read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("Duplicate manifest key")
            out[key] = value
        return out
    raw = Path(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite manifest value")))
    return value, hashlib.sha256(raw).hexdigest()


def _stream_binding(stream):
    h, size = hashlib.sha256(), 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
        size += len(chunk)
    return {"bytes": size, "sha256": h.hexdigest()}


def file_binding(path):
    with Path(path).open("rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("A regular local archive file is required")
        return _stream_binding(stream)


def _code_binding():
    if file_binding(__file__)["sha256"] != _LOADED_SOURCE_SHA256:
        raise ValueError("Exporter source changed since this module was loaded")
    return _LOADED_SOURCE_SHA256


def _new_output(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    # All descendants contain private derivations. This prevents ordinary Git
    # add operations; it is not a security boundary against force-add/copying.
    (out / ".gitignore").write_text("*\n", encoding="utf-8")
    return out


def _check_identity(dataset_id, binding):
    expected = expected_archive(dataset_id)
    if binding != {k: expected[k] for k in ("bytes", "sha256")}:
        raise ValueError("Original archive size/SHA256 differs from the fixed published identity")


def _member_name(name):
    if not name or "\0" in name or "\\" in name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
        raise ValueError("Unsafe or ambiguous archive member name")
    return name


def _i32(value):
    if not -(2 ** 31) <= value < 2 ** 31:
        raise ValueError("Civil integer metadata exceeds TFDS int32 range")
    return value


def _universal_newlines(value):
    # TFDS opens the CSV text stream with newline=None. Our lossless CSV reader
    # retains original line endings for lineage, then mirrors that translation.
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _civil_value(raw):
    """Match the upstream base parser, then its declared feature dtypes."""
    result = {"id": raw["id"], "text": raw["comment_text"], "parent_text": raw["parent_text"],
              "publication_id": raw["publication_id"], "created_date": raw["created_date"],
              "parent_id": int(float(raw["parent_id"] or 0)), "article_id": int(raw["article_id"])}
    for name in CIVIL_LABELS:
        if not raw[name]:
            return None, "missing_base_label:" + name
        result[name] = float(raw[name])
    # Feature encoding happens only after all base labels are present. A row
    # omitted by the generator must not encounter a later dtype conversion.
    for name in ("article_id", "parent_id"):
        result[name] = _i32(result[name])
    for name in CIVIL_LABELS:
        number = result[name]
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("Civil base label is not a finite original fraction")
        result[name] = struct.unpack("<f", struct.pack("<f", number))[0]
    return result, None


def _civil_render(path, archive, records, lineage, members):
    counts = Counter()
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(sys.maxsize)
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            # ZIP already holds its central directory in memory. No row corpus
            # or complete label/source table is retained here.
            names = [i.filename for i in infos]
            if len(names) != len(set(names)) or names.count("civil_comments.csv") != 1:
                raise ValueError("Expected exactly one unambiguous civil_comments.csv member")
            target = None
            for ordinal, info in enumerate(infos, 1):
                _member_name(info.filename)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("ZIP symlinks are not accepted")
                with z.open(info) as stream:
                    content = _stream_binding(stream)
                if content["bytes"] != info.file_size:
                    raise ValueError("ZIP member size mismatch")
                item = {"member_ordinal": ordinal, "name": info.filename, "is_directory": info.is_dir(),
                        "compressed_bytes": info.compress_size, "crc32": info.CRC, **content,
                        "disposition": "csv_input" if info.filename == "civil_comments.csv" else "not_used_by_upstream_builder"}
                _line(members, item)
                counts["members_seen"] += 1
                counts["member_uncompressed_bytes"] += content["bytes"]
                if info.filename == "civil_comments.csv":
                    if info.is_dir():
                        raise ValueError("Expected Civil CSV is not a regular member")
                    target = item
            with z.open("civil_comments.csv") as binary, io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
                reader = csv.DictReader(text)
                header = reader.fieldnames
                required = {"id", "comment_text", "parent_text", "publication_id", "created_date", "parent_id", "article_id", "split", *CIVIL_LABELS}
                normalized_header = [_universal_newlines(name) for name in header] if header else []
                if not header or len(header) != len(set(normalized_header)) or not required.issubset(normalized_header):
                    raise ValueError("Civil raw CSV header is missing or ambiguous")
                for row_ordinal, raw in enumerate(reader, 1):
                    if None in raw or any(value is None for value in raw.values()):
                        raise ValueError("Civil CSV row width differs from the original header")
                    counts["rows_seen"] += 1
                    upstream = {_universal_newlines(k): _universal_newlines(v) for k, v in raw.items()}
                    split = CIVIL_SPLITS.get(upstream["split"])
                    value, omission = _civil_value(upstream) if split else (None, "not_in_base_official_splits")
                    loc = {"archive_sha256": archive["sha256"], "member_ordinal": target["member_ordinal"],
                           "member_name": target["name"], "member_sha256": target["sha256"],
                           "row_ordinal": row_ordinal, "csv_physical_end_line": reader.line_num,
                           "raw_values_sha256": _digest(raw), "raw_split": raw["split"], "official_split": split}
                    if value is not None:
                        counts["rows_exported"] += 1
                        counts["official_split:" + split] += 1
                        loc["export_ordinal"] = counts["rows_exported"]
                        value["_archive_lineage"] = loc
                        _line(records, value)
                    else:
                        counts["rows_omitted"] += 1
                        counts["omission:" + omission] += 1
                    _line(lineage, {"location": loc, "disposition": "exported" if value is not None else "omitted",
                                    "omission_reason": omission, "raw_values": raw})
                counts["csv_physical_lines"] = reader.line_num
    finally:
        csv.field_size_limit(previous_limit)
    return dict(counts)


def _news_value(raw):
    if not isinstance(raw, dict) or not set(NEWS_FIELDS.values()).issubset(raw):
        raise ValueError("CC-News article lacks required original fields")
    out = {}
    for name, original in NEWS_FIELDS.items():
        value = raw[original]
        if value is not None and not isinstance(value, str):
            raise ValueError("CC-News upstream field is neither string nor null")
        out[name] = value.strip() if value is not None else ""
    return out


def _news_render(path, archive, records, lineage, members):
    counts = Counter()
    # Streaming mode avoids seeking or unpacking the archive. Clear TarFile's
    # otherwise-growing member cache. Repeated names remain distinct instances.
    with tarfile.open(path, "r|gz") as tar:
        ordinal = 0
        while True:
            info = tar.next()
            if info is None:
                break
            ordinal += 1
            _member_name(info.name)
            if not (info.isfile() or info.isdir()):
                raise ValueError("Only regular CC-News files and directories are accepted")
            counts["members_seen"] += 1
            item = {"member_ordinal": ordinal, "name": info.name, "is_directory": info.isdir(),
                    "tar_header_offset": info.offset, "tar_data_offset": info.offset_data, "declared_bytes": info.size}
            if info.isdir():
                item.update({"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest(), "disposition": "directory"})
                _line(members, item)
                tar.members.clear()
                continue
            selected = posixpath.basename(info.name).endswith(".json")
            with tar.extractfile(info) as stream:
                if selected:
                    data = stream.read()
                    content = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                else:
                    content = _stream_binding(stream)
            if content["bytes"] != info.size:
                raise ValueError("Tar member size mismatch")
            item.update(content)
            item["disposition"] = "json_article" if selected else "not_selected_by_upstream_loader"
            _line(members, item)
            counts["member_uncompressed_bytes"] += content["bytes"]
            if selected:
                raw = json.loads(data)
                value = _news_value(raw)
                counts["rows_seen"] += 1
                counts["rows_exported"] += 1
                loc = {"archive_sha256": archive["sha256"], "member_ordinal": ordinal, "member_name": info.name,
                       "member_sha256": content["sha256"], "raw_values_sha256": _digest(raw),
                       "upstream_example_index": counts["rows_exported"] - 1, "export_ordinal": counts["rows_exported"],
                       "official_split": "train"}
                value["_archive_lineage"] = loc
                _line(records, value)
                _line(lineage, {"location": loc, "disposition": "exported", "omission_reason": None, "raw_values": raw})
                del data, raw, value
            else:
                counts["nonarticle_files_omitted"] += 1
            tar.members.clear()
    return dict(counts)


def _render_archive(dataset_id, archive_path, output_dir, archive):
    """Private grammar renderer, NOT archive acceptance. Used in format tests."""
    out = Path(output_dir)
    with (out / DATA_FILES[0]).open("xb") as records, (out / DATA_FILES[1]).open("xb") as lineage, (out / DATA_FILES[2]).open("xb") as members:
        counts = _render_streams(dataset_id, archive_path, archive, (records, lineage, members))
    return {"counts": counts, "outputs": {name: file_binding(out / name) for name in DATA_FILES}}


def _render_streams(dataset_id, archive_path, archive, streams):
    if dataset_id == "civil_comments":
        return _civil_render(archive_path, archive, *streams)
    if dataset_id == "cc_news":
        return _news_render(archive_path, archive, *streams)
    raise ValueError("Unsupported archive grammar")


class _HashSink:
    """Consume every rendered byte without retaining another corpus copy."""
    def __init__(self):
        self._hash = hashlib.sha256()
        self._bytes = 0

    def write(self, value):
        self._hash.update(value)
        self._bytes += len(value)
        return len(value)

    def binding(self):
        return {"bytes": self._bytes, "sha256": self._hash.hexdigest()}


def _hash_reconstruction(dataset_id, archive_path, archive, sinks):
    counts = _render_streams(dataset_id, archive_path, archive, tuple(sinks[name] for name in DATA_FILES))
    return {"counts": counts, "outputs": {name: sinks[name].binding() for name in DATA_FILES}}


def _check_population(dataset_id, counts):
    if dataset_id == "civil_comments":
        actual = {key: counts.get("official_split:" + key, 0) for key in CIVIL_COUNTS}
        if actual != CIVIL_COUNTS or counts.get("rows_exported", 0) != sum(CIVIL_COUNTS.values()):
            raise ValueError("Civil exported official split counts differ from TFDS 1.2.4")
    elif counts.get("rows_exported", 0) != 708241:
        raise ValueError("CC-News exported article count differs from the pinned release")


def _adapter_options(dataset_id):
    return {"input_format": "jsonl", "input_schema": "tfds_1_2_4" if dataset_id == "civil_comments" else "raw"}


def _failed(out, schema, dataset_id, exc, *, partial_sinks=None):
    value = {"schema": schema, "status": "failed", "dataset_id": dataset_id,
             "error_type": type(exc).__name__, "error": str(exc), "machine_acceptance_passed": False,
             "confirmatory_study_ready": False, "code_sha256": _LOADED_SOURCE_SHA256}
    if partial_sinks is not None:
        value["partial_reconstruction_prefixes"] = {name: sink.binding() for name, sink in partial_sinks.items()}
        value["prefixes_are_incomplete_unaccepted_outputs"] = True
    _write_json(out / "FAILED.json", value)
    return value


def export_archive(dataset_id, archive_path, output_dir):
    """Export exactly the published original; retain failures and partial files."""
    out = _new_output(output_dir)
    _write_json(out / "attempt.json", {"schema": SCHEMA, "operation": "export", "dataset_id": dataset_id,
                                      "code_sha256": _LOADED_SOURCE_SHA256, "private_corpus_outputs": True})
    try:
        _code_binding()
        expected = expected_archive(dataset_id)
        archive = file_binding(archive_path)
        _check_identity(dataset_id, archive)
        rendered = _render_archive(dataset_id, archive_path, out, archive)
        _check_population(dataset_id, rendered["counts"])
        if file_binding(archive_path) != archive:
            raise ValueError("Original archive changed during export")
        manifest = {"schema": SCHEMA, "status": "completed", "dataset_id": dataset_id,
                    "evidence_scope": "published_original_archive_bytes", "archive": archive, "expected_archive": expected,
                    "adapter": _adapter_options(dataset_id), **rendered, "code_sha256": _code_binding(),
                    "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation()},
                    "published_archive_identity_verified": True, "source_permissions_verified": False,
                    "source_semantics_verified": False, "confirmatory_study_ready": False,
                    "private_corpus_outputs": True, "source_splits_are_lineage_not_study_partitions": True}
        _write_json(out / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        _failed(out, SCHEMA, dataset_id, exc)
        raise


def _validate_manifest(manifest):
    dataset_id = manifest.get("dataset_id")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "completed" or manifest.get("evidence_scope") != "published_original_archive_bytes":
        raise ValueError("Completed original-archive export required; format fixtures are refused")
    if manifest.get("expected_archive") != expected_archive(dataset_id):
        raise ValueError("Export does not bind the fixed upstream release")
    _check_identity(dataset_id, manifest.get("archive"))
    if manifest.get("code_sha256") != _code_binding() or manifest.get("adapter") != _adapter_options(dataset_id):
        raise ValueError("Export code/schema differs from this replay implementation")
    if set(manifest.get("outputs", {})) != set(DATA_FILES):
        raise ValueError("Export output inventory differs")
    required = {"published_archive_identity_verified": True, "source_permissions_verified": False,
                "source_semantics_verified": False, "confirmatory_study_ready": False,
                "private_corpus_outputs": True, "source_splits_are_lineage_not_study_partitions": True}
    if any(manifest.get(key) is not value for key, value in required.items()):
        raise ValueError("Export evidence boundary differs")
    _check_population(dataset_id, manifest.get("counts", {}))
    return dataset_id


def _compare_reconstruction(manifest, rendered):
    if manifest["counts"] != rendered["counts"] or manifest["outputs"] != rendered["outputs"]:
        raise ValueError("Archive replay differs in records, raw lineage, members, or counts")


def replay_export(export_dir, archive_path, output_dir):
    """Reconstruct every export byte into hash/count sinks in a NEW attempt.

The report establishes archive-to-export derivation only. Original rights,
native-source interpretation, encoder caches, and human work remain external.
"""
    out = _new_output(output_dir)
    supplied = Path(export_dir)
    _write_json(out / "attempt.json", {"schema": REPLAY_SCHEMA, "operation": "replay", "code_sha256": _LOADED_SOURCE_SHA256,
                                      "private_corpus_outputs": True})
    dataset_id = None
    sinks = None
    try:
        _code_binding()
        if (supplied / "FAILED.json").exists():
            raise ValueError("Failed export cannot be accepted")
        manifest, manifest_hash = _read_json(supplied / "manifest.json")
        dataset_id = _validate_manifest(manifest)
        archive = file_binding(archive_path)
        _check_identity(dataset_id, archive)
        if archive != manifest["archive"]:
            raise ValueError("Supplied archive differs from export binding")
        for name in DATA_FILES:
            if file_binding(supplied / name) != manifest["outputs"][name]:
                raise ValueError("Stored export output differs: " + name)
        sinks = {name: _HashSink() for name in DATA_FILES}
        rendered = _hash_reconstruction(dataset_id, archive_path, archive, sinks)
        _compare_reconstruction(manifest, rendered)
        if file_binding(archive_path) != archive:
            raise ValueError("Archive changed during replay")
        if file_binding(supplied / "manifest.json")["sha256"] != manifest_hash:
            raise ValueError("Export manifest changed during replay")
        for name in DATA_FILES:
            if file_binding(supplied / name) != rendered["outputs"][name]:
                raise ValueError("Stored export changed during replay: " + name)
        bound = {"dataset_id": dataset_id, "archive": archive, "export_manifest_sha256": manifest_hash,
                 "outputs": rendered["outputs"], "counts": rendered["counts"], "code_sha256": _code_binding()}
        report = {"schema": REPLAY_SCHEMA, "status": "completed", **bound, "replay_binding_sha256": _digest(bound),
                  "machine_acceptance_passed": True, "published_archive_identity_verified": True,
                  "all_export_rows_replayed": True, "source_permissions_verified": False,
                  "source_semantics_verified": False, "confirmatory_study_ready": False,
                  "private_corpus_outputs": True, "reconstruction_storage": "streaming_hashes_only"}
        _write_json(out / "replay.json", report)
        return report
    except Exception as exc:
        _failed(out, REPLAY_SCHEMA, dataset_id, exc, partial_sinks=sinks)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("export")
    p.add_argument("dataset_id", choices=("civil_comments", "cc_news"))
    p.add_argument("archive"); p.add_argument("--out", required=True)
    p = sub.add_parser("replay")
    p.add_argument("export_dir"); p.add_argument("archive"); p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        report = export_archive(args.dataset_id, args.archive, args.out) if args.operation == "export" else replay_export(args.export_dir, args.archive, args.out)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                          "confirmatory_study_ready": False}), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "dataset_id": report["dataset_id"],
                      "published_archive_identity_verified": True, "confirmatory_study_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
