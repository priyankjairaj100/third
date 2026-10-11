#!/usr/bin/env python3
"""One-pass identity check against a pinned community-mirror metadata response."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time
import zlib


class Refusal(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise Refusal(code)


def binding(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def stat_binding(value):
    return {name: getattr(value, "st_" + name) for name in
            ("dev", "ino", "mode", "size", "mtime_ns", "ctime_ns")}


def stream_checksums(path):
    before = stat_binding(path.stat())
    require(stat.S_ISREG(before["mode"]), "archive_not_regular_file")
    hashes = {"sha256": hashlib.sha256(), "sha1": hashlib.sha1(usedforsecurity=False),
              "md5": hashlib.md5(usedforsecurity=False)}
    size, crc = 0, 0
    with path.open("rb") as stream:
        require(stat_binding(os.fstat(stream.fileno())) == before, "archive_changed_before_read")
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            crc = zlib.crc32(block, crc)
            for digest in hashes.values():
                digest.update(block)
        after_fd = stat_binding(os.fstat(stream.fileno()))
    after_path = stat_binding(path.stat())
    result = {"bytes": size, **{k: h.hexdigest() for k, h in hashes.items()}, "crc32": f"{crc & 0xffffffff:08x}"}
    return result, {"before": before, "after_open_descriptor": after_fd, "after_path": after_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "metadata", "out", "expectations-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    out = Path(args.out).absolute()
    require(out.parent.is_dir() and not out.exists(), "new_output_file_in_existing_directory_required")
    os.umask(0o077)
    started = time.perf_counter()
    report = {"schema": "third-source-mirror-identity-v1", "passed": False,
              "mirror_identity_verified": False, "publisher_authenticity_verified": False,
              "snapshot_completeness_verified": False, "attribution_review_completed": False,
              "primary_input_acceptance": False, "full_population_cache_acceptance": False,
              "confirmatory_study_ready": False, "frozen_source_route_changed": False,
              "network_requests": 0, "archive_content_passes": 0,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Identity against pinned community-mirror metadata only; no official origin or corrected-release authentication."}
    try:
        script = Path(__file__).resolve(strict=True)
        expectations = script.with_name("expectations.json")
        metadata = Path(args.metadata).resolve(strict=True)
        archive = Path(args.archive).resolve(strict=True)
        watched = {path: binding(path) for path in (script, expectations, metadata)}
        report["input_bindings"] = {str(path): value for path, value in watched.items()}
        require(watched[expectations]["sha256"] == args.expectations_sha256, "expectations_sha256_mismatch")
        pins = json.loads(expectations.read_bytes())
        require(watched[metadata] == pins["metadata_file"], "metadata_binding_mismatch")
        value = json.loads(metadata.read_bytes())
        require(value["metadata"]["identifier"] == pins["item_identifier"], "mirror_item_mismatch")
        entries = [entry for entry in value["files"] if entry.get("name") == pins["file_entry"]["name"]]
        require(entries == [pins["file_entry"]], "mirror_file_entry_mismatch")
        require(archive.name == pins["file_entry"]["name"], "archive_filename_mismatch")
        report["metadata_source"] = {key: pins[key] for key in ("metadata_url", "metadata_retrieved_utc", "item_identifier")}
        report["archive_path"] = str(archive)
        read_start = time.perf_counter()
        observed, stats = stream_checksums(archive)
        report.update({"archive_content_passes": 1, "observed": observed, "archive_stats": stats,
                       "stream_read_seconds": time.perf_counter() - read_start})
        wanted = {key: pins["file_entry"][key] for key in ("md5", "sha1", "crc32")}
        wanted.update({"bytes": int(pins["file_entry"]["size"]), "sha256": pins["verified_local_sha256"]})
        report["expected"] = wanted
        require(stats["before"] == stats["after_open_descriptor"] == stats["after_path"], "archive_stat_changed_during_read")
        require(observed == wanted, "archive_checksum_or_size_mismatch")
        require(all(binding(path) == expected for path, expected in watched.items()), "code_metadata_or_expectations_changed")
        report.update({"passed": True, "mirror_identity_verified": True, "initial_final_bindings_equal": True})
    except Exception as error:
        report["error"] = str(error) if isinstance(error, Refusal) else type(error).__name__
    report["seconds_before_result_write"] = time.perf_counter() - started
    report["peak_rss_native_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_unit"] = "bytes" if sys.platform == "darwin" else "KiB"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({key: report[key] for key in ("passed", "mirror_identity_verified")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
