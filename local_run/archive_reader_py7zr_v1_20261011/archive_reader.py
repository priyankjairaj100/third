#!/usr/bin/env python3
"""Only: archive_reader.py x -so -y ARCHIVE Posts.xml|PostLinks.xml."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
# Preserve the real output pipe, then send even native-library diagnostics to
# stderr. Only the writer below receives the preserved descriptor.
_cli_sink = None
if __name__ == "__main__":
    _cli_sink = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
with contextlib.redirect_stdout(sys.stderr):
    import py7zr
    from py7zr.io import Py7zIO, WriterFactory

VERSION = "1.1.3"
MEMBERS = frozenset({"Posts.xml", "PostLinks.xml"})


class ReaderError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise ReaderError(code)


def check_version():
    require(importlib.metadata.version("py7zr") == VERSION and py7zr.__version__ == VERSION,
            "py7zr_version_mismatch")


def binding(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def runtime_binding():
    """For the parent's separate launch receipt; never emitted by extraction."""
    check_version()
    package = Path(py7zr.__file__).resolve().parent
    distribution = importlib.metadata.distribution("py7zr")
    files = {str(p.resolve()) for p in package.rglob("*.py")}
    for item in distribution.files or ():
        if str(item).endswith((".dist-info/METADATA", ".dist-info/RECORD", ".dist-info/WHEEL")):
            files.add(str(distribution.locate_file(item).resolve()))
    return {"schema": "third-archive-reader-py7zr-runtime-v1", "py7zr_version": VERSION,
            "python": sys.version, "executable": sys.executable,
            "adapter": binding(__file__), "module_and_distribution_files": [binding(p) for p in sorted(files)],
            "scope": "Current local reader identity; no publisher authentication or historical execution claim"}


class StdoutWriter(Py7zIO):
    """Append-only; recognize py7zr's two terminal rewind notifications."""
    def __init__(self, sink, expected_size):
        require(type(expected_size) is int and expected_size >= 0, "invalid_member_size")
        self.sink, self.expected_size = sink, expected_size
        self.written, self.terminal_seeks, self.closed = 0, 0, False

    def write(self, data):
        require(not self.closed and self.terminal_seeks == 0, "write_after_terminal_marker")
        require(isinstance(data, (bytes, bytearray, memoryview)), "nonbyte_write")
        require(self.written + len(data) <= self.expected_size, "member_larger_than_declared")
        remaining = memoryview(data)
        while remaining:
            count = self.sink.write(remaining)
            require(type(count) is int and 0 < count <= len(remaining), "stdout_short_or_failed_write")
            self.written += count
            remaining = remaining[count:]
        return len(data)

    def read(self, size=None):
        raise ReaderError("stream_read_forbidden")

    def seek(self, offset, whence=0):
        # py7zr 1.1.3 first seeks to zero after decompression (before its CRC
        # comparison), then MemIO.close() does so again. These are terminal
        # notifications only: stdout never rewinds and no writes remain legal.
        require(not self.closed and offset == 0 and whence == 0
                and self.written == self.expected_size and self.terminal_seeks < 2,
                "seek_forbidden_or_premature")
        self.terminal_seeks += 1
        return 0

    def flush(self):
        self.sink.flush()

    def size(self):
        return self.written

    def close(self):
        allowed_seeks = {2} if self.expected_size else {1, 2}
        require(not self.closed and self.written == self.expected_size
                and self.terminal_seeks in allowed_seeks, "incomplete_member_close")
        self.flush()
        self.closed = True


class SingleMemberFactory(WriterFactory):
    def __init__(self, member, size, sink):
        require(member in MEMBERS, "unexpected_member")
        self.member, self.writer, self.created = member, StdoutWriter(sink, size), False

    def create(self, filename):
        require(filename == self.member and not self.created, "unexpected_or_repeated_output_stream")
        self.created = True
        return self.writer


def stream_member(archive_path, member, sink):
    check_version()
    require(member in MEMBERS, "unexpected_member")
    path = Path(archive_path).resolve(strict=True)
    require(path.is_file(), "archive_not_file")
    # A supplied read-only handle disables py7zr's parallel path. Factory mode
    # skips filesystem extraction and metadata writes, including for other items.
    with path.open("rb") as source, contextlib.redirect_stdout(sys.stderr):
        with py7zr.SevenZipFile(source, mode="r") as archive:
            require(not archive.needs_password(), "encrypted_archive_unsupported")
            selected = [item for item in archive.list() if item.filename == member]
            require(len(selected) == 1, "missing_or_duplicate_member")
            item = selected[0]
            require(item.is_file and not item.is_directory and not item.is_symlink, "nonregular_member")
            factory = SingleMemberFactory(member, item.uncompressed, sink)
            archive.extract(targets=[member], recursive=False, factory=factory)
            require(factory.created and factory.writer.closed
                    and factory.writer.written == item.uncompressed, "incomplete_extraction")


def main(argv=None, sink=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 5 and args[:3] == ["x", "-so", "-y"] and args[4] in MEMBERS,
                "unsupported_command_contract")
        stream_member(args[3], args[4], _cli_sink if sink is None else sink)
        return 0
    except Exception as error:
        # No member contents, archive paths, or arbitrary library exception text.
        print(str(error) if isinstance(error, ReaderError) else type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    result = main()
    _cli_sink.close()
    raise SystemExit(result)
