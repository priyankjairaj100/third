Archive reader adapter v1: engineering compatibility only

This separately versioned executable supplies exactly the existing source-lineage
audit's command: archive_reader.py x -so -y ARCHIVE Posts.xml|PostLinks.xml.
No other commands, flags, members, duplicate target names, or extra output streams
are accepted. It requires already-installed py7zr 1.1.3 and installs nothing.
The frozen source-lineage audit and all original inputs remain unchanged.

The executable opens the archive read-only and passes a supplied file handle to
py7zr, disabling its parallel extraction path. extract(targets=[member],
recursive=False,factory=...) writes only to the provided append-only stdout sink.
The requested archive member must occur once and be a regular file. No extracted
files/directories are created. Bytecode writes are disabled. The executable
preserves the stdout pipe and redirects descriptor 1 to stderr before loading
py7zr, so diagnostics cannot contaminate the preserved member-byte pipe.

Terminal-seek compatibility is explicit. Pinned py7zr 1.1.3 seeks its output to
zero after decompression before checking CRC, then MemIO.close() seeks zero again.
After EXACTLY the archive-declared byte count, the adapter recognizes these two
calls solely as terminal notifications; stdout is never rewound. The first call
permanently prohibits further writes. All reads, other seeks, premature markers,
extra streams, and incomplete closes fail. Empty members permit one or two terminal
markers because their archive representation may omit the CRC-stage seek.
CRC/decompression errors return nonzero even if bytes already
reached stdout; the source audit independently requires successful process exit.

Launch receipt
--------------
Use the intended cluster Python via PATH for the executable's /usr/bin/env python3
shebang. The current intended interpreter is /nfs_home/software/miniconda/bin/python3.
The parent launch must save a new private receipt before the source audit, calling
archive_reader.runtime_binding() under that same interpreter. This returns the
actual executable/Python version, pinned py7zr version, adapter hash, and hashes of
every py7zr .py module plus distribution METADATA/RECORD/WHEEL. The parent also binds
its launch command, dependency distributions, and sealed adapter package. Neither
the receipt nor any diagnostics is emitted into an extracted member stream.
Pass the absolute executable path as the existing audit's --seven-zip argument.
There is no additional adapter CLI receipt switch or runtime download fallback.

Tests
-----
PYTHONDONTWRITEBYTECODE=1 /path/to/py7zr-1.1.3-python -m unittest discover \
  -s local_run/archive_reader_py7zr_v1_20261011 -p test_reader.py

Tests use actual tiny software 7z archives, including both allowlisted members,
an empty file, an unrelated member, duplicate names, CRC corruption and truncation.
They check exact subprocess stdout, no extracted files, source-byte preservation,
strict command/member validation, terminal-seek refusal, partial stdout writes,
diagnostic separation, and runtime binding. No natural source data is used.
Local compatibility does not establish cluster execution, publisher authenticity,
historical extraction identity, full archive completeness, or scientific acceptance.

Primary API/source references reviewed:
https://py7zr.readthedocs.io/en/stable/api.html
https://raw.githubusercontent.com/miurahr/py7zr/v1.1.3/py7zr/io.py
https://raw.githubusercontent.com/miurahr/py7zr/v1.1.3/py7zr/py7zr.py
