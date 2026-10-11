# AskUbuntu community-mirror identity check

This standard-library checker reads the existing archive once. It compares its
size, SHA1, MD5 and CRC32 with the exact saved HTTPS Internet Archive metadata
response, and its SHA256 with the already verified local archive identity.
The metadata response, full file entry, retrieval URL/date, checker and expectation
bytes are bound in the result. Archive inode/device/size/mode/mtime/ctime must
remain unchanged across the open descriptor and final path check; metadata,
expectations and checker bytes are checked again. Inputs are opened read-only.

Run only in the authorized CPU allocation, with a new private output filename:

```bash
python check_mirror.py --archive /private/inputs/askubuntu.com.7z \
  --metadata /private/archive_metadata_raw.json --out /private/result.json \
  --expectations-sha256 EXPECTED_SHA256
```

The launcher must pin the checker bytes and supply the reviewed expectation hash.
The output directory must already exist; existing result files are never replaced.
No downloads, extraction, source edits, model work, or additional archive pass
occur. Failures return exit 2 and preserve a failure receipt when possible.
Timing includes checks through finalization before the result write; it is not
encoding performance. RSS units are stated explicitly.

A match establishes identity to the named community mirror. The mirror declares
an April 2, 2024 dump copied on April 11. Its uploader-supplied creator and
`source: original` metadata do not prove official origin, corrected-release
identity, publisher completeness, or attribution compliance. Those gates, primary
admission, cache acceptance and study readiness remain false. The frozen source
route is unchanged. Raw metadata and resulting private receipts stay outside Git.
