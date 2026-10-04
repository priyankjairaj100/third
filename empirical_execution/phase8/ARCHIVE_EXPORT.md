# Original archive exports and replay

`archive_export.py` implements the previously missing Civil and CC-News export boundary.
It uses only the Python standard library and local files.
It never downloads archives or publishes their contents.
Phases 3–7 remain unchanged.

The genuine archives are still absent.
Positive checks exercise software format fixtures, including existing natural Civil text and explicitly artificial metadata.
They are not authentic corpus acceptance or empirical results.
The production entry points reject those fixtures.

## Fixed archive identities

| Dataset ID | Original file | Bytes | SHA256 |
| --- | --- | --- | --- |
| `civil_comments` | `civil_comments_v1.2.zip` | 448174578 | `89a60573acbbfaf189b2c77278115eefcbc1654dbd8235dd8fbecc1b83f1585c` |
| `cc_news` | `cc_news.tar.gz` | 845131146 | `1aaf8e5af33e3a73472b58afba48c6a839ebc2dd190c4e0754fc00f8899a9cec` |

There is no command-line or public-API identity override.
The archive size and complete SHA256 are checked before parsing begins.
The original file is hashed again after reconstruction.
Renamed copies are permitted only when the bytes match exactly.

Civil authority is TFDS revision `ff070f916bdbc7f5f833e2bf9e48336946d6fb56`:
[builder](https://github.com/tensorflow/datasets/blob/ff070f916bdbc7f5f833e2bf9e48336946d6fb56/tensorflow_datasets/text/civil_comments.py),
[checksums](https://github.com/tensorflow/datasets/blob/ff070f916bdbc7f5f833e2bf9e48336946d6fb56/tensorflow_datasets/url_checksums/civil_comments.txt),
and [catalog counts](https://www.tensorflow.org/datasets/catalog/civil_comments).
The pinned code and checksum file were read through the official GitHub repository.

News authority is maintainer revision `0ae17dffa45f621cb8061a8393e2dd225d0876a0`:
the [hosting commit](https://huggingface.co/datasets/vblagoje/cc_news/commit/0ae17dffa45f621cb8061a8393e2dd225d0876a0)
records the original tarball's LFS digest and size.
The [subsequent conversion diff](https://huggingface.co/datasets/vblagoje/cc_news/commit/81eb2ce0d2a9dad6ad16b68ef750ec290880fa36)
contains the complete removed loader, including member selection and field mapping.
Direct pinned loader pages were unavailable through the web reader; the official diff supplied the code.
Converted Parquet shards are not accepted by this original-tarball interface.

## Construction contract

Civil reads exactly one `civil_comments.csv` ZIP member.
Duplicate ZIP member names are refused rather than resolved ambiguously.
All other members are hashed and recorded without being used as examples.
Every parsed CSV record receives a raw-value lineage entry, including omitted rows.
Raw string lexemes, unknown columns, and original line endings remain in that ledger.

The full-feature export uses the upstream base configuration, `CivilComments/1.2.4`:

- `comment_text` becomes `text`; parent text remains separate.
- Comment, publication, article, parent, and date fields are preserved through the declared upstream conversions.
- An empty parent becomes zero; other parents use `int(float(value))`; articles use `int(value)`.
- Common fields are parsed before the seven labels, in the upstream order.
- The first empty required label omits the row. Earlier parse errors remain errors.
- Label values become FP32, represented exactly as JSON numbers containing their Python float values.
- The raw decimal fractions remain available in the lineage ledger and original archive.
- `train`, `test_public`, and `test_private` map to official train, validation, and test lineage respectively.
- Other split values are omitted before parsing common fields, matching the base split generators.

The local TFDS text reader uses universal newline translation.
The exporter applies that translation to feature values while preserving untranslated CSV strings separately.
This covers embedded CRLF and bare CR inside quoted fields.
The behavior is checked against the upstream parser and the local `epath` OS-backend contract.
This is a value export, not a recreation of TFRecord shards or their ordering.
Output order follows original CSV records; official splits remain explicit lineage fields.

Before success, exported split counts must equal 1,804,874 / 97,320 / 97,320.
These are counts before the frozen adapter's later exclusions.
Int32 overflow, nonfinite fractions, out-of-range fractions, malformed headers, and inconsistent row widths fail closed.
Those integrity checks are stricter than permissive parsing of malformed input; they are not silent row omissions.

News processes logical TAR members in archive order without extracting files to disk.
It selects regular files whose literal POSIX basename ends with `.json`, matching the loader's filter.
Directories and nonselected regular files remain in the member ledger.
Links and special files are refused. Unsafe member names are refused even though extraction is never performed.
Repeated member names, article text, and URLs remain distinct instances with different member ordinals.

| Original JSON field | Export field |
| --- | --- |
| `maintext` | `text` |
| `source_domain` | `domain` |
| `date_publish` | `date` |
| `title`, `description`, `url`, `image_url` | Same names |

The loader strips strings and converts nulls to empty strings.
Missing fields are errors; the exporter never invents dates, domains, labels, or native source identifiers.
Original JSON objects are preserved in lineage; complete member-byte hashes bind their original serialization.
Duplicate JSON keys follow the upstream JSON parser's last-value behavior; the member digest preserves the exact source bytes.
Nonfinite values cannot be serialized into the strict JSON lineage and therefore fail closed.
Blank articles or invalid dates remain exported for the frozen adapter to account for independently.
Success requires exactly 708,241 exported article instances.
This count does not prove complete calendar coverage or native-source meaning.

The original official split fields are provenance, **not the study's train/calibration/test assignments**.
The frozen source/date preparation rules still define those study partitions.
No export step fits a model, selects a threshold, or uses held-out outcomes for a decision.

## API and files

```python
manifest = export_archive(dataset_id, archive_path, output_dir)
report = replay_export(export_dir, archive_path, new_replay_dir)
```

All output directories must be new.
The code records an attempt, preserves partial outputs, and writes `FAILED.json` when a production call fails.
Existing directories are refused without changing them.
The module captures its source hash when loaded and refuses completion after source changes.
No output is marked accepted merely because it has a matching manifest checksum.

| File | Meaning |
| --- | --- |
| `records.jsonl` | Deterministic JSONL in the schema consumed by frozen Phase 4. |
| `lineage.jsonl` | Every Civil input row or selected News article, including original values and omissions. |
| `members.jsonl` | Every logical archive member, its ordinal, size, content hash, and disposition. |
| `manifest.json` | Successful export identity, counts, file hashes, adapter schema, source hash, and evidence boundary. |
| `replay.json` | Successful complete reconstruction, including `replay_binding_sha256`. |

`_archive_lineage` accompanies each exported record and survives in the adapter's `original_fields`.
It binds archive, member, raw values, output ordinal, and official source split.
Civil additionally binds CSV record ordinal and physical ending line.
News additionally records the upstream zero-based example index and TAR offsets in the member ledger.
Archive/member hashes bind headers and serialization details that are not reconstructed from parsed values.

Replay first verifies the fixed original identity and every stored output hash.
It then reconstructs all records, raw lineage, and member inventory from the original archive into streaming hash/count sinks.
All three complete file bindings and all counts must match exactly.
Every generated byte contributes to its file's SHA256; no second corpus copy is written.
Failed replay attempts retain error details and any partial-output prefix bindings, clearly marked incomplete and unaccepted.
It rechecks the original archive, supplied manifest, and supplied outputs before success.
The replay receipt binds those exact inputs and the exporter source version.
The Phase 8 dossier wrapper must carry this receipt alongside ordinary frozen source/cache acceptance.

`machine_acceptance_passed` means this archive-to-export construction passed.
It does not establish permissions, source semantics, calendar completeness, encoder correctness, or genuine human collection.
`confirmatory_study_ready` remains false.
Private format-rendering helpers never issue an original-archive manifest.
Production replay rejects fixture scopes, substituted identities, failed runs, edited outputs, and changed reconstruction counts.

## Local commands after authorized intake

```bash
python empirical_execution/phase8/archive_export.py export civil_comments \
  "$CIVIL_ARCHIVE" --out "$PRIVATE_CIVIL_EXPORT"
python empirical_execution/phase8/archive_export.py replay \
  "$PRIVATE_CIVIL_EXPORT" "$CIVIL_ARCHIVE" --out "$PRIVATE_CIVIL_REPLAY"

python empirical_execution/phase8/archive_export.py export cc_news \
  "$NEWS_ARCHIVE" --out "$PRIVATE_NEWS_EXPORT"
python empirical_execution/phase8/archive_export.py replay \
  "$PRIVATE_NEWS_EXPORT" "$NEWS_ARCHIVE" --out "$PRIVATE_NEWS_REPLAY"
```

These variables refer to supplied original files and new private output paths.
The Civil adapter options are `input_format=jsonl`, `input_schema=tfds_1_2_4`.
News uses `input_format=jsonl`, `input_schema=raw`, plus the separately pinned public suffix list.
No raw article bodies belong in the public repository.
Generated attempt directories include a Git ignore-all rule to prevent ordinary accidental adds.
That rule does not prevent deliberate force-adds or copying data into another directory.

## Resource and verification limits

Neither exporter keeps the complete corpus, labels, or source table in memory.
Civil retains ZIP central-directory metadata and several copies of the current CSV record.
News clears Python's otherwise growing TAR-member cache after each member.
Its working set includes several copies of the largest JSON article, decompression buffers, and TAR/PAX metadata.
There is no claim of constant RSS independent of member size or metadata.
The implementation has not been measured on the authentic full archives.

Each production export makes two complete archive hash passes around parsing.
Civil also decompresses its CSV once for member hashing and once for parsing.
Replay repeats reconstruction and hashes existing outputs before and after comparison.
The raw-value ledger intentionally adds disk cost; replay keeps only attempt/failure/receipt metadata and its streaming digest states.
All emitted data-file sizes are recorded in the manifest.
Retained original archives, export records, raw-value lineage, member inventory, metadata, and failed exports all consume disk.
Downstream adapter outputs, prepared records, model assets, encoder caches, and task artifacts add separate storage costs.
The parent session observed about 2 GiB of free overlay disk during this work.
Authentic full-corpus output sizes are unmeasured; no feasibility estimate is inferred from the streaming memory structure.

The checker verifies format semantics, adapter compatibility, deterministic reconstruction, and production refusal boundaries.
The independent review found and corrected source-change binding, Civil embedded-newline translation, and literal News basename selection.
No fake authentic-data success or fabricated human response was used.

Earlier check reports preserve their outcomes and code hashes.
Exact intermediate source snapshots were not retained before review edits.
Those earlier versions therefore cannot be claimed independently replayable from this checkpoint.
The final report and frozen final source snapshot are the authoritative software check evidence.
