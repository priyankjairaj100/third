# Prospective original-corpus acquisition map

Reviewed 2026-10-04 against the narrative protocol and Phase 6 required inputs.
This review acquired metadata and source code, **no original corpus bytes**.
It does not change the frozen protocol, parsers, or acceptance gates.

## Civil Comments: exact archive identified; raw import remains blocked

The official TFDS builder specifies `civil_comments/CivilComments`, version `1.2.4`.
Its archive is [`civil_comments_v1.2.zip`](https://storage.googleapis.com/jigsaw-unintended-bias-in-toxicity-classification/civil_comments_v1.2.zip).
The archive member expected by the builder is `civil_comments.csv`.

| Published binding | Value |
| --- | --- |
| Archive bytes | 448,174,578 |
| SHA256 | `89a60573acbbfaf189b2c77278115eefcbc1654dbd8235dd8fbecc1b83f1585c` |
| Inspected TFDS code revision | `ff070f916bdbc7f5f833e2bf9e48336946d6fb56` |
| Required configuration | `CivilComments`, base seven labels; not Covert, Identities, or Context |

Sources: immutable [builder](https://github.com/tensorflow/datasets/blob/ff070f916bdbc7f5f833e2bf9e48336946d6fb56/tensorflow_datasets/text/civil_comments.py)
and [checksum registry](https://github.com/tensorflow/datasets/blob/ff070f916bdbc7f5f833e2bf9e48336946d6fb56/tensorflow_datasets/url_checksums/civil_comments.txt).
Both were read through the GitHub connector.
The [official catalog](https://www.tensorflow.org/datasets/catalog/civil_comments) identifies the dataset and underlying comment text as CC0.
It reports 1,804,874 train, 97,320 validation, and 97,320 test examples for this configuration.

The following mapping is established by the upstream code, not by inspection of an acquired CSV.
These are the required raw columns; the complete archive header remains uninspected.

| Raw CSV column | TFDS 1.2.4 output | Upstream conversion |
| --- | --- | --- |
| `id` | `id` | String |
| `comment_text` | `text` | Own comment text |
| `parent_text` | `parent_text` | String; excluded from our model input |
| `publication_id` | `publication_id` | String |
| `created_date` | `created_date` | String |
| `article_id` | `article_id` | `int(...)`, then declared int32 |
| `parent_id` | `parent_id` | Empty becomes 0; otherwise `int(float(...))`, then declared int32 |
| `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, `sexual_explicit` | Same names | Parse floats; TFDS declares float32 |
| `split` | Dataset split, not an example feature | `train` → train; `test_public` → validation; `test_private` → test |

The builder skips rows missing any required base label.
All required article, publication, parent, and date fields survive the official full-feature export.
The supervised `(text, toxicity)` view discards those fields and cannot support our source-rich intake.
The archive lacks user IDs; article withdrawal cannot become an author-withdrawal claim.

**Unresolved integration:** `phase4/adapters.py:adapt_civil` accepts a TFDS-style export containing `text`.
It does not accept the archive's raw `comment_text` schema.
Its CSV mode means CSV encoding of the export schema, not arbitrary original CSV compatibility.
An audited, separately versioned archive-to-export step is still required.
No exporter was implemented in this review.

That step must bind the original ZIP, member bytes, row ordinals, raw lexical values, official splits, and exported bytes.
It must reproduce the declared TFDS conversions and omissions, including float32 labels.
Preserve raw fractions alongside converted values to make precision changes auditable.
Record every omitted row and retain split lineage without importing held-out labels into fitting or threshold decisions.
Verify the upstream counts before downstream eligibility exclusions; disclose any mismatch.
Do not silently rename columns, guess absent sources, or replace this release with Civil100.

## WCEP: exact author file links identified; content bindings remain unknown

Inspected author repository revision: `8aa8418d55a4e6710277bed18bfc5349b50a5d96`.
The [README](https://github.com/complementizer/wcep-mds-dataset/blob/8aa8418d55a4e6710277bed18bfc5349b50a5d96/README.md)
labels its extracted-data update `6.10.20` and links an
[author Drive folder](https://drive.google.com/drive/folders/1T5wDxu4ajFwEq77dG88oE95e8ppREamg?usp=sharing).
The [author notebook](https://github.com/complementizer/wcep-mds-dataset/blob/8aa8418d55a4e6710277bed18bfc5349b50a5d96/wcep_getting_started.ipynb)
provides these download names and file IDs:

| Name in author notebook | Author download URL | Published byte size or digest verified here |
| --- | --- | --- |
| `train.jsonl.gz` | https://drive.google.com/uc?id=1kUjSRXzKnTYdJ732BkKVLg3CFxDKo25u | None |
| `val.jsonl.gz` | https://drive.google.com/uc?id=1_kHTZ32jazTbXaFRg0vBeIsVcpI7CTmy | None |
| `test.jsonl.gz` | https://drive.google.com/uc?id=1qsd5pOCpeSXsaqNobXCrcAzhcjtG1wA1 | None |

These are original extracted event/article-instance files, not the separate initial release without article text.
File IDs and the code revision are locators, not content hashes for the Drive files.
Current Drive contents, sizes, hashes, completeness, URL coverage, and event-size distribution remain unverified.
The folder's web metadata was inaccessible through the research tool.

The complete [pinned Git tree](https://api.github.com/repos/complementizer/wcep-mds-dataset/git/trees/8aa8418d55a4e6710277bed18bfc5349b50a5d96?recursive=1)
contains no extracted article archive; the tree response was not truncated.
Its data directory contains a crawl-path list, not the required article bytes.
The [releases collection](https://api.github.com/repos/complementizer/wcep-mds-dataset/releases) was empty on review.
The [MIT license](https://github.com/complementizer/wcep-mds-dataset/blob/8aa8418d55a4e6710277bed18bfc5349b50a5d96/LICENSE)
covers repository software; this is not evidence of blanket publisher-text rights.
Keep article bodies outside public backups under the project policy.

After authorized transfer, retain and hash all three original compressed files.
Review release identity and complete split coverage before claiming original-release acceptance.
The existing Phase 5 parser accepts JSONL or gzipped JSONL and preserves event/article lineage.
Verify the intended WCEP-100 scope from actual bytes; do not silently truncate larger events.
Unverified article URLs leave this arm record-only under the protocol.

## Other required corpora

| Corpus | Sourced input facts | Remaining questions |
| --- | --- | --- |
| AskUbuntu and English Stack Exchange | Corrected April 2024 company snapshot; `Posts.xml` and `PostLinks.xml` are required by our protocol. The company reports replacement completion on 2024-04-07 at 19:38 UTC. | A currently authorized official route for that historical snapshot was not established. Exact site archive names, byte sizes, and digests remain unverified. Do not substitute the initial buggy April release, today's release, or community mirrors. |
| CC-News | Maintainer `vblagoje/cc_news`; 708,241 English articles dated January 2017–December 2019. Original hosted `data/cc_news.tar.gz` is pinned below. | Transfer is unavailable here. Verify calendar coverage, original rows, duplicates, null dates, and source URLs from actual bytes. The maintainer lists the license as unknown. |

Stack sources: the [release catalog](https://meta.stackexchange.com/questions/224873/all-stack-exchange-data-dump-releases)
distinguishes corrected April 2024 from the initial faulty release.
The [company correction notice](https://meta.stackexchange.com/questions/398279/shifting-the-data-dump-schedule-a-proposal/398606)
records the replacement time and fixes, including UTF-8 XML and original tag delimiters.
[Current download help](https://stackoverflow.com/help/data-dumps) offers the latest snapshot through account settings.
It requires a declaration about intended LLM training; no declaration or account action was performed here.
That flow does not establish historical April 2024 access.
[Content licensing](https://stackoverflow.com/help/licensing) identifies CC BY-SA 4.0 for contributions from 2018-05-02 onward.
Preserve applicable attribution and revision metadata; access conditions are a separate question.

### CC-News original tarball and export boundary

The maintainer's [June 2023 hosting commit](https://huggingface.co/datasets/vblagoje/cc_news/commit/0ae17dffa45f621cb8061a8393e2dd225d0876a0)
records the original object:

- Revision: `0ae17dffa45f621cb8061a8393e2dd225d0876a0`.
- Repository path: `data/cc_news.tar.gz`.
- Published bytes: `845131146`.
- Published SHA256: `1aaf8e5af33e3a73472b58afba48c6a839ebc2dd190c4e0754fc00f8899a9cec`.
- Former loader URL: `https://storage.googleapis.com/huggingface-nlp/datasets/cc_news/cc_news.tar.gz`.

The immutable repository [file page](https://huggingface.co/datasets/vblagoje/cc_news/blob/0ae17dffa45f621cb8061a8393e2dd225d0876a0/data/cc_news.tar.gz)
could not be opened by the web tool; current binary availability was not established.
The hosting commit itself supplies the object digest and size.
The [January 2024 conversion](https://huggingface.co/datasets/vblagoje/cc_news/commit/81eb2ce0d2a9dad6ad16b68ef750ec290880fa36)
replaced this tarball with five `plain_text/train-0000N-of-00005.parquet` files, for N=0…4.
It reports a combined download size of 1,122,805,586 bytes.

The upstream loader maps raw `maintext`→`text`, `source_domain`→`domain`, and `date_publish`→`date`.
It also carries `title`, `description`, `url`, and `image_url`; it strips strings and converts nulls to empty strings.
Current Phase 4 intake expects exported `text`, `domain`, and `date` fields.
The tarball therefore needs an audited export step preserving member identity, original values, and declared transformations.
The converted Parquet release needs its own pinned derivation evidence if used.
Neither schema conversion nor the advertised date span establishes complete monthly coverage.
The [maintainer card](https://huggingface.co/datasets/vblagoje/cc_news) supplies the population description and marks licensing unresolved.

## Connector feasibility and executable next step

The installed GitHub fetch capability supports approved GitHub URLs and UTF-8 responses.
Its contract rejects external hosts, oversized responses, and non-UTF-8 binary downloads.
It legitimately retrieved the author code, checksum registry, and tree metadata above.
It cannot turn external Drive or GCS links into authorized archive imports.
The installed HF inspection connector returned repository metadata, not a binary download channel.
No denied-host binary request, mirror substitution, browser session, or external computation was attempted.

The next data action is authorized transfer of the exact originals into private local input storage.
For Civil, this verification is executable after setting `CIVIL_ARCHIVE` to the supplied local ZIP:

```bash
python - "$CIVIL_ARCHIVE" <<'PY'
import hashlib, pathlib, sys, zipfile
p = pathlib.Path(sys.argv[1])
assert p.stat().st_size == 448174578, 'Wrong archive size'
h = hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
assert h.hexdigest() == '89a60573acbbfaf189b2c77278115eefcbc1654dbd8235dd8fbecc1b83f1585c'
with zipfile.ZipFile(p) as z:
    assert z.namelist().count('civil_comments.csv') == 1
print('Expected archive bytes and member name verified; export replay remains required.')
PY
```

This verifies the expected bytes, not acquisition authority or downstream correctness.
The subsequent Civil step is a versioned exporter, followed by independent export replay and ordinary acceptance.

For WCEP, after transfer and review, the existing parser can run locally:

```bash
python empirical_execution/phase5/replication.py wcep \
  "$WCEP_INPUT/train.jsonl.gz" "$WCEP_INPUT/val.jsonl.gz" "$WCEP_INPUT/test.jsonl.gz" \
  --out "$WCEP_OUTPUT" --snapshot-revision "$REVIEWED_WCEP_RELEASE" \
  --collection-scope all_official_splits
```

The variables denote reviewed local paths and release evidence, not values generated by this review.
The parser deliberately leaves authenticity and complete-release verification false.
Original-file review, actual encoder replay, and genuine human calibration remain prerequisites for empirical activation.
