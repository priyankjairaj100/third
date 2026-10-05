# Local inputs and preparation

This guide prepares authentic inputs for the frozen empirical program.
Run commands from the repository root in Bash on your local machine.
The examples use the private directory `local_workspace/`.
Use a new attempt directory after every failure or configuration change.
Never overwrite a previous attempt or edit frozen Phases 3–10.

Read `EXECUTION.md` for orchestration and the empirical-program document for the scientific schedule.
Preparation success is not a primary result.
The first real semantic runtime and all original corpora still require local qualification.

## 1. Directory and environment contract

Use these paths consistently:

| Path | Contents |
| --- | --- |
| `local_workspace/inputs/` | Original archives, original XML, acquisition evidence, source specifications, coverage evidence |
| `local_workspace/models/{e5,mpnet}/{model,tokenizer}/` | Regular local files from the reviewed immutable publisher revision |
| `local_workspace/models/{e5,mpnet}/assets.json` | Generated inventory, outside both inventoried directories |
| `local_workspace/prepared/DATASET/a01/` | Export, adapter, complete prepared population, E5 and MPNet caches |
| `local_workspace/calibration/DATASET/ENCODER/a01/` | Accepted dossier, blank packs, genuine responses, threshold and validation |
| `local_workspace/reviews/` | Acquisition review, evidence review, actual reviewer records |
| `local_workspace/reports/` | Environment observations and later sanitized result exports |

Keep all raw corpora, full caches, human identities, and unsanitized outputs outside tracked Git files.
In particular, never publish CC-News or WCEP article bodies.
The private directory must be ignored before creating these artifacts.

The saved engineering environment used Python 3.12.14 and the versions in `empirical_execution/phase4/requirements.txt`.
That file includes the earlier NumPy, SciPy, scikit-learn, joblib, threadpoolctl, and idna pins.
These pins describe an observed engineering environment.
They are not a tested transformer environment for your operating system.

Install a compatible local environment before importing model assets.
Retain the resolved installation lock and installation logs locally.
Required encoder packages are NumPy, torch, transformers, tokenizers, and safetensors.
The selected tokenizer may need another package, such as sentencepiece.
Faiss is needed for relevant ANN or refit branches, not for the exhaustive reference graph.
No semantic package versions are invented by this handoff.

Record installed versions with:

```bash
python local_run/templates/prepare_runtime.py \
  --out local_workspace/reports/environment-a01
```

The script exits 2 when an encoder package is missing.
It then records the observation without writing a usable encoder-version file.
Installed package metadata alone does not qualify imports, numerical behavior, or model execution.
After completing installation, create a new environment attempt and use its `encoder_versions.json`.

**The frozen encoder runs on CPU in FP32.**
It uses eager attention, deterministic algorithms, disabled MKLDNN, and fixed document-local padding.
Start with batch size 8 and one thread, as shown below.
Model encoding on CUDA, MPS, mixed precision, ONNX, or a sentence-transformers wrapper changes this contract.
Do not silently substitute those routes because a GPU is available.
A faster route needs a new version, numerical qualification, and an explicit amendment before primary use.

Graph construction and calibration use the frozen CPU FP64 scorer.
Exact ridge auditing uses rational arithmetic.
The native helper is platform-specific to the documented LP64 Linux GMP ABI.
Do not run its headerless build recipe on another platform without review.
Read `empirical_execution/ccu/native/README.txt` for its self-test and Python fallback.
Backend changes require new resource observations even when represented states agree.

Total disk and memory requirements have not been measured on the authentic program.
Allow for originals, expanded lineage, both encoder caches, replay outputs, graphs, failed attempts, and result artifacts.
Full source/cache acceptance re-encodes the complete prepared population.
It cannot be replaced by checking a handful of rows.
Exhaustive calibration and leakage guards can dominate work before the small selected panels are constructed.

## 2. Obtain the exact original inputs

The identifiers below are frozen study inputs or previously reviewed candidates.
They are not a statement that every download endpoint remains available.
Keep acquisition URLs, retrieval dates, original bytes, byte counts, hashes, and applicable terms locally.
Do not substitute a current release, connector preview, or mirror without a documented amendment.

### Civil Comments

Use the base `CivilComments` configuration corresponding to TFDS 1.2.4.
Download the [official archive](https://storage.googleapis.com/jigsaw-unintended-bias-in-toxicity-classification/civil_comments_v1.2.zip).
Save it as `local_workspace/inputs/civil_comments/civil_comments_v1.2.zip`.

- Exact bytes: `448174578`.
- SHA256: `89a60573acbbfaf189b2c77278115eefcbc1654dbd8235dd8fbecc1b83f1585c`.
- Required member: `civil_comments.csv`.
- Expected base export counts: 1,804,874 train, 97,320 validation, and 97,320 test.

The [official catalog](https://www.tensorflow.org/datasets/catalog/civil_comments) identifies the dataset and underlying comments as CC0.
It also explains the fractional labels and absence of user IDs.
The source unit is publication plus article, not author identity.
Do not use the supervised text/label-only view.
Required metadata include `id`, `publication_id`, `article_id`, `parent_id`, `created_date`, and all seven base fractions.
The archive's `split` remains original lineage; our source split defines study partitions.
The export retains raw values and conversion evidence.

### AskUbuntu and English Stack Exchange

Use the **corrected April 2024** company snapshot for each site.
Keep each original site archive and its extracted `Posts.xml` and `PostLinks.xml`.
Place the XML files under `local_workspace/inputs/askubuntu/` and `local_workspace/inputs/english_stackexchange/`.

The project has not established an authorized historical download route or published archive digest.
The local LLM must resolve this factual input gap and record the actual route and hashes.
Do not write a plausible archive name or fill a digest with placeholder characters.
The [release catalog](https://meta.stackexchange.com/questions/224873/all-stack-exchange-data-dump-releases)
and [company correction notice](https://meta.stackexchange.com/questions/398279/shifting-the-data-dump-schedule-a-proposal/398606)
identify the required corrected release.
Current account download access does not prove access to that historical release.
If it cannot be obtained, mark the affected arms unavailable and report the limitation.

Retain original question IDs, creation dates, owner IDs, tags, title, rendered body, and available attribution/license fields.
Preserve all native duplicate links with `LinkTypeId=3`, including endpoints outside the retained question range.
The parser keeps questions dated 2018-05-02 through 2023-12-31.
It never concatenates answers.
The source unit is site plus the observed owner ID.
Missing or nonpositive owners remain unknown singletons, excluded from genuine source withdrawal sampling.
The [official license page](https://stackoverflow.com/help/licensing) explains the applicable CC BY-SA dates.
Retain attribution and original revision/license metadata.

### CC-News

Use `vblagoje/cc_news`, original tarball revision `0ae17dffa45f621cb8061a8393e2dd225d0876a0`.
The [immutable hosting commit](https://huggingface.co/datasets/vblagoje/cc_news/commit/0ae17dffa45f621cb8061a8393e2dd225d0876a0)
records `data/cc_news.tar.gz`.
Save it as `local_workspace/inputs/cc_news/cc_news.tar.gz`.

- Exact bytes: `845131146`.
- SHA256: `1aaf8e5af33e3a73472b58afba48c6a839ebc2dd190c4e0754fc00f8899a9cec`.
- Expected export count: `708241` article instances.

The newer Parquet conversion is not a drop-in replacement for the pinned tarball export.
The [maintainer card](https://huggingface.co/datasets/vblagoje/cc_news) marks the license as unknown.
Keep article bodies private.
Retain URL, declared domain, publication date, title, body, and archive/member identity.
Article identity is not a deduplicated URL.

Supply an actual [Public Suffix List](https://publicsuffix.org/list/) file and retain its reviewed revision and SHA256.
The parser includes ICANN and PRIVATE sections by default.
Registrable domain grouping requires agreement between the URL and declared host.
It does not establish corporate ownership.

Copy `templates/prepare_news_coverage.template.json` to `local_workspace/inputs/news_coverage.json`.
Bind its input checksum to the exported `records.jsonl`.
Supply actual local acquisition/coverage evidence and all 36 month statuses.
The template deliberately declares every month unavailable until reviewed.
Counts or observed date endpoints cannot establish extraction completeness.
Calibration uses source-held-out domains from 2017; analysis uses training domains from 2018–2019.
Do not run the generic preparation command for News.

### WCEP replication

Use the author's extracted article-instance release described at repository revision
`8aa8418d55a4e6710277bed18bfc5349b50a5d96`.
Read the [author repository](https://github.com/complementizer/wcep-mds-dataset/tree/8aa8418d55a4e6710277bed18bfc5349b50a5d96)
and [getting-started notebook](https://github.com/complementizer/wcep-mds-dataset/blob/8aa8418d55a4e6710277bed18bfc5349b50a5d96/wcep_getting_started.ipynb).
The notebook identifies these original files:

| File | Author link |
| --- | --- |
| `train.jsonl.gz` | https://drive.google.com/uc?id=1kUjSRXzKnTYdJ732BkKVLg3CFxDKo25u |
| `val.jsonl.gz` | https://drive.google.com/uc?id=1_kHTZ32jazTbXaFRg0vBeIsVcpI7CTmy |
| `test.jsonl.gz` | https://drive.google.com/uc?id=1qsd5pOCpeSXsaqNobXCrcAzhcjtG1wA1 |

Store all three under `local_workspace/inputs/wcep100/`.
No published byte digests were verified in the checkpoint.
Review and hash the actual files; a Drive locator is not a content identity.
The repository's MIT software license does not grant blanket rights to publisher article text.
Keep article instances, native event IDs, article ordinals, dates, and original URLs.
Do not truncate events to 100 articles or use summaries as encoded inputs.
The registered panel keeps whole events until the 25,000-article target is met.
It is record-only and inherits the accepted News encoder and threshold.
WCEP event membership is not duplicate gold.

## 3. Obtain and qualify both encoders

| Encoder | Reviewed candidate revision | Published safetensors SHA256 |
| --- | --- | --- |
| `intfloat/multilingual-e5-base` | `d128750597153bb5987e10b1c3493a34e5a4502a` | `a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7` |
| `sentence-transformers/all-mpnet-base-v2` | `e8c3b32edf5434bc2275fc9bab85f82640a19130` | `78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658` |

Use the publisher repositories, not similarly named repackages.
Retain the complete configuration, tokenizer files, and applicable license at the same reviewed revision.
The pinned MPNet card declares Apache-2.0.
Re-read the local E5 license bytes during acquisition; this handoff does not invent an unverified license claim.
Place model and tokenizer files in their respective regular-file directories.
Do not use cache symlinks, pickle fallback, custom Python code, or remote-code loading.
Avoid downloading alternate ONNX, OpenVINO, or pickle weights without a separate need.

Create the actual E5 inventory:

```bash
python empirical_execution/phase4/embeddings.py inventory \
  --model-dir local_workspace/models/e5/model \
  --tokenizer-dir local_workspace/models/e5/tokenizer \
  --encoder-id intfloat/multilingual-e5-base \
  --encoder-revision d128750597153bb5987e10b1c3493a34e5a4502a \
  --tokenizer-revision d128750597153bb5987e10b1c3493a34e5a4502a \
  --versions local_workspace/reports/environment-a01/encoder_versions.json \
  --out local_workspace/models/e5/assets.json

python empirical_execution/phase4/embeddings.py inventory \
  --model-dir local_workspace/models/mpnet/model \
  --tokenizer-dir local_workspace/models/mpnet/tokenizer \
  --encoder-id sentence-transformers/all-mpnet-base-v2 \
  --encoder-revision e8c3b32edf5434bc2275fc9bab85f82640a19130 \
  --tokenizer-revision e8c3b32edf5434bc2275fc9bab85f82640a19130 \
  --versions local_workspace/reports/environment-a01/encoder_versions.json \
  --out local_workspace/models/mpnet/assets.json
```

Use the environment attempt that actually passed package presence checks.
These revision strings must match the downloaded bytes and acquisition evidence.
Inventory accepts declarations; it does not prove publisher provenance.

Before full encoding, qualify the actual models on genuine development texts.
Check repeated and interleaved document execution, stable FP32 vectors, padding, and complete long-document chunk logs.
Retain code, source IDs, hashes, runtime, and measured costs for that development check.
Do not label a sliced input file as the complete primary prepared population.
The later acceptance gate still requires full-cache replay.

The frozen representation uses every consecutive chunk of at most 256 content tokens.
E5 prepends literal `query:` tokens; MPNet does not.
Pooling includes attended model tokens, then weights chunks by content-token count.
Documents are normalized and stored as FP32.
Default model-card truncation is not this full-document representation.

## 4. Export, parse, and prepare

These commands use attempt `a01`.
Only parent directories may exist before a stage; each stage output itself must be new.

### Civil

```bash
python empirical_execution/phase8/archive_export.py export civil_comments \
  local_workspace/inputs/civil_comments/civil_comments_v1.2.zip \
  --out local_workspace/prepared/civil_comments/a01/export
python empirical_execution/phase8/archive_export.py replay \
  local_workspace/prepared/civil_comments/a01/export \
  local_workspace/inputs/civil_comments/civil_comments_v1.2.zip \
  --out local_workspace/prepared/civil_comments/a01/archive_replay
python empirical_execution/phase4/adapters.py civil \
  local_workspace/prepared/civil_comments/a01/export/records.jsonl \
  local_workspace/prepared/civil_comments/a01/adapter \
  --input-format jsonl --input-schema tfds_1_2_4
python empirical_execution/phase3/run_preparation.py prepare \
  --dataset civil_comments \
  --records local_workspace/prepared/civil_comments/a01/adapter/records.jsonl \
  --target 10000 --out local_workspace/prepared/civil_comments/a01/population
```

`--target 10000` records a provisional panel target; it does not truncate the complete prepared population.
Later registered groups derive their own whole-source panels.

### Both Stack sites

Run the block separately for `askubuntu` and `english_stackexchange`.
Set each archive hash and reviewed corrected release from actual acquisition evidence.

```bash
DATASET=askubuntu
STACK_ARCHIVE_SHA256=REQUIRED_ACTUAL_CORRECTED_ARCHIVE_SHA256
STACK_RELEASE=REQUIRED_REVIEWED_CORRECTED_APRIL_2024_RELEASE_ID
python empirical_execution/phase4/adapters.py stack \
  "local_workspace/inputs/$DATASET/Posts.xml" \
  "local_workspace/inputs/$DATASET/PostLinks.xml" \
  "local_workspace/prepared/$DATASET/a01/adapter" \
  --dataset-id "$DATASET" --archive-sha256 "$STACK_ARCHIVE_SHA256" \
  --archive-release "$STACK_RELEASE"
python empirical_execution/phase3/run_preparation.py prepare \
  --dataset "$DATASET" \
  --records "local_workspace/prepared/$DATASET/a01/adapter/records.jsonl" \
  --duplicate-links "local_workspace/prepared/$DATASET/a01/adapter/duplicate_links.json" \
  --target 10000 --out "local_workspace/prepared/$DATASET/a01/population"
```

Replace every `REQUIRED_...` value before running.
The parser cannot verify that an arbitrary release string describes the authentic corrected archive.

### News

```bash
python empirical_execution/phase8/archive_export.py export cc_news \
  local_workspace/inputs/cc_news/cc_news.tar.gz \
  --out local_workspace/prepared/cc_news/a01/export
python empirical_execution/phase8/archive_export.py replay \
  local_workspace/prepared/cc_news/a01/export \
  local_workspace/inputs/cc_news/cc_news.tar.gz \
  --out local_workspace/prepared/cc_news/a01/archive_replay
PSL_SHA256=REQUIRED_ACTUAL_PINNED_PSL_SHA256
python empirical_execution/phase4/adapters.py news \
  local_workspace/prepared/cc_news/a01/export/records.jsonl \
  local_workspace/prepared/cc_news/a01/adapter \
  --input-format jsonl --input-schema raw \
  --psl-path local_workspace/inputs/public_suffix_list.dat --psl-sha256 "$PSL_SHA256"
python empirical_execution/phase4/news_calendar.py \
  local_workspace/prepared/cc_news/a01/adapter \
  local_workspace/inputs/news_coverage.json \
  local_workspace/prepared/cc_news/a01/population
```

Complete and review the coverage file before its preparation command.
Unknown coverage remains unknown; do not infer it from exported counts.

### Encode the complete prepared populations

Run E5 for each of `civil_comments`, `askubuntu`, `english_stackexchange`, and `cc_news`.
Civil alone also requires MPNet for the registered curator/learner cross.
For Civil, both encoders must use the identical complete ordered population.
Do not create unregistered MPNet caches for the other corpora.

```bash
DATASET=civil_comments
ENCODER=e5
python empirical_execution/phase4/embeddings.py encode \
  --records "local_workspace/prepared/$DATASET/a01/population/records.jsonl" \
  --preparation-audit "local_workspace/prepared/$DATASET/a01/population/audit.json" \
  --asset-manifest "local_workspace/models/$ENCODER/assets.json" \
  --model-dir "local_workspace/models/$ENCODER/model" \
  --tokenizer-dir "local_workspace/models/$ENCODER/tokenizer" \
  --batch-size 8 --threads 1 --evidence-role confirmatory_calibration \
  --out "local_workspace/prepared/$DATASET/a01/${ENCODER}_cache"
```

The evidence-role argument declares intended use; it does not qualify authenticity or semantic quality.
`confirmatory_study_ready` remains false until all required gates pass.

### WCEP's separate route

```bash
WCEP_RELEASE=REQUIRED_REVIEWED_RELEASE_ID_BOUND_TO_THREE_ACTUAL_FILE_HASHES
python empirical_execution/phase5/replication.py wcep \
  local_workspace/inputs/wcep100/train.jsonl.gz \
  local_workspace/inputs/wcep100/val.jsonl.gz \
  local_workspace/inputs/wcep100/test.jsonl.gz \
  --snapshot-revision "$WCEP_RELEASE" --collection-scope all_official_splits \
  --out local_workspace/prepared/wcep100/a01/adapter
python empirical_execution/phase5/replication.py panel \
  local_workspace/prepared/wcep100/a01/adapter --target 25000 \
  --out local_workspace/prepared/wcep100/a01/panel
ENCODER=e5
python empirical_execution/phase5/replication_encoding.py encode \
  --adapter-dir local_workspace/prepared/wcep100/a01/adapter \
  --panel-dir local_workspace/prepared/wcep100/a01/panel \
  --asset-manifest "local_workspace/models/$ENCODER/assets.json" \
  --model-dir "local_workspace/models/$ENCODER/model" \
  --tokenizer-dir "local_workspace/models/$ENCODER/tokenizer" \
  --batch-size 8 --threads 1 \
  --out "local_workspace/prepared/wcep100/a01/${ENCODER}_cache"
```

Use the identical accepted News E5 encoder/runtime inventory for the registered WCEP configuration.
Read `phase5/REPLICATION_ENCODING_README.md` before inheritance.
The later WCEP candidate needs the accepted News bundle and its exact registered group.
WCEP does not create a fresh semantic threshold or classification labels.

## 5. Source specifications and genuine calibration

Copy the matching `prepare_*_source.template.json` into `local_workspace/inputs/`.
Name it `civil_source.json`, `askubuntu_source.json`, `english_stackexchange_source.json`, `news_source.json`, or `wcep_source.json`.
Paths inside the templates resolve relative to that directory.
Keep the resulting reviewed absolute paths stable.
Update attempt paths when a preparation stage uses another attempt ID.
Replace `REQUIRED_...` fields using actual evidence only.
The templates do not constitute accepted input dossiers.

Run this sequence for exactly five registered corpus/encoder pairs:
Civil/E5, Civil/MPNet, AskUbuntu/E5, English Stack Exchange/E5, and News/E5.
Do not create MPNet calibration packs for the other corpora.

```bash
DATASET=civil_comments
ENCODER=e5
SOURCE_SPEC=local_workspace/inputs/civil_source.json
CAL_ROOT="local_workspace/calibration/$DATASET/$ENCODER/a01"
python empirical_execution/phase8/dossiers.py calibration \
  --source-spec "$SOURCE_SPEC" --encoder "$ENCODER" --out "$CAL_ROOT/dossier"
python empirical_execution/phase7/prepare_calibration.py \
  --dataset "$DATASET" --encoder "$ENCODER" \
  --dossier "$CAL_ROOT/dossier/dossier.json" --out "$CAL_ROOT/selection"
```

This performs full replay before creating accepted blank forms.
Give raters only their assigned text pairs and the fixed annotation guide.
Never give them private sampling manifests, scores, desired outcomes, task labels, or another rater's judgments.
The coordinator obtains three independent people per pair.
An LLM must not fill human fields, invent rater IDs, or declare collection complete.
Human consent, sensitive-content handling, and compensation arrangements require real implementation and documentation.

Use `empirical_execution/phase3/CALIBRATION_ANNOTATION_GUIDE.txt` unchanged.
The six categories are exact copy, substantially same meaning, overlapping information, merely related, unrelated, and uncertain.
At least two positive judgments make a pair positive; uncertain is nonpositive.
Selection samples up to 30 pairs in each of 20 score bins.
There can therefore be up to 600 pairs and 1,800 selection judgments per corpus/encoder.

The selection output nests its files inside `selection/selection/`.
Copy its generated `responses.template.json` to a private responses file before genuine collection.
After all genuine selection responses exist:

```bash
python empirical_execution/phase4/calibrate_cache.py threshold \
  --manifest "$CAL_ROOT/selection/selection/private_sampling_manifest.json" \
  --responses "$CAL_ROOT/selection/actual_responses.json" \
  --out "$CAL_ROOT/threshold.json"
python empirical_execution/phase4/calibrate_cache.py validation \
  --records "local_workspace/prepared/$DATASET/a01/population/records.jsonl" \
  --preparation-audit "local_workspace/prepared/$DATASET/a01/population/audit.json" \
  --vectors "local_workspace/prepared/$DATASET/a01/${ENCODER}_cache/vectors.npy" \
  --cache-manifest "local_workspace/prepared/$DATASET/a01/${ENCODER}_cache/cache_manifest.json" \
  --asset-manifest "local_workspace/models/$ENCODER/assets.json" \
  --lock "$CAL_ROOT/threshold.json" --out "$CAL_ROOT/validation"
```

Validation samples up to 200 pairs, with three fresh independent judgments each.
Previously selected pairs remain eligible but require fresh blinded assignments.
After genuine validation collection:

```bash
python empirical_execution/phase4/calibrate_cache.py quality \
  --manifest "$CAL_ROOT/validation/private_sampling_manifest.json" \
  --responses "$CAL_ROOT/validation/actual_responses.json" \
  --lock "$CAL_ROOT/threshold.json" --out "$CAL_ROOT/quality.json"
```

Selection needs weighted precision at least 0.95 and nonzero support.
Validation needs the exact one-sided 95% lower precision bound at least 0.90.
If selection fails, the recorded 0.995 value is diagnostic, not a passed threshold.
If validation fails, stop that primary configuration and report failure without threshold retuning.
Civil/MPNet has its own threshold and responses.
The two Civil/E5 threshold-sensitivity variants require additional independent validation packs, as specified in the empirical program.
Only E5 defines the shared training-only semantic leakage guard.
The dossier candidate stage reconstructs that guard and the complete whole-source panels.
Do not apply an additional MPNet guard or manually remove difficult evaluation examples.

The later program also requires genuine admission and contextual human audits.
Their assignments come from actual admitted records and their graphs, not these calibration packs.
Use the main empirical program and `phase9/GRAPH_DOSSIER.md` for their separate sampling and review.

## 6. Handoff from preparation to execution

Give the local execution agent these actual artifacts:

1. Acquisition evidence, original-file hashes, applicable terms, and remaining coverage limitations.
2. Successful original export/replay, adapter, and complete preparation artifacts.
3. Both real model inventories, exact runtime lock, required complete caches, and real-model qualification observations.
4. Accepted calibration dossiers, genuine responses, threshold locks, and independent validation reports.
5. Stable source specifications and the required News/WCEP inheritance evidence.
6. Genuine development, resource, and review evidence required by the frozen acceptance contracts.

Do not upload these private artifacts indiscriminately.
Use the reporting workflow to return sanitized summaries, failures, hashes, and artifact inventories.
Accountable review of supplied evidence remains required; a checksum or `approved=true` field cannot replace it.

Historical acquisition facts and conversion details remain in `phase7/ACQUISITION_REVIEW.md` and `phase8/ARCHIVE_EXPORT.md`.
Current commands above were checked against the frozen CLI/source interfaces.
No original archive download, transformer run, or human collection was completed while preparing this guide.
