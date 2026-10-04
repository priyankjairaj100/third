OFFLINE CORPUS ADAPTERS — VERSION 1
4 October 2026

WHAT IS COMPLETED
adapters.py implements streaming original-schema readers for source-rich Civil
Comments exports, Stack Exchange Posts.xml/PostLinks.xml, and extracted CC-News
CSV/JSONL. It makes no network requests and never creates corpus records, labels,
source ownership, or human ratings. The prior semantic experiments remain blocked
by missing authentic corpora, encoder assets and human annotations.

Run from repository root:

  python3 empirical_execution/phase4/adapters.py civil civil.jsonl output/civil
  python3 empirical_execution/phase4/adapters.py civil civil.csv output/civil_csv --input-format csv
  python3 empirical_execution/phase4/adapters.py stack Posts.xml PostLinks.xml output/askubuntu --dataset-id askubuntu --archive-sha256 ACTUAL_ARCHIVE_SHA256
  python3 empirical_execution/phase4/adapters.py news news.jsonl output/news --psl-path public_suffix_list.dat --psl-sha256 ACTUAL_PSL_SHA256
  python3 empirical_execution/phase4/check_adapters.py

An output directory must not already exist. Exceptions leave a failed audit and
partial output explicitly marked incomplete; downstream code must reject it.
Dependencies are Python standard library, idna, and NumPy via the existing phase3
source helper. The idna version, adapter hash and source-helper hash are recorded.

FILES AND STREAMING
Every run writes records.jsonl, exclusions.jsonl, identity_index.sqlite3 and
audit.json. SQLite enforces unique natural IDs on disk with a 2 MiB page cache;
text working storage is one row/HTML parse at a time. PSL rule sets are resident.
Input hashing, XML DTD scanning, parsing and output verification require separate
linear passes, which must be charged. SQLite retained bytes and all output sizes
are recorded; those are disk bytes, not measured peak RSS.

Stack additionally writes duplicate_links.jsonl, duplicate_links.json (a streamed
JSON array directly accepted by phase3 run_preparation.py), and the original
duplicate-link row metadata. Link endpoints outside filtered questions are kept
and counted; this preserves unresolved-link evidence rather than silently hiding
it. Only LinkTypeId=3 is a duplicate relation. No answer is concatenated.

NATIVE IDENTITIES, SOURCES AND ORIGINAL FIELDS
Civil record IDs are str(original id), preserving the existing parent guard's
native lookup. Stack IDs are str(Id), within the separately named site dataset.
CC-News has no native unique record ID in the declared export schema: identity is
immutable input file SHA256 plus physical row ordinal. Repeated URLs and repeated
texts remain separate natural instances. Reexporting/reordering News makes a new
identity-bound snapshot and must not be mixed with the prior one.

source_unit_id/source_kind come from frozen phase3.panels.source_unit. Civil uses
[publication_id,article_id]; Stack [declared archive site,OwnerUserId]; News uses
the pinned-PSL registrable domain only when URL and declared host agree exactly
after canonicalization. No source_id alternative is introduced.

Original fields are retained with explicit schema typing: CSV is text-only, so
Civil article/parent integer fields and toxicity receive canonical schema types;
Stack owner receives its integer type. Every altered lexical value is preserved
under adapter_input.raw_lexemes_for_typed_fields. This prevents CSV-versus-JSONL
encodings from splitting one source. Unknown Civil article sentinels (nonpositive,
empty or null) become a missing value with the raw value preserved. Stack absent
or nonpositive owners remain distinct unknown singletons. Malformed integer owner
strings are counted exclusions, never genuine sources. Site is explicitly added
archive-route context; absent owner metadata is explicitly declared, not invented.
Original revision, attribution and license fields in Posts.xml are copied. They
do not establish exclusive authorship or automatically verify the applicable license.

CIVIL
The source-rich schema requires id, publication_id, article_id, parent_id,
created_date, text and toxicity, as in TFDS CivilComments 1.2.4. Own comment text
only is used; parent_text is never concatenated. Original fractional toxicity
becomes labels=[toxicity] and label=toxicity. No binary training label is created.
NFC and whitespace collapse preserve wording. Invalid fractions fail; blank text
is counted/excluded. Dates are syntax-checked when available, with missing counts;
no new Civil date window is imposed. A Kaggle comment_text/target export requires
an explicitly reviewed conversion rather than guessed aliases.

STACK
Only PostTypeId=1 and recorded creation dates from 2018-05-02 through 2023-12-31
inclusive are retained. The recorded date is used without inventing a timezone.
Question-local tags are retained, with no frequency/vocabulary selection here.
Title precedes parsed rendered Body HTML. The parser keeps inline wording, code,
quotations, numbers, negation, image alt text and author link destinations; block
boundaries become whitespace. Only explicitly marked question-status,
question-status-container, post-notice, or special-status containers are removed.
Generic blockquotes and author prose are preserved. Unmarked legacy notice-like
phrases are flagged for archive inspection, not automatically deleted as prose.
HTML repairs and removed notices are counted. DTD/entity declarations are refused;
normal XML/HTML character references are decoded. This parser needs validation on
the actual corrected April 2024 archive before declaring boilerplate coverage.
Declared archive release/hash are consistency metadata, not acquisition proof.

NEWS
Title precedes article body. One exact normalized first nonempty body line is
removed only if it equals the normalized title; a longer line beginning with the
same words, case differences, or later occurrences remain. No description/summary
is appended. Blank total article or invalid/missing date is counted/excluded.
The maintainer's 2017–2019 window is enforced. 2017 records are marked
calibration_candidate; 2018–2019 are analysis_candidate. These are date roles,
not finished source-disjoint partitions. The downstream prospective News policy
must still keep calibration domains out of the analysis source universe and use
complete calendar months for the time-window sensitivity. Month counts are saved.
There are no News task labels: labels=[]. Article text is not redistributed in the
GitHub backup or deliverable package. Local output is intentionally not an export
license. Existing preview News checks create temporary outputs and delete them.

OFFLINE PSL
Supply an actual pinned UTF-8 Public Suffix List file; SHA256 must match. ICANN and
PRIVATE section markers are required. PRIVATE rules are included by default;
--icann-only is an explicitly separate setting. Exact, wildcard and exception
rules follow the official longest-match/exception semantics. Hostnames use the
recorded idna package version, UTS46 mapping and STD3 rules, lower-case Punycode,
and removal of one trailing root dot. IPs, malformed/conflicting hosts, public
suffixes alone and unknown suffixes remain unknown singletons. The default-star
PSL fallback is intentionally not accepted as evidence of a known source.
PSL grouping is a domain grouping, never a corporate ownership assertion.

VERSIONED INTAKE CORRECTION
verify_adapter_output(directory) checks version-bound hashes, record construction,
original labels/tags and source derivation by streaming the prepared records.
It implements the News title+body consistency rule, correcting the legacy
phase3/intake_v2.py expectation that prepared text equal the body alone. Frozen
phase3 code is unchanged. Do not use intake_v2's News equality as a primary gate.
The new verifier is only a construction-consistency gate; it is not the complete
study intake, a license review, an originality test, or a claim of authenticity.
Both input_authenticity_verified and confirmatory_study_ready remain false.

WHAT HAS ACTUALLY BEEN CHECKED
check_adapters.py covers tiny software-only CSV/XML/PSL fixtures, endpoints,
Unicode, native parent linkage, missing/community owner distinctions, malformed
owners, notice/code/link parsing, duplicate links, exact repeated-title behavior,
URL-instance identity, date roles, raw-schema refusal and overwrite refusal.
It also parses the 100 existing natural Civil preview rows and, when locally
available, the 90 existing natural News preview rows. A tiny PSL fixture in that
News plumbing check is not native-source evidence or a real PSL validation.
News may be unavailable after checkout because its bodies are not redistributed;
the audit states that limitation instead of inventing replacements. No original
Stack archive or full official PSL was available in this workspace for validation.
No synthetic empirical dataset is used; fixture tests are not experiment results.

PRIMARY SCHEMA SOURCES, READ 4 OCTOBER 2026
https://www.tensorflow.org/datasets/catalog/civil_comments
  CivilComments 1.2.4 lists article/publication/parent/date fields and label fractions.
https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
  Posts rendered HTML, nullable owner, original tags/revision/license fields;
  PostLinks type 3 denotes a duplicate, type 1 denotes a link.
https://huggingface.co/datasets/vblagoje/cc_news
  Maintainer describes 708,241 English articles, 2017–2019, title/text/domain/date/URL.
https://github.com/publicsuffix/list/wiki/Format
  PSL rule syntax, section divisions, wildcard/exception and registrable-domain rules.
