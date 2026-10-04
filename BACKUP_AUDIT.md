# Project backup audit

Audit date: 4 October 2026. Scope: the counterfactual semantic-curation project in this workspace, for the user-authorized repository `priyankjairaj100/third`. This audit inspected project files only. It did not read credentials, system configuration, other conversations, or Library files; it did not push anything.

## Decision

Back up the authored research, implementation, delivered reports, experiment outputs, independent audits, public Civil Comments engineering fixture, and context/handoff documentation. Preserve superseded scientific results as explicitly historical evidence. Exclude the locally cached CC-News article bodies and their connector responses, runtime caches, duplicated extraction trees, and rendering/build byproducts. Rescue the unique authored files currently stored under `tmp` before excluding that directory.

This is a completeness and accidental-disclosure check, not a declaration that the primary experiment is complete. At this initial inspection, `empirical_execution/CURRENT_STATUS.json` recorded `phase3_preparation_complete_primary_semantic_study_blocked`. The later Phase 4 supplement below supersedes that status reference.

## Include

- All top-level project `.md`, `.py`, and verification `.json` files. The round 1/2/3 theory notes and checks document the development history and should not be silently replaced by the newest summary.
- All authored modules and scripts in `empirical_execution/`, including `ccu/`, its native C++ source and build instructions, phase 2, and phase 3.
- All current and historical result directories: `results/`, `memory_results/`, `canonical_results/`, `phase2/results/`, and the current phase 3 results. Preserve `archive/pilot_v1_variable_curation/` because the engineering changelog explicitly records why it was superseded.
- All provenance notes, design locks, manifests, checksums, independent reviews, theory addenda, annotation guidance, and blank response templates. File names beginning `private_sampling_manifest` mean blinded-annotation administration, not private credentials; these are engineering-only public-dataset examples and contain no genuine human ratings.
- `empirical_execution/data/civil_comments_engineering_preview.jsonl` and `civil_comments_preview_raw.json`, together with the acquisition/provenance notes. These are the same public Civil Comments preview material already included in delivered packages. The source-rich primary corpus is not present.
- All seven deliverable ZIPs in `output/`, the PDFs and their authored TeX, the empirical protocol and schemas, and the generated PNG/SVG figure under `empirical_execution/figures/`.
- New repository README, resumption instructions, file manifest, this audit, and backup status, once finalized.

The existing historical `STATUS.json` must remain visibly distinguished from the current `CURRENT_STATUS.json`; preserving history must not create a false current-status statement.

## Exclude or clearly archive

| Path/pattern | Recommendation | Reason |
| --- | --- | --- |
| `empirical_execution/data/cc_news*` | Exclude all 23 local files | These contain article bodies and connector responses deliberately omitted from earlier distributable packages. Keep metadata, acquisition notes, and body-free audit results. |
| `tmp/` | Exclude after rescuing the five authored files listed below | PDF rendering images, TeX build files, temporary workflow products, and an extracted release-check copy are reproducible intermediates. |
| `**/__pycache__/`, `*.pyc`, `*.pyo` | Exclude | Interpreter caches. |
| `missfont.log`, transient `*.aux`, `*.log`, `*.out`, `*.toc`, `*.fmt` | Exclude generated copies | Build byproducts. Do not broadly exclude authored JSON/TXT result logs. |
| `empirical_execution/phase3/results/calibration_engineering_pack_pre_scope_fix/` | Prefer a clearly labeled historical archive, or explicitly omit as superseded | The three files differ from the current pack. They are not the current assignments or evidence. A broad inclusion must not make their status ambiguous. |
| `empirical_execution/ccu/native/exact_chart` | Optional recorded platform artifact; source is mandatory | It is a 42 KB compiled Linux helper, rebuildable from `exact_chart.cpp`; document platform/dependencies if retained. |

No model weights, semantic embedding arrays, original source-rich corpora, completed human ratings, or GPU training products were found among the project assets inspected.

### Unique authored files under `tmp`

The following five files have no byte-identical copy outside `tmp`. Copy them into an explicitly historical source/build directory before excluding `tmp` (or preserve only these exact paths):

- `tmp/pdfs/build_empirical_pdf.py`
- `tmp/pdfs/package_empirical.py`
- `tmp/pdfs/rebuild_pdf.py`
- `tmp/pdfs/round1_theory.tex`
- `tmp/pdfs/round2_theory.tex`

The first three are authored build/package helpers containing the original workspace path. They are useful provenance but should be labeled legacy until paths are made portable. The last two are earlier theory drafts; the current theory is `output/pdf/counterfactual_curation_theory.tex`. A tiny `tmp/memory_theory_fragment_check/check.tex` is only a compile harness around the preserved addendum and can be discarded.

## Inspection results

- Inspected 291 candidate loose nonbinary files for common GitHub/Hugging Face/OpenAI token formats, private-key headers, signed URLs, quoted credential assignments, and private-IP patterns. No matches were found. This is a targeted scan, not a mathematical guarantee of absence.
- Repeated the token/key/URL scan inside all seven generated ZIPs. No matches were found.
- Inspected all seven ZIP inventories: none contains a CC-News data file.
- Searched candidate loose text and every ZIP member for a 120-byte article-body fingerprint from each sufficiently long local News article. No matching article excerpt was found outside the excluded News files. Matches of dataset names or schema field names alone were code/schema references, not article bodies.
- All seven ZIP CRC integrity checks passed.
- Confirmed existence and nonzero size of the current theory TeX/PDF, empirical protocol, study design, memory and canonical theory addenda, strict-mode summary, phase 2 summary and independent result audit, and phase 3 independent review, reviewed-code hash manifest, workflow audit, finite-population audit, and scoring audit.
- The current phase 3 blank assignment pack is preserved separately from its superseded pre-scope-fix predecessor. Neither contains completed human judgments.

## Resumption-critical boundaries

The continuation guide must say that engineering and software checks are complete, but the primary semantic study has not begun. Required inputs still include original source-rich corpus snapshots, pinned E5 embeddings or usable local model/tokenizer assets, and genuine independent blinded ratings. Do not present the lexical preview experiments, declared provenance fields, or blank annotation templates as semantic empirical confirmation.

The backup should preserve both exact-canonical strict-mode results and the phase 3 pair-local scoring amendment. Historical phase 2 BLAS-based experiments were intentionally left unchanged; their computational target must not be silently conflated with the new pair-local reference.

Any final repository verification should check the committed tree against its inclusion manifest and confirm the excluded News files are absent, including inside the included archives. This audit reports the inspected workspace before the publishing step; the publishing agent should record the final commit separately.

## Implemented preservation decisions

The publishing agent copied the five unique temporary sources byte-for-byte into `archive/theory_history/` and `tools/legacy_build/`; `archive/RESCUED_SOURCES.json` records original paths and checksums. The superseded calibration pack is retained in its clearly named original folder with `README_SUPERSEDED.txt`. It remains distinct from the current frozen pack. The compiled native helper is retained alongside its source and platform/build instructions. Git attributes disable automatic text normalization so checkpoint checksums survive cross-platform checkouts.

## Phase 4 preservation supplement — 4 October 2026

The checkpoint now includes the complete `empirical_execution/phase4/` authored code, instructions, engineering outputs, independently reviewed audits, blank admission assignments, failed convex run/source snapshot, and explicitly historical intermediate runs. The authoritative ridge run is `results/execution_engineering_final/`; the authoritative convex run is `results/convex_engineering_release_v2/`. Historical positive convex runs omitted the initial head artifact and are not the final replay target.

The 16,463,560-byte request trace exceeds the connected publishing API request limit when encoded. Its deterministic gzip archive preserves every byte; `large_result_archive.json` records the original and compressed hashes. `tools/restore_large_results.py` checks both and restores the original without replacing a differing file. Only the redundant uncompressed generated copy is excluded from Git. This is a storage-format change, not omitted experimental data.

`phase4/results/publication_content_audit.json` records the targeted credential-pattern and excluded-News-body fingerprint scan, including the decompressed archive. The new blank admission pack uses the already published natural Civil preview only; it contains no completed ratings or added News bodies. Transient subprocess software fixtures are outside the backed-up project artifact set.

No authentic primary corpora, semantic encoder assets, genuine human ratings, or complete primary systems/full-refit results are newly claimed. `CURRENT_STATUS.json`, `phase4/TODO.json`, and both root handoffs explicitly preserve those remaining dependencies and implementation gaps. The manifest generator now reads the current status instead of hard-coding the old Phase 3 stage.


## Phase 5 preservation supplement — 4 October 2026

The checkpoint includes compact payload and summary implementations, both
versioned worker/decoder paths, kernel capability attempts, pinned official
SemDeDup sources/configuration/license, NumPy reference evidence, convex and
boundary implementations, WCEP/News adapters, task/human/statistical analysis,
study registry, independent reviews and updated continuation context. Initial
failures and superseded runs remain in explicitly identified directories.

The authoritative isolated v1 result directory stores 3,888 per-process files
losslessly in `jobs.tar.gz`. `jobs_archive.json` binds each member's original
length/SHA256 plus the archive's hash. `tools/phase5_result_archive.py` verifies
or restores all members without overwriting a differing file. Only redundant
unpacked members are git-ignored. Temporary large service-state snapshots were
not retained; their hashes/byte accounting and exact generation code are saved,
while released heads and reports are all archived. This distinction is explicit
in result documentation and is not a claim to back up original corpora.

The original 213-row journal and reconciled 216-row ledger are both retained,
with exact per-job evidence for the three recovered rows. No outcomes were
rerun to replace the missing journal entries. Later numerical-worker failures
also retain their source snapshot and measured outputs.

`phase5/audit_publication.py` scans candidate public files and nested gzip/tar/
NPZ archive members for credential patterns and fingerprints from the excluded
local News bodies. Its latest inventory-bound output is
`phase5/results/publication_content_audit.json`; no News article body or real
human response has been added. Civil engineering data were already public in
this checkpoint history. Actual future rater identities require separate
publication review and must not be automatically uploaded with raw responses.

The Phase 5 completion ledger maps all 19 historical Phase 4 TODO items. It
explicitly retains missing authentic data/model/human inputs, real-backend and
primary activation acceptance, prospective recipe obligations and measurement
limits. The final GitHub tree/backup-manifest check establishes preservation,
not an ACL-ready empirical study.


## Phase 6 preservation checkpoint — 4 October 2026

The user requested a pause and a complete checkpoint for another chat.
No new scientific work was started after that request.
The already-running independent audit finished with 244 passed checks.
Phase 6 preserves all authored code, reports, recipe decisions, input requirements, and source-bound audit evidence.
The primary semantic study remains unstarted.

Verbose result directories are preserved as complete deterministic tar/gzip archives.
Each adjacent JSON manifest records every member, byte count, and SHA256 hash.
The archive tool verifies every saved member and refuses differing restoration targets.
Loose duplicate directories remain locally available but are excluded from Git.
No failed or interrupted result is replaced by a later successful run.
The archive inventory is recorded in phase6/results/archive_checkpoint_inventory.json.

The publication scan covers new project files and recursively decompressed archive members.
It checks excluded News-body fingerprints and common credential patterns.
Its report is phase6/results/publication_content_audit.json.
This targeted scan does not establish universal absence or corpus authenticity.
The final BACKUP_MANIFEST and remote tree are verified separately during publishing.

Real corpora, model files, human ratings, and primary execution remain absent.
See RESUME_CONTEXT.md and phase6/COMPLETION_LEDGER.md for continuation.


## Phase 7 preservation supplement — 4 October 2026

The checkpoint adds current command-line interfaces and the original input map.
It preserves both dispatcher wrapper versions and their separate checks.
The final wrapper reuses historical natural evidence with explicit source bindings.
One existing Civil20 lexical job and its three releases remain saved.
No original source archive, model file, or genuine rating was added.

The calibration tests preserve reports only.
Their temporary forms contained existing Civil text and blank responses.
The positive replay route was explicitly mocked.
The resource preflight evaluates formulas and configured caps only.
It supplies no native runtime or memory measurement.

The Phase 7 publication report covers new project files and compressed members.
The main backup manifest covers the complete saved checkpoint.
Frozen Phase 3 through Phase 6 code and reports remain unchanged.

## Phase 8 preservation supplement — 4 October 2026

This checkpoint adds archive exports, streaming replay, staged input assembly, and scoped observed policy selection.
It records remaining software routes and missing scientific inputs explicitly.
No original corpus, semantic model file, completed human response, or primary experiment was added.

Archive format checks use existing Civil text with explicit software-only metadata.
The production archive gate refuses these substitute identities.
Intermediate check reports retain their original source hashes.
Some intermediate source snapshots were not saved before corrections.
Those earlier versions are not claimed independently replayable.
The final archive source snapshot and its final check report are saved together.

The publication scan covers changed project files and compressed members.
The final backup manifest records every included file and checksum.
No News article body or real rater identity belongs in this checkpoint.
Future private inputs and derivations remain excluded from ordinary Git additions.
Frozen Phase 3–7 preservation is verified against the previous checkpoint manifest.
