COUNTERFACTUAL CURATION: OFFLINE EMPIRICAL PREPARATION
Date: 4 October 2026 | Intended paper: ACL 2027

WHAT IS COMPLETE
Source-disjoint population preparation, exact/link guards, a common pair-local
floating-point similarity reference, whole-source panels, blinded threshold
sampling, independent quality-validation sampling, exact finite-population
confidence bounds, a local CLI, and versioned intake corrections are implemented.
Independent software audits and natural-text engineering checks passed.

WHAT IS NOT COMPLETE
The primary semantic empirical study has not started. The existing100 Civil
Comments and90 CC-News records are reused engineering previews. Civil lacks native
article/publication IDs. News host groups are not verified native source units.
No E5/MPNet embeddings, original source-rich corpus snapshot, or actual independent
human ratings are available. No semantic threshold or quality pass exists.
This package does not establish publication readiness or reviewer-proof claims.

START HERE
Run commands from the extracted package root with Python3.12, NumPy2.3.5,
SciPy1.17.0 and scikit-learn1.8.0 (versions used for the recorded checks).
No command below downloads data or launches external compute.

  python3 empirical_execution/phase3/check_reference_graph.py
  python3 empirical_execution/phase3/check_workflow.py
  python3 empirical_execution/phase3/check_intake_v2.py
  python3 empirical_execution/phase3/audit_finite_population_independent.py
  python3 empirical_execution/phase3/audit_scoring_and_pack_independent.py
  python3 empirical_execution/phase3/audit_guards_independent.py

check_calibration.py verifies an identical existing assignment pack and refuses
replacement if hashes differ. Preserve the delivered pack when making code changes.
check_panels.py additionally needs the locally acquired News preview,
which is deliberately excluded from the distributable package. Its recorded
48-check audit is included. Natural-text data are never synthesized to fill gaps.

LOCAL PRIMARY WORKFLOW
1. Supply authenticated raw corpus files and the records.jsonl normalization of
   those files, preserving stable native IDs and original_fields. Parsers, date
   filtering, source completeness and duplicate-link extraction need original
   snapshot verification; this CLI does not invent or authenticate those inputs.
   Follow assets_manifest_v2.template.json and intake_v2_changes.txt.

  python3 empirical_execution/phase3/run_preparation.py prepare \
    --records INPUT/records.jsonl --dataset civil_comments \
    --duplicate-links INPUT/duplicate_links.json --out RUN/prepared

   duplicate_links.json is a list of native stable-ID pairs, not semantic guesses.
   This writes fixed-guard records, preliminary panel IDs and a sealed audit.
   Training against calibration/test semantic filtering is still pending.

2. Supply the pinned E5 FP32 document vectors aligned exactly to prepared records,
   plus an embedding derivation log: model/tokenizer revisions, input hashes,
   chunking, pooling, normalization, dtype and runtime. Cache file format: N-by-768
   FP32 .npy, memory mappable. cache_manifest.json has these exact required fields:

   row_ids: complete ordered record ID list
   normalized_records_sha256: SHA256 of canonical [{record_id,text},...]
   prepared_rows_sha256: SHA256 of canonical full prepared row objects
   cache_file_sha256: SHA256 of .npy bytes
   provenance:
     dataset_id: civil_comments
     encoder_id: intfloat/multilingual-e5-base
     encoder_revision: actual pinned revision
     population_scope: original fixed calibration population description
     evidence_role: confirmatory_calibration

   Canonical JSON = UTF8, sorted keys, no spaces, ensure_ascii=False,
   allow_nan=False (phase3.calibration.digest). These fields detect inconsistency;
   they do not prove that an encoder or historical data source is authentic.

  python3 empirical_execution/phase3/run_preparation.py selection \
    --records RUN/prepared/records.jsonl --preparation-audit RUN/prepared/audit.json \
    --vectors INPUT/e5.npy --cache-manifest INPUT/cache_manifest.json \
    --out RUN/selection

3. Independently collect three genuine blinded human ratings per sampled pair
   using CALIBRATION_ANNOTATION_GUIDE.txt. Give raters only their text assignments;
   the private sampling manifest and response collection declaration remain with
   the collector. No files are automatically sent. Do not flip attestation fields
   or fill labels without actual collection. A declaration is not authentication.

  python3 empirical_execution/phase3/run_preparation.py threshold \
    --manifest RUN/selection/private_sampling_manifest.json \
    --responses INPUT/completed_selection_responses.json --out RUN/selection_lock.json

4. With the frozen threshold, create a fresh independent SRS validation pack.
   Reuse the records, preparation audit, vector and cache-manifest arguments from
   step2. The threshold lock fixes the scorer, software hashes and block size.

  python3 empirical_execution/phase3/run_preparation.py validation \
    --records RUN/prepared/records.jsonl --preparation-audit RUN/prepared/audit.json \
    --vectors INPUT/e5.npy --cache-manifest INPUT/cache_manifest.json \
    --lock RUN/selection_lock.json --out RUN/validation

   Collect fresh independent blinded ratings, including any pair that happens to
   overlap selection. Overlap is allowed; reusing old judgments is not.

  python3 empirical_execution/phase3/run_preparation.py quality \
    --manifest RUN/validation/private_sampling_manifest.json \
    --responses INPUT/completed_validation_responses.json \
    --lock RUN/selection_lock.json --out RUN/quality.json

5. Only after a passed matching statistical/scope report, apply the primary E5
   training-only guard and make final whole-source panels. Independent external
   provenance review and the full pre-execution lock remain necessary.

  python3 empirical_execution/phase3/run_preparation.py guard \
    --records RUN/prepared/records.jsonl --preparation-audit RUN/prepared/audit.json \
    --vectors INPUT/e5.npy --cache-manifest INPUT/cache_manifest.json \
    --lock RUN/selection_lock.json --quality RUN/quality.json --out RUN/guarded

   The resulting IDs define both E5 and MPNet populations; never refilter for
   MPNet. reference_graph.build_reference_graph constructs the main fixed graph
   using the same ordered scorer and narrative priority-v1|seed|record-id rule.
   The old ccu.core graph builder is retained for historical engineering replay.

MEMORY AND NUMERICAL SCOPE
Pair scoring uses O(BD+B^2) temporary arrays. It still requires exhaustive
O(N^2D) arithmetic for all calibration pairs, and graph storage is O(N+m) for
m threshold edges (quadratic in the worst case). The CLI explicitly gathers
O(N_cal*D) FP32 calibration vectors. Input bytes, record metadata, graph state,
cache gathers and resident memory must all be charged in the systems study.
This is not a demonstrated speedup at the planned10k-200k scales.

The reference means frozen FP32 inputs promoted to FP64, coordinate-ordered
norms and dot products, within the locked software/runtime. It does not certify
real-arithmetic cosine or equality across arbitrary hardware. A pair's score and
stable-ID priority do not depend on other records, so retained graph reconstruction
is the induced graph of the original population under these fixed assumptions.

DESIGN CORRECTIONS AND NON-CLAIMS
See PREPARATION_AMENDMENT.txt. Earlier phase2 development outputs are preserved,
not recomputed or retroactively promoted. The real preview assignment pack in
results/calibration_engineering_pack contains241pairs and723 EMPTY assignments.
It is a plumbing example, not the actual E5 human study's annotation pack.

SHIPPED EVIDENCE
panel_engineering_audit.json:48 software/engineering checks.
results/calibration_software_checks.json:31 checks.
intake_v2_audit.json:69 regression assertions.
results/reference_graph_checks.json:10 checks,10,000 score entries.
results/workflow_checks.json:9 CLI success/refusal cases.
independent_finite_population_audit.json:2,900 bound cells,5,200 coverage cells.
independent_scoring_pack_audit.json:9,900 scalar pair oracles and graph comparisons.
independent_guard_audit.json:27 directed guard cases plus source-ID checks.
independent_review.txt: review scope and remaining scientific gates.

DATA HANDLING
The Civil100 fixture remains a preview with its original acquisition/provenance
note and labels. The News article text is not redistributed; its local engineering
audit remains included without article bodies. No completed human ratings exist.
