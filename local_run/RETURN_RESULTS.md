# Return results without losing context

Send a report after setup, semantic calibration, the development gate, the confirmatory dispatch, and final analysis. Send a blocked/failed stage immediately if it prevents progress. Keep each report immutable and keep all underlying evidence locally.

## Create the private report

```bash
cp local_run/templates/results_report.template.json local_workspace/reports/report-001.json
```

Have the local LLM fill this file from actual observations. Remove every `REQUIRED_` placeholder. Keep unavailable numbers `null`; never replace them with zero. Set `reviewed_for_sharing` to `true` only after checking this report contains no private document text, personal annotation details, credentials or local account paths. The report is copied into the return bundle verbatim; the collector cannot assess the truth or privacy of free text.

Use the following fields:

- `stage`: `setup`, `preparation`, `calibration`, `development`, `confirmatory`, or `analysis`.
- `evidence_role`: `preparation`, `engineering`, `development`, `confirmatory`, or `analysis`.
- `status`: `not_started`, `in_progress`, `blocked`, `failed`, `completed`, or `completed_with_limitations`. This is a report declaration, not software acceptance.
- `inputs`: one object per corpus/model/cache, with `name`, actual `revision_or_archive`, `sha256` if available, `accepted` and a concise missing-input reason. No raw text, download credentials or local paths.
- `completed_work`, `blocked_work`, `deviations`, `negative_findings`: concise factual statements, including all failed gates and changed settings. Separate code changes from new scientific evidence.
- `results`: actual aggregate rows. Each row should name `claim`, `dataset`, `panel`, `encoder`, `group_id`, `method`, `arm`, `metric`, `estimate`, `unit`, `interval`, `independent_units`, `planned_units`, `failed_units`, `missing_units`, `zero_admission_units`, `source_artifact_sha256`, and `scope`. Use null for unavailable estimates/intervals; interval includes its method and level. State the comparison and denominator. Do not report 21,332 jobs as independent observations.
- `human_collection`: actual counts and a receipt hash; no annotator names or individual judgments in this shared summary.
- `next_action` and `questions_for_research_assistant`: make a blocked run actionable.

For the final paper report, include these parallel result groups, whether favorable or not:

1. Curation changes: selection/addition/removal counts, blocker distribution and zero-admission rates by corpus and request arm.
2. Learning consequences: oracle-relative prediction/loss effects with paired, source-aware uncertainty; utility relative to uncurated/control baselines.
3. Correctness: selected-set/state/head agreement, certificates, failures and scope of exact versus numerical claims.
4. Systems: actual total persistent bytes, peak working memory, build/repair/verification/serialization cost and amortized lifecycle; retain compact eligible payload as a main comparator.
5. Robustness/extensions: projection, encoder, refit, precision, convex, replication, negative controls and human assessment as specified in the program.
6. Complete coverage: all planned jobs/estimands, failed or unavailable cells, deviations, analysis lock and multiple-testing decisions.

Use the exact metrics and statistical definitions from the empirical program. The field suggestions above do not define a new estimator.

## Package a partial report

```bash
python3 local_run/collect_results.py \
  --report local_workspace/reports/report-001.json \
  --out local_workspace/reports/report-001.zip
```

## Package a finalized dispatch report

```bash
python3 local_run/collect_results.py \
  --report local_workspace/reports/report-002.json \
  --dispatch local_workspace/runs/REPLACE_WITH_FINAL_DISPATCH \
  --out local_workspace/reports/report-002.zip
```

The optional dispatch directory must contain its actual final `summary.json` and `jobs.json`. The collector verifies their hash binding and complete registered job identities. It exports only canonical job/group IDs, declared statuses, method IDs and acceptance flags, plus aggregate accounting. It omits artifact paths, error messages, documents, predictions and annotation content. It does not rerun semantic acceptance or verify statistical estimates in your report. `completed_dispatch` can include blocked or failed jobs; all remain in the returned accounting.

The ZIP contains `report.json`, `checkpoint.json`, `manifest.json`, and, if supplied, `dispatch_accounting.json`. Source filenames are replaced by fixed public bundle names. The checkpoint records the repository commit/tree, tracked-worktree cleanliness, and protocol/registry/report hashes. A dirty checkout is disclosed, never silently called reproduced. No corpus/model/raw-human file is auto-attached. Output overwrite is refused.

**Upload the ZIP in the next chat** and say: “Review this local report against https://github.com/priyankjairaj100/third and `local_run/EMPIRICAL_PROGRAM.md`. Separate verified ledger accounting from reported measurements; tell me the next supported action.” Keep the complete private workspace on your machine. The ZIP is a compact handoff, not a substitute for raw reproducibility evidence. If a discrepancy needs predictions, certificates or a particular log, we can specify a narrow additional evidence export after inspecting this report.

Do not commit the private workspace or report bundle to the public repository automatically. Public result tables and a redacted reproducibility release can be prepared after inspecting the actual results and source permissions.
