Review of actual private job15243 task locks and job15241 Qwen audit receipts.

verify.py takes --inputs (the private exported files), --repo (this checkout)
and --out (a new JSON path). It has no network, scheduler or model operations.
It does not import the frozen selector when recomputing selected-loss arithmetic,
source folds and all20F1 thresholds. It checks receipt seals, exact file bindings,
source/package pins and preserved Qwen failure provenance.

The compact completed review is saved under local_run/local_llm_execution_20261014/
task_and_audit_independent_review_20261016.json. Per-record labels, predictions,
source IDs, source text and raw model responses are not published here.
The review explicitly distinguishes local arithmetic checks from original
cluster source/vector/grammar execution; it is not a second model run or a
human semantic-validity judgment.
