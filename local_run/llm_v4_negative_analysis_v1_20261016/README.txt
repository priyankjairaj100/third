Completed v4 negative selection: separate aggregate replay

The finite controller completed on2026-10-16 at09:52:01UTC. Both model output
audits passed; the independently recomputed selection decision was negative.
No fresh validation, threshold-dependent guard or effect experiment followed.

analyze.py is a standard-library, local-only reporting program. It does not
import the frozen statistics implementation, contact a model/server, modify a
protocol, or publish private pairs. Exact input byte counts/SHA256 are pinned
to the controller's completed independent review. It verifies canonical seals,
complete panels, sampling weights/bins, all51 complete cut records, and the
predefined swap diagnostics. It also produces descriptive aggregate category
cross-counts and bin counts; those are not additional confirmatory tests.

Run from the repository with an unused output path:
python3 -B local_run/llm_v4_negative_analysis_v1_20261016/analyze.py \
  --private local_workspace/reports/llm-v4-independent-selection-review-a01 \
  --output <new-aggregate-report.json>

The committed result is
local_run/local_llm_execution_20261014/llm_v4_negative_diagnostics_20261016.json.
The companion llm_selection_v4_terminal_result.json contains sanitized terminal
receipts and the controller review. Analysis source is bound in the report.
An existing output is never overwritten. Do not change the pinned inputs to
accept another outcome; new reporting versions must identify their scope.

Observed result:27/360 raw canonical joint positives; maximum weighted joint
proportion2/5 versus19/20 required; favorable sensitivity maximum8/15. The raw
stratified-sample ratio is not a population prevalence estimate. Sensitivity
is missing/uncertain imputation, not a confidence bound. Six-way disagreements
308/355 refer only to pairs with two valid responses. Rater agreement is not
human validity. No humans, retries, prompt changes or cut substitutions.

The output contains aggregate counts and input bindings only. Per-pair labels,
IDs, account metadata, source text and raw model output remain private. This is
an author arithmetic replay, separate from the executed CPU audit and controller
review; it is not an independent human or model assessment of semantic labels.
