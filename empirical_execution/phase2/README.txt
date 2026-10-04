COUNTERFACTUAL CURATION: EMPIRICAL DEVELOPMENT, PHASE 2
4 October 2026 | Everything executed in this conversation workspace

What actually advanced
The empirical pipeline now measures prediction and loss consequences of
deletion-induced admissions on separate diagnostic evaluation records. It
also has an executable intake validator for the planned semantic study.
This stage is completed development work, not the primary publication study.
The original protocol and previous engineering/strict-mode results are intact.

Fixed experiment
The reused 100 Civil Comments preview records were grouped by NFC-normalized,
whitespace-collapsed text, preserving case. A fixed salted hash assigned 82
records to development training and 18 to diagnostic evaluation. No split
retry or outcome selection occurred. All records had already been available
during earlier engineering work; the evaluation subset is not untouched test
data. Native source IDs are absent. Exact-text groups do not cross the split,
but source and semantic leakage guards are unavailable. There are 119
train/evaluation lexical pairs above the uncalibrated threshold of 0.6.

Training-only curator: fixed 128-dimensional lexical word/bigram hashing,
strict cosine threshold 0.6, stable priority. Learner hashing dimensions 64
and 768, original fractional toxicity labels, lambda 0.01, no intercept and
no clipping of scores. Neither representation is an E5 semantic embedding.

Before computing losses, the runner saved design/code/data hashes, all split
IDs and all graph-only request paths. The same 64 random-record R, 32 initially
excluded-record U and 16 topology-stress A trajectories were evaluated at
1,2,4,8 cumulative deletions in both dimensions: 896 checkpoint rows. There is
no source arm. Each trajectory resets its state; prefixes are sequential.

Methods and outcomes
O-G independently rescores the retained training graph, reconstructs the
selected set and trains ridge from fresh moments. B-F trains on the surviving
members of the INITIAL selected set, using the same lambda*n normalization.
B-F is not the unchanged initial head. Prediction arrays and original
evaluation labels are saved so every MSE/RMS can be independently recomputed.
The O-G cache holds selected sets and predictions, not dense Gram histories.

All structural zeros and negative signed effects remain in the output.
MSE_effect = MSE_frozen - MSE_oracle. Relative effect divides by fixed
predeletion curated MSE; normalized prediction RMS divides by fixed
predeletion score SD. Zero denominators remain undefined without epsilon
floors. Paired method differences are aggregated per reset trajectory.
The 10,000-replicate percentile intervals are descriptive, conditional on this
fixed corpus/split/feature map. They do not include test-source uncertainty,
support population inference, or define a confirmatory significance test.
Arm streams are independent and are not paired by their ordinal IDs.

Observed signal at eight deletions
R: 19/64 paths had additions; mean additions 0.34375. At d=768, mean relative
MSE effect 0.09993%, descriptive 95% interval [-0.04319%,0.26377%].
U: 17/32 paths had additions; mean additions 0.90625. At d=768, mean relative
MSE effect 0.12681%, interval [-0.12929%,0.45036%].
A: 16/16 paths had additions; mean additions 6.0625. At d=768, mean relative
MSE effect 4.45329%, interval [4.27438%,4.62364%]. This is targeted graph stress,
not natural random-request prevalence. A has only one distinct deletion set
at each of checkpoints 1, 2, 4 and 15 distinct sets at checkpoint 8. Identical
stress prefixes must not be counted as distinct empirical mechanisms.

Important unfavorable context
At d=768, predelete curated MSE 0.02364 exceeds uncurated 0.02093,
same-size hash-selected 0.01919 and training-mean baseline 0.02029.
The d=64 panel also has weaker curated performance. Thus this preview does
not establish a credible curation utility benefit. The positive stress
effect cannot substitute for semantic calibration, a source-disjoint main
test set, native-source evidence, or the second labeled corpus.

Numerical integration
At d=768, first R/U/A paths were separately replayed through the strict exact
state at all four checkpoints. All 12 states matched fresh retained rebuilds
byte for byte, and all 12 released heads passed the exact residual certificate
at tolerance 1e-10. The largest radius was 4.5091e-14. These 12 audits do not
turn all 896 floating outcomes into exact numerical certificates.

Main files
run_development.py             frozen execution runner
analysis.py                    paired trajectory analysis/undefined handling
audit_development_results.py   independent saved-result verification
results/                       design, requests, split, predictions and results
intake_validate.py             local corpus/semantic-cache validation
intake_checks.py               real-fixture and invalid-metadata checks
assets_requirements.txt         minimum real inputs and detailed format
assets_manifest.template.json  fill only with actual local artifacts
assets_manifest.schema.json    row/cache/response structure
assets_readiness.json           actual blocked main-study status

Replay from workspace/extracted archive root
Preserve phase2/results before rerunning; runner replaces its result files.
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 empirical_execution/phase2/run_development.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 empirical_execution/phase2/audit_development_results.py
No network
or remote job is used. Existing numerical dependencies are in requirements.txt.
The exact backend needs the included platform-specific native helper or the
Python Fraction fallback; see ccu/native/README.txt.

Next real empirical step
Provide the source-rich original Civil Comments corpus files and locally
usable pinned encoder/runtime, or a corpus bundle with aligned frozen FP32
semantic embeddings and derivation evidence. See assets_requirements.txt.
The first useful import can be partial: we can inspect originals, prepare
source-disjoint panels and create the blinded calibration tasks here. Genuine
human judgments and the remaining design lock are required before confirmation.
The validator distinguishes machine-checked fields from human/provenance
evidence; a success status is never permission to invent missing content.
