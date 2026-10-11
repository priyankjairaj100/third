V5 LOCAL LLM ANNOTATION ENGINEERING PILOT — PROSPECTIVE

Purpose
V4 was stopped after its token guard reproduced LMFE 0.11.3's stripping of
trailing U+FFFD replacement characters. Invalid byte tokens could become
invisible to that guard while accumulating in the final raw text. Its 13
completed outcomes and all 44 planned requests remain preserved.

V5 accepts a regular output token only if its standalone raw tokenizer decode
is nonempty and contains no U+FFFD. EOS is handled separately and requires
complete guarded JSON. Every accepted non-EOS transition must emit characters;
the full accumulated raw token decoding must equal the guarded text exactly.
This narrows the available output-token vocabulary, including exclusion of
legitimate split-codepoint paths and literal replacement-character tokens.
The token inventory and its hash are independently reconstructed in the audit.
This does not remove or truncate input text, change labels, or repair outputs.

Scope and controls
Same 40 canonical engineering pairs plus 4 swapped diagnostics, byte-for-byte.
Same pinned Qwen2.5-7B-Instruct model, original prompt, schema and 80-word bound.
32768 total context, 512 output limit, greedy batch 1, no retry, no fallback.
BF16 on one A100 80GB with forced Flash SDPA; model weights remain on cluster.
All 44 outcomes must remain accounted for, including any failures.
Vocabulary policy, runtime/library sources, input assets and old v2-v4 folders
are SHA256-bound in runtime_plan.json and immutable run receipts.

Validation
SOFTWARE_VALIDATION.json records actual local test results and skipped checks.
Local tokenizer assets are official pinned bytes, not a full model download.
Local CPU tests cannot establish CUDA generation or complete-run qualification.
The Slurm preflight must execute the full test suite with zero skips before
preparing inputs or loading model weights. Independent artifact audit follows
inference. Full logits are not saved; score ranking remains runtime-attested.
No semantic accuracy or population inference follows from format validity.

Execution after cluster connection is restored
Use an existing authenticated SSH session; stop after refusal or auth failure.
Fast-forward cluster repo to the backed-up commit, keep the checkout clean,
and use a new private annotation_v5/qwen-pilot-v5-20261011-a01 run directory.
The pilot.sbatch positional arguments are: repository, private workspace,
pinned model snapshot, new run root, unchanged v3 judge Python, prior v4 run.
Submit only through Slurm using --propagate=NONE. Do not overwrite a run root
or resubmit after an ambiguous receipt; inspect existing launch evidence first.
No v5 cluster job has been launched as of this package's local checkpoint.

Scientific prerequisites
AskUbuntu source-lineage acceptance and the complete frozen CPU/E5 cache still
precede substantive annotation. This exposed 420-pair engineering pack is not
the scientific pair population. Keep original study/human readiness false.
Your 50-100 blinded reviews remain unused and reserved for the later audit.
