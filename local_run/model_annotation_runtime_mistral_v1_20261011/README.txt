MISTRAL SECOND-JUDGE ENGINEERING PILOT — FROZEN EXECUTION PACKAGE

Purpose and source boundary
Separately qualify mistralai/Mistral-Nemo-Instruct-2407 at revision
04d8a90549d23fc6bd7f642064003592df51e9b3. This model was already named in the
original model-assisted plan. Preserve Qwen v2-v5 code, outputs and failures.
Reuse the byte-identical v5 bounded_reason.py implementation and its disclosed
restricted token policy; guard schema identifiers intentionally remain v5.

The 40 canonical pairs and four swapped diagnostics must match the earlier
Qwen pilot input file byte-for-byte. They come from the exposed 420-pair
engineering pack. The same original system prompt, task definition, judgment
schema and 80-word reason bound apply. Prior judgments are not shown to Mistral.
Full prompt tokenization covers all 840 engineering pair orientations; that
is not the complete scientific calibration population. All 44 planned model
outcomes remain in the denominator, including failures and overflows.

Model and inference contract
The pinned native context capacity is 131072 tokens. This pilot uses a 32768
execution subcap INCLUDING 512 output tokens, without input truncation, RoPE
scaling, retries, format repair or an attention fallback. Longest full prompts
run first; ties retain original input order. The model is MistralForCausalLM,
with the pinned PreTrainedTokenizerFast native chat template and BOS 1/EOS 2.
Run greedy batch 1 BF16 on one A100 80GB using forced Flash SDPA. Model weights
stay private; no model inference is performed by local software tests.

The reused output guard excludes regular tokens whose raw standalone decode
is empty or contains U+FFFD. This deliberately also excludes legitimate split
Unicode-codepoint paths and literal replacement-character tokens. Every
non-EOS transition must emit text and exactly match accumulated raw decoding;
EOS requires completed JSON. Final strict JSON, original schema and decoded
Python str.split reason length remain authoritative. Raw failed outputs and
token inventories are retained. The independent audit reconstructs token
eligibility and source text, but full-logit argmax remains runtime-attested.

Validation and execution
SOFTWARE_VALIDATION.json records observed local tests and actual skips.
The actual pinned tokenizer regression passed locally: 46 discovered tests,
44 passed and two explicit PyTorch integration skips. All eight nonweight
assets matched recorded Git blob SHA1 and size. The runtime amendment remains
unsealed until the package is reviewed. Local tests do not establish CUDA behavior,
semantic correctness, judge independence or population precision. The Slurm
preflight must pass all 46 tests with zero skips before preparation/inference.
The final independent audit requires a completed 44-outcome run and preserves
individual failed statuses; integrity passing is separate from all 44 valid.

pilot.sbatch requests 4 CPUs, 64 GiB host RAM, one A100 80GB,gpu-short, 2 hours,
excludes csis.mn1, and disables inherited Slurm resource propagation. Its seven
positional arguments are: repo, private workspace, pinned model snapshot,
new existing 0700 private run root, dedicated judge Python, prior Qwen run,
and EXPECTED SHA256 OF THE SEALED runtime_plan.json. The wrapper checks that
hash, every runtime package member, preserved v2-v5 bindings and dependency
versions BEFORE importing tests or preparation. Existing output names are
refused. Submit only through an existing authorized connection and Slurm;
no remote actions are performed by this package preparation.

Source facts for separate v2 development are documented in
local_run/source_acceptance_v2_20261011. The full CPU/E5 representation and
prospective model/author-audit requirements remain separate. Genuine human
collection, original v1 three-rater admission and confirmatory readiness
remain false. The user's 50-100 blinded reviews are reserved for later work.

Tokenizer warning diagnosis
Transformers4.57.6 emitted a general Mistral-regex warning. Inspection of its
actual code and the pinned tokenizer shows that the existing regex equals
the proposed replacement character-for-character with identical split
behavior. Its warning branch checks model/vocabulary metadata without checking
the existing regex. The warning is a false positive for these pinned bytes;
no fix flag, production tokenization change or source modification was made.
SOFTWARE_VALIDATION.json binds the saved diagnosis and exact source hashes.
