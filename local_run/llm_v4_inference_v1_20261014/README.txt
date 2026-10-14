V4 inference, independent audits and selection statistics.

The derivative preserves the exact execute_call generation/grammar function
from the completed v3 implementation. Changes concern v4 source pins, identity
domains, response envelope versions and prepared-bundle metadata. Local tests
exercise single-attempt identities, missing outcomes and exact preservation of
the generation function. These tests are not model accuracy measurements.

The workflow first replays every actual v4 prompt/token sequence on CPU using
both tokenizers. Only successful terminal qualification permits the separate
activation writer. Each model then runs under a fresh exclusive Slurm stage,
with complete weight hashes, per-call artifacts, no retries, full denominators
and bounded lifecycle capture. Qwen's independent CPU audit must pass before
Mistral can launch. Statistics requires both independent audits. Selection
failure is retained; passing selection still requires fresh validation.

CPU stages: 2 CPUs, 8 GiB RAM, 20 minutes. GPU stages: 4 CPUs, one A10080GB
on the previously qualified csis.cn3, 48/64 GiB host RAM for Qwen/Mistral,
6 hour limit per family, 256 MiB output cap plus 16 MiB console cap. The
gpu-short partition's observed maximum is 8 hours. Fresh quota must retain
8 GiB plus the stage allowance. Quota snapshots are not reservations.
All stages retain exclusive launch intent/receipt, tests and process exit
receipts. Never restart an occupied stage or retry an uncertain submission.

The output auditor separately reconstructs call identities, prompt token IDs,
tokenizer/grammar configuration, decoded output and token guard transitions.
It binds the model-run weight attestations; it does not independently reread
weight payloads or replay full logits. Invalid JSON remains nonpositive.
Statistics uses the v4 thirteen-stratum weights and unchanged 51-cut/19-of-20
selection rule. No scientific threshold or semantic claim is guaranteed.

Old v3 sources and outputs remain immutable. Human annotation remains zero.
This is the selection workflow only. Independent validation and downstream
repair/utility/cost work are not completed by a selection run.
