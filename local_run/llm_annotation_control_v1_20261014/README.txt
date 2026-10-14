One-time LLM selection execution control, 14 October 2026.

This controller first submits a tokenizer-only CPU qualification, then permits
serial Qwen and Mistral jobs under the same separately published activation.
It cannot generate sampling randomness, create an activation, retry a call or
resume an existing output directory. A durable intent precedes each single
sbatch; ambiguous submissions require reconciliation and never automatic retry.

Qualification uses the pinned judge environment on csis.cn1, 4 CPUs, 8 GiB and
20 minutes, with CUDA disabled. It binds all final packages and prepared request
artifacts; the execution package tests both actual tokenizers without loading
weights. It additionally replays fixed call-0000 from each already exposed
historical pilot through the new raw/token auditor, creating no new labels.
The qualification allowance is 32 MiB, including at most 16 MiB private
output and 14 MiB console logs. Admission also requires the future two-family
192 MiB commitment above an 8 GiB quota reserve.

Each production family uses one A100 80GB, 4 CPUs, the already-qualified host
allocation (Qwen 48 GiB, Mistral 64 GiB), a prospective four-hour cap and a fixed
64 MiB artifact cap. A 96 MiB allowance per family includes 14 MiB console logs
and metadata margin. Quota checks are fresh snapshots, not reservations. Qwen
admission retains both families' commitment; Mistral retains its remaining one.
The first real GPU generation is a planned attempt under activation, with no
unrecorded qualification probe. CPU checks do not establish new GPU execution.

Control verifies package members, resource/request/audit pins, completed request
job 13572, the executing frozen checkout, fresh quota and a clear project queue.
Production requires the exact admission/activation and terminal CPU qualification;
Mistral additionally requires Qwen's successful process termination. A failed
Qwen process is retained for review before any further family submission.
The child checks its exact intended command and interpreter, captures bounded
logs through process exit, and preserves a completion receipt. Slurm enforces
the time and resource ceilings; no login-node model work is permitted.

Raw results are accepted only after the independent output audit and statistics
consumer. Missing or failed planned calls remain in the denominator. An incomplete
or integrity-failed run stays quarantined for explicit reconciliation. Nothing
here claims human agreement, source authenticity, logit replay or semantic truth.
