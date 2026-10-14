One-time tokenizer-only preparation controller, 14 October 2026.

Requires completed full census job 13571, its successful 22-test/process receipts,
exact result and independently bound sample review, and the existing immutable
sampling lock. It cannot accept the engineering qualifier or create fresh seeds.
Fresh private stage root, clear relevant queue, clean unchanged executing checkout
and fresh 8 GiB quota reserve plus 128 MiB allowance are required before one sbatch.
An exclusive durable intent precedes submission. Failure/unknown outcome remains
for reconciliation; no retry or alternate output slot is offered.

Preparation runs on csis.cn1 in CPU-short, 2 CPUs, 8 GiB, 20 minutes, excluding
csis.mn1. It uses the existing qualified judge Python for both tokenizers, offline,
one numeric thread, CPU tensors and disabled visible CUDA. No model weights or
forwards are authorized. Exact tokenizer/config and source payload reads happen
only in the Slurm child. The preparer enforces a 64 MiB total private output cap.
The 128 MiB stage allowance includes output, 1 MiB test/14 MiB child logs, wrapper
margin and contingency. Quota is a snapshot, not a reservation or runtime guarantee.

Read/hash only small code, metadata, selection artifacts and existing asset
manifests on the login node. Existing qualified sampler/frame control supplies
verified quota, bounded capture and durable single-submission helpers. The child
receives exact original source/snapshot roots separately and revalidates matching
payloads. Neither a passing preparation nor tokenizer feasibility activates
inference, establishes semantic validity or passes historical source/human gates.

Tests cover engineering/full-census separation, bound review identity, unexpected
prior activation, relevant-job refusal and exact argument/source/lock separation.
