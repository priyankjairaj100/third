THIRD — CONSTRAINED LOCAL ANNOTATION PILOT V3

Read runtime_plan.json for this prospective engineering amendment.
Original v2 plans, code and results remain unchanged.
This version reuses the same 40 pairs and 4 order checks, preserves raw outputs,
and uses pinned JSON-constrained generation with no format retry.
BF16 Qwen2.5-7B uses PyTorch SDPA with FLASH_ATTENTION as the only allowed
backend. Native total context 32768 includes 512 output tokens. No fallback,
truncation, extended RoPE or summaries are allowed.

Preparation reconstructs exact prior input bytes and verifies source/asset hashes.
It also counts full prompt lengths for all 420 engineering pairs in both
orientations. That pack is not the full scientific pair population.
The runner schedules all 44 selected calls longest-first, preserving original
indices and pair seeds. Failures remain in the full denominator.

The dedicated v3 judge venv adds only pinned grammar dependencies; the CUDA
stack is read-only from the unchanged encoder-cuda environment. Raw texts,
model outputs, token ledgers and runtime manifests stay in local_workspace/.
No human review is consumed. Passing this bounded execution check does not
validate semantic accuracy, human agreement, or confirmatory-study readiness.
