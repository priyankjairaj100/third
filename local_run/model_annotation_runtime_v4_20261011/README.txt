THIRD — ANNOTATION CONSTRAINT CORRECTION V4

Prospective engineering version retaining the identical 40 pairs and 4 swapped
checks, cached model, BF16 Flash-only backend, original rubric and 80-word limit.
V3 code and results remain unchanged. The new decoder guards decoded reason
words during generation and verifies effective tokenizer-alphabet configuration.
Native LMFE proposals may use its free-text shortcut. The final candidate guard
disables shortcuts and checks candidates in descending score/token-ID order.
The first legal candidate is selected in the same greedy decoding step; no
generation retry occurs. Non-prefix-stable token transitions are rejected.
No retries, truncation, semantic relabeling or human reviews are permitted.
Passing the bounded pilot does not qualify semantic accuracy or study readiness.

Dependencies reuse the unchanged dedicated v3 judge environment. No new models
or libraries are downloaded for this correction. The exact decoder policy and
source hashes will be sealed in runtime_plan.json before any inference.
