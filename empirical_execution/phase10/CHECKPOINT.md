# Phase 10 checkpoint — 4 October 2026

This checkpoint strengthens the memory theorem and its algorithmic correspondence.
It closes the finite-input and sequential-state qualifications left by Phase 9.
It does not start the primary semantic study.

## Scientific result

The construction stores the same FP32 feature rows for curation and ridge learning.
All labels and forgotten blocker payloads are fixed and public.
Private information consists of `m` candidate sign blocks, each containing `D` bits.

The frozen FP64 scorer yields the same graph for every private sign assignment.
The proof assumes the stated IEEE operations and `2+k+D <= 2^20`.
Its score error stays below `2^-32`.
Its edge and nonedge margins exceed `0.029` and `0.027`.

Every head through record-deletion budget `b` is zero.
At budget `b+1`, designated requests reveal one candidate's private block.
The initial exact private-state minimum therefore rises from zero to `mD` bits.
No continuity assumption enters this finite-state bound.

After a fixed cumulative deletion set `F`, the exact minimum is `e(F)D` private bits.
Here, `e(F)` counts candidates still reachable within the remaining original horizon.
The codec stores those blocks in immutable canonical bytes.
It needs no retained-feature access during repair.
Fresh retained initialization and sequential repair agree under the declared contract.

The proof also derives an explicit expected head-error bound using classical rate-distortion reasoning.
A separate corollary covers sufficiently accurate certified numerical releases.
An actual exact-target error certificate is required.
Declaring a tolerance does not establish the corollary's premise.

## Verification and scope

The source-bound reports are authoritative for exact versions and counts.

| Report | Scope |
| --- | --- |
| `results/finite_word_review_final.json` | Final proof review and source bindings. |
| `results/scorer_transfer_final/checks.json` | Frozen scorer and rational margin checks. |
| `results/finite_word_review_checks_final.json` | Independent algebra and output-contract checks. |
| `results/codec_checks_corrected.json` | Owner codec, fresh graph, exact target, and history checks. |
| `results/codec_independent_final/checks.json` | Independent implementation and byte-contract checks. |
| `results/frozen_preservation.json` | Byte preservation of Phases 3–9. |
| `results/checkpoint_verification.json` | Final report/source consistency and scope checks. |

The main proof is `FINITE_WORD_THEORY.md`.
The manuscript fragment is `finite_word_theory.tex`.
Read both independent reviews and `SCORER_TRANSFER.md` with the theorem.

The scorer suite passes 85 bounded checks.
The independent algebra suite passes 22 assertions.
It covers 256 tiny private assignments and 512 exact normal equations.
It also checks 7,424 low-budget queries and 39,424 state-drop transitions.
The codec owner suite passes 51 checks across 260 tiny family assignments.
It compares 5,888 fresh graphs and exact heads.
It also checks 19,780 sequential transitions.
The independent codec suite passes 51 checks across 1,560 state cases.
It verifies 20,280 exact normal-equation coordinates.
These are mathematical software fixtures, not empirical datasets.
Enumeration does not establish the universal entropy or arithmetic arguments.

The exact model target is rational ridge on stored input values.
It does not prescribe a legacy floating solver's output bytes.
Signed zero, numerical equality, and byte equality remain distinct release contracts.
The private-information minimum excludes public construction storage and deletion metadata.
Padding, materialized outputs, and workspace add real costs.
The approximate coding upper bound is an initial-query existence result.
It is not an efficient sequential lossy codec.

Review corrected two input contracts: string subclasses and serialization of large rational integers.
It also corrected two LaTeX statement gaps and a checker report-path error.
The failed attempt, source snapshots, and theorem correction history remain saved.
No result establishes total-byte optimality, physical erasure, or a natural NLP task effect.
Earlier lexical evidence still favors compact eligible payload in total bytes.

## Novelty assessment

The GRACE author-manuscript gap is resolved.
The recent counterfactual-auditing comparison is narrower after inspecting official author code.
Its full paper and precise approximate-success event remain unresolved.
The Dominici thesis remains a metadata-only comparison.

The broad prediction-versus-counterfactual memory distinction is unsafe as a novelty claim.
The defensible object is the joint curation, stored-feature, ridge-output, and sequential-state construction.
Standard coding, entropy, rounding, and bit-packing tools receive explicit attribution.
The extension strengthens the same paper rather than supplying another independent headline.
No search guarantees priority or acceptance.

## Preservation and next work

The baseline is commit `9b21ef0475134ce6c9f9c5254a1009715cc57c47`.
The frozen preservation report verifies all 2,208 Phase 3–9 files against that checkpoint.
Their total size is 111,668,883 bytes.
New files live in Phase 10; root handoffs and current status are updated.
The publication audit and backup manifest cover the resulting checkpoint.

Executed primary jobs remain zero.
Genuine human responses remain zero.
No original corpus, encoder model, or semantic cache was added.
No external compute or person-directed annotation collection was started.

Proceed with authentic input intake and the ordered tasks in `REMAINING_TASKS.md`.
The planned six figures, three tables, and empirical manuscript still require accepted study evidence.
