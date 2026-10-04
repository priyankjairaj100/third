# Phase 10: finite-format memory and exact sequential repair

This phase closes two specific gaps in the Phase 9 construction.
It uses exactly stored FP32 features under the frozen FP64 cosine scorer.
It also supplies a canonical sequential codec for the constructed family.
These are theory and software results.
They are not natural-language experiments.

The primary semantic study remains unstarted.
Original corpus archives, pinned encoder assets, and genuine human judgments remain absent.
Earlier phases remain frozen.

Read `CHECKPOINT.md` for verification and `REMAINING_TASKS.md` for the next scientific work.

## Files and authority

| File | Purpose |
| --- | --- |
| `FINITE_WORD_THEORY.md` | Family, access model, memory bounds, approximation limits, and sequential state proof. |
| `finite_word_theory.tex` | Manuscript theorem fragment. |
| `SCORER_TRANSFER.md` | Uniform graph proof for the actual frozen scoring source. |
| `FINITE_WORD_REVIEW.md` | Independent proof review and stronger-claim counterexamples. |
| `finite_word_codec.py` | Immutable canonical bytes, cumulative repair, and exact rational head. |
| `CODEC.md` | Interfaces, storage accounting, execution costs, and scope. |
| `CODEC_REVIEW.md` | Independent implementation review. |
| `LITERATURE_FOLLOWUP.md` | New primary-source comparisons and remaining retrieval limits. |
| `CONTRIBUTION_POSITION.md` | Recommended paper claims and rejected novelty claims. |
| `TODO.json` | Current machine-readable task ledger. |

Use the proof, scorer contract, and independent review together.
Read the codec document before calling its interfaces.
The retained checks use small algebraic fixtures only.
They do not establish theorems through empirical enumeration.

## Main distinction

The same stored feature rows define curation and the ridge learner.
Public blocker payloads and labels remain fixed across all private instances.
Every target through deletion budget `b` is zero.
At budget `b+1`, designated requests expose one hidden candidate each.

The exact initial private-information minimum is `mD` bits on this family.
At a fixed reachable deletion set, the minimum is `e_F D` bits.
Here, `e_F` counts candidates that can still enter within the remaining horizon.
The codec retains those sign blocks and discards the rest.
Its bytes depend on the retained instance and remaining horizon, not deletion order.

Those minima exclude public construction data and deletion metadata.
Physical bytes, padding, output precision, and workspace remain separate costs.
All private side information counts toward each information lower bound.
The exact output is the rational ridge target on stored inputs.
It is not a claim about legacy floating solver bytes.

The approximate result measures head error under a specified norm and access model.
It does not imply a fixed NLP task-loss effect.
Its classical rate-distortion component is attributed explicitly.

## Empirical continuation

Phase 9 remains the authority for multi-family assembly and resource evidence.
Phase 8 retains original archive export and staged input preparation.
Frozen Phase 6 interfaces retain the accepted execution routes.
No primary acceptance rule changes in this phase.

Run `python3 tools/verify_backup.py` before changing saved evidence.
Use an isolated checkout for checks that write reports.
Preserve failed attempts and their bound source versions.
