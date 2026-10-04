# Phase 10 literature follow-up and contribution assessment

Audit date: **4 October 2026**. This targeted follow-up preserves the frozen
Phase 9 audit. It revisits three retrieval gaps and assesses the finite-format
extension. It is not an exhaustive novelty certificate. No experiments,
benchmarks, source-paper copies, or author contacts were made for this audit.
The companion `sources/literature_followup_sources.json` records primary URLs,
versions, retrieval outcomes, and the inspected local sources.

## Decisive assessment

**The finite-format extension closes substantive mathematical-model gaps; it
does not establish a new general memory or coding principle.** Its strongest
role is to sharpen the same paper's principal theorem. Exact stored FP32
inputs, a proof for the frozen numerical cosine scorer, and a canonical
sequential private-bit state replace two limitations explicitly left by
Phase 9: ideal-real feature variation and an initial-query-summary upper
bound. These are material improvements to theorem applicability and
implementation correspondence. They do not establish practical optimality or
semantic prevalence. The empirical program remains necessary for an ACL paper.

The defensible distinct object is the **joint realization**: a specified
ordered raw-neighbor curator, shared stored curator/learner features, fixed
binary labels, hidden candidates absent from every smaller-budget head, an
adjacent-budget ridge-output separation, and an exact sequential state under
the stated access contract. No inspected source was found to supply that
complete object. This qualified comparison does not establish priority.

## The three prior gaps

| Lead | New primary route and result | Remaining uncertainty |
| --- | --- | --- |
| GRACE, arXiv `2608.28361` | Author publication page links a readable 20-page manuscript. Main definitions, algorithm, evaluation, and limitations inspected. **Author-manuscript gap resolved.** | No byte identity with arXiv v1 established. The author's page lists forthcoming EMNLP 2026 Findings; it does not supply a verified manuscript revision date. |
| SSRN `7520178` | Author's official repository found; pinned release, exact queries, constructive decoder, and verification code inspected. **Exact-query gap narrowed.** | Full paper and formal approximate-success event still unavailable. Code is not a substitute for the missing theorem statement. |
| Dominici MSc thesis, 2025 | Author/lab pages and focused institutional/code searches inspected. **Still metadata only.** | No thesis or implementation obtained; no theorem, deletion semantics, or numerical result can be excluded or compared conclusively. |

### GRACE: distinguish the curator's role

[Bushipaka et al., author manuscript](https://retis.santannapisa.it/~tommaso/publications/EMNLP-2026.pdf),
linked by [Cucinotta's publication page](https://retis.santannapisa.it/~tommaso/papers/emnlp26.php),
formalizes retained-data retraining as a gold target (§3). Its algorithm
constructs forget/retain coresets from compressed gradients: direction-based
ranking, nonnegative orthogonal matching pursuit, and clustered retained
selection (§4, Algorithm 1). Experiments use behavioral forgetting/utility
proxies rather than strictly comparing with gold retraining (§5.4, Appendix
B.3). The author page lists EMNLP 2026 Findings, 24–26 October; that event is
future relative to this audit. The [official repository](https://github.com/dangeloandrea14/grace)
is linked in the manuscript.

This is direct prior work on coreset-assisted unlearning. Our question fixes
the original training curator and repairs its counterfactual admissions after
raw deletion. The inspected GRACE method does not characterize the memory
required to reproduce that fixed pipeline. Avoid “first coreset/curation
unlearning” wording. This comparison uses the author-hosted version, not an
unretrieved arXiv version.

### SSRN: exact reconstruction overlap is now concrete

[Tadakamalla, Surisetty and Asad](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7520178)
was written 24 September and posted 26 September 2026. Its
[official code](https://github.com/PranayTadakamalla/predictive-state-not-explanatory-state/tree/4143f408843394e21a0622c9eb5d5de10a687a8c)
declares software version 1.0.0, released 24 September.
At that commit, `queries.py` flips one event in a binary history and tests
every threshold; `decoder.py` reconstructs the history from the complete
answer tuple for length at least three. Thus the exact counterfactual family
really identifies all history bits. The README also explicitly uses joins
of query partitions to characterize sufficient state. These are substantial
conceptual overlaps with general task-relative memory framing.

The README and cached primary abstract claim the approximate lower bound
`t + log2(1−delta)`. Neither retrieved code nor README specifies its full
probability event. We therefore **do not** equate it with our expected squared
parameter-error condition, or assert it requires simultaneous rather than
per-query success. The complete paper remains a priority retrieval before
submission. No full PDF or TeX was present in the pinned repository tree.

### Dominici: unresolved, and consequential for broad framing

[The author's page](https://leodom01.github.io/) and
[DEEM's thesis list](https://deem.berlin/) identify *End-to-end Machine
Unlearning Through Data Preparation Pipelines and Feature Stores*.
The author dates graduation to 28 August 2025; this is not a verified
publication date. The page describes provenance and deletion propagation.
Fresh searches included the exact title, university domains, PDF variants,
and the author's code namespace. They yielded no thesis text. The UvA
repository route failed to load; the attempted GitHub user-list endpoint was
unsupported, and a guessed website repository tree returned 404. Those
failures say nothing about whether a public thesis or code exists.

Retain this as an unresolved close pipeline lead. Do not claim that end-to-end
preparation-aware unlearning is new, or that this thesis lacks our results.
We deliberately do not promote author-announcement performance figures into
verified findings.

## What the finite-format result adds

The inspected Phase 10 construction stores the same raw FP32 rows for cosine
curation and ridge learning. Public orthogonal blocks encode IDs/signatures;
`m` independent candidate blocks contain `D` private signs each. With
`k,D` powers of four and `2+k+D <= 2^20`, nonzero stored coordinates are exact
normal binary32 values. The scorer proof separately bounds the actual ordered
FP64 normalization/dot arithmetic and preserves strict decisions at the
stored threshold `float(0.4)`. The head target remains **exact rational ridge
on those stored rows**, not the output bytes of a floating-point optimizer.

For public positive rational regularization, a hard request exposes
`w_i / (16641/16384 + 2 lambda)`, so its private signs are identifiable.
The exact finite-alphabet obstruction is therefore `mD` private bits,
without continuity assumptions on the encoder. The sequential codec retains
only currently future-eligible sign blocks, derives their ordering from
public metadata, and targets byte equality with fresh initialization at the
actual remaining deletion horizon. Its contribution is a faithful witness
and scoped optimality statement for this family, not a general compressor.
The independent proof and codec reviews, rather than this literature audit,
determine correctness of the final bounds and implementation.

The separate certified-release bridge in `FINITE_WORD_THEORY.md` is useful:
a sufficiently small rigorous parameter-error radius preserves all private
signs, so exact rational outputs are not the only informative releases.
It is conditional on obtaining that certificate; it does not qualify an
existing optimizer. The fixed format and dimension cap also mean the finite
inequalities, not an unbounded-dimension asymptotic under FP32, are primary.

| Improvement | What it settles | What it does not settle |
| --- | --- | --- |
| Exact FP32 sign family | Information lower bound survives a fixed input alphabet. | Arbitrary finite precision, natural encoder outputs, or one independent bit per stored floating-point bit. |
| Frozen-scorer transfer | The graph used in the proof matches the specified arithmetic under its IEEE assumptions. | All hardware modes, fast-math/BLAS alternatives, or empirical hardware certification. |
| Actual ridge heads and binary labels | The private bits are visible in admissible model outputs without engineered private labels. | Fixed-test prediction risk, NLP loss, or task usefulness. |
| Canonical sequential codec | IDs-only logical updates can match fresh retained-state initialization on this family. | Physical erasure, transcript privacy, efficient rational decoding, or optimal total resident bytes. |
| Approximate information bound | Quantifies the stated expected parameter distortion. | A universal error metric, or a novel rate-distortion theorem. |

Public signatures, IDs, priority, deletion metadata, padding, headers,
initialization inputs, decoder workspace and returned rational heads must
remain separately charged. The initial `mD` payload and the current
`e(F)D` payload are not claims that the entire service consumes those bits.
The latter is a worst-case statement for each fixed admissible ledger `F`,
not an entropy claim conditional on an arbitrary adaptive request policy.
The reviewed `B`-bit convention permits at most `2^B` distinguishable values;
usable variable length or framing must also be charged. Exactness for each query and
approximate guarantees averaged over queries are different contracts.

## Classical machinery and safe final positioning

[Shannon's 1959 primary paper](https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf)
defines the distortion-constrained information minimum, gives product-source
arguments, and explicitly derives `1−H2(p)` for an equiprobable binary source
with Hamming distortion (reprint p. 340). General reproduction alphabets are
also discussed. The proposed squared-error scalar function is a specialization
that must be derived with its own reproduction alphabet; it must not be
silently replaced by the displayed Hamming formula. Covering, entropy,
randomized symmetrization, fixed-width bit packing and roundoff bounds remain
standard tools. None should be presented as newly invented here.

Recommended claim: **“For a fixed ordered semantic-suppression rule, we give
a shared-FP32-feature, binary-label ridge family whose exact bounded-future
repair requires `mD` private bits, and a canonical sequential representation
attaining that private-payload requirement under explicit public-state and
access contracts.”** Attach the precise scorer, budget, state and arithmetic
conditions from the reviewed theorem. Present approximation as a classical
information-theoretic corollary with an explicit parameter-error scale.

This is stronger than a cosmetic qualification: it answers two concrete
counterexamples to interpreting Phase 9 as a deployed finite-state claim.
It is still a worst-case theorem for the same research contribution. Practical
model relevance is only partial until genuine embeddings and retained-data
experiments show meaningful admissions, honest memory/runtime tradeoffs and
prediction consequences. Two of the three complete-paper gaps remain; the
SSRN gap is materially narrower. Refresh them and forward citations before
submission, without presenting an unsuccessful search as proof of absence.
