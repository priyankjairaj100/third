# Contribution position after the novelty audit

Date: 4 October 2026.
This note separates the proposed contribution from established tools.
It does not claim exhaustive novelty or publication acceptance.
Read `THEORY_EXTENSION.md`, `THEORY_REVIEW.md`, and `NOVELTY_AUDIT.md` together.

## Main paper claim

Future deletions can require information that the original trained model never used.
We characterize this requirement for an explicit ordered cosine curator.
We give exact repair summaries under a cumulative deletion horizon.
We show that actual ridge outputs can force feature-scale memory.
The natural-data study must establish whether this requirement matters in NLP.

The main target remains `A(C(D minus F))`.
The curator uses all earlier raw neighbors.
It does not use only previously selected neighbors.
Frozen features, priorities, and the strict threshold define its contract.
Corpus-fitted curator refitting remains a separate branch.

## Stronger memory theorem

Let `m` denote hidden retained candidates.
Let `d` denote the common curation and learner feature dimension.
Let `q` denote the response dimension.
The construction requires enough dimensions for its public signatures.
Logarithmic dimension in `m` suffices; arbitrary `m` at fixed dimension is not asserted.

Every original head is zero.
Every target head through deletion budget `b` is also zero.
All candidate features vary within small unit-sphere caps.
Their strict cosine graph remains identical across the construction.
All forgotten blocker payloads remain fixed and public.
Candidate responses remain coordinatewise bounded.

At budget `b+1`, designated requests expose one candidate each.
The resulting ridge head determines that candidate's feature direction and response parameters.
Thus exact continuous query summaries require exactly

\[
                 m(d-1+q)
\]

private real coordinates on the stated family.
A separate packing argument gives a randomized finite-bit lower bound.
At the specified fixed error scale, this bound is `Omega(m(d+q))`.
The error scale depends on regularization and the fixed cap radius.
It can be small.

This closes an important gap in the earlier memory argument.
Arbitrary independent statistics need not correspond to realizable ridge moments.
The new proof uses actual ridge optimizers throughout.
It also prevents forgotten payloads from revealing the hidden features.

A separate corollary uses only binary labels.
Every hidden candidate has the same public label, one.
All other labels are zero.
Only hidden features vary.
The exact continuous requirement remains `m(d-1)` coordinates.
A separate finite-bit packing bound also remains.
Thus the feature-memory obstruction does not depend on private continuous labels.

For fixed `b`, eligible population size is proportional to `m`.
The lower bound matches the existing `O(E(d+q))` numerical-coordinate order in a worst-case sense.
The matching chart upper bound answers counterfactual queries from the initial state.
It is not a canonical sequential erasure algorithm.
The implemented canonical state has its own stronger contract and costs.

## Contribution boundaries

| Component | Defensible role | Boundary |
| --- | --- | --- |
| Exact ridge addition and deletion | Established learner machinery | Do not claim this operation as new. |
| Nonmonotone incremental maintenance | Established general framework | Deletion-induced additions alone are insufficient novelty. |
| Curation-specific activation signatures | Structural specialization | State the exact raw-neighbor rule and finite horizon. |
| Sparse polynomial repair | Algorithmic contribution under the stated access contract | Compare against incremental maintenance and compact eligible payload. |
| Canonical exact state | Strong numerical and history contract | Excludes physical erasure, transcript privacy, and arbitrary refitted curators. |
| Feature-scale ridge memory lower bound | New strengthening within this project | Exact continuous coordinates and approximate finite bits are separate claims. |
| Natural semantic study | Required evidence for NLP importance | No primary semantic result exists yet. |

The proof tools are classical.
These include Sherman–Morrison, invariance of domain, volumetric packing, and Fano's inequality.
General matroid-union rank results also have direct predecessors.
None should be presented as newly discovered mathematics.

## Evidence needed for the paper

The existing prospective registry remains unchanged.
This theory update does not justify selecting favorable trajectories after observing results.
It also does not justify suppressing zero admissions or failures.

| Scientific question | Existing study obligation | Decision consequence |
| --- | --- | --- |
| Do real deletions admit excluded records? | Structural panels, source requests, budget curves, and zero mass | Rare admissions weaken the broad practical motivation. |
| Do admissions preserve useful language information? | Blinded relevance and complete context audits | Near-identical replacements weaken the NLP claim. |
| Does the changed target affect predictions? | Signed task effects against full counterfactual retraining | Negligible effects narrow the paper's practical claims. |
| Does the repair method help under full costs? | Persistent bytes, peak memory, lifecycle time, access contracts | Compact eligible payload can remain the practical winner. |
| Does fixed-curator theory describe refitted pipelines? | Official full-refit branch and mismatch measurements | Large mismatch limits deployment scope. |

The lower bound predicts a possible information burden.
It does not predict its prevalence in natural text.
It does not prove practical compression or speedup.
Keep the existing unfavorable compact-payload comparison visible.

## Manuscript order

1. Define the full counterfactual target and the concrete curator.
2. Derive the activation rule and deletion-induced admissions.
3. Present the bounded-horizon repair algorithm and exact state contract.
4. State the realizable ridge memory theorem and its access assumptions.
5. Report accepted natural-data evidence, including zeros and negative findings.
6. Explain total costs, refitting limits, and unresolved scope.

The novelty audit records the closest verified sources and unresolved search leads.
Resolve those leads before the final submission claim.
Do not describe this checkpoint as a completed ACL paper.
