COUNTERFACTUAL SEMANTIC CURATION: FINALIZED SCOPED THEORY, VERSION 3
ACL 2027 research specification, 3 October 2026

MAIN ARTIFACT
counterfactual_curation_theory.tex -- Complete editable LaTeX source; PDF supplied separately.

IMPLEMENTATION AND VERIFICATION
verify_counterfactual_curation.py / curation_verification.json
  First-pass exhaustive coefficient, sequential-state, finite-field rank,
  activation, and numerical ridge checks.
algorithm_round2.py / round2_algorithm_verification.json
  Exact indexed canonical state repair with sparse tuple keys and stable handles.
  External counters verify sequence-wide bounds; rebuild and retry oracles.
round2_structure_verification.py / round2_structure_verification.json
  Reduced-signature rank, source incidence rank, source canonical states,
  path independence, and the SemDeDup threshold-equivalence boundary.
round2_lower_bounds_check.py / round2_lower_bounds_check.json
  Fixed cosine geometry and scalar/shared-embedding ridge targets in the
  constant-accuracy lower-bound construction.
round3_audit_check.py / round3_audit_check.json
  Adversarial generic-polynomial checks for the indexed update engine.
round3_compression.py / round3_compression_verification.json
  Sharp factor-two storage, current eligible metadata, and an exact rank-optimal
  canonical scanning implementation (GaugeState), including partition sources.
round3_ridge.py / round3_ridge_check.json
  Moment-only CG decoding, exact rational comparison, iteration bounds,
  residual/conditioning inequalities, and the average-loss count-shift example.
round3_lower_check.py / round3_lower_check.json
  Fixed-test prediction separation, ridge objective constants, and exhaustive
  small-code mask/permutation symmetrization.
round3_envelope.py / round3_envelope.json
  Sharp count-only curation envelopes and task-sensitive gradient bounds.

RUN FROM THE EXTRACTED DIRECTORY
python verify_counterfactual_curation.py --output curation_verification.json
python algorithm_round2.py --output round2_algorithm_verification.json
python round2_structure_verification.py
python round2_lower_bounds_check.py
python round3_audit_check.py
python round3_compression.py --output round3_compression_verification.json
python round3_ridge.py --output round3_ridge_check.json
python round3_lower_check.py
python round3_envelope.py

Python 3 and NumPy suffice. The indexed algorithm uses only the standard
library. The third-pass ridge and audit scripts import algorithm_round2.py;
keep the scripts in the same directory. Seeds are fixed. Runtime fields
are hardware-dependent. Recorded runs: Python 3.12.14 / NumPy 2.3.5.

COMPILE WITH A STANDARD LATEX INSTALLATION
pdflatex counterfactual_curation_theory.tex
pdflatex counterfactual_curation_theory.tex
Repeat if the contents or cross-reference warnings request it.

FINAL THIRD-PASS IMPROVEMENTS
1. The fast indexed map is within a sharp factor two of minimum linear statistic
   coordinates: M <= 2r. Its live keys/indices give O((s+h+1)*r + N_alive) words.
   A separately implemented scanning baseline uses rank-optimal statistic values
   with explicitly charged current eligible-record metadata and rebuilding.
2. Scalar ridge decoding uses current moments only. Exact zero-start CG needs at
   most rank(M) <= min(d,n) iterations, so model solving is now included in the
   total cost. A single added row generally causes a full-rank normalized ridge
   system change because lambda*n changes. Classical solver results are credited.
3. The lower bound now applies to prediction error on one fixed public test
   distribution. A matching information-theoretic squared-error curve and
   retained-objective excess-loss threshold are proved. Coding upper bounds are
   fixed-query public-randomness results, not efficient state-clean algorithms.
4. Given uniformly certified fixed blocker envelopes, three maintained counts
   give the sharp interval worst-case TV mismatch. Two extra count summaries
   maintain the certificate without actual refitting. Validity of the envelopes
   is a required certified input, not a free property of learned clustering.
5. Primary-source positioning includes DBSP, exact continual frozen-head ridge
   unlearning, adaptive history independence, and September 2026 retroactivity.
   Neither generic antijoin maintenance nor ordinary ridge updates are new.

SCOPE
The selector retains a record exactly when no earlier retained raw neighbor
blocks it. Embeddings, partition eligibility, threshold, and relative order are
fixed under restriction. The analyzed strict nonnegative-threshold suppression
step matches the SemDeDup code rule; it does not automatically reproduce
refitted clusters or priorities. It is not greedy independent-set selection.

The deletion budget is cumulative and decreases. Exact canonical state means
the designated abstract coefficient map, remaining membership/horizon, and live
indices modulo handle renaming. It does not mean literal equality of Python
heap layout, historical output deletion, or whole-system physical erasure.
The public ID universe distinguishes retries from invalid requests. Floating
point ridge checks do not replace a rigorous roundoff certificate.

Rank optimality concerns linear query coordinates with specified public decoder
metadata. The fast sparse implementation stores at most twice that number of value
coordinates. This is not an optimality theorem for constrained ridge moments
or total bits. The rank-optimal baseline trades metadata/rebuilding work for
fewer value coordinates. Both implementation contracts are specified in the PDF.
The memory lower bound counts every hidden-label-dependent resource and allows
no retained-label reread or free data-dependent ticket. The fixed-geometry
upper/lower comparison concerns hidden-label information, not total metadata.

IMPLEMENTATION NOTES
The canonical indexed engine uses exact integer test statistics. The ridge
verification uses exact rational arithmetic and floating demonstrations; its
explicit system matrix takes O(d^2) workspace. The theoretical matrix-view CG
variant can use O(d) workspace, but is not misrepresented as the demonstration.
The gauge baseline uses sorted current blocker tuples, interned integer DSU
handles, and one-pass component minima. Canonical snapshot sorting/export is
excluded from online update bounds. Expected hashing remains a model assumption.
The coding checks use a small exhaustive code, not an efficient general codec.

EVIDENCE LIMITS
All recorded checks passed. Finite checks supplement the written proofs; they
are not a universal proof, novelty certification, natural-language experiment,
neural unlearning evaluation, or demonstration of useful end-to-end speedup.
The theory specification is closed for the stated core. NLP relevance and
practical advantage still require the planned corpus experiments.
