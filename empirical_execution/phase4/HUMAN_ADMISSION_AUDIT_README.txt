HUMAN ADMISSION AUDIT — PAIR PACK PREPARATION ONLY
Prospective implementation choices fixed 4 October 2026, before any primary frame.

admission_audit.py implements the pair-sampling portion of the original protocol's
semantic information audit. It creates a private sampling manifest, blinded text
assignments, blank responses and instructions. It produces no human judgments,
contacts nobody and never marks annotation dispatch or the confirmatory study ready.
This audit is separate from threshold calibration; calibration judgments must not
be silently reused as admission judgments.

API
---
prepare_admission_audit(corpus_inputs, *, master_seed=20271003,
                       evidence_role='engineering_nonconfirmatory')

corpus_inputs keys are civil_comments, askubuntu and cc_news. Missing corpora
remain missing with all their quotas unfilled; there is no replacement corpus.
Each supplied corpus has:
  records: original nonempty text and record_id in graph/cache input order;
  features: aligned fixed FP32 representation;
  graph: fixed earlier-raw-neighbor-suppression BlockerGraph;
  request_manifest: sealed core request manifest for that graph;
  existing_labels: optional original target map, never invented or imputed;
  provenance: corpus_snapshot, representation_id, representation_revision,
              evidence_role declarations.

The parent must select one intended corpus/panel/curator configuration. This API
does not mix different graphs or repeated size panels into an unlabeled population.
Graph/frame/text/label/code/configuration hashes bind the resulting private file.
None of those hashes authenticates corpus provenance or validates semantic quality.

Frozen population and observational control definition
-----------------------------------------------------
The sampling frame is the COMPLETE union of admissions at every declared logged
checkpoint of the supplied R/S/U/A trajectories. The code independently derives
admitted IDs from the graph and deleted prefix, checking the supplied diagnostics.
Matched-volume, excluded-blocker-only and other secondary arms are not mixed into
this union; their ignored path counts are disclosed. It records all path/checkpoint
occurrences and per-arm multiplicities privately, including repeated appearances
of one record. Those repetitions never create independent human observations.

For each eligible initially excluded record, select one former blocker uniformly
from its complete stable-ID-ordered original blocker list using a public independent
seed. This choice is fixed across every occurrence. A former blocker need not have
been selected for training; the curator suppresses using all earlier raw neighbors.

The observational control population is fixed as initially excluded records that
are retained and still excluded at one or more logged core checkpoints, and are
NEVER admitted anywhere in this same finite checkpoint union. One former blocker
is selected by the identical rule. Thus control records and admitted records are
disjoint for this observed frame. This is not a randomized causal control, and
"never admitted" does not extend to other deletions or unobserved intermediate steps.

Unique admitted-record/former-blocker pairs are the human units. An admitted pair
can belong to both R/S and U/A strata. It still receives only one set of three
independent ratings if selected by either or both sampling routes. Controls cannot
overlap admitted pairs under the above definition. Pair multiplicities remain
available for precisely defined secondary summaries but are not duplicated ratings.

Prospective quotas and strata
-----------------------------
The near-equal fixed corpus quotas are:
  Civil:       134 admissions (67 R/S + 67 U/A), 67 controls.
  Ask Ubuntu:  133 admissions (67 R/S + 66 U/A), 67 controls.
  CC-News:     133 admissions (67 R/S + 66 U/A), 66 controls.
Totals are at most 400 admission pairs and 200 control pairs. Integer remainders
are assigned in this fixed corpus/role order, independent of outcomes.

Four similarity bands use the cosine excess relative to the fixed threshold:
  m = (score - tau)/(1 - tau)
  [0,.25), [.25,.5), [.5,.75), [.75,infinity).
The upper band permits harmless FP64 score overshoot above one. A former-blocker
score must satisfy the same strict reference score > tau predicate and tau < 1.
The code computes pair scores with the shared phase-3 coordinate-order reference.
Relative bands avoid a threshold above .95 putting every pair in the single top
absolute-cosine band. Band boundaries never move in response to the observed frame.

Civil and Ask Ubuntu have three existing-target relation cells: exactly equal,
unequal, or unavailable. Civil uses original toxicity fractions without inventing
a binary toxicity class. Ask Ubuntu compares the already frozen original-tag
multihot target vectors; all-zero vectors are valid. Missing targets are explicitly
unavailable, never labeled agreement/disagreement. News has only the unavailable
cell, because it has no invented task labels. Label disagreement is a sampling
stratum, never a human semantic truth label.

Each corpus-role quota is divided as equally as possible across its four bands
times relation cells, assigning remainders in ascending band then equal/unequal/
unavailable order. News therefore has four cells, labeled tasks twelve. Every cell
samples min(quota, population) unique pairs uniformly without replacement, or takes
a census. Missing/empty/small cells retain their shortfalls. No cell, role or corpus
borrows unused quota. This may yield substantially fewer than 600 total pairs.

Probability interpretation
--------------------------
The complete private frame lists every eligible pair and every sampling stratum.
For a pair belonging to stratum h, n_h/N_h is its conditional SRS inclusion
probability. Independent stratum samples can overlap across R/S and U/A. Its
probability of appearing at least once in the deduplicated human pack is
  pi(pair) = 1 - product_h [1 - n_h/N_h].
The manifest records this exact rational number, alongside each stratum's sample
and frame. A selected pair is annotated once; do not attach separate independent
outcomes to duplicate stratum selections. The fixed former-blocker draw probability
1/|B(record)| is recorded separately and is not silently multiplied into a claimed
marginal inclusion probability over all possible blocker pairs.

These sampling probabilities condition on the DECLARED FROZEN CHECKPOINT UNION and
the REALIZED FIXED FORMER-BLOCKER CHOICES. They support estimates about that finite
pair frame after genuine ratings exist. They do not estimate all possible deletion
requests, all original blocker pairs, all NLP corpora, or semantic novelty relative
to all retained texts. An equal-corpus summary would additionally need to declare
its balanced target population and account for absent corpora. No result estimator
or judgment is fabricated by this preparation module.

Blinding and collection
-----------------------
write_admission_pack(manifest, records_by_corpus, directory) creates a NEW directory
and refuses overwrite. Every selected unique pair receives three distinct blank
assignment IDs. IDs are bound to the full frozen frame/input context, so changing
text under the same raw record ID does not silently reuse old assignment IDs.
Text orientation is independently masked. The exposed fields are only opaque
assignment/pair IDs, text A, text B and the task definition. Methods, request arms,
gold labels, similarities, model outcomes, record IDs and sampling multiplicities
remain private. The collector must assign three distinct fluent humans independently.

Questions separately cover meaning relation, a consequential entity/number/
negation/time/assertion distinction, and possible task relevance. Responses start
blank with human_completed=false. No majority labels, AI judgments, completed
responses or quality result are produced. Instructions include content warnings
and a skip option. Preserve skips, uncertainty and disagreement; do not replace
skipped pairs or impute ratings. Adjudication follows independent collection.

Before dispatch, the actual study still needs annotator recruitment/qualifications,
payment and estimated duration, consent/withdrawal procedures, content protections
and permissions for any protected text. These are not inferred from source hashes.
The files are local preparation only. A public release should contain permitted
annotations and IDs/short permitted excerpts, not publisher article bodies.

The two 100-admission contextual audits remain explicitly pending:
  1. Complete former-blocker sets, with a fixed context limit and logged truncation.
  2. Nearest surviving SELECTED text at a prespecified actual retained checkpoint,
     with stable ties and an independently fixed record/checkpoint subset.
Pair judgments alone do not complete either contextual audit or prove global novelty.

Verification
------------
Run: python empirical_execution/phase4/check_admission_audit.py

The available Civil-100 natural texts, original toxicity fractions and lexical
engineering vectors exercise 20 declared R/U/A paths. The observed frame has
10 unique admitted records and 23 disjoint observational controls; the frozen
strata select 24 unique pairs, giving 72 BLANK assignments. Missing Ask Ubuntu
and News quotas remain unfilled. These counts are software preparation results,
not a semantic experiment or human conclusion.

An independent scalar graph calculation checks the complete frame, every observed
multiplicity and all controls. Rational arithmetic checks conditional union
probabilities; a tiny combinatorial sample-space calculation verifies the overlap
formula independently. Other checks cover quota sums, no refill, unavailable-label
handling, text-only blinding fields, three blanks per unique pair, tampering and
overwrite rejection. No synthetic empirical corpus or completed rating is created.
