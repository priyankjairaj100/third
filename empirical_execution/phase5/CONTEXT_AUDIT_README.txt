CONTEXTUAL HUMAN ADMISSION AUDITS — TWO BLANK LOCAL PREPARATION WORKFLOWS

The frozen Phase 4 pair audit remains unchanged. context_audit.py implements
the two additional protocol audits: complete former-blocker neighborhood and
nearest surviving selected text. No annotator was contacted, no rating exists,
and the selected documents have not thereby been shown to contain novel meaning.

Frame and prospective sampling conventions
-----------------------------------------
For each supplied Civil, Ask Ubuntu or News corpus, start with the COMPLETE
union of unique admissions at every declared core R/S/U/A logged checkpoint.
Secondary matched-volume/stress arms are not silently included. Independently
recompute selected sets from each deletion prefix, preserve repeated occurrences
as private metadata, and count each admitted record once per audit kind.
This frame is conditional on one declared panel/curator/request bundle, not all
possible deletions, all panels or all corpora.

Before reading semantic outcomes, allocate100 slots per audit as34Civil,
33AskUbuntu and33News. This near-equal deterministic corpus allocation completes
the previously unspecified rounding convention. Each corpus/kind uses its own
seeded SRSWOR stream. Take min(quota,N); preserve shortfalls and absent corpora.
There is no cross-corpus refill. The two subsets are independently selected,
may overlap, and remain separate contextual questions with separate assignments.
The complete private frame includes every eligible record and all occurrences.
Each record has exact inclusion probability n/N and Horvitz-Thompson weight N/n
conditional on this frame. Census probability is one. These do not by themselves
justify a global all-corpus average when corpora are absent.

For each record and audit kind, choose one actual admitted occurrence uniformly
with an independent fixed seed from its stable ordered occurrence list. Record
its conditional1/multiplicity probability and actual request/checkpoint privately.
Weights for finite-record estimands condition on these realized context choices.
Occurrence-level inference requires an explicitly declared target; repeated
occurrences are not independent human observations. Neither context choice nor
record choice depends on labels, model utility, nearest score or ratings.

Actual context
--------------
Former-neighborhood audit uses ALL original earlier raw neighbors B(v), including
blockers that never trained the model. At the chosen admission checkpoint every
one is removed. The private frame lists the complete set before context limits.
This is not the complete original SELECTED corpus, so a positive judgment cannot
be presented as novelty relative to everything ever trained.

Nearest-selected audit exhaustively scores all surviving selected records at the
chosen retained checkpoint EXCLUDING THE ADMITTED QUERY ITSELF. It uses the frozen
ordered-coordinate finite-precision reference scorer, not an ANN/top-K proxy.
Pick largest cosine, with lexicographically smallest stable ID on exact computed
score ties. Save the exact selected candidate population, score and tie policy
privately. No surviving comparator means an explicit missing comparison, never
a fabricated text or an automatic novelty label. One nearest text is not a proof
about all surviving information, and it does not imply corpus recovery.

Context computations occur only AFTER subset sampling. Checkpoint populations
are shared once per distinct checkpoint, rather than duplicated in every frame
row. Unselected units remain in the complete weighted frame without computing
their nearest neighbor. Exact nearest scanning is bounded working memory for
scores, with total selected-record comparisons charged conceptually to the audit;
no primary systems speed or peak-RAM claim is made for annotation preparation.

Limits and missingness
---------------------
Defaults fixed prospectively:8000Unicode code points for the target and20000
for ALL comparison texts combined. These are characters, not model tokens.
Former blockers are placed in a fixed hash order independent of labels/scores.
Take prefix characters up to the total budget; log complete IDs, original and
displayed lengths and every partially/fully omitted context. The assignment
explicitly says whether target/context is complete. Complete former-blocker
meaning can only be judged when the complete set and target were displayed;
truncated judgments concern the displayed context and require separate reporting.
Budget changes do not resample records, replace contexts or move quotas.

The three distinct assignment slots per chosen audit unit are BLANK. Exposed
fields contain only opaque IDs, original displayed text, task/question, and
context/comparator availability indicators. Arms, method, labels, model outcomes,
raw record/source IDs, similarities, sampling weights and private multiplicities
are withheld. Task definitions retain toxicity, technical-tag relevance or factual
assertions. Responses separately ask distinct information, consequential entity/
number/negation/time/assertion difference, and possible task relevance.

Preserve skips, uncertainty, incomplete context and missing comparator. Do not
replace or impute. Report weighted response/skip/truncation rates; complete-case
novelty fractions alone do not estimate the full frame. With genuine binary
outcomes, an explicit lower/upper analysis can assign all missing outcomes0/1,
while retaining the original sampling weights; no outcome analysis is performed
here. Selection probability corrects sampling, not voluntary skip bias or the
latent semantic truth of human judgments.

Before dispatch, finalize actual recruitment/fluency, independent assignment,
payment/duration, consent/withdrawal, content protections and text permissions.
No such completed process is fabricated. News article bodies must not enter a
public backup. The pack writer is local preparation, not authorization to contact
humans. Existing calibration judgments must not silently replace contextual ones.

Reproduction and release
------------------------
 python3 empirical_execution/phase5/check_context_audit.py \
   --output-dir /absolute/path/to/new/output

results/context_audit_engineering/ uses the existing Civil100 natural texts and
lexical reference graph. It is an engineering census where fewer than100 actual
admissions are available. Missing AskUbuntu/News quotas remain unfilled. Tests
independently reconstruct the complete admission frame, every former blocker
set and selected population, compare nearest choices to a scalar oracle, and
check blank fields, fixed quotas, exact inclusion probabilities, truncation,
tampering, overwrite and engineering-to-primary relabel refusal. A three-vector
exact-tie control is purely a software fixture, never empirical evidence.
