OFFLINE THRESHOLD CALIBRATION
============================

Status: executable preparation and fail-closed analysis, not completed human
annotation. The packaged Civil100 lexical assignment file is a plumbing artifact;
it supplies neither semantic embeddings nor a valid semantic threshold.

API (import phase3.calibration with empirical_execution on sys.path)
----------------------------------------------------------------
prepare_selection(vectors_fp32, records, provenance, seed=20271003, block_size=256)
    Returns a private, SHA256-sealed manifest. records require record_id and text;
    confirmatory_calibration additionally requires partition="calibration" and
    768 columns. Input provenance includes dataset_id, encoder_id,
    encoder_revision, population_scope, evidence_role. evidence_role is either
    engineering_nonconfirmatory or confirmatory_calibration. It is a declaration,
    not verification of source authenticity or the embedding derivation.

write_blinded_pack(manifest, records, directory)
    Creates private_sampling_manifest.json, blinded_assignments.json, and an
    EMPTY responses.template.json. Refuses overwrite. Nothing is distributed or
    sent. Give annotators only assigned text pairs, never the private manifest.

select_threshold(selection_manifest, completed_response_bundle)
    Requires all three assignments for every sampled pair, three distinct rater
    IDs, nonempty unique response IDs, completion markers, six categories and the
    matching collapsed labels, plus an actual human collection declaration.
    Uncompleted templates fail. Software cannot authenticate humans by file fields.
    Returns a sealed selection lock with the complete candidate ledger. If no
    candidate qualifies, tau=.995 is diagnostic and selection remains failed.

prepare_validation(vectors, records, selection_lock, seed=20271004, block_size=256)
    Requires the frozen lock and identical vectors/records/scoring implementation.
    Independently samples up to200 from the entire strict-above-tau population.
    Previously sampled pairs remain eligible and get new pair/assignment IDs.
    The block size and both code hashes are retained for replay.

quality_gate(validation_manifest, completed_response_bundle, selection_lock)
    Pure function; rejects missing or mismatched responses. Computes the exact
    finite-population one-sided95% lower bound. A primary gate can pass only when
    selection qualified and the lower bound is at least.90. No valid pair population
    means precision is undefined. Failure cannot be rescued by retuning the primary
    threshold. Overall confirmation still requires independently verified source,
    embedding, annotation and panel provenance outside this module.

Operational clarifications (prospectively locked; not claims of prior specificity)
-------------------------------------------------------------------------------
* Retain all six categories from the original narrative protocol. exact_copy and
  substantially_same_meaning are positive. overlapping_information, merely_related,
  unrelated and uncertain are nonpositive. Pair success is at least2 of3 positive
  independent judgments. uncertain maps to legacy "unsure", never a positive.
* Selection estimates precision as the ratio of Horvitz-Thompson weighted positive
  counts to weighted sampled above-threshold counts. The known bin population and
  exact n_bin/N_bin inclusion probability are logged. Raw and Kish effective
  support accompany the estimate; neither is an independent quality certificate.
* Twenty bins use nearest-FP64 versions of decimal edges -1,-.9,...,1; left closed,
  right open, except the final right endpoint. Raw computed scores beyond endpoints
  because of rounding saturate the bin index, but are NOT clipped for thresholds.
* All pair scores use the shared panels.graph_normalize/reference_cosines functions:
  promote frozenFP32 toFP64; accumulate norms and products in coordinate order.
  Working score memory is O(BD+B^2), not a fullN-by-N matrix. It still performs
  O(N^2D) arithmetic. Stable scoring across tile sizes is checked; this implementation
  favors a well-specified numerical target and makes no scale-speed claim.
* Scores within1e-12 of any grid threshold receive independent math.fsum accumulation
  over the already normalized FP64 products. Disagreements are logged; the ordered
  reference decides. This is a numerical diagnostic, not exact-real cosine proof.
* Reservoir AlgorithmR implements SRS without replacement within each selection bin
  and once over the validation population. Recorded seeded PythonMT19937 streams
  are domain-separated by stage. SRS guarantees refer to ideal uniform random draws
  in the algorithmic model, not a claim of physical randomness or cryptographic RNG.

Exact statistical calculation
----------------------------
For populationN, sample n and x majority positives, let
  T(K)=sum_{j>=x} comb(K,j)*comb(N-K,n-j) / comb(N,n).
The lower success count is the smallest integerK with T(K)>.05. The gate compares
10*K>=9*N using integers, with no normal approximation or floating probability.
Integer binomial arithmetic and monotone binary search implement inversion. Census
returns exactly x/N. No sample/population pairs gives undefined precision.

The coverage target is a finite set of fixed or potential judgments under the
prespecified three-rater protocol, conditional on rater assignment being independent
of pair sampling. It is not latent semantic truth, and this interval does not cover
additional annotator noise or changed annotation instructions. The outside study
must verify actual collection, blinding, independence, consent/pay and provenance;
turning declaration fields to true without collection is not an admissible action.

Engineering validation
----------------------
python3 empirical_execution/phase3/check_calibration.py
    Uses the existing100 natural Civil Comments records and frozen lexical features.
    Checks4950 distinct pairs, deterministic selection, exact inclusion probabilities,
    no duplicate pairs, tile-size invariant scores, new validation assignment IDs,
    and refusal of missing judgments. Creates no human labels. Reservoir software
    fixtures exhaustively enumerate random choices on small integer IDs; these are
    not synthetic empirical datasets. The independent audit separately checks exact
    bound inversion and frequentist coverage by exhaustive finite combinatorics.

The engineering pack intentionally contains blank responses. It is not the actual
semantic study's annotation pack, and human completion of it would not turn lexical
features or reused development records into confirmatory E5 evidence.
