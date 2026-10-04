FROZEN REQUEST MANIFESTS — PHASE 4, 4 OCTOBER 2026

Scope
-----
requests.py freezes the original protocol's R, S, U, A and matched-R workloads
for a supplied, fixed earlier-raw-neighbor blocker graph. It accepts no target
labels, task losses, predictions or model gradients. No record is excluded
because its target is zero, empty/multihot-zero, inconvenient, or uninformative.
The only record universe is exactly the graph's input IDs. Callers must finish
the independent population, semantic calibration and task configuration gates
before making any confirmatory use of a request manifest.

No source authenticity is inferred from a source-kind string. The input mapping
is only a declaration. Every manifest explicitly says that this module has not
verified external provenance and that the confirmatory study is not ready.
The Civil-100 preview has unknown singleton sources only. Its S and matched-S
initializations are retained as unavailable/zero-release statuses, not evidence
about real article, publisher or annotator withdrawal.

API
---
generate_manifest(graph, *, dataset_id, panel_id, source_kinds, design=None,
                  allocations=None, evidence_role='engineering_nonconfirmatory',
                  master_seed=20271003, record_horizon=None, source_horizon=None,
                  include_excluded_blocker_stress=True, timing_sink=None)

graph is a ccu.core.BlockerGraph. Build it with phase3.reference_graph for the
prospective common pair-local FP64 scoring rule. The generator also accepts a
mathematical adjacency fixture for software checks; that is not an experiment.

source_kinds maps EVERY graph source ID to one of:
  genuine_native: declared original source group, eligible for S sampling;
  unknown_singleton: exactly one owned record; excluded from S sampling;
  engineering_proxy: nonnative engineering grouping, excluded from S sampling.
The full source service universe always includes all three categories. Source
membership is a disjoint partition supplied by the graph; whole groups are
expanded within the already fixed post-guard panel. The code cannot establish
that a panel contains every original snapshot record of a source.

design uses the original study_design.json structure. Only requests/seeds are
read for request construction; its entire supplied JSON value is hash-bound.
Defaults are R=256, S=256, U=128, A=32, record horizon 128 with checkpoints
1/8/32/128, source horizon 8 with checkpoints 1/2/4/8. A-excluded-blockers has
the same number of paths as A and is a separate stress sensitivity. Other blocks
take the first stable trajectory IDs from these paths; do not regenerate a new
sample because one method finishes earlier or one outcome looks interesting.

R: one uniform partial permutation of all stable record IDs, equivalent in law
to the required full-permutation prefixes. State resets only between independent
trajectory IDs. Prefix checkpoints are cumulative and unique.

S: one uniform partial permutation of declared genuine native sources. All
records owned by each requested source are removed, including initially excluded
records. Unknown sources remain in the service universe and may survive after
all genuine sources are requested. Genuine source size and complete source size
quantiles are reported with a fixed inverse empirical-CDF convention.

U: one permutation of the initially excluded records, defined once before any
deletion. The feasible initial horizon is min(requested horizon, excluded count).
The eligible-state universe remains ALL records, not the excluded sampling pool.
An empty excluded pool has an explicit status and no release. This matches the
earlier helper's clipping semantics and fixes the reading used prospectively.

A: sample at most 1,024 initially excluded candidates without replacement whose
full blocker sets fit the record horizon; census smaller candidate pools. Score
the actual unpadded blocker deletion set by admissions, breaking ties by stable
candidate ID. The entire sampled candidate list, blocker set and admission score
are logged. Identical blocker sets can reuse an exact graph-only score. Delete
the winning blockers in sorted stable-ID order, then append one random order of
remaining records until the horizon. Padding can delete the targeted candidate;
therefore unpadded scores are distinct from reported checkpoint outcomes.
Failure to find a candidate is retained and uses ordinary random padding only.
This procedure is a sampled graph stress test, not an optimal adversary.

A-excluded-blockers: restrict only the candidate's initial blocker set to the
initially excluded pool. Padding still ranges over all records. The metadata
states this explicitly; it is not an excluded-only trajectory beyond that prefix.
U supplies the complete excluded-only workload. This is a prospective resolution
of the narrative's unspecified padding convention, consistent with the previous
engineering helper, and is not a new prevalence arm.

R-volume-matched-to-S: one independently seeded all-record permutation per S
trajectory. Checkpoints equal that S trajectory's cumulative RAW record counts.
These are nested prefixes, never separately sampled subsets. Its separately
declared record horizon is the last source raw count, which may exceed 128.
Missing genuine sources produce explicit zero-release matched-control statuses.

Determinism, hashes and numerical scope
--------------------------------------
The master seed is 20271003 unless explicitly amended. A stream key hashes
length-prefixed UTF-8 (master, request salt, dataset ID, panel ID, arm, 3-digit
trajectory index), taking the first 128 SHA-256 bits. Frames are sorted by stable
ID. Python's exact version is recorded because random.sample is an implementation
dependency. The graph hash is bound to the manifest but not to the RNG key:
R/S/matched-R requests consequently remain paired across curator/threshold/
priority alternatives on the same panel and source partition. U/A share stream
keys but their frames or objectives can differ as the graph changes. They are
not thereby identical request sets. The existing phase-2 results are unchanged.

The manifest binds all graph IDs, priority, complete edge relation, FP64 threshold
hex representation, source partition, source-kind declaration, code files,
configuration and supplied design. verify_manifest checks its content seal and,
if a graph is supplied, exact graph identity. A hash verifies integrity, not the
truth of source provenance or the quality of semantic edges. A graph constructed
by an unrelated implementation can be supplied; the metadata never certifies
how those edges were originally generated. Main-study use separately requires
the acquisition/encoder/reference-scoring locks.

Wall-clock candidate-search timings are appended to timing_sink, outside the
deterministic manifest seal. They include the implementation's inverse incidence
construction and candidate scoring. Request preparation and graph/audit work are
shared preparation costs, not repair-worker service speed. No speedup is claimed.

Every path has trajectory_id, arm, unit, seed, initial_horizon, deletion_order,
checkpoints and observations. Observations include all zeros and empty selected
targets, exact cumulative deletions, remaining budget, admissions, originally
selected deletions, current selected count and membership. The field
previous_release_selected_deleted counts deletions of records selected at the
prior RELEASE; it is not the sum of singleton intermediate removal counts.

The manifest contains graph-only oracle diagnostics. NEVER send this whole object
to the identifier-only repair worker. The parent runner must pass only each
incremental request's identifiers and the declared service contract. Diagnoses,
selected/admitted IDs and the benchmark graph stay with the isolated auditor.

write_manifest creates a new path exclusively and refuses overwrite. Distinct
ordered paths and distinct cumulative request sets are counted separately. In a
small finite population many paths can coincide or share terminal sets; no such
case is resampled away or presented as another independent corpus.

Checks and limits
-----------------
Run from the repository root:
  python empirical_execution/phase4/check_requests.py

The check uses the existing natural Civil-100 texts with 128-dimensional lexical
engineering features at tau=0.6. These are software/development checks, not E5,
semantic precision, primary-scale prevalence, or source-withdrawal evidence.
The primary allocation sizes are exercised only to check exact generation and
retention of all statuses. Independently calculated scalar adjacency predicates
verify each checkpoint and every sampled A candidate. Genuine-source expansion,
matched-R raw counts, unknown-source retention and own-source blocking are tested
on an explicitly named four-vertex mathematical fixture, not a synthetic corpus.
An empty graph and an edgeless graph test zero/failure behavior. No human ratings
or task labels are invented. Tests also reject tampering, a wrong graph and
overwriting a frozen file; reproducible generation and paired R sensitivity
requests are checked.

Structure-only extensions (implemented separately)
-------------------------------------------------
requests_extensions.py completes the separate mass-proportional source sampler
and ceil(0.01*N) record-horizon preparation. It does not modify frozen requests.py.

generate_structure_extensions(graph, *, dataset_id, panel_id, source_kinds,
                              design=None, master_seed=20271003,
                              mass_source_paths=256, source_horizon=8,
                              one_percent_allocations=None,
                              evidence_role='engineering_nonconfirmatory',
                              timing_sink=None)

S-mass draws whole declared genuine source groups sequentially without replacement,
each next source with probability equal to its initial post-guard panel record
count divided by the remaining source record count. Integer uniform randrange
draws avoid floating probability/key ambiguities. Every draw and its exact
conditional probability numerator/denominator is logged. Those step probabilities
are not a claim that final k-source inclusion probabilities remain proportional
to source size. Fallback singletons and proxy groups never enter this sampler.
This distribution is distinct from uniform S: never pool the two or apply the
uniform-source admission formula to S-mass. As with core S, all groups in the
ownership partition remain in the source SERVICE universe.

The one-percent record branch uses the exact integer horizon (N+99)//100. This
can be smaller than 128 for small panels and larger for large panels; its name
does not promise a larger budget everywhere. It produces a separately sealed
child core manifest for R/U/A (defaults 256/128/32). A-excluded-blockers and source
paths are absent from this particular extension. Checkpoints clip 1/8/32/128
and additionally include the one-percent final horizon, so a 200k panel actually
reaches 2,000 deletions. U still clips to its initially excluded pool.

These are separate service-state constructions with their own initial horizons,
construction costs, supported future-request sets and lifecycle costs. Never
append them to an exhausted core state or reset a core budget implicitly. A
distinct request salt separates this branch from the core; there is no claimed
cross-horizon paired trajectory. Within either branch, seeds exclude graph edges
and preserve shared R/S-mass randomness across same-panel curator alternatives.
The source-mass and one-percent namespaces also differ from each other.

verify_structure_extensions checks its top-level seal, child request seal and
optional bound graph. It still never authenticates source evidence, passes a
semantic gate, or declares the confirmatory study ready. Its graph diagnostics
remain isolated-auditor data, with the same worker access restriction as above.

Run the small extension verification:
  python empirical_execution/phase4/check_requests_extensions.py

This runs only 10 actual lexical Civil-100 one-percent checkpoints (K=1), keeping
four empty S-mass initializations because real source metadata is unavailable.
Exact sampler correctness is separately checked over every integer draw trace
for a three-weight algebra fixture; all six ordered two-source probabilities
match the sequential PPS formula as exact rational numbers. Mathematical fixtures
also test complete source-group expansion, retained unknown records, integer
ceil boundaries including K=129, paired curator sensitivity streams and seal
rejection. These fixtures are software mathematics, not source-withdrawal data.

Descriptive news chronology and branch-specific full-refit requests still require
their actual dated corpus and a pinned SemDeDup branch state. No core request
manifest is substituted for those missing inputs.
