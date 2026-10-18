Third: Stage1 development structural evidence (public aggregate bundle)

This bundle preserves every one of the 51 original score cutoffs: 0.50 through
0.99 in steps of 0.01, plus 0.995. No cutoff was selected. The study uses the
fixed 1001-row whole-source, pre-guard reserved development-training panel
(target 1000 extended to whole sources), under the unchanged E5 scoring and
priority rules. It does not establish a semantically validated graph.

Bounded findings
---------------
Across the 51 cuts, 25 graphs are complete and one is empty. Singleton record
deletion has at least one structural admission at 48 cuts; deletion of an
initially excluded record has an admission at 13 cuts. A genuine source request
has an admission at 47 cuts. These are counts of cutoffs on this fixed panel,
not population prevalence, semantic quality or task utility. All individual
counts and exact expectations remain in run/threshold_00.json through
run/threshold_50.json and independent_review.json.

All 51 full graphs matched a second block tiling of the same frozen scorer.
A separately implemented scalar arithmetic check passed 816 blocker-set
comparisons on the first 16 rows across all cuts. The scalar check is bounded;
the full alternate-tiling check is not an independent scoring implementation.
The independent aggregate audit verified all 53 completion-bound outputs,
204 exact expected-admission fractions and 50,050 blocker-CDF comparisons.
It did not recompute the private graphs locally.

Replay
------
Dependency: Python 3.9 or newer, standard library only. No additional packages,
network, cluster, credentials or private data are needed for aggregate replay.
From this bundle directory, run:

    python3 replay_aggregates.py

The script locates its inputs relative to itself, so the bundle can be moved.
Do not use Python -O. It checks the exact file inventory and SHA-256 manifest,
completion output hashes, all 51 cuts, exact fractions, arithmetic bounds,
monotonic blocker CDFs, scope flags, and agreement with the independent review.
It prints a JSON receipt to standard output and writes no files. Redirect that
receipt outside this bundle if desired; extra files inside fail the inventory.
The manifest is an integrity inventory, not an external authenticity signature.

The original scientific execution used Linux, Python 3.13.7, NumPy 2.5.3 and
SciPy 1.17.1. Observed stage time was 248.513 seconds; wrapper time was 249.615
seconds and reported child peak RSS was 260,792,320 bytes. It requested two
CPUs, 8 GiB memory and no GPU. These are one-run engineering measurements,
not a comparative speed or whole-system memory claim.

Contents and provenance
-----------------------
run/*.json: 54 byte-identical aggregate and completion records from execution.
independent_review.json/.txt: the original independent aggregate audit.
replay_aggregates.py: portable replay of the shareable aggregate checks.
study_protocol.json: explicitly deidentified view of the executed protocol,
  not the executed document, a replacement protocol or new authorization.
resource_receipt.json: deidentified numerical resources and original binding.
SOURCE_BINDINGS.json: actual executed package/protocol/source hashes, private
  input hashes, and the separately bound operational dispatcher. Hashes of
  omitted files record provenance; this bundle cannot verify their bytes.
PRIVACY_INVENTORY.json: public inventory and exclusion/inspection summary.
MANIFEST.json: SHA-256 and size of every other file in this bundle.

No private source text, record IDs, contributor IDs, labels, vectors, per-record
membership states, raw logs, scheduler identities, node names, user names or
private filesystem locations are included. Original process/transport receipts
and the raw retrieval archive are excluded. Hashes remain linkable identifiers
of artifacts; they contain no source payload. Operational strings were removed
only from the public protocol/resource views. The executed protocol and source
bindings are retained, and all run records are unchanged.

Scientific limits
-----------------
This is a separately declared structural follow-up after the negative v4
calibration. The original v4 negative result, unmet semantic agreement gate,
held-out boundaries and unmet original human gates remain unchanged. No human
or LLM annotation was performed here. Future annotation remains fully LLM with
zero human review. No learner/head, task utility, deletion-effect performance,
original full experimental registry or confirmatory experiment was completed
by Stage1. All original development sources remain excluded from future
confirmation. Singleton/source admission totals need not be monotone across
cutoffs; no such requirement was imposed.
