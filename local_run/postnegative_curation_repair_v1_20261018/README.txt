Execution checkpoint: Stage 2 COMPLETED0:0 as job15416 and passed independent aggregate audit.
Audited results: ../postnegative_curation_repair_results_v1_20261018/.
The original launch checkpoint is preserved in EXECUTION_CHECKPOINT.launch.json.
The earlier prospective plan remains unchanged in PLAN.prospective.json.
PLAN.json adds only reviewed runtime policy; package_manifest.json seals the eight runtime files.
Do not resubmit. Exact curation membership/state repair only; no labels, learner or semantic validation.

POSTNEGATIVE CURATION REPAIR CORRECTNESS, STAGE 2 VERSION 1

Prepared prospectively before inspecting Stage 1 outcomes. This stage does not
select a threshold or depend on observing useful activations. It runs every one
of the original 51 cutoffs on the same natural 1,001-record development prefix.

The original algorithm_round2.IndexedState implementation supports generic
integer statistics. Here each record's statistic is its deterministic original
panel membership basis vector e_v, dimension 1,001. This records selection and
allows exact comparisons of the whole integer polynomial state. It is not a
fabricated annotation, task label, ridge model or NLP head. No text, label,
held-out feature slice or model response is read.

For each graph, one frozen R sequence and one frozen S sequence are generated
using the original request generator, seed20261018 and the dedicated namespace
in the plan. They are the same across every cutoff. Sources sample only the
original native contributor proxies; unknown singleton sources never enter
that sampling frame. The original generator clips short frames explicitly.
There is no graph-selected stress path or effect-based request choice.

Initial deletion horizon128 decreases cumulatively. Every singleton selected
set is checked against an independently implemented raw-blocker predicate.
At checkpoints1,8,32,128, compare the same cumulative deletion as singleton,
batch and reverse-singleton operations. Require equality of every coefficient,
inverse index, degree index, alive-unit set and remaining horizon. Independently
recompute the retained-feature graph, selected set and membership polynomial
at the remaining horizon; compare all logical state, not only its readout.

The original generic service uses idempotent retries. This harness explicitly
tests its unchanged retry contract; it does not claim the separate strict
service API has the same retry behavior. Runtime invariant/operation-count
checks remain enabled. Do not invoke Python with -O.

Resource proposal: approved CPU compute node, 2allocated CPUs, one numerical
thread, 8GiB RAM, 3,600seconds. Use the same separately approved scratch/NFS
policy as Stage1, with8GiB scratch reserve and2MiB total publication/log budget.
A fresh review, quota check, and immutable output path are required. Timeout
or partial completion remains a failed attempt. All private inputs/states stay
on the cluster; export aggregate reports only.

Usage after root review and protocol freeze:
  python -B curation_repair.py --repo FROZEN_REPO --protocol PLAN.json \
    --protocol-sha256 HASH --allocation PRIVATE_ALLOCATION \
    --prefixes PRIVATE_PREFIXES --vectors PRIVATE_CACHE --out NEW_SCRATCH_OUTPUT
Set plan status frozen_for_stage2 only after review. The wrapper must bind the
script/helper/test/protocol package and pin observed runtime dependencies.
Do not update the frozen checkout. This script imports its original code.

Six small algebraic software tests cover both record/source compositions,
owner-source blocker cancellation, fixed requests across cutoffs, unavailable
source frames, integer membership meaning, invalid-request atomicity and the
idempotent retry contract. These fixtures are not empirical results.

This stage can establish the exact curation repair mechanism on these natural
input graphs. It cannot establish supervised effects, semantic-quality claims,
efficiency against compact eligible payload, native ridge certification,
physical erasure, or completion of the original semantic experiment registry.
A later native-label learner comparison needs its own prospective protocol.
