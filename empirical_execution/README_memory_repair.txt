MEMORY REPAIR IMPLEMENTATION AND EVIDENCE
4 October 2026 | All execution in the current workspace

Start here for the new memory result. README.txt describes the earlier dense
engineering pilot; its recorded outputs remain unchanged.

Decision
Use JointSpanRidgeSummary as the aggregate-summary candidate. Keep the original
dense implementation as an audit reference, FactoredRidgeSummary as an ablation,
and CompactEligiblePayloadState as a mandatory stronger payload comparator.
No superiority to eligible-payload retention is claimed.

What changed
The dense algorithm expands each record's rank-one ridge contribution into one
or two full ambient Gram matrices. The new representation stores each current
coefficient as G=U A U.T, H=U B and an exact count, or packed dense moments if
smaller. It uses the JOINT range of G and H: signed Gram cancellation can leave
a nonzero cross moment. Deletion still performs the same indexed substitution,
collision addition and decreased-horizon pruning.

The exact-rank theorem gives O(E_h(d+c)) numeric words, plus key/membership
metadata. The implementation-safe envelope uses initial eligible count E_0.
Rank residuals are retained, and dense entries are not demoted. Joint-state
bytes need not decrease monotonically. The construction allocation cap is not
a peak-RSS cap or a lifecycle cap. See the PDF/theory fragment for full bounds.

The reduced decoder solves (A+lambda*n I)C=B and returns U C. It avoids an
ambient d-by-d matrix for compact current coefficients. Its residual is an
ordinary numerical diagnostic, not a certified bound on moment roundoff.

Files
- ccu/joint_span_summary.py: strongest summary, reduced/common decoders,
  canonical projector basis orientation, streaming non-pickle checkpoints.
- ccu/factored_summary.py: signed Gram factors with separate dense cross matrix.
- ccu/compact_payload.py: eligible rows without a persistent Gram, kernel or
  historical head; release uses primal/dual ridge as appropriate.
- run_memory_repair.py: matched natural-data validation with independent oracle.
- memory_process_probe.py: one fresh-process high-water measurement per method.
- memory_theory_addendum.tex: mathematical derivation, bounds, costs, caveats.
- memory_results/: design, checkpoint registry, measured results and audit.
- build_memory_report.py: renders the accompanying report using the local TeX
  format already available in this workspace. Standard LaTeX can also render
  the included complete report source when its listed packages are installed.

Measured evidence
100 real Civil Comments records, fixed lexical curator128 at threshold0.6,
original targets, horizon8 and lambda0.01. No synthetic corpus or labels.
These are implementation experiments, not semantic-encoder or NLP-utility
results. Multioutput uses all seven original toxicity label fields.

  d    c   Dense arrays   Gram-factor arrays   Joint-span arrays   Payload arrays
  64   1     1,698,840          119,141               70,512            21,912
 128   1     6,640,920          270,949              172,144            43,160
 768   1   234,483,480        1,628,326            1,039,232           255,640
 128   7     7,249,176          879,205              183,136            47,144
 768   7   238,133,016        5,277,862            1,047,200           259,624

All entries are initial owned numerical array bytes, not total memory. Python
counts, key/index metadata and workspace are additional. Public identifiers
use the same integer representation in the four new comparisons. The old
string-ID payload baseline appears only as a historical array comparison.

544 release checkpoints passed for all four methods; the joint reduced
decoder was also exercised544times. Largest joint common-head discrepancy:
1.1102230246251565e-15. Largest reduced-head discrepancy:
1.1657341758564144e-15. Another100full-deletion releases passed; joint maximum
head discrepancy2.064945557871121e-14, final headexactlyzero. There are12atomic
rejection checks,3order/batching checks, and20complete coefficient-map audit
checkpoints per compressed method. Coefficient audits reference the dense
implementation; released moments/heads use a fresh independent oracle.

Fresh whole-process peaks after construction and four releases, one observation
each: dense356.71MiB, Gram-factor135.42MiB, joint/reduced121.28MiB,
compact-payload118.39MiB. The Python/numerical runtime and original inputs are
included. Do not interpret setup-peak subtraction as exact allocation or the
array reduction as a whole-process RAM reduction. Timings are not speed claims.

Independent audit
Actual coefficient bases of rank65,2,3 were changed by coordinate permutations
and signs. Projector canonicalization preserved the coefficient and basis to
ordinary floating tolerance. A1,135,027-byte checkpoint roundtripped exactly;
its loaded arrays were owned/disjoint and a subsequent release matched exactly.
Saved source hashes identify the reviewed and executed code.

Important state distinction
Canonical basis orientation removes coordinate-rotation history in exact
arithmetic. It does not make the entire state canonical: dense/compact mode can
depend on history, and FP64 rank/pivot choices are not bit-canonical. The fast
implementation guarantees decoded coefficient/head equivalence in exact
arithmetic and measured numerical agreement here. A strict canonical-state
variant needs current-coefficient mode re-encoding, potentially costing O(d^3)
for dense entries. No new privacy or allocator/backups guarantee is claimed.

Factors may reveal information already present in their coefficients. The
contract is no SEPARATE feature/label table and no repair-time payload reread,
not that factor values can never coincide with or reveal an original vector.
The payload comparator intentionally retains eligible rows.

Replay
Use the existing numerical dependencies pinned in requirements.txt. From this
directory, preserve memory_results/ before replay because outputs are replaced:

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_memory_repair.py

Each standalone peak probe is a fresh invocation:

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python memory_process_probe.py joint_span

Other method arguments: dense, gram_factor, compact_payload. The complete
root-run peak table is saved in memory_results/process_peaks.json. Runtime has
no network dependency and uses only the included CC0 Civil fixture.

Main API
  state = JointSpanRidgeSummary.build(x, y, blockers, horizon=8)
  state.delete([integer_record_id])
  head = state.decode_compact(lambda_reg=0.01).weights
  state.save(path)
  state = JointSpanRidgeSummary.load(path)

For source deletion pass owners and universe_size; blockers remain record
indices and are mapped to distinct source owners. The proof and interface
cover it, but these new model tests use record deletion because original Civil
source identifiers are unavailable. The broader ACL semantic study remains
subject to its original acquisition and human-quality gates.
