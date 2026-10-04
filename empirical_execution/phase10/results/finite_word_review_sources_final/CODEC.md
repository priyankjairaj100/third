# Canonical finite-word sequential codec

`finite_word_codec.py` implements only the stated finite-sign hard family.
It is not a compressor for arbitrary text features or arbitrary curation
graphs. The checks are tiny mathematical software fixtures, not synthetic
empirical datasets or NLP experiments.

The initial private sign information is exactly `mD` bits. The implementation
packs those data bits into `ceil(mD/8)` bytes and separately stores public-
contract binding and deletion metadata. Padding, public inputs, initialization
caches, output materialization, and workspace are not free. No total-byte
optimality, practical speedup, or physical-erasure claim follows.

## Fixed public contract

The implementation requires `b >= 1`, `m >= 2`, powers of four `k,D`, and
`d = 2+k+D <= 2^20`. Public signature rows contain exactly `k` ordinary integer
signs. Their normalized pairwise inner products have absolute value at most
`1/6`. All public identifiers are distinct, nonempty, exact Python strings.

The constructor sorts the entire supplied universe using the actual frozen
`phase3.reference_graph.stable_priority` and its declared seed. It then assigns
roles in that order: one anchor, `b` common blockers, `m` private blockers,
and `m` candidates. Roles are not assigned before hashing. Thus the anchor and
all blockers precede the corresponding candidates under the actual curator.

The orthogonal coordinate blocks are `e0,e1,U(k),H(D)`. For public normalized
signatures `u_i` and private normalized hidden sign words `z_i`, the stored
vectors are

\[
a=e_0,\qquad s_j=\tfrac12e_0+\tfrac78e_1,\qquad
p_i=\tfrac12e_0+\tfrac78u_i,\qquad
w_i=\tfrac12e_1+\tfrac78u_i+\tfrac1{128}z_i.
\]

The same exactly representable FP32 rows serve the curator and learner.
The learner does not normalize them. Candidate labels are public binary one;
all other labels are public zero. The curator threshold is the stored binary64
`float(0.4)`, hexadecimal `0x1.999999999999ap-2`.
The dimension cap is the regime of the separate frozen-scorer transfer proof.
The constructor also checks exact FP32 representability of the nonzero scales.

The public regularization parameter is a positive `Fraction`, positive integer,
or finite positive Python/NumPy FP64 value. A floating value is converted using
its exact stored binary64 ratio. For example, stored `0.1` is not silently
replaced by the decimal rational `1/10`. Booleans and FP32 lambda inputs are
refused by this interface.
Public rational numerators and denominators use canonical hexadecimal strings
in the serialized contract, so arbitrary rational size is not limited by
Python's decimal-integer string conversion limit.

Public identifiers, signatures, seed, lambda, and all other contract fields
must be fixed independently of private signs. This is a semantic precondition;
the code cannot establish historical independence from a supplied object.
The public contract contains no actual `z_i` block or candidate feature row.

## Eligibility and exact sequential state

Let `K=b+1` and `B_i={s_1,...,s_b,p_i}`. For cumulative deleted set `F`, the
codec stores candidate `i`'s sign word exactly when

\[
w_i\notin F,\qquad |B_i\setminus F|\le K-|F|.
\]

Because every `B_i` has size `K`, this is equivalent to `F` being a subset of
`B_i`, with candidate survival. The implementation evaluates the general rule.
It does not rely on a list of special cases.

- Deleting common blockers alone retains all candidate words.
- Deleting one private blocker and any allowable common prefix retains only
  that blocker's candidate word.
- Deleting an anchor, a candidate, or two different private blockers leaves
  no future-eligible candidate.
- On the exact activating deletion `F=B_i`, the terminal state still needs
  candidate `i`'s word: its current head is nonzero, even though no budget remains.

Future eligibility is monotone as cumulative deletions grow and the remaining
horizon shrinks. The transition copies only still-eligible bits from the
current immutable state. It receives identifiers only and never calls the
retained initializer or full-vector materializer.

For a fixed public contract and cumulative `F`, state bytes depend only on
the surviving eligible sign words. They do not depend on batch boundaries,
deletion order, or any discarded candidate signs. They exactly equal a fresh
retained initialization at horizon `K-|F|` under that same public universe.
This is the claimed canonicality; it is not a claim that a smaller universe
with new priorities or a reset horizon has the same state.

For this family, every remaining eligible word can still be individually
probed by completing `F` to its `B_i`. The resulting exact head distinguishes
all its `2^D` sign values. Consequently the conditional private information
minimum is `D` times the eligible population. Public/deletion metadata and
physical byte padding are separate from that model-specific bit statement.
See the accompanying theorem and independent proof review for the lower bound.

## Binary format

The authoritative state has exact type `bytes`:

| Field | Encoding |
| --- | --- |
| Version | Eight bytes `CCUFW01` followed by zero |
| Public contract binding | 32-byte SHA256 of canonical public JSON |
| Deleted count | Eight-byte unsigned little-endian integer |
| Deleted record indices | Increasing, unique public-universe indices; fixed minimal whole-byte width for that universe |
| Eligible sign payload | Candidate role order, then hidden-coordinate order; positive sign is one; least significant bit first |
| Tail padding | Zero bits through the end of the last byte |

The header occupies 48 bytes. Eligible positions, remaining horizon, and payload
length are derived from public information and deletion indices; no separate
private headers or eligibility bitmap are stored. The parser rejects a wrong
version/contract, noncanonical or out-of-range indices, excess budget, truncated
or trailing bytes, and nonzero padding. Mutable byte arrays are refused as
authoritative state.

SHA256 is a contract/version consistency check under ordinary hash assumptions.
It is not an authentication mechanism or an unconditional cross-contract
collision theorem. A payload bit flip can encode another legitimate sign word.
Format validation does not authenticate the original private data.

## API and exact target

`FiniteSignContract(...)` constructs only public parameters.
`initialize_retained(contract, blocks, deleted_ids=(), remaining_horizon=None)`
requires exactly the sign blocks of all surviving candidates. It validates
even ineligible surviving blocks, then persists only eligible ones. Deleted
candidate blocks are neither required nor read. An explicit remaining horizon
must equal `K-|F|`; initialization cannot silently reset the budget.

`transition(contract, state, request_ids)` is functional.
`CanonicalService(contract, state)` wraps it with two persistent fields only:
the public contract and current authoritative bytes. `delete(ids)` replaces
the state only after a successful transition. Unknown, duplicate, already
deleted, or overbudget identifiers are refused atomically. An empty request
returns the same bytes. Requests provide no forgotten or retained private
payload.

`head_exact(contract, state)` and `service.head()` return a transient tuple of
exact rational coordinates. The selected learner target is

\[
(X^TX+n\lambda I)W=X^Ty.
\]

Except at a unique activating deletion, every selected label is zero and the
head is exactly zero. At `F=B_i`, the selected set is `{a,w_i}`. The candidate
is orthogonal to `a` and has exact squared norm `16641/16384`, so

\[
W=\frac{w_i}{16641/16384+2\lambda}.
\]

The exact head is generally not representable as FP32 or FP64. Returning an
ordinary floating array would change this exact-output contract. No head is
retained inside the service.

`materialize_family` exists for mathematical input preparation and oracle
checks only. It constructs the entire original FP32 private feature cache and
binary targets. Neither transition nor head decoding calls it.

## What is counted

`accounting` reports the actual serialized state split into header, deletion
indices, packed private bytes, data bits, and padding bits. It separately
reports canonical public-contract JSON bytes, identifier UTF8 bytes, optional
public full-graph int64 CSR bytes, public fixed FP32 rows, and the original
full FP32 feature cache and FP64 labels if materialized.

The public full graph is derivable from the family. If retained explicitly,
its arrays and identifiers must be counted. The public contract's Python
tuple/string/integer overhead is not equal to its serialized JSON size.
The routine does not measure Python resident memory, allocator overhead,
imported runtime/library memory, or a concurrent-process peak.

Initialization may hold the complete original private input alongside the
packed result. Fresh retained initialization may reread all surviving private
blocks; that is an oracle operation, not the sequential transition. During an
update, old bytes, a new payload buffer, new serialized bytes, deletion and
eligibility structures, public hashing buffers, and temporary bit operations
can coexist. Therefore shrinking persistent payload is not a peak-memory bound.

`materialized_head_accounting` explicitly constructs the exact output and
reports the size of one canonical fraction-pair JSON representation and the
integer magnitude bits. Fraction integers use signed hexadecimal numerator
strings and positive hexadecimal denominator strings. It is not a minimum-size head representation and
excludes Python object overhead. Its own output/serialization workspace is a
cost. A deployment retaining returned heads, original caches, private seeds,
old snapshots, remote stores, or logs must count that private information too.

Old immutable snapshots can remain in caller memory. The API supplies no
physical erasure, observer privacy, secure anti-rollback mechanism, or general
hostile-Python-object boundary. Save/resume under the same contract preserves
the encoded horizon; deliberately restoring an old snapshot is outside the
monotone current-service deletion history.

## Verification boundary

The owner checker exhausts all 256 private words for `m=2,D=4,b=1` and all four
words for `m=2,D=1,b=2`. It rebuilds the actual frozen Phase 3 graph after every
legal deletion set, independently constructs full rational Gram/cross normal
equations, and compares exact heads. It checks every deletion order, batch
equivalence, every intermediate fresh retained state, independent bit packing,
partial disposal, state-image collapse after dropping information, malformed
bytes, atomic refusals, original-input independence, and explicit accounting.

The mathematical fixtures use orthogonal public Hadamard signature rows.
They do not test the logarithmic-dimension existence result by sampling.
The independent codec reviewer supplies separate controls and source-bound
findings. Finite enumeration supports implementation behavior; it does not
replace the graph-margin, information, or sequential-invariant proofs.

Run the checker into a new output path, preserving prior attempts:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 empirical_execution/phase10/check_finite_word_codec.py \
  --out empirical_execution/phase10/results/codec_checks.json
```

No natural corpus, genuine human response, or primary NLP result is produced.
