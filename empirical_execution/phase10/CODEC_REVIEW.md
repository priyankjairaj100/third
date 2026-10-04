# Independent implementation review of the finite-word codec

The review covers the exact mathematical family in `FINITE_WORD_THEORY.md`
and the actual implementation in `finite_word_codec.py`. It does not review a
natural-language dataset, a general compression scheme, or native empirical
performance. Phases 3–9 remain unchanged.

Final source bindings and focused outcomes are recorded in
`results/codec_independent_final/checks.json`. The checker preserves the exact
reviewed codec and its own source beside that report. The implementation owner
also maintains its separate codec checker; the two reports should not be pooled
as independent empirical samples.

## Contract and exact target

The implementation restricts the public family to `b>=1`, `m>=2`, powers of
four `k,D`, and `2+k+D<=2^20`. It verifies the signature coherence bound with
integer dot products. IDs receive their public roles after the actual frozen
priority sort. The candidate signs alone carry private information; the public
contract must be fixed independently of them. That independence is an access
assumption, not a fact established by checking Python types or a checksum.

The materialized candidate norm is exactly `16641/16384`. The codec returns
zero when no candidate has become selected. Any valid activating deletion set
must contain every common blocker and exactly the relevant private blocker,
exhausting the original horizon. Its selected records are the anchor and that
candidate. Their average-loss ridge solution is the stored candidate vector
divided by `16641/16384 + 2*lambda`.

The independent checker does not merely reuse that head formula. It constructs
the actual frozen graph from the stored FP32 rows, obtains retained membership,
interprets those stored rows as exact rationals, and checks every normal-equation
coordinate against the codec head. Positive regularization gives uniqueness for
nonempty selected data. The codec's prescribed empty target is zero.
No learner normalization is introduced. Rational outputs need not be exactly
representable by an FP32/FP64 array.

## Canonical state and disposal

The authoritative representation is immutable `bytes`: version, public-contract
digest, little-endian deleted-count field, sorted fixed-width public ID indices,
and packed eligible sign words. Eligible words occur in fixed candidate order.
Zero tail padding makes the byte representation unique for a fixed contract,
retained eligible words and cumulative deletion set.

The remaining horizon is always `b+1-|F|`. A candidate is retained only when it
survives and its remaining blockers fit this horizon. In this family that is
equivalent to `F` being a subset of its original blocker set. Hence eligibility
can only decrease. The transition copies the appropriate existing sign bits;
it never invokes fresh initialization, materialization, or an original-input
cache. The fresh retained initializer is a separate oracle that may read every
surviving candidate word and stores only those still eligible.

The checks compare batch, singleton, reverse-order and save/resume histories
with that fresh initializer at the same remaining horizon. They include the
important disposal boundaries:

- Common-blocker deletions retain every candidate word.
- One private-blocker deletion retains only its candidate word.
- Deleting the anchor, a candidate, or two distinct private blockers discards
  every private word for the remaining fixed horizon.
- At an activating query with zero remaining horizon, the selected candidate's
  word remains necessary to reproduce the current head on an empty continuation.

Changing a discarded word does not change the canonical state. Mutating the
original initializer's containers after initialization does not change service
bytes or the target. These statements do not mean old Python byte objects or
released heads have been physically erased. Any such copies retained by a
caller are extra sign-dependent storage and must be charged under the theorem.

## Validation finding and resolution

Two concrete input-contract inconsistencies were found and corrected. First, the frozen
priority helper accepts subclasses of `str`, while the deletion API required
exact `str` values. A contract could therefore expose an ID object that its own
deletion API rejected. The implementation now requires exact public `str` IDs,
matching the deletion interface. The independent checker exercises this case.

Second, arbitrary positive rational regularization was accepted, but sufficiently
large integer numerators failed later decimal JSON serialization under the
observed Python integer-to-decimal digit limit. The preserved reproduction uses
`Fraction(10**5000)`. Public rational fields and the optional exact-head accounting
now use explicit hexadecimal integer strings, keeping exact arithmetic while
avoiding that decimal conversion limit. The checker covers initialization,
activation, exact decoding and output accounting for this case. The pre-fix
source and observed failure remain in `results/codec_review_large_integer_finding`.

Requests reject unknown identifiers, repeated IDs within a batch, already
deleted IDs, nontext values and cumulative budget overflow before assigning a
new service state. Rejected updates preserve the original immutable bytes.
Empty requests are identity operations. Fresh reconstruction refuses a reset
horizon, missing surviving words, wrong word lengths and noninteger signs.

State parsing rejects wrong versions/digests, truncated or extra bytes, invalid
counts, noncanonical deleted-index order, duplicate indices and nonzero padding.
It validates representation consistency; it does not authenticate private
payloads against original data. A payload-bit change may simply describe another
legitimate family member. The public-contract digest is likewise a consistency
binding under ordinary hash assumptions, not a proof of historical authenticity
or an unconditional collision-free encoding of arbitrary public contracts.

## Byte-accounting assessment

For `e` eligible candidates and `f` deleted IDs, the serialized representation
has exactly

`HEADER_BYTES + f*index_bytes + ceil(e*D/8)` bytes.

The header is 48 bytes. The last term is the packed private payload, including
at most seven padding bits. Its private information content is `eD` bits.
The public deletion ledger, header and public contract are additional costs;
the `eD` lower bound is conditional on them.

The reported materialized graph CSR extent is checked against the actual
priority, offset and edge array byte counts. Materialized fixed public rows,
full original feature caches and targets are separately described hypothetical
storage choices. These categories may overlap with the serialized public
contract and must not be summed indiscriminately into a supposed optimum.

The optional exact-head accounting reports the length of one explicit
hexadecimal numerator/denominator JSON representation. It is not a minimum output encoding,
Python object size, allocator measurement or peak RSS. The regularizer's bit
length affects exact output sizes. Transition workspace can include old bytes,
new buffers, metadata, public hashing and temporary arithmetic. A caller may
also keep earlier states and outputs. No total-byte optimality, constant-memory,
native-speed or physical-erasure claim is supported.

## Verification scope

`check_codec_review.py` uses bounded mathematical fixtures only. It exercises
three private dimensions (`1`, `4`, `16`), bit/nibble/byte packing boundaries,
multiple private sign patterns, every legal deleted set in those small
contracts, and an exhaustive smallest private alphabet for conditional state
cardinality. It supplements the analytic monotonicity and scorer-transfer
proofs; it cannot prove the universal information theorem by enumeration.
No synthetic empirical dataset, semantic embedding, human response, training
run, or benchmark is created.

The final report is authoritative for the exact source hash, check count and
case counts. Review conclusions remain conditional on the stated frozen scorer
arithmetic, fixed public contract, complete accounting of private side
information, and unchanged initial deletion horizon.
