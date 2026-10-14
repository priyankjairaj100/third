THIRD — LLM-only selection sampler v1 — 2026-10-14

This new adapter performs deterministic score/bin/reservoir preparation for the
prospective model-only v3 protocol. It invokes no model, collects no human labels,
selects no threshold, and never activates inference. All previous packages,
analysis locks, source files, numerical routines and recorded failures remain
unchanged. Passing this adapter does not satisfy historical human gates or prove
source authenticity, semantic validity or confirmatory study readiness.

Bindings and admission
- Exact protocol package, PLAN, actual frame13556 result and three payloads are
  pinned in expectations.json. The frame identity used in every v3 ID/RNG is the
  RAW result.json SHA256 d4fc45ec0d38ddc94c2a9c82ee3948020d0acd1f47fd84582b87ad5e273aedb1.
  The frame's distinct internal seal is recorded separately.
- The protocol's 39 historical file bindings are supplied as an exact separately
  pinned private snapshot. They are evidence for protocol.validate, not a claim
  about the current executing checkout or a replacement for source admission.
  Actual phase3 calibration.py/panels.py imports resolve to the executing repo
  and are separately byte-pinned. No protected lock is edited or hidden.
- Rehash the complete FP32 frame, mapping, input receipt, original prepared source
  records/audit and small encoder-asset/SciPy-repair provenance. Check source
  partition hashes, unique record IDs, original full-row order, all three mapping
  digests, shape, FP32 dtype, finiteness and nonzero rows. Recheck all watched files
  and closed frame/evidence/lock inventories before publishing success.
- The frame13556 already executed full source/cache/shard/replay admission. This
  adapter binds that result and its actual frame bytes; it does not rerun encoder
  forwards or all 32 encoder/replay validations. Original source limitations stay.

Qualification, prior to production seeds
  sampler.py qualification --repo R --workspace W --protocol-package P \
    --protocol-evidence E --protocol-evidence-sha256 ESHA --frame-package FPKG \
    --frame-dir F --frame-result-sha256 FSHA --package-sha256 SSHA --out NEW

The fixed first512 ACTUAL calibration rows form an explicitly exposed engineering
subset, not a substitute full population. Use the registered B256 frozen scorer,
20 bins, cap30 and a public deterministic engineering-only pair of seeds/stages.
No production entropy or sampling lock is read. Score/audit/bin/reservoir/hash time
is measured as a single pass; independent parity and final input hashing have
separate timing fields. Node contention is not controlled.

Independent bounded parity constructs one512x512 matrix using the frozen ordered
FP64 authority, enumerates upper tiles independently, computes bins independently,
and implements the reservoir loop independently. All130816 emitted scores/order,
20 counts and selected tuples must match. Additionally171 first19 pair-at-time
scores must equal matrix entries; all selected identities/orientations, rational
weights and diagnostic swap selections receive direct checks. This is finite
software/numerical parity, never an accuracy or real-arithmetic certificate.

The subset has3 tiles and196608 computed tile entries; full selection has3486 tiles
and223102992 entries. Python emitted-pair work scales differently. A bounded
measurement is not a full-runtime guarantee. Root must choose resource/time bounds
from the actual completed qualification and retain failure if full work times out.

Production selection, only after independently reviewed qualification and lock
  sampler.py selection <the same common arguments> \
    --lock-package LP --lock-package-sha256 LSHA \
    --sampling-lock L --sampling-lock-sha256 LRAW \
    --qualification-result Q --qualification-result-sha256 QSHA \
    --qualification-review QR --qualification-review-sha256 QRSHA

The separate sealed sampling-lock helper binds four once-only256-bit OS seeds,
frame/protocol/scorer/runtime/order, this sampler package, completed qualification
and its independent review. This sampler draws no entropy. It accepts only a
complete durable lock with intent and no failure/extra file, the fixed a01 lock
slot, and the corresponding fixed a01 full-selection output. The helper package
is pinned; dependency sealing is acyclic (helper first, sampler second).

Full selection includes ALL20996 rows and220405510 unordered i<j pairs, including
same-source pairs and unknown-owner singletons. No length, context, availability,
prior-pilot exposure or duplicate-link filter is applied. The unchanged frozen
iter_pair_scores(vectors,256,audit), _bin and _reservoir_insert are called directly.
Raw FP64 scores remain unclipped, including out-of-range values; only bins saturate
as originally specified. At most600 tuples are retained; no dense full matrix or
complete pair archive is allocated. The streamed SHA256 input is exactly ASCII
  decimal_i TAB decimal_j TAB float.hex(raw_score) LF
in frozen upper-tile traversal order. Sample rows serialize in ascending(i,j).

For each bin, select min(30,Nbin), with exact reduced-rational probability n/N and
inverse weight N/n. IDs/orientation/model seeds come from the pinned v3 protocol.
Each bin has its own RNG from the selection stage and committed seed. Diagnostic
swaps use a separate committed selection_swap RNG (bin_index=null), the unchanged
reservoir over the complete sorted sample, capacity ceil(n/10), and sorted selected
indices. They are diagnostic requests, never extra units or independent votes.

Artifacts and limits
Both modes publish input_bindings.json, selection.private.jsonl,
selection_swap.private.jsonl and result.json (passed result published last).
Qualification additionally publishes diagnostic_frame_vectors.npy and
  diagnostic_frame_rows.private.jsonl
containing the exact first512 vector/metadata slices for independent replay;
there are no document bodies. The NPY is1572992bytes for512x768FP32. Every output is
hash-bound. A16MiB whole output-directory cap is checked before writes and final
publication. Each sample/input payload is capped8MiB; result cap4MiB. Root's separate
128MiB stage allowance includes bounded console output, filesystem and contingency.
Failures retain FAILED.json and incomplete artifacts, never a usable partial sample;
progress checkpoints are every4096 pairs and may lag a failing traversal. No resume,
retry, truncation, algorithm substitution or partial-census success is supported.

Runtime and software checks
Actual CLI work requires Linux, positive SLURM_JOB_ID, csis.cn1, exact encoder-cpu
Python3.13.7 and NumPy2.5.3, four numeric-thread environment values1, offline flags,
and no visible GPU. CPU affinity, interpreter, job, max RSS and elapsed fields are
recorded. 8GiB is an allocation estimate, not a measured ceiling; OOM must fail.
Only execution inside Slurm is intended. This package submits no job and uses no SSH.

Unit discovery: python -m unittest discover -s PACKAGE -p test_sampler.py -v
Staged overrides: THIRD_SAMPLER_REPO, THIRD_SAMPLER_PROTOCOL,
THIRD_PROTOCOL_EVIDENCE. Tests use tiny labeled algebraic fixtures, not natural
texts, live samples, model calls or production seed draws. The pinned cluster
interpreter must repeat all22 tests without skips before actual qualification.
SOFTWARE_VALIDATION.json records local software checks only. Independent natural
qualification and full selection are later evidence, not outcomes of this package.
