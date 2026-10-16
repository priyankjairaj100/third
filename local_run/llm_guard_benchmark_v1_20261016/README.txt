Bounded resource preparation for the fully LLM development leakage guard.

No model threshold or semantic validity is accepted here. This stage uses the
first256 pre-reserved development training rows and first1024 stable-ID ordered
original heldout rows. It reads no text or task labels, computes the complete
262144 cross-pairs using frozen ordered FP64 normalization/dot operations, and
prepares all51 original thresholds without selecting one. The role reservation,
full cache and original scoring source are bound by exact bytes.

The new streaming routine visits heldout rows in stable-ID order. It stores the
first above-threshold witness and exact score, not the largest-score witness;
these differ in the original guard. Strict greater-than and all51 original cuts
are unchanged. Every intended pilot pair is scored. Seven synthetic tests cover
all-cut parity with original semantic_guard, chunk/order invariance, exact ties,
canonical witnesses and malformed inputs. The real-cache job additionally
compares exact retained IDs, witnesses and scores with the unchanged original
semantic_guard at0.50,0.99 and0.995.

Full development would score all10314 reserved training rows against all42063
original calibration/test rows. The small pilot supplies a linear pair-count
time projection, not a guaranteed runtime or ETA. It excludes scheduler queue,
full-scope input checks, final publication and scaling effects. Whole-child RSS
and time include memmaps, metadata, gathered vectors and reference comparisons.

No whole-population guard, effect study, new label, or full-guard automatic launch
is performed. A separate full execution package must bind the passed fresh v4
model agreement validation, actual admitted threshold and reviewed resources.

Run one CPU Slurm job on cn1 with2CPUs,8GiB and20-minute cap. BLAS1, no CUDA,
downloads or retries. Pause only local controller transitions during this brief
qualification; Mistral keeps running. Resume immediately at terminal state or
definite prelaunch rejection. Preserve and reconcile ambiguous launch receipts.
The utility intake.py is copied byte-for-byte from the executed intake package;
only its parsing, publishing and binding helpers are imported. It is not rerun.
