# Accepted calibration preparation

This command connects the existing acceptance check to a blank calibration pack.
It accepts local files only.
It does not download a model or contact annotators.
It does not select a threshold.

Run this command from the repository root:

```bash
python3 empirical_execution/phase7/prepare_calibration.py \
  --dataset civil_comments \
  --encoder e5 \
  --dossier local_inputs/dossier.json \
  --out local_outputs/calibration_attempt_01
```

The output directory must not exist.
The command returns zero only after it creates the blank pack.
A rejected input produces a receipt and exit code 2.
A failed replay cannot produce an accepted pack.
If writing fails, the receipt marks any partial pack as unaccepted.

## Input files

Start with `templates/calibration_dossier.template.json`.
Replace each path and hash with values from actual local files.
The template cannot pass acceptance without these files.

The dossier has exactly four fields:

| Field | Meaning |
| --- | --- |
| `source_acceptance` | Original input paths, parser configuration, prepared files, and encoder files. |
| `records` | Every record in the calibration partition, in its original cache order. |
| `features` | Corresponding stored FP32 vectors from the actual encoder. |
| `provenance` | The complete derived calibration provenance. |

File descriptors use exactly `path`, `sha256`, and `kind`.
Supported kinds are `json`, `jsonl`, and `npy`.
Paths resolve from the dossier directory.
Descriptors cannot leave that directory.
The command rejects repeated JSON keys, invalid hashes, and reference cycles.

Inside `source_acceptance`, existing path fields remain strings.
Do not replace those path fields with loaded records or arrays.
Use the interface in `phase6/ACCEPTANCE_README.md`.

The existing `phase4.calibrate_cache.load_aligned_cache` function derives the last three fields.
It preserves the complete calibration partition.
Do not select a smaller subset to reduce the acceptance cost.
Creating those input files still requires the original prepared population and actual encoder cache.

## Checks and outputs

The command uses the frozen study design.
It uses selection seed 20271003 and scorer block size 256.
It replays the original parser and complete encoder cache through Phase 6.
It then creates the selection manifest and checks input and code hashes again.

The output contains:

- `acceptance.json`, when replay begins.
- `receipt.json`, which records the outcome and file hashes.
- `selection/private_sampling_manifest.json`, after successful preparation.
- `selection/blinded_assignments.json`, after successful preparation.
- `selection/responses.template.json`, with blank responses only.

The command leaves publisher authenticity and human collection unverified.
A successful result permits preparation only.
It does not authorize primary interpretation or satisfy the quality gate.
Human collection and independent validation remain separate requirements.
WCEP cannot select its own threshold.
WCEP inherits the accepted News threshold.

Keep actual source files, model files, and human responses outside public result directories.
Review their publication permissions before any repository upload.

## Qualification scope

Positive interface checks mock the upstream acceptance result explicitly.
They use existing Civil text and lexical vectors to check file handling.
They do not establish original archive acceptance or real encoder execution.
Unmocked checks reject missing files and the engineering preview.
The saved report identifies these scopes separately.
