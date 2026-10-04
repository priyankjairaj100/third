# Phase 6: complete experiment routes and acceptance

This phase closes the remaining software and design gaps from Phase 5.
The primary semantic study remains unstarted.
Original corpora, real encoder assets, and human responses remain unavailable.

Read `COMPLETION_LEDGER.md` for the final checked results and remaining requirements.
Read `REQUIRED_INPUTS.md` for the inputs needed to start the primary study.
Earlier phases remain unchanged.
Their results retain their original scope.

## Main components

| Component | Purpose |
| --- | --- |
| `recipes.py` | Resolves the 21 prior recipe obligations prospectively. |
| `dispatch.py` | Joins registered experiment branches and preserves every job outcome. |
| `extensions.py` | Executes additional request laws, controls, audits, human analysis, and statistics. |
| `acceptance.py` | Replays supplied source bytes, preparation, caches, guards, panels, and original labels. |
| `methods.py`, `workers.py`, `run_isolated.py` | Runs declared method variants under the common process and numerical policies. |
| `measurement.py` | Measures declared state, logical operations, process memory, and dependency scope. |
| `state_audit.py` | Rebuilds designated state and binds a separate audit service to its recorded snapshots. |
| `statistics_evidence.py` | Binds statistical identities, predictions, evaluation metadata, and paired timing receipts. |
| `logistic_selection.py` | Selects logistic output thresholds using separate calibration probabilities. |
| `human_inference.py` | Computes finite population intervals under the actual sampling design. |
| `dyadic_convex.py` | Accelerates exact stored-value verification while preserving rational bounds. |
| `audit_independent.py` | Checks the new implementations independently and binds its report to source files. |
| `audit_publication.py` | Scans publishable files and compressed members for accidental disclosure. |

## Evidence boundaries

All current natural experiments reuse the Civil100 lexical preview.
They are engineering checks, not independent replications.
Small algebraic fixtures verify software only.
They do not create synthetic empirical evidence.

Source replay establishes derivation from supplied bytes.
It does not authenticate publishers, completeness, permissions, or human collection.
The dispatcher requires those evidence records separately.
Recorded trust decisions are not cryptographic proof of their substance.

The memory qualification remains unchanged.
Compact eligible payload outperformed larger summaries on the earlier natural fixture.
Coordinate compression does not establish minimum total storage.
Shared host timings do not establish a publication speedup.

FP32 variants have separately named states and recorded numerical outcomes.
Their approximate moments do not inherit exact arithmetic guarantees.
Residual gates check their declared linear system.
They do not alone bound accumulated moment error.

No actual human ratings exist.
Blank ratings produce missing outcome bounds, not positive semantic findings.
The confidence intervals cover fixed operational outcomes under the specified sampling design.
They do not cover latent truth or unspecified variation between raters.

## Reproduction and saved history

Use a new result directory for each run.
Do not replace historical results or failure records.
Use one BLAS thread when reproducing the measured worker checks.
The final ledger lists the exact commands and authoritative directories.

Some verbose historical directories may be stored as lossless archives.
Each archive has a complete file manifest with byte counts and SHA256 hashes.
Restore one with:

```bash
python3 tools/phase6_result_archive.py restore DIRECTORY_NAME
```

The tool refuses unsafe paths and differing existing files.
The main backup manifest covers the archives and their manifests.

Before reproducing the consolidated audit from a fresh checkout, restore the checkpoint archives:

```bash
python3 tools/verify_backup.py
python3 - <<'PYRESTORE'
import json
from tools.phase6_result_archive import verify
from pathlib import Path
p = Path('empirical_execution/phase6/results/archive_checkpoint_inventory.json')
for row in json.loads(p.read_text())['entries']:
    print(verify(row['directory'], restore=True))
PYRESTORE
```

All archive manifests preserve complete historical directories.
The final dispatcher, extension, and state check reports also have loose summary copies under `results/`.
The fresh-checkout audit needs restored files before it follows historical evidence paths.
