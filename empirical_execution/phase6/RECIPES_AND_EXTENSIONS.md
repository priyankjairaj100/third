# Prospective recipes and executable extensions

This release resolves the 21 earlier recipe decisions.
It does not supply missing research inputs.
Every new choice precedes the primary semantic study.
Historical Phase 5 files remain unchanged.

The compiler retains all 16,912 Phase 5 experiment jobs.
It replaces only the 21 unresolved placeholder jobs.
It adds 4,420 explicit extension jobs.
The complete registry contains 21,332 jobs.

Five recipes specify policies without additional jobs.
These cover method scope, matched controls, numerical acceptance, measurements, and the optional Civil comparison.
The optional Civil comparison remains inactive.
This choice protects the required experiment program.

| Extension | Jobs | Fixed allocation |
|---|---:|---|
| Source sampling by mass | 1,536 | 64 weighted source paths and 64 matched record paths on each structure panel |
| One percent horizon | 1,344 | 64 R, 32 U, and 16 A paths on each structure panel |
| Excluded blocker stress | 384 | 32 paths on each structure panel |
| Fresh graph audit | 48 | 8 R, 4 S, 2 U, and 2 A paths per core 10k panel |
| Graph envelope | 288 | 32 R, 32 S, and 32 matched paths per core 10k panel |
| Cached graph oracle | 160 | First eight R and S paths, five repetitions, two labeled corpora |
| Warm service | 160 | Same timing paths and repetitions |
| Persistence after every release | 160 | Same timing paths and repetitions |
| Negative controls | 320 | Separate exact-text, greedy, and no-curation paths |
| Initial utility | 2 | Four fixed learner references per labeled corpus |
| Count normalization diagnostic | 1 | Civil R000 at its final feasible checkpoint |
| News chronology | 1 | One stream of complete recorded days |
| Human threshold analyses | 12 | Five corpus/encoder pairs, plus two threshold sensitivity validations |
| Human admission analysis | 1 | The complete balanced pair sample |
| Human context analyses | 2 | Separate former-neighborhood and nearest-survivor audits |
| Statistical analysis | 1 | Complete planned cells and source uncertainty |

Source sampling uses exact integer tickets.
Each draw selects a remaining source proportional to its original record count.
Unknown source singletons never enter this sampling frame.
Matched record controls use their own sufficient initial horizon.

The one percent horizon equals `ceil(N/100)`.
Its checkpoints come from `[1, 8, 32, 128, K]`.
The algorithm clips these values and removes duplicates.
It constructs a separate state before any request.

The News stream preserves complete days.
It stops before a day exceeds the cumulative budget of 128 records.
It neither divides that day nor skips it.
Its outcomes remain descriptive.

The negative controls use their own excluded records.
Exact-text curation uses normalized text and fixed priority.
Greedy curation tests neighbors that it selected earlier.
Neither control replaces the primary curator.
Empty control pools remain visible.

`recipes.py` supplies the machine-readable rules.
`extensions.py` supplies executable routes for every added job class.
The dispatcher verifies inputs before it starts these routes.
It records missing inputs, failed gates, and unavailable arms.
A missing dataset cannot become an empty successful study.
Compatible jobs share an accepted in-memory input snapshot.
The cache key also binds the corpus, panel, design, resource policy, and base directory.
Every job rechecks the loaded arrays and metadata.
Workers separately verify their isolated input packages.
Original archives and encoder files are not reread during these jobs.
Their accepted historical hashes remain the provenance binding.
This scope does not claim protection against a malicious filesystem.

The cached graph oracle has a separate implementation name.
Its reports identify `O-G-cached` and preserve the requested `O-G` identity.
Its permitted state contains the exhaustive graph and all learner rows.
Shared preparation still includes graph construction.

All eight FP32 variants remain required.
Their reports identify their actual state precision.
FP32 results describe an accuracy and memory tradeoff.
They do not establish exact FP64 target recovery.

The common numerical limit remains `eta <= 1e-10`.
CG starts from zero and permits at most `10*d` iterations per output.
The code does not relax this limit after a failure.
Residual verification alone does not supply an interval certificate.

Human threshold analysis reconstructs the deterministic pair samples.
Primary use also requires original-source and cache replay.
Human admission analysis reconstructs its complete frame from accepted upstream inputs.
It reports completed, skipped, substantive, and missing assignments separately.
Partial collection does not count as completed collection.
No person received an annotation request from this work.

Human intervals preserve missing judgments and missing corpora.
The pair and context families each receive error probability `1/40`.
Their simultaneous coverage concerns the fixed recorded sampling design.
It does not establish semantic truth or authenticity of human collection.

Statistical analysis binds measurements to saved dispatcher outputs.
It retains every planned cell and every failed cell.
It recomputes prediction metrics from saved arrays.
It separates whole-trajectory resampling from whole-source resampling.
Timing repetitions remain within their original trajectory.
They do not become additional request trajectories.

`precision_analysis` evaluates fixed candidate allocations using disjoint development effects.
It reports conditional empirical power and projected interval width.
It does not change the registered allocation automatically.
Zero development variance cannot establish adequate power.
The simulation interval covers simulation randomness only.

`derive_resource_policy` uses actual local memory and development timings.
It assigns the same limits to every method.
No external compute is assumed.
The policy preserves infeasible native cells.

`select_verification_budget` can qualify a larger exact-verification budget.
It requires dense measurements with the target dimension and output count.
Sparse observations cannot qualify dense native work.
The forecast uses four times the largest observed cost per coordinate.
Its memory estimate includes the additional input arrays.
The forecast remains an estimate, not a worst-case guarantee.
A forecast failure is not an observed timeout.

The supporting policies fix the ANN parameters and the full-refit defaults.
The full-refit panel uses whole training sources outside both main and replication panels.
Its hash salt is `ccu-v1-refit-panel`.
A shortage remains a shortage.
The code does not reuse sources to hide it.

## Current execution scope

The route checks use 40 actual Civil comments and 20 separate evaluation comments.
The features remain lexical.
The evaluation subset is a software check, not an untouched benchmark.

Natural route checks cover the structural, learner, oracle, and persistence routes.
They also reconstruct and analyze blank human packs.
No genuine source withdrawal ran on this preview.
The chronological path has a symbolic schema check only.
Human threshold collection and the primary source bootstrap still need actual inputs.
Their absence remains explicit.

An early test expected a nonempty exact-text control pool.
The natural records had no such pool.
The algorithm correctly reported an unavailable arm.
The revised test accepts that outcome and preserves the failed test record.

## Interfaces

```python
from phase6.recipes import build_recipe_book, build_registry
book = build_recipe_book()
registry, jobs = build_registry()
```

```sh
PYTHONPATH=empirical_execution python3 -m phase6.recipes /absolute/new/export
PYTHONPATH=empirical_execution python3 empirical_execution/phase6/check_recipes.py \
  --output /absolute/new/recipe_checks.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=empirical_execution \
  python3 empirical_execution/phase6/check_extensions.py --out /absolute/new/extensions
```

Exports use new directories.
The code refuses to overwrite a saved export.

The final result directories identify their source hashes.
Earlier candidate directories remain historical evidence.

## Authoritative verification

The registry lives in `results/recipes_release/`.
The recipe checks passed 44 checks in `results/recipes_checks_final.json`.
The extension checks passed 48 checks in `results/extensions_release_v3/checks.json`.
The latter report lists evidence for all 16 extension classes.
Ten classes completed on actual Civil records with lexical features.
No class completed a primary semantic experiment.

The source sampling route preserves the absent genuine-source frame.
Chronology has a date-schema check and an explicit missing-input refusal.
Human routes reconstruct actual samples and preserve missing judgments.
Statistics uses saved natural predictions within an explicit software metadata fixture.
That fixture does not establish a genuine-source bootstrap result.
Greedy and no-curation controls completed.
The exact-text excluded pool was empty and remains visible.

`extensions_final` and `extensions_release_v2` preserve earlier completed attempts.
Later state-audit corrections superseded their global source bindings.
Their files remain unchanged.
