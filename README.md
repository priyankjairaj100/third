# Unlearning What Was Never Trained

Research workspace for **counterfactual repair after semantic data curation**, targeting ACL 2027.

**Start a new chat with [RESUME_CONTEXT.md](RESUME_CONTEXT.md).** It records the current state, user constraints, evidence boundaries, and next steps. [THEORY_AND_RESULTS_HANDOFF.md](THEORY_AND_RESULTS_HANDOFF.md) provides the deeper theory and results history.

## Local empirical execution — 5 October 2026

The user now runs empirical work on their own machine and sends results back here.
Start with **[local_run/START_HERE.md](local_run/START_HERE.md)** and give the local LLM **[LOCAL_LLM_BRIEF.md](local_run/LOCAL_LLM_BRIEF.md)**.
The **[exact empirical program](local_run/EMPIRICAL_PROGRAM.md)** covers all 21,332 registered jobs, preparation/human gates, statistics and reporting.
Use the [execution guide](local_run/EXECUTION.md) and [results-return workflow](local_run/RETURN_RESULTS.md).
This changes execution ownership, not the scientific protocol or frozen implementations.
Original corpora, model assets and genuine human responses are still required locally.
No primary experiments were run while preparing this handoff.

## Scientific status

Phase 10 proves finite-format memory bounds and implements exact sequential repair on the constructed family.
Read its [checkpoint](empirical_execution/phase10/CHECKPOINT.md) for verified scope.
Read the [theory](empirical_execution/phase10/FINITE_WORD_THEORY.md), [scorer proof](empirical_execution/phase10/SCORER_TRANSFER.md), and independent reviews together.
The [literature follow-up](empirical_execution/phase10/LITERATURE_FOLLOWUP.md) narrows the novelty claim using newly retrieved primary material.
The [remaining-task ledger](empirical_execution/phase10/REMAINING_TASKS.md) separates completed qualification from missing empirical evidence.
Phase 9 retains multi-family experiment assembly and resource qualification.
Phase 8 retains pinned archive exports, streaming replay, and staged candidate construction.
Phase 7 retains the calibration CLI, explicit policy dispatcher, and native work-count preflight.
Phase 6 implements the experiment routes and adds source replay, measurement, and inference checks.
It resolves the 21 prior recipe obligations prospectively.
The final results and limits are in the [completion ledger](empirical_execution/phase6/COMPLETION_LEDGER.md).

**The primary semantic study has not started.**
Original source-rich corpora, real encoder assets, and genuine human responses remain unavailable.
See [required inputs](empirical_execution/phase6/REQUIRED_INPUTS.md).
No external compute or human collection was started.

The available natural checks reuse Civil100 with lexical features.
They verify software within that scope.
They do not establish semantic task value, source withdrawal effects, or a publication speedup.
The earlier memory finding remains unfavorable to large summaries.
Compact eligible payload remains the essential comparator.

The [Phase 7 README](empirical_execution/phase7/README.md) gives the current command-line entry points.
The [Phase 6 README](empirical_execution/phase6/README.md) identifies current code and result files.
The [machine-readable status](empirical_execution/CURRENT_STATUS.json) separates completed software from missing scientific evidence.
Earlier phases remain unchanged.
Their results and failures retain their original scope.

## Read in this order

1. [RESUME_CONTEXT.md](RESUME_CONTEXT.md) — exact continuation entry point.
2. [THEORY_AND_RESULTS_HANDOFF.md](THEORY_AND_RESULTS_HANDOFF.md) — theory, algorithm evolution, experiments, and unresolved work.
3. [Current machine-readable status](empirical_execution/CURRENT_STATUS.json).
4. [Empirical protocol source](output/empirical_program/counterfactual_curation_empirical_protocol.tex) and [study design](output/empirical_program/study_design.json).
5. [Phase 3 preparation README](empirical_execution/phase3/README.txt), [prospective amendment](empirical_execution/phase3/PREPARATION_AMENDMENT.txt), and [independent review](empirical_execution/phase3/independent_review.txt).
6. [Phase 6 implementation](empirical_execution/phase6/README.md), [completion ledger](empirical_execution/phase6/COMPLETION_LEDGER.md), and final independent audit. Earlier phases remain historical context.

The original protocol's narrative resolves omissions in the compact JSON. Later explicit corrections and qualifications supersede older drafts only within their stated scope. Historical results remain historical.

## Project map

| Path | Contents |
| --- | --- |
| `local_run/` | Local-machine handoff, exact experiment matrix, preparation/execution guides, launcher, report exporter and independent review |
| Root `*theory*`, `round2_*`, `round3_*` | Original theory development, verification scripts, and successive qualifications |
| Root `empirical_*.md` | Detailed empirical design, statistical and reviewer-oriented planning |
| `output/pdf/` | Theory, empirical protocol, execution, memory, strict mode, Phase 2 and Phase 3 reports; available LaTeX sources |
| `output/*.zip` | Previously delivered reproducibility and execution packages |
| `output/empirical_program/` | Protocol, study configuration, claim ledger, schemas and preflight |
| `empirical_execution/ccu/` | Curation, moments, compressed summaries, exact canonical state and certified ridge code |
| `empirical_execution/results/` | Initial natural-preview engineering results |
| `empirical_execution/memory_results/` | Memory-repair experiments and audits |
| `empirical_execution/canonical_results/` | Exact canonical checkpoints, certificates and independent audits |
| `empirical_execution/phase2/` | Downstream consequence development experiment, predictions, analyses and historical intake |
| `empirical_execution/phase3/` | Source panels, shared scorer, calibration, exact quality gate, intake v2, CLI and audits |
| `empirical_execution/phase4/` | Original-schema adapters, offline encoding, task choices, request manifests, engineering comparison, systems harness and explicitly tracked unfinished integrations |
| `empirical_execution/phase5/` | Compact and summary method services, kernel access boundary, numerical/convex/refit/replication/analysis integrations, natural development matrix and final review |
| `empirical_execution/phase6/` | Complete routes, source acceptance, precision variants, measured workers, exact verification, human inference, and independent review |
| `empirical_execution/phase7/` | Calibration CLI, explicit policy dispatch, native resource preflight, and acquisition map |
| `empirical_execution/phase8/` | Pinned archive export, streaming replay, staged dossiers, scoped observed policy selection, and remaining-task ledger |
| `empirical_execution/phase9/` | Realizable feature-scale memory theorem, novelty audit, multi-family assembly, broader resource evidence, and independent reviews |
| `empirical_execution/phase10/` | Stored-FP32 construction, exact sequential bit codec, independent reviews, and literature follow-up |
| `empirical_execution/archive/` | Superseded pilot outputs retained for provenance |
| `BACKUP_MANIFEST.json` | File sizes, SHA256 checksums and backup exclusions |
| `BACKUP_AUDIT.md` | Backup scope and public-data handling review |

## Quick local verification

The recorded runtime was Python 3.12.14 with CPU execution. Dependencies are pinned in `empirical_execution/requirements.txt`. These commands require no model download and do not launch external compute:

```bash
python3 tools/verify_backup.py
python3 empirical_execution/phase3/check_workflow.py
python3 empirical_execution/phase3/check_reference_graph.py
python3 empirical_execution/phase3/audit_finite_population_independent.py
python3 empirical_execution/phase3/audit_scoring_and_pack_independent.py
python3 empirical_execution/phase3/audit_guards_independent.py
python3 -m empirical_execution.phase4.check_model_selection
python3 -m empirical_execution.phase4.check_systems
```

Read the phase-specific instructions before running long experiments. Several checks write result files; run in an isolated checkout when preserving the original snapshot. The compiled GMP helper is platform-dependent; its source and build/fallback instructions are included.

For continuation, acquire and accept the actual source-rich files and pinned local semantic assets, then execute the blinded selection/independent validation workflow with genuine humans. Use Phase 10 for the current checkpoint and remaining tasks. Use Phase 9 for experiment assembly. Use Phase 8 for original preparation. The frozen Phase 6 acceptance and experiment routes remain authoritative. Completed code and a checksum inventory do not establish scientific input authenticity or complete these experiments.

## Backup scope

Authored research sources, code, reports, configurations, result files, audits, earlier packages and continuation documentation are preserved. Temporary renders, interpreter caches and redundant extraction folders are omitted. News article bodies and their connector-response copies remain excluded from public redistribution under the existing project packaging policy; acquisition metadata and recorded results remain available. The manifest documents exclusions and available checksums. No credentials or unrelated conversation material belong in this repository.

This repository is the continuation location requested by the user. Keep handoff/status files current and commit meaningful milestones here during subsequent authorized project work. A GitHub commit is a research checkpoint, not a declaration that the primary study is complete.
