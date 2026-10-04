# Unlearning What Was Never Trained

Research workspace for **counterfactual repair after semantic data curation**, targeting ACL 2027.

**Start a new chat with [RESUME_CONTEXT.md](RESUME_CONTEXT.md).** It records the current state, user constraints, evidence boundaries, and next steps. [THEORY_AND_RESULTS_HANDOFF.md](THEORY_AND_RESULTS_HANDOFF.md) provides the deeper theory and results history.

## Current status — 4 October 2026

Theory, algorithm implementations, memory-repair qualifications, strict finite-precision certification, natural-text development experiments, and the Phase 3 preparation workflow are available here. Phase 4 adds original-schema parsers, offline semantic-encoding code, calibration-only task selection, frozen request generation, a fixed-input comparison runner and a fresh-process measurement harness. **The primary semantic empirical study has not started.** Authentic source-rich archives, pinned semantic model/cache assets and usable transformer runtime, and genuine independent human ratings remain missing. Parser software is implemented; acceptance against the actual archives is pending. Primary optimized methods/access isolation, full-refit integration and the complete pre-execution lock remain unfinished.

The saved empirical results use small, reused natural-text previews and lexical features. They are engineering/development evidence, not E5/MPNet results or confirmation of an ACL paper's empirical claims.

The final Phase 4 core replay has **80 checkpoints, zero failures and 54 zero-admission rows**. An independent reconstruction checks **480 saved heads**. The separate convex pilot completes **12 checkpoints and 25 rigorous optimizer-error certificates**, preserving its earlier failed run. News calendar/encoding bridges, per-encoder calibration and the admission-pair pack are implemented and checked; the latter contains **72 blank assignments and no human responses**. This advances preparation and integration verification; it establishes neither a systems speedup nor semantic utility. [Phase 4 README](empirical_execution/phase4/README.md) and [TODO ledger](empirical_execution/phase4/TODO.json) separate completed software, completed engineering runs, missing inputs and unfinished research implementation. Final consolidated review and repository checkpoint remain tracked separately.

## Read in this order

1. [RESUME_CONTEXT.md](RESUME_CONTEXT.md) — exact continuation entry point.
2. [THEORY_AND_RESULTS_HANDOFF.md](THEORY_AND_RESULTS_HANDOFF.md) — theory, algorithm evolution, experiments, and unresolved work.
3. [Current machine-readable status](empirical_execution/CURRENT_STATUS.json).
4. [Empirical protocol source](output/empirical_program/counterfactual_curation_empirical_protocol.tex) and [study design](output/empirical_program/study_design.json).
5. [Phase 3 preparation README](empirical_execution/phase3/README.txt), [prospective amendment](empirical_execution/phase3/PREPARATION_AMENDMENT.txt), and [independent review](empirical_execution/phase3/independent_review.txt).
6. [Phase 4 execution/preparation README](empirical_execution/phase4/README.md), [TODO ledger](empirical_execution/phase4/TODO.json), and module-specific instructions/audits.

The original protocol's narrative resolves omissions in the compact JSON. Later explicit corrections and qualifications supersede older drafts only within their stated scope. Historical results remain historical.

## Project map

| Path | Contents |
| --- | --- |
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

For continuation, acquire and accept the actual source-rich files and pinned local semantic assets, then execute the blinded selection/independent validation workflow with genuine humans. In parallel, finish the primary optimized/access-contract/full-refit integrations listed in the TODO ledger. Completed preparation code must not be used to mark those scientific dependencies complete.

## Backup scope

Authored research sources, code, reports, configurations, result files, audits, earlier packages and continuation documentation are preserved. Temporary renders, interpreter caches and redundant extraction folders are omitted. News article bodies and their connector-response copies remain excluded from public redistribution under the existing project packaging policy; acquisition metadata and recorded results remain available. The manifest documents exclusions and available checksums. No credentials or unrelated conversation material belong in this repository.

This repository is the continuation location requested by the user. Keep handoff/status files current and commit meaningful milestones here during subsequent authorized project work. A GitHub commit is a research checkpoint, not a declaration that the primary study is complete.
