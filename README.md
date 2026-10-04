# Unlearning What Was Never Trained

Research workspace for **counterfactual repair after semantic data curation**, targeting ACL 2027.

**Start a new chat with [RESUME_CONTEXT.md](RESUME_CONTEXT.md).** It records the current state, user constraints, evidence boundaries, and next steps. [THEORY_AND_RESULTS_HANDOFF.md](THEORY_AND_RESULTS_HANDOFF.md) provides the deeper theory and results history.

## Current status — 4 October 2026

Theory, algorithm implementations, memory-repair qualifications, strict finite-precision certification, natural-text development experiments, and the Phase 3 preparation workflow are available here. **The primary semantic empirical study has not started.** Original source-rich corpus assets, pinned semantic embeddings or usable model/runtime files, and genuine independent human ratings are missing. Corpus-specific parsing and the rest of the pre-execution lock also remain unfinished.

The saved empirical results use small, reused natural-text previews and lexical features. They are engineering/development evidence, not E5/MPNet results or confirmation of an ACL paper's empirical claims.

## Read in this order

1. [RESUME_CONTEXT.md](RESUME_CONTEXT.md) — exact continuation entry point.
2. [THEORY_AND_RESULTS_HANDOFF.md](THEORY_AND_RESULTS_HANDOFF.md) — theory, algorithm evolution, experiments, and unresolved work.
3. [Current machine-readable status](empirical_execution/CURRENT_STATUS.json).
4. [Empirical protocol source](output/empirical_program/counterfactual_curation_empirical_protocol.tex) and [study design](output/empirical_program/study_design.json).
5. [Phase 3 preparation README](empirical_execution/phase3/README.txt), [prospective amendment](empirical_execution/phase3/PREPARATION_AMENDMENT.txt), and [independent review](empirical_execution/phase3/independent_review.txt).

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
```

Read the phase-specific instructions before running long experiments. Several checks write result files; run in an isolated checkout when preserving the original snapshot. The compiled GMP helper is platform-dependent; its source and build/fallback instructions are included.

## Backup scope

Authored research sources, code, reports, configurations, result files, audits, earlier packages and continuation documentation are preserved. Temporary renders, interpreter caches and redundant extraction folders are omitted. News article bodies and their connector-response copies remain excluded from public redistribution under the existing project packaging policy; acquisition metadata and recorded results remain available. The manifest documents exclusions and available checksums. No credentials or unrelated conversation material belong in this repository.

This repository is the continuation location requested by the user. Keep handoff/status files current and commit meaningful milestones here during subsequent authorized project work. A GitHub commit is a research checkpoint, not a declaration that the primary study is complete.
