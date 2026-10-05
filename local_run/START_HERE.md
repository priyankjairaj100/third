# Run the empirical study locally

This is the execution handoff for the user's **5 October 2026** decision to run empirical work locally and report results back. All preserved project code, theory, protocols and historical results are in this repository. Original corpora, model weights, genuine human responses and new run outputs must be supplied locally; they are not in Git.

## Start in a fresh checkout

```bash
git clone https://github.com/priyankjairaj100/third.git
cd third
python3 tools/verify_backup.py
python3 local_run/run_local.py init
python3 local_run/templates/prepare_runtime.py --help
```

If you already have a checkout, preserve local changes and private outputs before updating it. Do not use a destructive reset. Run commands from the repository root. Read the environment instructions before installing packages: the numerical engineering environment is pinned, but a complete semantic runtime has not yet been qualified. The frozen encoder path is CPU FP32; do not silently substitute GPU encoding.

**Give your local LLM [`LOCAL_LLM_BRIEF.md`](LOCAL_LLM_BRIEF.md).** It contains the full assignment, current scientific status, authority order, evidence rules and exact entry points. Ask it to execute the stages below and maintain the results report. You need not reconstruct previous chats.

## The order to follow

| Stage | Your local program | Gate before advancing |
|---|---|---|
| 0. Inventory | Record actual hardware, package versions, free storage and input availability. Verify the repository. | Honest setup report; no assumed hardware or corpus access. |
| 1. Acquire and prepare | Obtain source-rich Civil Comments, the registered Stack Exchange archives, CC-News/WCEP, and pinned E5/MPNet assets. Export/adapt and replay actual provenance. | Authentic archives/source identities and deterministic semantic-cache replay. Preserve any unavailable archive as blocked. |
| 2. Calibrate | Produce blinded selection packs; collect genuine independent human judgments; choose the calibration-only threshold; produce fresh validation packs and collect judgments. Prepare source-disjoint panels and task choices. | Exact registered semantic-quality bound, leakage guards and frozen downstream choices. No fabricated ratings. |
| 3. Develop and qualify | Begin with a Civil/E5 development path, then cover the registered families' actual operation profiles. Check correctness, admissions, utility diagnostics and complete resource accounting. | Versioned fixes if needed; observed resource qualification, accountable evidence review and no tuning on confirmatory results. |
| 4. Freeze and execute | Assemble canonical study owners. Run the registered A–G core and H extension program in a complete Phase 9 accounting invocation. | Every planned job retained, including failures, blocked jobs, inapplicable arms and zeros. Final ledger hash bound to summary. |
| 5. Analyze and return | Build the registered statistics dossier from the finalized dispatch; run human/extension analyses when their real inputs exist. Report all requested estimands and limitations. | Source-aware paired inference, valid predeclared tests, complete failure accounting and an honestly reviewed compact report bundle. |

The exact scientific specification is **[`EMPIRICAL_PROGRAM.md`](EMPIRICAL_PROGRAM.md)**; the enumerated configuration is **[`EXPERIMENT_MATRIX.json`](EXPERIMENT_MATRIX.json)**. Do not reduce the campaign silently to the first pilot. The full registry has **21,332 jobs**, not 21,332 independent samples and not a guaranteed runtime.

Use **[`INPUTS_AND_PREPARATION.md`](INPUTS_AND_PREPARATION.md)** for source/model acquisition, environment and human collection. Use **[`EXECUTION.md`](EXECUTION.md)** for exact commands, launch templates, resource qualification, reviews and dispatch/analysis. These documents disclose genuine manual boundaries rather than pretending the scientific inputs already exist.

## Send results back

Follow **[`RETURN_RESULTS.md`](RETURN_RESULTS.md)**. A setup failure or a negative pilot is useful: send it with the precise missing input or failed gate. Do not wait for all jobs to succeed. The collector never includes source documents, model files or raw annotation forms by default, and it never turns a declared result into verified evidence.

At this handoff, the primary semantic study is unstarted. The goal of the local program is to determine whether the proposed phenomenon matters on natural data and which repair methods offer real benefits. A favorable paper outcome is not assumed.
