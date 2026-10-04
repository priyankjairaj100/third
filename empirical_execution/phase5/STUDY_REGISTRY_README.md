# Prospective study registry and input preflight

`study_registry.py` materializes the bounded A–G experiment design and preserves missing work as explicit planned jobs. It does not submit a preregistration, generate real request IDs before the panels exist, or authorize scientific execution from file presence. The original design's status remains `design_reviewed_not_acquired_or_preregistered`.

The initial export at `results/study_registry_release/` has **43 configuration groups and 16,933 planned jobs**, including matched-volume controls, timing repeats, refit seed repeats and **21 unresolved protocol obligations**. There are **158 required artifact keys**. None is an executed experiment; an empty input inventory verifies zero primary assets.

## Coverage

* A: Civil, Ask Ubuntu and CC-News; nested 10k/25k/100k/200k source-complete panels; 256 R, 256 S, 128 U, 32 A per panel.
* B: First Civil and Ask Ubuntu 10k panels; O-G/B-F at 256 R, 256 S, 128 U, 32 A. Before clipping this is 2,688 checkpoint releases per corpus per method.
* C: The same first labeled panels; all eight methods at the first 64 R, 64 S, 32 U, 16 A paths. Before clipping this is 704 releases per corpus per method. First 16 R and 16 S paths carry full-state rebuild obligations; first eight R and eight S get five fresh-process repetitions each.
* D: Three second disjoint core 10k panels, English StackExchange 10k and whole-event WCEP25k; 64 R, 64 S when genuinely available, 32 U. WCEP reuses frozen CC-News threshold/quality and does not require a new semantic split guard. Missing WCEP URLs/source validity keep its S and matched-S jobs unavailable while allowing record-only jobs; they never disappear from the accounting.
* E: Separate Civil/Ask Ubuntu 5k full-refit panels; own original fitted curator/head; 32 R, 32 S, 16 A; first two R/S paths require every service step. Five same-seed and three alternative-seed no-deletion fits remain separate jobs.
* F: Civil/Ask Ubuntu10k, 32 R and 32 S, with fresh optimum, eligible-payload retraining and certified convex repair.
* G: Fixed Civil10k dimension projections32/64/128/256, complete E5/MPNet curator/learner cross, priority seeds1/2, thresholds max(0,tau−.02) and min(1,tau+.02), lambda factors.1/10, FP32 state and common zero-start CG. News fixed calendar window adds ANN comparison and natural-window/source-panel comparison. Each named alternative has 32 shared R and 32 shared S, never a full factorial.

Each S trajectory additionally has a separately identified matched-volume R job with its own sufficient record horizon. It is not counted as an independent primary R trajectory. Main groups share a `request_pairing_key`; actual generated request manifests must establish the intended pairing. B/C link parent paths so repeated O-G/B-F outcomes can be reused without treating them as independent replication. E5/E5 boundary results similarly reuse the corresponding native cell.

A job is a paired trajectory with its listed method workers, not one row per method. `planned_checkpoint_units` stores the protocol's unclipped requests. Actual horizon, clipped/deduplicated checkpoints and actual request IDs stay unknown until the authentic source/record/excluded universes are locked. Empty or unavailable arms remain status records; they do not disappear from the registry.

## Underspecified obligations stay visible

The prose does not supply every implementation detail or allocation. The registry retains named unresolved jobs for mass-PPS, the separate1% horizon, excluded-blocker stress, chronological News withdrawals, fresh-graph audit arm allocation, cached-graph cost tier, utility references, negative controls, normalization-error diagnostic, conditional graph-envelope study, every-release persistence, cold/warm policy, optional official Civil splits, three human audit/collection branches, boundary/matched-R method scope, test-source statistics, primary numerical release gates and full dependency/read accounting.

This prevents a compact JSON specification from quietly dropping prose requirements. Their recipes must be completed prospectively where applicable; a script existing elsewhere may satisfy implementation but does not supply the missing scientific inputs. No arbitrary trajectory count has been invented for these unresolved jobs.

Three explicit prospective registry choices fill otherwise ambiguous *planning scope*: second panels cover all three core structural corpora; matched-volume controls carry the same listed methods at their own horizon; dimension/FP32/lambda/CG alternatives list all eight methods. Narrowing these expensive scopes requires a recorded amendment before confirmation. The registry makes these choices visible rather than calling them previously registered facts.

## Artifact preflight

`artifact_inventory_template.json` lists concrete dependencies by corpus, encoder, panel and global policy. They include original archives, parser/source reviews, post-guard panel and request manifests, pinned model/tokenizer assets, vector caches/manifests, actual selection/validation responses and locks, quality reports, task settings/test inputs (including MPNet-specific lambda/test caches), sensitivity-specific threshold quality audits, source-specific metadata, projections, SemDeDup runtime/configuration, numerical policies and resource locks.

For a supplied descriptor `{ "path": "relative/file", "sha256": "lowercase-64-hex" }`, preflight checks that the file exists inside the inventory root and verifies its actual bytes. Missing files, path escapes, absent hashes and altered bytes remain explicit failures. Every planned group/job remains in the output regardless of missing assets.

Passing this inventory check only means that bytes match a supplied checksum. It does **not** authenticate a corpus, reconstruct the encoder execution, validate source completeness, recompute calibration outcomes, prove independent human collection or establish the worker access/resource contract. Even a fully populated inventory has `execution_allowed: false` pending the external scientific/implementation acceptance. Capability declarations are recorded as unverified declarations; invented `primary_ready` or human-count fields cannot activate a run.

The existing Phase3 calibration machinery and Phase4 input inspectors remain separately necessary. The registry does not mutate their frozen contracts or pretend to replace their semantic checks. Actual collected human ratings remain zero.

## Commands and APIs

```sh
python3 empirical_execution/phase5/study_registry.py --output-dir /path/to/new/registry
python3 empirical_execution/phase5/study_registry.py --inventory /local/inputs/inventory.json --output-dir /path/to/new/preflight
python3 empirical_execution/phase5/check_study_registry.py --output-dir /path/to/new/check-output
```

Python APIs: `registry_groups(design)`, `build_registry() -> (registry, jobs)`, `preflight(registry, inventory, base_dir)`, `export(output_dir, inventory=None, base_dir=None)`.

`registry.json` holds group definitions, requirements, protocol/source hashes and a digest of the expanded jobs. `jobs.jsonl.gz` contains every individual planned job in deterministic gzip form; decompress with `gzip -dc`. `preflight.json` records group blockers and per-artifact hash status. `export_manifest.json` binds the generated files. The separate check report validates cardinalities, source conditions, shared-path links, timing/refit counts, missing-input retention, artifact corruption/path rejection and inability of declarations to authorize execution.

The initial version passes **41 software checks**. This is orchestration/inventory software evidence, not scientific confirmation or an empirical result.
