# WCEP encoding and inherited News curator

`replication_encoding.py` supplies the executable connection between the original-event WCEP parser/panel and the shared encoder/graph workflows. It defines an explicit WCEP cache schema; it does **not** manufacture a fixed-source-split or leakage-guard audit for the Phase 3 loader.

The default replication is record-only, as permitted when genuine original URL/domain evidence is unavailable. Repeated article instances remain separate, including repeated URLs or strings. Event summaries, categories, summaries' annotations and event membership never enter the model input or provide duplicate gold labels. The title/body rule is exactly the existing News adapter rule.

`load_wcep_panel(adapter_dir, panel_dir)` reopens and verifies the sealed parser audit and panel, file hashes, current parser/text-adapter code, whole-event hash-prefix choice, event/article counts, ordered instance IDs, snapshot-file/event/ordinal ID derivation, title/body construction, source singleton convention, and per-record event-date/collection lineage. It compares full panel rows against the selected original-adapter rows. An empty panel, incomplete parse, altered bytes or a partially selected boundary event is rejected. File consistency does not establish external snapshot authenticity or completeness.

`encode_wcep` reuses the **unchanged** `phase4.embeddings.verify_assets`, `LocalTransformer` and `encode_document` implementations. E5 and MPNet use the identical pinned all-256-content-token-chunk contract: local safetensors and tokenizer, record-local batches, attended-token masked means, per-chunk normalization, content-token-weighted pooling, final normalization and FP32 storage. The adapter loads a genuine local transformer; there is no fake-backend option. Missing or mismatched local assets/runtime fail **before** an output directory is created. Failure during actual encoding leaves `FAILED.json` and no successful cache manifest.

Outputs contain `vectors.npy`, ordered IDs, the copied asset inventory, per-record chunk logs, resource/pooling summary and a sealed WCEP cache manifest. The manifest binds the entire panel and records, source code, model/tokenizer revisions, exact asset inventories, preprocessing, batch/thread settings and runtime versions. It remains pending News-curator inheritance. Memory accounting explicitly includes the resident selected adapter/panel records and event ledger (unselected adapter records are streamed), model/tokenizer, full token/chunk list for one document, batch activations and memory-mapped output; no constant-memory claim is made.

`inherit_news_threshold` is a separate operation. It reopens the complete prepared CC-News cache through the frozen versioned loader, requires its authentic-input *declared* confirmatory calibration scope, matches the **same complete asset manifest, model/tokenizer revision, runtime, batch and pooling contract**, and reconstructs the exact calibration frame. It recomputes threshold selection from the supplied selection responses and recomputes quality validation from the fresh validation responses, requiring an exact match to the supplied saved locks and a passed declared-scope quality gate. Blank/absent responses cannot pass. It seals a WCEP inherited-curator lock containing the unchanged News threshold and all input hashes. There is no WCEP threshold-selection or retuning operation.

`load_inherited_graph_inputs` provides aligned WCEP rows, the FP32 memory-mapped cache and inherited threshold for the exhaustive graph builder. It refuses a missing or mismatched inherited lock and checks the inherited fixed scorer's source/Python/NumPy contract. The lock establishes internal artifact consistency, not human authenticity. News calibration coverage is not asserted to transfer as a WCEP semantic precision guarantee; WCEP is the explicitly event-enriched replication arm. The record-only branch does not require a fabricated domain source arm.

## Commands

Create the same asset manifest used for News with the existing Phase 4 `embeddings inventory` CLI. Encode the parsed whole-event panel:

```bash
python3 -m empirical_execution.phase5.replication_encoding encode \
  --adapter-dir /local/wcep/parsed --panel-dir /local/wcep/panel \
  --asset-manifest /local/assets.json --model-dir /local/model \
  --tokenizer-dir /local/tokenizer --out /local/wcep/e5-cache \
  --batch-size 8 --threads 1
```

The separate `inherit-news --help` lists the complete News cache, preparation, selection-response and validation-response dossier. `check-cache` validates the cached artifacts without loading the transformer, and `graph-inputs` requires the inherited lock. No command fetches files or starts external compute.

## Actual validation and remaining evidence

`check_replication_encoding.py` exercises transient, explicitly named **software schema fixtures** and negative missing-runtime checks for both encoders. It verifies whole-event preservation, ordered IDs, text/event isolation, resealed-but-inconsistent input rejection, absent inherited-lock rejection and fail-before-output behavior. It creates no successful fake semantic cache, no completed human responses and no passed quality dossier. The underlying chunk math has the earlier Phase 4 software checks; the shared real transformer backend is still unavailable in this environment.

Thus the WCEP bridge is implemented, while real WCEP encoding and actual inherited-threshold execution remain blocked by authentic original WCEP/CC-News records, real pinned model/runtime assets and completed genuine News calibration/validation ratings. A successful authentic-asset acceptance run remains necessary; the negative software checks do not claim one occurred.
