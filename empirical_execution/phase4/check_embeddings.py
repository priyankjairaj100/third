#!/usr/bin/env python3
"""Software fixtures only: this audit does not generate semantic embeddings."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from phase4 import embeddings as e


class TokenizerFixture:
    """Deliberately nonlinguistic fixture. Integers spell content length only."""
    def encode(self, text, add_special_tokens=False, truncation=False):
        assert not add_special_tokens and not truncation
        return [91, 92] if text == "query: " else list(range(1, int(text) + 1))

    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return 2

    def prepare_for_model(self, ids, **kwargs):
        assert kwargs["truncation"] is False and kwargs["padding"] is False
        ids = [9001] + ids + [9002]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def main():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def reject(name, function, exception=ValueError):
        try:
            function()
        except exception:
            checks.append(name)
            return
        raise AssertionError("Did not reject: " + name)

    tokenizer = TokenizerFixture()
    for count, expected in [(1, [1]), (256, [256]), (257, [256, 1]), (1025, [256, 256, 256, 256, 1])]:
        for encoder in (e.E5, e.MPNET):
            chunks, audit = e.chunk_inputs(tokenizer, str(count), encoder, 512)
            prefix = [91, 92] if encoder == e.E5 else []
            reconstructed = []
            for chunk in chunks:
                check(f"prefix_specials_{count}_{encoder}_{chunk['content_start']}",
                      chunk["input_ids"][:1 + len(prefix)] == [9001] + prefix
                      and chunk["input_ids"][-1] == 9002)
                reconstructed.extend(chunk["input_ids"][1 + len(prefix):-1])
            check(f"all_content_exactly_once_{count}_{encoder}", reconstructed == list(range(1, count + 1)))
            check(f"weights_{count}_{encoder}", audit["chunk_content_tokens"] == expected)
            check(f"first_window_counts_{count}_{encoder}",
                  audit["model_first_window_omitted_tokens"] == max(0, count - 512 + len(prefix) + 2))
    reject("zero_token_text_refused", lambda: e.chunk_inputs(tokenizer, "0", e.E5, 512))
    reject("no_silent_capacity_truncation", lambda: e.chunk_inputs(tokenizer, "257", e.E5, 259))
    reject("unknown_encoder_refused", lambda: e.chunk_inputs(tokenizer, "1", "unknown", 512))

    means = np.array([[3, 0, 0], [0, 4, 0], [0, 0, 5]], dtype=np.float32)
    calls = []
    def backend(chunks):
        calls.append([c["content_start"] for c in chunks])
        return np.stack([means[c["content_start"] // 256] for c in chunks])
    pooled, detail = e.encode_document(tokenizer, backend, "513", e.E5, 512, batch_size=2, dimension=3)
    independent = np.array([256, 256, 1], dtype=np.float64)
    independent = (independent / np.sqrt(np.dot(independent, independent))).astype(np.float32)
    check("weighted_normalized_chunks_agree_with_independent_value", np.array_equal(pooled, independent))
    check("only_intrarecord_batches", calls == [[0, 256], [512]])
    check("document_vector_stored_fp32", pooled.dtype == np.float32)
    check("no_chunk_lost_in_final_batch", detail["batch_count"] == 2 and detail["content_tokens"] == 513)
    e.encode_document(tokenizer, backend, "1", e.E5, 512, batch_size=2, dimension=3)
    replay, _ = e.encode_document(tokenizer, backend, "513", e.E5, 512, batch_size=2, dimension=3)
    check("other_document_does_not_change_chunk_grouping", np.array_equal(pooled, replay))
    reject("zero_vector_refused", lambda: e.normalize_reference(np.zeros(3)))
    reject("nan_vector_refused", lambda: e.normalize_reference(np.array([np.nan, 1])))
    reject("backend_fp64_refused", lambda: e.encode_document(tokenizer,
        lambda parts: np.ones((len(parts), 3), dtype=np.float64), "1", e.E5, 512, dimension=3))
    reject("backend_wrong_dimension_refused", lambda: e.encode_document(tokenizer,
        lambda parts: np.ones((len(parts), 2), dtype=np.float32), "1", e.E5, 512, dimension=3))
    reject("zero_batch_refused", lambda: e.encode_document(tokenizer, backend, "1", e.E5, 512, batch_size=0, dimension=3))

    with tempfile.TemporaryDirectory(prefix="ccu_software_asset_fixture_") as temporary:
        directory = Path(temporary)
        model, tok = directory / "model", directory / "tokenizer"
        model.mkdir(); tok.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"SOFTWARE FIXTURE: NOT MODEL WEIGHTS")
        (tok / "tokenizer.json").write_text("{}")
        versions = {key: "software-fixture-only" for key in e.VERSIONS}
        manifest = e.asset_inventory(model, tok, e.E5, "a" * 40, "b" * 40, versions)
        e.verify_assets(manifest, model, tok, verify_runtime=False)
        checks.append("byte_manifest_round_trip_without_runtime_or_model_load")
        reject("fixture_cannot_pass_production_runtime_gate", lambda: e.verify_assets(manifest, model, tok), RuntimeError)
        (model / "model.safetensors").write_bytes(b"MUTATED FIXTURE")
        reject("mutated_weights_refused", lambda: e.verify_assets(manifest, model, tok, verify_runtime=False))
        (model / "unexpected.json").write_text("{}")
        reject("unlisted_asset_refused", lambda: e.verify_assets(manifest, model, tok, verify_runtime=False))
        reject("floating_revision_refused", lambda: e.asset_inventory(model, tok, e.E5, "main", "b" * 40, versions))
        (model / "custom.py").write_text("raise RuntimeError('never execute')")
        reject("custom_script_refused", lambda: e.local_inventory(model))
        (model / "custom.py").unlink()
        (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "../../outside.safetensors"}}))
        external = e.asset_inventory(model, tok, e.E5, "a" * 40, "b" * 40, versions)
        reject("external_shard_refused", lambda: e.verify_assets(external, model, tok, verify_runtime=False))
        (model / "model.safetensors.index.json").unlink()
        (tok / "tokenizer_config.json").write_text(json.dumps({"vocab_file": "/tmp/unbound-vocab.json"}))
        external = e.asset_inventory(model, tok, e.E5, "a" * 40, "b" * 40, versions)
        reject("external_tokenizer_asset_refused", lambda: e.verify_assets(external, model, tok, verify_runtime=False))

    result = {"passed": True, "checks": len(checks), "check_names": checks,
        "evidence_role": "software_fixtures_only_not_empirical_data",
        "semantic_model_forward_executed": False, "semantic_embedding_cache_created": False,
        "runtime_versions_observed": e.installed_versions(),
        "actual_offline_transformer_adapter_validation": "blocked_missing_local_model_assets_and_runtime",
        "production_code_sha256": e.file_hash(e.__file__), "audit_code_sha256": e.file_hash(__file__)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/embeddings_software_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
