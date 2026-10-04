# Signed graph descriptors for human audit dossiers

`graph_dossier.py` closes a serialization gap in the human admission/context routes. Those frozen routes require a graph object, while the bound JSON/NPY loader returns ordinary dictionaries. A raw `BlockerGraph` also cannot pass the frozen JSON-based dossier fingerprint. The new adapter provides both interfaces using the same signed values.

This is preparation software. It creates no judgments, review testimony, source identities, or semantic evidence. Frozen graph construction, source acceptance, human sampling, response analysis, and external-review rules remain unchanged.

## Interface

```python
from phase9.graph_dossier import (
    graph_payload, hydrate_human_graphs, reviewed_dossier_sha256,
)
from phase8 import dossiers

# Pack author: graph must already come from the intended source/cache inputs.
frame_inputs["graph"] = graph_payload(existing_graph)
dossiers.dump_value(dossier, output_path)

# Reader: retain the frozen bound loader, then hydrate only human graph fields.
loaded = dossiers.load_value(output_path, base=stable_root)
loaded = hydrate_human_graphs(loaded)

# An actual external reviewer can bind the dossier to this digest.
# This call does not create or accept a review.
review_digest = reviewed_dossier_sha256(loaded)
```

`study_assembly._bound_json` invokes hydration after the frozen bound loader and before computing the dossier fingerprint. The scheduler includes this adapter's source in its code bindings. Other bundle shapes are unchanged.

Hydration transforms only `corpus_inputs[corpus]["graph"]`. Records, features, request manifests, provenance, responses, and review values are not changed. The returned top-level/frame dictionaries are copies; the underlying bound feature arrays retain their existing loader behavior.

The canonical graph descriptor has exactly these seven fields:

| Field | Representation |
| --- | --- |
| `schema` | `ccu-phase9-canonical-blocker-graph-1` |
| `record_ids` | Complete unique nonempty IDs in feature-row order |
| `source_ids` | Complete aligned source ownership |
| `priority_indices` | Full permutation of input row positions |
| `indptr` | Nonnegative int64-safe CSR row offsets |
| `indices` | Sorted, distinct, earlier-neighbor row indices within each row |
| `threshold_hex` | Exact canonical finite `float.hex()` value in `[-1,1]` |

`graph_payload(graph)` exports a plain JSON dictionary, sorting CSR rows while requiring the frozen graph/source binding to stay identical. It preserves feature-row alignment and the exact priority and threshold. It does not certify the origin of the graph or its features.

`CanonicalGraph` is a dictionary subclass with primitive tuple values. The frozen `task.fingerprint` normalizes those tuples to the same lists in the original JSON descriptor. Thus its signing digest is identical before serialization and after hydration. `reviewed_dossier_sha256` computes precisely the frozen narrow-review expression, excluding only `external_evidence_review`. It creates no statement about who reviewed the data.

The graph interface exposes the fields, properties, selection method, and accounting method of the frozen `BlockerGraph`. Ordinary mapping mutation and attribute replacement are refused; `copy`, shallow copy, and deep copy return the same immutable object. Arrays returned by the graph interface have immutable bytes-backed buffers, rather than a reversible readonly flag on an owning array. No hidden cached graph can diverge from the signed dictionary values.

The supported boundary is trusted Python code reading bound JSON descriptors, not an adversarial Python-object sandbox. Python's explicit base-class mutation methods can bypass an overridden dictionary method. Even then the graph interface derives from the changed signed fields rather than a stale hidden graph; the fingerprint and frozen graph binding change. Structural validation and accepted-graph comparison still apply. Do not treat a descriptor checksum as proof of collection or source authenticity.

## Binding and acceptance

Construction validates record/source alignment, integer types excluding booleans, integer range, priority permutation, all CSR offsets and indices, strict earlier-neighbor order, canonical row ordering, and the exact threshold representation. It also invokes the frozen request module's graph validator.

The frozen `human_job` independently rebuilds the graph from its accepted source/cache bundle and compares the descriptor graph's full frozen binding. It also compares features, records, and complete requests before regenerating the sampling frame. This check must remain in place: a structurally valid descriptor alone is insufficient. Original source acceptance, genuine-source requirements, all-corpus primary requirements, genuine human collection, and external review are unchanged.

Metadata storage is `O(N+E)`, and JSON loading plus tuple conversion creates ordinary in-memory metadata. Graph-array or method access reconstructs and validates `O(N+E)` metadata and allocates immutable graph arrays. Scalar and tuple properties read constructor-validated values directly, avoiding a graph scan for every threshold access in the frozen human-frame loop. This is shared preparation/human-analysis overhead, not repair runtime, constant-memory streaming, or native-scale feasibility evidence. Original feature caches and OS page-cache costs remain additional.

## Focused verification

```bash
python3 empirical_execution/phase9/check_graph_dossier.py \
  --out empirical_execution/phase9/results/graph_dossier_final
```

The checker refuses an existing output directory and snapshots its exact sources before execution. It uses the existing Civil40 natural-text lexical fixture and its existing unknown-singleton engineering ownership. It does not relabel those identifiers as genuine sources or the lexical features as semantic embeddings.

Checks cover canonical JSON/fingerprint/digest parity, complete graph/source identity, immutable arrays and copy behavior, malformed descriptors, selection parity, bound JSON/NPY loading, actual frozen admission/context frame reconstruction, and both frozen human execution routes with blank forms. Changed valid source ownership or priority is rejected by the frozen comparison with the accepted engineering graph; changed feature-file bytes are rejected by the frozen bound loader.

Both positive routes must end at `awaiting_real_human_responses` with `primary_output_accepted=false`. No authentic primary source gate is mocked, no human response is completed, and no external-review testimony is created. The temporary blank packs are discarded after checking; the checker and source-bound receipt preserve their reproducible construction and assignment counts without publishing another copy of the natural text.
