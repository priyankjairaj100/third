# Model input candidates

The required encoders remain E5 and MPNet.
This review did not acquire or execute either encoder.

`results/publisher_model_candidates.json` records immutable candidate revisions and published weight hashes.
These values came from the publishers' file metadata.
They do not constitute a verified local asset manifest.
The frozen study design remains unchanged.

| Encoder | Candidate revision | Published weights |
| --- | --- | --- |
| `intfloat/multilingual-e5-base` | `d128750597153bb5987e10b1c3493a34e5a4502a` | `model.safetensors`, displayed as 1.11 GB |
| `sentence-transformers/all-mpnet-base-v2` | `e8c3b32edf5434bc2275fc9bab85f82640a19130` | `model.safetensors`, 437,971,872 bytes |

Sources: [E5 immutable file metadata](https://huggingface.co/intfloat/multilingual-e5-base/blob/d128750597153bb5987e10b1c3493a34e5a4502a/model.safetensors)
and [MPNet immutable file pointer](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/e8c3b32edf5434bc2275fc9bab85f82640a19130/model.safetensors).

Import the configuration and complete tokenizer files from the same reviewed revision.
Use ordinary local files instead of cache symlinks.
The existing encoder requires safetensors weights.
It rejects custom Python code and external configuration paths.
It does not need duplicate pickle weights, ONNX models, or OpenVINO models.
Avoid downloading every alternate model format without a purpose.

The local runtime must provide the five versions required by the existing asset manifest.
Those packages are NumPy, torch, transformers, tokenizers, and safetensors.
Tokenizer-specific dependencies must also be available.
Use the inventory command in `phase4/embeddings.py` after importing the actual files.
Run complete encoder replay before accepting a cache.

No compatible runtime was installed during this review.
No denied download route or remote computation was used.
Available disk space must cover original inputs, model files, derived caches, and acceptance replay outputs.
The metadata file records weight storage only.
It does not estimate the complete storage requirement or peak memory.
