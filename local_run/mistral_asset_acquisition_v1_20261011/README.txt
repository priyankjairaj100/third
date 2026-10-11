Mistral Nemo immutable asset acquisition, 11 October 2026

Acquire only the 13 exact files in the pinned asset expectation receipt for
mistralai/Mistral-Nemo-Instruct-2407 revision
04d8a90549d23fc6bd7f642064003592df51e9b3. Total: 24,510,683,806 bytes.
The alternative consolidated weight file is deliberately excluded to avoid a
second 24.5 GB copy. One isolated HF cache holds blobs and snapshot symlinks.

The installed huggingface_hub 0.36.2 exposes the hf entry point but its environment
has no hf binary wrapper. Invoke that same installed console entry point through
Python; no environment is modified. This is the current hf CLI, not the deprecated
huggingface-cli command. The exact CLI options were checked on the cluster.

Run acquire.sbatch with repo, workspace, new private run root, acquire.py SHA256,
and asset_expectations.json SHA256. It uses the existing judge runtime on a CPU
compute node. Before any download, require the full expected total plus an 8 GiB
reserve below the quota soft limit on the observed /dev/sdb1 filesystem. Before
each file, recheck its size plus the reserve. Downloads are one file at a time;
Xet and its chunk cache are disabled. Existing cache/output paths are refused.
No authentication change, model loading, inference, external publication, or
corpus text transfer is performed. Incomplete attempts remain private and do not
produce a passing receipt; there is no automatic resubmission or cleanup.

Check large files against publisher LFS SHA256 and size, ordinary files against
publisher Git blob SHA1 and size. Record a local SHA256 for every file and rehash
all at completion. The SHA1 identity is the Git blob digest including its header,
not a claim that the publisher supplied SHA256 for ordinary Git files.

All acquisition logs/results remain under ignored local_workspace. They establish
asset identity only; second-judge technical and semantic qualification are later
separate gates. The user-authorized 50–100 human reviews remain unused.
