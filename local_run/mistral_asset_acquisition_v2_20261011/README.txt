Mistral asset acquisition v2: fresh login-node quota grants

V1 job 13361 stopped before downloads because compute-node /nfs_home is NFS
(csis.mn1:/nfs_home), while login-node /nfs_home is /dev/sdb1 ext4. Compute
quota RPC returned Connection refused. V1 and its failed receipt are preserved.

V2 downloads the same 13 immutable public model files, 24,510,683,806 bytes,
once into local_workspace/models/mistral-nemo-hub-v2. No model is loaded.
It requires the exact observed NFS mapping on the worker and ext4 mapping on
the one-shot login controller. The controller uses quota -v -w in KiB;
filesystem free space is not substituted for user quota.

The worker publishes 14 requests: one for the entire acquisition plus 8 GiB,
then one for each file plus 8 GiB. Each request includes job ID, nonce, exact
file index/name/bytes, cache path, package hash and expectation hash.
The controller may grant only the pending request of the specified job.
It publishes raw quota evidence and its grant atomically without overwrites.
Worker acceptance binds the exact request bytes, timestamp, quota arithmetic
and raw evidence. Grants older than 120 seconds are rejected. The worker waits
at most 300 seconds per request, then fails without starting that download.
Clock differences up to 5 seconds are explicitly tolerated; clocks must agree.

Controller invocation on the login node, using the existing authenticated
connection, is one-shot and contains no SSH or persistent loop:

  python grant_quota.py --out RUN/acquisition \
    --cache WORKSPACE/models/mistral-nemo-hub-v2 \
    --expectations REPO/local_run/model_annotation_runtime_mistral_v1_20261011/asset_expectations.json \
    --package-sha256 SEALED_PACKAGE_SHA --job-id SLURM_JOB_ID

Invoke again for each newly pending request. An idle response grants nothing.
The worker never requests a password or opens a connection to the login node.
If authentication fails while servicing, stop servicing; the worker times out.
Controller errors do not grant permission. Partial downloads and failure
receipts remain preserved; there is no retry or cleanup of prior caches.

Fresh quota observations are not reservations against other concurrent writers.
The 8 GiB reserve and 120-second age cap limit this risk but cannot remove it.
The final receipt reports every grant; it does not claim a final quota query.
Original model identity is checked via published LFS SHA256 for weights and
Git blob SHA1 for ordinary files, then a final local SHA256 recheck of all 13.
The alternate consolidated weight copy is excluded; Xet chunk caching and
implicit HF token use are disabled. No scientific or human gate is advanced.

Run local synthetic checks with python -B -m unittest discover -s . -p test_quota.py.
Cluster/software integration remains unexecuted until its own receipt exists.
