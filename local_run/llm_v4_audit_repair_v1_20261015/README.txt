V4 exact-byte audit repair, 15 October 2026.
Qwen 13663 completed 396 calls. CPU audit 14187 failed because text-mode universal
newline conversion changed CR/CRLF sequences before hashing six stored responses.
All original bytes, hashes, token counts, model calls, labels and thresholds remain
unchanged. This package performs complete fresh CPU audits in distinct roots.
It does not infer audit success from the byte diagnostic or rerun model judgments.

After a corrected audit succeeds, publish_bridge.py creates a previously absent
run/result.json exact copy for frozen downstream readers, plus a sealed
ACCEPTED_AUDIT_REPAIR.json pointing to the corrected audit's actual job and source.
Original failed launch/lifecycle/log files remain unchanged; job 14187 stays FAILED.
Only continue_launch.py may admit corrected audit predecessors for Mistral/statistics.
It retains the original execution manifest, activation and unoccupied output slots.
No original model, audit-source, request, seed, sampling or statistics kernel is edited.
Validation uses the same repair prospectively, preserving its fresh calls and rules.
Use explicit --scope and --repair-package-sha256 on every command. No automatic retry.
