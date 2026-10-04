# Isolated natural-text development matrix

All 216 jobs passed. The 864 released heads agree with the independent scalar-graph/dual-ridge oracle within 9.44e-16. These are engineering measurements on the reused Civil100 lexical preview, not primary semantic, source-withdrawal or task-utility results.

| Dimension | Method | Jobs | Mean initial state bytes | Mean final state bytes | Maximum child RSS bytes | Mean lifecycle seconds |
|---:|---|---:|---:|---:|---:|---:|
| 64 | B-A | 20/20 | 82296 | 82297 | 122634240 | 0.4831 |
| 64 | B-E | 20/20 | 74151 | 67529 | 122634240 | 0.4755 |
| 64 | B-E-compact | 20/20 | 49260 | 42876 | 122634240 | 0.4763 |
| 64 | O-G | 20/20 | 139322 | 137756 | 122634240 | 1.1610 |
| 64 | O-T | 20/20 | 91586 | 90020 | 122634240 | 0.4742 |
| 64 | P-I | 20/20 | 1707470 | 39432 | 122634240 | 0.4867 |
| 64 | P-I-jointspan | 20/20 | 161632 | 24907 | 122634240 | 0.5237 |
| 64 | P-R | 20/20 | 1471486 | 39616 | 122634240 | 0.4939 |
| 64 | P-S | 20/20 | 1707470 | 39432 | 122634240 | 0.4893 |
| 768 | B-A | 4/4 | 5055357 | 5055358 | 123879424 | 0.6216 |
| 768 | B-E | 4/4 | 5002156 | 4944142 | 123879424 | 0.6439 |
| 768 | B-E-compact | 4/4 | 285808 | 228032 | 123879424 | 0.5623 |
| 768 | O-G | 4/4 | 420922 | 419356 | 123879424 | 1.2760 |
| 768 | O-T | 4/4 | 373186 | 371620 | 123879424 | 0.6298 |
| 768 | P-I | 4/4 | 232140762 | 2390804 | 764555264 | 2.0081 |
| 768 | P-I-jointspan | 4/4 | 1144264 | 477369 | 123879424 | 0.6597 |
| 768 | P-R | 4/4 | 198985738 | 2390988 | 714579968 | 3.2096 |
| 768 | P-S | 4/4 | 232140762 | 2390804 | 740941824 | 2.3086 |

The compact eligible-payload comparator remains substantially smaller than the joint-span summary on this fixture. Incidence-rank compression reduces coefficient storage relative to dense P-I but incurs additional construction workspace. Neither observation establishes a general memory optimum.

All methods used the same prospective 1 GiB address-space limit, 60-second CPU/wall limits and one BLAS thread. Each construction and repair ran in a fresh process. Repair loaded its own accounted snapshot before seccomp TSYNC denied all new file opens. The kernel boundary is enforced for trusted project code; this is not a malicious-native-code or physical-erasure proof.

Five d64 repetitions and one d768 feasibility repetition reuse the same four paths (two R, one U, one A). Shared-host concurrent development checks affect latency, so this table supports no paper speedup claim. Serialized state excludes head files; process RSS includes runtime, loaded state, working arrays and serialization buffers. Shared input preparation and independent audit costs are recorded separately.

The canonical journal was reconstructed from all 216 immutable per-job audits after three append entries were missing. `journal_reconciliation.json` records exact original/canonical hashes and the recovered IDs; `job_ledger_original.jsonl` is preserved. No method was rerun to replace an outcome.
