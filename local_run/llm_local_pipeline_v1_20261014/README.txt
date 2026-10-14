Local controller for the existing fully LLM v3 selection, 14 October 2026.

This is one bounded execution of the already activated 198-request workflow.
It is not a recurring automation. It consumes existing Qwen job13574, preserves
and verifies all terminal artifacts, launches the existing sealed CPU audit,
then submits Mistral once through the unchanged serial controller. It audits
Mistral and runs the existing frozen statistics consumer. A separate local
review recomputes every one of the51 exact weighted cut decisions and diagnostics.

No activation, seeds, prompts, models, thresholds or denominator rules change.
No human annotations, model retries, validation activation or Git pushes occur.
A failed selection criterion is a completed negative result, not a trigger for
more model calls. Qwen ordinary runtime failure is audited and then holds the
Mistral submission. Any SSH refusal, timeout, integrity/audit/source mismatch or
uncertain submission stops the controller; existing jobs are not cancelled.

The active copies and private inputs are in the original task workspace:
/Users/priyankjairaj/Documents/Codex/2026-10-11/see-i-am-working-on-a/work/resume-20261014/sampling_v1
These archived sources are exact copies. Do not run a second controller. Its
exclusive state folder is remaining_pipeline_a01; intent.json records PID,
source/input bindings, existing job13574, a10-hour maximum coordinator lifetime
and120-second metadata polling. The four-hour per-model and20-minute per-audit
Slurm limits remain unchanged. Receipt files and stage outputs are exclusive.
The existing authenticated SSH session is reused with no authentication fallback.

Read remaining_pipeline_a01/events.jsonl, completion.json or failure.json and
existing Slurm receipts before resuming. Never restart after an uncertain stage.
The parent process is local and does not consume an extra GPU or run weights on
the login node. Each sealed remote launcher still checks its queue/quota/source
prerequisites. Output bodies and private token/source inputs remain ignored.

Ten local software-fixture tests passed before starting. They cover ordering,
Qwen failure audit/hold, audit failure, changed-source refusal, uncertain receipt
no-retry, negative/positive/uncertain panels and corrupt threshold/denominator
rejection. These fixtures create no model or natural-data observations.
Local terminal checks do not independently re-execute model weights or logits;
the CPU auditor separately replays actual prompt tokens, output tokens/grammar,
response identities and complete denominators. Labels remain model-defined.
