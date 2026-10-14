Prospective operational launcher correction before any Mistral submission.

The original sealed control's Mistral preflight reused an assembly helper that
requires every successful predecessor on CPU worker csis.cn1. Qwen13574 correctly
ran on GPU worker csis.cn3. The v1 preflight rejected it before creating any
Mistral intent or job. That failed local receipt and controller are preserved.

This explicit adapter delegates all ordinary helpers unchanged, except the
terminal-parent check for exact completed Qwen job13574: every accounting row
must be COMPLETED0:0 on csis.cn3, with unique parent and recorded2634-second
parent runtime. CPU request/qualification/audit checks still use the original
strict CPU-worker helper. The exact independently replayed Qwen audit13623 and
its pinned reports are additionally required. Seven tests exercise the repair
and reject wrong nodes, failed/active jobs, duplicate/unrelated/missing rows,
wrong identity/runtime and relaxed success flags.

Original source, package hashes, resource plan, admission, activation, models,
requests, inference child command, environment, seeds and statistics are unchanged.
The GPU worker executes original execute.sbatch and original control/runner code.
Every source/queue/quota guard still runs twice where originally required. The
adapter identity and explicit exception are recorded in the launch intent; this
is an operational code amendment, not a claim that unmodified v1 launched Mistral.
Only the existing absent Mistral slot is accepted. No overwrite/retry is allowed.
This package does not lower the8GiB reserve or change any scientific threshold.
