# Full-refit implementation acceptance history

The authoritative scoped acceptance run is `refit_engineering_final/`.
It has 94 passing software checks, 24 completed natural Civil100 lexical
checkpoints, zero checkpoint failures, and 72 independently reconstructed
saved heads. Five fits repeat the same declared seed and three use distinct
alternative seeds. Original fitted selection contains 75 of the 100 preview
records. No genuine source trajectory or official Faiss/Torch backend ran.

`refit_engineering/` preserves the initial attempted acceptance run. The natural
24 checkpoints completed; a subsequent algebraic source-fixture assertion had
an incorrect expected total row count (12). The correct count is 10: four R
all-step rows, four S all-step rows, and two matched-volume R checkpoint rows.
This was a check expectation error, not a repair error. That initial check
did not finish and has no passing final check report.

`refit_engineering_release/` is an intermediate passing 92-check run. The final
release adds explicit initial-graph/selection consistency and rejection of
changed frozen curator feature bytes, and records the final source hash.
Earlier output locks bind earlier module revisions and are preserved as
implementation history; the current module is validated only against the
final release. Do not pool these repeated preview runs or treat them as
independent evidence.

The final run's module hash is
`af949b35a3bfa5ed044f71eb4f4f0b05d42b8a3f2f12a6b15e6b1f825dd2e37d`;
the acceptance script hash is
`1615221221591838c2b7051816906cb975a0433a4b86e4a5bcb93c76353fe712`.

Official dependencies remain unavailable (`torch`, `faiss`, `tqdm`).
The reference backend is intentionally not presented as Faiss-equivalent.
The full semantic, source-complete 5k-panel boundary study remains unexecuted.
