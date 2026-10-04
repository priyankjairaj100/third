# Registry exports

`study_registry_release/` is the authoritative prospective registry export:
43 configuration groups, 16,933 planned jobs, 158 artifact keys, 41 passing
software checks. No primary assets are verified by its empty inventory and
no scientific execution is authorized.

`study_registry/` preserves the first reviewed draft. Independent review found
two mismatches with the narrative protocol: WCEP must inherit the frozen News
threshold and may remain record-only without URLs; threshold alternatives
are explicitly clamped at zero/one. That draft is superseded.

`study_registry_final/` records the correction of those two issues. The release
adds explicit sensitivity-specific threshold quality audits and MPNet-specific
learner lambda/test caches. These exports are evolving plans, not experimental
runs; do not aggregate their job counts.

The release registry code SHA256 is
`3d4419bde7efeac3654ec4aba091eab877e547d0384a147fddbcfc6ed13c72ef`.
Check-script SHA256 is
`b04d58ec8e7132b23cad74efe129defa65a850af4d69eb377668d4b0cc9f39c5`.
