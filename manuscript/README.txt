Current layout checkpoint,18October2026:
Native compilation and authorized alternate15-page visual review PASS. Native computer-use runtime remains incompatible with this host; no security controls changed.
Run: python3 -B manuscript/tmlr_layout_20261018/check_layout.py --root . --output /tmp/third-layout-check.json
The old TMLR checker remains bound to its historicalsource/archive and is replayed inside that archive by this new checker.
Current manuscript differs only in the revision-attribution paragraph formatting.

Third TMLR manuscript — current source and artifact guide

The existing third_working_draft.tex is now the TMLR-targeted source. It embeds
the unmodified official anonymous style and author-year bibliography, requiring
no extra project files in the native editor. The style's Apache license is at
tmlr_20261018/TMLR_STYLE_LICENSE.txt. The default template's review header does
not mean a submission has been made. Keep this same source/editor open.

This paper presents finite-horizon exact repair, finite-format private-state
bounds and two completed negative LLM calibrations. It does not claim natural
semantic utility, a runtime/storage win, or completion of the original empirical
program. The unexecuted experimental design is explicitly in an appendix.

Current verification (from repository root):
  python3 -B manuscript/tmlr_20261018/check_tmlr.py --root . --output /tmp/third-tmlr-check.json --replay-historical

The check verifies source/citation structure, exact official style bytes,
preservation of prior mathematical blocks, both negative results, calibration
table arithmetic,49 historical source bindings, the new delta-review binding,
and the anonymous supplement hash. Historical source checking runs only in a
clean extraction of the prior archive. The previous build_evidence.py,
package_public.py and evidence_20261016.json remain historical and must not be
rebound or run as if they review the TMLR revision. The new source check is an
author check, not a new proof, private-data or visual review.

The separate AI delta review found one abbreviation issue, now corrected; F1/F2
from the prior review remain corrected. Original reviews retain their old source
bindings. BUILD_STATUS.json tracks the successful native compilation and the
unavailable visual check. Do not manufacture a PDF export or visual approval.

Anonymous reviewer supplement:
  tmlr_20261018/tmlr_anonymous_supplement_v1.zip
It contains code and aggregate evidence only. Extract and run check_aggregates.py
with Python3.10+. All102 cut quotients/decisions and72 deidentified compute rows
are checked;12 v4 bin-boundary cuts are reconstructed. It is not private-data,
fresh-inference, full-experiment or complete theorem-suite replay.

Development backup archives contain historical reports and identifying links.
They are NOT anonymous submission supplements. The current dedicated supplement
omits that history; exact source/hash matching can still link public artifacts.

See tmlr_20261018/SUBMISSION_CHECKLIST.txt for the remaining PDF and author-owned
submission items. Both inaccessible full texts are acknowledged without claiming
novelty clearance. The original calibration gate continues to block effect jobs.
No human pair annotation or pair review is required or newly introduced.
