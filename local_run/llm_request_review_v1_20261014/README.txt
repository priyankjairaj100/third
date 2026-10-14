Independent request audit, 14 October 2026.

Replays all prepared full-text prompts and input token IDs using the existing
qualified offline tokenizer assets, without torch/model weights or inference.
Checks exact sample identities, weights, swaps, request and terminal receipts.
The frozen full source/frame and 220M-pair census remain execution-attested;
this audit does not independently replay them or establish semantic validity.

The exact actual audit is preserved separately in the compact public report.
Local tokenizer-only assets and private terminal exports remain ignored.
Eight mutation/coverage tests passed independently. A missing local Jinja2
dependency was resolved using already-installed packages without downloads.
No production random draws, activation or human annotation occurs here.
