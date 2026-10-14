V4 tokenizer request preparation. No model weights or GPU context are loaded.

The new wrapper accepts only the sealed v4 sampler, including its full census
replay, first512 independent parity and pairwise score checks. It independently
reconstructs every selected identity, score, inverse weight and swap choice.
It verifies all calibration text mappings against the complete prepared source,
then uses the unchanged, hash-pinned v3 prompt/tokenizer primitives and pinned
model assets. All original texts remain complete; overflow stays unavailable
in the planned denominator. V3 selection outputs cannot enter this adapter.

New sample and prompt files remain private. Stage budget: CPU Slurm csis.cn1,
2 CPUs, 8 GiB RAM, 20 minute limit, 128 MiB payload cap, 256 MiB stage allowance
plus 8 GiB quota reserve. Submission is exclusive and never retried automatically.
This package creates no inference activation or scientific acceptance.
