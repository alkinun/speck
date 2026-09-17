# Experiments

[qualification](qualification/README.md) is the first hardware check: the existing 1.2B hybrid at 4K,
synthetic inputs, finite steps, and fresh-process checkpoint/RNG replay.

`make smoke` creates a tiny offline end-to-end experiment in a temporary directory. Use its
`--output-dir` option to inspect the generated configs, shards, checkpoints, and report.

[pilot](pilot/README.md) fixes a 105M-token real-data engineering recipe, pinned evaluation inputs,
and a bounded retained-data preparation path. Its README and PLAN.md distinguish preparation
progress from hardware qualification and GPU execution.
Earlier 140M releases and research matrices remain in [history](../archive/README.md).
