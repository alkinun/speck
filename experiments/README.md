# Experiments

[qualification](qualification/README.md) is the first hardware check: the existing 1.2B hybrid at 4K,
synthetic inputs, finite steps, and fresh-process checkpoint/RNG replay.

`make smoke` creates a tiny offline end-to-end experiment in a temporary directory. Use its
`--output-dir` option to inspect the generated configs, shards, checkpoints, and report.

The real-data pilot is not materialized yet. Add one directory when its data, optimization settings,
token endpoint, evaluation, and cost are fixed under [PLAN.md](../PLAN.md).
Earlier 140M releases and research matrices remain in [history](../archive/README.md).
