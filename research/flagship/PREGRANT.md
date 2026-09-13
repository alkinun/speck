# Pre-grant readiness

Current state and next actions are maintained in [research/status.json](../status.json):

```bash
python -m scripts.research_catalog --status
```

The [first-wave proposal](FIRST_WAVE.md) now provides the reviewable E1/E3 recipes, logical run list,
and per-source preparation targets. Resolve its review items before turning it into launch manifests.

The remaining work covers tokenizer selection, final corpus preparation, per-arm materializers and
analysis, long-document yield, GH200/DDP/Slurm qualification, comparators, and recurrent export.
The reorganized implementation also requires execution qualification before new scientific runs.

For each launch, record the implementation revision, qualified hardware/software, exact model/data/
tokenizer identities, analysis and stopping rules, checkpoint/resume behavior, and compute charge.
Site-specific allocation, scheduler, storage, and accounting values belong in the launch environment.

The [execution plan](EXECUTION.md) defines dependencies and the protected reserve. Dated pre-grant
snapshots remain in the [archive](../../archive/pregrant-history/research/flagship/PREGRANT.md).
