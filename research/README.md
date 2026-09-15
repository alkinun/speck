# Current research

Speck's first serious allocation targets **efficient long-context intelligence**: a general-purpose
1.2B hybrid with strong document/history understanding and retained broad capability.

The central experiment crosses dense versus KDA/global memory with standard versus dependency-requiring
supervision. The flagship uses a 320B-token base target (400B conditional on measured fit), followed by
substantial context and instruction development. Useful 32K capability is the first milestone; 128K is
the target, with 64K evaluated in between. The allocation is 5,000 GPU-hours on four GH200s over about
90 calendar days. All new claims are pre-results.

| Need | Source |
| --- | --- |
| Direction and model | [Lab direction](DIRECTION.md), [charter](flagship/README.md) |
| Scope change and retired work | [Pivot decision](flagship/PIVOT.md) |
| Budget, dependencies and fallbacks | [Execution](flagship/EXECUTION.md) |
| First allocation work | [First wave](flagship/FIRST_WAVE.md) |
| Controlled study | [Study](flagship/STUDY.md) |
| Data and coherent-context preparation | [Data](flagship/LONG_CONTEXT_DATA.md) |
| Capability development | [Context](flagship/CONTEXT_EXTENSION.md), [post-training](flagship/POST_TRAINING.md) |
| Quality, public models and retrieval baseline | [Evaluation](flagship/EVALUATION.md) |
| Paper and planned claims | [Paper](flagship/PAPER.md), [registry](../paper/claims.json) |
| Readiness and selected versions | [Status](status.json), [catalog](catalog.json) |
| Previous scope and completed evidence | [Transition snapshot](history/README.md), [archive](../archive/README.md) |

The catalog selects versioned designs, not launch manifests. Exact costs, datasets, thresholds and
hardware are qualification work; missing values are explicit launch blockers. Checked historical
source stock remains useful but old E1/E3 quotas are not new preparation requirements.

```bash
python -m scripts.research_catalog
python -m scripts.research_catalog --status
python -m scripts.archive check
```

Status owns current work; contracts own decisions; results/findings own evidence. Linear and W&B are
mirrors. Use the [workflow](WORKFLOW.md) and [artifact policy](DATA_MANAGEMENT.md).
