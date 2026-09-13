# Current research

Speck's first flagship studies how a fixed compute and memory budget should be divided across
high-information training data, recurrent processing, and periodic exact attention.

**Target:** a 1.2B dense-width KDA/GQA model, 400B training tokens with a 320B throughput fallback,
followed by context extension and an Instruct release. The budget is 5,000 GH200 GPU-hours,
approximately 52.08 four-GPU node-days, within a three-month allocation.

## Working entry points

| Need | Source |
| --- | --- |
| Model and scope | [Flagship charter](flagship/README.md) |
| Current state and next actions | [Status](status.json), or `python -m scripts.research_catalog --status` |
| Selected contracts | [Catalog](catalog.json) |
| Experiment order and budget | [Execution](flagship/EXECUTION.md) |
| Paper question and evidence requirements | [Paper contract](flagship/PAPER.md) |
| Supported prior conclusions | [Findings](findings/README.md) |
| Literature and source surveys | [Library](literature/README.md) |
| Manuscript claims | [Claim registry](../paper/claims.json) |
| Completed work and earlier plans | [Archive](../archive/README.md) |

The catalog explicitly selects contract versions. Their original JSON bytes and referenced historical
inputs are preserved in the archive. These planning contracts are not runnable launch manifests.
The reorganized runtime needs a new execution qualification before it produces scientific outputs.

```bash
python -m scripts.research_catalog
python -m scripts.research_catalog --status
python -m scripts.archive check
```

## Record ownership

`research/status.json` is the maintained status source. Protocols define methods and decisions;
`results/` stores current outcomes; findings interpret evidence; `paper/claims.json` defines manuscript
claim status. Linear and W&B remain operational mirrors.

Use the short [workflow](WORKFLOW.md) and [artifact policy](DATA_MANAGEMENT.md). The
[lab direction](DIRECTION.md) records the longer-term research aim.
