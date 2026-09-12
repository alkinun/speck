# Capacity-adjusted 500K starter dataset

Completed [pilot](../../research/notebook/2026-09-12-instruct-data-pilot.md) recipe: 500,000 training
and 10,000 validation conversations. It retains the original 17 sources, revisions, filters,
category caps, seed, and reference tokenizer. The owner approved rebalancing after smaller sources
could not supply their scaled quotas.

| Training skill | Original | 500K |
| --- | ---: | ---: |
| General | 40% | 48.5% |
| Precise instruction following | 10% | 5% |
| Writing | 20% | 20% |
| Math | 10% | 10% |
| Code | 10% | 10% |
| Grounded | 8% | 5.7% |
| Clarification | 2% | 0.8% |

Human-edit shortfalls go to rewriting/summarization; remaining shortages go to general sources.
Independent validation weights accommodate the small sources after prior validation is excluded.
Exact source quotas and selection rules are in `mixture.json`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.instruct_prepare \
  --recipe experiments/Speck-Instruct-Starter500K/mixture.json \
  --exclude /mnt/speck-data/speck/evaluations/Speck2-Instruct-Starter500K/exclusions.jsonl \
  --output-dir /mnt/speck-data/speck/data/instruct-starter-500k-v2
```

The exclusion file contains the fixed diagnostic, qualitative, BananaMind, Open SLM, and prior
validation prompts/families. See the [compiler guide](../Speck-Instruct-Starter/README.md) for
setup, exclusions, and resume. Native [Speck2 preparation](../Speck2-140M-Instruct-Starter500K/README.md)
retains 496,537 training rows after its 4K check. This is a rebalanced recipe, not a size-only ablation.
