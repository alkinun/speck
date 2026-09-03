# 01 — 131M-token mixer screen

## Question

The initial screen asked whether Gated DeltaNet, gated causal convolution, sliding attention, or
global attention could be ranked by short-context validation loss under a small matched budget.

## Fixed recipe

- Family: `SpeckLC-150M-MixerScreen-131M`
- Six 20-layer models
- Hidden size: 768
- Parameters: 151–157M
- Training tokens: exactly 131,072,000 per variant
- Sequence length: 4,096
- Batch tokens: 65,536
- Optimizer: Muon
- Schedule: cosine
- Training seed: 42
- Final validation: approximately 20M tokens
- Hardware: one RTX 3090
- Attention placement in hybrids: every fourth layer, `GGGA` repeated

## Variants

| Variant | Layer layout |
| --- | --- |
| `gdn-local` | 15 Gated DeltaNet + 5 sliding GQA, window 2,048 |
| `gdn-global` | 15 Gated DeltaNet + 5 global GQA |
| `full-local` | 20 sliding GQA, window 2,048 |
| `full-global` | 20 global GQA |
| `conv-global` | 15 gated causal convolution + 5 global GQA |
| `pure-gdn` | 20 Gated DeltaNet |

## Original results

| Variant | Validation loss | PPL | tok/s | GPU-h | Peak allocated | W&B run |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gdn-local` | 2.810548 | 16.6190 | 31,945.7 | 1.2260 | 13,881 MiB | `zgwquxt5` |
| `gdn-global` | 2.819378 | 16.7664 | 47,724.3 | 0.8178 | 13,246 MiB | `v3onm3is` |
| `full-local` | 2.822668 | 16.8217 | 15,731.0 | 2.5037 | 17,028 MiB | `8fl41l42` |
| `full-global` | 2.826207 | 16.8813 | 40,596.0 | 0.9543 | 11,913 MiB | `bi3lho9g` |
| `conv-global` | 2.869402 | 17.6265 | 55,552.6 | 0.7046 | 20,571 MiB | `oree92ca` |
| `pure-gdn` | 2.965539 | 19.4052 | 51,130.4 | 0.7597 | 13,679 MiB | `bmv41bf6` |

All final checkpoints are complete at
`~/.cache/speck/checkpoints/SpeckLC-150M-MixerScreen-131M-*/model_002000.pt`.

## What survived later scrutiny

Two conclusions are much larger than the measured seed range and remain credible:

- Gated DeltaNet beats the matched convolution hybrid by `0.05002` nats.
- Removing attention entirely costs `0.15499` nats relative to `gdn-local`, far outside the
  observed seed noise.

One conclusion did not survive as a ranking:

- The four attention-bearing non-convolution models span only `0.01566` nats.
- The apparent `gdn-local` advantage over `gdn-global` is `0.00888` nats.
- The later three-seed range for one fixed architecture is `0.00965` nats.
- Therefore the screen cannot rank the top four from one seed.

## The unmeasured axis

Analytic bfloat16 resident state and training FLOPs at 128K were already radically different:

| Variant | State @4K | State @128K | GFLOP/token @128K |
| --- | ---: | ---: | ---: |
| `pure-gdn` | 2 MiB | 2 MiB | 0.92 |
| `gdn-local` | 9 MiB | 9 MiB | 1.02 |
| `full-local` | 30 MiB | 30 MiB | 1.32 |
| `conv-global` | 15 MiB | 480 MiB | 6.96 |
| `gdn-global` | 17 MiB | 482 MiB | 6.96 |
| `full-global` | 60 MiB | 1,920 MiB | 25.10 |

The screen ranked models on the axis where they were nearly tied and did not measure the axis where
they differed by up to roughly 200×. That observation redirected the project away from the prepared
500M-token 4K queue.

## Artifacts

- Experiment: [experiments/SpeckLC-150M-MixerScreen-131M](../experiments/SpeckLC-150M-MixerScreen-131M)
- Corrected sweep ledger:
  [sweep.json](../experiments/SpeckLC-150M-MixerScreen-131M/sweep.json)
- The prepared but intentionally stopped 500M family:
  [experiments/SpeckLC-150M-Rank-500M](../experiments/SpeckLC-150M-Rank-500M)
