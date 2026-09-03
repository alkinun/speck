# 13 — Synthetic MQAR calibration and mixer comparison

## Question

Does KDA's channel-wise decay improve multi-query associative recall over scalar GDN when output
gating, head geometry, task data, optimization horizon, and validation cases are controlled?

This is the first direct capability comparison motivated by Kimi Linear. It is a synthetic
qualification gate, not evidence of natural-language long-context quality.

## Protocol reconstruction

The Kimi paper specifies two layers, two heads, head dimension 128, a maximum of 20,000 steps, and
the learning-rate grid `{5e-5, 1e-4, 5e-4, 1e-3}`. It does not disclose the number of associations,
batch size, optimizer, full model block, or generator.

MQAR data therefore follows the primary
[Zoology implementation](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/data/multiquery_ar.py)
at pinned revision `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`:

- vocabulary: 8,192, split into disjoint key and value halves;
- unique keys and values within each example;
- power-law query gaps with `power_a=0.01`;
- random non-query distractors;
- loss only on the value following each query.

Speck uses hidden size 256, two mixer-plus-SwiGLU blocks, intermediate size 768, two 128-dimensional
key/value heads, short-convolution kernel 4, AdamW with weight decay 0.1, gradient clipping at 1.0,
and cosine decay to zero. Training batches contain eight sequences. Every evaluation uses the same
256 held-out examples and seed `1,000,042`.

The full-vocabulary random accuracy is `1/8192 = 0.0122%`. After learning only that answers belong
to the value half, chance is `1/4096 = 0.0244%` and cross-entropy is `ln(4096) = 8.3178`.

## Implementation defects caught before comparison

The first KDA module inherited Speck GDN's zero decay bias. At length 1,024 this produced overly
short initial memory and a 20K-step run that learned only the value-token prior. The official FLA
KDA layer instead initializes per-channel time steps log-uniformly from 0.001 to 0.1 and decay rates
from 1 to 16.

Speck now implements that KDA initialization. A configurable `fla` initialization was added to GDN
while preserving `speck` as the default for every existing checkpoint. Synthetic GDN controls opt
into the FLA initialization, using rate range 0 to 16 and the same time-step range. The invalid KDA
run remains checked under `results/.../diagnostics`.

Corrected KDA still failed with 256 associations. The initializer defect was real but did not
explain the dense task's capacity failure.

## Query-density calibration

The sequence length, vocabulary, KDA architecture, LR `1e-3`, seed, and 20K horizon were fixed while
the number of associations changed:

| Associations | Best accuracy | Step reaching 99% | Final loss | Result |
| ---: | ---: | ---: | ---: | --- |
| 4 | 99.41% | 7,500 | 0.3711 | pass |
| 16 | 99.12% | 6,000 | 0.0646 | pass |
| 32 | 99.01% | 12,500 | 0.0814 | pass |
| 64 | 0.067% | none | 8.3102 | fail |
| 256 | 0.037% | none | 8.3179 | fail |

Sixteen associations converge faster than four because each batch contains four times as many
supervised targets. Query count mixes memory load with gradient density; it is not a monotonic
difficulty scalar. The sharp 32-to-64 collapse nevertheless identifies a capacity/optimization
boundary for this fixed model and horizon.

Thirty-two associations were frozen as the architecture-comparison point: difficult enough to
produce a long phase transition, but demonstrably solvable.

## Model sizes and execution

| Variant | Parameters | Analytic F/tok | Warm-cache training tok/s | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| GDN-SiLU | 3,941,896 | 25.018M | 45.4k preflight; ~0.95M long run | 788.7 MiB |
| GDN-sigmoid | 3,941,896 | 25.018M | 49.1k preflight; ~0.98M long run | 788.7 MiB |
| KDA-sigmoid | 4,072,452 | 24.930M | 28.3k preflight; ~0.83M long run | 790.7 MiB |

The ten-step preflight rates include residual setup overhead and are not comparable with long-run
steady state. KDA is about 13% slower than GDN in the full runs on this 3090 despite a slightly
lower analytic FLOPs estimate, showing that kernel realization matters. KDA has 3.31% more
parameters in the tiny model because of its channel-decay projection.

## Seed-42 learning-rate grid

Best validation accuracy; parenthesized values are the first 250-step evaluation at or above 99%:

| Variant | `5e-5` | `1e-4` | `5e-4` | `1e-3` |
| --- | ---: | ---: | ---: | ---: |
| KDA-sigmoid | 3.784% | **99.878% (4,750)** | 0.354% | 99.011% (12,500) |
| GDN-SiLU | 2.222% | **99.097% (9,750)** | 0.073% | 99.084% (12,250) |
| GDN-sigmoid | **97.131%** | 0.110% | 0.085% | 0.110% |

The LR response is strongly non-monotonic. In particular, KDA succeeds at `1e-4` and `1e-3` but
fails at the intervening `5e-4`. Reporting only one convenient rate would be misleading.

At the discovery seed, best-tuned KDA reaches the gate in 38,912,000 training tokens versus
79,872,000 for best-tuned GDN-SiLU, initially suggesting about a 2.05× sample/FLOPs advantage. That
one-seed magnitude did not survive replication.

## Three-seed confirmation

The best discovery rate was repeated with training seeds 43 and 44. Validation data remained fixed.
For GDN-sigmoid, which had no passing rate, the highest-accuracy `5e-5` point was repeated.

| Variant | LR | Seed 42 | Seed 43 | Seed 44 | Passes | Median step | Step range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KDA-sigmoid | `1e-4` | 4,750 | 4,500 | 6,000 | 3/3 | 4,750 | 1,500 |
| GDN-SiLU | `1e-4` | 9,750 | 4,750 | 5,000 | 3/3 | 5,000 | 5,000 |
| GDN-sigmoid | `5e-5` | fail, 97.13% | fail, 0.54% | fail, 97.00% | 0/3 | — | — |

KDA's mean gate step is 5,083 versus 6,500 for GDN-SiLU, a 1.28× mean token/FLOPs advantage. The
median ratio is only 1.05×. Per seed, KDA is faster twice and GDN-SiLU once. Mean wall time through
the gate is 60.9 seconds for KDA and 69.4 seconds for GDN-SiLU because KDA's kernel is slower.

The defensible result is therefore:

- KDA and GDN-SiLU both solve 32-query MQAR in 3/3 seeds.
- Their median convergence is nearly tied at the 250-step measurement resolution.
- KDA has a smaller observed tail/range and better mean because GDN-SiLU has one slow seed.
- KDA-sigmoid robustly beats the output-gate-matched scalar GDN-sigmoid, which passes 0/3.

## Output-gate interpretation

The weight- and initialization-matched GDN comparison favors SiLU on this task: SiLU passes 3/3,
while sigmoid passes 0/3. This does not refute Kimi's language-model ablation, which reports
validation perplexity rather than MQAR convergence. It does show that output-gate conclusions are
task- and optimization-dependent at Speck scale.

KDA makes sigmoid viable, indicating that channel-wise memory control interacts with the output
gate. A future KDA-SiLU cell would be required to measure that interaction fully; it is not needed
before the more important length-scaling curve.

## Decision

KDA passes the synthetic promotion gate. The effect is not “universally faster than every GDN”; it
is more precise:

1. Channel-wise decay rescues sigmoid-gated MQAR and reduces observed convergence-tail variance.
2. A well-tuned SiLU GDN remains highly competitive at 32 associations.
3. Both architectures have a sharp phase transition, making seed and LR grids mandatory.
4. The present KDA fails between 32 and 64 simultaneous associations at length 1,024, so finite
   recurrent capacity remains a real limit.

Next, measure two orthogonal length curves before language-model pretraining:

- fixed load: 32 associations at lengths 256, 512, 1,024, and 2,048, isolating distance;
- fixed density: 8, 16, 32, and 64 associations at those lengths, measuring the combined scaling
  problem.

Run KDA-sigmoid and GDN-SiLU at `1e-4` over the three confirmed seeds. Retain GDN-sigmoid as a
negative control only where budget permits. Palindrome and 64-stack tests remain required before
the 131M-token architecture staircase.

## Artifacts

- Frozen protocol:
  [experiments/SpeckLC-SyntheticMemory/protocol.json](../experiments/SpeckLC-SyntheticMemory/protocol.json)
- Consolidated result:
  [results/SpeckLC-SyntheticMemory/summary.json](../results/SpeckLC-SyntheticMemory/summary.json)
- Raw preflights, diagnostics, density calibration, LR grid, and seed confirmations:
  [results/SpeckLC-SyntheticMemory](../results/SpeckLC-SyntheticMemory)
