# 14 — MQAR distance and load scaling

## Question

Does KDA's advantage emerge from handling longer distances, more simultaneous associations, or
both? The initial length-1,024 experiment conflated these axes.

## Controlled design

Two curves use KDA-sigmoid and GDN-SiLU, their selected LR `1e-4`, the same two-layer
head-dimension-128 architecture, 8,192-token vocabulary, batch size 8, fixed validation seed, and
20K-step/99%-accuracy rules from [13](13_synthetic_mqar.md).

1. **Fixed load:** 32 associations at lengths 256, 512, 1,024, and 2,048. This holds target count
   and association load fixed while increasing distance and distractors.
2. **Fixed density:** 8, 16, 32, and 64 associations at those lengths—one pair per 32 tokens. This
   increases distance, memory load, and supervised targets together.

The length-1,024/32 three-seed result is reused by both curves. Seed 42 was run across every new
point. The two length-2,048 endpoints were then repeated at seeds 43 and 44 because they determine
the long-distance conclusion.

## Seed-42 fixed-load curve

First evaluation step reaching 99%:

| Length | Associations | KDA-sigmoid | GDN-SiLU |
| ---: | ---: | ---: | ---: |
| 256 | 32 | 5,750 | 5,250 |
| 512 | 32 | 4,250 | 7,250 |
| 1,024 | 32 | 4,750 | 9,750 |
| 2,048 | 32 | 4,750 | fail; best 0.134% |

KDA passes every distance without a monotonic step increase. GDN is slightly faster at 256, then
slows with length and fails at 2,048. Because every sequence retains 32 supervised answers, this
curve is not explained by changing gradient count.

The KDA result shows no distance ceiling through 2,048 at fixed load. The scalar-GDN seed-42 result
suggests one, but replication is required because phase-transition timing was already seed-sensitive
at length 1,024.

## Seed-42 fixed-density curve

| Length | Associations | KDA-sigmoid | GDN-SiLU |
| ---: | ---: | ---: | ---: |
| 256 | 8 | 4,000 | 5,250 |
| 512 | 16 | 4,250 | 5,250 |
| 1,024 | 32 | 4,750 | 9,750 |
| 2,048 | 64 | 8,000 | 8,500 |

Both mixers pass all four seed-42 density points. KDA reaches the gate first everywhere, although
the 2,048 gap is only one 500-step interval. The higher number of supervised targets at 2,048/64
also helps GDN relative to fixed-load 2,048/32, so this curve cannot be interpreted as memory load
alone.

## Cross-control correcting the density calibration

The initial calibration had found KDA failure at length 1,024/64 using LR `1e-3`, while the
2,048/64 density point passed using the selected LR `1e-4`. To separate LR from spacing, both models
were run at 1,024/64 and `1e-4`:

| Variant | Step reaching 99% |
| --- | ---: |
| KDA-sigmoid | 5,000 |
| GDN-SiLU | 6,000 |

Both pass. The former “capacity boundary between 32 and 64” was an LR-dependent optimization
boundary. Longer spacing was not required to store 64 associations. This correction is reflected
in the consolidated summary and amended [13](13_synthetic_mqar.md).

## Replicated 2,048 endpoints

### Fixed load: 32 associations

| Variant | Seed 42 | Seed 43 | Seed 44 | Passes |
| --- | ---: | ---: | ---: | ---: |
| KDA-sigmoid | 4,750 | 7,500 | 7,500 | 3/3 |
| GDN-SiLU | fail, 0.134% | fail, 0.061% | 6,000 | 1/3 |

### Fixed density: 64 associations

| Variant | Seed 42 | Seed 43 | Seed 44 | Passes |
| --- | ---: | ---: | ---: | ---: |
| KDA-sigmoid | 8,000 | 4,750 | 6,250 | 3/3 |
| GDN-SiLU | 8,500 | fail, 0.043% | fail, 1.038% | 1/3 |

Scalar GDN can solve either endpoint, so this is not an absolute expressivity impossibility. It is
a reliability difference under fixed optimization and compute ceilings. KDA passes all six
replicated 2,048 runs; GDN passes two of six, one on each curve.

The endpoint also illustrates why final sub-threshold accuracy matters. Failed GDN runs range from
near value-prior chance to 1.04%, but none enters the high-accuracy phase. Classifying all as “fail”
is appropriate for the predeclared gate, while preserving raw curves avoids losing graded signal.

## Interpretation

At length 1,024, KDA and SiLU GDN both pass 3/3 and have nearly tied median convergence. At length
2,048, their reliability separates sharply. Channel-wise decay does not merely shift median steps;
it makes the learned retrieval phase substantially more likely to appear as distance/load stress
increases.

The most defensible architectural claim from the synthetic work is now:

> Under a pinned MQAR generator, LR grid, and 20K-step ceiling, KDA preserves 3/3 learnability at
> 2,048 tokens for both 32 and 64 associations, while scalar SiLU GDN preserves only 1/3.

This remains a tiny-model synthetic result. It does not establish 32K/128K language capability,
and phase-transition probabilities need more than three seeds for a precise estimate. It is strong
enough to promote KDA to Palindrome/Stack qualification and then to the controlled 131M-token
language-model staircase.

## Artifacts

- Fixed-load curves:
  [results/SpeckLC-SyntheticMemory/mqar-fixed-load](../results/SpeckLC-SyntheticMemory/mqar-fixed-load)
- Fixed-density curves:
  [results/SpeckLC-SyntheticMemory/mqar-fixed-density](../results/SpeckLC-SyntheticMemory/mqar-fixed-density)
- Matched-LR cross-control:
  [results/SpeckLC-SyntheticMemory/mqar-cross-controls](../results/SpeckLC-SyntheticMemory/mqar-cross-controls)
- Consolidated summary:
  [results/SpeckLC-SyntheticMemory/summary.json](../results/SpeckLC-SyntheticMemory/summary.json)
