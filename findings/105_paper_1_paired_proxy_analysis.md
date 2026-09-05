# 105 — Paper 1 three-pair proxy quality screen

## Completed fixed sample

Candidate pair 2 completes at 2.795503 nats versus its paired dense control at 2.832686, a descriptive
fixed-token difference of -0.037184. All six frozen cells are complete and qualified, every run has zero
non-finite steps, and the automatic transition produced the final analysis without quality-dependent
branching.

Across the three pairs, candidate-minus-control fixed-token loss is
`[-0.039170, -0.042662, -0.037184]`. The mean is -0.039672 nats and the upper one-sided 95% Student-t
bound is -0.034996, safely below the frozen +0.01 non-inferiority margin. Every one of the eleven source
guardrails passes; even the least favorable upper bound is negative (-0.015018 for textbook exercises),
well below the +0.02 source margin. The whole-architecture proxy quality screen therefore passes.

## Frozen secondary views

All three candidate arms reach the control-only 2.839127 target. Their paired time-to-quality
improvements are 18.81%, 21.28%, and 18.36%; mean improvement is 19.48% and the lower one-sided 95%
bound is 16.83%. No pair is right-censored.

At matched analytic training FLOPs, interpolated candidate-minus-control loss averages -0.120095 nats
(upper bound -0.115455). At each pair's common steady-time budget, it averages -0.072674 nats (upper
bound -0.068398). These are loss differences at matched budgets, not percent speedups.

At the fixed 131.072M-token endpoint, the candidate uses 21.49% fewer analytic FLOPs/token. Across
pairs, steady training time is 10.01% shorter, active time 10.42% shorter, and steady throughput 11.13%
higher on average. Peak allocated memory is 20.77% higher. Candidate pair 2 individually records
2,935.98 steady seconds, 3,277.70 active seconds, 44,643 tok/s, and 13.88 GiB peak allocation.

## Disposition

The frozen `quality_screen_passed` branch authorizes materializing and separately qualifying exactly the
six-pair, twelve-run proxy-finalist design at 1,539,833,856 tokens per run. It does not authorize that
training yet: output absence, storage, analysis/stopping rules, and the exact hardware/runtime path must
first be requalified.

This result belongs to the complete five-cache KDA/GQA whole architecture. It does not attribute benefit
to KDA, NoPE, GQA, or the 3:1 mixer ratio; it does not promote the architecture, establish a systems or
long-context claim, satisfy the separate novelty gate, or authorize paper-scale training. The three
dense controls still fail the frozen one-hour proxy envelope, and RULER v2, NoLiMa, and HELMET remain
unexecuted or blocked release gates.

## Artifacts

- [Final paired analysis](../results/Speck-Paper1/baseline-analysis.json)
- [Candidate pair 2](../results/Speck-Paper1/runs/Speck-Paper1-Baselines-131M-pair-2-seed-44-order-1073741824-five_cache_kda_gqa.json)
- [Automatic transition](../results/Speck-Paper1/transitions/Speck-Paper1-Baselines-131M-pair-2-seed-44-order-1073741824-five_cache_kda_gqa.json)
- [Frozen disposition](../research/paper-1/proxy_disposition_v1.json)
