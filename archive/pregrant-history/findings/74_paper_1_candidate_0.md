# 74 — Paper 1 KDA/GQA candidate pair 0

## Qualified run

The first five-cache KDA/GQA candidate completed 2,000 steps and 131,072,000 tokens at seed 42 and
packed-data offset zero. The automatic one-shot finalizer disabled itself, waited for process exit,
collected and validated the checkpoint, committed the result, and scheduled candidate pair 1 after the
frozen cooldown. No quality-dependent branch ran.

Final validation loss is 2.794477 nats over 19,988,480 tokens, versus 2.833646 for its paired dense
control: a descriptive candidate-minus-control difference of -0.039170 nats. Every one of eleven source
deltas is negative, ranging from -0.046295 to -0.024414 nats. These are one-cell observations, not a
confidence bound or proxy decision.

## Cost observations

The candidate has 153,958,938 parameters and 1,021,601,280 analytic FLOPs/token at 4K, 21.49% below
the dense control's analytic count. Steady training took 2,931.57 seconds, versus 3,260.47 seconds:
10.09% shorter in this pair. Total active time is 3,344.99 seconds (0.9292 GPU-hours), 11.13% shorter.

Peak allocated memory is 14,897,357,824 bytes (13.874 GiB), 20.84% above the dense control despite
lower analytic FLOPs and shorter time. This is why runtime, memory, and arithmetic remain separate axes.
The candidate individually fits the v1 one-hour/16-GiB/40K-token/s envelope, but the paired control does
not; no complete comparative cost-envelope pass follows.

## Decision boundary

Candidate pair 0 is complete and qualified. It cannot pass the aggregate or source non-inferiority gates
alone, estimate paired variance, support component attribution, or justify a systems claim. Candidate
pairs 1 and 2 remain mandatory regardless of their outcomes, followed by the frozen six-cell analysis.

## Artifacts

- [Qualified candidate result](../results/Speck-Paper1/runs/Speck-Paper1-Baselines-131M-pair-0-seed-42-order-0-five_cache_kda_gqa.json)
- [Automatic transition](../results/Speck-Paper1/transitions/Speck-Paper1-Baselines-131M-pair-0-seed-42-order-0-five_cache_kda_gqa.json)
