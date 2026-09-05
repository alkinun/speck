# 85 — Paper 1 KDA/GQA candidate pair 1

## Qualified run

The second five-cache KDA/GQA candidate completed 2,000 steps and 131,072,000 tokens at seed 43 and
packed-data offset 536,870,912. The automatic one-shot finalizer disabled itself, waited for process
exit, collected and validated the checkpoint, committed the result, and scheduled candidate pair 2
after the frozen cooldown. No quality-dependent branch ran.

Final validation loss is 2.796465 nats over 19,988,480 tokens, versus 2.839127 for its matched dense
control: a descriptive candidate-minus-control difference of -0.042662 nats. Every one of eleven source
deltas is negative, ranging from -0.056500 to -0.018805 nats. Together with pair 0 this is encouraging,
but two observations cannot estimate the frozen three-pair confidence bound or make a proxy decision.

## Cost observations

The candidate has 153,958,938 parameters and 1,021,601,280 analytic FLOPs/token at 4K, 21.49% below
the dense control. Steady training took 2,930.33 seconds versus 3,260.22 seconds, 10.12% shorter. Active
time was 3,275.82 seconds (0.9100 GPU-hours), 10.16% shorter, and steady throughput was 44,729 versus
40,203 tokens/s.

Peak allocation was 14,899,848,192 bytes (13.877 GiB), 20.74% above the dense control. The candidate
individually passes the one-hour/16-GiB/40K-token/s envelope, while its dense control fails the one-hour
limit. The hard paired envelope therefore does not pass, and time, memory, and arithmetic remain
separate evidence axes.

## Decision boundary

Candidate pair 1 is complete and qualified. Candidate pair 2 remains mandatory regardless of these two
directions, followed by the frozen six-cell analysis. No architecture promotion, component attribution,
release claim, or paper-scale authority follows from this interim description.

## Artifacts

- [Qualified candidate result](../results/Speck-Paper1/runs/Speck-Paper1-Baselines-131M-pair-1-seed-43-order-536870912-five_cache_kda_gqa.json)
- [Automatic transition](../results/Speck-Paper1/transitions/Speck-Paper1-Baselines-131M-pair-1-seed-43-order-536870912-five_cache_kda_gqa.json)
