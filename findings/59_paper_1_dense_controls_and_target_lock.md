# 59 — Paper 1 dense controls complete and target locked

## Third control

Dense control pair 2 completed 2,000 optimizer steps and 131,072,000 tokens at initialization seed 44
and packed-data offset 1,073,741,824. Its final 19,988,480-token validation loss is 2.832686 nats.
All six validation boundaries, eleven source losses, checkpoint identities, and finite-state checks
qualify. Steady training took 3,256.06 seconds.

## Three-control envelope

The three final dense losses are 2.833646, 2.839127, and 2.832686 nats. Their mean is 2.835153,
sample standard deviation is 0.003475, and range is 0.006440 nats. These are control-only variation
estimates across the frozen seed/data cells, not comparisons with the candidate.

Steady training times are 3,260.47, 3,260.22, and 3,256.06 seconds. Their 4.41-second range is 0.135%
of the mean, showing a stable realized dense-control runtime after excluding frozen startup and
evaluation categories.

## Target lock

The preregistered rule takes the maximum final loss across all three complete controls and rounds it
upward to six decimal places. The immutable time-to-quality target is therefore 2.839127 nats. The
lock pins all three control-result hashes and was created before any candidate checkpoint directory or
candidate result record existed.

The target is guaranteed reachable by every control. A candidate that does not cross it remains
right-censored; no complete-case substitution is allowed.

## Decision

The control phase is complete. Candidate pair 0 may begin after the frozen cooldown and live launch
gate. All three candidate cells remain mandatory regardless of interim loss, and no control-only
observation has architecture-promotion authority.

## Artifacts

- [Dense control 2 result](../results/Speck-Paper1/runs/Speck-Paper1-Baselines-131M-pair-2-seed-44-order-1073741824-dense_global_param_match.json)
- [Control-only target lock](../results/Speck-Paper1/baseline-time-to-quality-target.json)
- [Automatic transition](../results/Speck-Paper1/transitions/Speck-Paper1-Baselines-131M-pair-2-seed-44-order-1073741824-dense_global_param_match.json)
