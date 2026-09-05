# 56 — Paper 1 dense control 0 and collection correction

## Run

The first new Paper 1 baseline completed its fixed 2,000 optimizer steps and 131,072,000 training
tokens on the named RTX 3090. It is the conventional dense-global, parameter-matched control for pair
0: initialization seed 42 and packed-data offset zero.

The final 19,988,480-token validation loss is 2.833646 nats. The trace falls from 10.408564 at step
zero through 3.646370, 3.150873, 2.933668, and 2.834252 at the frozen intermediate boundaries. All
eleven source losses are retained. No non-finite step occurred.

Steady training took 3,260.47 seconds; optimizer time was 3,375.62 seconds, evaluation took 378.99
seconds, and peak allocated device memory was 12,328,394,240 bytes. The model, optimizer, metadata,
timing, completion marker, and run summary are retained on the dedicated volume with independent
hashes.

## Collection failure and correction

The initial collector rejected the valid checkpoint because it compared the requested 20,000,000
validation-token budget with the evaluated token count. The unchanged runtime evaluates only complete
`4 × 4,096 = 16,384`-token validation steps, so the realizable final count is
`floor(20,000,000 / 16,384) × 16,384 = 19,988,480`; intermediate validations similarly evaluate
4,997,120 of the requested 5,000,000 tokens.

The correction was frozen before successful collection and before any candidate checkpoint or result
existed. The collector now derives both counts from the frozen arm geometry and checks every history
entry, including the single-GPU identity stored in resolved checkpoint settings. No training data,
loss, timing, statistical threshold, stopping rule, target lock, or execution order changed. Rerunning
the control could not produce the previously demanded unattainable count and was therefore forbidden.

The first event trigger used `PathExists` and retriggered the failed collector until systemd applied
its start limit. That trigger pattern is retired; later transitions require a single explicit status
and collection action.

## Decision

Dense control 0 is complete and qualified. It is one of three control observations and has no
standalone comparative, component, or promotion authority. Dense controls 1 and 2 must complete before
the control-only time-to-quality target is locked or any candidate result is created or inspected.

## Artifacts

- [Batch-rounding correction contract](../research/paper-1/baseline_collection_v2.json)
- [Qualified control result](../results/Speck-Paper1/runs/Speck-Paper1-Baselines-131M-pair-0-seed-42-order-0-dense_global_param_match.json)
- [Collector implementation](../speck/paper_baseline_analysis.py)
