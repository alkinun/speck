# 138 — Post-language systems protocol frozen before finalist results

## Removing the calendar confound prospectively

The active language sequence must remain control-first to lock its quality target, so its multi-day
timing is descriptive. Before any result is accepted, the successor systems design is now frozen as six
paired blocks over the exact finalist seed/data cells. Three blocks run control then candidate and three
run candidate then control; every seed receives one of each order. No language outcome selected the
order, workload, endpoint, or threshold.

Each of twelve fresh-process trials restores its retained final checkpoint, uses ten warmup and thirty
measured 4K BF16 optimizer steps, writes no checkpoint, and persists no changed model state. Trials are
surrounded by matched idle and thermal gates. One-hertz GPU board power, temperature, clocks, power
limit, P-state, utilization, memory utilization, fan, throttle, host, and process telemetry are required.

Missing energy intervals are conservatively bounded from zero watts to the recorded power limit. Any
energy claim uses candidate-upper/control-lower ratios, not neighbor-based interpolation. This correction
was made during adversarial qualification, before execution.

## Frozen inference

The two primary endpoints are measured wall seconds/token and gross GPU-board joules/token. Each uses
six paired log ratios, an upper one-sided 97.5% t bound with df=5, an exact 64-assignment sign-flip test,
and same-direction AB/BA point estimates. A joint training-systems claim requires both endpoints plus a
directionally favorable incremental-energy sensitivity result. Missing/failed blocks, outlier deletion,
replacement, retry, or budget extension fail rather than repair the claim.

This protocol cannot run until all twelve language results pass v3 acceptance, the language analysis is
complete, all checkpoints remain, the GPU is free, and the harness, sampler, integrator, and analyzer are
separately qualified. It does not authorize whole-system energy, dollars, serving, datacenter, component,
or paper-scale claims.

## Artifacts

- [Frozen systems protocol](../research/paper-1/finalist_systems_v2.json)
- [Protocol qualification](../results/Speck-Paper1/finalist-systems-protocol-qualified-v1.json)
