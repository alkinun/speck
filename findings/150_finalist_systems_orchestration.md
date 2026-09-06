# 150 — Systems block orchestration qualifies with recording adapters

## Exact successful path

One sampler spans the 120-second pre-idle, first gated trial, bounded thermal recovery, second gated
trial, and 120-second post-idle. Only after sampler stop may traces and the block be assembled. Both
trials require the exact UUID, at most 45°C, zero utilization, and no other compute PIDs. The second
start must also be within 2°C of the first before its engine launches.

The four-hour program budget is checked before a block and after assembly. Only a qualified block can
authorize a successor, and block 5 never does.

## Failure path

The first exception stops progression, attempts sampler cleanup, retains any cleanup failure plus the
event prefix, and authorizes neither replacement, retry, nor successor. Exhausted budget fails before
starting a sampler. Review moved the paired-temperature check ahead of the second engine rather than
invalidating work afterward.

Seven recording-adapter tests pass. No real sleep, sampler, engine, GPU, checkpoint, trace, or failure
write occurred. The live adapter, real concurrency, runtime-probe production, end-to-end pipeline, and
execution remain blocked behind the absent activation artifact.

## Artifact

- [Orchestration qualification](../results/Speck-Paper1/finalist-systems-orchestration-qualified-v1.json)
