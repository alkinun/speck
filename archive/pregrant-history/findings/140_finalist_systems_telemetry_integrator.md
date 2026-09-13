# 140 — Conservative systems telemetry integration qualified synthetically

## Strict trace identity

Each sample now requires strictly increasing monotonic and timezone-aware UTC timestamps, the exact GPU
UUID, finite physical power/thermal/clock/utilization/fan fields, P-state and throttle state, host load,
memory and disk counters, display state, and GPU/benchmark process inventories. Power above its recorded
limit, a UUID change, malformed process data, or an unbracketed interval is rejected.

All eight frozen phase markers and both exact 120-second flanking idle intervals are mandatory. Warmup
ends exactly where measurement begins, measured markers bind the integration interval, and the process
must remain between its idle blocks.

## Conservative missing-sample propagation

The point estimate uses trapezoidal board-power integration. Each gap receives one nominal observed
second; excess duration is explicitly missing and bounded from zero watts to the maximum recorded power
limit spanning it. Coverage and longest gap are preserved, with the frozen 99% and 2.5-second gates.

Incremental energy subtracts the duration-weighted flanking idle baseline with interval arithmetic:
gross lower minus idle upper, point minus point, and gross upper minus idle lower. A removed sample
expands the interval and fails coverage in the synthetic adversary rather than being silently filled.

## Boundary

Ten synthetic tests pass. The schema and integrator are qualified, but live `nvidia-smi`/NVML
acquisition is not invoked or qualified. The checkpoint workload harness and execution gate remain
blocked, and the active language experiment is unchanged.

## Artifact

- [Telemetry-integrator qualification](../results/Speck-Paper1/finalist-systems-telemetry-integrator-qualified-v1.json)
