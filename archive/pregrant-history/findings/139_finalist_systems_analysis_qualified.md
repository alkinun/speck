# 139 — Fail-closed systems analysis qualified offline

## Exact implementation of the frozen estimands

The CPU-only analyzer accepts up to six hash-bound block artifacts and validates their protocol, block,
pair, order, arm, step/token, timing, energy-interval, telemetry, thermal, OOM, fallback, and finiteness
contracts. A complete block requires both arms; a retained failed block remains in the output and cannot
be replaced or imputed.

For each endpoint it computes the six paired log candidate/control effects, geometric-mean ratio,
one-sided 97.5% df=5 upper bound, all 64 sign-flip assignments, and separate AB/BA means. Time uses the
equal-token wall-time ratio. Gross board energy uses candidate-upper/control-lower ratios; favorable
point estimates are reported separately and cannot override conservative failure.

Metric-specific time and energy decisions remain separate. The joint claim requires both plus a
directionally favorable positive incremental-energy ratio. Missing/failed blocks, a zero conservative
control denominator, or an unfavorable order stratum fail closed.

## Qualification boundary

Eight synthetic adversaries cover a clear joint pass, time-only result, order reversal, missing and
failed blocks, point-versus-bound conflict, zero lower energy, and thermal mismatch. The frozen protocol
is unchanged. The sampler, energy integrator, checkpoint harness, and all execution gates remain blocked.

## Artifact

- [Systems analysis qualification](../results/Speck-Paper1/finalist-systems-analysis-qualified-v1.json)
