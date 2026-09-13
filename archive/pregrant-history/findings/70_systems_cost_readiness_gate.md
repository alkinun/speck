# 70 — Systems-cost readiness and proxy envelope failure

## Preserved negative result

The v1 RTX 3090 proxy profile requires at least 40,000 steady tokens/s, at most 16 GiB peak allocation,
zero non-finite steps, and at most 1.0 GPU-hour per discovery run. All three dense controls pass
throughput, memory, and finite-state limits but fail total GPU-hours:

| Pair | Active GPU-hours | Steady tokens/s | Peak GiB | Complete envelope |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1.04553 | 40,200.38 | 11.4817 | fail |
| 1 | 1.01288 | 40,203.41 | 11.4927 | fail |
| 2 | 1.01142 | 40,254.82 | 11.4927 | fail |

GPU-hours retain their physical meaning: total active seconds divided by 3,600, including construction,
compilation, validation, training, and checkpoint work represented by the run's active clock. The limit
is not widened and GPU-hours are not relabeled as steady optimizer time. This is a negative cost-envelope
result, not a language-quality failure and not authority to stop the frozen six-run sample.

## Evidence hierarchy

The new gate separates five levels: analytic complexity, isolated operator, full-model runtime, online
serving system, and monetary deployment. Evidence at one level cannot claim the next. In particular,
FLOPs/cache bytes do not prove speed or capacity; a model microbenchmark does not prove online service;
and physical savings do not become dollars without explicit accounting inputs.

Training reports paired GPU seconds and gross/incremental energy to locked quality, steady throughput,
achieved utilization, allocated/reserved memory, and total active GPU-hours. Environment setup,
compilation, initial/intermediate validation, optimizer work, checkpointing, and orchestration remain
separate categories.

Serving crosses prompt lengths 512/4K/32K/128K, batch one, maximum resident batch, and arrival rates at
10/30/50/70/90% of saturation. It separately reports cold, exact-hit, and partial-hit prefixes plus
eager, compiled, and production paths. At least 1,000 requests support p99; failures remain in the
denominator. Sparse-prefill/full-decode or maximum-batch gains cannot claim batch-one sparse decode.

## Energy, memory, and dollars

One-hertz UUID-bound power samples produce trapezoidal gross joules and an idle-baseline-subtracted
incremental value; both are reported. Missing telemetry yields bounded intervals, not invented samples.
Memory separates weights, optimizer/gradients, exact/compressed/sparse/local KV, recurrence, AttnRes,
experts, workspace, fragmentation, host, and disk state.

Monetary claims remain blocked because v1 has neither amortization nor electricity inputs. Owned and
cloud scenarios are separate to avoid double-counting. A datacenter v2 must name accelerators, host,
interconnect, parallelism, runtime, scheduler/cache policy, power scope, prices, and a complete resource/
recovery envelope.

## Decision

The proxy control hard envelope failed. Candidate cost is not yet available. No consumer serving,
datacenter, dollar, architecture-cost, or paper-scale claim is authorized. After the fixed proxy, its
physical results are descriptive; a prospective v2 envelope is required before larger execution.

## Artifact

- [Systems-cost readiness gate](../research/paper-1/systems_cost_readiness_v1.json)
