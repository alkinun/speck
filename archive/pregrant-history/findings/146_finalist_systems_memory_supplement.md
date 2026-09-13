# 146 — Additive NVML used-memory producer qualifies with mocks

## No primary-protocol change

The interface audit found that peak NVML bytes had no producer. A pre-result additive supplement now
adds only `memory.used`; v2's time/energy endpoints, blocks, workload, thresholds, resource envelope,
claims, and activation gate remain byte-identical.

The successor sampler requests used memory in the same CSV row as UUID, power, thermal, clocks, and
utilization. Finite non-negative MiB values are converted using 1,048,576 bytes/MiB and serialized as
`memory_used_bytes`. Both protocol and supplement hashes travel with the series.

Peak reduction requires at least two samples inside the measured interval plus samples bracketing both
boundaries, then reports the maximum observed bytes without interpolation or imputation. This is a
one-hertz observed peak, not a hardware high-watermark register.

## Boundary

Seven mock/synthetic tests pass. The static producer gap is closed, but live RTX 3090 field support and
assembler consumption are not. Pipeline, execution, and systems claims remain blocked.

## Artifacts

- [Memory supplement](../research/paper-1/finalist_systems_memory_v1.json)
- [Mock qualification](../results/Speck-Paper1/finalist-systems-memory-supplement-qualified-v1.json)
