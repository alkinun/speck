# 141 — Systems sampler implementation qualified without a live query

## Exact acquisition surface

The sampler requests one CSV row containing the frozen UUID, power, temperature, graphics/memory
clocks, power limit, P-state, GPU/memory utilization, fan, active throttle reasons, and display state.
A second query binds compute PIDs to that UUID. Unsupported numeric fields, malformed rows, invalid
PIDs, multiple GPU rows, or downstream physical/schema violations fail.

Host acquisition records aggregate CPU delta utilization, load average, available memory, explicit
disk read/write counters, and benchmark process identity. Disk devices must be named explicitly and all
must exist in `/proc/diskstats`; automatic “all device” aggregation is forbidden because it can
double-count device-mapper and its backing disk.

The finite sampler uses monotonic deadlines rather than sleeping one second after each query, preventing
query latency from accumulating into schedule drift. Actual overruns remain visible in the timestamps
and are handled by the already-qualified coverage/gap rules.

## Boundary

Nine tests use only mocked command output and synthetic host files. The parser, composition, deadline
schedule, protocol hash, and explicit device serialization qualify. No live `nvidia-smi`, `/proc`, or
disk query ran. Live field support, backing-device selection, co-running sampler behavior, workload
harness, and execution remain blocked.

## Artifact

- [Mock sampler qualification](../results/Speck-Paper1/finalist-systems-sampler-qualified-v1.json)
