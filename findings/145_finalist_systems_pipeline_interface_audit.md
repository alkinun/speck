# 145 — Qualified components do not yet form an executable systems pipeline

## Three interfaces align

The integrator emits the analyzer's gross/incremental energy and coverage/gap fields. The engine emits
its primary wall time and torch allocated/reserved peaks. Those three bridges are exact.

## Seven fail-closed blockers

The sampler currently serializes samples, not a full interval/phase-bound trace. Its frozen vector also
lacks `memory.used`, while the analyzer requires peak NVML used bytes. No block orchestrator enforces
idle/thermal/recovery/resource gates or binds sampler PIDs to the engine. Compiled fallback and CUDA
phase timing remain unattested. Batch fingerprints are produced but not compared. No assembler converts
engine+trace evidence into complete or retained-failure block artifacts. Finally, the full software
identity vector is not carried across all stages.

The `memory.used` mismatch is reproduced directly from the current sampler and telemetry required-field
sets. GPU memory-utilization percentage is not treated as a byte measurement.

## Decision

All individual CPU/mock qualifications remain valid; none grants end-to-end or execution authority.
The activation artifact remains absent. The next safe work is additive memory telemetry and a pure
trial/block assembler, followed by orchestration and runtime attestation.

## Artifact

- [Pipeline interface audit](../results/Speck-Paper1/finalist-systems-pipeline-interface-audit-v1.json)
