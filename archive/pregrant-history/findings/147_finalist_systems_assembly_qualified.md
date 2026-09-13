# 147 — Synthetic systems trial and block assembly qualifies

## File-bound trial assembly

The assembler binds engine, trace, and runtime-attestation paths and hashes to the frozen protocol,
memory supplement, and exact trial. Phase markers must match; benchmark and GPU process PIDs must remain
present in their intervals; conservative energy integration, observed peak NVML bytes, bracketing start
temperature, compiled/no-fallback runtime state, model config, engine hash, and positive CUDA phase times
must all qualify.

Batch evidence now requires a 64-hex digest plus the full 161-microbatch/40-step/accumulation-4/CPU/H2D
contract before comparison. This closes the `None == None` edge discovered during review.

## Paired block behavior

Control and candidate must occupy their frozen positions and share the batch digest and every common
software identity. Model config is intentionally arm-specific. A missing trial, batch mismatch, common
software drift, or greater-than-2°C paired start mismatch produces a retained failed block with no
replacement or retry. A complete synthetic block is accepted by the frozen analyzer.

## Remaining boundary

Eleven tests pass. Runtime attestations and full traces are synthetic inputs; their live producers and
orchestration remain blocked, as do global 12-trial identity validation, actual checkpoint/CUDA preflight,
pipeline qualification, and execution.

## Artifact

- [Assembly qualification](../results/Speck-Paper1/finalist-systems-assembly-qualified-v1.json)
