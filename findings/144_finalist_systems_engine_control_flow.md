# 144 — Non-persisting systems engine control flow qualifies on CPU fixtures

## Why the full trainer cannot be reused directly

Resuming a complete finalist checkpoint under the frozen horizon executes zero steps. Starting a normal
branch is also unsuitable: the full lifecycle always performs final validation, checkpoint serialization,
and summary output. The systems engine therefore reuses only the frozen initialization, verification,
model/optimizer construction, parent restore, loader, compilation preparation, and `optimization_step`.

It fixes learning rate at the frozen terminal cosine value (`0.0015 * 0.1 = 0.00015`), skips tracking,
validation, checkpointing, and summaries, resets peak memory after ten warmup steps, and measures exactly
thirty more steps. Cleanup runs even when optimization raises.

## Paired data and H2D semantics

To satisfy both frozen data materialization and H2D-inclusive timing, the engine rebuilds the checkpoint
continuation stream on CPU. A real trial holds `1 + 40 * 4 = 161` microbatches, hashes every input/target
shape, dtype, byte sequence, and loader state, then replays CPU-to-device transfers inside optimizer
windows. Future paired arms must have identical fingerprints.

## Hard activation boundary

The CLI requires a future activation artifact that pins the protocol and engine, records twelve v3-
accepted language results, and passes an actual-path GPU preflight. That artifact does not exist; direct
invocation fails before checkpoint or CUDA access and creates no output.

Seven CPU-fixture tests pass. Actual checkpoint loading, compiled CUDA, fallback detection, phase timing,
paired real fingerprints, retained failure orchestration, and live sampler co-execution remain blocked.
This is engine-control-flow evidence, not performance evidence.

## Artifact

- [Engine control-flow qualification](../results/Speck-Paper1/finalist-systems-engine-qualified-v1.json)
