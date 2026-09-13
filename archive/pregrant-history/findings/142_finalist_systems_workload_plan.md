# 142 — Exact systems workload plan and persistent-mutation detector

## Twelve trials derived without checkpoint access

The planner binds the frozen protocol, run qualification, and materialization hashes, then derives all
twelve block-position trials. Each record carries its exact role, arm, run, experiment, checkpoint,
pair, continuation data window, step/token counts, and unique output path. Planning opens no checkpoint
and creates no output. Checkpoint writes, persisted model/optimizer state, and execution remain false.

Output paths are rejected if they contain or fall inside an experiment/checkpoint tree. The command
guard hashes every protected file before and after a mocked child, detecting content, inventory, or
deletion changes even when the child also raises. A nonzero child with unchanged protected bytes is
propagated normally.

## Honest isolation boundary

Before/after identity is persistent-mutation detection, not prevention. A transient write restored
byte-for-byte could escape it. Kernel-enforced read-only path isolation and the benchmark engine must be
separately qualified, followed by a post-sequence no-output GPU preflight. The active checkpoints were
not opened or statted by this work.

## Artifact

- [Workload-plan qualification](../results/Speck-Paper1/finalist-systems-workload-plan-qualified-v1.json)
