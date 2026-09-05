# 102 — Novelty artifact inventory after KL-selector audit

## Inventory

The KL-selector adds an eighth code repository and a tenth immutable source/checkpoint snapshot. Its
203-file tree contains broad conversion and selection source plus 171 experiment configs, but no root
license or tests. Across nineteen sources, root-licensed code paths remain three. Seven metadata-only
trees have been inspected; no working tree, import, or third-party execution occurred.

## Boundary

The paper's held-out-KL and early-stop procedures are not reproduced by the inspected ranking script;
dependencies, data order, checkpoints, and W&B logs are unpinned. Execution, reuse, behavior, and full
reproduction remain blocked, so the new tree creates no qualified reproduction path.

## Artifact

- [Novelty code availability v8](../research/paper-1/novelty_code_availability_v8.json)
