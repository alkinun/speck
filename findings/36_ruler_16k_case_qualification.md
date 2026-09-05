# 36 — RULERv1 16K deterministic case qualification

The 16K stage passes the same frozen gate: all 13 tasks, 100 cases per task, two complete
byte-identical generations, zero application-layer network attempts, and no fully accounted case over
16,384 tokens. The 1,300-case identity is
`7ddcb6b201298fd3d6c6501b0d307719b7c411580f76b84c8e94e1e138042084`.

The retained first pass and logs occupy 69MB. This remains approximately linear relative to the 17MB
4K and 34MB 8K artifacts. HotpotQA reaches 16,383 accounted tokens and terminates under the same
control-flow-only compatibility patch.

This qualifies the 16K data matrix, not model capability. Reproducibility, context fit, network denial,
and storage growth support advancing to 32K. The 32K, 64K, and 128K matrices and all candidate
executions remain blocked.

## Artifact

- [16K case qualification report](../results/Speck-Architecture-Promotion-v1/ruler-cases-16384-qualified.json)
