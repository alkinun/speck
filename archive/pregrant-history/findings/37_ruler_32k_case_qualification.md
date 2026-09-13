# 37 — RULERv1 32K deterministic case qualification

The complete 32K stage qualifies under the frozen controls: 13 tasks by 100 cases, two byte-identical
generations, zero application-layer network attempts, and no fully accounted case above 32,768 tokens.
The case-stream identity is
`a0826626d51c75e01d5846c0a646b982226a7fced674dda469f780e1eebd5c39`.

The retained first pass and logs occupy 140MB, continuing the approximately linear 17/34/69/140MB
storage sequence. HotpotQA reaches 32,763 accounted tokens and terminates under the same hashed
compatibility patch. With 21GB free after retention, a projected ~280MB 64K artifact and its bounded
temporary generations fit the local evidence volume.

This qualifies the 32K data matrix, not model capability. The measured reproducibility, length fit,
runtime progression, and storage headroom support advancing to 64K. The 64K and 128K matrices and all
candidate executions remain blocked.

## Artifact

- [32K case qualification report](../results/Speck-Architecture-Promotion-v1/ruler-cases-32768-qualified.json)
