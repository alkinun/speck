# 38 — RULERv1 64K deterministic case qualification

The complete 64K stage qualifies under the frozen controls. Both full generations produce identical
hashes for all 13 tasks, all 1,300 rows pass schema and full context accounting, and the socket guard
records zero network attempts. The case-stream identity is
`eed73e822a8f5ab27e9d91b399583b40000bf9f5105115968c57e0c5768f864e`.

The retained first pass and logs occupy 282MB, continuing near-linear growth. The largest accounted
cases reach exactly 65,536 tokens. HotpotQA reaches 65,527 and terminates under the same patch. With
21GB free, the projected ~564MB retained 128K artifact and two temporary generation trees are within
the local storage envelope.

This qualifies 64K data generation only. The reproducibility, fit, network, and storage controls
support the final 128K case-generation stage. The 128K matrix and every candidate capability execution
remain blocked.

## Artifact

- [64K case qualification report](../results/Speck-Architecture-Promotion-v1/ruler-cases-65536-qualified.json)
