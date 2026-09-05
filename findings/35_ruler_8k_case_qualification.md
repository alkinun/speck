# 35 — RULERv1 8K deterministic case qualification

## Result

The same pinned and patched pipeline used for the 4K gate completes all 13 tasks at 8K, with 100
cases per task and seed 42. Two independent complete generations produce identical hashes for every
task and record zero application-layer network attempts. All 1,300 retained rows pass schema and full
length accounting; the largest cases reach exactly 8,192 tokens. The combined case-stream identity is
`d7f4894fd29c8139c8af324ae3c643c4535993bf938eaf45ee73f8ac0ea5b954`.

The retained first pass and logs occupy 34MB, exactly the expected order of growth from the 17MB 4K
cell. `qa_2` reaches at most 8,139 accounted tokens, so the required-document compatibility repair
continues to terminate without forcing an overlength prompt.

## Decision

The 8K data matrix qualifies independently; this still makes no model-capability claim. Determinism,
context fit, network denial, and storage scaling support proceeding to the 16K generation gate. Four
longer matrices and all candidate executions remain blocked.

## Artifact

- [8K case qualification report](../results/Speck-Architecture-Promotion-v1/ruler-cases-8192-qualified.json)
