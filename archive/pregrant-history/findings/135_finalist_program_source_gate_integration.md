# 135 — One-control integration proof for the pre-commit source gate

## Append-only successor

Finding 134 unit-qualified the exact finite 11-source predicate and established its location in the
existing pre-commit program-validation call. Its cited v1 artifact remains byte-identical. V2 closes
the narrower remaining evidence gap without rewriting v1.

The new temporary fixture presents the complete validator with the first accepted dense control, its
exact result identity, five validation points, eleven finite source losses per point, preserved failure
and rerun history, and the exact second-control successor. The full validator accepts this state. The
same fixture with only `dclm` removed is rejected through the full validator path.

## Decision

Both the successful acceptance path and the incomplete-source failure path are now integration-tested
before the first real result is accepted. The runner, analyzer, plan, thresholds, trainer, active run,
and v1 artifact are unchanged. The independent post-commit acceptance verifier remains mandatory.

## Artifact

- [Program source gate integration](../results/Speck-Paper1/finalist-program-source-gate-v2.json)
