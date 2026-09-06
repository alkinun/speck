# Research organization

Only [`flagship/`](flagship/) defines current work. It separates the stable scope, flexible execution
order, pre-grant readiness, machine-readable budget, and non-launchable model targets:

- [`flagship/README.md`](flagship/README.md) — model, paper, experiment, data, evaluation, and release
  scope.
- [`flagship/EXECUTION.md`](flagship/EXECUTION.md) — 90-day phase order and flexibility rules.
- [`flagship/PREGRANT.md`](flagship/PREGRANT.md) — readiness checklist before allocated compute.
- [`flagship/plan.json`](flagship/plan.json) — exact GPU-hour, dependency, reserve, and contingency
  contract.
- [`flagship/targets/`](flagship/targets/) — planning geometry only; never launch these as experiments.

## Reusable governance

[`architecture-promotion-v1/`](architecture-promotion-v1/) retains the statistical and evaluation
infrastructure that the flagship reuses:

- `policy.json` — paired non-inferiority, power, and promotion rules.
- `cost_envelopes.json` — named training and serving measurements.
- `evaluation_manifest.json` — pinned internal protocols and RULER v2 disposition.
- `evidence_matrix.json` — historical component evidence.
- `internal/` — structured retrieval and symbolic composition protocols.
- `external/` — pinned upstream-source records. NoLiMa and HELMET are reference-only and cannot block
  the first flagship.

This directory is governance, not an invitation to reactivate every archived axis. The active
flagship plan narrows its use to relevant statistics, internal protocols, RULER v2, cost accounting,
and evidence provenance.

Validate the reusable contract with:

```bash
uv run --extra cpu python -m scripts.research_contract_validate \
  research/architecture-promotion-v1 \
  --tokenizer-experiment experiments/SpeckLC-150M-KimiTransfer131M/kda-sigmoid-nope
```

Preflight the internal protocol cases without loading a checkpoint:

```bash
uv run --extra cpu python -m scripts.promotion_case_preflight \
  research/architecture-promotion-v1/internal/structured_retrieval_v2.json \
  --tokenizer-experiment experiments/SpeckLC-150M-KimiTransfer131M/kda-sigmoid-nope
```

## Archive boundary

`paper-1/` contains JSON contracts from the retired tri-axis/finalist program. They remain because
checked results hash or cite them; they have no launch or scope authority. The chronological narrative
is under [`findings/ARCHIVE.md`](../findings/ARCHIVE.md), while
[`findings/README.md`](../findings/README.md) exposes only evidence relevant to the active flagship.

Historical results and experiment configs remain reproducibility artifacts. A path under
`research/paper-1/`, `results/Speck-Paper1/`, or an archived findings index must not be interpreted as
planned work.
