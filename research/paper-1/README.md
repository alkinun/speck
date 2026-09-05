# Speck Paper 1 research program

## Working thesis

> A language model can improve quality per training and serving cost by controlling information flow
> independently across sequence, depth, and width, then composing only mechanisms whose isolated
> benefits survive replication, scale transfer, and realized runtime measurement.

The three axes are:

- **Sequence:** local raw context, fixed-state recurrence, compressed global coverage, and selective
  precise retrieval.
- **Depth:** content-dependent access to earlier representations without unbounded residual dilution.
- **Width:** sparse expert activation with explicit stability, balance, memory, and communication
  contracts.

This is a research hypothesis, not the name of a finished architecture. Candidate mechanisms include
SWA, KDA, GQA/MQA/MLA, HCA, CSA, Block Attention Residuals, and Stable LatentMoE. A component enters the
final architecture only after passing the active
[`architecture-promotion-v1`](../architecture-promotion-v1/) policy.

## Novelty gate

Combining published mechanisms is not sufficient novelty. Before any paper-scale pretraining run,
Speck must establish at least one of the following under a matched and replicated design:

1. a new sequence, depth, or width mechanism with a reproducible advantage;
2. a new composition rule that predicts when known mechanisms cooperate or interfere and survives a
   held-out interaction test;
3. a generalizable empirical law connecting an internal mechanism diagnostic to quality and realized
   cost across at least three scales; or
4. a systems method that changes the feasible quality-cost frontier and is inseparable from the model
   design.

An implementation contribution alone may support the paper, but it cannot satisfy this gate without a
scientific claim and evidence.

## Reference-paper standard

The program adopts the coverage—not the conclusions—of three primary references:

- [Kimi Linear](https://arxiv.org/abs/2510.26692) develops an operator and chunkwise algorithm, tests
  synthetic mechanisms, isolates key components and layer ratios, fits scaling laws, compares matched
  large models through pretraining/SFT/RL, and reports long-context and serving efficiency.
- [Kimi K3](https://arxiv.org/abs/2607.24653) organizes architecture around information flow across
  sequence, depth, and width, then covers data, scaling, training, long-context extension,
  infrastructure, serving, broad evaluation, and cost efficiency.
- [DeepSeek-V4](https://arxiv.org/abs/2606.19348) specifies compressed/sparse attention mathematically,
  analyzes cache and FLOPs, documents training and inference systems, reports data/training/stability
  details, and separates base, post-trained, real-world, and limitation sections.

Their reports also illustrate a limitation Speck should improve upon: whole-system gains do not isolate
every component. Paper 1 therefore requires explicit component, removal, and interaction evidence before
the final combined model.

## Files

- [`claims.json`](claims.json) is the claim and falsification ledger.
- [`baseline_matrix.json`](baseline_matrix.json) audits historical controls and freezes the first
  parameter-matched dense/KDA paired design, data orders, matching views, storage gate, and non-claims.
- [`experiment_program.json`](experiment_program.json) freezes baselines, stages, scales, axes, and the
  paper-scale pretraining gate.
- [`paper_outline.md`](paper_outline.md) defines the manuscript structure and required evidence in each
  section.
- [`reference_audit.md`](reference_audit.md) maps the evidence depth of Kimi Linear, Kimi K3, and
  DeepSeek-V4 to explicit Speck requirements.
- [`reporting_checklist.md`](reporting_checklist.md) is the release-grade completeness checklist.

Validate the program with:

```bash
uv run --extra cpu python -m scripts.paper_program_validate research/paper-1
uv run --extra cpu python -m scripts.paper_baseline_prepare \
  research/paper-1/baseline_matrix.json --check
```

## Current state

The five historical sequence controls are now identity-audited as discovery evidence only. A new
153.96M-parameter dense/KDA baseline pair is materialized across three paired initialization/data-order
cells, but launch remains blocked on the evaluation-manifest dependency, paired GPU preflight, frozen
analysis code, and the 16GiB free-space floor. The project otherwise has strong evidence for GDN/KDA
trade-offs, the need for some global attention, a global-cache sharing failure frontier, and rigorous
promotion infrastructure. It does **not** yet have:

- a promoted sequence architecture;
- a local implementation or isolation of HCA/CSA, AttnRes, or Stable LatentMoE;
- a demonstrated Speck-specific architectural novelty;
- medium-scale transfer;
- independent long-context results; or
- a production serving runtime.

Accordingly, the paper is in **thesis and experiment-design**, not model-training or manuscript-claim,
status.
