# 26 — Paper 1 research program

## Objective

Define the evidence required for Speck's first efficient-architecture paper before selecting a final
architecture or launching paper-scale pretraining.

The target is the methodological depth of Kimi Linear, Kimi K3, and DeepSeek-V4: mathematical
operators, controlled ablations, scaling, training and serving systems, long-context behavior,
mechanistic analysis, broad evaluation, negative results, limitations, and reproducible artifacts.
Page count and model size are not the target.

## Working thesis

The program organizes efficiency as information flow across three independent axes:

1. **Sequence:** local raw context, fixed-state recurrence, compressed global coverage, and selective
   precise retrieval.
2. **Depth:** content-dependent access to earlier representations through bounded depth routing.
3. **Width:** sparse expert activation with stability, balance, memory, and communication contracts.

Candidate ingredients include SWA, KDA, GQA/MQA/MLA, HCA, CSA, Full/Block Attention Residuals, and
Stable LatentMoE. None is selected by this finding.

## Novelty decision

An assembly of published mechanisms is insufficient for Paper 1. Before paper-scale pretraining, at
least one matched and replicated result must establish:

- a new Speck mechanism;
- a new predictive composition rule;
- a generalizable mechanism/quality/cost relationship over at least three scales; or
- a systems method that materially changes the feasible architecture frontier.

If the novelty is narrower than the tri-axis thesis, the paper and title narrow with it. The project
must not preserve a broad narrative by weakening its evidence requirements.

## Experiment order

### Sequence

1. Establish dense global, SWA, GDN/global, and KDA/global controls.
2. Hold five independent global memories fixed and compare GQA3, MQA1, and NoPE MLA.
3. Test HCA compression rates against the selected exact-cache representation.
4. Test block-level CSA only after dense compressed attention and selector metrics qualify.
5. Add the smallest raw SWA branch that recovers local and incomplete-block information.
6. Sweep recurrent/global ratio and placement after the operators are fixed.

### Depth

1. Compare standard residual, Full AttnRes, and Block AttnRes.
2. Sweep summary count.
3. Repeat the depth/width geometry search for both standard and promoted residual rules.

### Width

1. Establish dense SwiGLU versus conventional top-k MoE.
2. Isolate latent projection and latent normalization.
3. Isolate bounded activation.
4. Isolate balancing strategy.
5. Select expert count/top-k geometry only after stability is established.

Only independently promoted mechanisms enter pairwise interactions. The final three mechanisms require
the complete `2^3` presence/absence cube and a final removal matrix.

## Scale ladder

- 30M–60M mechanism qualification.
- 130M–170M, 131M-token discovery without promotion authority.
- 130M–170M finalists at a minimum ten tokens per parameter with a `3 × 2` seed/data-order design.
- 300M–600M three-pair scale transfer on hardware and parallelism frozen in advance.
- One matched 1B–1.5B/20B-token sentinel to detect reversal.
- Paper-scale pretraining only after every launch gate passes.

A scaling-efficiency claim additionally requires at least five compute-optimal points per final
architecture, uncertainty on the fitted curve, residual analysis, and a held-out point.

## Required analysis depth

- KDA decay/timescale, overwrite, state norm, and write strength.
- Compression fidelity and dense-attention mass captured by compressed/sparse selection.
- Selector recall connected to retrieval and composition failures.
- AttnRes source weights, entropy, age, residual magnitude, and gradient flow.
- Expert utilization, routing margins, specialization, redundancy, overflow, communication, and
  activation/gradient outliers.
- Operator-level wall-clock, energy, persistent state, workspace, fragmentation, and peak memory.
- Counterexamples where an attractive internal diagnostic fails to predict held-out capability.

## Paper-scale pretraining status

**Blocked.** The checked [`experiment_program.json`](../research/paper-1/experiment_program.json) lists
ten required gates, including novelty, isolated component promotion, interaction evidence,
longer-horizon proxy replication, medium-scale transfer, independent evaluation, target kernels,
consumer/datacenter cost envelopes, and a complete resource budget.

## Artifacts

- [Paper 1 program](../research/paper-1/)
- [Claim and falsification ledger](../research/paper-1/claims.json)
- [Experiment program](../research/paper-1/experiment_program.json)
- [Reference depth audit](../research/paper-1/reference_audit.md)
- [Manuscript outline](../research/paper-1/paper_outline.md)
- [Reporting checklist](../research/paper-1/reporting_checklist.md)

Primary reference reports:

- [Kimi Linear](https://arxiv.org/abs/2510.26692)
- [Kimi K3](https://arxiv.org/abs/2607.24653)
- [DeepSeek-V4](https://arxiv.org/abs/2606.19348)

