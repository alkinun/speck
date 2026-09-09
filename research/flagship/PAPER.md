# Flagship paper contract

Status: pre-results narrative and evidence contract, 2026-09-09. This document defines the one paper
the grant is intended to support. It is not a promise that every hoped-for claim will pass. Failed
gates narrow the claim; they do not trigger an unrelated experiment search.

[`paper/claims.json`](../../paper/claims.json) is the machine-readable status and evidence registry for
the claim families below. Manuscript prose, figures, and tables cannot promote a claim independently of
that registry and its checked sources.

## Central question

> How do data quality and the allocation of recurrent versus exact-attention memory determine the
> quality, training cost, and long-context serving cost of a small language model?

The paper follows one chain: qualify the data, select the data recipe, isolate the sequence design,
verify that data and architecture effects transfer and compose, test scale transfer, train one useful
flagship, extend it to 128K, and measure the released system on named hardware. Sparse attention,
MoE, depth routing, and unrelated architecture searches cannot enter this chain during grant 1.

## Contribution and claim ladder

| Claim family | Required evidence | If the gate fails |
| --- | --- | --- |
| Reproducible data improvement | E1W/E1S, E2, E3, E4, neutral two-view held-out data, sealed audit | Publish the measured trade-off and use the frozen fallback |
| Data transfer across architectures | I1 three-seed 2×2 dense/hybrid × prior/selected-mixture comparison | State the interaction and restrict the data claim to the supported architecture |
| Hybrid quality/compute advantage | Completed 150M proxy, grant decisions, dense/hybrid scale ladder, held-out flagship point | Remove the scaling or equivalence claim that failed |
| Component causality | D2, D3, D7, D8 against exact C0 with paired bounds and long-context gates | Keep C0's declared default and report the negative result |
| Combined model validity | I2 assembled recipe versus exact C0 over three seeds | Launch complete C0; never search favorable subsets after output |
| Useful 128K capability | 4K→32K→128K training, RULER v2, internal retrieval/composition, trailing loss, original-4K retention | Report the highest measured effective length, not the allocation ceiling |
| Real systems benefit | Interleaved GH200/RTX 3090/CPU measurements of time, energy, memory, state, TTFT, and TPOT | Retain only analytic claims or suppress the unsupported deployment claim |
| Strong model at its size | Pinned comparator quality table with disclosed token budgets and identical evaluation contracts | Release honestly without a best-at-size headline |

Inherited components are labeled inherited. A contribution is the controlled knowledge established
about a component, its interaction, scale transfer, or realized cost—not the use of an existing
operator.

## Paper spine

1. Problem, cost axes, claims, and non-claims.
2. Model equations and exact parameter/FLOP/state accounting.
3. Data provenance, filtering, deduplication, decontamination, and tokenizer decision.
4. Data source, mixture, repetition, and decay ablations.
5. KDA/global component ablations and completed negative results.
6. I1 data-by-architecture transfer and I2 assembled-recipe confirmation.
7. Dense-versus-hybrid scale law with uncertainty and the flagship as a held-out point.
8. Flagship training trajectory, failures/resumes, throughput, MFU, and energy.
9. Progressive 32K/128K extension and useful-context evaluation.
10. Annealing, merge, and SFT deltas.
11. Quality comparisons with each comparator's disclosed training-token budget.
12. Serving frontier on GH200, RTX 3090, and CPU.
13. Mechanistic interpretation, negative results, limitations, and reproducibility.

The introduction, abstract, and conclusion use this same order. There is no standalone section for an
experiment that does not change a model/data decision, validate an interaction, test transfer, or price
the released system.

## Required figures and tables

1. Architecture diagram plus exact state growth from 4K to 128K.
2. Data funnel and six-category Pareto/guardrail view.
3. E1–E4 effect plot with uncertainty and all domains.
4. D2/D3/D7/D8 forest plot with every seed and capability gate.
5. I1 2×2 main-effect/interaction plot and I2 assembly confirmation.
6. Dense/hybrid quality-versus-FLOPs scale curves with uncertainty and held-out flagship residual.
7. Full flagship learning curve marking stable, pre-decay, decay, and extension boundaries.
8. Position/trailing loss and RULER/internal capability curves through 128K with 4K retention.
9. Training time/energy-to-quality and serving TTFT/TPOT/throughput/state/memory frontiers.
10. Comparator table with model revision, parameters, disclosed tokens, benchmark contract, and result.
11. Negative-result table linking every stopped arm to its preregistered gate.

Each public value resolves to an append-only result artifact, immutable config/data/model identities,
and the exact analysis revision. Aggregate plots retain source/task and per-seed supplements.

## Headline gate

No headline is written before the comparator, held-out flagship, long-context, and systems results
exist. The preferred form remains: the model matches a named comparator at a measured fraction of
training compute and resident 128K state under a named serving envelope. Every variable in that
sentence must come from a checked result; otherwise the sentence is narrowed.

## Scope discipline

- MoBA, MLA, sparse/compressed attention, MoE, and depth routing remain follow-on work.
- E5 curriculum shape is retired before outputs. Stable sampling uses the declared uniform default,
  followed by the E4-selected decay mixture; no curriculum-shape claim is made.
- The 100-hour integrated-validation envelope is mandatory evidence, not general exploration budget.
- Protected reserve repairs mandatory work or preserves the flagship; it never creates a new axis.
- A failed interaction, reversal, or scale transfer is reported and narrows the paper instead of being
  hidden by a new control or favorable subset.
- Before release, SPE-171 performs an independent clean-room regeneration and claim audit; unresolved
  reviewer findings remain limitations rather than being removed from the record.
