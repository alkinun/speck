# Flagship paper contract

Status: pre-results allocation-thesis and evidence contract, 2026-09-10. This document defines the
one paper supported by grant 1. It is not a promise that every hoped-for result passes. Failed gates
narrow the relevant claim; they do not trigger a new mechanism, favorable subset, or replacement
paper question.

[`paper/claims.json`](../../paper/claims.json) is the machine-readable status and evidence registry.
Manuscript prose, figures, W&B, and Linear cannot promote a claim independently of checked results and
findings.

## Central thesis

> Under fixed training compute and serving memory, efficient small language models improve through
> allocation: high-information data reduces tokens-to-quality, recurrent layers compress routine
> sequence processing, and a small number of periodically placed exact-attention layers preserve
> global access. These gains count only when they transfer, compose, survive scale and token horizon,
> and materialize on hardware.

The paper follows one causal chain: qualify and select data; isolate recurrent versus exact-memory
allocation; test data-by-architecture interaction and assembled settings; test scale and mature-horizon
transfer; train one fixed 1.2B held-out flagship; extend only to measured useful context; and price the
released system. The model is the held-out consequence of the evidence program, not a configuration
chosen after its results.

## Four primary claims

| Claim | Required evidence | If the gate fails |
| --- | --- | --- |
| C-DATA — data allocation | E1W/E1S, E2–E4, two parser views, six category guardrails, one-opening sealed audit | Publish the trade-off and use the frozen balanced fallback |
| C-MEMORY — exact-memory allocation | Prior paired proxy, C0/D2/D3/D7/D8, dense/hybrid comparisons, 32K/128K and 4K-retention gates, analytic and measured cost | Keep the complete default, report the boundary, and remove unsupported equivalence or component language |
| C-TRANSFER — transfer and composition | Five-seed I1, three-seed I2, scale ladder, 350M mature-horizon S2, flagship held out from fitting | Report interaction/reversal, use complete C0 when required, and restrict the claim to measured scales/horizons |
| C-SYSTEM — held-out realized system | Fixed 1.2B trajectory, pre-decay/final/extended checkpoints, comparator table, quality and useful-context evaluation, prefill/decode/state/energy measurements, release parity | Release honestly without the failed best-at-size, useful-length, or deployment headline |

Inherited components are labeled inherited. The contribution is controlled knowledge about resource
allocation, interaction, transfer, mechanism, and realized cost—not use of an existing operator.

## Paper spine

1. **Resource-allocation problem.** Fixed compute and total deployment footprint; training, prefill,
   decode, runtime state, persistent state, and output-token axes; claims and non-claims.
2. **Controlled framework.** Shared controls, paired seeds and data orders, neutral held-out data, BPB,
   non-inferiority, fixed-token/FLOP/time views, immutable execution, and failure handling.
3. **Allocating training data.** Source funnel, mixture response surface, repetition, decay, category
   guardrails, and sealed confirmation.
4. **Allocating exact memory.** Dense versus KDA/global, component attribution, exact-attention ratio,
   state frontier, middle-integration/final-readout roles, and cache-sharing negative evidence.
5. **Transfer and composition.** I1 interaction, I2 complete assembly, scale transfer, S2 mature-token
   horizon, and every reversal.
6. **Held-out flagship.** Fixed 1.2B training trajectory, predicted versus observed quality, pre-decay
   and final checkpoints, 32K/128K progression, and original-4K retention.
7. **Hardware and release frontier.** Training time/energy, prefill, decode, runtime HBM state,
   persistent-prefix boundary, weights plus state, maximum resident batch, output tokens to fixed task
   quality, comparators, and export parity.
8. **Boundaries.** Negative results, limitations, openness boundary, reproducibility, and one concise
   future direction.

Tokenizer construction, complete source-rights records, every screening arm, post-training operations,
and full artifact tables belong in the supplement unless they change one of the four claims.

## Required figures and tables

1. Resource map decomposing dense and hybrid training compute plus fixed and length-growing state.
2. Six-category data funnel, response surface, Pareto frontier, and category guardrails.
3. E2/E3/E4 paired effects with uncertainty and every domain.
4. Quality–FLOP–state frontier over dense, 3:1, and eligible 5:1 memory allocation.
5. D2/D3/D7/D8 forest plot with every pair and capability gate.
6. I1 factorial interaction and I2 complete-assembly confirmation.
7. Scale curve, S2 mature-horizon sentinel, and held-out flagship residual.
8. Position/trailing loss, RULER/internal capability, and original-4K retention through the highest
   supported length.
9. Training, prefill, decode, energy, runtime-state, peak-memory, and output-token frontiers.
10. Comparator and negative-result tables with exact model, token, hardware, and evidence identities.

Every public value resolves to an append-only result, immutable config/data/model identities, and the
exact analysis revision. Main figures retain per-seed, per-source, and per-task supplements.

## Mechanism contract

I3 preregisters evaluation-only interventions rather than adding training arms:

- reset KDA recurrent state at declared sequence boundaries;
- suppress declared middle versus final global-cache contributions;
- measure effects on language loss, trailing loss, retrieval, composition, and state cost;
- interpret interventions together with prior global-layer and Reader Attention distance evidence.

These interventions test the proposed integration/readout roles. Distribution-shift limitations remain
explicit, and a diagnostic cannot promote an architecture independently of the trained comparisons.

## Systems taxonomy

Report these outcomes separately; absence of an implementation suppresses the corresponding claim:

- training analytic FLOPs, achieved FLOP/s, time, and energy to fixed quality;
- prompt-prefill FLOPs, TTFT, throughput, time, and energy;
- token-decode FLOPs, TPOT, throughput, time, and energy;
- fixed recurrent state and length-growing exact state, both total and bytes per token;
- runtime HBM state, workspace, fragmentation, and peak allocation;
- persistent prefix state, transfer bytes, restore time, and cache-miss recomputation where supported;
- weights plus state at batch one and maximum resident batch;
- generated output tokens and wall-clock time to fixed task quality.

DeepSeek-V4.1-Flash's reported 890 global-cache bytes/token is analytic context only. It is not a
locally measured comparator. Speck state claims are against named matched dense-GQA controls unless a
broader checked comparison exists.

## Headline gate

No headline is written before the comparator, held-out flagship, useful-context, and systems results
exist. The preferred form is: the fixed 1.2B model matches a named comparator or dense control at a
measured fraction of training compute and 128K state under named quality and hardware envelopes.
Every variable must resolve to checked evidence. Allocation length is never described as effective or
usable context.

## Scope discipline

- The public target is fixed at 1.2B/400B; 1.2B/320B is the throughput fallback. The retained 600M
  geometry is not selectable in grant 1.
- CED, CSA2, Reader Attention, FP4 cache QAT, mHC, Engram, DSpark, multimodality, MoE, depth routing,
  and other sparse/compressed-attention work remain outside grant 1.
- The five-seed I1 interaction, three-seed I2 assembly, and S2 mature-horizon sentinel are mandatory
  for their corresponding claims, not general exploration budget.
- I1 cannot reopen E2. I2 applies every compatible promoted setting once; failure launches complete
  C0 without subset search.
- Protected reserve repairs or continues declared work under frozen triggers. It never creates a new
  axis.
- Before release, SPE-171 performs independent clean-room regeneration and claim audit. Unresolved
  reviewer findings remain limitations.
