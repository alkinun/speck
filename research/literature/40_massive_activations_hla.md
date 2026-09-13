# Massive Activations in Hybrid Linear Attention LLMs

- **Paper:** [arXiv:2608.12149v2](https://arxiv.org/abs/2608.12149v2)
- **Version reviewed:** v2, 24 August 2026
- **Code:** [official analysis repository](https://github.com/StartLuxLabs/Massive-Activations-HLA)
- **Models:** [controlled-pretraining checkpoint collection](https://huggingface.co/collections/startlux-models/massive-activations-hla)
- **Primary topic:** architecture-aligned massive-activation dynamics around full-attention layers in
  interleaved linear-attention/state-space hybrids

## Method

The paper identifies a consensus attention-sink token from full-attention maps, then traces that token's
maximum absolute residual-stream coordinate across depth. A local maximum immediately before a
full-attention layer is called a pre-attention spike (PAS). Relative activation retained between
successive spikes is summarized by an inter-spike retention score (ISR); denser attention produces
inter-spike plateaus (ISP).

The main inference-time matrix uses the M-A-P checkpoints: 24-layer RetNet, HGRN, GLA, DeltaNet, and
Gated DeltaNet models at 340M/20B tokens and 1.3B/100B tokens. It covers pure linear attention,
24:1, 12:1, 6:1, and 3:1 layer-to-full-attention ratios, plus full attention. The diagnostic analysis
uses 100 inputs from each of five domains and 10,000 stratified bootstrap resamples. The paper also
checks Kimi Linear, Qwen3.5, Nemotron-H, and Zamba2 checkpoints from 1.2B to 397B parameters.

For controlled pretraining, the authors train 24-layer GDN hybrids from scratch on FineWeb-Edu. The
340M arms use 10B tokens and the 1.3B arms use 50B. A single full-attention layer is placed at depth 4,
12, or 20 in the 340M model; the middle placement is repeated at 1.3B. Separate 3:1 arms study ISP.
Full-attention output gates are added in one intervention and all GDN output gates are removed in
another.

## Evidence

PAS aligns immediately before full attention across the five recurrent mixers, ratios, scales, and
input domains. Denser full attention increases ISR for every architecture/scale pair: all twenty paired
12:1-to-6:1 and 6:1-to-3:1 bootstrap intervals exclude zero. Similar layer alignment appears in the
large public hybrids despite differences in sequence mixer and post-training stage.

The single-attention placement result is direct N1 evidence. At 340M, layer-4, layer-12, and layer-20
models all reach 99.53--100% PAS alignment and similar perplexity/commonsense scores, yet the middle and
late placements substantially outperform the early placement on most retrieval tasks. For example,
FDA accuracy is 8.02/60.43/53.71 and NIAH-3 is 2.40/25.80/69.00 for early/middle/late. Thus PAS location
is nearly deterministic given the architecture, but its binary alignment alone does not rank placement
quality. PAS magnitude grows with placement depth, while retrieval rankings vary by task.

Output gating changes activation magnitude without cleanly controlling capability. Gating the sparse
full-attention layers attenuates PAS and ISP much more than removing every GDN output gate amplifies
them, but the morphologies persist. Retrieval changes sharply and inconsistently across tasks while
language modeling remains similar. The intervention establishes which module regulates the diagnostic;
it does not establish that PAS/ISP magnitude causes better retrieval.

At a representative fixed token-feature coordinate, the authors decompose a PAS into a large pre-
attention write, attention-sink behavior in the full layer, and an opposite-signed update that cancels
the outlier. ISP is interpreted as delayed cancellation. The paper explicitly leaves the computational
roles of transient PAS and persistent ISP unresolved.

## Evidence limits

The placement comparison has three 340M layouts and only the middle layout at 1.3B; no placement
replicates or layout-level uncertainty are reported. Retrieval is evaluated at 2K or 4K rather than a
broad length curve, and the individual tasks disagree enough that there is no demonstrated universal
placement ranking. PAS/ISP is observed after a layout is instantiated; it is not a pretraining-free,
prospective feature for unseen layouts.

The released repository is organized for checkpoint analysis, not from-scratch training. Baseline and
no-GDN-output-gate checkpoints use public FLA, but the gated-full-attention integration is weights-only.
The exact sampled JSONL inputs used for all panels are not released. Code/model identity and behavioral
reproduction require a separate immutable artifact audit before execution.

## What matters for Speck

PAS, ISP, sink-conditioned activation tracing, gate interventions, and early/middle/late single-layer
retrieval are mandatory N1 diagnostics and baselines. A Speck claim cannot treat activation spikes near
full attention, their cross-mixer recurrence, or later-attention retrieval gains as new.

More importantly, the paper supplies a strong counterexample to a simplistic diagnostic placement law:
near-perfect PAS alignment coexists with large placement-quality differences. Any residual prospective
predictor must add incremental held-out information beyond architecture-determined spike location and
must demonstrate task/scale calibration before training held-out layouts. This paper does not show such
a predictor, but it further lowers the prior that the remaining N1 question will produce a useful law.

## Bottom line

Hybrid massive-activation morphology is a real and unusually well-controlled role diagnostic, not yet a
placement selector or causal quality mechanism. It expands the mandatory baseline set and strengthens
the case for independent review before spending compute on N1.
