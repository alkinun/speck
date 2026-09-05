# 89 — Systematic from-scratch hybrid placement overlap

## Direct evidence

Hybrid Architectures for Language Models trains Transformer/Mamba hybrids from scratch on 60B DCLM
tokens and studies placement at 350M and 1B. It crosses 1:1, 1:3, 1:5, and 1:12 attention-to-Mamba
ratios with early, scattered-middle, and later layouts. At 1:12 it moves the single attention block
through seven or eight depths.

Early attention is consistently worse on validation NLL. At 1B, layer-1 attention yields 2.771 DCLM
NLL versus 2.740--2.741 around layers 7--11; at 350M the corresponding values are 2.898 versus
2.861--2.862 around layers 6--8. The denser layout families show the same front-placement penalty.
The authors connect this to nearly uniform early Transformer attention conflicting with Mamba's local
bias, while a learned intra-layer mixture favors Mamba early and Transformer in the middle.

## Scope correction

From-scratch non-uniform placement, avoiding early attention, middle/later attention roles, and joint
ratio-placement recipes are direct prior art. They cannot support a Speck novelty claim.

The source does not learn a frozen predictor or prospectively rank arbitrary unseen layouts. That
technical distinction remains, but it is only an unestablished incremental empirical question. The
study's layouts are hand-designed, it reports no placement uncertainty or replication, and its
mechanistic account is not a fixed-budget causal intervention. None of those limitations is evidence
that Speck's predictor would work or matter.

## Decision

There remain zero established architecture-novelty candidates. N1 placement experiments stay blocked;
the residual prospective-prediction question receives landscape and independent claim review only.
The already-frozen baseline remains independent and continues.

## Artifacts

- [Paper note](../papers/39_hybrid_architectures_systematic.md)
- [Novelty landscape v6](../research/paper-1/novelty_landscape_v6.json)
