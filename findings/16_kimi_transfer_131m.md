# 16 — Kimi-transfer language-model staircase

> **Interpretation update:** finding [19](19_retrieval_specificity_and_replay.md) shows that the
> original counterfactual metric was sensitive to arbitrary record changes and did not prove
> target-specific retrieval. The language-loss results here remain valid; retrieval claims are
> superseded by distractor-controlled exact evaluation.

## Question

Which pieces transferred from Kimi Linear improve Speck's recurrent/global hybrid, and can a
position-free model retain short-context language quality while becoming sensitive to content far
beyond its 4K training length?

The staircase isolates four changes instead of importing Kimi Linear as an inseparable bundle:

1. Speck's historical GDN decay initialization → FLA-style timescales;
2. SiLU → sigmoid output gate;
3. partial RoPE → NoPE in the five global layers;
4. scalar GDN decay → channel-wise KDA decay.

Our KDA keeps Speck's full-rank output gate rather than Kimi Linear's low-rank gate. This makes the
fourth step a controlled decay-granularity comparison, not an exact reproduction of the paper.

## Frozen protocol

All variants use the original `gdn-global` 20-layer layout (`GGGA` repeated), hidden size 768,
sequence length 4,096, seed 42, identical packed-data order, Muon, cosine schedule, and a requested
131,072,000 tokens. The three GDN models have 152,916,468 parameters and 1.019435520 GFLOP/token.
KDA has 153,958,938 parameters (+0.68%) and 1.021601280 GFLOP/token (+0.21%). Final validation uses
the same 19,988,480-token sample as the historical control.

The packed corpus is frozen by manifest SHA-256
`b84b09e0b701e35d84487cf6f91e6da9c9fb686b7f6efe67b2e2f5f301fda98e`. The original control's
final loss is 2.819377661, and the measured three-seed range is 0.00965 nats. Differences smaller
than that range are called unresolved.

## Preflight

All four variants passed compiled forward, backward, clipping, and Muon update on the RTX 3090.

| Variant | Parameters | GFLOP/tok | Preflight tok/s | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| GDN FLA/SiLU/RoPE | 152.916M | 1.0194 | 29.3K | 13.0 GiB |
| GDN FLA/sigmoid/RoPE | 152.916M | 1.0194 | 29.1K | 13.0 GiB |
| GDN FLA/sigmoid/NoPE | 152.916M | 1.0194 | 31.2K | 13.0 GiB |
| KDA/sigmoid/NoPE | 153.959M | 1.0216 | 45.3K | 13.9 GiB |

The first two-step KDA measurement reported 5.4K tok/s because one deferred compilation occupied
the first measured step. It is preserved as a cold-compile diagnostic. A cached five-step rerun had
1.4475-second steps and 45.3K tok/s. Full training is the more reliable systems measurement below.

## Language-model result

| Variant | Final loss | Versus previous | Versus control | tok/s | GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: |
| historical GDN Speck/SiLU/RoPE | 2.819378 | — | — | 47,724 | 0.82 |
| GDN FLA/SiLU/RoPE | 2.827965 | +0.008588 | +0.008588 | 47,774 | 0.816 |
| GDN FLA/sigmoid/RoPE | **2.790629** | **−0.037337** | **−0.028749** | 47,802 | 0.814 |
| GDN FLA/sigmoid/NoPE | 2.814996 | +0.024367 | −0.004382 | 47,892 | 0.811 |
| KDA/sigmoid/NoPE | **2.795380** | **−0.019616** | **−0.023998** | 44,384 | 0.876 |

FLA timescales alone are tied with the control. Sigmoid is the first clear natural-language gain:
it improves every held-out source and exceeds the noise range by roughly 3.9× versus its matched
SiLU predecessor. NoPE gives back 0.0244 nats, though a from-scratch NoPE model is tied with the
historical control rather than suffering the large loss of a late positional switch.

KDA recovers 0.0196 nats of the NoPE penalty. Its final loss is only 0.00475 above the best
RoPE+sigmoid model, inside the noise range, while remaining 0.0240 below the historical control.
The key result is therefore a position-free KDA hybrid statistically tied with the best
short-context staircase point on this seed.

KDA full training is 7.1% slower than GDN and allocates 0.91 GiB more peak VRAM. It uses the same
resident recurrent and KV state at inference; the extra allocation is training workspace and
parameters, not length-growing state.

## Exact-length counterfactual retrieval

The selected three variants were evaluated at 4K, 8K, 16K, 32K, 64K, and 128K with ten samples at
each of three needle depths. Every case has a matched factual/counterfactual prompt. Scores below
are mean log-probability advantages for the answer supported by the prompt.

| Length | RoPE GDN | NoPE GDN | NoPE KDA |
| ---: | ---: | ---: | ---: |
| 4K | −0.0047 | 0.7387 | **1.7065** |
| 8K | −0.0048 | 0.5956 | **1.2079** |
| 16K | −0.0186 | 0.4672 | **0.7167** |
| 32K | −0.0074 | 0.3361 | **0.3858** |
| 64K | 0.0191 | 0.1910 | **0.2108** |
| 128K | 0.0049 | 0.1010 | **0.1448** |

The RoPE model fails even at 4K: its directional test is not significant and its contrastive score
is approximately zero. Both NoPE variants obtain 30/30 correct directions at all six lengths
(`p = 9.31e-10` at 4K) and therefore pass the existing directional-retention gate through 128K.
KDA has the larger absolute score at every length: 2.31× GDN at 4K and 1.43× at 128K.

This is not solved retrieval. Exact match is zero and ten-way candidate accuracy is 23.3% for every
variant at every length. NoPE preserves causal sensitivity to the needle; it does not yet make the
150M model reliably emit the answer. Contrastive score also decays sharply with length even when
directional accuracy remains perfect.

At 128K all three models require 504,860,160 bytes (481.47 MiB) of resident BF16 state. Median
prefill is 3.2540 seconds for RoPE GDN, 3.2376 for NoPE GDN, and 3.3167 for NoPE KDA. NoPE's benefit
is representational, not a material systems shortcut.

## What we learned

1. The output gate mattered more than the imported decay initializer at this budget.
2. RoPE was the dominant blocker for extrapolative content sensitivity.
3. NoPE's short-loss cost is real, but late conversion exaggerated it; from-scratch training is the
   correct protocol.
4. Channel-wise KDA is not merely a synthetic-task effect. It recovers most of NoPE's natural-
   language penalty and strengthens counterfactual sensitivity at every tested length.
5. A 4K-trained position-free KDA hybrid can be short-loss competitive and causally sensitive at
   128K, but it still cannot perform open-vocabulary retrieval.

## Decision

Advance both top architectures to seeds 43 and 44:

- GDN FLA/sigmoid/RoPE is the short-loss control;
- KDA/sigmoid/NoPE is the long-context treatment.

Do not promote either to 32K training from one seed. Replication must establish whether the
0.00475-nat tie and the KDA retrieval advantage survive initialization variance. If they do, the
next controlled stage is matched 32K continuation with the same long-dependency data and mandatory
original-4K regression evaluation.

## Artifacts

- Experiment contracts:
  [experiments/SpeckLC-150M-KimiTransfer131M](../experiments/SpeckLC-150M-KimiTransfer131M)
- Machine-readable summary:
  [results/SpeckLC-150M-KimiTransfer131M/summary.json](../results/SpeckLC-150M-KimiTransfer131M/summary.json)
- Full retrieval records:
  [results/SpeckLC-150M-KimiTransfer131M/retrieval](../results/SpeckLC-150M-KimiTransfer131M/retrieval)
- W&B runs: `shranqi0`, `neyzh2kc`, `khwwdylw`, and `gq3hqrkw`
