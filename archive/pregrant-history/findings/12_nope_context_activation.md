# 12 — Same-parent NoPE context activation

## Question

Does replacing partial scaled RoPE with NoPE in all five promoted global layers improve a GDN
hybrid during the existing 32K context-activation stage?

This directly tests the most transferable result from Kimi Linear while holding Speck's recurrent
mixer fixed. It is a continuation intervention, not a from-scratch NoPE architecture comparison.

## Controlled design

Both variants start from the exact original `gdn-local` checkpoint:

- parent step: 2,000 at 131,072,000 tokens;
- parent model SHA-256: `ae692777c0f3603261a935c7c56c5a05be8d9768e85eb4eddd444236e5fe46e1`;
- global layers: 3, 7, 11, 15, and 19;
- sequence length: 32,768;
- continuation request: 32,000,000 tokens;
- consumed continuation: 32,047,104 tokens over 489 optimizer steps;
- data manifest: `0a14833ad84d0f240fd7787e542c47c2f77f40d73427c207cc7ae6b2a95f9da0`;
- seed: 42;
- Muon, learning rate `1e-4`, 25-step warm-up, cosine decay to 10%;
- batch tokens: 65,536;
- Liger loss and activation checkpointing;
- 152,916,468 parameters and 1.6800384 GFLOP per training token.

The completed frontier's `global-5` is the control. It uses 32 of 64 RoPE dimensions in every
global head and 8× global RoPE scaling. The treatment changes exactly those five `rope_dim` values
from 32 to 0. A recursive comparison found no other normalized model-config difference.

## Runtime qualification

The treatment passed compiled forward, backward, gradient clipping, and Muon update before
training:

| Variant | Synthetic tok/s | Peak allocated |
| --- | ---: | ---: |
| RoPE control | 15,446.5 | 4.64 GiB |
| NoPE treatment | 15,143.0 | 4.64 GiB |

These two-step preflights are noisy and prove execution rather than stable throughput.

## Training result

| Variant | Initial 32K loss | Final 32K loss | Change | tok/s | GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: |
| RoPE control | 2.855830 | 2.634904 | -0.220925 | 19,043.1 | 0.4621 |
| NoPE treatment | 3.142328 | 2.699041 | -0.443288 | 19,207.8 | 0.4584 |

The immediate NoPE intervention costs `0.286499` nats at 32K. Continuation recovers far more loss
than the RoPE control, but NoPE still finishes `0.064136` nats worse. This is more than six times the
measured `0.00965`-nat seed range.

NoPE is worse on every held-out long-document source:

| Source | RoPE | NoPE | NoPE minus RoPE |
| --- | ---: | ---: | ---: |
| FineMath 4+ | 2.489268 | 2.545711 | +0.056443 |
| peS2o | 2.901843 | 2.965723 | +0.063880 |
| Wikimedia | 2.562148 | 2.636797 | +0.074650 |

The 32K validation set contains only 327,680 evaluated tokens, but the direction is consistent
across all three sources and large relative to the measured base-stage noise range.

## Original-4K regression

The exact comparison protocol uses batch 4 and 19,988,480 original-corpus tokens:

| Checkpoint | 4K loss | Change from common parent |
| --- | ---: | ---: |
| common parent | 2.810548 | — |
| RoPE control | 2.826227 | +0.015679 |
| NoPE treatment | 2.874594 | +0.064045 |

NoPE is `0.048367` nats worse than the trained RoPE control. It therefore fails both the final 32K
language-loss objective and the original-4K retention gate.

An initial diagnostic accidentally used batch 1, evaluating 19,996,672 tokens and obtaining
`2.874986`. It was preserved, renamed explicitly, and replaced by the exact batch-4 protocol rather
than silently discarded. The difference does not change the conclusion.

## Counterfactual retrieval

Every length contains 30 paired factual/counterfactual cases over depths 0.1, 0.5, and 0.9.

| Length | RoPE directional accuracy / score | NoPE directional accuracy / score |
| ---: | ---: | ---: |
| 4K | 93.3% / 0.1777 | 100% / 1.4238 |
| 8K | 86.7% / 0.0538 | 100% / 1.2331 |
| 16K | 83.3% / 0.0281 | 100% / 1.1194 |
| 32K | 70.0% / 0.0134 | 100% / 0.8876 |
| 64K | 50.0% / 0.0029 | 100% / 0.6645 |
| 128K | 63.3% / 0.0040 | 100% / 0.5870 |

For NoPE, all six directional tests have 30/30 successes and one-sided binomial
`p = 9.31e-10`. Its effective length under the existing 85%-of-4K directional-retention rule and
its longest statistically detectable length are both 128K. The corresponding RoPE values are 16K
and 32K.

The mean NoPE contrastive score is about 8.0× the RoPE score at 4K and 145× at 128K. The ratio at
128K should not be overinterpreted because the RoPE denominator is near zero; the absolute NoPE
score of `0.5870` is the meaningful observation.

Open-vocabulary exact match remains zero for both models at every length. Candidate accuracy is
23.3% for both at every length. The result demonstrates strong causal sensitivity to distant
content, not reliable answer generation.

## Systems result

State geometry is identical: both use 504,860,160 bytes (481.47 MiB) of BF16 resident state at
128K. Canonical NoPE prefill is 3.236 seconds at 128K versus 3.262 seconds for RoPE. Full training
throughput differs by only 0.87% in NoPE's favor. These differences are too small to claim a systems
advantage; removing rotary work is not material beside five global attention layers.

## Reporting defects caught

The first retrieval artifact incorrectly marked NoPE as extrapolating global RoPE because the
evaluator inferred that condition from attention scope alone. The evaluator now records active RoPE
dimensions by scope and requires a positive global `rope_dim` before reporting extrapolation. A
test covers pure NoPE, RoPE, and mixed local-RoPE/global-NoPE cases.

The incorrect-metadata artifact is preserved under `results/.../diagnostics`; the canonical curve
was rerun after the fix and reproduced every quality metric exactly. Only timings changed slightly.

## Interpretation and decision

The outcome is a real Pareto conflict:

- NoPE makes the pretrained hybrid substantially more sensitive to distant content.
- The same intervention damages next-token modeling at both 4K and 32K.
- Thirty-two million continuation tokens recover much, but not enough, of the abrupt positional
  distribution shift.

This partially validates Kimi Linear's mechanism but rejects naive post-hoc conversion as our
training recipe. Kimi trains NoPE global layers as part of the base architecture; Speck switched
already-trained sliding-RoPE layers only at context activation. The next fair test is therefore
from-scratch GDN-sigmoid/NoPE and KDA-sigmoid/NoPE pretraining, after the synthetic memory
factorization. Spending more tokens on this one-seed late-conversion branch would answer a narrower
adaptation question and reuse the same weak long-document supervision.

Do not promote this checkpoint as a release candidate. Retain it as evidence that positional
encoding strongly controls the loss-versus-retrieval frontier.

## Artifacts

- Experiment contract:
  [experiments/SpeckLC-150M-KimiTransfer32K](../experiments/SpeckLC-150M-KimiTransfer32K)
- Consolidated result:
  [results/SpeckLC-150M-KimiTransfer32K/summary.json](../results/SpeckLC-150M-KimiTransfer32K/summary.json)
- Canonical retrieval and short-loss records:
  [results/SpeckLC-150M-KimiTransfer32K](../results/SpeckLC-150M-KimiTransfer32K)
- Checkpoint:
  `~/.cache/speck/checkpoints/SpeckLC-150M-KimiTransfer32K-global-5-nope`
- Model SHA-256: `b963e89d6154c00deb9741853369aef9e51c55e3d0cf568b2339e30eaac84716`
- W&B run: `bgohscb1`
