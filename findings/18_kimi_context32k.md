# 18 — Matched 32K KDA/NoPE context activation

## Question

After the three-seed base-stage result, does KDA/sigmoid/NoPE retain its retrieval advantage when
trained on 32K natural documents, and what does it cost in long-document and original-4K loss?

The matched control is GDN with FLA timescales, sigmoid output gate, and partial RoPE. The treatment
is channel-wise KDA with sigmoid and NoPE. Both start from their exact seed-42 131M-token
checkpoints from finding [16](16_kimi_transfer_131m.md).

## Controlled protocol

- sequence length: 32,768;
- requested continuation: 32,000,000 tokens;
- consumed continuation: 32,047,104 tokens over 489 optimizer steps;
- long-document manifest:
  `0a14833ad84d0f240fd7787e542c47c2f77f40d73427c207cc7ae6b2a95f9da0`;
- mixture: 50% FineMath 4+, 40% peS2o, 10% Wikimedia;
- Muon, learning rate `1e-4`, 25-step warm-up, cosine decay to 10%;
- batch tokens 65,536, Liger loss, and activation checkpointing;
- RoPE control scaling factor 8; KDA NoPE scaling factor 1 because no active rotary dimensions.

The parent model and optimizer states are restored exactly. The two branches differ in their base
architecture and positional regime; they are not same-checkpoint interventions.

## Preflight

| Variant | GFLOP/tok | tok/s | Peak allocated |
| --- | ---: | ---: | ---: |
| GDN/sigmoid/RoPE | 1.6800384 | 19,211 | 4.64 GiB |
| KDA/sigmoid/NoPE | 1.6822042 | 18,347 | 4.84 GiB |

Both passed compiled forward, backward, clipping, and Muon update with the production Liger loss.
An initial GDN preflight accidentally used the benchmark's Torch-loss default. It is preserved as a
labeled diagnostic; a clean Liger rerun replaced it as the canonical measurement.

## Long-document training result

| Variant | Initial 32K loss | Final 32K loss | Change | tok/s | GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDN/sigmoid/RoPE | 2.79745 | **2.615723** | −0.18173 | 19,058 | 0.4620 |
| KDA/sigmoid/NoPE | 2.87334 | 2.627167 | **−0.24617** | 18,291 | 0.4829 |

KDA begins 0.07589 nats behind because the RoPE control receives explicit 8× scaling while KDA
retains its position-free base representation. KDA learns more during continuation and reduces the
gap to 0.011444 nats, recovering about 85% of the initial deficit.

The final held-out split contains only 327,680 tokens, not the configured 20M-token target. The
comparison is paired and every source favors RoPE, but the evidence is much weaker than the
base-stage loss evaluation:

| Source | RoPE | KDA | KDA minus RoPE |
| --- | ---: | ---: | ---: |
| FineMath 4+ | 2.475444 | 2.476877 | +0.001434 |
| peS2o | 2.876604 | 2.896745 | +0.020140 |
| Wikimedia | 2.541879 | 2.557975 | +0.016095 |

## Original-4K retention

The mandatory regression check uses the original packed corpus, batch size 4, and 19,988,480
evaluated tokens.

| Variant | Parent 4K loss | Post-32K 4K loss | Regression |
| --- | ---: | ---: | ---: |
| GDN/sigmoid/RoPE | 2.790629 | 2.805224 | +0.014595 |
| KDA/sigmoid/NoPE | 2.795380 | **2.799689** | **+0.004309** |

KDA retains general short-context capability better and finishes 0.005534 nats below the RoPE
control after continuation. It is better on ten of eleven original validation sources; peS2o is
0.003576 worse. This is a stronger measurement than the small long-document validation split.

## Exact-length counterfactual retrieval

| Length | RoPE direction / score | KDA direction / score |
| ---: | ---: | ---: |
| 4K | 93.3% / 0.1747 | **100% / 2.5674** |
| 8K | 83.3% / 0.0751 | **100% / 2.2404** |
| 16K | 63.3% / 0.0105 | **100% / 1.6182** |
| 32K | 20.0% / −0.0149 | **100% / 0.9559** |
| 64K | 40.0% / −0.0052 | **100% / 0.4742** |
| 128K | 36.7% / −0.0023 | **100% / 0.3180** |

The scaled-RoPE control becomes insensitive by 16K and fails at its trained 32K length. Its
effective directional length is 8K. KDA obtains 30/30 correct directions at every length and keeps
effective directional length 128K.

Continuation strengthens KDA rather than merely preserving its base behavior. Its score rises from
1.7065 to 2.5674 at 4K and from 0.1448 to 0.3180 at 128K. The 128K signal improves 2.20× while the
original-4K language loss regresses only 0.0043 nats.

Exact match remains zero at every length. Candidate accuracy is 26.7% at 4K and 8K, then 23.3%
through 128K. The treatment has learned a strong internal dependency on distant content but not a
reliable answer-emission policy. This distinction remains the largest open evaluation gap.

## Systems result

KDA is 4.0% slower in realized 32K training and 1.9% slower in 128K prefill. Both require exactly
504,860,160 bytes (481.47 MiB) of BF16 resident state at 128K. KDA therefore improves the quality
axis without changing the current five-global-layer state cost; global-layer compression remains
necessary.

## Decision

KDA/sigmoid/NoPE becomes the **lead long-context research architecture**, with
GDN/sigmoid/RoPE retained as the short-loss control. It is not yet a release architecture:

- it misses one of three strict base-stage loss ties;
- the long-document validation set is too small;
- open-vocabulary exact match is still zero;
- independent RULER, NoLiMa, and HELMET evaluations have not run;
- five global layers still cost 481.47 MiB at 128K.

Do not launch 128K continuation yet. The next work is to add a task-appropriate inference/evaluation
adapter, verify real retrieval and aggregation rather than contrastive sensitivity, and then repeat
the KDA/NoPE result with one and two global layers. That experiment attacks the actual frontier:
maintain the replicated signal while reducing 128K KV state from 482 MiB toward 96–192 MiB before
INT8 compression.

## Artifacts

- Experiment contracts:
  [experiments/SpeckLC-150M-KimiContext32K](../experiments/SpeckLC-150M-KimiContext32K)
- Machine-readable summary:
  [results/SpeckLC-150M-KimiContext32K/summary.json](../results/SpeckLC-150M-KimiContext32K/summary.json)
- Short-loss and retrieval records:
  [results/SpeckLC-150M-KimiContext32K](../results/SpeckLC-150M-KimiContext32K)
- W&B runs: `5rztcowr` and `bcpiz02u`
