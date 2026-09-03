# Nemotron-H: Accurate and Efficient Hybrid Mamba-Transformer Models

- **Paper:** [arXiv:2504.03624](https://arxiv.org/pdf/2504.03624)
- **Version reviewed:** v4, 5 September 2025
- **Primary topic:** aggressive Mamba-2/attention hybrids, FP8 pretraining, pruning and distillation

## Central claim

Nemotron-H replaces most Transformer attention layers with Mamba-2 while retaining roughly 8% evenly
spaced GQA layers. The 8B and 56B base models report accuracy competitive with similarly sized public
Transformers and up to `3×` higher inference throughput. The report also describes full FP8 pretraining
of the 56B model and a deployment-constrained pruning/distillation pipeline called MiniPuzzle.

## Architecture

The paper counts Mamba-2, attention, and FFN modules as separate layers:

| Model | Total layers | Attention | Mamba-2 | FFN | Hidden size | Mamba state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nemotron-H-8B | 52 | 4 | 24 | 24 | 4,096 | 128 |
| Nemotron-H-56B | 118 | 10 | 54 | 54 | 8,192 | 256 |

Attention layers are evenly distributed, always precede an FFN, and use GQA with eight KV heads. The
first layer is Mamba-2 and the last is an FFN. Mamba-2 keeps head dimension 64, expansion 2, kernel-4
convolution, and eight groups. The models use squared-ReLU FFNs, RMSNorm, no dropout, and **no
positional embeddings**.

The 8B model was trained on 15T tokens and the 56B model on 20T, both at sequence length 8,192. A
matched Nemotron-T-8B Transformer uses the same data and most width/FFN choices, making it the most
useful architecture control in the report.

## FP8 pretraining

The 56B model computes most linear layers in FP8 in both forward and backward passes: E4M3 for weights
and activations, E5M2 for gradients. The first and last four layers remain BF16. Per-tensor dynamic
scaling preserves the largest absolute value and flushes values too small for the format.

The authors warn that small-model FP8 conclusions did not transfer reliably; they validated the recipe at
at least 8B and trillion-token horizons. In their 15T 8B control, downstream accuracy is equal to or better
than BF16 even when likelihood curves are not uniformly better.

## Evidence

- Nemotron-H-56B is reported at `2.4×` the per-GPU output throughput of Qwen2.5-72B and Llama-3.1-70B
  in the stated H100 setup; Nemotron-H-8B reaches `1.8×` and `3×` versus Qwen2.5-7B and Llama-3.1-8B
  at long input.
- The 8B hybrid beats its same-data Transformer control on 7 of 15 reported tasks. This is stronger
  evidence than comparisons to differently trained public models, but it is not an all-task win.
- MiniPuzzle ranks depth and FFN-neuron removals with forward-only sensitivity, filters architectures by a
  1M-context/32 GiB FP4 target, briefly distills three candidates for 7B tokens, then distills the winner
  for 63B tokens.
- The final 47B model keeps 5 attention, 44 Mamba-2, and 49 FFN layers and narrows the FFN from 32,768
  to 30,720. It reports near-parent quality and `1.2×` faster long-context inference.

## What matters for Speck

Nemotron-H provides credible evidence that attention fractions near 8% can work, especially when layers
are distributed throughout depth. It also suggests architecture and quantization sensitivity interact:
global attention and adjacent Mamba layers can require higher precision even when most of the network is
FP8.

For a small Speck model, copy the experimental questions rather than the scale:

- compare 5%, 10%, and 25% global layers with fixed parameter/FLOP budgets;
- preserve an evenly spaced arm but include measured middle/final placement arms;
- quantify whether NoPE works from scratch rather than through late conversion;
- run selective precision sweeps by module, including the mixer feeding global attention.

## Limitations and cautions

- Public-model accuracy comparisons are not matched for tokens, data, tokenizer, or post-training.
- Even the matched 8B result wins only a subset of tasks; mean scores can hide retrieval regressions.
- The base models train at 8K. Later long-context behavior depends on additional alignment and extension
  data, not on the hybrid architecture alone.
- MiniPuzzle is deployment-targeted compression, not evidence that particular low-importance attention
  layers would have been unnecessary during pretraining.

## Bottom line

Nemotron-H validates an aggressive global-attention budget and gives unusually actionable FP8 and
compression details. It strengthens the case for a low-attention-ratio Speck arm, provided retrieval,
precision sensitivity, and placement are evaluated directly.
