# Nemotron 3 Nano: Efficient MoE Hybrid Mamba-Transformer for Agentic Reasoning

- **Paper:** [arXiv:2512.20848](https://arxiv.org/pdf/2512.20848)
- **Version reviewed:** v1, 23 December 2025
- **Primary topic:** 30B-A3B hybrid MoE, 1M extension, and selective FP8 quantization

## Central claim

Nemotron 3 Nano is a 31.6B-total/3.2B-active hybrid Mamba-2/GQA MoE trained on 25T text tokens, then
extended and post-trained for agentic reasoning and context lengths up to 1M. It combines six attention
layers with a mostly Mamba-2 backbone and uses selective FP8 quantization to protect sensitive modules.

## Architecture and training

- 52 total modules/layers, model width 2,688, 32 query heads, two KV heads, head dimension 128.
- 64 Mamba heads of dimension 64, state dimension 128, and eight Mamba groups.
- 128 routed experts, six active, plus two shared experts; expert hidden dimension 1,856.
- No positional embeddings, dropout, or linear biases; RMSNorm and untied embeddings/output weights.
- Pretraining uses 23.5T broad tokens followed by a 1.5T high-quality phase. The learning-rate schedule
  warms to `1e-3`, holds for 80% of training, then decays over the final 5T tokens.

The long-context continual-pretraining phase uses 121B tokens. It mixes 512K and 4K sequences after an
all-512K attempt caused a small short-context regression. Document QA and synthetic retrieval data reach
256K and make up stated portions of the blend. Training uses eight-way context parallelism.

## Long-context evidence

The base-model comparison reports RULER `87.50`, `82.92`, and `75.44` at 64K, 128K, and 256K. The
post-trained model reports RULER-100 `92.92` at 256K, `91.25` at 512K, and `86.34` at 1M. These are
strong reported results and show graceful degradation in this model.

**Source correction:** the linked v1 paper does **not** contain the supplied claim that a dense 12B
sibling scores about 85/80/75 and then collapses to 23 at 1M. No 12B dense sibling appears in the
paper text or tables. That “last doubling cliff” should not be cited to arXiv:2512.20848 without a
different primary source.

The paper still reinforces the right operational lesson: report every length, because its own model drops
from `92.92` at 256K to `86.34` at 1M rather than treating the maximum configured window as uniform
capability.

## Selective FP8

Post-training quantization found all six self-attention layers most sensitive. The six Mamba layers
feeding those attention layers are also kept BF16, as are all Mamba Conv1D operations. Most remaining
weights and activations and the KV cache use FP8.

The resulting model reports about 99% median benchmark recovery. FP8 KV enables larger batches and is
a major throughput contributor. In the stated single-H200 8K-input/16K-output comparison, the paper
reports up to `3.3×` throughput versus Qwen3-30B-A3B-Thinking and `2.2×` versus GPT-OSS-20B, using
the better of vLLM and TensorRT-LLM for each model.

## What matters for Speck

- Mixed-length context activation is safer than an all-long phase; always re-evaluate original context.
- Precision should be selected by module sensitivity, not applied uniformly. Global layers and the
  recurrent layers that prepare their inputs deserve explicit protection tests.
- Two KV heads across only six attention layers are a useful minimum-cache reference.
- Long-context post-training data includes targeted retrieval tasks. Architecture alone does not explain
  the 1M curve.

## Limitations and cautions

- The release is a large MoE trained on 25T tokens; its data advantage overwhelms small architectural
  differences in public-model comparisons.
- RULER-100 is synthetic. The paper's AA-LCR result is materially weaker and reminds us that retrieval
  plus reasoning is a different capability.
- The 1M evaluation exceeds the maximum 512K sequence used in the stated long-context pretraining
  mixture, so the final result also depends on extrapolation and post-training.
- Selective FP8 keeps a nontrivial sensitive subset in BF16; describing the whole model as uniformly FP8
  would be inaccurate.

## Bottom line

The most transferable lessons are mixed-length activation, per-module quantization sensitivity, and
full per-length reporting. The paper is positive evidence for a six-global-layer Mamba hybrid, but not the
source of the alleged dense-12B 1M collapse.
