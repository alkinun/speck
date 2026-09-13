# MiniMax-01: Scaling Foundation Models with Lightning Attention

- **Paper:** [arXiv:2501.08313](https://arxiv.org/pdf/2501.08313)
- **Version reviewed:** v1, 14 January 2025
- **Code and weights:** [MiniMax-AI](https://github.com/MiniMax-AI)
- **Primary topic:** 7:1 Lightning/full-attention MoE at million-token scale

## Central claim

MiniMax-Text-01 scales a hybrid of Lightning Attention and full softmax attention to 456B total and
45.9B active parameters. Seven linear-attention blocks are followed by one full-attention block, repeated
through an 80-layer MoE. The model trains to 1M context and passes a 4M single-needle pressure test.

The report is as much a systems paper as an architecture paper: its million-token result depends on
variable-length packing, context parallelism, fused kernels, and communication/computation overlap.

## Architecture

- 80 layers, model width 6,144.
- 64 attention heads of dimension 128.
- One GQA full-attention block after every seven Lightning Attention blocks.
- RoPE on half the head dimensions, base 10,000.
- 32 experts per layer, top-2 routing, expert FFN dimension 9,216.
- DeepNorm/PostNorm-style residual scaling rather than the currently common PreNorm recipe.

Lightning Attention is an I/O-aware implementation of causal linear attention with a blockwise decay.
It computes intra-block interactions explicitly and propagates a compact KV summary between blocks. Pure
Lightning Attention is fast but weak on retrieval; the periodic softmax layer supplies exact global access.

## Controlled evidence

Scaling runs from 70M to 7B and up to 300B training tokens compare softmax, pure Lightning, and the
hybrid. Pure Lightning has similar short-task performance but substantially worse NIAH. The 1B hybrid
scores `95.7` on the paper's weighted NIAH versus `43.6` for hybrid CosFormer2 and `91.8` for hybrid
HGRN2; the 3B hybrid scores `98.0`.

At matched training throughput, hybrid Lightning also beats tested hybrid SWA windows 256–1,024 on the
reported short, NIAH, and SCROLLS aggregates. These are promising small-scale results, later qualified by
MiniMax's M2 experience.

## Long-context recipe

Base training uses length 8,192 and a long constant-learning-rate phase. Context extension proceeds:

| Training length | RoPE base | Tokens | Short/medium/long data |
| --- | ---: | ---: | ---: |
| 128K | 5M | 300B | 30% / 70% / 0% |
| 512K | 10M | 32B | 35% / 35% / 30% |
| 1M | 10M | 26B | 30% / 30% / 40% |

Ten percent high-quality long-context QA is added during the final fifth of each stage. The paper itself
warns that vanilla NIAH saturates early and is inadequate for tracking continued improvement.

## Systems contributions

- **Varlen Ring Attention** applies ring attention to packed sequences without forcing every sample to a
  context-parallel multiple.
- **LASP+** replaces serial rank-to-rank linear-state propagation with local prefix calculations,
  all-gather, and parallel reconstruction. It spends additional communication and temporary memory to
  remove the serial dependency.
- Separate fused kernels cover Lightning prefill and decode. The report claims over 75% end-to-end MFU
  on H20 and notes that at 1M, softmax layers dominate attention latency despite being only one eighth.
- The model-size target was constrained by fitting more than 1M context on eight 80GB GPUs with 8-bit
  weights.

## What matters for Speck

MiniMax-01 establishes that a 7:1 hybrid can be trained and served at million-token scale, but it also
shows that a few global layers dominate asymptotic prefill. Sequence packing and context-parallel state
semantics deserve the same priority as the mixer equation.

Use its extension ladder and hard-evaluation warning. Do not use 4M vanilla NIAH as evidence of a 4M
effective reasoning window.

## Limitations and cautions

- The full model comparison bundles architecture, MoE, 10T+ base tokens, extension data, post-training,
  and custom infrastructure.
- Reported commercial-model parity is sensitive to evaluation date and harness.
- The 4M claim is inference extrapolation on a single-needle test; maximum training length is 1M.
- Periodic full attention means training/prefill remains quadratic in sequence length, with a smaller
  coefficient rather than a new asymptotic bound.

## Bottom line

MiniMax-01 is the “successful large hybrid” systems reference. Read it together with MiniMax-M2: the
former proves feasibility and efficiency, while the latter documents capability failures that small-scale
benchmarks missed.
