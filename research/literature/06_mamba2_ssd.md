# Transformers are SSMs: Mamba-2 and Structured State Space Duality

- **Paper:** [arXiv:2405.21060](https://arxiv.org/pdf/2405.21060)
- **Version reviewed:** v1, 31 May 2024; published at ICML 2024
- **Primary topic:** mathematical duality and hardware-efficient linear sequence mixing

## Central claim

Structured State Space Duality (SSD) identifies a class of selective state-space models whose sequence
transformation is also a structured masked-attention matrix. This common representation enables an
algorithm that is recurrent across chunks and matrix-multiplication-heavy within chunks. Mamba-2 is the
neural architecture built around that SSD layer.

The paper's enduring contribution is not merely a faster Mamba block. It provides the mathematical and
systems vocabulary used by later chunkwise linear mixers, including Gated DeltaNet and KDA.

## The dual views

An SSM exposes a recurrent, linear-time view that maintains a fixed-size state. Attention exposes a
quadratic, highly parallel view that materializes interactions between positions. SSD describes sequence
mixing matrices with low semiseparable rank, allowing the same operator to be expressed in both forms.

The SSD algorithm partitions the sequence into blocks. Within a block it uses quadratic matrix
multiplications with high arithmetic intensity; between blocks it propagates a much smaller recurrent
state. With block length fixed, total work remains linear in sequence length. This trades some extra
local arithmetic for substantially better tensor-core utilization than a scan dominated by elementwise
operations.

The simplified SSD transition uses a scalar-identity state transition per head rather than Mamba-1's more
general diagonal transition. State dimension and head dimension are arranged to make the block products
regular. The paper develops MHA-, GQA-, and MQA-like parameter-sharing interpretations for SSD heads,
which later hybrids inherit.

## Mamba-2 block changes

Mamba-2 produces the SSD inputs `A`, `B`, `C`, and `X` with parallel projections rather than generating
some of them deep inside a sequential block. It also changes normalization and head structure. These
macro choices reduce parameter overhead and make tensor/model/context parallelism cleaner; SSD is not
a drop-in kernel replacement for an otherwise identical Mamba-1 network.

## Evidence

- The dedicated SSD implementation is reported as `2–8×` faster than Mamba-1's optimized selective scan
  as state dimension grows, and competitive with FlashAttention-2 over a broad length range.
- Scaling experiments from roughly 125M to 1.3B parameters show Mamba-2 matching or exceeding Mamba-1
  and Transformer baselines on the Pile.
- A 2.7B Mamba-2 model trained on 300B tokens is competitive with a matched Transformer++ baseline.
- At 350M, 48 layers, and 7B tokens, a mixture with roughly 10% attention layers gives the best perplexity
  among the tested SSD/attention counts. A small number of exact-attention layers already helps.
- At 2.7B/300B tokens, the paper compares Transformer++, pure Mamba-2, alternating SSD/MLP, 58 SSD + 6
  attention layers, and a 28 SSD + 32 MLP + 4 attention arrangement. The hybrids improve on the pure
  endpoints under the shared recipe.

The authors report that attention placement is fairly insensitive when layers are spaced out and not
concentrated at the very beginning or end. That result is based on small-scale experiments and should not
override Speck's measured distinction between integration and readout placement.

## What matters for Speck

SSD explains why “linear-time” alone is not a performance specification. State geometry, chunk size,
matrix shapes, and GPU utilization determine whether a recurrence is actually fast. Every new mixer
should therefore have three audited paths: a readable recurrence, a chunkwise train/prefill path, and a
one-token decode path.

The paper also gives a strong prior that sparse exact-attention layers are complementary to finite-state
mixers. Its approximately 10% optimum is an ablation starting point, not a universal rule; Kimi Linear's
25% and Nemotron-H's roughly 8% bracket a wide plausible region.

## Limitations and cautions

- SSD does not generalize softmax attention. It corresponds to attention with a finite feature map and a
  structured mask, so copying and high-load retrieval limitations remain.
- The quoted speedups are kernel-level A100 results and vary with state size, sequence length, dtype, and
  competing kernel maturity.
- Many Mamba-2 gains come from the full block redesign, so “SSD versus scan” and “Mamba-2 versus
  Mamba-1” are different comparisons.
- The main pretraining lengths are short relative to the long-context claims later architectures make.

## Bottom line

Use Mamba-2 as the foundation for reasoning about state geometry and chunkwise kernels, not as the final
memory design. It establishes the efficient computation pattern; GDN and KDA add more capable update
rules, and exact attention remains the high-resolution retrieval complement.
