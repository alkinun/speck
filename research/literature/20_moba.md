# MoBA: Mixture of Block Attention for Long-Context LLMs

- **Paper:** [arXiv:2502.13189](https://arxiv.org/pdf/2502.13189)
- **Version reviewed:** v1, 18 February 2025
- **Code:** [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA)
- **Primary topic:** MoE-style routing over context blocks

## Central claim

MoBA partitions history into contiguous blocks and lets every query route to its top-k blocks using the
dot product with each block's mean key. The current block is always included with a causal mask. Because
the attention parameters are unchanged, a layer can switch between MoBA and full attention during
training or serving.

MoBA is deliberately less structured than fixed sink/window patterns: the model chooses historical
blocks from content while preserving a guaranteed local path.

## Mechanism and implementation

The block score is `q · mean(K_block)`. Future blocks are masked before top-k. Current-block attention is
computed causally; historical selected blocks are computed without an internal causal mask because all
their tokens precede the query. The two results are combined with online softmax so normalization matches
one attention over the union.

The kernel reorganizes queries by assigned K/V block, applies variable-length FlashAttention, then
restores query order. This is analogous to dispatching tokens to experts. Fine-grained segmentation helps
quality but increases routing and launch overhead.

## Controlled evidence

- Five 568M–2.1B models are trained near compute-optimal token counts at length 8K. With block 512 and
  top-3, MoBA and full-attention fitted losses differ by about `1e-3`.
- At 32K, trailing-token loss is consistently a little worse for MoBA but the fitted gap narrows with
  scale. Trailing loss is more informative than an average dominated by early tokens.
- At fixed 75% sparsity, selecting many fine blocks is about `0.01` loss better than selecting two coarse
  blocks from eight.
- In a 1.5B/30B-token study, training 90% of tokens with MoBA and the last 10% with full attention nearly
  matches full-attention position-wise loss without a switch-time spike.
- During SFT, keeping the last several layers full lowers both average and trailing loss, likely because
  prompt loss masking weakens long-range gradient flow through sparse layers.

## Million-token experiment

The Llama-3.1-8B continuation grows from 128K to 1M. After full-attention extension, MoBA is activated for
100B tokens with block size 4,096 and top-12; the last three of 32 layers remain full attention.

Important caveat: downstream evaluations use MoBA for **prefill only and switch to full attention during
generation**. The reported RULER-128K score is `0.7818` versus `0.7849` for full attention, but this is not
evidence for end-to-end sparse decoding. The 1M result is a standard needle test.

At 1M prefill the paper reports up to `6.5×` attention-layer speedup. A synthetic fixed-sparsity scale-up
to 10M reports `16×` lower attention computation time than FlashAttention. Neither number is an
end-to-end generation speedup.

## What matters for Speck

MoBA is the simplest content-routed sparse global layer in the set. Its ability to switch modes suggests
a practical curriculum: sparse for most long pretraining, a short full-attention consolidation, and a few
full final layers for SFT/readout.

Speck must nevertheless evaluate four modes separately: full/full, MoBA/MoBA, MoBA prefill/full decode,
and a layerwise hybrid. Otherwise quality and speed claims will silently mix execution paths.

## Limitations and cautions

- Mean-key routing can miss a small but critical token whose signal is diluted by its block.
- Routing-score computation still scans block summaries, and top-k/dispatch overhead matters at modest
  lengths.
- The headline real-task comparison switches to full attention for generation.
- Sparse SFT is explicitly weaker unless final full layers are retained.
- The method reduces attention compute but ordinarily keeps all raw K/V available, so it does not by
  itself shrink cache capacity like MLA or eviction/compression.

## Bottom line

MoBA is a good first sparse-attention prototype because it is simple, switchable, and based on existing
MoE/FlashAttention primitives. Its paper also models the honesty Speck needs: trailing-token curves,
full-attention consolidation, and explicit disclosure of sparse-prefill/full-decode evaluation.
