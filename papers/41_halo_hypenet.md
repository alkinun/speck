# Hybrid Linear Attention Done Right: HALO and HypeNet

- **Paper:** [arXiv:2601.22156v1](https://arxiv.org/abs/2601.22156v1)
- **Version reviewed:** v1, 29 January 2026
- **Code:** [official THUNLP repository](https://github.com/thunlp/hybrid-linear-attention)
- **Models:** [HypeNet collection](https://huggingface.co/collections/chen-yingfa/hypenet)
- **Primary topic:** task-guided full-attention layer selection during Transformer-to-hybrid
  distillation, plus a long-context hybrid architecture and positional encoding

## HALO selection method

HALO first instantiates one recurrent counterpart for every attention layer using transferred Q/K/V/O
weights. Stage 1 independently trains each counterpart to match its teacher attention output by hidden-
state MSE. The selector then creates one model per depth in which only that attention layer is replaced
by its trained recurrent counterpart.

For each single replacement, it evaluates recall on SQuAD, FDA, and SWDE and commonsense reasoning
(CSR) on HellaSwag, ARC-Easy, and ARC-Challenge. Its importance score favors a large recall drop and a
small CSR drop, then retains the top quarter of layers as full attention. End-to-end KL distillation and
16K fine-tuning follow selection.

This is an outcome-guided per-layer sensitivity method. It uses 8.36M evaluation tokens for every
single-layer model—234M inference-token presentations for the 28-layer 1.7B teacher—after 320M Stage-1
training tokens. It is cheaper than repeatedly retraining every candidate, but it is neither
training-free nor prospective with respect to the selection tasks.

## Selection evidence

On Qwen3-1.7B conversion, HALO's selected layout reports 55.9 CSR and NIAH scores of
94.9/90.3/94.1/90.6/79.9/74.3 from 8K through 256K. In the same HALO pipeline, Jet-Nemotron and KL-LS
rankings, an even layout, a latter-half even layout, and an RNN-only control are worse overall. The
latter-half recipe is especially poor despite earlier from-scratch evidence favoring middle/later full
attention, emphasizing that conversion-time layer substitutability is not interchangeable with a
from-scratch placement rule.

The comparison does not rerun the full original Jet-Nemotron or KL-LS procedures; it substitutes their
rankings into HALO. It is reported for one converted 1.7B teacher without layout replicates or
uncertainty. The score is based on independent single replacements and does not model interactions
among the several recurrent replacements ultimately selected.

HALO also reports importance rankings for Qwen3-4B and 8B, but the final table's HTML rendering is
ambiguous about highlighted top-k membership. Section 4.3 explicitly states that the final model keeps
25% of layers as attention; the rendered configuration table appears to reverse attention and recurrent
counts, so exact model artifacts must resolve the release identity.

## HypeNet evidence

HypeNet combines hybrid positional encoding (including RoPE/NoPE choices and attention-logit scaling),
QK normalization, conversion of recurrent GQA projections to MHA, output gates, a larger parameter
budget, and Lightning Attention. Conversion from Qwen3 uses 2.3B training tokens across three stages.
At 512K, the authors report up to 3.4x prefill and 3.0x decode speedups over Qwen3-1.7B on one A800,
while Qwen runs out of memory at 1M.

The architecture ablations show large long-context effects from positional encoding and QK
normalization, and smaller/mixed effects from attention/recurrent output gates. Separate 500M models
trained from scratch on 20B tokens compare positional encodings and recurrent mixers; Lightning
Attention gives the best reported length generalization among the tested mixers. These are useful
sequence-design baselines, but the whole HypeNet bundle does not isolate the serving gain of layer
selection itself.

## What matters for Speck

Per-layer recurrent substitutes, local hidden-state alignment, recall-versus-CSR sensitivity scoring,
task-guided top-k attention retention, and its Jet/KL/even/latter-half controls are mandatory N1
baselines. Together with DtR, this occupies inexpensive task-outcome-guided layer selection during
conversion.

The residual Speck question remains narrower only because HALO evaluates every layer on the same tasks
used for selection, uses a pretrained teacher, and does not prospectively rank unseen from-scratch
multi-layer layouts. That distinction is increasingly procedural rather than architectural. Any
retained N1 claim would have to beat HALO on held-out tasks/layouts while explaining interactions and
showing why from-scratch prediction yields scientific or systems value beyond cheap conversion.

HyPE, recurrent QK normalization, GQA-to-MHA conversion, and output gating must also be treated as prior
composition choices if Speck later studies long-context hybrid representation design.

## Bottom line

HALO is a direct, strong placement-selection baseline, though only in conversion. It further reduces
the likely novelty and practical value of N1; independent review should retire N1 unless it can state a
claim that is more than a costlier from-scratch analogue of task-guided sensitivity selection.
