# FlashMorph: joint layer selection for hybrid conversion

- **Paper:** [arXiv:2606.30562](https://arxiv.org/pdf/2606.30562)
- **Version reviewed:** v1, 29 June 2026
- **Code:** [LanDisen/FlashMorph](https://github.com/LanDisen/FlashMorph)
- **Primary topic:** budget-constrained joint selection of full-attention layers during
  Transformer-to-hybrid conversion

## Mechanism

FlashMorph starts from a pretrained full-attention Transformer. It first trains a linear-attention
replacement for every attention layer by matching hidden states while the original model remains
frozen. The resulting morphable layer has full and linear branches mixed by one scalar gate.

During placement selection, both branches and the backbone are frozen. Only all layer gates are jointly
optimized. Answer-token hidden states are aligned to the full-attention teacher, while a linearization
regularizer penalizes reliance on full attention. The reported default coefficient is 0.1. Synthetic
passkey retrieval supplies a long-range selection signal. The largest gates are then discretized under a
fixed full-attention budget, followed by logits distillation and long-context finetuning.

This differs from independent layer scoring: every gate is optimized under the current global mixture,
so complementarity and redundancy can affect the selected set.

## Evidence boundary

The paper uses Qwen3-0.6B and 1.7B pretrained backbones with several linear-attention replacements. It
compares placement methods across 6:1, 3:1, and 1:1 linear/full ratios and reports general, recall-heavy,
and RULER results. Its layer-selection stage uses 20M tokens and reports 2.1 GPU-hours at the 1.7B scale,
substantially below the compared search/layerwise methods.

The reported single-GPU 1.7B system uses batch one. Prefill comparisons span 4K–1M and decode grows from
a 1K prompt; long-context speed and memory improve versus the original full-attention Qwen model. These
measure the converted hybrid system, not placement search alone.

## What matters for Speck

Joint budget-constrained hybrid placement is not a novel Speck contribution. FlashMorph is also a
stronger placement baseline than uniform, random, greedy, or isolated layer sensitivity when its code
can be reproduced.

A potentially distinct Speck claim must differ in both question and evidence. A role-grounded law would
need to explain and prospectively predict unseen *from-scratch* layouts, tasks, and scales from causal
integration/refresh/readout diagnostics. FlashMorph instead optimizes gates for conversion using a
pretrained full-attention teacher, trained replacement branches, synthetic retrieval, and recovery
finetuning. That distinction is a hypothesis boundary, not proof of novelty.

## Limitations for transfer

- The method selects layers for conversion, not from-scratch pretraining architecture.
- Teacher alignment and passkey supervision may favor retention of the teacher and retrieval behavior.
- Gate optimization produces a configuration but does not by itself establish a generalizable causal
  placement law.
- Reported runtime compares the resulting hybrid with full attention; it does not isolate every
  placement mechanism or transfer speedups to Speck hardware.

## Bottom line

FlashMorph directly occupies “jointly optimize hybrid attention placement under a budget.” Speck must
either beat it as a baseline with a genuinely predictive and causal law or drop placement novelty.
