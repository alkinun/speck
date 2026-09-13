# Rethinking efficient-attention roles in hybrid architectures

- **Paper:** [arXiv:2606.15378v1](https://arxiv.org/abs/2606.15378v1)
- **Version reviewed:** v1, 13 June 2026
- **Code/checkpoints:** [thunlp/rethinking-hybrid-attention](https://github.com/thunlp/rethinking-hybrid-attention)
- **Primary topic:** scaling and causal/mechanistic analysis of how efficient mixers affect full-
  attention retrieval learning in from-scratch hybrids

## Design and scale

The study compares a full Transformer with 1:1 layer-wise hybrids using SWA windows 128/512/2,048,
Lightning Attention, Mamba-2, or Gated DeltaNet. Five model sizes range from 15M to 477M non-embedding
parameters (71M–665M total). S1–S4 are measured at six complete schedules from 100 to 1,000 tokens per
non-embedding parameter; S5 supplies larger-scale extrapolation checks. Training uses 16K contexts and
a 1:1 long/short data mixture.

Separate power laws are fit for held-out C4 loss and log LongPPL, with S4 held out from S1–S3 fitting.
Short-context loss is nearly insensitive to efficient-mixer choice. LongPPL differs strongly in the
low-data regime but converges with more training, suggesting that efficient attention changes capability
emergence speed more than the final level when full-attention structure is fixed.

## Mechanistic evidence

Inference-time receptive-field interventions restrict either efficient or full attention to about 2K.
Restricting full attention sharply hurts LongPPL; restricting efficient attention has minor effect,
including for recurrent mixers with nominally unbounded history. Layer-wise NIAH probes place nearly all
incremental long-range accuracy in middle full-attention layers, while middle efficient layers add little
or sometimes reduce probe accuracy.

The paper interprets efficient attention as an optimization prior for full attention. Natural-data
gradient influence decays toward baseline beyond about 2K; a large SWA window can absorb the nearer
dependency signal and delay pressure on full-attention retrieval heads. Retrieval-head entropy, Q/K
distance, and gradient traces support this “Large-Window Laziness” account, especially for SWA-2,048.

The evidence is stronger than correlation alone but not a complete placement law. The receptive-field
intervention tests operator roles after training; probe heads are chosen from final checkpoints; gradient
influence uses Llama-3.1-8B as a model-agnostic proxy; all main hybrids use uniform 1:1 alternation.

## Design consequences

A 1:3 full-to-efficient SWA-128 variant converges more slowly at small scale but closes the gap as scale
grows. Head-wise SWA/full mixing does not outperform layer-wise mixing and learns LongPPL more slowly.
Applying NoPE only to full-attention layers materially improves long-context results while leaving short
tasks similar. At 665M total and about 100B tokens, SWA-128-NoPE reaches 52.88 RULER versus 46.13 for
SWA-128 and 47.17 for Full at 16K; after a 5B-token 32K extension it remains strongest on reported RULER
and LongBench aggregates.

## What matters for Speck

The following cannot be claimed as new: full attention as the main long-range carrier, middle full-layer
integration, efficient attention as an optimization prior, receptive-field role interventions,
layer-wise long-range probes, retrieval-head training dynamics, small-window pressure on global
retrieval, scale-dependent attention density, or NoPE strengthening the full path.

N1 must now predict *which non-uniform from-scratch placements* succeed beyond this study's uniform
alternation and diagnostics. It must compare against LongPPL scaling, receptive-field restriction,
middle-layer probe gains, retrieval-head entropy/QK/gradient traces, and small-SWA/NoPE controls. If its
proposed integration/refresh/readout variables reduce to “middle full attention carries retrieval” or do
not improve unseen-placement prediction, N1 is overlapped or falsified.

## Bottom line

This is the strongest direct N1 overlap so far. A prospective non-uniform placement law may remain, but
the role interpretation and its main diagnostics are occupied; N1 cannot stay primary without a new
versioned landscape and substantially narrower claim.
