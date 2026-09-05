# Budgeted Attention Allocation

- **Paper:** [arXiv:2605.05697](https://arxiv.org/pdf/2605.05697)
- **Version reviewed:** v1, 7 May 2026
- **Primary topic:** one checkpoint with monotone attention-head usage controlled by a requested budget

## Controller and training

Each attention head receives a sigmoid gate whose logit is an intercept plus a nonnegative softplus
sensitivity times the requested budget's logit. Nonnegative sensitivity makes head use monotone in the
external budget. Training samples budgets, penalizes mean gate mass and budget overshoot, and mostly
uses soft gates.

A hard operating point keeps the globally largest requested number of gates across layers. Structural
materialization requires at least one active head per layer. Because hard top-k is not differentiable,
the paper also uses one epoch of straight-through hard-gate adaptation with distillation from the soft
checkpoint.

Dense warm-starting is important: one from-scratch synthetic seed collapses, while dense initialization
stabilizes the budgeted model. Fixed-budget learned gates, post-hoc pruning, and recovered pruned models
are strong baselines and sometimes match or beat the multi-budget checkpoint at one operating point.

## Evidence boundary

The study covers a synthetic marked-token task and sampled AG News/DBpedia14 classification with small
custom Transformers, BERT-Tiny, and BERT-Mini. Main comparisons use three seeds. Soft gate mass is only
estimated cost and does not reduce latency. Hard structural execution produces measured single-thread CPU
speedups, but the implementation is naive and GPU acceleration is not automatic without fused kernels.

The contribution is controllability across many budgets from one checkpoint, not universal dominance
over one specialist per fixed budget.

## What matters for Speck

An externally requested attention-cost knob and monotone layer/head gates are not novel. Any dynamic
Speck budget claim must compare against this controller, fixed-budget specialists, recovered structural
pruning, and the cost of storing/deploying multiple specialist artifacts.

Speck's current architecture program selects one fixed evidence-backed model rather than promising one
checkpoint at many price points. Adding budget conditioning would be a separate conditional-compute axis,
not evidence for the fixed architecture.

## Transfer cautions

- Estimated gate mass is not wall-clock cost.
- Hard execution changes optimization and needs adaptation.
- Small encoder classification does not establish autoregressive LLM quality or long-context serving.
- Eager GPU execution does not automatically exploit inactive heads.

## Bottom line

Cost-conditioned attention is direct prior art. Speck should not add it merely as a novelty feature; a
future use needs autoregressive quality, fused execution, and comparison to per-budget specialists.
