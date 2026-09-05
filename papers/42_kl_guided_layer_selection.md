# Distilling to Hybrid Attention Models via KL-Guided Layer Selection

- **Paper:** [arXiv:2512.20569v1](https://arxiv.org/abs/2512.20569v1)
- **Version reviewed:** v1, 23 December 2025
- **Code:** [official FLA repository](https://github.com/fla-org/hybrid-distillation)
- **Primary topic:** generic-text KL marginal utility for selecting non-uniform softmax layers during
  Transformer-to-linear-attention distillation

## Method

The method first converts a softmax Transformer into an all-linear student using RADLADS-style
distillation: 100M tokens of per-layer hidden-state alignment followed by 600M tokens of end-to-end
teacher/student KL matching. For every depth, it restores exactly that teacher softmax block into the
otherwise all-linear student, briefly redistills, and uses held-out generic-text KL as the layer's
marginal-utility score. The top K restored layers form the final hybrid, which is distilled again.

The primary GA-S2 variant measures utility relative to the all-linear base using Stage-2 KL, then takes
the top K. Although called greedy addition, the published algorithm ranks independent one-restoration
neighbors rather than sequentially recomputing marginal utilities after every chosen layer. Ablations
cover greedy removal, averaged rankings, and Stage-1 MSE equivalents.

This is far more compute-intensive than a static probe: the full selector uses the 700M-token
distillation pair for every layer. A rolling-Jaccard/backbone stopping rule finds nearly final top-K sets
after 27--42% of Stage 2 in the two main teachers, suggesting 58--74% lower selection tokens.

## Evidence

The main experiments convert Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct to GDN hybrids across 12.5%,
25%, 33%, and 50% softmax budgets. Controls include uniform placement; attention-entropy, multi-head,
value-trace, channel-wise entropy, activation-MSE, and LM-perplexity heuristics; SMART/PostNAS; and the
Stage-1/Stage-2 addition/removal/average variants.

At 25% softmax, GA-S2 reports RULER 0.8631 for Qwen and 0.7539 for Llama, versus 0.6904/0.4610 for
uniform and 0.6401/0.6274 for SMART. Stage-2 KL strongly beats Stage-1 MSE, and addition from an
all-linear base beats removal from an all-softmax base. At Qwen's 12.5% budget, GA-S2 reaches 0.662
versus 0.542 for the strongest control and 0.441 for uniform.

The result extends within Qwen to 1.5B and 7B teachers at 25% and 33% budgets. It also tests mixer
transfer: GDN-selected layers produce GLA students with RULER 0.6927/0.8407 for Llama/Qwen, higher than
GLA-selected GLA students at 0.6498/0.6921. This directly shows that one recurrent mechanism can serve
as a placement probe for another, although only within teacher conversion.

Selected layers are non-uniform and often clustered. Qwen repeatedly selects groups around depths
3--5, 19--22, and 31--33; Llama has a main middle group around 12--18. Explicitly penalizing nearby
selections lowers Qwen RULER from 0.8713 to 0.8509 or worse, showing that forced even scattering removes
useful local groups in this setting.

Selection uses generic DCLM KL, while RULER, SWDE, FDA, SQuADv2, and 8K--128K needle retrieval assess
downstream recall. Thus the method has more held-out-task evidence than HALO's task-outcome selector.
The 25% Qwen hybrid retains nearly perfect needle retrieval through 65K, then falls to 0.684 at 131K
versus the teacher's 0.954.

## Evidence limits

This remains conversion from an instruction-tuned teacher, not from-scratch architecture prediction.
Every layer's score requires training a one-restoration hybrid, so it observes proxy outcomes for all
candidate depths. The top-K combination assumes independent marginal utilities and is not a prospective
predictor of arbitrary interacting layouts.

Results are not reported with seeds, placement-level confidence intervals, or a frozen cross-family
prediction test. GDN-to-GLA transfer is encouraging but asymmetric: the GDN probe unexpectedly beats the
specialized GLA probe, so “architecture consistency” is not a stable law. The long-context extension is
single-needle retrieval for one model and does not establish complex long-document reasoning.

## What matters for Speck

Generic-text KL marginal utility, restore-one-layer neighborhoods, all-linear addition, all-softmax
removal, MSE/KL comparisons, cross-mixer probe transfer, rolling-set early stopping, clustered-layout
analysis, and distance-regularized scattering are all mandatory N1 baselines. Crucially, KL selection
already uses one mechanism to find layers that transfer to another and evaluates on task families not
used for scoring.

After this source, the only formal Speck distinction is prediction for unseen jointly trained
*from-scratch* layouts without training every one-layer neighbor. The held-out-task and cross-mixer parts
are no longer distinct by themselves. That remaining distinction is procedural and likely too weak to
justify an architecture paper unless independent review identifies a genuinely causal, compute-saving
law that conversion evidence cannot answer.

## Bottom line

KL-guided selection is the strongest direct overlap yet for a transferable placement diagnostic. It
does not solve arbitrary from-scratch layout prediction, but it removes most of the residual's claimed
generality and makes N1 retirement the evidence-favored outcome pending independent review.
