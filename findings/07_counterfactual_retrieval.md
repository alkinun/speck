# 07 — Counterfactual retrieval diagnostic

## Why the metric changed

The evaluator went through three designs, and the failures are part of the result:

1. A random six-digit access code tokenized to seven tokens. All 150M checkpoints had zero exact
   match at 4K. Partial token accuracy existed, but formatting and next-token skill dominated.
2. A common single-token answer still had zero open-vocabulary exact match because these are base
   models, not instruction-following models.
3. Ten single-token candidates (`A` through `J`) produced 7/30 controlled-choice successes for both
   parents and both initially promoted models. The exact same answer symbols succeeded, revealing a
   stable answer-prior artifact rather than retrieval.

Ordinary multiple-choice accuracy was therefore insufficient.

## Paired counterfactual design

For every factual prompt, the evaluator creates a second prompt with identical:

- archive label;
- filler tokens;
- total token length;
- needle placement;
- question wording.

Only the needle's answer changes from candidate `a` to the next candidate `b`.

Let `L_f(x)` be the candidate log probability under the factual prompt and `L_c(x)` under the
counterfactual prompt. The paired score is:

```text
0.5 * [(L_f(a) - L_f(b)) + (L_c(b) - L_c(a))]
```

Metrics:

- **Directional accuracy:** fraction of pairs with a positive paired score.
- **Pair accuracy:** fraction where each prompt individually prefers its own needle answer over the
  alternate.
- **Mean score:** average paired log-probability movement; retains effect magnitude.
- **Short-context significance:** one-sided binomial test of directional successes against 50%.
- **Effective length:** longest tested length retaining 85% of a significant 4K directional
  baseline.
- **Detectable length:** longest tested length whose directional result independently has `p<0.05`.

Each full curve uses 30 pairs per length: 10 cases at depths 0.1, 0.5, and 0.9. Lengths are 4K, 8K,
16K, 32K, 64K, and 128K. Every measured length follows an unmeasured warm-up.

## Sanity checks observed

- `gdn-local` at 4K has strong effects for needles near the end, smaller effects around the 2K
  boundary, and no consistent effect for early needles outside the local window.
- The same local model drops to chance once total length grows and all placements are effectively
  distant.
- Global-attention models show effects across all depths at short lengths and decay gradually with
  length.
- A middle-only global layer improves long-document loss but does not yield significant output-side
  counterfactual retrieval. A final global layer does, supporting the interpretation that layer
  placement controls whether retrieved information reaches the logits.

## Claim boundary

This diagnostic fixes answer priors and detects causal content sensitivity, but it remains a narrow
synthetic passkey task:

- Open-vocabulary exact match is zero for every model.
- Thirty pairs per point give limited resolution near the significance boundary.
- Repeated templates may not predict natural multi-hop or aggregation performance.
- The evaluator is not an independent upstream benchmark.

It is valid for internal regression and architectural comparison. It is not sufficient for a model
card claim or 128K promotion.

## Implementation and artifacts

- Implementation: [speck/long_context.py](../speck/long_context.py)
- Runner: [scripts/long_context_eval.py](../scripts/long_context_eval.py)
- Method commit: `0670ddb`
- Full raw results:
  [results/SpeckLC-150M-GlobalCount32K/retrieval](../results/SpeckLC-150M-GlobalCount32K/retrieval)
