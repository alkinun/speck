# Systematic hybrid linear-attention ratio and mixer study

- **Paper:** [arXiv:2507.06457v2](https://arxiv.org/abs/2507.06457v2)
- **Version reviewed:** v2, 24 June 2026
- **Models/materials:** [M-A-P hybrid-linear-attention collection](https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research-686c488a63d609d2f20e2b1e)
- **Primary topic:** controlled comparison of recurrent/linear mixers and uniform full-attention ratios
  in from-scratch hybrids

## Experimental matrix

The study trains 72 models: 36 around 340M parameters on 20B FineWeb-Edu tokens and 36 around 1.3B on
100B tokens. It compares six mixer families across uniform linear-to-full ratios 24:1, 12:1, 6:1, 3:1,
and pure linear, plus a full Transformer. Evaluation separates general zero-shot tasks from RULER recall.
Contexts are limited to 4,096 tokens.

The mixer families span vector recurrence, fixed/data-dependent/hierarchical outer-product state, and
delta-rule controlled forgetting. The paper's central warning is that standalone linear-model quality
does not predict hybrid ranking: interactions with sparse full-attention layers can reorder mixers.

## Results and evidence limits

General-language averages are nearly flat across ratios, whereas RULER improves as full attention becomes
denser. Averaged recall rises from 0.256 for pure linear to 0.390/0.397 at 6:1/3:1; the full Transformer
is about 0.423. Gated DeltaNet peaks around 0.436 at 3:1 and HGRN-2 around 0.434 at 6:1 in the reported
table. The practical recommendation is a gated hierarchical/controlled-forgetting mixer at 3:1–6:1.

The paper attributes strong hybrids to selective gating, hierarchical recurrence, and controlled
forgetting, but explicitly calls this comparison observational and leaves component ablations to future
work. Full layers are uniformly interleaved, so the study does not compare non-uniform placements or fit
a placement law. The 340M models are considered too weak for meaningful recall analysis; transfer beyond
1.3B, 4K, base pretraining, and English remains open.

## What matters for Speck

Mixer selection and full-attention count must be crossed: a mixer that wins alone cannot be assumed to
win at the chosen ratio. Short-context loss cannot select attention density because recall moves much
more strongly. HGRN-2/GatedDeltaNet and the 3:1–6:1 region are mandatory baselines, not constants to copy.

For N1, this paper fixes the ratio/mixer baseline but leaves placement open. Any prospective placement
law must hold mixer and count fixed, compare non-uniform layouts against uniform interleaving, and show
held-out gains beyond the paper's observed mixer-by-ratio interactions. Claims about gating,
hierarchical recurrence, or controlled forgetting require causal removals because this source does not
provide them.

## Bottom line

The broad recipe “gated recurrent mixer plus occasional full attention” and its 3:1–6:1 recall knee are
not novel. N1 can address non-uniform role placement only after conditioning on this strong, large-token
from-scratch ratio/mixer baseline.
