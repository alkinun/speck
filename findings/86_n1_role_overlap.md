# 86 — N1 role-overlap audit and scope reduction

## New direct evidence

Three full-text sources materially narrow the last primary novelty hypothesis.

Rethinking the Role of Efficient Attention trains five scales and six 1:1 hybrid families from scratch.
Receptive-field interventions and layer probes show that middle full-attention layers carry most
long-range information, while efficient mixers mainly shape how quickly full-attention retrieval heads
learn. Retrieval-head entropy, Q/K distance, gradient traces, and natural dependency distance explain
Large-Window Laziness; small SWA plus NoPE full layers is a derived design.

A 72-model systematic study crosses six recurrent/linear mixers with five uniform ratios at 340M/20B
and 1.3B/100B. Language quality is nearly flat while recall moves strongly with attention density;
standalone mixer ranks do not predict hybrid ranks. GatedDeltaNet/HGRN-2 around 3:1–6:1 are direct
baselines, though its gating/hierarchy/forgetting interpretation remains observational.

Distill-then-Replace constructs task-specific non-uniform hybrids through interaction-aware greedy
replacement. It beats uniform, random, and local sensitivity. A hidden-state probe correlates with
replacement order but loses 2.6–3.8 points to greedy feedback at the same layer count, directly showing
that static probe importance does not capture interactions.

## Scope correction

The broad `N1_role_grounded_placement_law` is rejected as novelty. Full attention as long-range carrier,
middle-layer integration, efficient attention as optimization prior, retrieval-head learning dynamics,
small-window pressure, full-layer NoPE, mixer-by-ratio tradeoffs, greedy non-uniform replacement, and
probe-based substitutability are occupied.

Only `N1_prospective_nonuniform_from_scratch_placement_law` remains: a predictor fit on discovery
layouts must rank and calibrate unseen non-uniform from-scratch layouts across tasks, scales, and a new
mixer without observing held-out layout outcomes. It must beat count/ratio, middle-role/LongPPL,
retrieval-head probes, sensitivity, DtR greedy conversion, FlashMorph gates, role layouts, and matched
random placements, then pass fixed-budget role interventions.

## Decision

This residual is an unestablished empirical architecture law, not a mechanism or selected architecture.
There are currently zero established Speck architecture-novelty candidates. No placement experiment is
authorized until code/source identity, remaining citation/patent/deployed-system review, and independent
claim-level expert review find a defensible distinction.

## Artifacts

- [Hybrid-role mechanism note](../papers/36_rethinking_hybrid_attention.md)
- [Systematic ratio/mixer note](../papers/37_systematic_hybrid_linear.md)
- [DtR greedy placement note](../papers/38_distill_then_replace.md)
- [Novelty landscape v5](../research/paper-1/novelty_landscape_v5.json)
