# 100 — Cross-task, cross-mixer KL-guided placement overlap

## Direct evidence

KL-guided layer selection starts from an all-linear distilled model, restores each teacher softmax
layer independently, redistills the one-restoration neighbor, and ranks its held-out generic-text KL.
The selected non-uniform hybrids are evaluated on downstream recall tasks not used in the score.

GA-S2 beats uniform and a broad heuristic/MSE/KL control matrix across fixed budgets for Qwen and
Llama teachers. It transfers within Qwen from 1.5B to 7B. More importantly, GDN-selected layers produce
strong GLA students and can outperform GLA's own selected set, occupying cross-mixer diagnostic
transfer. Rankings cluster by teacher; forcing more even spacing harms RULER. A rolling-Jaccard/backbone
rule reports 58--74% lower selection tokens.

## Residual correction

Held-out-task generalization, cross-mixer transfer, cross-scale selection, non-uniform clustering,
spacing intervention, and ranking stability are no longer distinct. The only formal N1 difference is
teacher-free prediction of interacting from-scratch layouts without training every neighbor.

That difference is procedural, not a mechanism. Current evidence favors retirement unless independent
review finds a causal architecture law with unique predictions and realistic evidence cost. No
placement protocol, artifact execution, or training is authorized.

## Artifacts

- [KL-guided selection note](../papers/42_kl_guided_layer_selection.md)
- [Novelty landscape v9](../research/paper-1/novelty_landscape_v9.json)
