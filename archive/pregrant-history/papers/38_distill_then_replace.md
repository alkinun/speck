# Distill-then-Replace: validation-driven hybrid layer placement

- **Paper:** [arXiv:2601.11667v2](https://arxiv.org/abs/2601.11667v2)
- **Version reviewed:** v2, 2 June 2026
- **Code:** no official implementation declared on the arXiv record
- **Primary topic:** task-specific conversion of pretrained full-attention models by blockwise
  distillation and greedy linear-layer replacement

## Method

DtR first trains one linear-attention counterpart for every full-attention block independently, matching
the original block output under the same hidden-state inputs. Starting from the full model, it then tries
each remaining one-layer replacement, evaluates the downstream validation set, commits the best feasible
replacement, and repeats until performance falls below a fixed threshold.

It returns both the highest-validation hybrid and the most-linear hybrid satisfying the threshold. The
search uses forward passes after local distillation, allowing layer interactions to change after every
accepted replacement. This is conversion from a task-capable pretrained teacher, not from-scratch
architecture selection.

## Evidence

The paper covers three pretrained base-model sizes, three linear counterparts, and eighteen downstream
tasks. Across 162 reported settings, 132 (81.5%) match or exceed the full-attention baseline, and every
task has at least one successful configuration. Replacement order is similar across linear mechanisms
but varies with base model and task.

Uniform interleaving, random replacement, and one-shot local sensitivity are direct baselines; greedy
validation performs better. A post-hoc probe impact score compares task-label information/alignment in
full and distilled linear hidden states and correlates with replacement order. Yet using the static score
to choose ten layers loses 2.6–3.8 accuracy points versus greedy on three tasks, showing that the probe
does not capture interaction-aware placement prospectively.

On one A800, the authors report 5/7/15 total GPU-hours for 1.5B/3B/8B bases, split between blockwise
distillation on about 100M general tokens and greedy search on 500 PubMedQA validation examples. These
are task-specific conversion/search costs, not from-scratch pretraining or named serving-system gains.

## What matters for Speck

Greedy interaction-aware replacement, local sensitivity, random/uniform placement, and hidden-state
probe substitutability are mandatory N1 baselines. “Predict important full-attention layers with a probe”
is not enough: DtR already tries it and shows static scores lose to validation feedback.

N1 remains distinguishable only if a law learned on discovery layouts predicts unseen *from-scratch*
layouts/tasks/scales without evaluating each held-out layout, then survives fixed-budget causal role
interventions. It must beat or explain DtR-style greedy placement where conversion is feasible, while
being honest that task-specific teacher conversion solves a different deployment problem.

## Bottom line

Automated greedy placement and probe-based layer substitutability are prior art. The from-scratch,
cross-task prospective aspect of N1 may remain, but its baseline and falsification burden are much
stronger than the current claim table records.
