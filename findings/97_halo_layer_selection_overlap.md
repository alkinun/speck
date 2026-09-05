# 97 — HALO task-guided layer-selection overlap

## Direct overlap

HALO transfers Q/K/V/O weights into a recurrent counterpart for every attention layer, independently
aligns each substitute to teacher hidden states, and evaluates every single replacement on recall and
commonsense tasks. An explicit ratio score ranks layers, and the top quarter remain full attention.

Within the same 1.7B conversion pipeline, HALO's layout beats even, latter-half even, RNN-only,
Jet-Nemotron, and KL-guided rankings on the reported CSR/NIAH matrix. It uses 320M Stage-1 training
tokens and 234M layer-selection inference-token presentations before joint distillation. Non-uniform,
task-outcome-guided attention retention is therefore strong prior art.

## Residual assessment

HALO does not predict arbitrary jointly trained from-scratch layouts: it uses a teacher, observes each
layer's selection-task outcomes, and scores one replacement at a time. That leaves a formal distinction
for held-out, from-scratch, interaction-aware prediction, but the difference is procedural. No evidence
shows that the costlier residual produces a transferable law or an improved architecture beyond cheap
conversion methods.

## Decision

Task-guided selection, hidden-state-aligned substitutes, and HALO's control family are mandatory N1
baselines. The residual remains low-prior and requires independent retention-or-retirement review. No
new artifact execution, protocol, placement experiment, or training is authorized.

## Artifacts

- [HALO paper note](../papers/41_halo_hypenet.md)
- [Novelty landscape v8](../research/paper-1/novelty_landscape_v8.json)
