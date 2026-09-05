# 101 — KL-guided selector source/config audit

## Available evidence

The official repository is pinned at `d7467e37d359ac873242d2902d23a9fd47de746b`. Its 203 files
include 24 Python files and 171 YAML configs spanning teachers, scales, GDN/GLA, budgets, heuristic
selectors, and per-layer sweeps. Conversion, distillation, one-restoration config generation, W&B
ranking, synthetic ablations, and final selected-layer configs are present.

## Blocking differences

There is no root license or test suite. More importantly, the paper defines importance as expected KL
on a held-out slice, while the released workflow ranks sampled W&B training loss. The published rolling-
Jaccard/backbone early stop is not implemented in the inspected ranking script. Paths and W&B state
need manual edits; FLA/data/evaluation dependencies, teacher/tokenizer/dataset revisions, DCLM order,
checkpoints, loss exports, and result tables are not pinned. Synthetic generators use unseeded Python
random calls. Nothing was checked out, imported, or executed.

## Decision

Source and config identities are qualified, but root rights, paper/code semantics, stopping behavior,
environment, data, logs/checkpoints, deterministic evaluation, reuse, and reproduction are blocked.
The selector remains a mandatory conceptual baseline with no execution or N1 protocol authority.

## Artifact

- [KL-selection code audit v1](../research/paper-1/kl_selection_code_audit_v1.json)
