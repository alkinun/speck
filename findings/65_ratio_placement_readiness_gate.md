# 65 — Recurrent/global ratio and placement readiness gate

## Specification correction

The earlier future grid named 1:1, 3:1, and 7:1 ratios in a 20-layer model. An exact 7:1 ratio would
require 2.5 global layers and is impossible. Before any ratio result exists, the grid is corrected to
integer-realizable 1:1, 3:1, and 9:1 arms: 10, 5, and 2 global layers with 10, 15, and 18 recurrent
layers. Counts and realized ratios must always be reported; nominal labels may not hide rounding.

## Ratio isolation

Count is tested before placement. Each ratio uses the same deterministic quantile rule,
`floor((i+1)×20/n)-1`, producing global indices:

- 1:1 — `[1,3,5,7,9,11,13,15,17,19]`;
- 3:1 — `[3,7,11,15,19]`; and
- 9:1 — `[9,19]`.

The selected KDA, exact-cache, HCA, CSA, raw-local, residual, data, and evaluation definitions remain
fixed. Uniform FFN width matches active parameters within 0.025% where possible; exact FLOPs and a
batch-aligned fixed-FLOP view remain visible instead of forcing two matches through hidden changes.
The lowest-cost arm passing every quality constraint wins only at discovery scope.

## Placement successor

Integration-heavy and readout-heavy layouts are not yet assigned indices. Their exact geometry depends
on the selected integer count; comparing a two-slot middle/final arm with five-slot layouts would
reintroduce the ratio confound. A successor contract must hold count fixed and include uniform,
integration-heavy with final readout, and readout-heavy with at least one middle integration slot.

Mechanistic evidence separates middle integration, final output access, refresh gaps, KDA overwrite,
attention/source recall, route/payload composition, and depth-wise logit contribution. Selection is
based on realized state, time-to-quality, prefill, decode, and arrival-rate frontiers under all quality
constraints—not on a published 3:1 ratio.

## Decision

No ratio or placement is selected. Implementation, training, and promotion remain blocked until every
sequence operator qualifies. The next eventual action is geometry-only materialization and correctness/
cost preflight for the three exact integer ratios.

## Artifact

- [Ratio/placement readiness gate](../research/paper-1/ratio_placement_readiness_v1.json)
