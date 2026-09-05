# 68 — Tri-axis interaction and removal readiness gate

## Unit of interaction

The final 2³ cube acts on three frozen axis-level bundles: sequence, depth, and width. A sequence bundle
may contain several retained operators, and the width bundle may contain several stability mechanisms.
The cube can estimate bundle interactions; it cannot attribute those contrasts to KDA, HCA, CSA,
AttnRes normalization, balancing, or another internal component. Those still require their own final
removals.

Each presence arm is already independently qualified. Sequence absence is its strongest matched control;
depth absence is a separately retuned standard-residual geometry rather than a naive residual swap at
the AttnRes optimum; width absence is a parameter/FLOP-view-matched dense SwiGLU.

## Complete cube

The discovery cube contains all eight `S,D,W` cells—`000`, three single-axis cells, three two-axis
cells, and `111`—in three shared seed/data cells, for 24 fixed model runs. Every cell within a pair uses
the same tokens, optimizer, data order, evaluation, and hardware. No cell is selected or abandoned from
interim quality.

Primary analysis uses within-pair main effects, all three difference-in-differences, the three-way
contrast, and each axis's conditional removal from `111`. It reports the equivalent saturated
pair-blocked regression only as a consistency view. Harmful interaction upper bounds must remain within
0.01 aggregate and 0.02 per-source nats. Holm correction covers pairwise and three-way superiority or
antagonism claims; failure to detect interaction is not independence or synergy.

## Removal rule

The final model faces matched removals of each axis bundle and each retained subcomponent. A component
survives only if removing it causes a preregistered quality failure or eliminates a realized systems
benefit clearing its threshold while quality remains non-inferior. If a simpler removal passes quality
and the component does not earn its complexity, the component is deleted.

Discovery interaction evidence has no promotion authority. Retention-changing interactions and final
removals require six paired finalist cells; the selected full model and strongest simpler removal then
need three medium-scale pairs without sign reversal.

## Decision

No axis bundle, cube, or combined architecture is selected or authorized. Once all axes independently
qualify, the next action is to freeze exact presence/absence configuration hashes and the 24-run resource
cost before materialization.

## Artifact

- [Interaction readiness gate](../research/paper-1/interaction_readiness_v1.json)
