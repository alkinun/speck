# 153 — Raw ratio family deconfounds recurrent/global count from FFN matching

## The matching alias

V1 correctly fixed integer-representable 1:1, 3:1, and 9:1 layouts, but proposed uniform FFN width
adjustment to match parameters. That would estimate ratio plus capacity adjustment.

At fixed FFN2304, existing model code gives raw 1:1/3:1/9:1 geometries of 155,024,828 / 153,958,938 /
153,319,404 parameters and 1,121,172,480 / 1,021,601,280 / 961,858,560 FLOPs/token at 4K. Their global
cache state is 7,680 / 3,840 / 1,536 BF16 bytes/token. These are the total deployable count effects.

Matching back to 3:1 requires FFN2281 for 1:1 and FFN2318 for 9:1. Those compound arms land within
6,050 and 5,586 parameters of control, but have no selection, placement-count, or pure-ratio authority.

V2 runs all three raw arms plus both sensitivities over three discovery cells (15 runs). Only the raw
family receives six-cell confirmation (18 runs), with Holm across the two candidates versus 3:1. A raw
failure cannot be rescued by a matched sensitivity without freezing a new compound design. Placement
still waits for a selected raw count and independent cells.

## Active boundary

V1 and the active experiment program remain byte-identical. V2 is frozen but unregistered and training-
blocked until the active finalist and all operator prerequisites finish.

## Artifacts

- [Corrected ratio design](../research/paper-1/ratio_placement_readiness_v2.json)
- [Qualification](../results/Speck-Paper1/ratio-placement-v2-qualified.json)
