# 152 — Raw MQA control deconfounds cache representation from FFN capacity

## The v1 attribution problem

V1's MQA1 arm changes five global memories from GQA3 to MQA1 and simultaneously widens all twenty
SwiGLU layers from 2,304 to 2,325. Its near-perfect parameter match is valuable for a whole-architecture
comparison, but the result cannot be attributed to cache representation alone.

## Code-derived raw arm

Changing only the five KV-head counts in memory yields 152,975,898 parameters and 1,015,703,040 analytic
FLOPs/token at 4K: 983,040 parameters and 5,898,240 FLOPs below GQA3, with FFN width fixed at 2,304 and
the same 66.7% cache-state reduction. The matched arm adds back 967,680 parameters and 5,806,080 FLOPs
through 21 extra FFN units per layer.

V2 therefore freezes four arms: GQA3, raw MQA1, parameter-matched MQA1, and the still-unimplemented
NoPE-MLA128 candidate. Raw-minus-GQA estimates the deployable compression effect at fixed FFN;
matched-minus-raw estimates compensation; matched-minus-GQA is explicitly compound. Three shared-control
quality comparisons use Holm correction, and the compensation contrast is a paired two-sided secondary
view.

If raw fails but matched passes, the compound architecture may remain a candidate but MQA alone receives
no preservation claim. If both pass, choose the cheaper non-dominated variant. No training, selection,
or promotion follows from analytic geometry.

## Active boundary

V1 and the active experiment program remain byte-identical during the live finalist. V2 is frozen but
unregistered and training-blocked; it may be registered only in a future clean program successor.

## Artifacts

- [Corrected design](../research/paper-1/sequence_cache_representation_v2.json)
- [Qualification](../results/Speck-Paper1/sequence-cache-representation-v2-qualified.json)
