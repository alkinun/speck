# 158 — Four Stable LatentMoE equation references qualify on CPU

## Result

Four isolated float64 CPU references now qualify without entering Speck's model or training paths:

- SiTU-GLU matches the direct Eq. 12 computation within `1.42e-14`, respects the source coordinate
  bound, has finite gradients in 35 grid/random cases, agrees with SwiGLU near zero to `1.71e-21`,
  and approaches it at large beta.
- Normalized LatentMoE matches an independently assembled Eq. 11 computation exactly in 3,072
  seed/shape cases spanning 96 geometries. All 26,112 inspected input, projection, expert, and norm
  gradient tensors are present and finite.
- Exact Quantile Balancing reproduces raw-score-only mixture weights, biased top-(k+1) cutoffs, the
  `(q+1)` required-bias order statistic, mean centering, and common-offset invariance in 128 cases.
- Histogram QB passes 384 cases, including 128 at the source's 1,000 bins. All 768 partition tests
  reproduce pooled integer counts exactly; maximum distance to the valid empirical quantile interval
  is `0.7981` bin widths. A fixture confirms that pooled recovery is not an average of shard quantiles.

## Tie finding

The finite-batch no-ties assumption is not generally available. Because a cutoff expert has required
bias equal to its current bias, repeated `(k+1)` cutoff identities can create exact boundary ties even
with continuous random router scores. Across the 128 cases, 204 expert thresholds had boundary ties
and only 10 complete cases were tie-free. The source-consistent rank bracket—strictly-below count at
most `q`, below-or-equal count above `q`—had zero failures. A future training design must freeze its
tie behavior rather than claiming exact per-expert load from the no-ties derivation.

## Boundary

These are dense correctness oracles, not kernels or a trainable MoE. They do not qualify conventional
MoE dispatch, primitive composition, model integration, distributed reduction, hardware efficiency,
training, or architecture promotion. The active finalist and every frozen source remain untouched.

## Artifacts

- [Frozen CPU protocol](../research/paper-1/stable_latentmoe_cpu_reference_v1.json)
- [Qualification](../results/Speck-Paper1/stable-latentmoe-cpu-reference-qualified-v1.json)
- [Reference implementation](../speck/stable_latentmoe_reference.py)
