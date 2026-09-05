# 69 — Scaling and held-out prediction readiness gate

## Two different claims

Scale transfer asks whether a selected candidate's paired effect keeps its sign and passes its gates at
a larger named scale. Scaling efficiency asks whether its fitted loss frontier versus compute and
realized time/energy is better across a declared range. Three-scale sign consistency supports the
architecture thesis but does not establish the stronger fitted-frontier claim.

The current 131,072,000-token baseline is below one token per roughly 154M parameters. It is a proxy
screen, not a compute-optimal scaling point.

## Conditional scale program

Five confirmatory fit targets are frozen at 30M, 60M, 150M, 350M, and 600M active parameters, with at
least three paired seed/data cells per architecture and point. Materialized models must remain within 2%
of nominal size and candidate/control within 0.25% per cell. A 1.2B-active, 20B-token matched sentinel
is held out for prediction and reversal testing, not population equivalence.

Compute-optimal allocation is not assumed. Identical candidate/control pilots cross 30M/60M/150M with
10/20/40 tokens per active parameter. A constrained joint
`L(N,D)=E+A·N^-α+B·D^-β` fit chooses batch-aligned budgets inside that range. Pilot losses are not reused
as the five confirmatory points.

The five points fit the same constrained `L(C)=E_C+A_C·C^-γ` family separately for each architecture
using deterministic multistart nonlinear least squares and paired-variance weights. The report includes
all solutions, raw/standardized/leave-one-out residuals, crossings, and unweighted sensitivity.

## Uncertainty and prediction

A 10,000-resample paired hierarchical bootstrap resamples seed/data cells and refits the complete
allocation and frontier pipeline. Boundary or failed fits remain counted. Before the 1.2B sentinel, its
loss, paired-effect, time/energy intervals, maximum residual, and ranking are frozen. The sentinel passes
only if both losses lie inside 95% predictive intervals, ranking keeps its sign, and both standardized
residuals are at most two. It cannot be folded back into the fit and still be called held out.

At 150M, checkpoints at 2/5/10/20 tokens per active parameter test architecture-by-horizon interaction.
An advantage that disappears or reverses by ten or twenty tokens per parameter blocks extrapolation.

## Decision

No architecture, geometry generator, hardware envelope, allocation pilot, fit, sentinel, scaling claim,
or paper-scale run is authorized. Capability reversal, missing suites, worse realized cost, crossing
frontiers, unstable fits, or a held-out miss narrows or rejects the scaling claim.

## Artifact

- [Scaling readiness gate](../research/paper-1/scaling_readiness_v1.json)
