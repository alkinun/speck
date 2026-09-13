# 114 — Finalist crossed-factor inference correction

## Flaw caught before results

The six finalist cells cross three initialization seeds with two data orders. V1 proposed a pooled
df=5 t-bound, but cells sharing a seed reuse initialization and cells sharing an order reuse the data
stream. Treating all six as independent would understate dependence. Standard two-way cluster methods
are asymptotic and indefensible with only three and two clusters; one observation per crossed cell also
cannot separate interaction from residual error without assumptions.

## V2 primary rule

The two data orders are now fixed robustness strata. Within each order, the three independently seeded
candidate-minus-control differences receive a one-sided 95% df=2 t-bound. Both strata must pass the
+0.01 margin, and every individual cell must also be at or below +0.01. Every source repeats the same
two-stratum +0.02 test and hard cell guard. This is an intersection-union screen: no failure can be
averaged away.

The pooled six-cell mean/range/naive df=5 bound and each seed's two-order mean are descriptive only.
The conclusion is explicitly limited to both frozen data orders; it does not infer to arbitrary orders.
Secondary FLOP/time/target views become order-stratified and retain censoring.

## Decision

V1 analysis is superseded before any finalist output. All 84 run configs remain byte-identical; only
analysis changes. Training stays blocked until the collector, qualification, launch contract, and
automation are updated to v2.

## Artifact

- [Finalist analysis v2](../research/paper-1/finalist_analysis_v2.json)
