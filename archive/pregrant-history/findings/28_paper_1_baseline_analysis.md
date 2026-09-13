# 28 — Paper 1 baseline analysis and stopping contract

## Question

How will the matched dense-global and five-cache KDA/GQA proxy runs be analyzed without choosing an
estimand, quality target, interpolation rule, or stopping decision after results are visible?

## Primary estimand

The candidate-minus-control direction is frozen as five-cache KDA/GQA minus dense global attention.
The primary endpoint is the paired difference in 20M-token final validation loss after 131,072,000
training tokens. The three initialization/data-order cells are the independent units; validation
batches are not treated as replicates.

The proxy screen uses the architecture-promotion-v1 one-sided paired Student-t rule. The upper 95%
bound must be at or below 0.01 nats. Every corpus-source paired upper bound must also be at or below
the 0.02-nat guardrail. With only three pairs, this is deliberately a variance-estimating proxy screen,
not architecture-promotion evidence.

## Secondary views

- Fixed analytic FLOPs compares the KDA endpoint at 131,072,000 tokens with dense loss interpolated at
  its batch-aligned 102,891,520-token compute match.
- Fixed time uses the smaller final `steady_training_seconds` budget within each pair and interpolates
  both loss curves at that time. Compile/startup, validation, checkpoint, and orchestration time remain
  separately reported rather than contaminating the training curve.
- Time to quality locks one loss target from the three dense controls only: their maximum final loss,
  rounded upward to six decimal places. The target-lock artifact must exist before candidate result
  records are created. Any pair in which either arm misses the target remains right-censored; the code
  does not form a complete-case confidence interval.

Piecewise-linear interpolation is frozen because the training cadence supplies only six validation
boundaries. It is a reporting approximation, not a claim about the shape between observations.

## Stopping decision

The design has zero interim efficacy looks and zero interim futility looks. All six model runs must
reach the fixed endpoint. Quality-based pair abandonment is forbidden. Safety, non-finite numerical
state, resource, lineage, evaluation-integrity, and operator-correctness failures may interrupt a run;
recovery resumes the same frozen cell. An irrecoverable attempt stays in the record and any rerun must
be preregistered rather than silently substituted.

## Implementation

Base checkpoints now retain every aggregate and per-source validation point with optimizer and steady
training clocks. The Paper 1 collector verifies the exact arm, seed, data offset, packed-data manifest,
parameter/FLOP geometry, final checkpoint, validation cadence, timing monotonicity, and run summary.
Separate target-lock and final-analysis modes preserve their input paths and SHA-256 identities.

The analysis implementation has no component-attribution authority. Passing it cannot credit KDA,
NoPE, GQA, the 3:1 mixer ratio, or a systems mechanism, and cannot authorize paper-scale training.

## Artifacts

- [Frozen analysis plan](../research/paper-1/baseline_analysis.json)
- [Baseline matrix](../research/paper-1/baseline_matrix.json)
- [Analysis implementation](../speck/paper_baseline_analysis.py)
- [Analysis CLI](../scripts/paper_baseline_analyze.py)
