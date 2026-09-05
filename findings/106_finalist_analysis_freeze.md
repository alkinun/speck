# 106 — Six-pair finalist analysis freeze

## Frozen design

Before creating any finalist config or output, the longer-horizon analysis is frozen at six paired
seed/data-order cells and twelve runs. Each arm trains for 1,539,833,856 tokens—exactly 23,496
65,536-token optimizer steps. Seeds 42/43/44 are crossed with offsets 0 and 1,610,612,736; candidate
and dense remain paired within every cell.

Validation occurs at step 0 and exact quartiles 5,874/11,748/17,622/23,496. The final validation keeps
the existing 20M requested-token complete-batch contract. The primary rule retains the +0.01 aggregate
and +0.02 every-source non-inferiority margins, now with a one-sided 95% Student-t bound at df=5.

All six dense controls must complete before the worst-control time-to-quality target is locked. Only
then may candidate results exist. There are zero interim efficacy/futility looks and no quality-based
pair abandonment. Fixed analytic-FLOP, steady-time, and time-to-quality views remain secondary.

## Authority

This freezes analysis and stopping rules only. It authorizes finalist materialization, not training.
Even a later language pass cannot attribute components, establish novelty, or promote without the
remaining capability, systems, runtime, scale-transfer, and release gates.

## Artifact

- [Finalist analysis v1](../research/paper-1/finalist_analysis_v1.json)
