# Paper outline

Working title: **Speck: Open Data Pipelines for Every Training Stage, Measured Across Scale**.

The paper reports the experiments the [program](program.md) defines. Every table is regenerated from
retained receipts; nothing is typed from memory. The [evaluation guide](evaluation.md) owns the
protocols it cites.

## 1. Introduction

The question, the step-by-step scaling motivation, and the contributions: per-stage pipelines,
controlled data experiments, measured scale transfer, and a complete open ledger.

## 2. Pipelines

One subsection per stage: sources and revisions, source-use conditions, extraction, filters,
exact and near deduplication, benchmark firewall, family partition, tokenization and packing.
Report raw stock, accepted unique tokens, exposure, replay and rejected material separately, with
CPU, storage and GPU cost per accepted token.

## 3. Method

The ladder and its shape rule, seed-noise calibration and minimum detectable effects, validation
packs, metrics per scale, predeclaration, and how every GPU-hour is accounted.

## 4. Pretraining data

Families P0 to P7. For each: the question, arms, per-rung results with seed spread, the decision,
and whether it held on the next rung. Negative and inconclusive families get the same space.

## 5. Scale transfer

For every family re-run on a larger rung, whether the smaller rung predicted the ranking and the
effect size. The ladder's predicted parent loss against the measured one. What this means for
choosing experiments at the next step's scale.

## 6. Decay and mid-training

Families D1 to D3 and M1 to M3: decay data, length and branch point; replay, repository context and
context-extension data, with useful-context evaluation beyond a configured maximum length.

## 7. Post-training

Families S1 to S3, R1, R2 and X1 on our parent and on the external control base, and where the two
parents disagree.

## 8. The released models

The chain of selected branches, their capability and cost, and a matched comparison with open models
of similar size. This section describes the models; it is not the paper's main claim.

## 9. Ledger and release

Every run, its cost and outcome, including failures and restarts; what is released and how to
rebuild what is not.

## 10. Lessons for the next step

What we would keep, change and stop doing at ten times the compute, stated as rules.
