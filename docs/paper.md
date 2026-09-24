# Paper outline

Working title: **Speck: Open Pretraining and Mid-Training Data Pipelines, Measured Across Scale**.

The paper reports the experiments the [program](program.md) defines. Every table is regenerated from
retained receipts; nothing is typed from memory. The [evaluation guide](evaluation.md) owns the
protocols it cites.

## 1. Introduction

The question, the step-by-step scaling motivation, and the contributions: pretraining and
mid-training pipelines, controlled data experiments, measured scale transfer, and a complete open
ledger.

## 2. Pipelines

Sources and revisions, source-use conditions, extraction, filters, exact and near deduplication,
benchmark firewall, family partition, tokenization and packing, for pretraining and mid-training
data. Report raw stock, accepted unique tokens, exposure, replay and rejected material separately,
with CPU, storage and GPU cost per accepted token.

## 3. Method

The ladder and its shape rule, per-rung learning rates, seed-noise calibration and minimum
detectable effects, validation packs, metrics per scale, the fixed SFT probe, predeclaration, and
how every GPU-hour is accounted.

## 4. Pretraining data

Families P0 to P7 and C. For each: the question, arms, per-rung results with seed spread, the
decision, and whether it held on the next rung. Negative and inconclusive families get the same
space.

## 5. Scale transfer

Family T: for every contrast re-run at 130m and 410m, whether the 50m rung predicted the ranking and
the effect size. The ladder's predicted parent loss against the measured one. Family DR against the
parent's decay results. What this means for choosing experiments at the next step's scale.

## 6. Decay

Families D1 to D3: decay data, length and branch point, in held-out loss and SFT-probe scores.

## 7. Mid-training

Families M1 to M4: replay, repository context and context-extension data, with short-task
retention and useful-context evaluation beyond a configured maximum length, in held-out loss and
SFT-probe scores.

## 8. The released models

The chain of selected branches, the light SFT assistant, their capability and cost, and a matched
comparison with open models of similar size. This section describes the models; it is not the
paper's main claim.

## 9. Ledger and release

Every run, its cost and outcome, including failures and restarts; what is released and how to
rebuild what is not.

## 10. Lessons for the next step

What we would keep, change and stop doing at ten times the compute, stated as rules, including
which post-training questions the next step should take up.
