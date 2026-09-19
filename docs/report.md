# Speck technical report — working outline

**Status: engineering results available; flagship training and final evaluation ahead.**
Release the report alongside identified base and always-thinking assistant checkpoints. This is
an outline, not a completed paper or a claim of flagship quality.

The [preparation receipt](../experiments/pilot/preparation.json),
[H100 rehearsal](../experiments/qualification/h100-result.json) and
[timing study](../experiments/qualification/timing-result.json) establish software/recovery behavior
and bounded performance measurements. The [completed pilot](../experiments/pilot/h100-run.json)
adds 800 steps / 104,857,600 tokens, source-wise learning curves and actual trainer timing.
[Development scoring](../experiments/pilot/development-result.json) covers all 2,619 frozen tasks;
results remain weak, including 0/253 GSM8K strict and 0/33 custom compiled code pass@1. This is an
engineering endpoint, not release-quality evidence. No final partition was evaluated.

All eight checkpoints, both exports and recovery/evaluation records are verified in the
[backup closeout](../experiments/pilot/backup-result.json). Preserve original failures, interruption
and recovery history. The earlier 2.20-hour training and 2.15-hour development scoring projections
are superseded by actual 2.265 trainer and 2.123 supervised evaluation hours for pilot accounting;
the approximately nine-hour final-partition estimate remains unmeasured. GH200/ARM64, distributed
scaling, long-context quality and flagship capability still require their own evidence.

Working title: **Speck: Developing a 1.2B Hybrid Assistant Under a Fixed Compute Budget**.

## Abstract

Write after final evaluation. State the actual model, training tokens and data composition, compute
consumed, measured strengths and weaknesses, and supported inference configuration. Include the
scope of comparison and material limitations. Do not substitute proposed budgets for measured cost.

## 1. Objective and scope

An English-first small assistant with broad usefulness and particular emphasis on math, code,
instruction following, and tools. Report each capability separately. Define the supported context,
output budget, and tool protocol from executed tests. Preserve distinct base and assistant artifacts.

The first release is a model-development report. An architectural advantage, causal data effect,
or general scaling claim needs an explicitly controlled experiment beyond the engineering pilot.
The inherited KDA/GQA building blocks are described as prior work.
The [competitive strategy](competitive.md) requires contemporary product comparisons as well as
compute/token-matched research references. SpeckLabs' first-step objective is an independently
pretrained base and a thinking assistant, with a reusable record for future larger allocations.
Disclose any teacher supervision separately; external pretrained weights are not the initialization
route. Measure both training efficiency and inference cost at useful quality. Publish exact configs,
manifests, evaluation definitions, costs and limitations alongside the checkpoints, with corpus
redistribution limited to eligible material. A single model's learning curve is not a scaling law.

## 2. Architecture and implementation

Record the exact producing revision and model configuration: 24 layers, width 2048, 18 KDA and six
NoPE GQA layers, SwiGLU, tied embeddings, and frozen Mistral tokenization. The selected all-active configuration
has 1,195,884,576 parameters and 32,003 embedding rows. Report actual released geometry if it changes.
Describe parameter/activation precision, optimizer groups, loss implementation, activation
checkpointing, recurrent state and attention cache, and supported export/runtime behavior.

Relevant foundations include [Kimi Linear](https://arxiv.org/abs/2510.26692),
[MiniCPM](https://arxiv.org/abs/2404.06395), and the sources in [research notes](research.md).
Separate inherited methods from the project's own measurements and implementation choices.
The [architecture program](architecture-program.md) records the selected 1.2B all-active model and
its runtime/context qualification. No dense-versus-MoE or all-GQA comparison is planned for this
release. Report measured quality-versus-cost curves and implementation improvements with their
numerical checks. Separate equal-token, equal-FLOP and equal-wall-time evidence where available;
they are not interchangeable. Without an architectural control, do not claim a causal advantage
from the KDA/GQA allocation. Attribute data and post-training gains only through executed controls.

## 3. Data and training

Publish a source-level summary of pinned revisions, permitted use, filters, exclusions, selected
unique stock, training exposure, repetition, and validation separation. Keep source corpus text
and packed shards outside the release. Explain the deterministic source ordering and document
partitioning, and disclose the limits of exact/near-duplicate and benchmark matching.

For each executed phase, report parent checkpoint, token endpoint, batch, optimizer, schedule,
precision, hardware, elapsed time, and total allocated GPU-hours including failed attempts.
The [105M-token pilot](../experiments/pilot/README.md) tests engineering readiness. Its mixture and
learning rate are starting choices, not optimized findings.

For SFT, count both total context and supervised tokens. Describe masked context turns, complete
conversation length coverage, reasoning format and output limits, and executable tool supervision.
Structural validity of a teacher trace does not establish answer correctness.
Report brief/deep reasoning coverage and reasoning/final-answer token shares, plus executable
verification coverage by source. The final assistant follows one always-thinking protocol; report
adherence and measured behavior without claiming that formatting proves reasoning quality.
Separate successfully executed trajectories from unexecuted or
unsuccessful traces. Record any capability-focused continuation as its own phase with broad-data
replay, distinct from masked SFT.

## 4. Evaluation

Describe pinned tasks, prompt templates, scorers, deterministic development/final assignment,
failures, denominators, decoding budgets, and comparator revisions. The prepared pilot protocol uses
custom subsets; published full-benchmark numbers are not directly comparable.

| Measurement | Required evidence |
| --- | --- |
| Learning | Source-wise train/validation curves and checkpoint token counts |
| General capability | Knowledge and commonsense results; representative writing/conversation review |
| Math | Checked final-answer accuracy with output tokens and parse failures |
| Code | Held-out executable tests in an isolated runner; pass@1 and timeouts |
| Instructions | Strict prompt- and instruction-level compliance |
| Tools | Correct selection/arguments, use of results, task completion, errors and no-tool cases |
| Efficiency | All-in training cost; prefill/decode latency, throughput, memory, workload sizes |
| Reliability | Missing-evidence cases, abstention, correction, and failure examples |

Record base and assistant results separately. Compare models under the same evaluated conditions.
Report sampling uncertainty and acknowledge that one training seed does not establish robustness
across training runs. Any narrowly controlled follow-up needs its question and cost fixed before outputs.
The [ZGCM-1 review](research.md#zgcm-1-review--2026-09-18) proposes a possible SFT selection-policy
comparison; it is not scheduled or evidence of an effect. If executed, disclose token/compute matching,
domain and length differences, repetition, and per-capability regressions alongside aggregate changes.
For reasoning and tools, record actual generated tokens, budget-exhaustion rates, tool calls, and
environment failures. Distinguish mean pass@1 over repeated samples from pass@k and best-of-k selection.

## 5. Results and limitations

**Pending model execution.** Preparation counts, software tests, and smoke losses are not capability
results. Report weak categories, unfavorable comparisons, unsupported serving features, contamination
limitations, and all material failed runs. Describe architecture/runtime trade-offs only where measured.

## 6. Reproducibility and release

Release identified checkpoints, tokenizer/chat metadata, native and supported export code, runnable
configurations, evaluation/analysis commands, aggregate results, and model cards. Hash the artifacts.
Regenerate the main tables from retained outputs and verify exported logits/generation against native
inference. The report and model cards must agree on supported context, protocols, costs, and limits.
