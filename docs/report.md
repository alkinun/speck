# Speck technical report — working outline

**Focus: data and training across pretraining, mid-training and post-training on a fixed 1.2B model.**
Release the report alongside identified base and always-thinking assistant checkpoints. The model's
attention, size and implementation choices provide supporting context. Broader architectural research
belongs to later releases with larger allocations; no architecture search is scheduled here.

**Status: engineering results available; flagship training and final evaluation ahead.** The
[preparation receipt](../experiments/pilot/preparation.json),
[H100 rehearsal](../experiments/qualification/h100-result.json) and
[timing study](../experiments/qualification/timing-result.json) establish bounded software, recovery
and performance evidence. The [pilot](../experiments/pilot/h100-run.json) adds 104,857,600 tokens over
800 steps and 2.265 trainer hours. [Development scoring](../experiments/pilot/development-result.json)
took 2.123 evaluation hours over 2,619 tasks, including 0/253 GSM8K strict and 0/33 custom compiled
code pass@1. This weak engineering endpoint does not establish release quality or a data effect.
No final partition was evaluated. [Backups](../experiments/pilot/backup-result.json) retain all eight
checkpoints, both exports and recovery/evaluation records. Keep failed attempts and original receipts.

Working title: **Speck: Data and Training Recipes for a 1.2B Coding and Reasoning Assistant**.

## Abstract

Write after final evaluation. State the data/training contribution, actual stage exposures and
compute, measured capability and cost, comparison scope and limitations. Identify the fixed model
briefly. Proposed budgets and preparation counts are not measured training results.

## 1. Objectives and research questions

Develop a useful English-first coding, math and tool assistant from scratch under a fixed compute
budget. Explain how data coverage, checked supervision and training stages contribute to capability,
while retaining general usefulness. External models are comparators or disclosed teachers.

The main empirical question is whether a bounded checked-code data intervention improves held-out
capability at matched training exposure. The complete training account additionally documents what
changes across pretraining, capability/context mid-training, SFT and any executed RL stage. Before/after
stage results describe progression; extra compute and changed objectives prevent automatic causal
attribution to data alone. The [program](program.md#training-lifecycle) defines stage and budget ownership.

## 2. Data construction and accounting

Describe each stage's data purpose, source revisions, selection criteria, correctness checks,
coverage, exclusions and sampling weights. Distinguish raw stock, qualified unique supply, actual
exposure, replay and rejected records. Publish permitted source manifests and processing recipes;
source corpus text and packed shards are not release artifacts.

Follow source families across natural documents/code, rewrites, exercises, assistant traces and RL
prompts. Keep held-out families out of every training and teacher-generation route. Explain joint
exact/near deduplication, partitioning, benchmark exclusions and their remaining blind spots.

| Stage | Data evidence to report |
| --- | --- |
| Pretraining | Domain/language/source coverage, unique tokens, quality filters, document packing and actual mixture |
| Capability mid-training | Task-oriented code/math/repair material, independent checks, broad-data replay and overlap with the base |
| Context mid-training | Coherent repositories/documents, length and position coverage, short replay and reused source families |
| SFT | Unique conversations/task families, verified outcomes, complete trajectories, context/supervised tokens and reasoning lengths |
| RL | Prompt/task families, reference answers/tests, verifier validity, task difficulty, rollout outcomes and environment failures |

Publisher quality labels, syntactically valid traces and passing self-generated tests are not
independent correctness evidence. Report verification coverage and sampled-audit uncertainty.
Explain any supply shortfall or mixture revision instead of treating collection quotas as results.

## 3. Training lifecycle

For every executed phase record its parent checkpoint, data manifest, objective/loss mask, token
endpoint, batch, optimizer-state policy, LR schedule, precision, hardware and allocated GPU-hours.
Include preparation/teacher costs, failures and recovery. Present source-wise learning curves and
capability versus tokens and cost at declared checkpoints.

### Pretraining

Describe from-scratch initialization, broad code/math/general coverage and the realized recipe.
Explain deterministic data order, packing boundaries and continuation checkpoints. Starting weights
and the engineering pilot's schedule are not optimized findings. State the achieved horizon and
its feasibility relative to the desired 320–400B base target.

### Mid-training

Separate capability-focused continuation from context extension. Describe selected code/math/repair
material and broad replay, then coherent long data and short-task retention. Freeze each transition's
objective, mixture, length, optimizer/schedule policy and cost. Capability continuation is a portion
of the declared base horizon; the proposed 8B context extension is additional. Report actual splits.

Use related-prefix benefit, positional retrieval and multi-file/document tasks to establish useful
context. A configured ceiling does not establish 128K capability. Document any unexecuted stage as
such; mixture-phase support alone is not evidence of a qualified changed-data continuation workflow.

### Post-training: thinking SFT and verified-reward RL

For SFT, explain assistant-only masks, complete conversations, tool serialization, brief/deep reasoning
and actual task outcomes. Count processed context, padding, supervised reasoning and final-answer
tokens separately. The always-thinking protocol does not make formatting evidence of reasoning quality.

For any executed RL stage, describe prompt selection/difficulty, reference answers, code tests and
verifiers, rollout limits, reward definitions and optimization. Report all-fail/all-pass groups,
reward failures, reasoning-length growth, held-out transfer and general regressions. Count generation,
scoring, updates and retries. Preserve the SFT checkpoint and explain whether RL justified promotion.
RL remains planned until its runtime and verifiers are qualified; an RL dataset is not an RL result.

## 4. Model choices and implementation

Give a concise account of the selected size, attention pattern and their rationale under the budget:
1,195,884,576 all-active parameters; 24 layers, width 2048, 18 KDA and six NoPE GQA layers, dense SwiGLU,
tied embeddings and frozen Mistral tokenization with 32,003 rows. Bind the producing config/revision.
The [model notes](model.md) distinguish inherited methods, design rationale and measured trade-offs.

Report training memory/throughput, cache behavior and supported context; explain limitations as well
as strengths. Put detailed kernels, numerical checks, optimizer groups and export implementation in
an appendix where appropriate. Neither this single size nor an unmatched external comparison can
establish an optimal size, attention ratio, architecture advantage or scaling law.

## 5. Controlled data study and evaluation

The [bounded code study](coding.md#first-comparison-to-prepare) compares natural code against natural
code plus checked exercises from a useful common checkpoint. Freeze both arms' manifests, code
share, exposure, schedule, evaluation endpoints and all-in cost before execution. It tests the
combined intervention; it does not isolate synthesis, selection and verification separately.
Do not expand this into a source/architecture sweep. Publish unfavorable or inconclusive results too.

Evaluate base, mid-trained, SFT and any RL checkpoints separately. Report source-held-out loss,
executable code, checked math, instructions, model-driven tools, useful context and general regressions.
The [evaluation guide](evaluation.md) and [competitive strategy](competitive.md) define protocols and
matched external comparisons. Freeze task/scorer/comparator identities, development/final partitions,
output budgets and failure denominators. Pilot subsets are not full-benchmark headline results.

Show quality versus training and inference budgets, including thinking tokens, tool calls and failed
attempts. Distinguish equal-token, equal-FLOP and equal-wall-time comparisons. Report uncertainty;
one training seed does not establish run-to-run robustness. Tool-environment golden checks are not
model capability scores. Additional studies require their own question, controls and budget.

## 6. Results, limitations and release

**Flagship results pending.** Describe realized gains and regressions, unsupported runtime/context
features, contamination limitations, teacher dependence, data-access restrictions and failed runs.
Explain what this fixed-model recipe can inform about later scaling without promising transfer.
MoE, attention residuals and broader attention/size research are future work, not current contributions.

Release identified checkpoints, tokenizer/chat metadata, source manifests, runnable training and
evaluation configs, processing/analysis code, curves, costs and model cards. Regenerate tables from
retained outputs and verify exported logits/generation against native inference. Report and model
cards must agree on stage lineage, supported context, protocols, costs and limitations.
