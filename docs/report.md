# Speck technical report — working outline

**Focus: an open data pipeline across all six training stages near 1.2B parameters** — pretraining,
endpoint decay, mid-training, post-training SFT, post-training RL and final self-distillation.
Release the report alongside identified base and always-thinking assistant checkpoints. The model's
attention, size and implementation choices provide supporting context only. Architectural research,
including the bounded matched-attention comparison, is deferred to a later allocation; the backbone
here is declared, not compared, so the report carries no architecture claim. The release includes
1,230 hours of data research across training stages.

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

The main empirical question is how a declared pretraining-data choice changes learning at matched
exposure on the fixed model. Run the bounded baseline/candidate comparison before selecting the main
starting recipe after costed screening; a checked-code intervention is one candidate after source
qualification. Downstream studies compare data on matched useful parent checkpoints before each
production stage. Report the
scope and limits of recipe selection rather than claiming a globally optimal mixture.
The complete training account additionally documents what
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
| Capability mid-training | Repository structure, issue/PR/commit context, verified repairs, tool schemas, workflows, trajectories and independent checks |
| Context/agentic mid-training | Coherent multi-file repositories, dependency distance, tool state, recovery, long-horizon trajectories, length and position coverage |
| SFT | Unique conversations/task families, verified outcomes, complete trajectories, context/supervised tokens and reasoning lengths |
| RL and final self-SFT | Prompt/task families, reference answers/tests, verifier validity, teacher lineage, anchor replay, rollout outcomes and environment failures |

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
its feasibility relative to the 80B 4K-base working horizon and 1,800-hour base reservation.

### Mid-training

Describe the 4K capability bridge, 16K repository reasoning stage and 32K long-horizon agentic
stage. Report repository families, dependency links, issues/PRs/commits, verified repairs, tool
schemas, grounded workflows, executable trajectories, replay and derived-data lineage. Freeze each
transition's objective, loss mask, packing, mixture, length, optimizer/schedule policy and cost.
Separate data continuation from context continuation; the combined production reservation is 600
GPU-hours and the comparison reservation is 300 GPU-hours.

For trajectories, report environment images, tool schemas, turn structure, valid/invalid calls,
test outcomes, failure/recovery labels, supervised positions and masked observations. Use related-prefix
benefit, dependency-distance retrieval, positional retrieval, multi-file repair, tool-state
continuation and short-task retention to establish useful context. A configured ceiling does not
establish capability at that length.

### Post-training: thinking SFT, verified-reward RL and final self-SFT

For SFT, explain assistant-only masks, complete conversations, tool serialization, brief/deep reasoning
and actual task outcomes. Count processed context, padding, supervised reasoning and final-answer
tokens separately. The always-thinking protocol does not make formatting evidence of reasoning quality.
The [post-training research synthesis](../experiments/main-data/post-training-research.json) defines the
separate general, reasoning, agent, verifier, preference and on-policy-feedback data banks. Treat any
hybrid or token-budget control as an evaluated hypothesis, not a release feature.

For any executed RL stage, describe prompt selection/difficulty, reference answers, code tests and
verifiers, rollout limits, reward definitions and optimization. Report all-fail/all-pass groups,
reward failures, reasoning-length growth, task-conditioned quality/token curves, held-out transfer and
general regressions. Count generation, scoring, updates and retries. Preserve the SFT checkpoint and
explain whether RL justified promotion. Correctness must be established before applying any efficiency
preference to response length.
RL remains planned until its runtime and verifiers are qualified; an RL dataset is not an RL result.

For final self-SFT, describe the selected parent, frozen prompt pool, teacher checkpoint, generation
settings, environment and verifier receipts, rejected records, anchor replay and the comparison to the
pre-self-SFT parent. Report held-out transfer and regressions before promoting the final checkpoint.

## 4. Model choices and implementation

Give a concise account of the selected size, attention pattern and their rationale under the budget.
The reference has 1,195,884,576 all-active parameters; 24 layers, width 2048, 18 KDA and six NoPE GQA
layers, dense SwiGLU, tied embeddings and frozen Mistral tokenization with 32,003 rows. The production
backbone is this reference, fixed by declaration. Bind each producing config/revision and report
reference-only measurements: actual parameter/FLOP counts, training cost/memory and
prefill/decode/cache measurements at declared lengths and batch sizes. State plainly that no
matched control was run, so no architecture comparison is claimed; the deferred design is
published so readers can see what was not done. Separate implementation effects from architecture.
The [model notes](model.md) distinguish inherited methods, design rationale and measured trade-offs.

Report training memory/throughput, cache behavior and supported context; explain limitations as well
as strengths. Put detailed kernels, numerical checks, optimizer groups and export implementation in
an appendix where appropriate. Neither this single size nor an unmatched external comparison can
establish an optimal size, attention ratio, architecture advantage or scaling law.

## 5. Controlled data study and evaluation

The [research design](../experiments/main-data/README.md#research-before-the-main-run) proposes a
baseline and at most three single-factor screening candidates (code-bank, natural-web and calibrated AI-generation/provenance filtering), followed by baseline-versus-selected
confirmation on two fresh paired seeds before main pretraining. Freeze the data contrast,
manifests, exposure, schedule, validation mixture, development endpoints and all-in cost. Report
seed variation, source/domain loss and capability curves, including uninformative endpoints.
If several data components change together, attribute results to the combined recipe, not one source.
Checked-code substitution does not isolate synthesis, selection and verification separately.
Publish the selection decision, unfavorable/inconclusive results and limits on extrapolating short
runs to the 80B working horizon. A later useful-checkpoint continuation study answers a separate question and cannot
retroactively justify the original mixture. The 700-hour pretraining study includes screening,
the three-arm endpoint-decay comparison, confirmation, evaluations and overhead. The
[mid-training packet](../experiments/main-data/mid-training-study-packet.json)
reserves 360 hours for proxy, objective/packing, context and 1.2B confirmation studies. The
[post-training packet](../experiments/main-data/post-training-study-packet.json) reserves 170 hours
for SFT selection, fixed-policy RL feasibility, a final self-SFT pilot and support. Start comparison
arms from the same useful parent, hold objective and exposure fixed where testing data effects, and
declare every combined intervention. Report capability per mid-training token and GPU-hour, SFT
convergence, early RL adaptation, self-distillation transfer and held-out results under fixed downstream recipes.
RL research currently covers fixed-policy prompt/verifier feasibility, not a policy-training data
ablation. Charge comparisons separately from production. Report inconclusive results and downstream
recipe decisions as well as gains.

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
