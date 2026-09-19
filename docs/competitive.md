# Competitive release strategy — 2026-09-19

Status: strategy and proposed release gates, not measured Speck capability or a new launch.
The selected flagship is the existing 1.2B-total, all-active KDA/GQA model with dense feed-forward
layers. MoE investigation and architectural comparison arms are outside this release program.
The user clarified that developing our own pretraining capability is essential; pretrained
adaptation is not an alternative first-release route.
A 100B budget-fit projection is not evidence that the result will be competitive. The desired
320–400B horizon also does not establish parity with current releases.

## Primary-source comparison

Publisher figures below are contextual evidence, not results from a matched Speck evaluation.
Reviewed September 19, 2026; pin immutable weight/config/tokenizer revisions before execution.
Different tokenizers, phase definitions, inference budgets and evaluation protocols prevent direct
conversion of training-token ratios into capability ratios.

| Reference | Published evidence | Role in our comparison |
| --- | --- | --- |
| [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 1.081B total / 0.680B non-embedding parameters; 128K context; 200B deep-thinking + 200B hybrid-thinking SFT tokens, followed by specialist RL and OPD | Primary contemporary reasoning/code/tool comparator; distinguish its base, SFT and final checkpoints |
| [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) | Reported GSM8K 85.60, MATH-500 87.96 and BFCLv3 56.97; thinking results average five runs | Direct size/thinking/efficiency comparator, not an assumed weak baseline |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | A smaller hybrid with text/vision support and a published thinking-mode evaluation | Lower-size contemporary comparator; restrict to text and account for total/non-embedding/vision parameters |
| [OLMo 2-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) | 4T stage-one tokens plus 50B mid-training; intermediate checkpoints are available | Base-learning reference; select nearby token checkpoints for research and final checkpoints for product comparisons |
| [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | 11T training tokens | Upper-size base/reference, not the only or strongest contemporary product baseline |

LFM's IFEval summary averages strict/loose prompt/instruction metrics; its BFCLv3 uses a custom
handler. Do not compare those directly to our strict metric or to Qwen's BFCL-v4. Training-token
figures are publisher definitions, not counts under our tokenizer. We have not reproduced these
results. Parameter count alone is also incomplete: Speck has about 65.5M tied embedding parameters,
so its non-embedding count is approximately 1.130B, unlike some nominally similar-size models.

## Recommendation

The first release develops a data and training recipe across pretraining, mid-training and
post-training on the selected 1.2B backbone. The [model notes](model.md) explain attention/size
choices and their measured limitations. MoE, attention residuals and broader architectural research
belong to later releases with larger allocations. Qualify the current runtime, improve the data and
measure useful thinking/code/tool behavior. The 91-hour bounded comparison is a data intervention.

The current 5,000-total-GPU-hour allowance does not support a confident broad best-in-class release
claim. A narrow competitive product is a hypothesis to test: reliable Python/JavaScript/TypeScript
repository repair and tool execution with bounded reasoning cost. Math and general instruction
following remain required capabilities/regression checks. Approximately 128K remains a development
target, not an advertised advantage until useful-context and cost tests pass against current models.

The first hardware work should establish whether our execution is unnecessarily limiting scale.
The H100 measurements used deterministic eager execution, FP32 parameters/optimizer and BF16
activations. A six-step microbatch-four probe improved steady throughput by about 20%, but did not
qualify restart or sustained operation. Profile batch/accumulation, optimizer and kernel costs,
input/validation/checkpoint overhead, then measure four-worker communication. Preserve update
semantics and validate numerical/recovery behavior for any production change. These are bounded
runtime qualification tasks, not an assumed 3–4x gain or a new GPU launch.

Keep the existing phase reservations until measured costs justify a revision. The 2,300-hour base
reservation needs 38.6K/48.3K effective tokens/s per allocated GPU for 320B/400B; four workers alone
do not close that per-GPU gap. All projections and protected stages remain in the
[scale plan](../experiments/main-data/README.md). An unchanged throughput result requires a smaller
horizon, measured budget reallocation or more compute; it cannot justify a quality promise.

## Before committing the main training budget

1. Freeze a small task panel and the intended claim before selecting new training data. Include
   held-out repository repair, code generation, multi-step tools, checked math and instruction
   retention. Exclude task families from all new corpus and teacher-generation routes. Do not tune
   on the final partition or replace standard benchmarks with convenient custom tasks.
2. Reuse the existing evaluation infrastructure for a bounded development comparison with MiniCPM,
   LFM and Qwen under matched tasks, tools, context, output caps and declared reasoning modes.
   Record success, actual output tokens, latency, memory and cost per successful task. Native chat
   templates are allowed; disclose adapters and format failures. Compare base models separately.
3. Set numerical success/regression tolerances from product requirements and development baselines,
   then freeze them before final evaluation. A competitive claim requires matching or exceeding the
   strongest selected comparator on the primary task endpoint, or an explicit useful efficiency
   tradeoff at an agreed quality floor. Report paired uncertainty and every declared endpoint;
   broad parity requires broad results. Passing a schema or nominal context limit is insufficient.
4. Assess the optimized runtime, eligible corpus and bounded learning evidence together. An early
   checkpoint comparison can reject a poor route but cannot promise the final ranking. Charge
   runtime qualification and the bounded data comparison to their existing reservations. Keep the
   selected architecture fixed; no MoE or all-GQA comparison arm is planned.

## Release evidence and later scaling

Success combines useful capability, measured cost and an open account of data and training.
Release identifiable base and thinking-assistant checkpoints with stage lineage, source manifests,
processing recipes, exact training/evaluation configurations, token/compute ledgers and failure
analysis. The [report outline](report.md) distinguishes controlled data results from stage progression
and matched product comparisons. Share permitted manifests and recipes without redistributing
restricted corpus bytes.

Preserve intermediate model/optimizer checkpoints, source-wise loss, capability versus tokens/cost,
and length-dependent runtime measurements. These inform later data and architecture studies as
compute/model sizes grow; a single 1.2B run cannot establish a scaling law. The future 50,000-hour
ambition is not funded by this allocation, and a tenfold compute increase does not promise a tenfold
quality improvement. Detailed stage/budget decisions belong in the [program overview](program.md).

If capability or efficiency falls short, publish that result and its limitations. Release claims
follow measured evidence; openness alone does not establish a competitive advantage.
