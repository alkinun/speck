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

The clarified objective is to develop SpeckLabs' first architecture and competitive flagship,
including its pretraining and post-training. The [architecture program](architecture-program.md)
records the selected 1.2B all-active design and distinct token/FLOP/runtime efficiency measurements.
The user has closed the MoE branch of investigation to focus this program on one model. Optimize
and qualify that model, improve its data, and establish useful thinking/code/tool performance.
The 91-hour bounded comparison remains a data study, not an architecture competition. Open reporting
and reusable infrastructure support the flagship objective; model quality still requires evidence.

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

## SpeckLabs first-step objective

The user clarified on September 19 that the 5,000-GPU-hour program must develop our own pretrained
model and the capability to scale to a future 50,000 GH200-hours and beyond. This supersedes the
previous recommendation to consider adapting an external base for a competitive product.
Keep external models as demanding comparators and, where qualified, teachers; disclose teacher
sources and generation/verification costs. Our base weights start from scratch.

Success has three parts: useful task capability, measured efficiency, and an open technical record.
Training efficiency means capability at a declared all-in training budget and throughput under a
specified workload. Inference efficiency means latency, memory and cost per successful task under
comparable quality, context and output conditions. More tokens/s or a larger context setting alone
does not establish either kind of advantage. The current hybrid must earn efficiency claims through
measurement; its global-attention layers still require a growing cache at longer context.

Keep the selected 1.2B all-active model, working data recipe and always-thinking post-training target.
Freeze the final token horizon after profiling, supply qualification and bounded learning evidence.
Do not automatically shrink to maximize token count, enlarge to imply capacity, or make 100B/400B
an unconditional release threshold. Phase reservations may be revised from measured needs while
keeping the 5,000-hour total and adequate evaluation/recovery coverage.

Ship identifiable base and thinking-assistant checkpoints with source manifests, exact training
configs, code/environment identities, token and compute ledgers, learning curves, evaluation
protocols and failure analysis. Publish data derivatives only where redistribution is permitted;
otherwise publish source identities, processing instructions and permitted manifests. Distinguish
an openly documented process from redistributing every underlying dataset.

Design this run to inform the next allocation: retain intermediate model/optimizer checkpoints,
source-wise loss, capability versus tokens/compute, length-dependent runtime and memory, and
recovery/distributed measurements. Freeze checkpoint/evaluation cadence before launch; do not
return to frequent progress polling. One 1.2B run supplies a learning curve, not a parameter-scaling
law. A future larger model needs separate controlled evidence and a new budget.
A tenfold compute increase must be divided among size, tokens, data work and post-training; it
cannot be promised as a tenfold token or quality increase.

A competitive flagship is the goal, with release claims set by results. If the first run misses the
capability/efficiency gates, report that outcome and preserve a reusable research release rather
than claiming an advantage that was not measured. No additional compute or parallel full-model
program is authorized by this strategy update.
