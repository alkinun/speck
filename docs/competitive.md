# Competitive release strategy — 2026-09-19

Status: strategy and proposed release gates, not measured Speck capability or a new launch.
Keep the current 1.2B KDA/GQA candidate while deciding whether a competitive release is feasible.
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

Keep 1.2B as the from-scratch research candidate; no evidence currently justifies shrinking it to
improve target-task performance at fixed compute. Equally, larger capacity does not prove superiority
over a smaller model trained longer. Do not select a size to enter an easier comparison bracket.

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
horizon, more compute, or a different initialization strategy; it cannot justify a quality promise.

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
   qualification/comparison work to existing reservations and avoid an architecture sweep.

## If product competitiveness is the priority

Adapting an eligible pretrained small model is the stronger route to investigate under a hard
compute cap: it reuses the upstream language foundation and spends our allocation on task data,
verified reasoning/tool supervision, context where needed and evaluation. Compare eligible base
and already-thinking starting checkpoints before choosing. This is a recommendation to consider,
not an authorized replacement for the current architecture or a guarantee of gains.

Such a release is a derivative with explicit upstream attribution. It cannot claim our from-scratch
architecture or training efficiency; disclose inherited pretraining separately from incremental
compute. The always-thinking interface remains a product requirement. The from-scratch candidate
could remain a bounded research artifact, but do not automatically fund two full model programs.

If owning the full from-scratch architecture is essential, retain 1.2B and use the report to establish
an honest measured contribution: data selection at a fixed budget, reproducible training, or a
verified quality/efficiency tradeoff. Publish a research checkpoint if competitive gates fail; do not
rename it a flagship solely because training completed. More compute can support a stronger attempt,
but 320–400B is not itself a release-quality threshold.
