# Research informing the first baseline

Reviewed 2026-09-19 using primary papers and official releases. These are lessons and comparisons,
not a promise to reproduce another lab's scores or compute budget. The decisions live in [PLAN.md](../PLAN.md).

OpenBMB is a primary ongoing reference for our data work. The [coding plan](coding.md) records the
specific UltraData-Code evidence, bounded local inspection, first comparison to prepare, and review
cadence. This prioritizes one testable data intervention while preserving independent baselines.

| Source | What it supports | Implication for Speck |
| --- | --- | --- |
| [Qwen3 report](https://arxiv.org/abs/2505.09388), sections 3 and 4.5–4.7 | Broad pretraining precedes capability-focused data and post-training. Small models use response and on-policy distillation. The reported 8B comparison favors distillation over direct RL in its tested setting. | Preserve general coverage while improving math/code. Start with supervised behavior and verified teacher examples; do not assume the reported distillation cost advantage transfers to 1.2B or our teachers. |
| [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) | A compact model uses a repeating three-Gated-DeltaNet/one-attention layout. It also includes a vision encoder. | Hybrid layouts are a relevant comparison, but GDN is not Speck's KDA. Similar ratios do not establish quality, kernel parity, or equivalent training cost. Compare text behavior under declared conditions. |
| [MiniCPM paper](https://arxiv.org/abs/2404.06395), section 4 | Warmup-stable-decay separates continued learning from endpoint decay and supports reuse of pre-decay checkpoints. | Retain continuation checkpoints and cost short pilot runs before committing to a long horizon. Its numerical learning rates use its parameterization and cannot simply be copied. |
| [MiniCPM5-1B release](https://huggingface.co/openbmb/MiniCPM5-1B) | Standard dense GQA, staged base/mid/post-training, SFT, RL teachers, and on-policy distillation target general, reasoning, coding, and tool behavior. The card reports 200B deep-thinking plus 200B hybrid-thinking SFT tokens. | A near-size capability reference. Its post-training scale alone warns against treating published quality as attainable with a small fraction of its training. Measure our own recipe and count teacher work. |
| [MiniCPM5-2B release](https://github.com/OpenBMB/MiniCPM) | The September 7 release describes capability-focused mid-training, 400B SFT tokens, and merging specialist RL teachers through distillation. | A stronger small-model reference, not another architecture lane. It reinforces the importance of post-training and executable agent evaluation. |
| [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) | A large MoE combines broad pretraining, systems work, SFT, and RL; it reports 14.8T pretraining tokens and 2.788M H800 GPU-hours for full training. | Learn from staged capability development and cost accounting. Its MoE, MLA, FP8 stack, and hardware totals are not a drop-in recipe for our scale. |
| [DeepSeek-R1 report](https://arxiv.org/abs/2501.12948), sections 2.3–2.4 and 4 | Cold-start examples address readability/language problems; verified rewards improve reasoning. Distilled small models use curated teacher samples, and the paper compares distillation with small-model RL. | Prefer a measurable SFT/distillation baseline before building RL infrastructure. Better math scores do not establish reliable tool use, instruction following, or general quality. |

## Practical conclusions

1. Quality and coverage of data, optimization, post-training, and evaluation all matter. Keep one
   model candidate while establishing the pipeline; do not turn each paper into another sweep.
2. Separate general pretraining from explicit instruction/reasoning/tool behavior. Domain emphasis
   must preserve enough broad data and replay to avoid losing ordinary usefulness.
3. Verified answers and executable code tests are useful supervision and evaluation signals. Include
   failed solutions, tool errors, corrections, and unanswerable cases in a carefully checked recipe.
4. Count input and output tokens, teacher/generation effort, verification, and failed attempts.
   Long reasoning can consume both training and inference budgets without improving answers.
5. Compare base with base and assistant with assistant. A parameter-matched public model can have
   vastly different data, compute, context, and distillation history.

## ZGCM-1 review — 2026-09-18

Reviewed [report v1](https://arxiv.org/html/2609.13356v1), the public
[dataset card](https://huggingface.co/datasets/zgcagi/ZGCM-1-Data/tree/a2e96e71e6d8316f7f40ca53bec15dad49b845cb),
[model card](https://huggingface.co/zgcagi/ZGCM-1-7B/tree/a0e10af50e6d11e3a3fd3575aec80deb98666489),
and [code documentation](https://github.com/zgcagi/ZGCM-1/tree/8c9677a03b1ae556f93b5cf6afa62b4bc0176d28).
No weights, gated corpus, or upstream implementation were imported or executed.

The model card describes 7.39B parameters, sliding-window/global attention, 4.19T pretraining
tokens, approximately 600B mid-training tokens, and 256K context. This is a much larger training
program than Speck's proposed allocation. Its attention is not KDA. Similar use of hybrid layers
and Muon does not validate our architecture or predict our quality.

Important qualifications from the report:

- Table 10's SFT filtering reduces approximately 2.08M examples to 1.145M: six-task mean
  67.78 → 68.83, but HumanEval+ 73.78 → 67.56 and IFEval 72.64 → 71.94.
- Section 3.1.3's 4.2× efficiency claim multiplies component estimates. Section 2.2 reports
  1.13× SWA/full-attention throughput at 4K; it does not compare against our KDA implementation.
- Table 6 implies 192 H100s for general pretraining (TP 2 × DP 96), not a GPU-hour total.
- Section 10.4's thinking/direct transfer is an observational checkpoint comparison.
- Section 5.1 allows 258,048 generated tokens and generally averages pass@1 across 32 runs;
  this is not best-of-32. Appendix 14 identifies external comparison values as cited releases.

### Changes to prioritize for Speck

These are proposed work after runtime qualification, not executed experiments or additions to the
frozen pilot. They refine the existing baseline rather than create another architecture search.

1. **Audit supervision before scaling it.** For our retained assistant stock, preserve each source
   identity and distinguish structural validity, answer verification, and actual environment success.
   Build a bounded, stratified manual audit before trusting a model-based quality score. Count
   accepted/rejected rows, total and supervised tokens, and domain coverage. Never set a blanket
   rejection percentage from another model's result.
2. **Make response style measurable.** Inventory direct answers and reasoning examples separately,
   including assistant reasoning-token share and length. Interleave ordinary assistance and tool
   examples in the first useful SFT recipe; the rehearsal's 50/50 row split is only coverage.
   Evaluate instruction compliance, correctness, and output length together. Treat any claim that
   reasoning supervision improves concise answers as a hypothesis requiring a matched comparison.
3. **Evaluate learned tool decisions.** Use `speck_tools_v1` for both training and inference.
   Extend our scripted environment checks with model rollouts, held-out task families, and explicit
   task-completion outcomes. Keep malformed calls, missing information, failed tools, and corrections
   in the score breakdown. A bounded local retrieval task is a possible later addition; live search
   requires its own provider, observation, judge, and cost contract.
4. **Cost a capability-focused continuation only after the pilot.** Consider a separate 4K phase
   mixing verified math/code/instruction data with broad replay, before SFT. Price its data supply
   and GPU-hours against simply extending base training. Keep this optional until our curves justify
   it. Longer context, lexical-difficulty curricula, FP8, and changing attention each need separate
   evidence; none belongs in the imminent rental rehearsal.
5. **Choose one scientific comparison.** A candidate is structurally valid SFT versus additionally
   verified SFT, from the same base, with fixed supervised-token budgets, domain/length strata,
   optimizer settings, and evaluation conditions. Report input-token cost and repetition as well.
   This tests the selection policy as a whole; it cannot isolate an abstract notion of quality.
   Freeze the question and affordable budget before viewing results; keep final tests untouched.

### Dataset reuse

The public card mixes full-text records with index-only locators, per-source terms, and sources
requiring separate access. The dataset is gated. Published token counts use GLM-5.1 tokenization;
they cannot be added to our Mistral-tokenized stock. The card's approximately 4.57M SFT rows also
differ from the report's 4,921,933; bind any future subset to its own immutable manifest.

Use its source inventory to investigate our measured supply bottlenecks, especially code. For any
candidate, first establish access and intended-use eligibility, then inspect a bounded sample,
retokenize, jointly deduplicate against retained sources, and run benchmark exclusions. Count only
materialized eligible tokens. A locator or already-held upstream document adds no new supply.
Prefer extending an already understood source when it resolves the same bottleneck.

The public [data pipeline README](https://github.com/zgcagi/ZGCM-1/blob/8c9677a03b1ae556f93b5cf6afa62b4bc0176d28/data-process/README.md)
describes a scaffold with representative adapters; tokenizer, quota, and indexed-writer integration
remain environment-specific. Borrow its traceable decision pattern where useful, rather than add a
second ingestion framework to Speck.

## OpenBMB web-data review — 2026-09-19

The [Ultra-FineWeb paper v1](https://arxiv.org/html/2505.05427v1), Table 4, reports
matched MiniCPM-1.2B runs at a nominal 100B tokens with zero-shot Lighteval:

| Benchmark | FineWeb-Edu | Ultra-FineWeb-en |
| --- | ---: | ---: |
| MMLU | 31.80 | 32.24 |
| ARC-C | 34.56 | 35.67 |
| ARC-E | 69.95 | 70.62 |
| CommonSenseQA | 31.53 | 36.45 |
| HellaSwag | 42.17 | 42.76 |
| OpenBookQA | 25.20 | 26.20 |
| PIQA | 72.14 | 73.67 |
| SIQA | 38.13 | 39.61 |
| WinoGrande | 55.56 | 55.80 |
| English mean | 44.560 | 45.891 |

All nine improve: **+1.331 percentage points** on average. The paper's printed deltas compare
against raw FineWeb, not FineWeb-Edu. Table 6's mixed-corpus English gain is smaller (+0.538),
with regressions; both English and Chinese sources change. Table 7's large-model results are
scaling predictions. These are publisher results, not a Speck replication or coding comparison.

The methodological lesson is to select classifier seeds using downstream learning results.
Their 10B-token verification uses 30% candidate / 70% background after a 1.1T-token base;
the quoted 110 H100-hours excludes making that base. Our 105M pilot is not an equivalent
foundation. Their classifier threshold is 0.5. Section 3.1 calls the budget 100B/104B, but its
printed product `4096 * 1024 * 26000` equals 109,051,904,000; retain “nominal 100B.”

The separate [Tiered Data Management paper v1](https://arxiv.org/html/2602.09003v1), Table 5,
reports English means of 52.26 / 53.36 / 53.96 for L1 / L2 / L3 under its verification protocol.
L3 improves the mean over L2 by 0.60, but ARC-C, ARC-E, BBH and PIQA regress. It does not
compare against Cosmopedia. Its different evaluation protocol prevents comparing these means
directly with the preceding table. Table 7 compares nominal 120B-token flat and staged mixtures:
overall 30.17 → 31.66 and code 7.25 → 9.70, with some commonsense regressions. This supports
testing a curriculum while retaining broad coverage. Its Code-L3 is a Stack-Edu rewrite;
do not identify it with the separately released UltraData-Code preview.

### Decisions for Speck

- **Prioritize natural Ultra-FineWeb qualification** for the main web component. Retain
  FineWeb-Edu as the control and DCLM as an independent comparator. Published matched evidence
  is sufficient to set this preparation priority; no new GPU replication is needed to begin it.
- Next prepare a bounded, comparable content/coverage audit using the
  [checked release pins](../experiments/corpus-audit/web-candidate-versions.json). Bind the English
  scored versus English HQ path explicitly. The archived scored-source cutoff was 0.8; quantify
  retained tokens and domain/length coverage before inheriting it. A stricter cutoff is a changed
  recipe, not an automatic improvement. Use released data first, preserving original text.
- Keep synthetic L3 source/answer checks and the checked-code comparison separate. Preserve
  source-family identity, joint deduplication and benchmark exclusions across candidate corpora.
- Price any later continuation comparison from a common competent base, counting base creation,
  preparation and evaluation. Use fixed background data and one changed component. Do not promise
  that a tiny from-scratch run will reliably rank sources, or copy their learning rate into our hybrid.

This changes preparation priority, not training admission or mixture weights. The frozen pilot and
current rental evaluation remain unchanged; they cannot resolve the web-corpus comparison.

## Evaluation references

[EvalPlus](https://github.com/evalplus/evalplus) supplies stronger execution tests for generated code.
[IFEval](https://arxiv.org/abs/2311.07911) measures verifiable instruction constraints. Use these as
starting references alongside checked math answers and a small deterministic tool environment.
The [pilot protocol](../experiments/pilot/evaluation.json) pins five dataset inputs, exclusions,
scorer revisions, and output limits. [Executable golden checks](evaluation.md) now pass for all five
graders, including 33 development code tasks and five scripted tool-environment episodes. A pinned
Qwen3-0.6B run exercises eight development tasks per benchmark. Those tiny denominators test the
pipeline and do not rank models. The broader dashboard and learned tool behavior remain open.

Start comparisons with MiniCPM5-1B and a pinned small Qwen release; use MiniCPM5-2B as a stronger
reference only when evaluation cost permits. Select exact revisions and compatible runtimes before
execution. Training benchmark answers or adapting recipes to final-test failures invalidates the
comparison. Earlier surveys and the long-context study remain available in [history](../archive/README.md).
