# Research informing data and training

Reviewed 2026-09-20 using primary papers and official releases. These are lessons and comparisons,
not a promise to reproduce another lab's scores or compute budget. The decisions live in [PLAN.md](../PLAN.md).
Use these references to inform pretraining, mid-training and post-training data and actual training.
Attention/size choices provide supporting context; MoE, attention residuals and broader architecture
research are deferred to later releases. A reviewed paper does not create another experiment arm.

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

## Post-training research synthesis — 2026-09-20

The detailed review is recorded in [post-training-research.json](../experiments/main-data/post-training-research.json). It covers MiniCPM5, Qwen3, GLM-4.5, Kimi K1.5/K2, DeepSeek-R1, Phi-4-reasoning, OLMo 3, SmolLM3, Nemotron Nano 2, on-policy distillation and DAPO. The common pattern is a staged pipeline: a capability prior, small high-quality cold-start SFT, reasoning or agent distillation, specialist verified-reward or preference training, then general capability repair or on-policy distillation. The data banks and their evidence requirements matter more to this program than published ratios or headline scores.

MiniCPM is the closest operational reference for our scale. Its public recipe separates reasoning/general SFT, specialist RL teachers and on-policy distillation, then merges the specialists. Qwen3 reports a large cost advantage for on-policy distillation over direct RL in its tested 8B comparison. GLM-4.5 and Kimi show why agent trajectories, verifiable rewards and non-verifiable rubric feedback need separate inventories. DeepSeek-R1 and Phi-4-reasoning support the cheaper sequence for a small model: teach useful reasoning with curated SFT or distillation, then use outcome-based RL as a finisher. OLMo 3 and SmolLM3 demonstrate the value of releasing intermediate stage data and checkpoints so the contribution of each stage can be measured.

These reports do not justify copying their teacher sizes, token counts, reward models or compute. Our first executable sequence remains structural SFT audit, independent outcome checks, tool replay, then fixed-policy verifier feasibility. Teacher generation, rejected rows, verifier work, rollouts and policy updates remain separate ledgers. No source is admitted by this review.

### Reasoning budgets and efficient RL

The evidence supports studying reasoning control, but not shipping a low/medium/high interface by default. Qwen3 exposes think/no-think and a thinking budget; MiniCPM5 exposes a think switch; Kimi K1.5 warms up its length penalty; DAPO uses a soft overlong region; Kimi K2.5 reports length overfitting under rigid budget constraints. A requested cap is an inference control, while a learned budget policy requires explicit training examples and held-out mode-following tests.

Speck therefore keeps the always-thinking release baseline. On a useful SFT parent, a bounded evaluation may compare always-thinking, explicitly trained hybrid mode, and three task-matched budget buckets. Promotion requires a quality/token and quality/latency frontier with no mode, format, truncation or long-task regression. For RL, correctness and safety remain primary; apply a delayed, soft, task-conditioned efficiency preference only after the verifier passes. Report Pareto curves, truncation and shortcut rates, and performance when the inference budget is increased. A global reward for shorter chains is not acceptable because it can reward guessing and incomplete tool work.

This changes the post-training questions and audit outputs inside the existing reservations. It does not add a training arm, alter the always-thinking contract, admit a corpus or expand the 5,000-GPU-hour envelope.

## Frontier data engineering synthesis — 2026-09-20

The supplied survey of DeepSeek, Kimi, GLM, MiniMax and MiMo reports a consistent shift from a
static corpus to a stage-conditioned data curriculum. The linked reports support the direction,
but their token counts, model scales, teacher systems and source permissions differ too much for
their percentages to be copied into Speck. The structured review is in
[frontier-data-research.json](../experiments/main-data/frontier-data-research.json).

The strongest lesson is to treat data as a schedule over source family, quality, representation,
dependency structure and transformation type. GLM-4.5 describes a general pretraining stage followed
by repository, synthetic reasoning and long-context/agent mid-training; MiMo-7B reports a three-stage
recipe that raises math/code to about 70% and later adds about 10% synthetic responses; Kimi K2
reports source-grounded rephrasing with fidelity checks and at most two rephrasings per corpus; and
DeepSeek-V4 reports progressive context training alongside a 32T/33T-token base. These are evidence
for experimental axes, not portable mixture settings. [GLM-4.5](https://arxiv.org/abs/2508.06471),
[MiMo-7B](https://arxiv.org/abs/2505.07608), [Kimi K2](https://arxiv.org/abs/2507.20534) and
[DeepSeek-V4](https://arxiv.org/abs/2606.19348) are the primary references.

The practical implications for Speck are concrete:

- Keep a mostly natural, quality-weighted core and separate source-grounded rewrites, generated
  reasoning, repository-event sequences and agent trajectories by lineage and stage.
- Make extraction fidelity a first-class math/code gate. Generic HTML/PDF cleaning can remove the
  equations, code blocks and forum structure that carry the intended signal.
- Represent repository training as linked files, issues, reviews, pull requests, commits, diffs and
  tests, with loss masks and family boundaries that distinguish context from targets.
- Measure useful dependency distance for context stages. Long documents alone do not establish a
  long-range learning signal.
- Use cheap proxy or bounded recipe experiments to choose quality weights and transformations, then
  confirm the selected recipe at matched exposure with held-out families.
- Keep benchmark decontamination separate from exact, fuzzy and semantic deduplication.

This research does not change the 35/25/40 working envelope, add a pretraining arm, admit a source,
or expand the 5,000-GPU-hour allocation. The next step is to map each finding to the pinned candidate
manifests and close only the source-use, family, correctness, supply and runtime gates supported by
primary evidence.

Those requirements are centralized in the [stage-conditioned data-design contract](../experiments/main-data/data-design-contract.json),
which the pretraining, mid-training and post-training packets now reference. This keeps the research
direction consistent across manifests without turning hypotheses into launch settings.
The [source-mapping receipt](../experiments/main-data/frontier-data-source-mapping.json) connects each
finding to the current pinned audits and records unresolved gate implications. No mapping entry closes
a source-readiness gate by itself.

## MidTool review — 2026-09-20

Reviewed [MidTool: Mid-training Data Synthesis for Agentic Tool Use](https://arxiv.org/abs/2608.20314), including its data construction, ablation and optimization analyses. The paper trains Qwen3-4B-Base and Qwen3-8B-Base on a 20.3B-token mixture spanning web, PDF, code and tool artifacts, then keeps the downstream SFT/RL recipe fixed while comparing mid-training corpora. Its mixture contains filtered source data, context-grounded augmentation and native executable trajectories.

The useful result for Speck is about data and optimization. In the fixed downstream comparison, processed source data alone improves tool-use outcomes over no mid-training; context-grounded and native trajectory branches help different endpoints; and the combined mixture is the only variant that improves all reported metrics. The authors also report lower SFT loss and faster early RL adaptation after mid-training, which makes downstream quality per SFT/RL token and per GPU-hour a first-class outcome. These results support testing mid-training as a capability prior, not merely as extra token exposure.

The paper does not establish a universal mixture or a direct intelligence-per-FLOP scaling law for Speck. It uses 32 H200s for mid-training, 8 B200s for RL, strong teacher models, a tool-use-specific corpus and a fixed downstream recipe. Its deep-search transfer remains weak, and its own limitations leave the interaction between mid-training and post-training open. We therefore borrow the causal structure, validation requirements and efficiency measurements, not its ratios, teachers or compute assumptions.

### Changes to the Speck study design

- Keep broad pretraining and capability mid-training within the same exposure accounting, but make capability mid-training a deliberate data intervention with a useful parent checkpoint.
- Compare a replay/source-only control with targeted grounded material and, only where environments and validation are qualified, executable trajectory data. Hold SFT/RL data, optimizer policy and evaluation identities fixed across these arms.
- Report downstream capability versus mid-training tokens and GPU-hours, SFT convergence area/steps to a fixed target, early RL adaptation and final held-out transfer. Count source preparation, teacher generation, validation, retries and discarded trajectories.
- Keep context extension separate from capability mid-training. A longer sequence is a systems and representation change; it cannot be used to attribute a data-mixture gain.

This adds a sharper question inside the existing 360-hour mid-training research reservation. It does not create a new production allowance, architecture arm or training admission.

## Cagliostro v3 review — 2026-09-20

The [Cagliostro v3 model card](https://huggingface.co/bench-labs/cagliostro-v3/tree/176bf92e5d6ff5be887309d675aafa47c61747e6)
reports a 146M-parameter English base model at 72.745B of a planned 75B pretraining tokens. The
checkpoint is still in its pretraining run: it has no instruction tuning, safety tuning or RL, and
its 2,048-token zero-shot results are not evidence for a post-training recipe. It uses one RTX 5090
for about nine days, with a warmup-stable-decay schedule and a mixture change when cooldown begins
at 85% of the run.

The card makes the phase change explicit. The stable mixture is 43.7% FineWeb-Edu, 28.3% DCLM,
16% Cosmopedia v2, 5% FineMath 3+, 3% OpenMathInstruct-2, 2% InfiWebMath 3+ and 2% SmolTalk.
During cooldown those shares become 37%, 5%, 25%, 15%, 13%, 0% and 5%, respectively. Math therefore
rises from 10% to 28% while broad web and DCLM fall; no source is intended to exceed 0.4 epochs.
The report also separates held-out validation loss from the temporary accuracy plateau, which is a
useful reporting practice for our own schedule and mixture studies.

This is a strong hypothesis for Speck, not a result to copy. The mixture, learning-rate cooldown
and checkpoint position change together, so the card cannot isolate the effect of the late math
reweighting. We should test a phase-conditioned mixture only as a predeclared replacement or later
ablation inside the existing 700-hour pretraining study, with the schedule, source exposures,
validation mixture and total tokens fixed. It must not become a fourth screening arm. The source
names also expose three candidates: NVIDIA's
[OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2), the
[InfiWebMath 3+](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/main/infiwebmath-3plus)
subset of FineMath, and [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk). The
source-readiness matrix now records a separately filtered InfiWebMath 4+ candidate with a bounded
arithmetic diagnostic; the 3+ viewer evidence, OpenMath and SmolTalk remain outside the matrix.
Every candidate needs the same source-use, family/contamination, correctness, finite-supply and
runtime gates; public presence does not admit it.

The card's `smollm-corpus` tag should not be counted as one undifferentiated corpus. Its named
components and their transformations need separate identities, overlap checks and exposure
accounting. SmolTalk used during base pretraining is also not post-training supervision. This keeps
the stage boundary clear while preserving the useful idea of small, high-value synthetic and
conversation-derived slices in a controlled mixture comparison.

## Practical conclusions

1. Quality and coverage of data, optimization, post-training, and evaluation all matter. Keep one
   model candidate while establishing the pipeline; do not turn each paper into another sweep.
2. Distinguish broad pretraining, capability/context mid-training, and SFT/RL post-training.
   Domain emphasis must preserve enough broad data and replay to retain ordinary usefulness.
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

## Marin corpus review — 2026-09-19

Reviewed upstream commit `d2d97e888ce1a59b0e344bfbaca9cc4871f4148a`, including source importers,
Datakit, and the 8B/32B retrospectives. Static checkout:
`/mnt/speck-data/speck/research-reviews/marin-20260919`. No upstream code or corpus was executed
or admitted. The [source-pool guide](https://marin.readthedocs.io/en/latest/reproducibility/pretraining-source-pool/)
describes acquisition recipes, not one ready-made public training corpus; some inputs need separate
hydration. A source catalog is not evidence of a particular model's actual mixture.

The [8B retrospective](https://github.com/marin-community/marin/blob/d2d97e888ce1a59b0e344bfbaca9cc4871f4148a/docs/reports/marin-8b-retro.md)
starts with 92.6% DCLM, 6.1% StarCoderData and 1.3% ProofPile 2; later stages change the recipe.
Its short cooldown comparisons found that lower validation loss from supposedly higher-quality
data could accompany worse task performance. Task-formatted data helped in some mixtures but
underperformed alone. These are stage-dependent results, not universal mixing ratios or evidence
that the full evolving training trajectory was a controlled data ablation.

The [current catalog](https://github.com/marin-community/marin/blob/d2d97e888ce1a59b0e344bfbaca9cc4871f4148a/lib/marin/src/marin/datakit/sources.py)
adds Nemotron-CC v2/v2.1 quality and synthetic subsets, HPLT, Common Pile, PDFs, Stack v3,
hydrated Nemotron code, math/STEM data and agent trajectories. Its token counts include estimates
and measurements with Marin's tokenizer; they are not Speck's eligible supply. Natural documents,
synthetic rewrites and task trajectories require separate lineage and evaluation treatment.

### Useful next work within the existing allocation

- **Qualify Stack v3 as a scalable natural-code candidate.** Marin's
  [importer](https://github.com/marin-community/marin/blob/d2d97e888ce1a59b0e344bfbaca9cc4871f4148a/lib/marin/src/marin/datakit/download/stack_v3.py)
  pins `HuggingFaceCode/stack-v3-train` at `bb2fa95033c00931906761bed7bc37b525155db6`, retains
  commit/file/license metadata, removes repeated file entries and groups files by directory.
  This addresses metadata gaps in retained Stack-Edu, but does not establish executable tests,
  dependency order or eligibility. Start with a bounded revision/schema/duplicate review and costed
  stratified sample. Check original versus redacted content identities and file-level source use;
  the [current dataset card](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train)
  includes unlicensed files and describes a v3.1 duplicate fix. Verify these at the chosen revision.
  Preserve the existing frozen audit and its unresolved outcomes; this candidate does not replace it.
- **Separate source selection from serialization.** Compare eligible source alternatives with
  fixed background data, tokenizer, serialization, horizon and training settings. Test repository
  grouping separately if affordable, measuring truncation and file-boundary behavior at 4K.
  Rank candidates by code/math outcomes and general retention alongside per-domain loss and cost.
- **Use useful parents for mid-training comparisons.** Short cooldowns can screen targeted data
  and task-format/replay interactions after base capability exists. They do not replace the planned
  fresh-seed pretraining comparisons. Confirm promising rankings at a longer affordable horizon;
  [Delphi](https://openathena.ai/blog/delphi/) illustrates the limits of short-run extrapolation.
- **Bind processing changes to consumed artifacts.** The
  [32B retrospective](https://github.com/marin-community/marin/blob/d2d97e888ce1a59b0e344bfbaca9cc4871f4148a/docs/reports/marin-32b-retro.md)
  reports cached GSM8K test contamination surviving a preprocessing fix, and correlated shuffling.
  Check exclusion versions against actual packs and inspect observed source order. Datakit's
  [reference pipeline](https://github.com/marin-community/marin/blob/d2d97e888ce1a59b0e344bfbaca9cc4871f4148a/experiments/datakit/README.md)
  separates reusable per-source work from global deduplication and changes store identity when
  the source set changes. Apply those principles through our existing pipeline.

This review adds a qualification candidate, not another training arm or infrastructure framework.
Keep the 700/360/170-hour data research split; the architecture study is deferred. No mixture weights,
production reservations, frozen evidence or training admissions change.

The subsequent [Stack v3 feasibility probe](coding.md#stack-v3-feasibility--2026-09-19) completes
the revision/schema check and selects the current corrected release for further qualification.
It records measured metadata/identity limitations; Marin's historical pin remains a reference.

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

## MAI-Thinking-1 review — 2026-09-19

Primary source: [Microsoft report](https://microsoft.ai/pdf/mai-thinking-1.pdf), focusing on
sections 2.4–2.6, 3.2–3.3 and appendices B.4/C. Local PDF receipt:
`/mnt/speck-data/speck/literature-reviews/mai-thinking-1-20260919/report.pdf`, SHA-256
`a267d745b1eb3792a8abf58e71e204a6f44f9c18eeb5ad8671320deae71986bd`.
This is a focused data/training review, not replication of all report results.

The report describes a 35B-active/~1T-total model trained on 30T natural pretraining tokens.
Table 5 assigns 54.6% exposure to code and separates unique supply from repeated exposure.
Its mixture experiments show rankings can change with scale and horizon. Appendix B.4 includes
repository files, commits and pull requests; pre-change file context is loss-masked for patches.
Section 3.3 verifies repair environments with empty-patch failure, golden-patch success and repeated
checks, while filtering ambiguous task statements. Appendix C finds repacking the preceding mixture
sufficient in its experiments; it evaluates fixed-suffix loss and positional retrieval. These results
come from a substantially larger model/program and different attention architecture.

### Decisions for our existing plan

- Keep 35% code / 25% math as an initial hypothesis. Qualify useful diversity within each bank and
  report unique tokens, exposure and repetition separately. Do not copy another model's percentages
  or treat our 105M-token engineering pilot as a mixture-ranking study.
- Extend the planned code qualification beyond isolated files: prepare source-linked changes and
  executable repair cases as described in [coding.md](coding.md#repository-change-data).
  Count full processed context separately from loss-bearing targets. This requires an explicit
  adapter/objective contract before admission; pasting raw patches into a pack is not equivalent.
- Add unchanged-domain repacking as the context-extension control, plus fixed-suffix loss and
  position-stratified task checks. Under the revised 360-hour mid-training research cap, leave stage tokens unset
  until measured qualification; 64K/128K is deferred. No context behavior is inherited from this paper.
- Preserve our supervised thinking baseline and bounded natural/refined comparisons. The report
  does not establish that abandoning synthetic data or starting our small model with RL is better.
  Its private processing pipeline is a research reference, not an acquired corpus.

These changes refine preparation and evaluation. They launch no experiments, alter no frozen pilot
records, and do not add compute to the confirmed 5,000-total-GPU-hour planning allowance.
