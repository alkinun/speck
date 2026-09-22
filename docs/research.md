# Research informing data and training

Reviewed through 2026-09-20 using primary papers and official releases. These are lessons and
comparisons, not a promise to reproduce another lab's scores or compute budget. **The decisions live
in [PLAN.md](../PLAN.md); this file records what each source actually showed and what it changed
here.** A reviewed paper does not create another experiment arm, admit a source, or expand the
5,000-GPU-hour envelope. Attention and size choices take supporting context only: the architecture
study is deferred, so no review here motivates a backbone change.

OpenBMB is a primary ongoing reference for our data work. The
[data guide](data.md#code-priority-and-qualification) records the specific UltraData-Code evidence,
the bounded local inspection and the first comparison to prepare. At each data-recipe freeze, review
the [MiniCPM releases](https://github.com/OpenBMB/MiniCPM), the
[UltraData framework](https://arxiv.org/abs/2602.09003), source cards and released classifiers,
recording exact revisions, matched ablations, verification methods, token definitions, source
coverage and processing cost. That is a review practice, not a background monitor.

## Standing references

| Source | What it supports | Implication for Speck |
| --- | --- | --- |
| [Qwen3 report](https://arxiv.org/abs/2505.09388), §3, §4.5–4.7 | Broad pretraining precedes capability data and post-training; small models use response and on-policy distillation. Its 8B comparison favours distillation over direct RL. | Preserve general coverage while improving math/code. Start with supervised behaviour and verified teacher examples; the reported cost advantage is not assumed to transfer to 1.2B or our teachers. |
| [Qwen3.5-0.8B card](https://huggingface.co/Qwen/Qwen3.5-0.8B) | A compact model uses a repeating three-Gated-DeltaNet/one-attention layout, plus a vision encoder. | Hybrid layouts are a relevant comparison, but GDN is not KDA. Similar ratios establish no quality, kernel parity or equivalent training cost. |
| [MiniCPM paper](https://arxiv.org/abs/2404.06395), §4 | Warmup-stable-decay separates continued learning from endpoint decay and supports reusing pre-decay checkpoints. | Retain continuation checkpoints and cost short pilot runs before a long horizon. Its numerical learning rates use its own parameterization. |
| [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) / [MiniCPM5-2B](https://github.com/OpenBMB/MiniCPM) | Staged base/mid/post-training with SFT, RL teachers and on-policy distillation; 200B deep-thinking plus 200B hybrid-thinking SFT tokens, and 400B SFT tokens at 2B. | The closest near-size capability reference. Its post-training scale warns against treating published quality as attainable with a fraction of that training. Measure our own recipe and count teacher work. |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | Broad pretraining plus systems work, SFT and RL; 14.8T tokens and 2.788M H800 GPU-hours. | Learn staged capability development and cost accounting. Its MoE/MLA/FP8 stack and totals are not a drop-in recipe at our scale. |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948), §2.3–2.4, §4 | Cold-start examples fix readability/language problems; verified rewards improve reasoning; distilled small models use curated teacher samples. | Prefer a measurable SFT/distillation baseline before building RL infrastructure. Better math scores establish no reliable tool use or general quality. |

## Post-training research synthesis — 2026-09-20

Full review: [post-training-research.json](../experiments/main-data/post-training-research.json),
covering MiniCPM5, Qwen3, GLM-4.5, Kimi K1.5/K2, DeepSeek-R1, Phi-4-reasoning, OLMo 3, SmolLM3,
Nemotron Nano 2, on-policy distillation and DAPO. The common pattern is a staged pipeline: a
capability prior, small high-quality cold-start SFT, reasoning or agent distillation, specialist
verified-reward or preference training, then general repair or on-policy distillation. **The data
banks and their evidence requirements matter here more than published ratios or headline scores.**
GLM-4.5 and Kimi show why agent trajectories, verifiable rewards and non-verifiable rubric feedback
need separate inventories. DeepSeek-R1 and Phi-4-reasoning support the cheaper small-model sequence:
teach useful reasoning with curated SFT or distillation, then use outcome-based RL as a finisher.
OLMo 3 and SmolLM3 show the value of releasing intermediate stage data and checkpoints.

None of this justifies copying their teacher sizes, token counts, reward models or compute. Our
executable sequence remains structural SFT audit, independent outcome checks, tool replay, then
fixed-policy verifier feasibility. Teacher generation, rejected rows, verifier work, rollouts and
policy updates stay in separate ledgers.

**On reasoning budgets:** the evidence supports *studying* reasoning control, not shipping a
low/medium/high interface. Qwen3 exposes think/no-think and a budget; MiniCPM5 exposes a think
switch; Kimi K1.5 warms up its length penalty; DAPO uses a soft overlong region; Kimi K2.5 reports
length overfitting under rigid budgets. A requested cap is an inference control; a learned budget
policy needs explicit training examples and held-out mode-following tests. Speck keeps the
always-thinking release baseline, and a bounded evaluation on a useful SFT parent may compare
always-thinking, a trained hybrid mode and three task-matched budget buckets. Promotion requires a
quality/token and quality/latency frontier with no mode, format, truncation or long-task regression.
In RL, correctness stays primary and any efficiency preference is delayed, soft and task-conditioned:
a global reward for shorter chains can reward guessing and incomplete tool work.

## Frontier data engineering synthesis — 2026-09-20

Structured review: [frontier-data-research.json](../experiments/main-data/frontier-data-research.json);
its mapping to pinned local audits is
[frontier-data-source-mapping.json](../experiments/main-data/frontier-data-source-mapping.json), an
evidence index that closes no gate by itself. Primary sources are
[GLM-4.5](https://arxiv.org/abs/2508.06471), [MiMo-7B](https://arxiv.org/abs/2505.07608),
[Kimi K2](https://arxiv.org/abs/2507.20534) and [DeepSeek-V4](https://arxiv.org/abs/2606.19348).

The consistent shift is from a static corpus to a stage-conditioned data curriculum: GLM-4.5 runs
general pretraining then repository, synthetic reasoning and long-context/agent mid-training; MiMo-7B
raises math/code to about 70% then adds about 10% synthetic responses; Kimi K2 does source-grounded
rephrasing with fidelity checks and at most two rephrasings per corpus; DeepSeek-V4 trains context
progressively over a 32T/33T-token base. **These are evidence for experimental axes, not portable
mixture settings** — their token counts, scales, teachers and source permissions differ too much.

The practical implications, now centralized in the
[stage-conditioned data-design contract](../experiments/main-data/data-design-contract.json):

- Keep a mostly natural, quality-weighted core; separate source-grounded rewrites, generated
  reasoning, repository-event sequences and agent trajectories by lineage and stage.
- Make extraction fidelity a first-class math/code gate. Generic HTML/PDF cleaning removes the
  equations, code blocks and forum structure that carry the intended signal.
- Represent repository training as linked files, issues, reviews, pull requests, commits, diffs and
  tests, with loss masks and family boundaries separating context from targets.
- Measure useful dependency distance for context stages; long documents alone establish no
  long-range learning signal.
- Use cheap proxy or bounded recipe experiments to choose quality weights, then confirm at matched
  exposure with held-out families.
- Keep benchmark decontamination separate from exact, fuzzy and semantic deduplication.

## MidTool review — 2026-09-20

[MidTool](https://arxiv.org/abs/2608.20314) trains Qwen3-4B/8B-Base on a 20.3B-token mixture of web,
PDF, code and tool artifacts, then holds the downstream SFT/RL recipe fixed while varying the
mid-training corpus. Processed source data alone improves tool-use outcomes over no mid-training;
context-grounded and native-trajectory branches help different endpoints; **only the combined mixture
improves every reported metric.** It also reports lower SFT loss and faster early RL adaptation after
mid-training, which makes downstream quality per SFT/RL token and per GPU-hour a first-class outcome.

It establishes no universal mixture and no intelligence-per-FLOP law for us: 32 H200s, 8 B200s,
strong teachers, a tool-specific corpus, weak deep-search transfer. We borrow the causal structure,
validation requirements and efficiency measurements — not the ratios, teachers or compute. Inside
the existing 360-hour mid-training research reservation this sharpens the question rather than
adding an arm: compare a replay/source-only control against targeted grounded material and, only
where environments and validation qualify, executable trajectory data, holding SFT/RL data,
optimizer policy and evaluation identities fixed. Report downstream capability versus mid-training
tokens and GPU-hours, SFT convergence, early RL adaptation and held-out transfer, counting source
preparation, teacher generation, validation, retries and discarded trajectories. Context extension
stays separate: a longer sequence is a systems and representation change and cannot carry a
data-mixture attribution.

## Cagliostro v3 review — 2026-09-20

The [model card](https://huggingface.co/bench-labs/cagliostro-v3/tree/176bf92e5d6ff5be887309d675aafa47c61747e6)
reports a 146M English base at 72.745B of a planned 75B tokens, on one RTX 5090 for about nine days,
with warmup-stable-decay and a mixture change at 85% of the run. Stable shares are 43.7%
FineWeb-Edu, 28.3% DCLM, 16% Cosmopedia v2, 5% FineMath 3+, 3% OpenMathInstruct-2, 2% InfiWebMath 3+
and 2% SmolTalk; at cooldown they become 37%, 5%, 25%, 15%, 13%, 0% and 5%. **Math rises from 10% to
28%** while broad web and DCLM fall, and no source exceeds 0.4 epochs. The card usefully separates
held-out validation loss from a temporary accuracy plateau.

This is a hypothesis, not a result to copy: mixture, learning-rate cooldown and checkpoint position
change together, so it cannot isolate the late math reweighting. Test a phase-conditioned mixture
only as a predeclared replacement or later ablation inside the 700-hour pretraining study, with
schedule, source exposures, validation mixture and total tokens fixed. **It must not become a fourth
screening arm.** The card exposes three candidates — [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2),
[InfiWebMath 3+](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/main/infiwebmath-3plus)
and [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) — each needing the same
source-use, family/contamination, correctness, finite-supply and runtime gates. Do not count the
`smollm-corpus` tag as one undifferentiated corpus, and note that SmolTalk used in base pretraining
is not post-training supervision.

## OpenBMB web-data review — 2026-09-19

[Ultra-FineWeb paper v1](https://arxiv.org/html/2505.05427v1), Table 4, reports matched
MiniCPM-1.2B runs at a nominal 100B tokens with zero-shot Lighteval:

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

All nine improve: **+1.331 percentage points** on average. The paper's printed deltas compare against
raw FineWeb, not FineWeb-Edu; Table 6's mixed-corpus English gain is smaller (+0.538) with
regressions, and Table 7's large-model results are scaling predictions. Their 10B-token verification
uses 30% candidate / 70% background after a 1.1T-token base, and the quoted 110 H100-hours excludes
building that base — our 105M pilot is not an equivalent foundation. Their classifier threshold is
0.5. §3.1 calls the budget 100B/104B while its printed product equals 109,051,904,000; retain
"nominal 100B". The separate [Tiered Data Management paper](https://arxiv.org/html/2602.09003v1)
reports English means of 52.26 / 53.36 / 53.96 for L1/L2/L3 under a different protocol, so those
means are not comparable with the table above; L3 gains 0.60 over L2 while ARC-C, ARC-E, BBH and
PIQA regress. Its Table 7 compares nominal 120B flat versus staged mixtures, 30.17 → 31.66 overall
and 7.25 → 9.70 on code. Its Code-L3 is a Stack-Edu rewrite, not the UltraData-Code preview.

**Decisions:** prioritize natural Ultra-FineWeb qualification for the main web component, with
FineWeb-Edu as the control and DCLM as an independent comparator — published matched evidence is
enough to set this preparation priority without a GPU replication. Prepare a bounded comparable
content/coverage audit using the
[checked release pins](../experiments/corpus-audit/web-candidate-versions.json), binding the English
scored versus English HQ path explicitly; the archived scored cutoff was 0.8, and a stricter cutoff
is a changed recipe, not an automatic improvement. Keep synthetic L3 source/answer checks separate,
and preserve source-family identity, joint deduplication and benchmark exclusions across candidates.

## Marin corpus review — 2026-09-19

Reviewed upstream commit `d2d97e888ce1a59b0e344bfbaca9cc4871f4148a`; static checkout at
`/mnt/speck-data/speck/research-reviews/marin-20260919`. No upstream code or corpus was executed or
admitted. Its source pool describes acquisition recipes, not a ready-made corpus. The
[8B retrospective](https://github.com/marin-community/marin/blob/d2d97e888ce1a59b0e344bfbaca9cc4871f4148a/docs/reports/marin-8b-retro.md)
starts at 92.6% DCLM / 6.1% StarCoderData / 1.3% ProofPile 2 and changes the recipe in later stages.
Its cooldown comparisons found **lower validation loss from supposedly higher-quality data could
accompany worse task performance**, and task-formatted data helped in some mixtures but
underperformed alone. Those are stage-dependent results, not universal ratios.

**Useful next work:** qualify Stack v3 as a scalable natural-code candidate — Marin's importer pins
`HuggingFaceCode/stack-v3-train`, retains commit/file/licence metadata and groups files by directory,
which addresses metadata gaps in retained Stack-Edu without establishing tests, dependency order or
eligibility. Separate source selection from serialization: compare eligible source alternatives with
fixed background data, tokenizer, serialization, horizon and settings, testing repository grouping
separately. Use useful parents for mid-training comparisons; short cooldowns can screen targeted
data but do not replace fresh-seed pretraining comparisons, and
[Delphi](https://openathena.ai/blog/delphi/) illustrates the limits of short-run extrapolation.
Bind processing changes to consumed artifacts: the
[32B retrospective](https://github.com/marin-community/marin/blob/d2d97e888ce1a59b0e344bfbaca9cc4871f4148a/docs/reports/marin-32b-retro.md)
reports cached GSM8K test contamination surviving a preprocessing fix, plus correlated shuffling, so
check exclusion versions against actual packs and inspect observed source order. The subsequent
[feasibility probe](../experiments/corpus-audit/stack-v3-feasibility.json) completes the
revision/schema check and selects the corrected release.

## MAI-Thinking-1 review — 2026-09-19

[Microsoft report](https://microsoft.ai/pdf/mai-thinking-1.pdf), §2.4–2.6, §3.2–3.3, appendices
B.4/C. Local receipt `/mnt/speck-data/speck/literature-reviews/mai-thinking-1-20260919/report.pdf`,
SHA-256 `a267d745b1eb3792a8abf58e71e204a6f44f9c18eeb5ad8671320deae71986bd`. A 35B-active/~1T-total
model on 30T natural pretraining tokens; Table 5 assigns 54.6% exposure to code and separates unique
supply from repeated exposure, and its mixture experiments show **rankings can change with scale and
horizon**. Appendix B.4 includes repository files, commits and pull requests with pre-change file
context loss-masked for patches. §3.3 verifies repair environments with empty-patch failure,
golden-patch success and repeated checks, filtering ambiguous task statements. Appendix C finds
repacking the preceding mixture sufficient in its experiments, evaluating fixed-suffix loss and
positional retrieval.

**Decisions:** keep 35% code / 25% math as an initial hypothesis, qualifying useful diversity within
each bank and reporting unique tokens, exposure and repetition separately — do not copy another
model's percentages or treat our 105M-token pilot as a mixture-ranking study. Extend code
qualification beyond isolated files to source-linked changes and executable repair cases as described
in [the data guide](data.md#code-priority-and-qualification), counting full processed context
separately from loss-bearing targets; that needs an explicit adapter/objective contract before
admission, since pasting raw patches into a pack is not equivalent. Add unchanged-domain repacking
as the context-extension control, plus fixed-suffix loss and position-stratified checks, leaving
stage tokens unset under the 360-hour mid-training research cap.

## ZGCM-1 review — 2026-09-18

[Report v1](https://arxiv.org/html/2609.13356v1) and its gated dataset/model cards; no weights,
corpus or upstream implementation were imported or executed. A 7.39B model with sliding-window/global
attention, 4.19T pretraining tokens, ~600B mid-training tokens and 256K context — a far larger
program than ours, with attention that is not KDA. Qualifications that matter more than its headline:

- Table 10's SFT filtering cuts ~2.08M examples to 1.145M for a six-task mean of 67.78 → 68.83, but
  **HumanEval+ falls 73.78 → 67.56** and IFEval 72.64 → 71.94.
- §3.1.3's 4.2× efficiency claim multiplies component estimates; §2.2 reports 1.13× SWA/full
  throughput at 4K and does not compare against our KDA implementation.
- Table 6 implies 192 H100s for general pretraining, not a GPU-hour total.
- §5.1 allows 258,048 generated tokens and averages pass@1 over 32 runs; this is not best-of-32.

**Decisions:** audit supervision before scaling it — preserve each source identity and distinguish
structural validity, answer verification and actual environment success, with a bounded stratified
manual audit before trusting any model-based quality score, and never set a blanket rejection
percentage from another model's result. Make response style measurable by inventorying direct
answers and reasoning examples separately, including reasoning-token share and length. Evaluate
learned tool decisions with model rollouts and held-out task families, keeping malformed calls,
missing information, failed tools and corrections in the breakdown. Its published token counts use
GLM-5.1 tokenization and cannot be added to our Mistral-tokenized stock; the card's ~4.57M SFT rows
also differ from the report's 4,921,933, so bind any future subset to its own immutable manifest.

## Practical conclusions

1. Quality and coverage of data, optimization, post-training and evaluation all matter. Keep one
   model candidate while establishing the pipeline; do not turn each paper into another sweep.
2. Distinguish broad pretraining, capability/context mid-training and SFT/RL post-training. Domain
   emphasis must preserve enough broad data and replay to retain ordinary usefulness.
3. Verified answers and executable code tests are useful supervision and evaluation signals. Include
   failed solutions, tool errors, corrections and unanswerable cases in a carefully checked recipe.
4. Count input and output tokens, teacher/generation effort, verification and failed attempts. Long
   reasoning can consume training and inference budget without improving answers.
5. Compare base with base and assistant with assistant. A parameter-matched public model can have
   vastly different data, compute, context and distillation history.

## Evaluation references

[EvalPlus](https://github.com/evalplus/evalplus) supplies stronger execution tests for generated
code; [IFEval](https://arxiv.org/abs/2311.07911) measures verifiable instruction constraints. Use
these alongside checked math answers and a small deterministic tool environment. The
[pilot protocol](../experiments/pilot/evaluation.json) pins five dataset inputs, exclusions, scorer
revisions and output limits, and [executable golden checks](evaluation.md) pass for all five graders
plus five scripted tool episodes. Those tiny denominators test the pipeline and do not rank models.
Start comparisons with MiniCPM5-1B and a pinned small Qwen release, using MiniCPM5-2B only when
evaluation cost permits; select exact revisions and compatible runtimes before execution. Training
benchmark answers or adapting recipes to final-test failures invalidates the comparison. Earlier
surveys and the long-context study remain in [history](../archive/README.md).
