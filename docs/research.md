# Research informing the first baseline

Reviewed 2026-09-17 using primary papers and official releases. These are lessons and comparisons,
not a promise to reproduce another lab's scores or compute budget. The decisions live in [PLAN.md](../PLAN.md).

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

## Evaluation references

[EvalPlus](https://github.com/evalplus/evalplus) supplies stronger execution tests for generated code.
[IFEval](https://arxiv.org/abs/2311.07911) measures verifiable instruction constraints. Use these as
starting references alongside checked math answers and a small deterministic tool environment.
Dataset versions, exclusions, scoring, and output limits still need to be pinned before runs.
The current toolkit has loss, generation, serving, and export checks; the complete math/code/tool/
reliability dashboard is not implemented or claimed to have run.

Start comparisons with MiniCPM5-1B and a pinned small Qwen release; use MiniCPM5-2B as a stronger
reference only when evaluation cost permits. Select exact revisions and compatible runtimes before
execution. Training benchmark answers or adapting recipes to final-test failures invalidates the
comparison. Earlier surveys and the long-context study remain available in [history](../archive/README.md).
