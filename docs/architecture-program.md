# First flagship architecture: development and evidence

2026-09-19. SpeckLabs' first 5,000-GPU-hour program must develop and train its own architecture,
then post-train and release a competitive base/thinking model with an open technical report.
Reproducibility supports the flagship objective; it does not replace capability and efficiency.
This is a comparison brief, not an executed experiment or a launch configuration.

## Starting hypothesis

Use the current 1.2B-class KDA/GQA hybrid as the starting candidate, not an immutable final design.
Test whether its recurrent/global allocation preserves useful code/math learning while improving
training cost and inference cost at matched useful quality. Develop the layer allocation, state and
cache geometry, positional behavior and implementation from measured constraints. Select one
consequential change at a time; do not launch a grid across these dimensions.

KDA is inherited from [Kimi Linear](https://arxiv.org/abs/2510.26692v2), whose published system
combines KDA with MLA at a different scale. Speck's GQA hybrid does not inherit its measured gains.
Distinguish the original building blocks from our architectural choices, implementation and evidence.
Architectural development need not introduce a new named operator, but claims of novelty must name
what changed and identify the matching ablation. No attention ratio or 128K capability is proven by
our engineering pilot. Global-attention layers retain length-growing caches and quadratic attention
work even when most layers are recurrent.

## One affordable architectural comparison

Prepare a conventional all-GQA control alongside the current hybrid before choosing additional
variants. First cost its forward/backward, optimizer and inference paths with identical workload
shapes, precision and comparable implementation effort. A synthetic timing result establishes
runtime only; it cannot establish learning efficiency or quality at long context.

For a bounded learning comparison, use the same qualified tokenizer/data stream, partitions,
sequence lengths and evaluation tasks. Declare both total and non-embedding parameters and any
geometry changes needed to approach the candidate's capacity. Freeze the optimizer policy and
small tuning allowance symmetrically; a setting optimized for only one candidate confounds the
conclusion. Do not force identical learning rates merely to claim fairness if parametrizations differ.

Report equal-token checkpoints and equal-compute checkpoints as separate comparisons. They answer
different questions and will not generally coincide. Hold data and post-training recipe fixed when
attributing gains to architecture; later data/post-training comparisons need their own controls.
Short pilot curves are screening evidence, not proof that rankings persist to the flagship horizon.
The dense control itself requires configuration, kernel/recovery qualification and a measured cost
before any paired run. Neither it nor an architecture variant is launched by this document.

The existing 91-hour bounded-comparison envelope cannot fund a full architecture comparison and
an independent data study by assumption. Cost this architecture control as the first candidate for
that envelope, with the existing data comparison as an alternative. Freeze the selected question,
arm horizons and all-in ceiling before execution; any expansion must explicitly reallocate the
5,000-hour total. CPU source audits continue regardless. Protect sufficient budget for the actual
flagship run, context qualification, post-training and final evaluation.

## What efficiency means

| Claim | Required evidence |
| --- | --- |
| Learning per training token | Held-out loss and capability curves over processed tokens, with unique supply and repetition disclosed |
| Learning per training FLOP | Those curves over an architecture-aware FLOP estimate, with counting conventions and omitted operations stated |
| Practical training efficiency | Useful throughput and time/GPU-hours to the same quality target, including validation, saves and distributed overhead |
| Inference efficiency | Quality versus total generated tokens, latency, memory and estimated FLOPs on the same tasks/hardware; separate prefill and decode |
| Agent efficiency | Success under bounded output tokens, tool calls and elapsed time; include failed attempts and environment cost |
| Useful long context | Retrieval/repair/reasoning benefit versus context length, memory and runtime, with short-task retention |

Avoid presenting accuracy divided by tokens or FLOPs as a universal efficiency score. Use matched
budgets and cost to reach an agreed quality level; report absolute quality alongside every ratio.
Training tokens, supervised tokens and inference tokens are different quantities. Count all thinking
and answer tokens, failed rollouts and teacher/verification work in the relevant ledgers. For agent
cost per success, count the costs of failures too and report the success rate and stopping policy.

The common 6*N*D approximation is only a coarse reference here: global attention, recurrent/chunk
operations, vocabulary projection, recomputation and long-context workloads need explicit treatment.
Separate model FLOP estimates from hardware time and measured kernel work; low estimated FLOPs do
not guarantee fast execution. Specify multiply-add convention and training/inference coverage.

## Release and later scaling

Maintain two evaluations: controlled in-house comparisons for architectural attribution, and matched
product evaluations against strong contemporary small models for competitiveness. Winning one
comparison does not imply winning the other. Post-training must deliver useful thinking/tool behavior
while preserving the base's gains; show base, SFT and any reward-trained checkpoints separately.

The paper should explain why the final design was chosen, where it wins and loses, the true cost of
its development and final training, and what remains uncertain at larger scale. Preserve intermediate
checkpoints and planned learning curves for the future 50K-hour program. Do not infer parameter
scaling or a future MoA advantage from a single-size run. The flagship objective is ambitious;
its performance and efficiency claims remain conditional on measured results.
