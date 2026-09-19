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

The user subsequently raised MoE as the likely long-term direction. Prioritize the dense-FFN
versus interleaved-MoE comparison below; defer the all-GQA control unless separately budgeted.
For either selected control, first cost forward/backward, optimizer and inference with identical workload
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
Any new control or MoE candidate requires configuration, kernel/recovery qualification and a
measured cost before a paired run. Neither it nor an architecture variant is launched by this document.

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

## Dense–MoE investigation — September 19 clarification

Recommendation: cost one modest interleaved-MoE challenger before locking the flagship architecture.
Keep the existing dense-feed-forward hybrid as its control and fallback. This prioritizes learning
about routing for later allocations without assuming the sparse candidate must win. The earlier
all-GQA control is deferred, not a third simultaneous training arm. No launch is authorized by a
paper review or this parameter illustration.

Two independent choices must remain distinct: KDA/GQA chooses how tokens exchange information;
dense/MoE chooses the feed-forward computation within a block. Keep all 24 token mixers and their
positions unchanged for this comparison. Replace only selected SwiGLU feed-forward layers.
Dense interleaving supplies shared processing but does not establish routing stability. Always-on
shared experts inside a routed layer are a separate design, not equivalent to a periodic dense layer.
Neither mechanism excuses measuring expert starvation, load imbalance, router logits and gradients.

Primary references:

- [OLMoE v1](https://arxiv.org/html/2409.02060v1) demonstrates an open 7B-total/1B-active MoE,
  trained on 5T tokens. This supports investigating the active-capacity regime, not a quality or
  throughput prediction for our smaller training budget.
- [DeepSeekMoE v1](https://arxiv.org/html/2401.06066v1), sections 3.2–3.3 and 5.1.2, studies shared
  experts and balancing. It retains a dense first FFN because balancing converged more slowly there.
  This does not establish a universal two- or three-MoE/one-dense optimum.
- [Sparse Upcycling](https://arxiv.org/abs/2212.05055) demonstrates reuse of dense checkpoints for
  MoE initialization in its tested models. Starting dense does not make later MoE work impossible,
  but conversion, continued learning and optimizer handling are not free or qualified here.

### One illustrative candidate to cost

Use 18 routed FFNs and six dense FFNs (three routed then one dense per four blocks), width 2048,
eight experts per routed layer, top-2 token routing, expert intermediate width 2560; dense FFNs
retain width 5120. This realizes the suggested 3:1 pattern without increasing selected expert width
per token: 2 × 2560 = 5120. No shared experts are added in this first illustration. Layer placement
and routing settings remain unqualified; first-layer behavior is one explicit diagnostic.

For a bias-free SwiGLU, parameters = 3 × model_width × intermediate_width. Starting from
1,195,884,576 parameters, replacing these 18 FFNs gives:

- Router parameters: 18 × 2048 × 8 = 294,912.
- Total: 1,195,884,576 + 18 × (8 × 3 × 2048 × 2560 − 3 × 2048 × 5120)
  + 294,912 = **2,894,872,608**.
- Active per token: replace eight by two in that expression = **1,196,179,488**.
- BF16 weights alone: approximately **5.79GB decimal**, versus 2.39GB for the current model.

This is roughly 2.9B total / 1.2B active, not a 1.2B-total model. It compares similar selected FFN
arithmetic while purchasing more capacity, not identical storage or wall-time cost. Routing,
permutation, grouped operations, optimizer work and communication add costs. Memory, optimizer
state and checkpoints depend on total weights; data-parallel communication may also scale with
total experts. Smaller inference batches can leave experts poorly utilized. Do not reuse the
100B/320B timing projections without measuring this candidate.

### Existing implementation and decision gates

The repository already provides RoutedSwiGLUSpec, dropless token-choice RoutedSwiGLU, FP32 router
math, balancing/z losses, grouped-CUDA and reference expert paths, expert-bank optimizer support,
and per-layer routing/gradient diagnostics. Source inspection and unit-test presence do not qualify
this full-size topology, sustained convergence, GH200 kernels, distributed training or serving.
In particular, the existing grouped path casts whole expert banks; profile that overhead rather
than assuming runtime tracks only active parameters.

First qualify a single-GPU replica if memory permits, then data parallelism across four devices.
Keeping experts local avoids expert-dispatch all-to-all, but not gradient communication or replicated
state. Do not introduce expert parallelism merely because the eventual program will be larger.
Measure balanced and skewed routing, no-dropped-token coverage, gradients, restart/export/generation
parity, optimizer cost and memory before any real-data comparison. Test both training batches and
small-batch inference. Compare pure language-model loss separately from auxiliary routing losses.

Promote MoE only if measured learning and inference quality/cost justify its overhead within the
program budget. Otherwise ship the dense-FFN candidate and retain the MoE evidence for the next
allocation. No guarantee of dense-equivalent quality, lower GPU-hours or greater stability follows
from sparse activation or interleaving alone. Freeze one question, not a sweep over expert counts,
layer ratios, shared experts and attention variants.
