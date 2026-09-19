# First flagship architecture: development and evidence

2026-09-19. SpeckLabs' first 5,000-GPU-hour program must develop and train its own architecture,
then post-train and release a competitive base/thinking model with an open technical report.
Reproducibility supports the flagship objective; it does not replace capability and efficiency.
This records the selected design and qualification work, not a new launch configuration.

## Selected flagship architecture

The user selected the existing **1.2B-total, all-active model** for the first flagship release.
This supersedes the earlier dense-versus-MoE comparison proposal. MoE research, configurations,
qualification and comparison runs are outside this program; there is no sparse challenger to prepare.
The existing model configuration remains unchanged by this decision.

- 1,195,884,576 total/active parameters; 24 layers, model width 2048.
- Three KDA recurrent blocks followed by one global GQA block, repeated six times.
- Dense SwiGLU feed-forward layers throughout, intermediate width 5120; no expert routing.
- Tied embeddings and the frozen Mistral tokenizer, with 32,003 embedding rows including role IDs.
- Start at 4K context; extend toward approximately 128K only through measured qualification.

Dense here describes the all-active feed-forward computation. The KDA/GQA token-mixing hybrid
remains the selected backbone. Its recurring layers and global layers are not an MoE mechanism.
KDA is inherited from [Kimi Linear](https://arxiv.org/abs/2510.26692v2); distinguish inherited building
blocks from Speck's configuration, implementation and evidence. No attention ratio or 128K capability
is proven by the engineering pilot. Global layers still retain length-growing caches and quadratic
attention work. Architectural novelty or superiority requires supporting evidence.

## Work before and after compute access

Before access, prepare the single-model GH200 qualification packet: exact configuration and
input identities, workload shapes, numerical/restart checks, measurements and stop conditions.
Keep the completed H100 pilot as the engineering baseline. Do not prepare an architecture sweep.

On access, qualify one GH200 first, then the four-worker configuration. Measure batch/accumulation,
kernels, optimizer, memory, input/validation/checkpoint overhead and distributed communication.
Validate update semantics, finite gradients, restart, export and generation after implementation
changes. Measure inference prefill and decode separately, including small batches. Faster synthetic
steps establish runtime only; they do not establish better learning or long-context quality.

Then freeze the training horizon from measured effective throughput and qualified data supply.
The existing 91-GPU-hour bounded-comparison reservation remains for the natural-code versus
checked-exercise data study, conditional on useful common-base quality and sufficient eligible
supply. Charge both arms and their evaluations to that ceiling; no architecture arm shares it.
Do not automatically spend the full reservation or move it into base training without revising the
budget. Protect context extension, thinking post-training, evaluation and recovery within 5,000
aggregate GPU-hours.

The remaining choices concern the training recipe and implementation: learning rate/schedule,
update size, qualified data, context curriculum, and thinking/tool post-training. They do not reopen
model size or routing as a sweep. Any measured issue requiring a structural departure should be
reported explicitly before changing the selected architecture.

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
scaling advantages from a single-size run. The flagship objective is ambitious;
its performance and efficiency claims remain conditional on measured results.
