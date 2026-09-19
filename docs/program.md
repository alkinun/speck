# SpeckLabs first program: execution overview

2026-09-19. SpeckLabs' first model and paper center on data and training across pretraining,
mid-training and post-training: our own pretrained base and an always-thinking assistant for
agentic coding, normal coding, math and tools. This document explains the program; [PLAN.md](../PLAN.md) records
current status and work order. The [numeric plan](../experiments/main-data/plan.json) owns working
quantities, while experiment configs and verified receipts own actual execution and measurements.
No GPU run starts from this document.

## Decisions and open questions

| Status | Decision or question |
| --- | --- |
| Selected | 1.2B total/all-active KDA/GQA model with dense feed-forward layers; no MoE investigation or architecture sweep |
| Selected | Pretrain from scratch; preserve base and thinking-assistant releases; one thinking protocol with variable effort |
| Confirmed envelope | 5,000 total GPU-hours across four GH200s; access/site details remain unconfirmed |
| Working recipe | 35% code, 25% math, 40% supporting data; 320B desired / 400B stretch, subject to supply and cost |
| Working stages | Capability mid-training within the base horizon; approximately 128K context via an additional proposed 8B extension; 1.5M SFT conversations within 1–2M; RL conditional |
| Measured | H100 engineering pilot, development evaluation and backups complete; the 105M-token base is weak |
| Unresolved | Main eligible supply, GH200/distributed speed, final token horizon, long-context cost/quality and useful thinking/tool performance |

The future 50K-GPU-hour program is an ambition, not part of this allocation. Prior external rental
receipts are separate from grant consumption. The completed pilot is evidence to reuse, not the
next training job. The [model notes](model.md) explain the fixed backbone and supporting attention/size analysis;
[competitive strategy](competitive.md) defines how its release claims must be measured.

## Training lifecycle

The paper's main subject is how selected data and actual training develop useful capabilities.
Keep the selected architecture fixed so data comparisons are interpretable. Explain its attention,
size and measured trade-offs as supporting analysis. MoE, attention residuals and broader
attention-layout/model-size research belong to later releases with separately funded experiments.

| Stage | Purpose and objective | Budget ownership and readiness |
| --- | --- | --- |
| Pretraining | Broad code/math/general foundations using next-token prediction | Shares the 320B desired / 400B stretch base horizon and 2,300 hours with capability mid-training; main data/runtime not yet qualified |
| Capability mid-training | Targeted code/math/repair continuation with broad-data replay | A separately reported portion of that base horizon, not extra tokens/hours; split, objective and changed-data branch remain to qualify |
| Context mid-training | Learn to use coherent longer repositories/documents while retaining short tasks | Proposed additional 8B tokens / 800 hours; length stages and runtime unqualified |
| Post-training: SFT | Verified reasoning and complete tool trajectories with assistant-only supervision | Working 1.5M unique conversations; small 4K rehearsal complete, production data unqualified |
| Post-training: RL | Improve outcomes using checkable tasks, reliable rewards and policy rollouts | Conditional within post-training's 800 hours; trainer, verifiers and rollout integration remain to qualify |

Record each stage's parent, data families, unique supply, actual exposure/replay, objective, schedule
and cost. Preserve checkpoints at transitions. Share family exclusions across all stages, including
teacher generation and RL prompts. Stage progression and matched external evaluations describe the
resulting model; controlled data arms are needed to attribute a gain to a data intervention.

## Compute and allocation

Four GH200s, 5,000 total GPU-hours: at four continuously allocated GPUs this is 1,250 elapsed hours
or 52.1 days. The historical roughly 90-day access window is not 90 days of continuous four-GPU
compute. CPU acquisition/verification, disk and external teacher API costs need separate ledgers.
Release GPU allocations during CPU-only waits where the provider permits it; count every allocated
GPU, including idle devices. A trainer stop is not necessarily a provider billing stop.

The following expands existing reservations without increasing their sum. Post-training and
protected-reserve subdivisions are new planning proposals, not measured costs:

| Work | GPU-hours | Share | Four-GPU elapsed equivalent |
| --- | ---: | ---: | ---: |
| Runtime qualification, engineering pilot and one bounded data comparison | 211 | 4.22% | 52.75h |
| Pretraining and capability mid-training | 2,300 | 46% | 575h |
| Context mid-training | 800 | 16% | 200h |
| Thinking SFT | 500 | 10% | 125h |
| Conditional verified-reward RL | 200 | 4% | 50h |
| On-allocation teacher / verification work | 100 | 2% | 25h |
| Comparative and final evaluation | 300 | 6% | 75h |
| Recovery and unresolved-cost reserve | 589 | 11.78% | 147.25h |
| **Total** | **5,000** | **100%** | **1,250h** |

The first row is the existing 70 + 50 + 91 reservations, not a direction to repeat completed pilot
work. The ledger must distinguish prior external rentals from consumption of the new allocation.
The final two rows divide the existing protected 889 hours. Reallocate only after actual needs and
remaining commitments are known; this is a conservative preparation envelope, not a proven optimal
split. A 200-hour RL allocation must be enforced including generation, not merely gradient updates.
If long SFT consumes more than 500 hours, revise the post-training split; no promise of all stages
at their maximum data volume follows from this table. Unused reserve is not a requirement to spend.
Capability mid-training has no additional reservation: freeze its token/hour split within the base
horizon before launch. The 8B context extension remains separate. These labels do not add compute.

## Model and runtime

- Selected 1,195,884,576 total/active parameters; 24 layers, width 2048, dense SwiGLU
  intermediate width 5120 throughout. No routed experts or architectural comparison arms.
- Repeating three KDA layers plus one NoPE GQA layer: 18 recurrent and six global-attention layers.
- Tied embeddings; frozen Mistral tokenizer with 32,003 embedding rows including three role IDs.
- Starting optimizer: Muon/AdamW; BF16 activations with existing FP32 parameters/optimizer state.
- One full model replica per GPU using the existing DDP route at 4K, subject to GH200 qualification.
  Global tokens per update = workers × microbatch sequences × sequence length × accumulation.
  Choose microbatch/accumulation from measured efficiency; changing worker count must not silently
  change update size or learning-rate assumptions.

The [H100 receipts](../experiments/qualification/timing-result.json) and
[completed pilot](../experiments/pilot/h100-run.json) provide the starting measurements:

| Measurement | Observed result | Boundary |
| --- | --- | --- |
| Full pilot trainer | 12,859 tokens/s, including trainer startup, validation and saves | Excludes external scoring, setup, transfer and provider idle time |
| Steady pilot optimization | 13,595 tokens/s; 19.3 GiB peak allocated memory | One H100, eager deterministic execution |
| Microbatch-four probe | 16,398 tokens/s | Six steady observations; not sustained/restart-qualified |
| 4K SFT rehearsal | 13,493 padded positions/s; approximately 3,110 supervised tokens/s | Small repeated corpus; not a long-SFT forecast |
| Native decode | 55 tokens/s at batch one; 428 aggregate at batch eight | 1K prefix + 256 output tokens; capability backend measured separately |

Profile kernels, optimizer, activation checkpointing, compilation, input loading and communication.
The H100 pilot is a correctness/engineering baseline, not an optimized throughput ceiling. Do not
change precision or disable deterministic behavior without separately validating numerical and
restart behavior. Hardware topology and ARM64 dependencies require qualification.

Long-context training may require additional memory/sequence distribution beyond the current DDP
path. Four replicas do not pool sequence memory. Its implementation and communication cost are an
explicit feasibility gate if a long example cannot fit one device. No tensor/context-parallel
training has been qualified by the existing pilot.

## Pretraining

Start at **4,096 tokens**, using packed, qualified material and a frozen run manifest. Specify
packing/document-boundary semantics and verify them consistently across KDA and global attention;
do not silently claim repository coherence or document isolation from concatenated shards.
Keep the starting mixture at 35% code, 25% math, 30% natural web, 5% reference/science and 5% refined
educational web. The eight proposed banks are:

| Bank | Base-token share | Candidate source route |
| --- | ---: | --- |
| Natural code, tests and documentation | 30% | Stack-Edu and source-resolved UltraData-Code L2; practical and multilingual coverage |
| Checked code and repair examples | 5% | Qualified natural-code derivatives; UltraData-Code L3 held until lineage and independent checks pass |
| Natural math/worked solutions | 20% | UltraData-Math L2, FineMath 4+, Nemotron-CC-Math `4plus` |
| Refined math | 5% | Checked UltraData-Math L3/derivatives |
| Selected natural web | 25% | Natural Ultra-FineWeb English, with pinned selection threshold |
| Independent web coverage | 5% | FineWeb-Edu and/or DCLM baseline/DCLM-Edu |
| Reference/science | 5% | FineWiki, peS2o and qualified English FinePDFs-Edu |
| Refined educational web | 5% | Ultra-FineWeb-L3 candidate, with retained Cosmopedia as comparator |

These are proposed exposure weights, not source quality scores or accepted token counts. Natural
Ultra-FineWeb and synthetic Ultra-FineWeb-L3 are different sources. OpenBMB informs the research;
source selection still needs local quality, coverage, lineage and overlap evidence. Keep broad
language coverage alongside specialized material. [Data](data.md) and [coding](coding.md) retain
audit findings and acquisition gates; [research notes](research.md) distinguish publisher evidence.

Candidate releases are not automatically admitted supply. Freeze source eligibility, deduplication,
benchmark exclusions, source-family partitions, language coverage and actual token counts first.

Preparation targets 400B eligible unique tokens for 320B exposure (500B for the 400B stretch),
allowing selection headroom and normally one pass. Current retained pilot-source stocks total only
6.799B before joint eligibility, including 0.477B code tokens. A 16-file code audit or passing generated
tests cannot close that supply gap. Keep raw stock, qualified unique supply, exposure/replay and
rejected material as separate counts. If refined supply fails, explicitly revise within-domain
shares or shorten the horizon; do not silently repeat examples or assume unbudgeted teacher generation.

Within 2,300 GPU-hours, 12,859 effective tokens/s/GPU would process about 106.5B tokens; the 100B
scenario leaves limited margin. Effective rates of 20K, 30K, 40K and 48.3K correspond to approximately
165.6B, 248.4B, 331.2B and 399.9B. These are conditional arithmetic, not GH200 predictions. Rates
must divide aggregate useful throughput by all allocated GPUs and include base-stage overhead.
The desired 320B/400B targets require 38.6K/48.3K per GPU within this reservation.

Freeze a continuation-compatible learning-rate schedule, update size, cadence and horizon before
launch. Retain optimizer/loader/RNG state at selected milestones and before endpoint decay.
The pilot's learning rate and 800-step cosine schedule are not a production recipe. Use source
losses and task learning curves to choose the transition to capability mid-training within the
declared base horizon, with any revised weights recorded separately from the initial mixture.

## Mid-training

### Capability continuation

Develop code/math/repair capability using selected, independently checked material and broad-data
replay. Specify source-family overlap with pretraining, difficulty, language/task coverage and actual
token exposure. Start with the existing next-token objective; any prompt/patch masking or alternative
objective needs an explicit adapter and qualification. Plain token packing does not implement it.

The capability portion, mixture, replay share and LR/optimizer policy remain unfrozen. Allocate it
inside the 320B/400B base horizon and 2,300-hour reservation. The existing bank totals are initial
mixture scenarios; update them if a staged mixture changes aggregate exposure. Do not count the same
tokens as both pretraining and an additional mid-training allowance.

The loader supports predeclared mixture phases. Ordinary checkpoint branches require the same data
manifest; changed-data branches currently use the context-extension contract. A dedicated data
continuation path must be qualified before either new-data capability continuation or the bounded
code comparison. Do not relabel those runs as context extension to bypass the contract. See
[training](training.md#mid-training-readiness) for current implementation boundaries.

Separate capability changes from length changes when drawing causal conclusions. Keep source-wise
loss and short-task checks at both transitions. Context extension follows below; these two parts of
mid-training have distinct data questions and accounting.

### Context extension

Base 4K → up to 16K → up to 32K → toward 128K. Current proposed exposures are 2B + 2B + 4B tokens,
including short replay. Use 8K/64K probes or intermediate stages if memory, stability or learning
requires them; charge their tokens within the existing 8B envelope. Maximum length is a ceiling,
not a requirement to pad every record or train all eight billion tokens at 128K.

Keep 25% short replay as a working proposal. Coherent long records include repository files/tests/
documentation, intact technical/math documents and grounded multi-document tasks. The proposed long
portion is 50% repositories, 30% math/science/technical and 20% cross-document material. Compare
against unchanged preceding-stage domain weights repacked into longer records before adopting this
reweighting. The [MAI review](research.md#mai-thinking-1-review--2026-09-19) motivates that control;
its results do not establish the right mixture or extension cost for our model.

Each transition requires finite/stable training, memory/runtime and recovery qualification, improved
use of additional context, and acceptable short-task retention. Evaluate fixed-suffix loss as related
prefix grows, retrieval across positions/distances, multi-file repair and grounded reasoning.
NoPE global layers mean this is not a standard RoPE-rescaling recipe. Global attention still has
quadratic attention work and a length-growing inference cache, despite the recurrent layers.

Eight billion tokens in 800 hours needs an average 2,778 effective tokens/s/GPU. That has not been
measured. Freeze actual stage lengths/exposures after qualification; do not advertise 128K from a
configuration field alone. Preserve both the short-context and extended base identities for release.

## Post-training

SFT and the conditional RL stage below together constitute post-training.

### Thinking SFT

Start from our qualified base and target 1.5M unique accepted conversations (1–2M range): 600K code
reasoning, 375K math, 375K agent/tool trajectories and 150K supporting instruction/general thinking.
Agentic coding counts once, in the agent category. Existing 500K rows are unqualified starting stock.
UltraData thinking/agent sources and selected independent reasoning data are candidates; no held
record becomes training data merely to fill a quota.

| SFT task family | Target conversations | Share of rows |
| --- | ---: | ---: |
| Code reasoning | 600,000 | 40% |
| Math reasoning | 375,000 | 25% |
| Agent/tool trajectories, including agentic coding | 375,000 | 25% |
| Supporting general/instruction tasks with reasoning | 150,000 | 10% |

Row quotas guide acquisition; they are not token sampling weights. The retained 500K stock contains
220K UltraData-SFT-2605 thinking rows, 110K UltraData-SFT-Agent-2609 rows, 120K glaive reasoning
rows and 50K SYNTHETIC-2-SFT-verified rows. Its census and tiny rehearsal do not certify correctness.
Prioritize code/math answer checks, useful reasoning, tool-observation consistency and actual task
completion. Recover long tails filtered by the old acquisition limits separately. Selected SmolTalk2
reasoning components are additional candidates; Dolci/non-thinking components can provide tasks or
reference answers only until independently checked thinking supervision exists. The
[assistant plan](assistant.md) records source identities, restrictions and adapter behavior.

Inventory complete lengths <=4K, 4–16K, 16–32K and 32–128K. A proposed sequence first establishes
behavior in qualified shorter buckets (up to 16K), then mixes long repository/tool/reasoning records
with short replay through the validated ceiling. These are exposure stages drawn from one unique
inventory; repeated rows/tokens must be counted explicitly. Do not truncate tool observations or
reasoning solutions to fit a short-only preparation pipeline.

Supervise assistant reasoning, final answers and tool calls; mask user/system/tool-result context.
Train brief and deep reasoning with one protocol and no supported non-thinking toggle. Verify task
outcomes and consistency of reasoning/tool observations. One pass over 1.5M conversations averaging
8K/16K total tokens is 12B/24B processed context tokens, before padding/replay. Count supervised
positions separately. The measured 4K SFT rate is not a long-context throughput prediction, so the
500-hour subdivision needs validation against the actual length mixture and workflow overhead.

### Conditional RL

RL is planned, not implemented/qualified as a production pipeline here. The working candidate is
GRPO-style training with verifiable rewards, after a useful SFT baseline. The primary reference is
[DeepSeekMath](https://arxiv.org/abs/2402.03300); its results do not establish gains for Speck.
Use checked math answers and sandboxed executable code tests first; add repository/tool episodes
only when environments, reset/replay and rewards are reliable. UltraData-RL supplies candidate
problems/reference material, not ready-made successful rollouts.
Treat task difficulty, reference/test validity, source-family overlap and verifiable outcomes as
part of the data recipe. Retain rollout success/failure and rejection counts by task family; report
held-out transfer rather than relying on training reward alone.

Propose a 16K total-context ceiling initially, considering 32K only after measured benefit/cost.
Freeze prompt/output limits, sample-group size, tool-step caps, reward normalization, reference/KL
policy and update settings from a bounded rehearsal. No full 128K online-RL campaign is promised.
Reward correctness/task completion; format is a requirement, not a substitute. Check trivial reward
hacks, flaky tests, all-fail/all-pass groups, general-skill regressions and reasoning-length growth.

The 200 hours include policy rollouts, scoring/reference passes, updates, retries and stage checks;
record CPU environment costs separately. Distributed rollout/training layout depends on compatible
inference support for our hybrid, not an assumed engine integration. Qualify it before committing
hours. Small GPU counts favor testing a simple bounded layout before asynchronous infrastructure.
Select the RL model only if development gains justify it; retain the SFT assistant as a release
candidate if RL fails to improve it. Large specialist RL teachers/OPD are not funded commitments.

## Evaluation and release

Freeze contemporary comparator revisions and task identities before new training-data selection.
Use cheap source-wise validation at planned intervals, bounded development evaluations at selected
checkpoints, and untouched final tests once the recipe/model selection is settled. Evaluation is
scheduled by milestones; no frequent manual progress polling or repeated full suites.

- Base: source-held-out losses, standard language/code/math tasks and learning curves. Compare
  base checkpoints with base checkpoints; token-matched references complement final-model references.
- Assistant: executable code generation and repository repair, math correctness, instruction
  compliance and model-driven multi-step tools. Preserve standard benchmarks plus separately named
  practical held-outs. Tiny pilot subsets are engineering continuity, not headline full-benchmark scores.
- Context: positional retrieval, fixed-suffix loss, coherent multi-file/document reasoning and
  short-context regression checks, at each actually supported length.
- Efficiency: training throughput and total allocated hours; prefill/decode latency, memory and
  success per inference budget across relevant context lengths and batch sizes. Match hardware,
  precision, tool access and reasoning budgets where possible; disclose differences and uncertainty.

Primary contemporary assistant comparators are MiniCPM5-1B, LFM2.5-1.2B-Thinking and Qwen3.5-0.8B.
The [coding plan](coding.md) lists candidate benchmark protocols; expanded suites are not all
implemented. Charge comparator inference and final scoring to the evaluation reservation.

Release identified base and thinking checkpoints, tokenizer/configs, training/evaluation code,
source manifests and processing recipes, actual exposure and compute ledgers, curves, ablations
that were actually run, limitations and failed attempts. Keep source bytes private where their
terms prohibit redistribution. A first report can establish a reproducible quality/efficiency
tradeoff without claiming an untested architecture advantage or universal leadership.
Keep the bounded code intervention as the primary controlled data study. Other source audits guide
selection but do not become training ablations automatically; each extra arm needs a declared cost.

## Storage, recovery and operational gates

At uint16, 320B token IDs alone take 640GB decimal; the 400B eligible bank takes 800GB. The stretch
500B bank takes 1TB. Masks, indexes, raw text, deduplication databases, environments, checkpoints
and backups are additional. The numeric plan's 2TB scratch allowance is provisional, not a measured
full footprint. Acquire in bounded units, verify hashes and manifests, and avoid staging every raw
source simultaneously without a storage budget. Keep corpus/checkpoint payloads outside Git.

Freeze clean source, environment, tokenizer and input identities for each launch. Preserve complete
optimizer/loader/RNG state, verified copies and failure records before remote cleanup. Rebuild the
[GH200 bundle](gh200.md) from current committed code; the historical H100 bundle is not a current
release artifact. Qualify [Slurm](slurm.md) interruption/requeue and account for setup, failed work,
validation, saves, allocated idle time and transfers separately. No SFT scheduler requeue or
accelerated serving integration is assumed qualified merely because base training works.

## Execution gates and next work

1. **Now, on CPU:** use the completed practical-code checks and the
   [qualification packet](../experiments/main-data/QUALIFICATION.md) to finish benchmark/family
   exclusions, then audit web/math and assistant quality/long-tail supply in bounded packets.
   Prepare one selected-model hardware packet, not another architecture branch.
2. **On GH200 access:** qualify one worker, then four; measure sustained effective throughput,
   recovery and inference. Reconcile prior external rentals and unused reservation headroom.
3. **Before main training:** freeze actual admitted manifests, mixture, repetition, batch, optimizer
   schedule, checkpoint cadence and affordable horizon. Run the data comparison only when its
   common-base/supply/cost gates pass; its 91-hour reservation is not a second full pretraining run.
4. **After a useful pretraining checkpoint:** execute qualified capability continuation within the
   base horizon, then qualify context stages and short-task retention. Prepare and train verified
   thinking SFT. Attempt RL only with working verifiers, runtime and a bounded budget.
5. **Before release:** finish matched development comparisons, freeze the selected checkpoint/recipe,
   then score final tests and publish measured capability, efficiency and limitations with artifacts.

The [technical report outline](report.md) follows these same stages. Decision changes update this
outline, PLAN.md and affected numeric fields together; completed receipts remain immutable.
