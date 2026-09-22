# SpeckLabs first program: execution overview

2026-09-22. SpeckLabs' first model and paper center on the data pipeline across all six training
stages: our own pretrained base and an always-thinking assistant for
agentic coding, normal coding, math and tools. This document explains the program; [PLAN.md](../PLAN.md) records
current status and work order. The [numeric plan](../experiments/main-data/plan.json) owns working
quantities, while experiment configs and verified receipts own actual execution and measurements.
No GPU run starts from this document.

## Decisions and open questions

| Status | Decision or question |
| --- | --- |
| Fixed substrate, not tested | 1.2B total/all-active KDA/GQA model, frozen by declaration rather than selected over a control; the bounded attention-baseline comparison was deferred on 2026-09-21 to a later allocation with its design preserved; no MoE or size sweep |
| Selected | Pretrain from scratch; preserve base and thinking-assistant releases; one thinking protocol with variable effort |
| Confirmed envelope | 5,000 total GPU-hours across four GH200s, with a nominal 90-day access window; start date/site details remain unconfirmed |
| Working recipe | 35% code, 25% math, 40% supporting data; 80B 4K-base working horizon, subject to supply and measured GH200 cost |
| Working stages | 4K capability bridge, 16K repository reasoning and 32K long-horizon agentic coding within a combined 600-hour mid-training production reservation; 64K/128K deferred; 1.5M SFT conversations within 1–2M; RL and final self-SFT conditional |
| Measured | H100 engineering pilot, development evaluation and backups complete; the 105M-token base is weak |
| Unresolved | Main eligible supply, GH200/distributed speed, final token horizon, long-context cost/quality and useful thinking/tool performance |

The future 50K-GPU-hour program is an ambition, not part of this allocation. Prior external rental
receipts are separate from grant consumption. The completed pilot is evidence to reuse, not the
next training job. The [model notes](model.md) define the reference backbone, which is the fixed
substrate for this release; the [report outline](report.md#matched-external-comparison) defines how
its release claims must be measured against contemporary comparators.

## Training lifecycle

The paper's main subject is the data pipeline across all six stages, and how selected data and
actual training develop useful capabilities. The backbone is declared as the fixed substrate so data
comparisons are interpretable; it is not selected over a control. Explain the size choice using
parameter and compute accounting, and report reference-only implementation efficiency. The matched
attention contrast, MoE, attention residuals and broader attention-layout/model-size research all
belong to later releases with separately funded experiments.

| Stage | Purpose and objective | Budget ownership and readiness |
| --- | --- | --- |
| Pretraining data research | Screen feasible source/filter/mixture contrasts and confirm the strongest candidate before main training | 700 of 1,230 data-research hours; matched initializations, fixed endpoints and costed arms |
| Downstream data research | Compare repository, workflow, trajectory, context, SFT, RL and self-distillation data on appropriate parent checkpoints | 360 hours for mid-training and 170 for post-training; separate from production exposure |
| Stage 1 — Pretraining, stable phase | Broad code/math/general foundations using next-token prediction | 1,550 of the 1,800-hour base reservation; uses the revised 80B 4K-base working horizon; main data/runtime not yet qualified |
| Stage 2 — Endpoint decay | Late-stage capability enrichment under a declared decay schedule, with natural, curated and derived arms separately identified by lineage | 250 production hours plus a 180-hour three-arm, two-seed research study; needs a matched stable parent and a per-arm decay manifest |
| Stage 3 — Mid-training (capability, context, agentic) | Repository, repair, tool-use and long-horizon continuation with replay, grounded workflows and executable traces; learn coherent repositories, tool state and recovery while retaining short tasks | 600 production hours for the whole 4K/16K/32K sequence; 32K is the first release target; selective loss masks and changed-data/context branches remain to qualify |
| Stage 4 — Post-training SFT | Verified reasoning and complete tool trajectories with assistant-only supervision | 450 production hours; working 1.5M unique conversations; small 4K rehearsal complete, production data unqualified |
| Stage 5 — Post-training RL | Improve outcomes with verified rewards on checkable math and sandboxed code | 200 conditional production hours plus 40 research hours. **No RL trainer exists in this repository.** Trainer, rollout engine, verifiers and recovery are all unbuilt and gate both this stage and stage 6 |
| Stage 6 — Final self-distillation | Rejection-sample the promoted RL parent, verify and deduplicate, mix with verified anchor data, then run assistant-masked self-SFT | 100 production hours plus a 30-hour pilot; preserve the pre-self-SFT parent and promote only on held-out transfer |

Record each stage's parent, data families, unique supply, actual exposure/replay, objective, schedule
and cost. Preserve checkpoints at transitions. Share family exclusions across all stages, including
teacher generation and RL prompts. Stage progression and matched external evaluations describe the
resulting model; controlled data arms are needed to attribute a gain to a data intervention.
Audit datasets for every stage now. The starting-mixture comparison precedes main pretraining;
downstream SFT/RL and continuation comparisons use useful parent checkpoints before their stages.
Dataset inspection alone does not establish how well a model learns from a mixture.

## Compute and allocation

Four GH200s, 5,000 total GPU-hours: at four continuously allocated GPUs this is 1,250 elapsed hours
or 52.1 days. The historical roughly 90-day access window is not 90 days of continuous four-GPU
compute. The grant allocation is confirmed, but the access start date and site/runtime details are
not yet confirmed. CPU acquisition/verification, disk and external teacher API costs need separate ledgers.
Release GPU allocations during CPU-only waits where the provider permits it; count every allocated
GPU, including idle devices. A trainer stop is not necessarily a provider billing stop.

The research-and-release allocation is:

| Work | GPU-hours | Share | Four-GPU elapsed equivalent |
| --- | ---: | ---: | ---: |
| Hardware/runtime and reference efficiency qualification | 120 | 2.4% | 30h |
| Pretraining data research | 700 | 14% | 175h |
| Mid-training data research | 360 | 7.2% | 90h |
| Post-training data research | 170 | 3.4% | 42.5h |
| 4K base production, stable phase | 1,550 | 31% | 387.5h |
| Endpoint decay production | 250 | 5% | 62.5h |
| Capability/context/agentic mid-training production | 600 | 12% | 150h |
| Thinking SFT production | 450 | 9% | 112.5h |
| Conditional verified-reward RL production | 200 | 4% | 50h |
| Final self-SFT production | 100 | 2% | 25h |
| Production teacher / verification work | 50 | 1% | 12.5h |
| Comparative/final evaluation | 250 | 5% | 62.5h |
| Recovery and unresolved costs | 200 | 4% | 50h |
| **Total** | **5,000** | **100%** | **1,250h** |

The architecture and efficiency line is gone. Its 200 hours were released on 2026-09-21 when the
comparison was deferred to a later allocation: 180 hours to data research and 20 to reference-only
training and inference profiling, which that study used to own. The
[packet](../experiments/main-data/architecture-study-packet.json) is preserved unmodified.

Endpoint decay and final self-SFT now hold their own lines. Both were previously folded into a
larger stage total, which made them invisible in the accounting even though the release claims to
cover them as named stages.

Data research totals 1,230 hours; its 700/360/170 caps include experiment preparation, evaluation,
retries and allocated idle time. The [research design](../experiments/main-data/README.md#research-cost-envelopes)
proposes subcaps and run counts, protecting pretraining confirmation before screening. The numeric
plan owns these ceilings; exact launch costs remain to be measured.
The 800-hour post-training total (450 SFT, 200 RL, 100 final self-SFT, 50 teacher/verification)
and the 450-hour protected subdivision are planning caps, not measured requirements. Charge every job once; production-stage tokens do not include discarded research
arms. A 200-hour RL cap includes rollouts and scoring, not just gradient updates.

The completed H100 rental remains separate. Qualify 16K then 32K within the 600-hour combined
mid-training production cap, leave stage tokens unset, and treat 64K/128K as future work requiring an affordable
revision. No production stage borrows reserve automatically.
Only short-context training has a measured cost reference, and it describes the inefficient frozen
pilot recipe. The selected throughput recipe was chosen on a 318M proxy; the flagship rate is
measured on GH200 before any horizon is re-anchored, under the derate and surplus rules predeclared
in `compute.throughput_reanchoring_rule`. The backbone no longer changes under a study decision. Freeze stage costs and stop
conditions; unused budget is not a requirement to spend.

## Model and runtime

- Reference 1,195,884,576 total/active parameters; 24 layers, width 2048, dense SwiGLU
  intermediate width 5120 throughout. No routed experts. The backbone is fixed by declaration; the
  bounded matched attention-control study is deferred to a later allocation and is not run here.
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
Keep the starting mixture at 35% code, 25% math, 30% natural web, 5% reference/science and 5%
refined educational web. The mixture was re-frozen over **six banks** on 2026-09-22: `checked_code`
and `refined_math` held zero retained stock of any kind, so each capped a one-pass run at zero
tokens at its declared weight. Each dropped share moved to the natural bank of its own domain, so
total exposure stays 80B and the declared domain split is unchanged. Derived code and refined math
are now unbanked candidates; qualifying either needs a revised freeze and its own exposure ledger,
never a quiet pour into a natural bank.

The bank table, with candidate source routes and preparation targets, lives once in the
[main data plan](../experiments/main-data/README.md#base-mixture); `plan.json` owns the numbers it
renders. These are proposed exposure weights, not source quality scores or accepted token counts.
Natural Ultra-FineWeb and synthetic Ultra-FineWeb-L3 are different sources. OpenBMB informs the
research; source selection still needs local quality, coverage, lineage and overlap evidence. Keep
broad language coverage alongside specialized material. [Data](data.md) retains audit findings and
acquisition gates; [research notes](research.md) distinguish publisher evidence.

Candidate releases are not automatically admitted supply. Freeze source eligibility, deduplication,
benchmark exclusions, source-family partitions, language coverage and actual token counts first.

Preparation targets 100B eligible unique tokens for an 80B 4K-base working exposure,
allowing selection headroom and normally one pass. Current retained pilot-source stocks total only
6.799B before joint eligibility, including 0.477B code tokens. A 16-file code audit or passing generated
tests cannot close that supply gap. Keep raw stock, qualified unique supply, exposure/replay and
rejected material as separate counts. If refined supply fails, explicitly revise within-domain
shares or shorten the horizon; do not silently repeat examples or assume unbudgeted teacher generation.

Within the revised 1,800-hour 4K reservation, the measured H100 full-trainer rate projects to about
83.3B tokens. The 80B working horizon therefore leaves little margin before long-context and
agentic overhead. At an illustrative 80% of that rate, only 66.7B fits; the horizon must shrink if
GH200 qualification does not sustain the required throughput. Rates divide aggregate useful
throughput by every allocated GPU and include base-stage overhead. The former 100B/320B/400B scales
remain deferred comparisons, not current supply targets or promised throughput gains.

Freeze a continuation-compatible learning-rate schedule, update size, cadence and horizon before
launch. Retain optimizer/loader/RNG state at selected milestones and before endpoint decay.
The pilot's learning rate and 800-step cosine schedule are not a production recipe. Use source
losses and task learning curves to choose the transition to capability mid-training within the
declared base horizon, with any revised weights recorded separately from the initial mixture.

## Mid-training

### Capability continuation

Develop repository, code, math, repair and tool-use capability using selected, independently checked
material and broad-data replay. Add issues, reviews, pull requests, commits, diffs, dependency-linked
files, tool schemas, grounded workflows and executable trajectories only with lineage and outcome
evidence. Start with next-token prediction; any output/action-only masking needs an explicit adapter,
mask fingerprint and resume qualification. Plain token packing does not implement it.

The production sequence is 4K capability bridge, 16K repository reasoning and 32K agentic
continuation. The 4K stage teaches repository structure, repair and tool schemas. The 16K stage
teaches multi-file dependencies and issue/PR/commit context. The 32K stage teaches repeated tool
calls, test failures, recovery, summaries and state tracking. Best-fit packing preserves complete
reasoning and repository records. Environment observations remain context; valid model actions are
the supervised target.

The loader supports predeclared mixture phases. Data branches require the qualified data-continuation
contract; context branches are the only path that changes sequence capacity. Do not relabel a data
intervention as context extension. Keep source loss, tool validity, recovery, long-horizon coherence
and short-task retention at every transition. See [training](training.md#mid-training-readiness) for
the implementation boundary.

### Context extension

Base 4K → qualify up to 16K → up to 32K, within the combined 600-hour mid-training production
reservation. Token counts remain unset until useful-context learning and runtime are measured.
64K/128K is future work requiring an explicit affordable revision. Context-mixture comparisons use
the separate 360-hour mid-training research cap.

Keep 25% short replay as a working proposal. Coherent long records include repository files/tests/
documentation, intact technical/math documents and grounded multi-document tasks. The proposed long
portion is 50% repositories, 30% math/science/technical and 20% cross-document material. Compare
against unchanged preceding-stage domain weights repacked into longer records before adopting this
reweighting. The [MAI review](research.md#mai-thinking-1-review--2026-09-19) motivates that control;
its results do not establish the right mixture or extension cost for our model.

Each transition requires finite/stable training, memory/runtime and recovery qualification, improved
use of additional context, and acceptable short-task retention. Evaluate fixed-suffix loss as related
prefix grows, dependency-distance retrieval, multi-file repair, tool-state continuation, recovery and
grounded reasoning.
NoPE global layers mean this is not a standard RoPE-rescaling recipe. Global attention still has
quadratic attention work and a length-growing inference cache, despite the recurrent layers.

Freeze actual stage lengths/exposures after qualification. Do not advertise 128K from a configured
ceiling; longer records can be retained for future work without counting them as release exposure.

## Post-training

Thinking SFT, conditional RL and final self-SFT together constitute post-training.

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
rows and 50K SYNTHETIC-2-SFT-verified rows. The [data-readiness closeout](../experiments/corpus-audit/data-readiness.json) adds full structural
checks, tool-aware sampled lengths and cross-stage prompt links; none certifies correctness.
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
450-hour SFT subdivision needs validation against the actual length mixture and workflow overhead.

### Final self-SFT

Run a bounded pilot after SFT and optional RL. Start from the promoted RL checkpoint, or the selected
SFT checkpoint if RL is not promoted. Freeze a prompt pool, generate responses and tool trajectories
in pinned environments, verify outcomes, remove duplicates and rejected records, and mix the accepted
records with a verified anchor set. Compare anchor-only continuation against self-distillation with
anchor replay. Preserve the pre-self-SFT parent and promote the final checkpoint only when held-out
task transfer improves without unacceptable general, protocol, length or tool-call regressions.
Teacher checkpoint, prompts, decoding, environment images, tool schemas, verifier versions, masks and
rejections are part of the derived-data manifest. Self-generated text is never counted as natural
pretraining data.

Final self-SFT holds its own **100-hour** production line, separate from initial SFT's 450; both
are frozen after measured cost and the pilot decision. Generation and verification draw on the
**50-hour** teacher/verification subdivision. Neither is an extra budget beyond post-training's
800 hours.

### Conditional RL

RL is planned, not implemented/qualified as a production pipeline here. The fixed-policy feasibility
harnesses in [`speck/evaluation/rl_feasibility.py`](../speck/evaluation/rl_feasibility.py) and
[`speck/evaluation/rl_code.py`](../speck/evaluation/rl_code.py) now provide deterministic tool
episodes, test-backed code-task receipts, reset/replay accounting, transcript/task-suite fingerprints
and bounded diagnostic rewards; they perform no policy updates. Failed attempts remain visible and
cannot be hidden by a later successful retry. The working candidate is
GRPO-style training with verifiable rewards, after a useful SFT baseline. The primary reference is
[DeepSeekMath](https://arxiv.org/abs/2402.03300); its results do not establish gains for Speck.
Use checked math answers and sandboxed executable code tests first; add repository/tool episodes
only when environments, reset/replay and rewards are reliable. UltraData-RL supplies candidate
problems/reference material, not ready-made successful rollouts.
Treat task difficulty, reference/test validity, source-family overlap and verifiable outcomes as
part of the data recipe. The post-training research design includes a conditional fixed-policy
prompt/rollout feasibility slot; it does not promise an RL training-data ablation. Retain success,
failure and rejection counts by task family; report held-out transfer beyond training reward.

Propose a 16K total-context ceiling initially, considering 32K only after measured benefit/cost and
the 32K mid-training parent is qualified.
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
The [code section of the data guide](data.md#code-priority-and-qualification) lists candidate
benchmark protocols; expanded suites are not all
implemented. Charge comparator inference and final scoring to the evaluation reservation.

Release identified base and thinking checkpoints, tokenizer/configs, training/evaluation code,
source manifests and processing recipes, actual exposure and compute ledgers, curves, ablations
that were actually run, limitations and failed attempts. Keep source bytes private where their
terms prohibit redistribution. A first report can establish a reproducible quality/efficiency
tradeoff without claiming an untested architecture advantage or universal leadership.
Keep the pretraining recipe-selection study before main training as the primary controlled data
study. A code-data intervention is one candidate contrast after qualification. Other source audits
guide selection but do not become training ablations automatically; each extra arm needs a cost.

## What success means

| Capability | Evidence to collect |
| --- | --- |
| Code and agentic coding | Standard executable code tasks plus held-out repository repair, regressions and actual task completion |
| Math and general usefulness | Checked answers, source-wise loss, knowledge, instruction compliance and representative language tasks |
| Thinking and tools | Correctness versus reasoning budget, valid calls, use of observations, error recovery and bounded loops |
| Context | Positional retrieval, related-prefix benefit, multi-file/document reasoning and short-task retention |
| Efficiency | All-in GPU-hours, tokens/FLOPs to useful quality, prefill/decode latency, memory and cost including failed attempts |
| Openness | Identified checkpoints, source/config manifests, processing recipes, curves, costs, failure analysis and executable evaluation |
| **Pipeline coverage** | **The primary deliverable. For each of the six stages: a data manifest with closed gates, an exposure ledger, a budget line and a receipt. A stage missing any of the four is reported as not covered.** |
| **Pipeline reusability** | **Whether the recipe can be rerun at larger scale by someone else: pinned source identities, deterministic processing, family exclusions shared across stages, and costs recorded per stage and per token** |

Coverage is claimed per stage, never in aggregate. A stage missing any of the four artifacts is
reported as not covered, however good the resulting model is. [PLAN.md](../PLAN.md#stage-5-and-6-coverage-is-unclaimed)
records which stages currently lack a trainer and the decisions that gate them.

No architecture superiority claim follows from an engineering pilot without a matching control, and
no such claim is available at all on this allocation: the comparison is deferred and the backbone
was fixed by declaration. Release claims must describe measured capability and limitations. The
[report outline](report.md) defines the comparison protocol, the matched external comparators and
the release evidence.

## Storage, recovery and operational gates

At uint16, 80B token IDs alone take 160GB decimal; the 100B eligible bank takes 200GB. Masks,
indexes, raw text, deduplication databases, environments, checkpoints and backups are additional. The numeric plan's 2TB scratch allowance is provisional, not a measured
full footprint. Acquire in bounded units, verify hashes and manifests, and avoid staging every raw
source simultaneously without a storage budget. Keep corpus/checkpoint payloads outside Git.

Freeze clean source, environment, tokenizer and input identities for each launch. Preserve complete
optimizer/loader/RNG state, verified copies and failure records before remote cleanup. Rebuild the
[GH200 bundle](compute-qualification.md#entry-gate) from current committed code; the historical H100
bundle is not a current release artifact. Qualify [Slurm](slurm.md) interruption/requeue and account for setup, failed work,
validation, saves, allocated idle time and transfers separately. No SFT scheduler requeue or
accelerated serving integration is assumed qualified merely because base training works.

## Execution gates and next work

1. **Now, on CPU:** use the completed practical-code checks and the
   [qualification packet](../experiments/main-data/QUALIFICATION.md) to finish benchmark/family
   exclusions, then audit web/math and assistant quality/long-tail supply in bounded packets.
   Prepare the reference-model hardware packet. The architecture/control study is deferred.
2. **On GH200 access:** follow the [GH200 qualification runbook](compute-qualification.md):
   qualify one worker, then four, then the scheduler canary; measure sustained effective throughput,
   recovery and inference. Reconcile prior external rentals and unused reservation headroom.
3. **Before main training:** the backbone is already fixed by declaration, so go straight to data.
   Freeze data-study manifests, paired fresh initializations, controls, endpoints and
   screening/confirmation costs. Run within the 700-hour pretraining research cap,
   record selection or an inconclusive result, then freeze the main admitted mixture, repetition,
   batch, schedule, cadence and affordable horizon. No main-run launch precedes this decision.
4. **After a useful pretraining checkpoint:** execute qualified capability continuation within the
   separate 600-hour mid-training production cap, then qualify context stages and short-task retention. Prepare and train verified
   thinking SFT. Compare downstream data recipes on useful parent checkpoints before committing
   each stage, within its declared budget. Attempt RL only with working verifiers and bounded cost,
   then evaluate final self-SFT against the preserved selected SFT/RL parent.
5. **Before release:** finish matched development comparisons, freeze the selected checkpoint/recipe,
   then score final tests and publish measured capability, efficiency and limitations with artifacts.

The [technical report outline](report.md) follows these same stages. Decision changes update this
outline, PLAN.md and affected numeric fields together; completed receipts remain immutable.
