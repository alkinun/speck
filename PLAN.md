# SpeckLabs: current decisions and next work

Updated 2026-09-23. This is the status and work order for the first flagship program.
Read the [program overview](docs/program.md) for the connected design, data, compute and release
outline. [Main-data plan.json](experiments/main-data/plan.json) owns working numeric targets;
experiment configurations and verified receipts own actual run settings and measured results.
Update these in place. Historical proposals and failures remain in Git and their original receipts.

## Goal and selected decisions

**The deliverable is an open data pipeline and recipe covering all six training stages, and a
1.2B model that proves the pipeline works end to end.** The stages are pretraining, endpoint decay,
mid-training, post-training SFT, post-training RL and final self-distillation from rejection-sampled
RL output. Each needs its own data manifest, exposure ledger, budget line and receipt; a stage
without all four is not covered, however good the resulting model is.

The goal is explicitly *not* a state-of-the-art 1.2B. It is a pipeline we can scale, since the
next allocation is expected to be several times larger and will fund architecture search and bigger
runs. Where a choice trades model quality against pipeline strength or openness, take the pipeline.

Release the base and a coding-centered generalist assistant with an open technical report. Primary
uses are agentic coding, normal coding, math reasoning and tools; general usefulness remains a
regression check. External models are comparators or qualified teachers, not our base
initialization.

- **Reference model:** 1,195,884,576 total/active parameters; 24 layers, width 2048; three KDA
  recurrent blocks then one global NoPE GQA block, repeated six times. Dense SwiGLU throughout,
  intermediate width 5120, tied embeddings, sigmoid KDA gates. **Fixed substrate for this release,
  frozen by declaration rather than selected over a control.** The bounded architecture/efficiency
  comparison is deferred to a later allocation with its design preserved; no MoE or size sweep.
- **Tokenizer fixed:** Mistral 32K; 32,003 embedding rows include three assistant role IDs.
- **Compute envelope:** 5,000 total GPU-hours across four GH200s, equivalent to 1,250 hours with
  all four allocated, within a nominal 90-day access window. The allocation is confirmed; access
  start timing, site details and GH200 throughput remain unconfirmed.
- **Behavior selected:** one always-thinking protocol with brief/deep reasoning and tool actions;
  no supported non-thinking response mode. Preserve separate base and assistant checkpoints.
- **Mid-training target:** move from a 4K capability bridge to 16K repository reasoning and 32K long-horizon agentic coding; the combined capability/context/agentic production reservation is 600 GPU-hours. 64K/128K is deferred.
- **Future ambition:** 50,000 GH200-hours and larger models; no future allocation is assumed funded.

The [model notes](docs/model.md) define the reference backbone. The architecture comparison was
deferred on 2026-09-21 and its 200 hours moved to data research and reference-only efficiency
profiling; the [packet](experiments/main-data/architecture-study-packet.json) is preserved intact
for the next allocation. **No architecture superiority or parity claim may appear in this report:**
the backbone was fixed, not compared. MoE, attention residuals and broad searches remain later work
alongside it. The first report centers on datasets and training, with reference-only
training/inference efficiency evidence. High efficiency and competitive quality are hypotheses to
measure. The [program overview](docs/program.md#training-lifecycle) defines stage boundaries.

## Working recipes, not launch settings

| Area | Preparation target | Still to resolve |
| --- | --- | --- |
| Pretraining | 80B 4K-base working horizon; initial 35% code, 25% math, 40% supporting material; explicit natural/curated/derived decay study | Qualified source supply and measured GH200 cost; 100B is deferred because the revised 1,800-hour 4K reservation must protect the expanded mid-training stages |
| Capability/mid-training | 4K repository/tool bridge, 16K multi-file repository reasoning and 32K long-horizon agentic coding with replay, grounded workflows and executable traces | Trajectory environments, selective loss masks, packing, changed-data continuation, context cost and useful transfer remain to be qualified |
| Thinking SFT | 1.5M unique qualified conversations, within a 1–2M range | Correctness, source-family deduplication, complete long examples and supervised/context token totals |
| Reward training | Conditional verifiable math/code rewards after useful SFT | Trainer, rollout integration, verifiers, recovery and affordable measured benefit |
| Endpoint decay | Three-way natural/curated/derived decay study on two paired seeds, 180 research hours; 250 production hours inside the 1,800-hour base reservation | A matched stable parent, a decay manifest per arm and separately identified derived lineage |
| Final self-distillation | Rejection-sampled from the promoted RL parent, verified and mixed with anchor data; 100 production hours plus a 30-hour pilot | A promoted RL or SFT parent, a frozen prompt pool, pinned environments and held-out transfer evidence |
| Data research | 1,230 hours: 700 pretraining, 360 mid-training, 170 post-training | Screening and confirmation before each production stage; useful parents for downstream comparisons |

The [main data plan](experiments/main-data/README.md) specifies candidate sources and counting rules.
Retained stock, admitted data and training exposure are different quantities. No small preview or
publisher quality label establishes flagship-scale supply. No automatic repetition fills a gap.

## The binding constraint is supply, not compute

[`supply-gap.json`](experiments/main-data/supply-gap.json) is derived from the plan and stock receipts by
`build_supply_gap.py` and checked by `make plan-check`, so it moves as acquisition proceeds and
cannot go stale. Today:

| Bank | Weight | Preparation target | Retained candidate stock | Coverage | One-pass exposure cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected_web | 25% | 25.0B | 0.351B | 1.41% | 1.40B |
| independent_web | 5% | 5.0B | 2.307B | 46.13% | 46.13B |
| natural_code | 35% | 35.0B | 0.477B | 1.36% | 1.36B |
| natural_math | 25% | 25.0B | 1.124B | 4.50% | 4.50B |
| reference_science | 5% | 5.0B | 1.402B | 28.04% | 28.04B |
| refined_web | 5% | 5.0B | 1.489B | 29.79% | 29.79B |
| **Total** | **100%** | **100.0B** | **7.150B** | **7.15%** | — |

Three facts follow, and they govern the order of work:

1. **Zero eligible tokens are established.** The 7.15B is retained stock with every gate still open.
   It is an upper bound on what the gates could admit, not usable data.
2. **No bank caps a one-pass run at zero any more.** The two that did, `checked_code` and
   `refined_math`, held zero retained stock of any kind and were **re-frozen out of the mixture on
   2026-09-22**. Each dropped share moved to the natural bank of its own domain, so total exposure
   stays 80B and the declared 35% code / 25% math / 40% supporting split is unchanged. Derived code
   and refined math are now *unbanked* candidates: qualifying either one later needs a revised
   freeze and its own exposure ledger, never a quiet pour into a natural bank.
3. **`natural_code` now binds at 1.36B — 1.70% of the 80B horizon.** The binding constraint moved
   from `selected_web`, because raising the code share to 35% divides the same 0.477B of retained
   code stock by a larger weight. Re-freezing created and destroyed no eligible token; it only
   moved where the horizon binds, and it moved it onto the bank whose supply is hardest to grow.

Closing the whole gap means roughly a **14x** increase in retained stock. That is acquisition,
filtering and tokenization work: CPU, bandwidth and storage, consuming **no grant GPU-hours**. It
can and should run before and during access, and it is why a throughput surplus buys nothing here.

**Source use is decided for all nine selected sources and closed on six.** The
[acceptance record](experiments/main-data/source-rights-acceptance.json) was signed on 2026-09-22 by
a named human authority over the nine sources the re-frozen mixture
[selects](experiments/main-data/source-registry.json), at a declared scope of research training,
paper publication and public model-weight release, with **no commercial use, no source-data
redistribution and no derived-shard redistribution**. Six sources closed outright; three carry a
recorded named decision with one specific mechanical dependency still open — origin and notice
recovery for both code routes, and authoritative peS2o v3 licence-selection documentation. Three
former candidates are recorded as **not selected**: `checked_code` and `nemotron_cc_math_4plus` with
their dropped banks, and `ultradata_math_l2_preview` because every one of its 169,058 retained rows
lacks host metadata, so no per-document attribution manifest can be produced for it at all.

Every approval carries conditions, and the conditions are the substance. The two code routes are
approved on a **per-file original-licence** basis with recovered notices, not on any dataset-level
grant: an unlabelled or unresolved record is excluded rather than assumed permissive. The remaining
44 unheld quota-blocked origin lookups stay out until evidence resolves them, and all recorded
content-family holds persist across every stage. The single record traced end to end carried a term its MIT label did not
express, which is why notice text governs and the scanner label does not.

The [joint candidate partitions](experiments/main-data/family-partition.json) now cover six text
stocks, including preprocessed UltraFineWeb-HQ, and the 218-file code review cohort. The frozen
rule assigns whole families to 90% train / 5% development / 5% final hash buckets; both legacy
firewall reference pools remain excluded. All 675 earlier text edges survive unchanged. A new
exact firewall match raises code cohort holds to 45. The full retained-code graph and intended
benchmark coverage still keep `family_partition` open. Raw supply bounds above precede these
holdout reservations and remaining eligibility gates.

`source_use` is open on three of nine selected sources; `family_partition` and `finite_supply` are
open on all nine. Source use being decided **admits nothing** — no source is admitted, no eligible
token exists, and no acquisition or training is authorized by that record.

## Completed evidence

| Work | What is established | Limits / evidence |
| --- | --- | --- |
| Runtime rehearsal | Full-size single-H100 base/SFT restart, loader/RNG recovery, numerical checks, generation and export | [H100 receipt](experiments/qualification/h100-result.json); ARM64 GH200, four-worker execution and scheduler recovery remain open |
| Engineering pilot | 800 steps / 104,857,600 tokens; 2.265 trainer hours, 12,859 full-trainer tokens/s, 13,595 steady optimizer tokens/s, 19.3 GiB peak allocated memory | [Execution receipt](experiments/pilot/h100-run.json); engineering evidence, not an optimized ceiling |
| Development scoring | 2,619 tasks in 2.12 evaluation hours: GSM8K strict 0/253, compiled code pass@1 0/33, IFEval strict prompts 12/101, ARC normalized 44/222, HellaSwag normalized 530/2,010 | [Results](experiments/pilot/development-result.json); weak base capability, custom subsets, no final-partition scoring |
| Backup closeout | Eight model/optimizer checkpoints, both exports, evaluation outputs and recovery logs retained locally | [Backup receipt](experiments/pilot/backup-result.json); pilot closed, no further rental work needed; provider billing/stop state not verified |
| Natural-code supply | All 1,999 retained archives reopened; 714,369 files / 476,774,847 retokenized tokens; 123 files under holds at census time | [Census](experiments/corpus-audit/code-supply.json), [lineage cohort](experiments/corpus-audit/natural-code-cohort.json); no new training admissions; stock covers 1.36% of the re-frozen 35B natural-code preparation target |
| Code bundle linkage | Four pinned repositories; 19 linked files / 5,650 tokens; four content flags hold 15 files by family | [Receipt](experiments/corpus-audit/code-bundles.json); static test weaknesses, no execution or admission; repository co-presence alone is insufficient |
| Code expansion feasibility | Upstream inventories pinned; one new Python metadata shard scanned, 16 blobs recovered, 13 length-matched / 9,541 tokens | [Receipt](experiments/corpus-audit/code-expansion.json); four content flags; four application modules selected for follow-up; no qualified yield or admission |
| Application origin/test review | All four source revisions, complete trees and MIT notices recovered; 29 response hashes verified | [Receipt](experiments/corpus-audit/code-application-origins.json); one direct but stale test link, no independent verification or admission |
| Stratified code preflight | 138 files / 459,615 tokens across 11 languages and 72 strata; exact offline replay | [Receipt](experiments/corpus-audit/code-yield-result.json); 30 content flags, 31 sample family holds; full eligibility and usable yield remain unresolved |
| Stack v3 preflight | All 16 frozen groups acquired; 29,347 repository rows / 379,942 entries; fixed 44-repository / 80-file cohort screened | [Receipt](experiments/corpus-audit/stack-v3-broader.json); four content flags, eight known-family holds; source use, quality and eligible yield unresolved; exact offline replay, no admission |
| Common code review | 176 complete texts / 412,974 tokens reviewed; 45 of 218 records now held; all 173 unheld records read (plus three held records) | [Reading closeout](experiments/corpus-audit/stylesheet-cohort-review.json), [current holds](experiments/main-data/family-partition.json); page/template/component roles and notice questions recorded; all holds preserved; no yield estimate or admission |
| Expanded data qualification | Practical CPU checks complete; 37 files screened against 22 benchmark lanes, with seven content flags and 22 files held after family propagation | [Qualification packet](experiments/main-data/QUALIFICATION.md); includes pinned LiveCodeBench v6 public text, not complete corpus admission |
| Stratified HQ web audit | Twelve pinned shards / 290,761 documents; 192 sampled, 24 reviewed; exact offline replay | [Receipt](experiments/corpus-audit/web-hq-stratified.json); high-score extraction defects and lower-score coverage candidates; no training admissions; later full token census below |
| Web extraction follow-up | Three matching archived captures; 32 fresh comparison documents, 125,453 sample tokens | [Receipt](experiments/corpus-audit/web-filter-validation.json); confirmed omissions/boundary issues; candidate flags remain review-only |
| Retained-data closeout | Full HQ token/overlap census, full math text/index reconciliation, 500K SFT format census and sampled tool-aware lengths, bounded RL inventory | [Receipt](experiments/corpus-audit/data-readiness.json); finite stock bounds measured; correctness, family eligibility and finite supply remain open |
| Joint family partitions | 5,492,692 candidate documents / 6,674,432,945 measured tokens; six text stocks and the code review cohort; 90/5/5 family buckets | [Receipt](experiments/main-data/family-partition.json); 45 code holds, full code inventory and benchmark coverage still open; no admission |
| Other data preparation | 6.799B retained pilot-source tokens before joint eligibility (7.150B with distinct HQ, as in the supply gap); 500K assistant rows inventoried; finite tool-aware SFT rehearsal | [Supply](experiments/pilot/supply.json), [assistant contract](docs/assistant.md); not main-run qualified supply |
| Training throughput pass | 2.084x over the frozen pilot recipe **on a 318M proxy**; proxy utilization 25.0% to 52.2%; selected configuration is checkpointing off, compiled with max-autotune, determinism retained, Liger loss | [Sweep](experiments/qualification/throughput-3090/sweep.json), [GH200 confirmation](experiments/qualification/throughput-gh200.json); RTX 3090 sm_86 only. **This is not a flagship speedup.** The only 1.2B point measured is the eager checkpointed baseline at 34.6% utilization, which establishes no optimized ceiling, and no checkpointing-off flagship configuration fit the 24 GiB card. Absolute rates, microbatch and checkpointing need GH200 measurement |
| Compiled restart parity | Coordinate-descent tuning found to vary kernel choice per process and removed; compiled base replay on the 318M proxy and compiled SFT replay on the 1.2B reference pass at the unchanged tolerance with exact RNG/loader state and optimizer parity | [Base](experiments/qualification/compiled-recovery-descent-3090.json), [SFT](experiments/qualification/compiled-sft-recovery-3090.json); one RTX 3090 worker, SFT with checkpointing on. About 0.5% proxy throughput cost. Hopper and four-worker compiled DDP remain unqualified |

The [timing measurements](experiments/qualification/timing-result.json) additionally cover saves,
validation, restart, SFT and prefill/decode. Microbatch four showed about 20% higher steady throughput
in a short probe; sustained operation and recovery need qualification before adoption.
Those pilot rates describe an inefficient configuration. The pilot ran eager, with activation
checkpointing, at microbatch one. Its full-trainer rate corresponds to about 9.8% estimated MFU;
the steady optimizer rate corresponds to about 10.3% under the same FLOP convention. The
[throughput pass](experiments/qualification/throughput-3090/sweep.json) measured 2.084x against
that recipe locally **on a 318M proxy, not on the flagship**. Treat 12,859 tokens/s as a property
of the frozen pilot, not as the machine's capability, and re-anchor the numeric plan only from
measured GH200 rates.

SFT padded positions/s are not supervised tokens/s. Earlier estimates and failed attempts remain
historical; do not repeat the completed pilot because a preparation document still contains commands.

## Immediate order of work

This is the one work order; other documents link here rather than keep their own. The data design
contract, candidate manifests and post-training audit protocol are recorded and described in the
[main data plan](experiments/main-data/README.md);
[`source-readiness.json`](experiments/main-data/source-readiness.json) is the one record of
per-source gates and blockers. None admits a source. Change a gate only on primary evidence; do not
reopen completed deterministic audits or infer eligibility from a paper's headline result, format,
length, classifier scores or notices. Extend the
[source-mapping receipt](experiments/main-data/frontier-data-source-mapping.json) whenever new
primary research evidence is added.

1. **Complete family partition and exclusion coverage.** Candidate splits exist for six text stocks
   and the code review cohort ([receipt](experiments/main-data/family-partition.json)). Close
   `family_partition` with the full retained-code graph and intended benchmark coverage before
   exercise derivation. Carry all 45 cohort holds into every stage. Resume the 44 unheld
   quota-blocked origin lookups; unresolved records and source 404s stay excluded. The
   [qualification packet](experiments/main-data/QUALIFICATION.md) owns this recovery state.
2. **Turn the inventory into finite eligible arms.** Count eligible tokens per bank after
   exclusions. Retained natural-code stock caps a one-pass run at 1.36B code tokens, so qualify
   additional supply or explicitly revise the finite recipe/horizon; no silent replay or
   reassignment of derived-data shares. Use the measured assistant lengths and RL prompt links
   before selecting downstream packs; adapter compatibility does not establish tool-task success.
3. **Close the trainer gaps on the workstation.** Stage 3 needs selective loss masks, best-fit
   packing and 16K/32K sequence support before any mid-training comparison. Stages 5 and 6 need the
   [build-or-decline decision](#stage-5-and-6-coverage-is-unclaimed), which is due before access.
   Neither spends grant hours.
4. **Prepare the reference-model GH200 packet before access.** Bind current source and exact inputs,
   checks, workload sizes, measurements and stop conditions, and rebuild the transfer bundle from
   current committed code. No paid run starts from this outline. In the same window, run the
   external [H100 throughput rental](docs/throughput-rental.md) to measure the flagship speedup
   before access; it costs no grant hours and keeps the 120-hour reservation from being the first
   place a checkpointing-off 1.2B configuration is measured. Its pre-flight is complete and it needs
   only a rented instance and SSH access. The [performance plan](docs/performance.md) is the
   pre-rental reference for MFU conventions, noise handling and kernel priorities.
5. **On access, qualify one worker then four.** Measure actual topology/ARM64 dependencies, kernels,
   batching, optimizer, communication, sustained throughput, restart/export and scheduler behavior.
   Declare every allocated GPU, including idle devices. Use results to cost the data studies and
   inform the main schedule, batch and horizon before the subsequent research gates.
   Freeze the throughput recipe here: microbatch, activation checkpointing and determinism are all
   immutable on resume. Qualify the compiled path under four-worker DDP with
   `scripts.training_replay --compile` before trusting any compiled production rate. Then apply the
   predeclared re-anchoring rule in either direction.
6. **Research the starting recipe.** The backbone is declared, not compared; the architecture study
   is deferred. Bind evaluation packs and numeric decision thresholds, and measure GH200 costs,
   before freezing seeds, common token horizons and executable configs. Run data screening, the
   decay study and confirmation from matched fresh initializations after source/runtime
   qualification; reduce the proposed matrix if needed without silently spending confirmation or
   production funds. Use fixed held-out source losses
   and development capability curves; keep final tests untouched. Record the result and limitations,
   then freeze the main mixture and schedule. Pretraining data research has 700 hours; reserve
   confirmation/evaluation cost before screening. The other 530 data-research hours belong to
   mid-training and post-training comparisons, 360 and 170 hours respectively, on useful parent checkpoints. The mid-training
   comparison will separate replay/source-only data from validated grounded or executable supervision and
   report downstream capability and SFT/RL optimization efficiency per token and GPU-hour.
7. **Pretrain, mid-train, post-train, evaluate and release through gates.** Freeze capability
   continuation and context extension separately, with their data, objectives and budget ownership.
   Preserve stage checkpoints and source-wise learning curves. Use useful stage checkpoints for
   downstream data comparisons before committing their stage budgets. Qualify context stages before
   long SFT. Promote reward training only for measured gains. Pin final endpoints/comparators before
   selection; keep final tests untouched.

Use milestone checks and bounded supervisors, not frequent manual progress polling or duplicate
workers.

## Readiness for experiment design

We can now draft hypotheses, controls and outcomes from the fixed direction and audited candidates.
Full production-corpus materialization is not a prerequisite for designing a bounded comparison.
Exact launch settings still depend on the evidence below; preparation receipts are not training results.

| Area | Established | Required before the relevant experiment launches |
| --- | --- | --- |
| Direction and budget | Six-stage data-pipeline report; architecture study deferred; stage ownership totals 5,000 hours | Costed arms, confirmation allowance and stop rules within each cap |
| Data | Pinned candidates, deterministic samples, exclusion methods, retained-stock counts and a signed source-use record over nine selected sources | Qualified finite arm manifests, family splits, eligible tokens and packing checks; origin/notice recovery for both code routes and peS2o v3 licence documentation |
| Backbone and runtime | 1.2B reference declared as the fixed substrate; completed H100 engineering baseline; throughput recipe selected on a 318M proxy | GH200/four-worker qualification, compiled-DDP qualification, flagship throughput measurement, FLOP accounting, recovery and measured cost |
| Evaluation | Pilot development evidence and pinned exclusion inputs | Primary endpoints, development/final partitions, regression tolerances and comparator protocol |
| Downstream stages | Stage objectives and separate research/production reservations | Useful parent checkpoints; qualified changed-data continuation, context and conditional RL paths |

The [proposed research design](experiments/main-data/README.md#research-before-the-main-run) now
specifies contrasts, controls, endpoints, maximum run counts and protected confirmation costs.
The [architecture control](docs/model.md#proposed-control-and-decision) keeps its checked parameter
count for the next allocation; it is not run here. Numeric subcaps live in the existing plan.

## Compute

The [numeric plan](experiments/main-data/plan.json) owns the reservation table and
[the overview](docs/program.md#compute-and-allocation) renders it with subdivisions, shares and
four-GPU equivalents. It reserves **120 hours for runtime and efficiency qualification, 1,230 for
data experiments, 1,800 for the 4K base reservation, 600 for capability/context/agentic
mid-training production, 800 for post-training production and 450 protected for
evaluation/recovery: 5,000 total GPU-hours**. Data research divides into 700/360/170 hours across
pretraining/mid-training/post-training. Inside the base reservation: **1,550 stable phase, 250
endpoint decay**. Inside post-training: **450 SFT, 200 conditional RL, 100 final self-SFT, 50
teacher and verification work**. `speck/operations/slurm.py` enforces the scheduled/protected split
of 4,550 + 450, and `make plan-check` fails if those constants, or any of these figures written in
prose, drift from the plan.

**The horizon has little margin.** At the measured H100 full-trainer rate the 1,800-hour
reservation supports about 83.3B tokens before long-context and agentic overhead; an illustrative
80% rate supports about 66.7B. Freeze the final horizon only after GH200 throughput and the 16K/32K
costs are measured; 100B and the old 320B/400B scales are deferred. The measured GH200 rate
re-anchors the horizon under three rules predeclared in `compute.throughput_reanchoring_rule`:

- **Apply the overhead derate.** The confirmation sweep runs `--mode compute` for thirty measured
  steps and excludes startup, inline validation and saves. The plan's anchor includes them. The
  measured pilot ratio is 0.9459; multiply by it before converting any benchmark rate into horizon
  hours, and remeasure the ratio on GH200 at production save cadence.
- **A surplus does not buy a longer base run.** If the measured rate beats the anchor, the 80B
  horizon does not move and the freed hours return to data research and to whichever stage is
  supply- or tooling-bound. Eligible-token supply, not compute, is what bounds this release.
- **A shortfall reduces the horizon**, never the protected mid-training, post-training or
  evaluation reservations.

**Four GPUs are not four times cheaper.** They increase aggregate speed while consuming four
GPU-hours per elapsed hour, and close no per-GPU efficiency gap. No GH200 or distributed speedup is
assumed, the rough 90-calendar-day window is not 90 days of continuous four-GPU funding, and
completed external rental costs stay outside this allocation. No GH200 jobs have launched.

## Stage 5 and 6 coverage is unclaimed

Coverage is claimed per stage, never in aggregate, and
[what that requires](docs/program.md#what-success-means) is fixed. Stages 1, 2 and 4 have trainers
today. Stage 3 still needs selective loss masks and best-fit packing; **stages 5 and 6 have no
trainer at all** — only the fixed-policy feasibility harnesses, which perform no policy updates.

That gap is two decisions, not one, and they have different costs and different deadlines:

- **Build or decline to build** the RL trainer, rollout engine and verifiers. This spends **no grant
  GPU-hours** — it is workstation software work measured in weeks of wall-clock — and it gates a
  third of the headline six-stage deliverable, because stage 6 rejection-samples a stage 5 parent.
  Deferring it to the closeout would mean writing an RL trainer *during* the access window. It is
  therefore **due before access**, and needs an owner and a date rather than a measurement.
- **Commit or release the 200 conditional RL hours.** This does need the measured throughput and the
  then-current supply position, so it stays at the GH200 qualification closeout, where it can be
  taken against real numbers. Building the trainer does not commit those hours; declining to build
  releases them to the supply-bound stages.

Until the build decision is recorded, stage 5 and 6 coverage is **unclaimed**, and no release text
may assume it. If the answer is not to build, report both stages as **not covered with the reason
recorded** — that is an acceptable outcome, stated plainly, and it is not the same as silence.
