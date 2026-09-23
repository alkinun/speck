# SpeckLabs: status and next work

Updated 2026-09-23. This is the status and the one work order for the first program. The
[program overview](docs/program.md) owns the design: goal, stages, model, data and compute.
[plan.json](experiments/main-data/plan.json) owns the numbers; experiment configurations and
verified receipts own actual run settings and results. Update these in place; history stays in Git.

The deliverable is an open data pipeline covering all six training stages, and a 1.2B model that
proves it end to end. The backbone is fixed by declaration, so no architecture claim is available
on this allocation. 5,000 GH200 GPU-hours are confirmed; access timing and GH200 throughput are not.

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

Three facts govern the order of work:

1. **Zero eligible tokens are established.** The 7.15B is retained stock with every gate still open:
   an upper bound on what the gates could admit, not usable data.
2. **`natural_code` binds at 1.36B, 1.70% of the 80B horizon.** Since the
   [2026-09-22 re-freeze](experiments/main-data/README.md#base-mixture) no bank caps a one-pass run
   at zero, and the binding bank is the one whose supply is hardest to grow.
3. **Closing the gap is acquisition, not compute.** Roughly a **14x** increase in retained stock is
   CPU, bandwidth and storage work that consumes no grant GPU-hours, so it runs before and during
   access, and a throughput surplus buys nothing here.

**Source use is decided and admits nothing.** The signed
[acceptance record](experiments/main-data/source-rights-acceptance.json) covers the nine
[selected sources](experiments/main-data/source-registry.json) for research training, paper
publication and public weight release, with no commercial use and no redistribution of source data
or derived shards. It is closed on six sources. The two code routes are approved per file, on the
original licence and recovered notice, so unresolved records stay excluded; peS2o v3 still needs
authoritative licence-selection documentation. Family holds persist across every stage.

**Family partitions are candidates, not splits.** Six text stocks and the 218-file code cohort are
[partitioned](experiments/main-data/family-partition.json) into whole-family 90/5/5 buckets with 45
code holds. The full retained-code graph and intended benchmark coverage keep `family_partition`
open on every source; the [qualification packet](experiments/main-data/QUALIFICATION.md) owns the
detail.

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
   exercise derivation. Carry all 45 cohort holds into every stage. Origins are verified for
   100/103 unheld Stack-Edu and 68/70 Stack v3 records; the rest stay excluded. The
   [qualification packet](experiments/main-data/QUALIFICATION.md) owns this recovery state.
2. **Turn the inventory into finite eligible arms.** Count eligible tokens per bank after
   exclusions. Retained natural-code stock caps a one-pass run at 1.36B code tokens, so qualify
   additional supply or explicitly revise the finite recipe/horizon; no silent replay or
   reassignment of derived-data shares. Use the measured assistant lengths and RL prompt links
   before selecting downstream packs; adapter compatibility does not establish tool-task success.
3. **Close the trainer gaps on the workstation.** Stage 3 best-fit masked rows and chat records
   are [implemented](docs/training.md#mid-training-readiness); 16K/32K runtime still needs GH200
   qualification. Stages 5 and 6 have minimal implementations: a
   [group-relative RL trainer](docs/training.md#reward-training) with checked verifiers and a
   [self-distillation dataset builder](docs/training.md#final-self-distillation). Remaining
   workstation work is multi-worker rollout and a pilot on a real SFT parent. None of it spends
   grant hours.
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

## Stage 5 and 6 coverage is unclaimed

Coverage is claimed per stage, never in aggregate, and
[what that requires](docs/program.md#what-success-means) is fixed. Stages 1, 2 and 4 have trainers
today, stage 3 has best-fit masked rows, and stage 5 has a minimal
[RL trainer](docs/training.md#reward-training) with checked verifiers; stage 6 builds a
[verified self-distillation dataset](docs/training.md#final-self-distillation) for the SFT trainer.
Neither is qualified at scale.

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

**Build decision, recorded 2026-09-23 by the principal investigator: build.** The scope is a
minimal GRPO-style trainer over checked-math and sandboxed-code verifiers, built on the existing
fixed-policy harnesses, then rejection-sampled final self-SFT. It is due before access. The 200
conditional RL hours stay uncommitted until the GH200 closeout. Stage 5 and 6 coverage remains
**unclaimed** until those trainers exist and pass their own qualification; if they do not, report
both stages as **not covered with the reason recorded**.
