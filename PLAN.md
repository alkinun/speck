# SpeckLabs: current decisions and next work

Updated 2026-09-20. This is the status and work order for the first flagship program.
Read the [program overview](docs/program.md) for the connected design, data, compute and release
outline. [Main-data plan.json](experiments/main-data/plan.json) owns working numeric targets;
experiment configurations and verified receipts own actual run settings and measured results.
Update these in place. Historical proposals and failures remain in Git and their original receipts.

## Goal and selected decisions

Develop a useful model from scratch through a measured data recipe spanning pretraining,
mid-training and post-training, then release the base and an always-thinking assistant with an
open technical report. Primary uses are agentic coding, normal coding, math reasoning and tools;
general usefulness remains a regression check. The first paper centers on data, training and
capability development. External models are comparators or qualified teachers, not our base initialization.

- **Reference model:** 1,195,884,576 total/active parameters; 24 layers, width 2048; three KDA
  recurrent blocks then one global NoPE GQA block, repeated six times. Dense SwiGLU throughout,
  intermediate width 5120, tied embeddings, sigmoid KDA gates. One bounded architecture/efficiency comparison before backbone freeze; no MoE or size sweep.
- **Tokenizer fixed:** Mistral 32K; 32,003 embedding rows include three assistant role IDs.
- **Compute envelope:** 5,000 total GPU-hours across four GH200s, equivalent to 1,250 hours with
  all four allocated. Site/access and GH200 throughput remain unconfirmed.
- **Behavior selected:** one always-thinking protocol with brief/deep reasoning and tool actions;
  no supported non-thinking response mode. Preserve separate base and assistant checkpoints.
- **Context target:** start at 4K, qualify 16K then 32K within 300 production GPU-hours; 128K is a stretch.
- **Future ambition:** 50,000 GH200-hours and larger models; no future allocation is assumed funded.

The [model notes](docs/model.md) define the reference backbone and 200-hour architecture/efficiency
study. Freeze the chosen backbone before the data experiments. MoE, attention residuals and broad
searches remain later work. The first report centers on datasets and training, with controlled
training/inference efficiency evidence. High efficiency and competitive quality are hypotheses to
measure. The [program overview](docs/program.md#training-lifecycle) defines stage boundaries.

## Working recipes, not launch settings

| Area | Preparation target | Still to resolve |
| --- | --- | --- |
| Pretraining | 100B working base horizon including capability mid-training; initial 35% code, 25% math, 40% supporting material | Qualified source supply and measured GH200 cost; H100 reference implies 106.5B within the base reservation |
| Capability mid-training | Targeted code/math/repair/tool-use continuation with broad-data replay, within the base horizon | Token split, source-only versus validated synthesis, objective and changed-data continuation support remain to be qualified; measure downstream quality per token and GPU-hour |
| Context mid-training | 300-hour production cap; 16K then 32K qualification, with 128K stretch and token counts unset | Memory, useful-context learning, positional behavior and stage costs |
| Thinking SFT | 1.5M unique qualified conversations, within a 1–2M range | Correctness, source-family deduplication, complete long examples and supervised/context token totals |
| Reward training | Conditional verifiable math/code rewards after useful SFT | Trainer, rollout integration, verifiers, recovery and affordable measured benefit |
| Architecture/efficiency study | 200-hour bounded reference-versus-control comparison; proposed 1.186B all-GQA/RoPE control shape-checked | GPU qualification, FLOP accounting, numeric quality/cost thresholds and final configuration |
| Data research | 900 hours: 600 pretraining, 150 mid-training, 150 post-training | Screening and confirmation before each production stage; useful parents for downstream comparisons |

The [main data plan](experiments/main-data/README.md) specifies candidate sources and counting rules.
Retained stock, admitted data and training exposure are different quantities. No small preview or
publisher quality label establishes flagship-scale supply. No automatic repetition fills a gap.

## Completed evidence

| Work | What is established | Limits / evidence |
| --- | --- | --- |
| Runtime rehearsal | Full-size single-H100 base/SFT restart, loader/RNG recovery, numerical checks, generation and export | [H100 receipt](experiments/qualification/h100-result.json); ARM64 GH200, four-worker execution and scheduler recovery remain open |
| Engineering pilot | 800 steps / 104,857,600 tokens; 2.265 trainer hours, 12,859 full-trainer tokens/s, 13,595 steady optimizer tokens/s, 19.3 GiB peak allocated memory | [Execution receipt](experiments/pilot/h100-run.json); engineering evidence, not an optimized ceiling |
| Development scoring | 2,619 tasks in 2.12 evaluation hours: GSM8K strict 0/253, compiled code pass@1 0/33, IFEval strict prompts 12/101, ARC normalized 44/222, HellaSwag normalized 530/2,010 | [Results](experiments/pilot/development-result.json); weak base capability, custom subsets, no final-partition scoring |
| Backup closeout | Eight model/optimizer checkpoints, both exports, evaluation outputs and recovery logs retained locally | [Backup receipt](experiments/pilot/backup-result.json); pilot closed, no further rental work needed; provider billing/stop state not verified |
| Natural-code supply | All 1,999 retained archives reopened; 714,369 files / 476,774,847 retokenized tokens; 123 files under holds at census time | [Census](experiments/corpus-audit/code-supply.json), [lineage cohort](experiments/corpus-audit/natural-code-cohort.json); no new training admissions; stock covers 1.59% of the revised 30B natural-code exposure |
| Code bundle linkage | Four pinned repositories; 19 linked files / 5,650 tokens; four content flags hold 15 files by family | [Receipt](experiments/corpus-audit/code-bundles.json); static test weaknesses, no execution or admission; repository co-presence alone is insufficient |
| Code expansion feasibility | Upstream inventories pinned; one new Python metadata shard scanned, 16 blobs recovered, 13 length-matched / 9,541 tokens | [Receipt](experiments/corpus-audit/code-expansion.json); four content flags; four application modules selected for follow-up; no qualified yield or admission |
| Application origin/test review | All four source revisions, complete trees and MIT notices recovered; 29 response hashes verified | [Receipt](experiments/corpus-audit/code-application-origins.json); one direct but stale test link, no independent verification or admission |
| Stratified code preflight | 138 files / 459,615 tokens across 11 languages and 72 strata; exact offline replay | [Receipt](experiments/corpus-audit/code-yield-result.json); 30 content flags, 31 sample family holds; full eligibility and usable yield remain unresolved |
| Stack v3 preflight | All 16 frozen groups acquired; 29,347 repository rows / 379,942 entries; fixed 44-repository / 80-file cohort screened | [Receipt](experiments/corpus-audit/stack-v3-broader.json); four content flags, eight known-family holds; source use, quality and eligible yield unresolved; exact offline replay, no admission |
| Common code review | 176 complete texts / 412,974 tokens reviewed; 44 of 218 records family-held; all 174 currently unheld records read (plus two now-held records) | [Reading closeout](experiments/corpus-audit/stylesheet-cohort-review.json); page/template/component roles and notice questions recorded; all holds preserved; no yield estimate or admission |
| Expanded data qualification | Practical CPU checks complete; 37 files screened against 22 benchmark lanes, with seven content flags and 22 files held after family propagation | [Qualification packet](experiments/main-data/QUALIFICATION.md); includes pinned LiveCodeBench v6 public text, not complete corpus admission |
| Stratified HQ web audit | Twelve pinned shards / 290,761 documents; 192 sampled, 24 reviewed; exact offline replay | [Receipt](experiments/corpus-audit/web-hq-stratified.json); high-score extraction defects and lower-score coverage candidates; no training admissions; later full token census below |
| Web extraction follow-up | Three matching archived captures; 32 fresh comparison documents, 125,453 sample tokens | [Receipt](experiments/corpus-audit/web-filter-validation.json); confirmed omissions/boundary issues; candidate flags remain review-only |
| Retained-data closeout | Full HQ token/overlap census, full math text/index reconciliation, 500K SFT format census and sampled tool-aware lengths, bounded RL inventory | [Receipt](experiments/corpus-audit/data-readiness.json); finite stock bounds measured, source use/correctness/family eligibility and missing banks remain open |
| Other data preparation | 6.799B retained source tokens before joint eligibility; 500K assistant rows inventoried; finite tool-aware SFT rehearsal | [Supply](experiments/pilot/supply.json), [assistant contract](docs/assistant.md); not main-run qualified supply |

The [timing measurements](experiments/qualification/timing-result.json) additionally cover saves,
validation, restart, SFT and prefill/decode. Microbatch four showed about 20% higher steady throughput
in a short probe; sustained operation and recovery need qualification before adoption.
SFT padded positions/s are not supervised tokens/s. Earlier estimates and failed attempts remain
historical; do not repeat the completed pilot because a preparation document still contains commands.

## Immediate order of work

The local data-preparation pass is now closed for the current evidence. The
[data closeout](experiments/main-data/data-closeout.json) records the supplied frontier-data
research synthesis. Map its hypotheses to pinned source evidence before changing source identities,
candidate weights or study arms; keep all five source-readiness gates explicit.
The [shared data-design contract](experiments/main-data/data-design-contract.json) now binds the
stage, lineage, quality, coverage, dependency and contamination fields required by every data-stage
manifest. It changes design metadata only; it does not admit sources or alter the allocation.
The [candidate manifest preflight](experiments/main-data/candidate-manifest-preflight.json) applies
those fields to all 12 candidates and currently finds no complete manifest or eligible token.

The [natural-web candidate manifest](experiments/main-data/natural-web-candidate-manifest.json) is the
first source-specific application. It preserves the measured FineWeb-Edu/Ultra-FineWeb overlap while
keeping the web contrast blocked until its remaining gates close.
The [natural-code candidate manifest](experiments/main-data/natural-code-candidate-manifest.json)
does the same for Stack-Edu, Stack v3 and the separate checked-code route; it records zero eligible
code tokens and keeps the code contrast blocked.
The [math candidate manifest](experiments/main-data/math-candidate-manifest.json) separates natural,
filtered-web, refined/generated and unavailable math candidates; it keeps all math routes blocked until
correctness, contamination and finite-supply gates close.
The [post-training candidate manifest](experiments/main-data/post-training-candidate-manifest.json)
binds the assistant and reward inventories while keeping structural validity, outcome verification,
tool trajectories and fixed-policy RL feasibility as separate downstream gates.

1. **Map the reviewed research to source evidence.** The
   [source-mapping receipt](experiments/main-data/frontier-data-source-mapping.json) now attaches each
   finding to pinned local audits and records the remaining gate implications. Extend it with a source
   configuration, sampling frame, processing method, baseline, measured result and limitation whenever
   new primary evidence is added.
   Update the candidate review and source-readiness matrix only when primary evidence supports the
   corresponding gate. Do not reopen completed deterministic audits or infer eligibility from a
   paper's headline result.
2. **After research intake, complete main-data eligibility and coverage.** The bounded Python/JS/TS practical checks are
   complete and [recorded](experiments/corpus-audit/practical-code-checks.json). The
   [qualification packet](experiments/main-data/QUALIFICATION.md) pins broader exclusion inputs
   and defines family separation. LiveCodeBench release-v6 public-text coverage is now pinned;
   finish the full source-family graph and intended scoring coverage before exercise derivation.
   The 16-row UltraData-Code L3 preview remains held for lineage.
3. **Turn the completed data inventory into finite eligible arms.** The
   [source-readiness matrix](experiments/main-data/source-readiness.json) binds the retained inventories,
   gate status and one-pass horizon bounds to the design-only data packet. The
   [data-readiness closeout](experiments/corpus-audit/data-readiness.json) measures full retained HQ
   tokens, retained-bank exact overlap, math normalization links, assistant format/context coverage
   and a bounded RL prompt/reference inventory. The
   [qualification packet](experiments/main-data/QUALIFICATION.md#next-bounded-data-packet) owns
   the remaining source-use, family/exclusion, correctness and finite-supply decisions. Do not repeat
   the completed code reading pass or infer quality from format, length, classifier scores or notices.
   GitHub origin recovery remains quota-blocked; preserve all 44 family holds and separate source 404s.
   The working eight-bank mixture still lacks qualified checked-code/refined-math supply. Retained
   HQ and natural-code totals also bound one-pass horizons below the maximum study-hour envelopes.
   Qualify additional supply or explicitly revise the finite research recipe/horizon; no silent replay
   or reassignment of synthetic shares. Use the measured assistant lengths and RL prompt links before
   selecting downstream packs. Adapter compatibility does not establish tool-task success.
4. **Prepare the reference-model GH200 packet before access.** Bind current source and exact inputs,
   checks, workload sizes, measurements and stop conditions. Rebuild the historical transfer bundle;
   do not treat its old source commit as the current release. No paid run starts from this outline.
5. **On access, qualify one worker then four.** Measure actual topology/ARM64 dependencies, kernels,
   batching, optimizer, communication, sustained throughput, restart/export and scheduler behavior.
   Declare every allocated GPU, including idle devices. Use results to cost the architecture/data
   studies and inform the main schedule, batch and horizon before the subsequent research gates.
   Reduce the 100B working horizon if measured
   cost or supply requires it.
6. **Complete the bounded architecture study, then research the starting recipe.** Use the
   200-hour architecture/efficiency cap to answer one consequential question and freeze the backbone.
   Run data screening and confirmation from
   matched fresh initializations after source/runtime qualification. Use fixed held-out source losses
   and development capability curves; keep final tests untouched. Record the result and limitations,
   then freeze the main mixture and schedule. Pretraining data research has 600 hours; reserve
   confirmation/evaluation cost before screening. The other 300 data-research hours belong to
   mid-training and post-training comparisons, 150 each, on useful parent checkpoints. The mid-training
   comparison will separate replay/source-only data from validated grounded or executable supervision and
   report downstream capability and SFT/RL optimization efficiency per token and GPU-hour.
7. **Pretrain, mid-train, post-train, evaluate and release through gates.** Freeze capability
   continuation and context extension separately, with their data, objectives and budget ownership.
   Preserve stage checkpoints and source-wise learning curves. Use useful stage checkpoints for
   downstream data comparisons before committing their stage budgets. Qualify context stages before
   long SFT. Promote reward training only for measured gains. Pin final endpoints/comparators before
   selection; keep final tests untouched.

Runtime and data preparation are the active tracks. Audit sources for all stages now. The starting
mixture study precedes main pretraining; mid-training and assistant model comparisons need useful
parent checkpoints and measured costs. Use milestone checks and bounded supervisors, not frequent
manual progress polling or duplicate workers.

## Readiness for experiment design

We can now draft hypotheses, controls and outcomes from the fixed direction and audited candidates.
Full production-corpus materialization is not a prerequisite for designing a bounded comparison.
Exact launch settings still depend on the evidence below; preparation receipts are not training results.

| Area | Established | Required before the relevant experiment launches |
| --- | --- | --- |
| Direction and budget | Data-centered report; bounded architecture study; stage ownership totals 5,000 hours | Costed arms, confirmation allowance and stop rules within each cap |
| Data | Pinned candidates, deterministic samples, exclusion methods and retained-stock counts | Qualified finite arm manifests, source-use decisions, family splits, eligible tokens and packing checks |
| Backbone and runtime | 1.2B reference, proposed parameter-matched GQA control and completed H100 engineering baseline | Control/GH200 qualification, FLOP accounting, recovery and measured cost; freeze backbone before data comparisons |
| Evaluation | Pilot development evidence and pinned exclusion inputs | Primary endpoints, development/final partitions, regression tolerances and comparator protocol |
| Downstream stages | Stage objectives and separate research/production reservations | Useful parent checkpoints; qualified changed-data continuation, context and conditional RL paths |

The [proposed research design](experiments/main-data/README.md#research-before-the-main-run) now
specifies contrasts, controls, endpoints, maximum run counts and protected confirmation costs.
The [architecture control](docs/model.md#proposed-control-and-decision) has a checked parameter count;
its runtime and production selection remain open. Numeric subcaps live in the existing plan.

The fixed code reading pass and retained-data closeout are complete. Next produce eligible finite arms,
then bind evaluation packs and numeric decision thresholds. Measure GH200 costs before freezing
seed values, common token horizons and executable configs. Reduce the proposed matrix if needed
without silently spending confirmation or production funds. Audit downstream sources now; run their
comparisons from useful parents only after the recorded runtime gaps are closed.

## Compute

The [numeric plan](experiments/main-data/plan.json) reserves **100 hours for runtime qualification,
200 for architecture/efficiency, 900 for data experiments, 2,300 for base/capability production,
300 for context production, 800 for post-training production and 400 for evaluation/recovery:
5,000 total**. Data research divides into 600/150/150 hours across pretraining/mid-training/post-training.
Each experiment includes its preparation, evaluations, retries and allocated idle time; production
exposures are separate. Capability production remains inside the 100B base horizon and 2,300 hours.
The [overview](docs/program.md#compute-and-allocation) records subdivisions and gates. Context tokens
remain unset until measured qualification; 128K is optional. The protected reserve is smaller and
must remain explicit. Completed external rental costs stay separate from this allocation.

At the measured H100 full-trainer rate, 2,300 GPU-hours corresponds to about 106.5B base tokens.
The 100B working horizon projects to 2,160 hours, leaving about 140 hours of margin. At an
illustrative 80% of the H100 effective rate, only 85.2B fits; measure before freezing the horizon.
The old 320B/400B scales are deferred comparisons, not first-allocation targets.
Four GPUs increase aggregate speed while consuming four GPU-hours per elapsed hour; they do not
close the per-GPU efficiency gap. No GH200 or distributed speedup is assumed. The rough 90-calendar-day
application window is not 90 days of continuous four-GPU funding. No GH200 jobs have launched.

## What success means

| Capability | Evidence to collect |
| --- | --- |
| Code and agentic coding | Standard executable code tasks plus held-out repository repair, regressions and actual task completion |
| Math and general usefulness | Checked answers, source-wise loss, knowledge, instruction compliance and representative language tasks |
| Thinking and tools | Correctness versus reasoning budget, valid calls, use of observations, error recovery and bounded loops |
| Context | Positional retrieval, related-prefix benefit, multi-file/document reasoning and short-task retention |
| Efficiency | All-in GPU-hours, tokens/FLOPs to useful quality, prefill/decode latency, memory and cost including failed attempts |
| Openness | Identified checkpoints, source/config manifests, processing recipes, curves, costs, failure analysis and executable evaluation |

The [competitive strategy](docs/competitive.md) and [report outline](docs/report.md) define the
comparison and release evidence. No architecture superiority claim follows from an engineering
pilot without a matching control. Release claims must describe measured capability and limitations.
