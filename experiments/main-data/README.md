# Main data mixture and scale — working plan

2026-09-19. [plan.json](plan.json) records preparation targets and reproducible cost arithmetic.
This is not a launch configuration or a claim that the required corpus is already qualified.
The frozen H100 pilot, development evaluation and backups are complete. This working plan does
not change their configurations or historical result receipts.
The first 5,000-GPU-hour program studies data and training across pretraining, mid-training and
post-training after a bounded architecture/efficiency study and backbone freeze near 1.2B. The [program](../../docs/program.md#training-lifecycle)
defines stage boundaries; [model notes](../../docs/model.md) cover supporting attention/size analysis.
Broader architecture research belongs to later releases. Our base starts from scratch; the future
50,000-hour allocation is not assumed funded.

## Scale

Use **100B combined pretraining and capability mid-training tokens as the first-allocation
working horizon**. At the measured H100 full-trainer rate this projects to 2,160 of the reserved
2,300 GPU-hours. The roughly 106.5B arithmetic capacity is not a guaranteed hardware ceiling or a
launch target. At an illustrative 80% of that effective rate, only 85.2B fits. Freeze the horizon
from qualified supply and measured GH200/multiworker cost; shorten it if necessary.

The previous 320B desired / 400B stretch scales remain deferred cost comparisons, not current
acquisition targets. Better data does not prove equal quality at different token horizons. The
first release studies data and staged training within the confirmed 5,000-hour envelope.

Capability mid-training's split within the 100B horizon remains unfrozen; it is not an additional
allowance. Follow with **300 hours of context qualification toward 16K/32K**, with tokens unfrozen
and 128K a stretch, and **1.5M unique
post-training conversations**, within a 1–2M planning range. These require their own cost and
quality qualification. All tokens use the frozen Mistral tokenizer; holdouts are outside training
supply, and exposure/replay is distinct from unique eligible material.

## Research before the main run

This is the proposed experiment design, with numeric caps in [plan.json](plan.json). Exact datasets,
seed values, horizons, learning rates and decision thresholds must be bound before execution.
Complete runtime qualification and the [architecture study](../../docs/model.md#proposed-control-and-decision)
before pretraining data runs. Audit sources for every stage now; downstream runs require useful
parent checkpoints. The full production corpus need not be materialized to design a finite study.

| Study | Proposed comparison and controls | Decision evidence |
| --- | --- | --- |
| Pretraining screening | One qualified baseline plus at most two single-factor candidates: one code-bank contrast and one natural-web-bank contrast. Same fresh initialization, other banks, domain shares, packing, objective and exposure. | Fixed source losses and development curves rank candidates for confirmation; one screening seed establishes no robust winner. |
| Pretraining confirmation | Baseline versus one selected candidate, each from two new paired initialization seeds. Identical initial tensors within each pair; fresh optimizer/data state. | Predeclared primary endpoint, consistent paired effects, acceptable regressions and affordable supply. Publish both seed pairs and inconclusive outcomes. |
| Capability mid-training | Preceding mixture versus targeted code/math/repair plus replay at 4K. Same useful parent, next-token objective, optimizer policy, LR schedule and exposure; one paired data-order seed. | Target development-task success, source losses and general retention. This tests the combined mixture, not each ingredient. |
| Context mid-training | At 16K, coherent records in preceding domain proportions versus proposed long-domain reweighting. Same parent, short replay, packing and token exposure; one paired seed. | Fixed-suffix loss at matched long prefix length, related-prefix benefit, positional/multifile checks and short-task retention. This does not isolate coherent packing or context length. |
| Thinking SFT | Eligible baseline versus outcome-verification selection from the same candidate pool. Same parent, task/length strata, masking, serialization and schedule; two paired data-order seeds with fresh optimizers. | One declared code/math or tool-task success endpoint, with general/protocol regressions and supervision density reported. Known-invalid examples enter neither arm. |
| RL prompt feasibility | Conditional fixed-policy rollouts from the selected SFT checkpoint on candidate eligible prompt strata. | Verifier reliability, solvable difficulty, nontrivial success/failure groups and rollout cost. No RL training-data ablation is promised by this slot. |

The baseline must itself pass eligibility. The 35/25/40 domain envelope and eight banks below remain
preparation hypotheses. If a bank lacks qualified supply, explicitly revise the experimental baseline
before making all arms; an unqualified checked-code bank is not a valid control. Within the code
candidate, choose either a natural-source contrast (Stack-Edu versus Stack v3) or checked-code
substitution after the fixed audits. Do not run both as undeclared extra arms. The natural-web
candidate compares a qualified FineWeb-Edu control with selected Ultra-FineWeb HQ at fixed bank share;
do not simultaneously tune the score threshold or serialization. DCLM/math alternatives enter only
through a recorded replacement before launch, not an expanding sweep.

A shared screening baseline is valid only if both candidates have the same unchanged settings.
If both candidates meet their frozen screening criteria, prioritize code for confirmation; raw loss
deltas from different domains are not directly comparable. If neither passes, keep the eligible
baseline or record a redesign within the remaining cap. Do not combine two individually favorable
changes without testing that combined recipe. Confirmation
uses fresh runs and seeds, not extensions of selected screening checkpoints. It may reuse qualified
training families, with exposure reported; these are not new independent source samples. Main
pretraining starts fresh after the recipe decision. Neither research exposure nor discarded arms count toward
the main model's 100B horizon.

Capability arms branch from the same preserved 4K production milestone before final LR decay.
Context arms use the selected capability-production endpoint; SFT arms use the same qualified
post-context base, and RL prompt checks use the selected production SFT checkpoint. After each
downstream research decision, production starts again from its unchanged parent using the selected
recipe. Research checkpoints do not silently become extra production exposure.

### Research cost envelopes

These are proposed ceilings within existing reservations, not measured durations or runnable jobs.

| Reservation | Training / study slots | Shared support | Total GPU-hours |
| --- | --- | ---: | ---: |
| Architecture | Two arms × one seed × 60h = 120h; profiling 40h | 40h | 200 |
| Pretraining data | Up to three screening arms × 40h = 120h; two confirmation arms × two new seeds × 90h = 360h | 120h | 600 |
| Mid-training data | Capability pair: two × 35h = 70h; context pair: two × 25h = 50h | 30h | 150 |
| Post-training data | SFT: two arms × two seeds × 25h = 100h; conditional RL prompt feasibility 20h | 30h | 150 |

Training slots include trainer startup, inline validation and saves. Shared support covers other
on-allocation preparation, capability scoring, qualification and recovery. Each operation is charged
once; CPU/storage and external API costs are separate. Production caps remain 2,300h base/capability,
300h context and 800h post-training, with 400h protected evaluation/recovery. The RL prompt study
does not fund optimizer updates; conditional RL implementation qualification, rollouts and production
updates must fit its separate 200h production subdivision.

Freeze one common token horizon per base-training comparison from the slowest arm's measured cost,
qualified unique supply and required learning signal, rounded to complete updates. Equal per-arm
hour caps are safety ceilings, not instructions to train each arm until its clock expires.
At the historical H100 rate, 40h and 90h correspond to approximately 1.85B and 4.17B tokens before
external scoring/setup costs. These illustrate scale only; they are not GH200 forecasts or selected
horizons. A longer confirmation horizon needs its own schedule frozen before it starts.

Protect confirmation first. If timing or supply does not support the proposed matrix, drop the
secondary screening candidate or revise the common horizon before launch. Do not silently spend
confirmation, production or recovery funds on more screening. If a scientifically informative study
cannot fit, record the limitation and revise scope explicitly. An inconclusive comparison can support
a documented baseline choice; it cannot support a positive data-efficiency claim.

### Measurements and selection

1. Freeze the union of exclusions and source-family partitions across all candidate arms and stages.
   Keep validation and development families disjoint from training and final evaluation. Freeze a
   common validation pack independent of arm manifests, including declared target-source coverage.
   The existing loss evaluator accepts a separate data experiment; an arm's own validation mixture
   must not be used as its comparator's score.
2. Before screening, name one primary endpoint per contrast, its direction, aggregation weights and
   minimum useful effect. For source substitutions, use held-out target-domain next-token loss;
   report code, math and supporting-domain losses separately. Capability outcomes and broad retention
   are guardrails. For capability/SFT studies, use the declared target task-success metric. Set numeric
   tolerances from baseline measurement noise and practical requirements before viewing treatment
   results. Do not choose a favorable metric or loss weighting after results arrive.
3. Use baseline and matched quarter/half/final exposure checkpoints for base-training loss curves;
   cost capability scoring at declared milestones rather than every save. Freeze exact update-aligned
   checkpoints and optional early-stop rules. A safety stop, failed arm or budget overrun stays in the
   result ledger; do not replace failures with undisclosed reruns or call a partial pair complete.
4. Screening is exploratory. Lock the selected contrast, longer-horizon schedule, primary endpoint,
   regression tolerances and new seeds before confirmation. Report paired differences for each seed
   and task/source-family uncertainty separately. Bootstrap task families or documents, not correlated
   tokens or translated rows. Two seeds give limited run-to-run evidence; one-seed architecture and
   mid-training pairs remain explicitly exploratory. Disclose development reuse and selection.
5. Promote a candidate only if the frozen useful-effect criterion, guardrails, seed-consistency rule
   and cost/supply gates pass. Otherwise retain the qualified baseline or report an explicit redesign.
   A floor-level code/math result cannot establish capability improvement; loss-only gains remain
   loss findings. Final tasks are scored only after recipe/checkpoint decisions are frozen.

Do not average unrelated loss, accuracy and latency into an improvised efficiency score. Report
quality versus tokens and allocated GPU-hours, and inference cost at declared quality/protocol.
Family-level uncertainty output, fixed-suffix context scoring and expanded task protocols still need
qualification; existing aggregate loss output does not provide them automatically.
For context, score the identical held-out suffix with matched long prefixes across arms. Also compare
related prefixes with short/irrelevant-prefix controls; a larger prefix benefit caused only by worse
short-prefix loss is not a context improvement.

### Runtime and launch requirements

Fresh pretraining and separate fixed-pack loss evaluation exist. Exact-initial-tensor checks belong
in the run packet. The [training guide](../../docs/training.md#mid-training-readiness) records the
changed-data capability branch gap; context branches cannot be used to bypass it. Qualify that path
before either capability arm. Context studies additionally need useful long records, memory/restart
checks and the fixed-suffix scorer; do not spend the production context cap to hide research overruns.

SFT currently derives steps from complete epochs and bucket cycles. Construct finite arms with
matched processed positions and bucket schedules; predeclare and verify a supervised-token mismatch
tolerance. Match reasoning-length/task distributions as far as possible and report residual differences.
If that cannot be achieved, qualify an explicit exposure-control change before launch or narrow the
claim to a combined data/exposure intervention. Equal row counts alone do not match training exposure.
SFT research arms are small finite studies, not four repetitions of the full 1.5M-conversation target.

Before each launch, retain the exact source/config revision, parent or initialization identities,
qualified data/exclusion manifests, objective/masks, exposure/batch/schedule, evaluation identities,
numeric decision/stop thresholds, run seeds, all-in cost estimate and recovery destination. Check the
remaining ledger before each job. These design notes do not replace the existing execution guards.

## Base mixture

These are explicit starting hypotheses chosen for the code/math/agent target, not measured optimal
weights. The table extrapolates the initial mixture across the base horizon. Freeze capability
mid-training weights separately and update aggregate bank exposures if a staged mixture is adopted.
The source names below are production hypotheses; the experimental baseline deliberately substitutes
its declared control source in the bank under study. Update the production source choices after
confirmation rather than assuming every preferred candidate wins.

| Component | Share | 100B exposure | Eligible unique preparation | Candidate sources / admission condition |
| --- | ---: | ---: | ---: | --- |
| Selected broad natural web | 25% | 25B | 31.25B | Natural Ultra-FineWeb English; bind scored/HQ path and threshold after the bounded audit |
| Independent web coverage | 5% | 5B | 6.25B | FineWeb-Edu and/or DCLM; select allocation after overlap and coverage measurements |
| Natural code, tests and documentation | 30% | 30B | 37.5B | Stack-Edu, source-resolved UltraData-Code L2, and Stack v3 under qualification; preserve practical and multilingual coverage |
| Checked code explanations, exercises and repair | 5% | 5B | 6.25B | Qualified natural-code derivatives; UltraData-Code L3 only if lineage and independent checks succeed |
| Selected natural math / worked solutions | 20% | 20B | 25B | UltraData-Math L2, FineMath 4+, Nemotron-CC-Math `4plus`; source allocation follows comparative audit |
| Refined math explanations / derivations | 5% | 5B | 6.25B | Qualified UltraData-Math L3 and verified derivatives |
| Reference / science / technical documents | 5% | 5B | 6.25B | FineWiki, peS2o and qualified English FinePDFs-Edu |
| Refined educational web | 5% | 5B | 6.25B | Qualified Ultra-FineWeb-L3; Cosmopedia remains a comparison source |
| **Total** | **100%** | **100B** | **125B** | **35% code, 25% math, 40% supporting material** |

Every document has one primary bank, including cross-domain material such as mathematical code.
Deduplicate across banks and original/derived families before counting supply. Candidate names
are not interchangeable licenses or evidence that content is already downloaded.

Use existing released content first; this plan does not assume we can afford generating billions
of new teacher tokens ourselves. The 5B checked-code exposure slot has a substantial
unresolved supply gap. All inspected UltraData-Code L3 rows remain held. If a refined source fails
qualification or is too small, explicitly re-freeze its share into eligible natural material in the
same domain, or shorten the horizon. Never silently repeat a small set to fill the target.

Preserve useful implementation, library/API usage, testing, build/configuration and debugging roles.
Python and JavaScript/TypeScript are priorities, with meaningful systems, JVM, SQL and shell
coverage. Language/role weights require measured supply; do not inherit tiny pilot language quotas.
Natural code need not be a standalone executable, while a claimed verified exercise must meet its
declared execution and independent-test checks. Math should cover foundational worked problems
and harder derivations, rather than exclusively competition problems or lengthy synthetic prose.

## Acquisition and storage

The current pilot-source stocks total **6.799B tokens before joint eligibility**, including only
**0.477B code tokens**. The separate natural UltraData-Math preview adds 0.385B before joint checks.
These figures do not establish qualified supply for the new recipe. Prepare a **125B eligible unique
token bank** as a 25% selection margin over 100B exposure: each bank's target is 1.25 times its
exposure in the table. Default to one pass through the selected training documents; the unused
margin is not a requirement to train everything. Holdouts and rejected raw records are additional.

Code acquisition is a critical feasibility gate. The [Stack-Edu card](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)
provides content locators rather than the actual files, so fetching and qualifying source bytes must
be included. [UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) is an additional
candidate, subject to the existing lineage hold. Publisher corpus sizes and tokenizer counts do
not establish our eligible supply. The retained 0.477B natural-code tokens bound a one-pass
baseline at 1.59B total tokens with a 30% natural-code share, before exclusions, validation and other
banks. Even the proposed short studies need additional qualified baseline supply or shorter common
horizons; their hour caps do not imply that 1.85B/4.17B-token arms are available.

At uint16, 100B training token IDs occupy **200GB decimal**; the 125B preparation bank occupies
250GB. These are alternative inventories, not automatically two distinct copies to sum.
Budget indexes, masks, source text, deduplication workspaces, checkpoints and backups separately.
A 2TB local scratch allowance is a working envelope, not a measured dataset size. The September19
filesystem check showed approximately 4.5TiB available. Stream bounded acquisitions and retain
qualified packs; do not mirror all upstream datasets or assume the rental's 200GB disk can hold this.

## Mid-training data

Capability continuation uses selected code/math explanations, independently checked exercises and
repair material with broad-data replay. Specify difficulty, source families, actual correctness
coverage and overlap/reuse from pretraining. Tokens and GPU-hours come from the existing base
horizon and 2,300-hour reservation; the targeted portion and replay weights remain to be frozen.
The [training guide](../../docs/training.md#mid-training-readiness) records the changed-data branch
and objective limitations. Data acquisition does not make this a runnable continuation recipe.

The pretraining recipe study precedes the main run and uses 600 of the 900 data-research hours.
Its fresh-run tokens do not count as main-model exposure; production starts fresh after selection.
Mid-training comparisons use their separate 150-hour research cap and cannot replace the initial study.

## Context extension

Qualify **4K → 16K → 32K within 300 production GPU-hours**. Keep stage token counts unset until
measured cost, useful-context learning and short-task retention justify them. Approximately 128K
is a stretch; the old 8B/800-hour curriculum is retired. Retain complete longer records as inventory
without counting them as first-release training exposure.

Short replay at 25% remains a hypothesis. For the long portion, compare unchanged domain weights
repacked into coherent longer records against the proposed 50% repository / 30% math/science /
20% grounded cross-document mix. Charge these comparisons to the 150-hour mid-training research
cap, distinct from the selected context production run. Preserve repository/import/document
relationships and record source-family reuse. Evaluate fixed-suffix loss with related prefixes,
retrieval across positions, multi-file tasks and short-task retention before advancing length.

## Post-training scale

Target **1.5M unique, qualified training conversations** with an initial collection split of
600K code reasoning, 375K math reasoning, 375K agent/tool trajectories and 150K supporting general
and instruction tasks with reasoning. Agentic coding belongs to the agent category for counting;
do not count it twice. These are row collection targets, not training-token weights. Preserve
brief and deep reasoning under one always-thinking protocol; no non-thinking mode is planned.

Sample complete length bands through 128K and recover missing upstream long tails. Freeze sampling
by both total context and supervised tokens after the audit. Default cost scenarios use one pass;
epochs are not silently multiplied to reach a row target. The 1–2M range allows quality and length
to determine final volume without filling quotas with duplicates or weak examples.

At 1.5M conversations, a mean **8K total tokens** implies **12B context tokens per pass**; a mean
16K implies 24B. At the measured **4K SFT padded-position rate** these arithmetic proxies are
247 and 494 GPU-hours, respectively. They are not long-context runtime predictions. Actual padded
positions, long attention, verification, teachers, evaluation and repeated passes can increase cost.
Measure the actual length mixture before claiming that 1.5M or 2M conversations fit the budget.

See the [program overview](../../docs/program.md) for the proposed SFT/RL subdivision,
distributed execution, stage promotion criteria and release evaluation.

## Compute allocation

The user confirmed the four-GH200 / 5,000-total-GPU-hour envelope; provider access and hardware
throughput remain unconfirmed. Use this reservation without treating previous rental reservations
as actual provider billing:

| Work | GPU-hours reserved |
| --- | ---: |
| Runtime qualification | 100 |
| Architecture and efficiency study | 200 |
| Pretraining and capability mid-training | 2,300 |
| Context production | 300 |
| Post-training, including any on-allocation teacher/reward work | 800 |
| Data experiments: 600 pretraining / 150 mid-training / 150 post-training | 900 |
| Protected evaluation and recovery | 400 |
| **Total** | **5,000** |

The completed pilot measured **12,859 tokens/s** over the full trainer process, including cold
startup, validation and checkpoint saves. At that single-H100 rate, 100B takes **2,160 GPU-hours**.
The 2,300-hour reservation leaves about 140 hours beyond that linear extrapolation. Four workers
at perfect scaling would take 22.5 days; this is arithmetic, not measured GH200 wall time. Charge
all allocated GPUs, including idle workers. Hardware, communication, input loading and cadence
can change the rate. GH200/multiworker qualification is still required.

| Base scenario | H100-rate GPU-hours | Four-GPU elapsed days, ideal / illustrative 80% scaling | Effective tokens/s per GPU needed within 2,300h |
| --- | ---: | ---: | ---: |
| **100B first-allocation working horizon** | 2,160h | 22.5 / 28.1 | 12,077 |
| 320B deferred scale comparison | 6,912h | 72.0 / 90.0 | 38,647 |
| 400B deferred scale comparison | 8,641h | 90.0 / 112.5 | 48,309 |

These elapsed times assume each device matches the measured H100 before communication losses;
80% scaling is an illustration, not a measurement. Four devices would deliver about 51.4K tokens/s
in aggregate at perfect scaling, or 41.1K at 80%. They consume four GPU-hours per elapsed hour:
5,000 aggregate GPU-hours permit 1,250 four-GPU hours (52.1 days), not 5,000 machine-hours.
The recorded 90-day access window does not establish 90 days of continuous four-GPU funding.
The user confirmed on September 19 that the allowance is **5,000 total GPU-hours**.
Provider access remains unconfirmed; the budget interpretation is now explicit.

At the H100 reference rate, keeping the other 2,700 reserved GPU-hours brings the full program to
approximately **9,612 GPU-hours for 320B** or **11,341 for 400B**, before any distributed penalty.
Alternatively, fitting the base into its current 2,300-hour reservation requires about **3.0× / 3.8×
more effective throughput per allocated GPU**. Merely adding four workers does not deliver that
per-GPU improvement. GH200 hardware gains, larger batches and implementation improvements remain
unmeasured; qualify single-worker efficiency and four-worker scaling separately. Do not assume the
engineering pilot is an optimized throughput ceiling or promise a particular speedup.

The 100B horizon is now the working preparation baseline, conditional on cost and supply; the
larger comparisons do not authorize expansion. Preserve continuation checkpoints and choose a
compatible learning-rate schedule before training; a fully decayed run does not extend at no cost.
The approved research allocation reserves 200 hours for architecture/efficiency and 900 for data
experiments, while preserving 2,300 base and 800 post-training production hours. Context production
is reduced to 300 hours and protected evaluation/recovery to 400. Historical H100 rental costs are
separate. Remeasure throughput if the architecture decision changes the reference model.

Charge experimental training, evaluation, retries and on-allocation preparation to the appropriate
research cap; charge selected production runs to their production phase. CPU/storage and external
teacher API costs remain separate. Context token counts are deliberately unset; the 300-hour cap
cannot inherit the former 8B curriculum without evidence. No extra experiment, 128K stage or RL
campaign is implicitly funded beyond the declared caps.

## Preparation workstreams

The ordered next task is in [PLAN.md](../../PLAN.md#immediate-order-of-work); the items below are
preparation coverage, not five simultaneous studies. The [readiness summary](../../PLAN.md#readiness-for-experiment-design)
separates inputs available for detailed design from the requirements for launching experiments.

1. Audit the pinned natural Ultra-FineWeb candidate and its chosen threshold against retained
   FineWeb-Edu using the existing sampler; report content coverage, overlap and eligible tokens.
2. Use the [completed practical checks](../corpus-audit/practical-code-checks.json) and
   [qualification packet](QUALIFICATION.md) to finish broader exclusions and family separation.
   Continue the fixed 138-file Stack-Edu and 80-file Stack v3 cohorts after the
   [C/C++ follow-up](../corpus-audit/c-cpp-cohort-review.json): common full-read coverage is 123 files,
   with 53 unheld files still outside it and 44 records family-held. The latest batch records test-harness
   limitations and template/SDK context, retaining original labels/weights. One metadata-only hold was
   added for a record under `svcomp`; no new or reused verified origins are available for this batch.
   Complete broader lineage and remaining notice/revision checks. The controlled syntax diagnostic
   confirms one redaction failure; semantic preservation and source-use/family gates remain unresolved.
   Qualify finite experiment arms from eligible supply before bulk packing. No examples are admitted.
3. Compare natural/refined math and educational candidates, keeping claimed checked solutions
   distinct from merely well-formed text. Measure source overlap and complete-document lengths.
4. Audit the retained SFT stock and long-source gaps against the 1.5M task/length targets. Build a
   unique-task inventory; do not assume all existing 500K will survive qualification.
5. Freeze actual admitted source manifests, weights, repetitions, exclusions and hardware cost
   before launch. Learning-rate schedule, batch geometry and checkpoint cadence belong to that
   run's contract. This plan starts preparation, not a new GPU run or bulk download.

## Qualification work

The [CPU qualification packet](QUALIFICATION.md) records eligibility rules, pinned exclusion inputs,
family separation and the next comparable source audit. It does not authorize training or change
the working numeric recipe above.
