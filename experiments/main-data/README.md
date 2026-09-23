# Main data mixture and scale — working plan

2026-09-23. [plan.json](plan.json) records preparation targets and reproducible cost arithmetic.
This is not a launch configuration or a claim that the required corpus is already qualified.
The frozen H100 pilot, development evaluation and backups are complete. This working plan does
not change their configurations or historical result receipts.
The first 5,000-GPU-hour program studies the data pipeline across all six training stages on a
1.2B backbone that is **fixed by declaration, not selected over a control**. The bounded
architecture/efficiency study was deferred to a later allocation on 2026-09-21.
The [program](../../docs/program.md#training-lifecycle)
defines stage boundaries; [model notes](../../docs/model.md) cover reference-only attention/size
analysis. Architecture research belongs to later releases. Our base starts from scratch; the future
50,000-hour allocation is not assumed funded.

The [architecture study packet](architecture-study-packet.json) preserves that deferred design
unmodified and draws zero hours from this allocation. Validate that it stays deferred:

```bash
PYTHONPATH=. python experiments/main-data/check_architecture_study.py \
  experiments/main-data/architecture-study-packet.json
```

It contains one reference, one matched all-GQA/RoPE control and one seed. The study is **deferred
to a later allocation**: it now draws zero GPU-hours and records the 200 it released. The design is
preserved so the next allocation can run it unchanged. It does not authorize training or broaden
the architecture search.

The [data preparation closeout](data-closeout.json) freezes the current evidence-only corpus pass.
The supplied frontier-data research is recorded in
[frontier-data-research.json](frontier-data-research.json). No source is admitted and no study
packet authorizes acquisition or training; per-source gates live in the
[source-readiness matrix](source-readiness.json).

## Scale

Use **80B 4K-base tokens as the revised first-allocation working horizon**. At the measured H100
full-trainer rate this projects to 1,728 of the reserved 1,800 4K-base GPU-hours. The roughly 83.3B
arithmetic capacity is not a guaranteed hardware ceiling or a launch target. At an illustrative 80%
of that effective rate, only 66.7B fits. Freeze the final horizon from qualified supply, GH200 cost,
and the separate 16K/32K mid-training costs; 100B is deferred.

The previous 320B desired / 400B stretch scales remain deferred cost comparisons, not current
acquisition targets. Better data does not prove equal quality at different token horizons. The
first release studies data and staged training within the confirmed 5,000-hour envelope.

Capability mid-training uses a combined **600-hour production reservation** across a 4K capability
bridge, 16K repository reasoning and 32K agentic continuation; its tokens are separately accounted
from the 4K base. Research uses **360 hours** for the four-arm proxy/objective/context/confirmation
funnel, and **1.5M unique
post-training conversations**, within a 1–2M planning range. These require their own cost and
quality qualification. All tokens use the frozen Mistral tokenizer; holdouts are outside training
supply, and exposure/replay is distinct from unique eligible material.

## The first executable data-study packet

[data-study-packet.json](data-study-packet.json) is the next bounded experiment artifact. It binds
one common baseline, code-bank, natural-web and AI-generation/provenance-filter contrasts, followed by an explicit
three-way decay comparison. It fixes the tokenizer, 4K context, objective,
serialization, exposure accounting and evaluation boundaries while leaving source admission open.

Validate the packet and its source-of-truth references offline:

```bash
PYTHONPATH=. python experiments/main-data/check_data_study.py \
  experiments/main-data/data-study-packet.json
```

The command checks design arithmetic and launch boundaries only. It does not acquire sources,
execute corpus content or authorize training.

The packet is deliberately design-only. Before any arm can run, every selected source needs its
source-use conditions met, family and near-duplicate partitions, independent correctness checks where
claimed, finite accepted-token counts, a disjoint evaluation pack and measured GH200 cost. The
[acceptance record](source-rights-acceptance.json) decided source use for the nine selected sources
on 2026-09-22: closed on six, and conditional on three pending origin/notice recovery for both code
routes and peS2o v3 licence documentation. It admits nothing. A future
launch request must bind the qualified manifests and the existing rights/operations/firewall records.
The packet does not authorize acquisition, training or a main-run mixture.

## Capability mid-training efficiency packet

[mid-training-study-packet.json](mid-training-study-packet.json) defines the next data-efficiency study. It compares replay, source-only repository data, grounded workflows and executable trajectories, then separately tests selective loss/packing and context transitions. Validate it offline:

```bash
PYTHONPATH=. python experiments/main-data/check_mid_training_study.py \
  experiments/main-data/mid-training-study-packet.json
```

The packet reserves 360 GPU-hours for mid-training research. Capability data interventions must use
`--branch-kind data` with `training_phase: data_continuation`; context interventions must use the
separate context branch. Linked SFT/RL convergence and adaptation measurements belong to the separate
post-training research reservation and must use the same downstream recipe across retained arms.

## Post-training data-study packet

[post-training-study-packet.json](post-training-study-packet.json) binds SFT selection, fixed-policy RL feasibility and a bounded final self-SFT pilot. It does not authorize policy updates or production self-SFT. Validate it offline:

```bash
PYTHONPATH=. python experiments/main-data/check_post_training_study.py \
  experiments/main-data/post-training-study-packet.json
```

The packet reserves 80 GPU-hours for SFT comparison, 40 for RL feasibility, 30 for the final self-SFT pilot and 20 for support. Existing 500K-row stock remains format and context evidence until source, family, correctness and held-out gates close.

Run `make plan-check` from the repository root to validate the cross-stage compute ledger, all
packet budgets and every recorded input reference together. A passing check is necessary bookkeeping,
not training authority.

The ledger checker confirms the plan's input records exist; the stage validators check their own
declared references. This is not a recursive audit of every historical receipt or evidence of
source eligibility. Study packets own arm counts and per-arm ceilings; mirrored plan fields must
match them. Match training exposure for data comparisons and report actual GPU-hours separately;
equal exposure and equal elapsed cost are not generally achievable together.

The [source-readiness matrix](source-readiness.json) is the one record of per-source identity,
inventory, gates, blockers and manifest-field completeness. It currently finds zero complete manifests
and zero eligible tokens. Validate it offline:

```bash
PYTHONPATH=. python experiments/main-data/check_source_readiness.py \
  experiments/main-data/source-readiness.json
```

The matrix is an evidence boundary: retained tokens are not eligible tokens, and an arm remains blocked until its listed gates close in a revised manifest.

All three study packets also reference the shared [stage-conditioned data-design contract](data-design-contract.json).
It makes the research synthesis operational without changing the study counts: manifests must preserve
stage, source family, transformation, quality, coverage, dependency, contamination and lineage fields;
sampling uses quality weighting with coverage constraints; and each stage has its own utility and
correctness measurements. The contract is design metadata only and does not admit sources or authorize runs.
The [frontier-data source mapping](frontier-data-source-mapping.json) is its companion review receipt:
it ties each finding to pinned local evidence, names the gate implications and records remaining
measurement gaps. It is an evidence index, not a source-admission decision.

The candidate manifests name readiness sources and add only domain evidence and comparison contracts:

- The [natural-web manifest](natural-web-candidate-manifest.json) binds the FineWeb-Edu control and
  the Ultra-FineWeb HQ candidate and preserves their measured overlap.
- The [natural-code manifest](natural-code-candidate-manifest.json) binds Stack-Edu and Stack v3; the
  checked-code route it also describes is unbanked.
- The [math manifest](math-candidate-manifest.json) binds FineMath 4+ and InfiWebMath 4+ beside the
  not-selected UltraData-Math and Nemotron and the review-only OpenMath/L3 candidates; arithmetic
  triage is a quality lead, not a correctness certificate.
- The [post-training manifest](post-training-candidate-manifest.json) binds the 500K-row assistant
  census and the 44,369-row reward-prompt inventory, keeping structural, outcome, tool-trajectory and
  RL-feasibility gates separate.
- The [post-training audit protocol](post-training-audit-protocol.json) fixes the next bounded
  review and its stop rules, measuring efficiency only after correctness.

OpenMathInstruct-2 and SmolTalk remain outside the readiness matrix
([candidate review](../corpus-audit/recipe-review.json)). A late math reweighting is a schedule
hypothesis, not an additional arm.

## Research before the main run

This is the proposed experiment design, with numeric caps in [plan.json](plan.json). Exact datasets,
seed values, horizons, learning rates and decision thresholds must be bound before execution.
Complete runtime qualification before pretraining data runs; the
[architecture study](../../docs/model.md#proposed-control-and-decision) is deferred and is not a
gate here. Audit sources for every stage now; downstream runs require useful parent checkpoints. The full production corpus need not be materialized to design a finite study.

| Study | Proposed comparison and controls | Decision evidence |
| --- | --- | --- |
| Pretraining screening | One qualified baseline plus at most three single-factor candidates: code-bank, natural-web-bank and calibrated AI-generation/provenance-filter contrasts. Same fresh initialization, other banks, domain shares, packing, objective and exposure. | Fixed source losses and development curves rank candidates for confirmation; one screening seed establishes no robust winner. |
| Pretraining confirmation | Baseline versus one selected candidate, each from two new paired initialization seeds. Identical initial tensors within each pair; fresh optimizer/data state. | Predeclared primary endpoint, consistent paired effects, acceptable regressions and affordable supply. Publish both seed pairs and inconclusive outcomes. |
| Capability mid-training | Replay/source-only control versus targeted code/math/repair/tool-use grounding, with validated executable trajectories only if environments qualify. Same useful parent, 4K next-token objective, downstream SFT/RL recipe, optimizer policy, schedule and exposure; one paired data-order seed. | Target capability, source losses, downstream SFT convergence, early RL adaptation and quality per mid-training token/GPU-hour. Branches are additive ablations; context extension stays separate. |
| Context mid-training | At 16K, coherent records in preceding domain proportions versus proposed long-domain reweighting. Same parent, short replay, packing and token exposure; one paired seed. | Fixed-suffix loss at matched long prefix length, related-prefix benefit, positional/multifile checks and short-task retention. This does not isolate coherent packing or context length. |
| Thinking SFT | Eligible baseline versus outcome-verification selection from the same candidate pool. Same parent, task/length strata, masking, serialization and schedule; two paired data-order seeds with fresh optimizers. | One declared code/math or tool-task success endpoint, with general/protocol regressions and supervision density reported. Known-invalid examples enter neither arm. |
| RL prompt feasibility | Conditional fixed-policy rollouts from the selected SFT checkpoint on candidate eligible prompt strata. | Verifier reliability, solvable difficulty, nontrivial success/failure groups and rollout cost. No RL training-data ablation is promised by this slot. |

The baseline must itself pass eligibility. The 35/25/40 domain envelope and six banks below remain
preparation hypotheses. If a bank lacks qualified supply, explicitly revise the experimental baseline
before making all arms; that rule is what removed the checked-code and refined-math banks on
2026-09-22. The code candidate is therefore the natural-source contrast (Stack-Edu versus Stack v3);
checked-code substitution is unbanked and would need a revised freeze before it could be an arm. The natural-web
candidate compares a qualified FineWeb-Edu control with selected Ultra-FineWeb HQ at fixed bank share;
do not simultaneously tune the score threshold or serialization. DCLM/math alternatives enter only
through a recorded replacement before launch, not an expanding sweep.

A shared screening baseline is valid only if every candidate has the same unchanged settings.
If several candidates meet their frozen screening criteria, confirm the code candidate when it is
among them; otherwise choose among the passing candidates by a rule declared before screening. Raw
loss deltas from different domains are not directly comparable. If none passes, keep the eligible
baseline or record a redesign within the remaining cap. Do not combine two individually favorable
changes without testing that combined recipe. Confirmation
uses fresh runs and seeds, not extensions of selected screening checkpoints. It may reuse qualified
training families, with exposure reported; these are not new independent source samples. Main
pretraining starts fresh after the recipe decision. Neither research exposure nor discarded arms count toward
the main model's 80B 4K-base horizon.

Capability arms branch from the same preserved 4K production milestone before final LR decay.
The decay study branches from one matched stable checkpoint and compares natural, curated-natural
and small verified-derived enrichment under the same decay schedule.
Context arms use the selected capability-production endpoint; SFT arms use the same qualified
post-context base, and RL prompt checks use the selected production SFT checkpoint. After each
downstream research decision, production starts again from its unchanged parent using the selected
recipe. Research checkpoints do not silently become extra production exposure.

### Research cost envelopes

Deduplication strength (exact versus near/family), quality thresholds, coverage-aware sampling,
grounded augmentation and difficulty/replay curricula remain candidate follow-ups. CPU audits can
measure retention, false positives, overlap and cost now; separate causal training claims require
a predeclared replacement of a funded arm or a later allocation. They are not extra funded runs.
Never relax benchmark-family separation as a deduplication ablation. A combined recipe needs its
own comparison before attributing gains to individual components.

These are proposed ceilings within existing reservations, not measured durations or runnable jobs.

| Reservation | Training / study slots | Shared support | Total GPU-hours |
| --- | --- | ---: | ---: |
| Pretraining data | Base screen: four arms × 30h = 120h; decay screen: three arms × two seeds × 30h = 180h; confirmation: two arms × two new seeds × 60h = 240h | 160h | 700 |
| Mid-training data | Proxy screen 120h; objective/packing 40h; context 80h; 1.2B confirmation 80h | 40h | 360 |
| Post-training data | SFT: two arms × two seeds × 20h = 80h; conditional RL feasibility 40h; final self-SFT pilot 30h | 20h | 170 |

The architecture study no longer holds a row. Its 200 hours were released on 2026-09-21 when the
comparison was deferred: 180 to data research (100 pretraining, 60 mid-training, 20 post-training)
and 20 to reference-only profiling inside runtime qualification. Each row above matches its study
packet's own `budget_check`, and `make plan-check` validates both against `plan.json`.

Training slots include trainer startup, inline validation and saves. Shared support covers other
on-allocation preparation, capability scoring, qualification and recovery. Each operation is charged
once; CPU/storage and external API costs are separate. Production caps are 1,800h 4K base,
600h capability/context/agentic mid-training and 800h post-training, with 450h protected
evaluation/recovery. The RL prompt study
does not fund optimizer updates; conditional RL implementation qualification, rollouts and production
updates must fit its separate 200h production subdivision.

Freeze one common token horizon per base-training comparison from the slowest arm's measured cost,
qualified unique supply and required learning signal, rounded to complete updates. Equal per-arm
hour caps are safety ceilings, not instructions to train each arm until its clock expires.
At the historical H100 rate, 30h, 40h and 60h correspond to approximately 1.39B, 1.85B and 2.78B tokens before
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
The sources below are the [registry's](source-registry.json) selection; the experimental baseline
deliberately substitutes its declared control source in the bank under study. Update the production source choices after
confirmation rather than assuming every preferred candidate wins.

This is the one prose copy of the mixture; [plan.json](plan.json) owns the numbers it renders, and
`make plan-check` fails if the two disagree.

| Bank | Share | 80B exposure | Eligible unique preparation | Selected sources |
| --- | ---: | ---: | ---: | --- |
| `selected_web` — selected broad natural web | 25% | 20B | 25B | Ultra-FineWeb English HQ (L1 route) |
| `independent_web` — independent web coverage | 5% | 4B | 5B | FineWeb-Edu |
| `natural_code` — natural code, tests and documentation | 35% | 28B | 35B | Stack-Edu and Stack v3; carries the whole declared code share |
| `natural_math` — selected natural math / worked solutions | 25% | 20B | 25B | FineMath 4+ and InfiWebMath 4+; carries the whole declared math share |
| `reference_science` — reference / science / technical documents | 5% | 4B | 5B | peS2o v3 and FineWiki |
| `refined_web` — refined educational web | 5% | 4B | 5B | Cosmopedia v2 |
| **Total** | **100%** | **80B** | **100B** | **35% code, 25% math, 40% supporting material** |

Checked code, Nemotron-CC-Math `4plus` and the UltraData-Math L2 preview were not selected on
2026-09-22. DCLM, UltraData-Code L2, Ultra-FineWeb-L3 and English FinePDFs-Edu are unbanked
alternatives; each enters only through a recorded replacement and a revised freeze.

The mixture was re-frozen over these six banks on 2026-09-22. `checked_code` (5%) and `refined_math`
(5%) were removed: each held zero retained candidate stock of any kind, so each capped a one-pass run
at zero tokens at its declared weight, however complete every other bank became. The merge was
**within-domain, not a reassignment** — the dropped code share went to `natural_code` and the dropped
math share to `natural_math` — so total exposure stays 80B, total eligible-unique preparation stays
100B, and the declared 35/25/40 domain split is unchanged. The re-freeze created and destroyed no
eligible token; it moved where the horizon binds, onto the bank whose supply is hardest to grow.
Checked code and refined math are now **unbanked** derived candidates: qualifying either later needs
a revised freeze and its own exposure ledger, and neither may be poured into a natural bank, because
derived and natural lineage are separately identified.

Every document has one primary bank, including cross-domain material such as mathematical code.
Deduplicate across banks and original/derived families before counting supply. Candidate names
are not interchangeable licenses or evidence that content is already downloaded.

Use existing released content first; this plan does not assume we can afford generating billions
of new teacher tokens ourselves. All inspected UltraData-Code L3 rows remain held. If a refined source fails
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
**0.477B code tokens**. The not-selected UltraData-Math L2 preview's 0.385B is not counted.
These figures do not establish qualified supply for the new recipe. Prepare a **100B eligible unique
token bank** as a 25% selection margin over 80B exposure: each bank's target is 1.25 times its
exposure in the table. Default to one pass through the selected training documents; the unused
margin is not a requirement to train everything. Holdouts and rejected raw records are additional.

Code acquisition is a critical feasibility gate. The [Stack-Edu card](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)
provides content locators rather than the actual files, so fetching and qualifying source bytes must
be included. [UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) is an unbanked
alternative, subject to the existing lineage hold. Publisher corpus sizes and tokenizer counts do
not establish our eligible supply. The retained 0.477B natural-code tokens bound a one-pass
baseline at 1.36B total tokens with a 35% natural-code share, before exclusions, validation and other
banks. Even the proposed short studies need additional qualified baseline supply or shorter common
horizons; their hour caps do not imply that 1.39B–2.78B-token arms are available.

At uint16, 80B training token IDs occupy **160GB decimal**; the 100B preparation bank occupies
200GB. These are alternative inventories, not automatically two distinct copies to sum.
Budget indexes, masks, source text, deduplication workspaces, checkpoints and backups separately.
A 2TB local scratch allowance is a working envelope, not a measured dataset size. The September19
filesystem check showed approximately 4.5TiB available. Stream bounded acquisitions and retain
qualified packs; do not mirror all upstream datasets or assume the rental's 200GB disk can hold this.

## Mid-training data

Capability continuation uses repository structure, dependency-linked files, issues, reviews,
pull requests, commits, diffs, verified repairs, tool schemas, grounded workflows and executable
trajectories with broad-data replay. Specify difficulty, source families, correctness, environment
receipts, loss masks and overlap/reuse from pretraining. The [training guide](../../docs/training.md#mid-training-readiness)
records the data/context branch and objective limitations. Data acquisition does not make this a
runnable continuation recipe.

The pretraining recipe study precedes the main run and uses 700 of the 1,230 data-research hours,
of which 180 belong to the three-arm, two-seed endpoint-decay study.
Its fresh-run tokens do not count as main-model exposure; production starts fresh after selection.
Mid-training comparisons use their separate 360-hour research cap and cannot replace the initial study.

## Context extension

Qualify **4K → 16K → 32K within the combined 600-hour capability/context/agentic production
reservation**. Keep stage token counts unset until measured cost, useful-context learning and
short-task retention justify them. 64K/128K is future work. Retain complete longer records as
inventory without counting them as first-release training exposure.

Short replay at 25% remains a hypothesis. For the long portion, compare unchanged domain weights
repacked into coherent longer records against the proposed 50% repository / 30% math/science /
20% grounded cross-document mix. Charge these comparisons to the 360-hour mid-training research
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

After SFT and optional RL, the final self-SFT pilot compares verified-anchor continuation against
verified self-distillation with anchor replay. Freeze prompt, teacher, environment, verifier and
rejection manifests before generation. Full final self-SFT remains inside the 800-hour post-training
production reservation and is promoted only on held-out transfer.

## Compute allocation

The grant confirms four GH200s and 5,000 total GPU-hours within a nominal 90-day window; the access
start, site details and hardware throughput remain unconfirmed. [plan.json](plan.json) owns the
reservation table and [the overview](../../docs/program.md#compute-and-allocation) renders it. This
section owns only the horizon arithmetic that the reservations imply.

The completed pilot measured **12,859 tokens/s** over the full trainer process, including cold
startup, validation and checkpoint saves. At that single-H100 rate:

| Base scenario | H100-rate GPU-hours | Four-GPU elapsed days, ideal / illustrative 80% scaling | Effective tokens/s per GPU needed within 1,800h |
| --- | ---: | ---: | ---: |
| **80B first-allocation 4K-base working horizon** | 1,728h | 18.0 / 22.5 | 12,346 |
| 320B deferred scale comparison | 6,912h | 72.0 / 90.0 | 49,383 |
| 400B deferred scale comparison | 8,641h | 90.0 / 112.5 | 61,728 |

So 80B fits the 1,800-hour reservation with about 72 hours to spare, before long-context and agentic
overhead. Fitting a deferred 320B/400B scale into the same reservation would instead need about
**3.0× / 3.8× more effective throughput per allocated GPU**, which adding workers does not deliver:
four devices give about 51.4K aggregate tokens/s at perfect scaling or 41.1K at 80%, while consuming
four GPU-hours per elapsed hour. These elapsed times assume each device matches the measured H100
before communication losses, and 80% scaling is an illustration, not a measurement. GH200 per-device
gains, larger batches and implementation improvements are all unmeasured; qualify single-worker
efficiency and four-worker scaling separately.

The throughput recipe was selected on a 318M proxy, so remeasure the flagship rate on GH200 before
re-anchoring any horizon, under the derate and surplus rules predeclared in
`compute.throughput_reanchoring_rule`. Preserve continuation checkpoints and choose a compatible
learning-rate schedule before training; a fully decayed run does not extend at no cost.

Charge experimental training, evaluation, retries and on-allocation preparation to the appropriate
research cap; charge selected production runs to their production phase. CPU/storage and external
teacher API costs remain separate, as do historical H100 rental costs. Context token counts are
deliberately unset; the 600-hour cap cannot inherit the former 8B curriculum without evidence. No
extra experiment, 128K stage or RL campaign is implicitly funded beyond the declared caps.

## Preparation workstreams

The ordered next task is in [PLAN.md](../../PLAN.md#immediate-order-of-work); the items below are
preparation coverage, not simultaneous studies. The [readiness summary](../../PLAN.md#readiness-for-experiment-design)
separates inputs available for detailed design from the requirements for launching experiments.

1. Finish broader exclusions, family separation and origin/notice recovery for the code cohorts;
   the [qualification packet](QUALIFICATION.md) owns the current holds and origin counts. Qualify
   finite experiment arms from eligible supply before bulk packing. No examples are admitted.
2. Use the [retained-data closeout](../corpus-audit/data-readiness.json) for exact overlap, math
   normalization links and complete-document length inventory. Check solution correctness and
   source families before selecting math/refined candidates; well-formed text is not a checked answer.
3. Use full SFT format checks, sampled serialized lengths and RL prompt links to prepare stage packs.
   Resolve adapter/completeness issues and verify outcomes; recover missing long tails separately.
   Do not assume all existing 500K will survive qualification or count RL tasks as SFT trajectories.
4. Freeze actual admitted source manifests, weights, repetitions, exclusions and hardware cost
   before launch. Learning-rate schedule, batch geometry and checkpoint cadence belong to that
   run's contract. This plan starts preparation, not a new GPU run or bulk download.

## Qualification work

The [CPU qualification packet](QUALIFICATION.md) records eligibility rules, pinned exclusion inputs,
family separation and the next comparable source audit. It does not authorize training or change
the working numeric recipe above.
