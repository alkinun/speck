# Main data mixture and scale — working plan

2026-09-20. [plan.json](plan.json) records preparation targets and reproducible cost arithmetic.
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
[frontier-data-research.json](frontier-data-research.json); source-specific gate updates are next.
No source is admitted and no study packet authorizes acquisition or training until the closeout gates
are updated with pinned evidence.

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

Validate the packet and its source-of-truth hashes offline:

```bash
PYTHONPATH=. python experiments/main-data/check_data_study.py \
  experiments/main-data/data-study-packet.json
```

The command checks design arithmetic and launch boundaries only. It does not acquire sources,
execute corpus content or authorize training.

The packet is deliberately design-only. Before any arm can run, every selected source needs named
source-use decisions, family and near-duplicate partitions, independent correctness checks where
claimed, finite accepted-token counts, a disjoint evaluation pack and measured GH200 cost. A future
launch request must bind the qualified manifests and the existing rights/operations/firewall records.
The packet does not authorize acquisition, training or a main-run mixture.

## Capability mid-training efficiency packet

[mid-training-study-packet.json](mid-training-study-packet.json) defines the next data-efficiency study. It compares replay, source-only repository data, grounded workflows and executable trajectories, then separately tests selective loss/packing and context transitions. Validate it offline:

```bash
PYTHONPATH=. python experiments/main-data/check_mid_training_study.py \
  experiments/main-data/mid-training-study-packet.json
```

The packet reserves 300 GPU-hours for mid-training research. Capability data interventions must use
`--branch-kind data` with `training_phase: data_continuation`; context interventions must use the
separate context branch. Linked SFT/RL convergence and adaptation measurements belong to the separate
post-training research reservation and must use the same downstream recipe across retained arms.

## Post-training data-study packet

[post-training-study-packet.json](post-training-study-packet.json) binds SFT selection, fixed-policy RL feasibility and a bounded final self-SFT pilot. It does not authorize policy updates or production self-SFT. Validate it offline:

```bash
PYTHONPATH=. python experiments/main-data/check_post_training_study.py \
  experiments/main-data/post-training-study-packet.json
```

The packet reserves 80 GPU-hours for SFT comparison, 20 for RL feasibility, 30 for the final self-SFT pilot and 20 for support. Existing 500K-row stock remains format and context evidence until source, family, correctness and held-out gates close.

Run `make plan-check` from the repository root to validate the cross-stage compute ledger, all
packet budgets and every recorded input receipt together. A passing check is necessary bookkeeping,
not training authority.

The ledger checker validates direct plan receipt hashes; the stage validators check their own
declared dependencies. This is not a recursive audit of every historical receipt or evidence of
source eligibility. Study packets own arm counts and per-arm ceilings; mirrored plan fields must
match them. Match training exposure for data comparisons and report actual GPU-hours separately;
equal exposure and equal elapsed cost are not generally achievable together.

The [source-readiness matrix](source-readiness.json) is the companion evidence index. It records the retained inventory, source-of-truth receipts, open gates and blocked arm status for each candidate bank. Validate it offline:

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

The [candidate manifest preflight](candidate-manifest-preflight.json) normalizes those requirements
across all 12 candidates. It currently finds zero complete manifests and zero eligible tokens; this is
the checklist for source-specific qualification, not a launch manifest.

The [natural-web candidate manifest](natural-web-candidate-manifest.json) is the first source-specific
application. It binds the FineWeb-Edu control and Ultra-FineWeb HQ candidate, preserves the measured
overlap, and leaves source-use, family, contamination, eligible-supply and runtime gates open.

The [natural-code candidate manifest](natural-code-candidate-manifest.json) now binds Stack-Edu,
Stack v3 and the separate checked-code substitution route. It preserves the finite-stock and cohort
limits, with zero eligible code tokens established and all provenance/correctness gates open.

The [math candidate manifest](math-candidate-manifest.json) now separates FineMath, UltraData-Math,
InfiWebMath, Nemotron and review-only OpenMath/L3 candidates. It preserves arithmetic triage as a
quality lead rather than a correctness certificate and leaves all math sources outside training arms.

The [post-training candidate manifest](post-training-candidate-manifest.json) now binds the 500K-row
assistant census and the 44,369-row reward-prompt inventory. It keeps structural serialization,
verified outcomes, tool trajectories and fixed-policy RL feasibility separate; no downstream rows are
admitted.

The [post-training audit protocol](post-training-audit-protocol.json) fixes the next bounded review:
deterministic SFT structure, stratified independent outcomes, tool-trajectory checks, reasoning-mode
measurement, and fixed-policy verifier feasibility with explicit stop rules. It measures efficiency
only after correctness, using task-conditioned length curves rather than a global shortness reward.
Its horizon accounting records the 80B/100B working target separately from the current one-pass bounds: retained code evidence bounds a 30% share at 1.589B total tokens before exclusions, and HQ tokens distinct from retained FineWeb-Edu bound a 25% share at 1.405B. These are constraints, not qualified supply.
The HQ comparison retains FineWeb-Edu as the control, keeps the pinned L1/HQ route as a candidate, and adopts no score cutoff or automatic repair rule; its sampled panels establish review evidence, not eligible yield.
The natural-code route is similarly bounded: the retained Stack-Edu census has 0 eligible tokens established, the fixed Stack-Edu/Stack v3 cohorts retain all 44 family holds, and unresolved origins/404s remain outside any arm. Checked-code substitution stays separate until provenance and correctness gates close.
The math route keeps FineMath 4+, UltraData-Math L2 and filtered InfiWebMath 4+ as separate candidates: the reversible FineMath directory view removes 1.67% of its stock as a conservative candidate, while all L2 rows still lack host metadata. The InfiWebMath sample has a conservative arithmetic triage and a hashed manual classification; it is not a correctness certificate. A single normalized cross-source link is recorded without automatic removal; Nemotron-CC-Math remains unavailable because its listed LFS hashes are unusable and no shards were acquired.
The external Cagliostro v3 review identifies OpenMathInstruct-2, FineMath's InfiWebMath 3+ subset and SmolTalk as additional candidates. The [candidate review](../corpus-audit/recipe-review.json) links bounded viewer diagnostics for OpenMath and FineMath/InfiWebMath, and the source-readiness matrix now records the separately qualified InfiWebMath 4+ candidate. OpenMath and SmolTalk remain outside that matrix; every candidate remains outside retained inventory and study arms until source-use, family/contamination, correctness, finite-supply and runtime evidence closes. A late math reweighting is a schedule hypothesis, not an additional arm.

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
| Architecture | Two arms × one seed × 60h = 120h; profiling 40h | 40h | 200 |
| Pretraining data | Base screen: four arms × 30h = 120h; decay screen: three arms × 30h = 90h; confirmation: two arms × two new seeds × 60h = 240h | 150h | 600 |
| Mid-training data | Proxy screen 120h; objective/packing 40h; context 40h; 1.2B confirmation 80h | 20h | 300 |
| Post-training data | SFT: two arms × two seeds × 20h = 80h; conditional RL feasibility 20h; final self-SFT pilot 30h | 20h | 150 |

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
The source names below are production hypotheses; the experimental baseline deliberately substitutes
its declared control source in the bank under study. Update the production source choices after
confirmation rather than assuming every preferred candidate wins.

| Component | Share | 80B exposure | Eligible unique preparation | Candidate sources / admission condition |
| --- | ---: | ---: | ---: | --- |
| Selected broad natural web | 25% | 20B | 25B | Natural Ultra-FineWeb English; bind scored/HQ path and threshold after the bounded audit |
| Independent web coverage | 5% | 4B | 5B | FineWeb-Edu and/or DCLM; select allocation after overlap and coverage measurements |
| Natural code, tests and documentation | 30% | 24B | 30B | Stack-Edu, source-resolved UltraData-Code L2, and Stack v3 under qualification; preserve practical and multilingual coverage |
| Checked code explanations, exercises and repair | 5% | 4B | 5B | Qualified natural-code derivatives; UltraData-Code L3 only if lineage and independent checks succeed |
| Selected natural math / worked solutions | 20% | 16B | 20B | UltraData-Math L2, FineMath 4+, Nemotron-CC-Math `4plus`; source allocation follows comparative audit |
| Refined math explanations / derivations | 5% | 4B | 5B | Qualified UltraData-Math L3 and verified derivatives |
| Reference / science / technical documents | 5% | 4B | 5B | FineWiki, peS2o and qualified English FinePDFs-Edu |
| Refined educational web | 5% | 4B | 5B | Qualified Ultra-FineWeb-L3; Cosmopedia remains a comparison source |
| **Total** | **100%** | **80B** | **100B** | **35% code, 25% math, 40% supporting material** |

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
These figures do not establish qualified supply for the new recipe. Prepare a **100B eligible unique
token bank** as a 25% selection margin over 80B exposure: each bank's target is 1.25 times its
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

The grant allocation confirms four GH200s and 5,000 total GPU-hours within a nominal 90-day window;
the access start, site details and hardware throughput remain unconfirmed. Use this reservation without treating previous rental reservations
as actual provider billing:

| Work | GPU-hours reserved |
| --- | ---: |
| Runtime and reference efficiency qualification | 120 |
| 4K base production: 1,550 stable phase / 250 endpoint decay | 1,800 |
| Capability/context/agentic mid-training production | 600 |
| Post-training: 450 SFT / 200 conditional RL / 100 final self-SFT / 50 teacher and verification | 800 |
| Data experiments: 700 pretraining / 360 mid-training / 170 post-training | 1,230 |
| Protected evaluation and recovery | 450 |
| **Total** | **5,000** |

The architecture and efficiency study no longer holds a reservation. Its 200 hours were released on
2026-09-21: 180 to data research and 20 to reference-only efficiency profiling inside runtime
qualification.

The completed pilot measured **12,859 tokens/s** over the full trainer process, including cold
startup, validation and checkpoint saves. At that single-H100 rate, 80B takes **1,728 GPU-hours**.
The 1,800-hour reservation leaves about 72 hours before long-context overhead. Four workers
at perfect scaling would take 22.5 days; this is arithmetic, not measured GH200 wall time. Charge
all allocated GPUs, including idle workers. Hardware, communication, input loading and cadence
can change the rate. GH200/multiworker qualification is still required.

| Base scenario | H100-rate GPU-hours | Four-GPU elapsed days, ideal / illustrative 80% scaling | Effective tokens/s per GPU needed within 1,800h |
| --- | ---: | ---: | ---: |
| **80B first-allocation 4K-base working horizon** | 1,728h | 18.0 / 22.5 | 12,346 |
| 320B deferred scale comparison | 6,912h | 72.0 / 90.0 | 49,383 |
| 400B deferred scale comparison | 8,641h | 90.0 / 112.5 | 61,728 |

These elapsed times assume each device matches the measured H100 before communication losses;
80% scaling is an illustration, not a measurement. Four devices would deliver about 51.4K tokens/s
in aggregate at perfect scaling, or 41.1K at 80%. They consume four GPU-hours per elapsed hour:
5,000 aggregate GPU-hours permit 1,250 four-GPU hours (52.1 days), not 5,000 machine-hours.
The recorded 90-day access window does not establish 90 days of continuous four-GPU funding.
The allowance is **5,000 total GPU-hours**. The allocation is confirmed, but provider start timing
and runtime qualification remain unconfirmed; the budget
interpretation is explicit.

At the H100 reference rate, keeping the other 3,200 reserved GPU-hours brings the full program to
approximately **9,612 GPU-hours for 320B** or **11,341 for 400B**, before any distributed penalty.
Alternatively, fitting the base into its current 1,800-hour reservation requires about **3.0× / 3.8×
more effective throughput per allocated GPU**. Merely adding four workers does not deliver that
per-GPU improvement. GH200 hardware gains, larger batches and implementation improvements remain
unmeasured; qualify single-worker efficiency and four-worker scaling separately. Do not assume the
engineering pilot is an optimized throughput ceiling or promise a particular speedup.

The 80B 4K-base horizon is now the working preparation baseline, conditional on cost and supply;
the 100B and larger comparisons do not authorize expansion. Preserve continuation checkpoints and
choose a compatible learning-rate schedule before training; a fully decayed run does not extend at
no cost. The research allocation reserves 1,230 hours for data experiments, while preserving 1,800
base, 600 mid-training and 800 post-training production hours. Historical H100 rental costs are
separate. The backbone no longer changes under a study decision, but the throughput recipe was
selected on a 318M proxy, so remeasure the flagship rate on GH200 before re-anchoring any horizon.

Charge experimental training, evaluation, retries and on-allocation preparation to the appropriate
research cap; charge selected production runs to their production phase. CPU/storage and external
teacher API costs remain separate. Context token counts are deliberately unset; the 600-hour cap
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
   The [reading closeout](../corpus-audit/stylesheet-cohort-review.json) completes all 174 currently
   unheld records in the fixed 138-file Stack-Edu and 80-file Stack v3 cohorts. The cumulative 176
   full reads include two now-held records; all 44 family holds remain intact. Page, template and
   component stylesheets retain their original labels, weights and document boundaries. The
   [notice follow-up](../corpus-audit/code-notice-provenance.json) adds seven verified host origins and
   reconfirms one. The [pinned-origin follow-up](../corpus-audit/pinned-code-origins.json) adds 42 more:
   the [retained-origin recovery](../corpus-audit/retained-code-origins.json) adds ten exact-match
   Stack-Edu origins. Coverage is now 28/104 unheld Stack-Edu and 68/70 unheld Stack v3. GitHub quota
   exhaustion blocks 75 records; one earlier Stack-Edu failure and two Stack v3 404s remain separate.
   Resume quota-blocked work only after access changes; continue independent web/math work meanwhile.
   Host-origin recovery and notice availability do not establish source-use approval or admission.
   Complete broader lineage and remaining notice/revision checks. The controlled syntax diagnostic
   confirms one redaction failure; semantic preservation and source-use/family gates remain unresolved.
   Qualify finite experiment arms from eligible supply before bulk packing. No examples are admitted.
3. Use the [retained-data closeout](../corpus-audit/data-readiness.json) for exact overlap, math
   normalization links and complete-document length inventory. Check solution correctness and
   source families before selecting math/refined candidates; well-formed text is not a checked answer.
4. Use full SFT format checks, sampled serialized lengths and RL prompt links to prepare stage packs.
   Resolve adapter/completeness issues and verify outcomes; recover missing long tails separately.
   Do not assume all existing 500K will survive qualification or count RL tasks as SFT trajectories.
5. Freeze actual admitted source manifests, weights, repetitions, exclusions and hardware cost
   before launch. Learning-rate schedule, batch geometry and checkpoint cadence belong to that
   run's contract. This plan starts preparation, not a new GPU run or bulk download.

## Qualification work

The [CPU qualification packet](QUALIFICATION.md) records eligibility rules, pinned exclusion inputs,
family separation and the next comparable source audit. It does not authorize training or change
the working numeric recipe above.
