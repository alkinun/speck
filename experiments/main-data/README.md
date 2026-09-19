# Main data mixture and scale — working plan

2026-09-19. [plan.json](plan.json) records preparation targets and reproducible cost arithmetic.
This is not a launch configuration or a claim that the required corpus is already qualified.
The frozen H100 pilot, development evaluation and backups are complete. This working plan does
not change their configurations or historical result receipts.
The first 5,000-GPU-hour program studies data and training across pretraining, mid-training and
post-training on the selected 1.2B backbone. The [program](../../docs/program.md#training-lifecycle)
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
allowance. Follow with a provisional **8B-token context extension toward 128K** and **1.5M unique
post-training conversations**, within a 1–2M planning range. These require their own cost and
quality qualification. All tokens use the frozen Mistral tokenizer; holdouts are outside training
supply, and exposure/replay is distinct from unique eligible material.

## Base mixture

These are explicit starting hypotheses chosen for the code/math/agent target, not measured optimal
weights. The table extrapolates the initial mixture across the base horizon. Freeze capability
mid-training weights separately and update aggregate bank exposures if a staged mixture is adopted.

| Component | Share | 100B exposure | Eligible unique preparation | Candidate sources / admission condition |
| --- | ---: | ---: | ---: | --- |
| Selected broad natural web | 25% | 25B | 31.25B | Natural Ultra-FineWeb English; bind scored/HQ path and threshold after the bounded audit |
| Independent web coverage | 5% | 5B | 6.25B | FineWeb-Edu and/or DCLM; select allocation after overlap and coverage measurements |
| Natural code, tests and documentation | 30% | 30B | 37.5B | Qualified Stack-Edu and source-resolved UltraData-Code L2; preserve practical and multilingual coverage |
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
not establish our eligible supply. The natural-code 16-file cohort qualifies the process, not scale.

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

Keep the controlled natural-code/checked-exercise study within its separate 141-hour reservation.
Do not count experimental arm tokens as main-model exposure unless that arm is actually continued.

## Context extension

Working stages: **2B tokens up to 16K, 2B up to 32K, 4B up to 128K**. These stage ceilings do not
mean every example is padded to the maximum. Within each stage reserve 25% of tokens for short
replay; the remaining 75% comes from coherent long material. Initially target half of the long
material from repositories, 30% from math/science/technical documents and 20% from grounded
cross-document material. These are preparation weights requiring content/length qualification. Use the preceding stage's
unchanged domain mixture, repacked into coherent longer records, as the control before adopting
these long-material weights. The [MAI review](../../docs/research.md#mai-thinking-1-review--2026-09-19)
supports testing this simpler control; it does not validate our KDA/GQA context extension.

Use repository relationships, imports, tests and documentation; preserve intact long reasoning
and agent traces for their relevant training phases. Record any reuse of base documents explicitly.
Freeze stage transitions only after measuring memory/runtime and evaluating retrieval across
positions, multi-file repair, cross-document reasoning and short-task retention. Also compare
held-out loss on the same suffix as related prefix length increases; stratify retrieval/QA by answer
position and distractor distance. The current model
has not been qualified at these lengths. Eight billion tokens is a budgeted hypothesis, not proof
that 128K will work. Add an intermediate length only if measured stability/cost calls for it.

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
| Runtime qualification | 70 |
| Pretraining and capability mid-training | 2,300 |
| Context mid-training | 800 |
| Post-training, including any on-allocation teacher/reward work | 800 |
| Bounded data comparison | 141 |
| Protected evaluation and recovery | 889 |
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
The retired 50-hour future pilot reservation now increases the bounded data study from 91 to
141 hours. Historical H100 rental costs stay separate. There is no architecture-search allocation.
The separate 8B context target needs an effective average of 2,778 tokens/s per allocated GPU to
fit 800 hours; its actual rate is unknown.

Charge shared-base creation to the main allocation once. Charge additional data-comparison arms
and their evaluations to the 141-hour comparison reservation. Synthesis/verification GPU costs
must be charged to their relevant preparation or post-training budget; CPU/storage and external
teacher API costs need separate accounting. Prefer existing eligible data over unbudgeted mass
synthesis. No optional RL phase or extra experiments are implicitly funded beyond these limits.

## Preparation workstreams

The ordered next task is in [PLAN.md](../../PLAN.md#immediate-order-of-work); the items below are
preparation coverage, not five simultaneous studies.

1. Audit the pinned natural Ultra-FineWeb candidate and its chosen threshold against retained
   FineWeb-Edu using the existing sampler; report content coverage, overlap and eligible tokens.
2. Use the [completed practical checks](../corpus-audit/practical-code-checks.json) and
   [qualification packet](QUALIFICATION.md) to finish broader exclusions and family separation.
   No examples are admitted. Establish a scalable acquisition route, then measure a bounded
   larger shard before bulk packing.
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
