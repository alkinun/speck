# Main data mixture and scale — working plan

2026-09-19. [plan.json](plan.json) records preparation targets and reproducible cost arithmetic.
This is not a launch configuration or a claim that the required corpus is already qualified.
The current pilot and rented-H100 evaluation remain unchanged.

## Scale

Prepare for **100B base-pretraining tokens**, followed by a provisional **8B-token context
extension toward 128K**. Aim for **1.5M unique post-training conversations**, with a 1–2M planning
range. The retained 500K rows are starting stock before final qualification, not the final target.
All token quantities use our frozen Mistral tokenizer; count actual exposure separately from unique
eligible supply. Holdouts are outside these training targets. A larger base horizon requires measured
throughput headroom; it must not consume the context-extension or post-training allocations.

## Base mixture

These are explicit starting hypotheses chosen for the code/math/agent target, not measured optimal
weights. Keep the mixture fixed for the first candidate; separate any later curriculum decision.

| Component | Share | Exposure | Candidate sources / admission condition |
| --- | ---: | ---: | --- |
| Selected broad natural web | 25% | 25B | Natural Ultra-FineWeb English; bind scored/HQ path and threshold after the bounded audit |
| Independent web coverage | 5% | 5B | FineWeb-Edu and/or DCLM; select allocation after overlap and coverage measurements |
| Natural code, tests and documentation | 30% | 30B | Qualified Stack-Edu and source-resolved UltraData-Code L2; preserve practical and multilingual coverage |
| Checked code explanations, exercises and repair | 5% | 5B | Qualified natural-code derivatives; UltraData-Code L3 only if lineage and independent checks succeed |
| Selected natural math / worked solutions | 20% | 20B | UltraData-Math L2, FineMath 4+, Nemotron-CC-Math `4plus`; source allocation follows comparative audit |
| Refined math explanations / derivations | 5% | 5B | Qualified UltraData-Math L3 and verified derivatives |
| Reference / science / technical documents | 5% | 5B | FineWiki, peS2o and qualified English FinePDFs-Edu |
| Refined educational web | 5% | 5B | Qualified Ultra-FineWeb-L3; Cosmopedia remains a comparison source |
| **Total** | **100%** | **100B** | **35% code, 25% math, 40% supporting web/reference/refined education** |

Every document has one primary bank, including cross-domain material such as mathematical code.
Deduplicate across banks and original/derived families before counting supply. Candidate names
are not interchangeable licenses or evidence that content is already downloaded.

Use existing released content first; this plan does not assume we can afford generating billions
of new teacher tokens ourselves. The 5B checked-code slot is a preparation target with a substantial
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

At uint16, 100B token IDs occupy **200GB decimal**; the 125B candidate bank occupies 250GB.
Budget indexes, masks, source text, deduplication workspaces, checkpoints and backups separately.
A 2TB local scratch allowance is a working envelope, not a measured dataset size. The September19
filesystem check showed approximately 4.5TiB available. Stream bounded acquisitions and retain
qualified packs; do not mirror all upstream datasets or assume the rental's 200GB disk can hold this.

## Context extension

Working stages: **2B tokens up to 16K, 2B up to 32K, 4B up to 128K**. These stage ceilings do not
mean every example is padded to the maximum. Within each stage reserve 25% of tokens for short
replay; the remaining 75% comes from coherent long material. Initially target half of the long
material from repositories, 30% from math/science/technical documents and 20% from grounded
cross-document material. These are preparation weights requiring content/length qualification.

Use repository relationships, imports, tests and documentation; preserve intact long reasoning
and agent traces for their relevant training phases. Record any reuse of base documents explicitly.
Freeze stage transitions only after measuring memory/runtime and evaluating retrieval across
positions, multi-file repair, cross-document reasoning and short-task retention. The current model
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

## Compute allocation

The requested four-GH200 / 5,000-GPU-hour allocation is not yet confirmed. Use this full-envelope
reservation without treating previous rental reservations as actual provider billing:

| Work | GPU-hours reserved |
| --- | ---: |
| Runtime qualification | 70 |
| Engineering pilot | 50 |
| Main 4K pretraining | 2,300 |
| Context extension | 800 |
| Post-training, including any on-allocation teacher/reward work | 800 |
| Bounded data comparison | 91 |
| Protected evaluation and recovery | 889 |
| **Total** | **5,000** |

The completed pilot measured **12,859 tokens/s** over the full trainer process, including cold
startup, validation and checkpoint saves. At that single-H100 rate, 100B takes **2,160 GPU-hours**.
The 2,300-hour reservation leaves about 140 hours beyond that linear extrapolation. Four workers
at perfect scaling would take 22.5 days; this is arithmetic, not measured GH200 wall time. Charge
all allocated GPUs, including idle workers. Hardware, communication, input loading and cadence
can change the rate. GH200/multiworker qualification is still required.

| Base target | H100 full-trainer-rate extrapolation | Effective tokens/s per allocated GPU needed within 2,300h |
| --- | ---: | ---: |
| 80B fallback | 1,728h | 9,662 |
| **100B working target** | **2,160h** | **12,077** |
| 120B conditional extension | 2,592h | 14,493 |
| 160B conditional extension | 3,456h | 19,324 |

The effective-rate thresholds include all work charged to the base reservation. Expand only after
measured throughput, usable supply and learning curves justify it while preserving the other
reservations. If 100B does not fit, reduce the base horizon or improve measured execution; do not
quietly sacrifice 128K or the thinking-model training. The separate 8B context target needs an
effective average of 2,778 tokens/s per allocated GPU to fit 800 hours; its actual rate is unknown.

Charge shared-base creation to the main allocation once. Charge additional data-comparison arms
and their evaluations to the 91-hour comparison reservation. Synthesis/verification GPU costs
must be charged to their relevant preparation or post-training budget; CPU/storage and external
teacher API costs need separate accounting. Prefer existing eligible data over unbudgeted mass
synthesis. No optional RL phase or extra experiments are implicitly funded beyond these limits.

## Next bounded work

1. Audit the pinned natural Ultra-FineWeb candidate and its chosen threshold against retained
   FineWeb-Edu using the existing sampler; report content coverage, overlap and eligible tokens.
2. Qualify the 16-file practical-code cohort, establish a scalable acquisition/provenance route,
   then measure a bounded larger shard. Close code supply feasibility before bulk packing.
3. Compare natural/refined math and educational candidates, keeping claimed checked solutions
   distinct from merely well-formed text. Measure source overlap and complete-document lengths.
4. Audit the retained SFT stock and long-source gaps against the 1.5M task/length targets. Build a
   unique-task inventory; do not assume all existing 500K will survive qualification.
5. Freeze actual admitted source manifests, weights, repetitions, exclusions and hardware cost
   before launch. Learning-rate schedule, batch geometry and checkpoint cadence belong to that
   run's contract. This plan starts preparation, not a new GPU run or bulk download.
