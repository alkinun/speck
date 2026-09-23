# Main data mixture and research design

2026-09-23. This file owns the base mixture and the pre-main-run research design;
[plan.json](plan.json) owns the numbers. The [program overview](../../docs/program.md) owns stages,
context extension, post-training scale and compute; [PLAN.md](../../PLAN.md) owns status and the
work order; the [qualification packet](QUALIFICATION.md) owns qualification state. Nothing here is
a launch configuration, admits a source or authorizes training.

The 1.2B backbone is **fixed by declaration, not selected over a control**. The
[architecture study packet](architecture-study-packet.json) preserves the deferred comparison
unmodified and draws zero hours from this allocation.

## Scale

The working horizon is **80B 4K-base tokens**. At the measured H100 full-trainer rate it projects to
1,728 of the reserved 1,800 4K-base GPU-hours; at an illustrative 80% of that rate only 66.7B fits.
Freeze the final horizon from qualified supply, GH200 cost and the separate 16K/32K mid-training
costs; 100B and the former 320B/400B scales are deferred. Prepare a **100B eligible unique token
bank**, 1.25 times each bank's exposure, and default to one pass. Holdouts are outside training
supply, and exposure/replay is counted separately from unique eligible material.

## Study packets

Each packet is design-only and validated offline by `make plan-check`, which also checks the
cross-stage ledger and every recorded reference. A passing check is bookkeeping, not training
authority. Study packets own arm counts and per-arm ceilings; mirrored plan fields must match.

- [data-study-packet.json](data-study-packet.json) binds one common baseline, the code-bank,
  natural-web and AI-generation/provenance-filter contrasts, and a three-way decay comparison, with
  fixed tokenizer, 4K context, objective, serialization, exposure accounting and evaluation.
- [mid-training-study-packet.json](mid-training-study-packet.json) compares replay, source-only
  repository data, grounded workflows and executable trajectories, then tests selective loss/packing
  and context transitions separately. Data interventions use `--branch-kind data`; context
  interventions use the context branch.
- [post-training-study-packet.json](post-training-study-packet.json) binds SFT selection, fixed-policy
  RL feasibility and a bounded final self-SFT pilot; it authorizes no policy updates.

Before any arm runs, every selected source needs its source-use conditions met (see the signed
[acceptance record](source-rights-acceptance.json)), family and near-duplicate partitions,
correctness checks where claimed, finite accepted-token counts, a disjoint evaluation pack and
measured GH200 cost.

The records behind the packets:

- The [source-readiness matrix](source-readiness.json) owns per-source identity, inventory, gates,
  blockers and manifest-field completeness; it finds zero complete manifests and zero eligible tokens.
- The [data-design contract](data-design-contract.json) fixes the manifest fields every stage keeps:
  stage, source family, transformation, quality, coverage, dependency, contamination and lineage.
- The [source mapping](frontier-data-source-mapping.json) ties each
  [research finding](frontier-data-research.json) to local evidence; it closes no gate.
- The [natural-web](natural-web-candidate-manifest.json), [natural-code](natural-code-candidate-manifest.json),
  [math](math-candidate-manifest.json) and [post-training](post-training-candidate-manifest.json)
  candidate manifests add domain evidence and comparison contracts. Arithmetic triage is a quality
  lead, not a correctness certificate.
- The [post-training audit protocol](post-training-audit-protocol.json) fixes the next bounded review
  and its stop rules, measuring efficiency only after correctness.

## Research before the main run

Numeric caps live in [plan.json](plan.json). Bind exact datasets, seeds, horizons, learning rates and
decision thresholds before execution. Runtime qualification precedes pretraining data runs;
downstream runs need useful parent checkpoints.

| Study | Proposed comparison and controls | Decision evidence |
| --- | --- | --- |
| Pretraining screening | One qualified baseline plus at most three single-factor candidates: code-bank, natural-web-bank and calibrated AI-generation/provenance-filter contrasts. Same fresh initialization, other banks, domain shares, packing, objective and exposure. | Fixed source losses and development curves rank candidates for confirmation; one screening seed establishes no robust winner. |
| Pretraining confirmation | Baseline versus one selected candidate, each from two new paired initialization seeds. Identical initial tensors within each pair; fresh optimizer/data state. | Predeclared primary endpoint, consistent paired effects, acceptable regressions and affordable supply. Publish both seed pairs and inconclusive outcomes. |
| Capability mid-training | Replay/source-only control versus targeted code/math/repair/tool-use grounding, with validated executable trajectories only if environments qualify. Same useful parent, 4K next-token objective, downstream SFT/RL recipe, optimizer policy, schedule and exposure; one paired data-order seed. | Target capability, source losses, downstream SFT convergence, early RL adaptation and quality per mid-training token/GPU-hour. Branches are additive ablations; context extension stays separate. |
| Context mid-training | At 16K, coherent records in preceding domain proportions versus proposed long-domain reweighting. Same parent, short replay, packing and token exposure; one paired seed. | Fixed-suffix loss at matched long prefix length, related-prefix benefit, positional/multifile checks and short-task retention. This does not isolate coherent packing or context length. |
| Thinking SFT | Eligible baseline versus outcome-verification selection from the same candidate pool. Same parent, task/length strata, masking, serialization and schedule; two paired data-order seeds with fresh optimizers. | One declared code/math or tool-task success endpoint, with general/protocol regressions and supervision density reported. Known-invalid examples enter neither arm. |
| RL prompt feasibility | Conditional fixed-policy rollouts from the selected SFT checkpoint on candidate eligible prompt strata. | Verifier reliability, solvable difficulty, nontrivial success/failure groups and rollout cost. No RL training-data ablation is promised by this slot. |

The baseline must itself pass eligibility. If a bank lacks qualified supply, revise the baseline
before making any arm. The code candidate is the natural-source contrast (Stack-Edu versus Stack
v3); the natural-web candidate compares a qualified FineWeb-Edu control with Ultra-FineWeb HQ at a
fixed bank share, without also tuning the score threshold or serialization. Alternatives enter only
through a recorded replacement before launch, not an expanding sweep.

A shared screening baseline is valid only if every candidate has the same unchanged settings.
If several candidates meet their frozen screening criteria, confirm the code candidate when it is
among them; otherwise choose among the passing candidates by a rule declared before screening. Raw
loss deltas from different domains are not directly comparable. If none passes, keep the eligible
baseline or record a redesign within the remaining cap. Do not combine two individually favorable
changes without testing the combination. Confirmation uses fresh runs and seeds, not extensions of
screening checkpoints. Research exposure and discarded arms never count toward the main horizon, and
main pretraining starts fresh after the recipe decision.

Capability arms branch from one preserved 4K production milestone before final LR decay. The decay
study branches from one matched stable checkpoint and compares natural, curated-natural and small
verified-derived enrichment under the same decay schedule. Context arms use the selected capability
endpoint, SFT arms the same post-context base, and RL prompt checks the selected SFT checkpoint.
Production then restarts from the unchanged parent with the selected recipe.

### Research cost envelopes

These are proposed ceilings within existing reservations, not measured durations or runnable jobs.

| Reservation | Training / study slots | Shared support | Total GPU-hours |
| --- | --- | ---: | ---: |
| Pretraining data | Base screen: four arms × 30h = 120h; decay screen: three arms × two seeds × 30h = 180h; confirmation: two arms × two new seeds × 60h = 240h | 160h | 700 |
| Mid-training data | Proxy screen 120h; objective/packing 40h; context 80h; 1.2B confirmation 80h | 40h | 360 |
| Post-training data | SFT: two arms × two seeds × 20h = 80h; conditional RL feasibility 40h; final self-SFT pilot 30h | 20h | 170 |

Training slots include trainer startup, inline validation and saves; shared support covers other
on-allocation preparation, scoring, qualification and recovery. Charge each operation once. The RL
prompt study funds no optimizer updates; RL implementation and updates fit the separate 200h
production subdivision.

Freeze one common token horizon per base-training comparison from the slowest arm's measured cost,
qualified supply and required signal. Hour caps are ceilings, not instructions to spend. At the
historical H100 rate, 30h, 40h and 60h correspond to about 1.39B, 1.85B and 2.78B tokens; these
illustrate scale only. Retained natural-code stock bounds a one-pass baseline at 1.36B total tokens,
so even these arms need more qualified supply or shorter horizons.

Protect confirmation first. If timing or supply does not fit, drop the secondary screening candidate
or revise the common horizon before launch; never spend confirmation, production or recovery funds
on more screening. An inconclusive comparison can support a documented baseline choice, not a
positive data-efficiency claim. Deduplication strength, quality thresholds, coverage-aware sampling
and curricula are CPU-measurable follow-ups, not extra funded runs; never relax benchmark-family
separation as an ablation.

### Measurements and selection

1. Freeze the union of exclusions and source-family partitions across all arms and stages. Keep
   validation and development families disjoint from training and final evaluation, and use one
   common validation pack independent of arm manifests.
2. Before screening, name one primary endpoint per contrast, its direction, aggregation weights and
   minimum useful effect. For source substitutions use held-out target-domain next-token loss,
   reporting code, math and supporting-domain losses separately; capability and retention are
   guardrails. Set tolerances from baseline noise before viewing treatment results.
3. Use matched quarter/half/final exposure checkpoints for loss curves and score capability at
   declared milestones. Failed arms and overruns stay in the result ledger; no undisclosed reruns.
4. Screening is exploratory. Lock the selected contrast, schedule, endpoint, tolerances and new seeds
   before confirmation. Report paired differences per seed; bootstrap task families or documents,
   not correlated tokens. One-seed pairs remain exploratory.
5. Promote a candidate only if the useful-effect criterion, guardrails, seed consistency and
   cost/supply gates pass. Loss-only gains remain loss findings. Score final tasks only after
   recipe/checkpoint decisions are frozen.

Do not average unrelated loss, accuracy and latency into an efficiency score. For context, score the
identical held-out suffix with matched long prefixes across arms, and compare related prefixes with
short/irrelevant-prefix controls.

### Runtime and launch requirements

The [training guide](../../docs/training.md#mid-training-readiness) records the changed-data branch
gap; qualify it before either capability arm, and do not use context branches to bypass it. Context
studies also need useful long records, restart checks and the fixed-suffix scorer. SFT arms must
match processed positions and bucket schedules within a predeclared supervised-token tolerance;
equal row counts do not match exposure.

Before each launch, retain the source/config revision, parent or initialization identities,
qualified data/exclusion manifests, objective/masks, exposure/batch/schedule, evaluation identities,
decision/stop thresholds, seeds, all-in cost estimate and recovery destination.

## Base mixture

These are starting hypotheses for the code/math/agent target, not measured optimal weights. The
sources are the [registry's](source-registry.json) selection; the experimental baseline substitutes
its declared control source in the bank under study.

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

The mixture was re-frozen over these six banks on 2026-09-22. `checked_code` (5%) and `refined_math`
(5%) held zero retained stock, so each capped a one-pass run at zero tokens. Their shares merged
**within domain** into `natural_code` and `natural_math`, so exposure stays 80B, preparation stays
100B and the 35/25/40 split is unchanged. Checked code and refined math are now unbanked derived
candidates: qualifying either needs a revised freeze and its own exposure ledger. Checked code,
Nemotron-CC-Math `4plus` and the UltraData-Math L2 preview were not selected; DCLM, UltraData-Code L2,
Ultra-FineWeb-L3 and English FinePDFs-Edu are unbanked alternatives.

Every document has one primary bank. Deduplicate across banks and original/derived families before
counting supply. If a source fails qualification, re-freeze its share within the same domain or
shorten the horizon; never silently repeat a small set. Preserve implementation, API usage, testing,
build and debugging roles in code, prioritizing Python and JavaScript/TypeScript with systems, JVM,
SQL and shell coverage; set language weights from measured supply. Math should span foundational
worked problems and harder derivations.

## Acquisition and storage

Retained stock and the per-bank gap are derived in [supply-gap.json](supply-gap.json). Code
acquisition is the critical gate: the [Stack-Edu card](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)
provides content locators rather than files, so fetching and qualifying source bytes is part of the
work. Publisher corpus sizes do not establish our eligible supply.

At uint16, 80B training token IDs occupy **160GB decimal** and the 100B preparation bank 200GB.
Budget indexes, masks, source text, deduplication workspaces, checkpoints and backups separately;
the 2TB scratch allowance is a working envelope. Stream bounded acquisitions; do not mirror whole
upstream datasets.

## Post-training scale

The [program overview](../../docs/program.md#thinking-sft) owns the 1.5M-conversation target and
its task split. At the measured 4K SFT padded-position rate, one pass at a mean of 8K or 16K total
tokens (12B or 24B context tokens) is an arithmetic proxy of 247 or 494 GPU-hours, not a
long-context runtime prediction; measure the actual length mixture before claiming it fits.
