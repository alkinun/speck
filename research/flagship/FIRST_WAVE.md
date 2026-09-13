# First-wave pretraining proposal

**Status: concrete proposal for review, not a frozen launch manifest.** This turns the open recipe
discussion into a run list and a bounded preparation target. Thinking-only post-training is being
explored separately; this proposal concerns the base model's training data.

**Review update, 2026-09-13:** the owner requested consideration of MiniCPM4 and the current UltraData
corpora before freezing recipes. The [source review](../literature/50_minicpm4_ultradata.md) adds serious
Code/Math candidates. The v1 run list and 17.5B envelope describe the preserved proposal; recompute
them after a budgeted source/tier revision is chosen. No new corpus is approved by this review.

## What we want to learn

1. **Source quality:** which source treatments improve a complete, balanced training mixture?
2. **Repetition:** how much reuse can the chosen incumbent tolerate at the same training-token budget?

The model, optimizer, training horizon, tokenizer, and category weights are held fixed within each
source comparison. Architecture selection belongs to its own experiments; these screens do not sweep
architecture or optimize the six category weights. E2 subsequently chooses the stable mixture.

## Proposed source recipes

The complete prior remains web/code/math/synthetic/science/reference = **55/15/10/10/5/5**.
The proposed incumbent is FineWeb-Edu, Stack-Edu, FineMath-4+, Cosmopedia v2, peS2o v3, and FineWiki.
It is a high-information starting prior chosen before screen results, not an established optimal mix.
Its education-heavy character limits how broadly E3 repetition results can be generalized.

| Screened category | Proposed treatments inside that category |
| --- | --- |
| Web | Ultra-FineWeb v1.4; FineWeb-Edu; DCLM; equal blend of those three |
| Code | restricted Stack v3; Stack-Edu; 50/50 blend |
| Math | FineMath-4+; MegaMath Web-Pro; 50/50 blend |
| Synthetic | Cosmopedia v2; Ultra-FineWeb-L3 Multi-Style; 50/50 blend |

Every arm replaces only its tested category; the other five keep the incumbent recipe. Equal blends
are deliberately simple pre-results hypotheses. This proposal replaces the earlier suggested web
35/30/25/10 tokenizer-sampling proportions with an equal-primary-source control. It does not add
FineWeb-base or code-prose components to that control. Existing source-use approval covers the proposed
IDs; exact language allocations, reader/filter variants, and source-capacity qualifications still need
their executable bindings.

The owner previously requested discussion before a freeze. These choices remain reviewable; the
proposal makes no claim that the owner approved the incumbent or revised blends.

## Experiment order and budget

| Work | Nominal model / tokens per run | Logical slots | Existing GPU-hour ceiling |
| --- | --- | ---: | ---: |
| E1W web screen + finalist replication | 350M / 8B | 4 seed-42 screens + 4 confirmation slots | 133 |
| E1S specialist screens + finalist replication | 150M / 2B | 9 seed-42 screens + 6 confirmation slots | 27 |
| E3 repetition | 150M / 6B | 1/2/4 epochs, each at seeds 42 and 43 | 32 |
| **Data first wave** | | **29** | **192** |

The ten confirmation slots refer to selected finalists, not guessed winners. E1W adds seeds 43/44
for each of two finalists; each specialist category adds seed 43 for its two finalists. E3's nominal
unique-pool targets are 6B, 3B, and 1.5B tokens, with the same 6B training exposure in every arm.
Its pool ordering, nested/subset semantics, token-boundary alignment, and exact stops must be bound
before execution; whole-document and batch alignment can add headroom to these nominal numbers.

D4/D6 retain their separate **38 GPU-hours**, bringing the existing P1 total to **230**. Qualify the
hardware and fix the applicable training/calibration settings before dependent source runs. Pack
independent runs across the four GPUs; launch confirmation only after its declared nomination step.
Calendar targets still depend on data availability and measured GH200 throughput.

Identical-looking E1S incumbent recipes are not automatically separate evidence or reusable runs.
Share a control only when its complete materialized model/data/order/optimizer/schedule/seed/evaluation
identities match. This proposal assumes no compute savings from reuse and keeps the existing ceilings.

## Data preparation target

The compiler takes the maximum demand for each source across all proposed recipes. Under eligible
reusable source views, the first-wave **source-capacity envelope is 17.5B tokens** before headroom:

| Source | Required token capacity |
| --- | ---: |
| Ultra-FineWeb v1.4 | 4.4B |
| FineWeb-Edu | 4.4B |
| DCLM | 4.4B |
| Stack-Edu | 1.2B |
| FineMath-4+ | 0.8B |
| Cosmopedia v2 | 0.8B |
| peS2o v3 | 0.4B |
| FineWiki | 0.4B |
| Restricted Stack v3 | 0.3B |
| MegaMath Web-Pro | 0.2B |
| Ultra-FineWeb-L3 Multi-Style | 0.2B |

These are required counts under the eventual D5 tokenizer, not measured supply. They are also not a
claim of 17.5B globally distinct tokens: alternatives can overlap. Do not sum separate banks as unique
capacity or let global first-source ownership silently distort a competing treatment. Each actual
arm needs a globally deduplicated, firewall-excluded view with its treatment membership and background
identity preserved. Materialize extra data where exclusions or view differences reduce eligible yield.

The 29 logical slots consume up to **130B training-token exposures**. Repeated seeds, confirmation, and
E3 epochs can reuse qualified stock; exposure is not the acquisition requirement. Conversely, quotas
must not be reduced merely because a source is difficult to acquire.

## What makes the first wave launch-ready

- Review and freeze the incumbent, substitution design, and blends.
- Complete D5 and bind source-specific filters/languages, corpus views, yields, and exact token stops.
- Bind the fixed proxy shapes and training settings, including D4/D6 dependencies, and qualify GH200.
- Attach selection, tie, failure, and confirmation-analysis manifests within the existing ceilings.
  Do not assume an extra replicated incumbent beyond the two budgeted finalists. If a proposed
  statistical rule needs another control, reconcile it before outputs rather than adding runs later.
- Prepare result tables and figures alongside the runs: source effects by category, every finalist
  and losing arm, repetition versus unique supply, and measured preparation/training costs.

The next concrete milestone is an approved recipe contract plus actual per-source preparation
manifests. The reviewable proposal and compiler help reach it; they do not launch training.

## Machine-readable proposal

[`first_wave_proposal_v1.json`](first_wave_proposal_v1.json) pins the governing data plan, registry,
and source-use record. The compiler validates source eligibility, category conservation, replication
counts, budgets, and integer token apportionment, then emits all logical slots and source demands:

```bash
uv run --no-sync python -m scripts.first_wave_plan \
  research/flagship/first_wave_proposal_v1.json \
  research/flagship/first_wave_preparation_v1.json
```

The output is a preparation proposal, not an experimental result or launch authorization.
