# Controlled study: learning to use long context

Selected machine contract: [long_context_study_v1.json](long_context_study_v1.json).
No confirmatory model or task result exists for this successor.

## Question and intervention

How do memory architecture and dependency-requiring supervision interact in long-context reasoning
at fixed resource budgets? The primary endpoint is equal-family task quality at 32K on document
reasoning and history/update understanding, with source/task and short/general guardrails.

| Architecture | Standard supervision | Dependency-requiring supervision |
| --- | --- | --- |
| Dense global GQA | D-S | D-T |
| 3:1 KDA/global GQA | H-S | H-T |

The two supervision branches share natural-document continuation and general replay. By default,
30% of assistant-target tokens is the experimental slice and 70% broad general replay. Standard
examples and targeted examples come from the same source-family frame with matched length, answer
style, processed tokens and assistant-target volume. Targeted examples require separated evidence,
updates, relations or recognizing missing evidence. Matching tolerances and exact examples freeze in R1.
If these cannot be matched, the study estimates a training-package effect, not dependency structure alone.

## R2: six parents, twelve branches, three independent seed blocks

Use paired seeds 42/43/44. Each architecture/seed gets its own 350M-class parent, targeting 10B broad
base tokens and a shared 250M-token 32K natural-document continuation. From that exact parent and
optimizer state, train standard and targeted branches, initially targeting 100M processed tokens each.
These endpoints are cost hypotheses, not executable commitments. R0/R1 re-cost all parent training,
continuation, branches and overhead within 360 GPU-hours before any confirmation run.

Hold architecture depth/width, FFN, tokenizer, shared data order and other named settings fixed.
Dense replaces recurrent positions with global GQA. Report its differing parameters/FLOPs/state.
Bounded optimizer checks must keep the dense baseline credible; distinguish a shared-recipe comparison
from architecture-specific tuning. This comparison does not isolate KDA gating, NoPE or one operator.

Do not treat the twelve branches as twelve independent parent training replicates. Compute each
contrast inside a seed block, then quantify training uncertainty across the three seed blocks.

## R3: transfer, 180 hours

Preferred: one fresh seed (45), dense/hybrid 750M parents targeting 15B broad tokens, shared long
continuation and all four supervision cells. If that entire program does not fit the R0/R1 forecast,
use 350M parents targeting 23B instead. Choose the route, endpoints and costs before R2 confirmation
outputs. A 350M fallback checks horizon, not model-size transfer. Charge full fresh parent training.

One seed gives directional transfer evidence. Neither route supports population equivalence, a fitted
scaling law, or a dense counterfactual at the flagship's 1.2B/320B horizon. No route fits: require a
costed successor before launch; do not erase the missing experiment from the limitations.

## Analysis

For higher-is-better task quality, compute within each seed:

- Training effect: `((D-T - D-S) + (H-T - H-S)) / 2`.
- Architecture effect: `((H-S - D-S) + (H-T - D-T)) / 2`.
- Interaction: `(H-T - H-S) - (D-T - D-S)`.

Report two-sided 95% Student-t intervals across the three paired contrasts, raw and Holm-adjusted
p-values for the three confirmatory contrasts, and clearly separate family/document bootstrap
uncertainty. Unadjusted intervals are labeled as such. A null interaction is unresolved, not evidence
of independence. R1 estimates detectable effects and discloses limited power before confirmation.

Fixed-token endpoints are primary; checkpoint interpolation gives fixed-FLOP/time views only over
observed support using frozen rules. Retain all tasks, seeds, failures, censored endpoints and costs.
Evidence-removal/update probes and KDA-state interventions are diagnostics with distribution-shift
limitations; they cannot independently promote an architecture.

## Selection and release

Promote targeted supervision when its paired average training effect is positive with a Holm-adjusted
two-sided p-value below 0.05, every frozen general/per-family non-inferiority guardrail passes, and cost
fits. The within-hybrid training contrast must also be positive and pass its frozen non-inferiority
guardrails; a pooled gain cannot override a hybrid regression. Otherwise retain standard supervision
and report the outcome. Interaction significance is not required
for a useful training improvement. No post-hoc rescue subset or quality-selected seed replacement.

The flagship retains the selected hybrid architecture. Its before/after development and final results
measure stage effects; lack of a full-budget dense 1.2B control remains explicit. If even standard
supervision does not support useful 32K performance, release the qualified general model with a
boundary/negative-result report and remove the long-context-performance headline.

## Required R1 freeze

Exact arm/config/parent identities; family-level partitions; numerical quality floors and retention
margins; scorer and primary family weights; processed/assistant tokens and tolerances; LR/schedules;
output budgets; failed-run rules; all costs; R3 route; and independent review. They must be bound before
R2 outputs. Missing numerical thresholds cannot be filled after looking at confirmation results.
