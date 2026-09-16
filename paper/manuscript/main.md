# Learning to Use Long Context Efficiently

## Data and Memory Trade-offs in a 1.2B Hybrid Language Model

**Working manuscript, selected scope, pre-results.** No new claim is supported. The owner-directed
2026-09-15 pivot replaces the earlier allocation-thesis manuscript, preserved with hashes in the
[transition snapshot](../../research/history/2026-09-15-allocation-thesis/manifest.json).

## Abstract

We plan a controlled study of how memory architecture and dependency-requiring supervision interact
in useful long-context reasoning under limited compute. A dense/global-GQA and a KDA/global-GQA hybrid
each receive standard or targeted supervision from matched source families and shared continuation
parents. Three paired seed blocks quantify training, architecture and interaction effects, followed by
one preselected scale or horizon transfer check. The resulting recipe informs a general-purpose
1.2B flagship within 5,000 GH200 GPU-hours. Final conclusions will depend on document/history capability,
retained general performance, public and retrieval comparisons, and measured quality-cost frontiers.

## 1. Problem and contribution

Affordable context requires both low processing/state cost and an ability to use supplied information.
We focus on document evidence reasoning and ordered histories with updates, while retaining ordinary
assistance, instruction following, math and code. KDA, GQA, hybrid ratios, continuation and synthetic
supervision are inherited. The exact new empirical question is fixed after comparison with the closest
prior work and before confirmation outputs. See [related work](../../research/literature/51_long_context_pivot.md) and the
[closest-work/focus review](../../research/literature/52_first_release_focus_review.md). The general-model
release is the primary product objective; the paper focus remains conditional on pre-results novelty,
small-parent task learnability, matching and cost qualification. A domain change would require a
coherent contract successor before confirmation.

The three hypotheses are registered in [claims.json](../claims.json): training improves useful context;
its effect interacts with architecture; and the flagship realizes a useful quality-cost operating point.
None is an achieved conclusion. A null interaction cannot establish independence.

## 2. Methods

### Models and control

The planned flagship has 24 blocks at hidden width 2048, a 3:1 KDA/global-GQA pattern, NoPE globals,
sigmoid recurrent output gates, SwiGLU, tied embeddings and the frozen Mistral tokenizer. BF16 and
Muon/AdamW remain defaults. The matched dense control replaces recurrent positions with GQA and
retains named depth/width/FFN/tokenizer/training controls; residual parameter and FLOP differences are
reported. The comparison estimates architecture packages, not an individual operator's causality.

### Data and intervention

One broad English-first base mixture precedes coherent natural-document continuation. Each parent
branches into standard and dependency-requiring supervision, with the same general replay. The target
slice asks for combining separated evidence, applying rules, resolving updates/conflicts and recognizing
missing evidence. Match source frames, length bins, answer style, processed input-plus-target tokens and
supervised target volume within predeclared tolerances; failed matching restricts attribution.
Partition by document/history family, including editions/forks and task variants. Existing D5/E2 audits
remain unopened exclusions. New development/final views have their own identities.

### Controlled study

R2 targets 350M-class models at seeds 42/43/44: six architecture parents and twelve supervision branches.
Common continuation precedes branching. Endpoints and costs freeze after disjoint R0/R1 pilots, within
360 GPU-hours. R3 uses fresh seed 45, preferably 750M/15B parents plus four branches, or a 350M/23B
horizon fallback chosen before R2 outputs, within 180 hours. Target endpoints are not runtime forecasts.

### Analysis

Compute training and architecture main effects and their interaction within each complete seed block.
Report two-sided 95% Student-t intervals across the three seed contrasts and Holm-adjusted tests over
three confirmatory contrasts. Label unadjusted intervals and report task/family bootstrap uncertainty
separately from training uncertainty. Freeze numerical quality floors, margins, family weights, output
budgets, interpolation and failed-run rules before confirmation. Retain every arm and failure.

## 3. Controlled results — pending

Required: all four cells and seeds, primary and per-family scores, general retention, fixed-token/FLOP/
time views, uncertainty, matching diagnostics and costs. No preparation table fills this section.

## 4. Transfer and mechanism — pending

Report the preselected R3 route, all outcomes and evidence-removal/update diagnostics. One seed is a
directional check; no scaling law or full-horizon flagship equivalence follows. Diagnostic interventions
carry distribution-shift limitations.

## 5. Flagship development — pending

A [source-specific preparation report](../tables/base-supply-v1/table.md) reconciles the current stocks
with base demand. It does not establish a global unique corpus or a validated repetition schedule.

320B base tokens are the default; 400B requires measured fit within 2,425 hours without taking capability
or evaluation budget. Preserve pre-decay/base, context and instruction checkpoints. The capability
program receives 700 hours. Useful 32K is the first milestone, 64K an evaluation point and 128K a target.
Each stage must retain general quality and re-establish useful context; advertise the highest passing
length. There is no full-horizon dense 1.2B counterfactual in this allocation.

The final assistant is required to emit reasoning inside `<think>...</think>` followed by its answer
([owner decision](../../research/flagship/release_behavior_decision_v1.json)). The post-training recipe
and exact template/parser remain pending. Evaluation must retain full generations, score the parsed
answer under frozen rules, and account for both reasoning and answer tokens in total cost.

## 6. Practical quality-cost frontier — pending

Compare pinned public near-footprint/larger models and BM25-based retrieval with the same source inputs
and output budgets. Distinguish architecture controls from deployed-system comparisons. Measure named
hardware, matched and best-qualified backends, resident state, weights/state/workspace, prefill, decode,
complete task latency and output cost. Include indexing/retrieval and teacher/judge costs. Missing energy
telemetry remains unavailable. CPU/GGUF claims require a qualified recurrent path.

## 7. Limitations and reproducibility

A three-seed interaction may be underpowered; report detectable effects and uncertainty. The 1.2B
flagship cannot establish a same-budget dense causal advantage. Periodic global attention still has
length-growing cache and quadratic prefill work. A narrow or modified benchmark subset is not its
full official score. Successful retrieval does not establish composition or useful context.

Final inference settings and claim tests freeze before the one-opening sealed audit. Failures narrow
claims. Every published value must resolve to checked results and independent regeneration. Funding,
interest and adoption are intended downstream outcomes, not evidence of a technical claim.

## Supplement: retained preparation evidence

Earlier tokenizer fallback, source-capacity, deduplication and recovery measurements remain useful
preparation evidence. The frozen Mistral decision did not open D5; incomplete historical all-attempt
costs remain disclosed. Existing generated assets and their immutable inputs are preserved under
[paper analysis](../README.md). Old E1/E3 quotas and pipeline projections are labeled historical and
must not be presented as completed supply for the new study or as model-quality evidence.
