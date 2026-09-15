# Paper contract: Learning to Use Long Context Efficiently

**Selected scope, pre-results, 2026-09-15.** Subtitle: *Data and Memory Trade-offs in a 1.2B Hybrid
Language Model*. The owner-directed [pivot](PIVOT.md) replaces the four allocation-thesis claims with
three new planned claims in [paper/claims.json](../../paper/claims.json). No new result is supported yet.

The owner prioritizes a strong general model and a high-quality paper and permits a change of domain.
The [first-release focus review](../literature/52_first_release_focus_review.md) retains this study as a
conditional candidate under its existing novelty, learnability, matching and cost gates. It is an
agent-authored review, not independent clearance. If those gates fail, record a coherent successor
before confirmatory outputs; the current title and domain are not ends in themselves.

## Central question

How do memory architecture and dependency-requiring supervision interact to produce useful
long-context reasoning at fixed resource budgets? The contribution must be a controlled new finding,
a useful measured frontier, or an informative failure boundary. Use of KDA, a 3:1 hybrid, long documents,
synthetic data or post-training alone is not novelty.

## Three claims

| ID | Question | Evidence required | Failure consequence |
| --- | --- | --- | --- |
| C-USE | Does targeted supervision improve useful long-context capability? | R2 three-seed paired training effect, every primary/general guardrail, frozen scoring and multiplicity handling | Keep standard supervision; report trade-off/null |
| C-INTERACTION | Does that effect depend on architecture? | Complete R2 factorial and uncertainty; R3 only at its measured scope | An interval spanning zero is unresolved; no independence/additivity claim |
| C-FRONTIER | Does the released 1.2B system earn a useful quality-cost advantage? | Useful-length floors, broad retention, public and RAG comparisons, named-hardware measurements, sealed audit and export parity | Remove unsupported efficiency/length headline; release qualified earlier model if needed |

Inherited short proxy and Reader Attention results motivate the design; they are not new claim evidence.
The new claim registry starts with no promoted outcomes. The prior claims and manuscript remain in the
[scope snapshot](../history/2026-09-15-allocation-thesis/manifest.json).

## Main paper spine

1. Problem: the cost of using supplied information across long inputs; task quality and several costs.
2. Controlled design: architectures, shared parents, standard/targeted supervision, partitions and costs.
3. R2 results: all cells/seeds, training/architecture effects and interaction with uncertainty.
4. Transfer and boundaries: one preselected larger-scale or longer-horizon check; evidence interventions.
5. Flagship development: broad base, context/SFT stages, general retention and measured useful length.
6. Practical frontier: public and retrieval alternatives, quality/state/latency/output cost.
7. Limitations and reproducibility: all failed gates, unmeasured extrapolation and all-in resources.

## Required main figures

1. Four-cell capability/cost table with all seed outcomes and family breakdowns.
2. Paired effects and interaction with training and evaluation uncertainty separated.
3. Quality-versus-context curves including 4K retention, oracle controls and failures.
4. Transfer result and flagship stage trajectory with scope boundaries.
5. Task-quality versus resident state and end-to-end latency against public/RAG alternatives.

Preparation funnels, tokenizer fallback, complete source rights/identities, all run logs and extra
systems plots belong in the supplement. Existing generated preparation assets remain labeled as
historical source preparation; they establish no long-context or quality result.

## Claims that this design cannot make

- A full-horizon dense 1.2B counterfactual: only one flagship is funded.
- A population transfer claim from the single-seed R3 check or a general scaling law.
- Linear overall prefill or constant complete-model context state with periodic global GQA.
- Broad frontier capability parity, universally optimal data, or universal superiority to retrieval.
- Usable 128K from allocation size, a needle test, or an earlier checkpoint's label.
- A novel operator, isolated KDA/NoPE causality or a causal effect of targeted dependencies if matching fails.

## Novelty and independent review

Review [the focused literature map](../literature/51_long_context_pivot.md) before R2. Olmo Hybrid
already provides controlled hybrid scaling; ProLong and LongPO address context training; task-directed
long-context training has substantial prior art. Write the closest-work comparison and the exact
remaining empirical question before confirming the study. If the question is already answered, adjust
within the bounded design before outputs; do not manufacture novelty afterward.

Numerical quality floors/margins, primary tests, costs and failure rules freeze before R2. Final model,
inference settings and headline-to-test freeze before the sealed audit. Independent review should
challenge baseline tuning, matching, low power, data leakage, selective task reporting and total cost.
No claim promotes until checked result and finding records support it.

## Model, paper and opportunity package

Ship usable checkpoints, one accelerated inference path, reproducible public/RAG measurements and two
held-out demos: a document packet and an updated conversation history. Prepare a short technical brief
showing measured advantage, limits, independent replication/trial status and a costed next allocation
question. Interest/funding is an intended consequence, not a scientific success metric or guarantee.
Preparation of this package does not send messages or publish artifacts externally.
