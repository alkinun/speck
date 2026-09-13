# 81 — N2 concept rejection after evidence-compression audit

## New evidence

Three cross-paradigm full-text audits invalidate the remaining conceptual novelty of “all required
sources survive compression.”

BRIEF decomposes multi-hop questions, identifies a helpful proposition for every subquestion, rejects
an example if any hop lacks one, requires the propositions to come from distinct documents, and uses
their concatenation as the compressor target. BRIEF-Pro extends source-aware compression to 10K-plus
word oracle-plus-distractor contexts, requested/automatic budgets, full-pipeline compute, and explicitly
analyzes failure to capture one hop.

IterCOMP is even closer. Its `Full` condition contains sub-answers for every required hop; `Partial`
omits later hops and is defined unanswerable. Its compression loop asks an LLM whether accumulated
evidence is sufficient, identifies the next missing information, asks a targeted follow-up question,
and accumulates evidence until sufficient or the iteration cap. Full/partial sufficiency, missing-source
identification, and targeted recovery are therefore all prior art.

## Scope correction

The v2 `N2_conjunctive_required_source_survival_law` is rejected as conceptual novelty. Conjunctive
source availability remains a useful measurement feature, but neither the conjunction nor the workflow
around finding/restoring a missing hop may be presented as new.

The only residual hypothesis is an empirical diagnostic law: a cheap statistic computed from internal
KV/sparse state before output might predict failures beyond BRIEF-style proposition coverage,
BRIEF-Pro helpfulness, IterCOMP sufficiency, HeadKV-R2 importance, GER/reachability/consensus, retained
mass, and output error. It would also need to predict equal-physical-budget removal/restoration effects
across real documents, three scales, and a held-out compression mechanism.

That distinction is not established. It is narrower than an architecture mechanism and may ultimately
be only an evaluation contribution. External LLM sufficiency judgments are themselves imperfect—full-
evidence classification falls sharply with hop count—and IterCOMP's published API cost table counts
only the final reader call, not iterative compression. Those limitations justify stronger experiments;
they do not restore conceptual novelty.

## Decision

N2 concept novelty is rejected. The residual internal-state law remains an unestablished hypothesis
pending STEC/current-literature review, artifact audits, prospectively frozen prediction/intervention,
and independent expert review. No architecture freeze, model integration, training, or paper-scale
authority follows.

## Artifacts

- [BRIEF full-text note](../papers/32_brief.md)
- [BRIEF-Pro full-text note](../papers/33_brief_pro.md)
- [IterCOMP full-text note](../papers/34_itercomp.md)
- [Novelty landscape v3](../research/paper-1/novelty_landscape_v3.json)
