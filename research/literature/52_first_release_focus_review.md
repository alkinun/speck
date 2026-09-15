# First-release focus review

Reviewed 2026-09-15 using primary papers and author repositories. This is an agent-authored positioning
review, not independent review, exhaustive novelty clearance, baseline reproduction or source-use
approval. The owner explicitly permits changing the research domain if that better serves a strong
general model and a high-quality paper. The selected execution contracts remain in force until a
recorded successor replaces them; permission to reconsider is not an experimental result.

## Closest work and attribution limits

| Primary source | Reviewed evidence and overlap | Consequence for Speck |
| --- | --- | --- |
| [Olmo Hybrid, v4](https://arxiv.org/html/2604.03444v4), §§2–4 and synthetic protocol | Controlled large-scale hybrid/transformer comparisons, scaling, and synthetic state-tracking/recall training already exist. Its theoretical and synthetic GDN construction includes negative eigenvalues. | Hybrid efficiency, scale transfer and state-tracking complementarity are prior territory. Theorems about that GDN construction cannot be imported as guarantees for our inherited KDA implementation. |
| [Token-level comparison, v1](https://arxiv.org/html/2606.20936v1), §§2–6 | Matched-prefix token diagnostics compare Olmo architectures; entity/pronoun and copying/bracket regimes differ. It explicitly proposes richer discourse-state tracking. | Neither discourse updates as a task idea nor token-level architecture diagnostics are new by themselves. A training intervention must have its own controlled evidence. |
| [LoongRL, v2](https://arxiv.org/html/2510.19363v2), §§3–4 | KeyChain synthesis and staged GRPO train long-context reasoning on Qwen2.5 7B/14B instruction models; the smaller model receives a difficulty warmup. | Dependency-requiring synthetic training and transfer beyond training length are established ideas. Gains on mature 7B/14B instruction models do not establish task learnability for our 350M/10B parents. |
| [Beyond Reward Engineering, v1](https://arxiv.org/html/2606.18831v1), §§3–4 | A roughly 14K-example recipe spans retrieval, multi-evidence synthesis and reasoning under GRPO, evaluated on Qwen3 4B/8B/30B-A3B. | Generic evidence-synthesis training gains are insufficient novelty. Our proposed matched supervision intervention differs from an RL data-package comparison, but that distinction needs verification. |
| [Long-dependency ProLong](https://arxiv.org/abs/2405.17915) and [author repository](https://github.com/October2001/ProLong) | The ACL 2024 work scores documents by dependency strength, distance and specificity for training-data selection. | Selecting stronger long dependencies is prior art. This is a different project from Princeton's similarly named ProLong; cite them separately. |
| [Princeton ProLong](https://github.com/princeton-nlp/ProLong) | Long-context continuation and SFT ablations build on Llama-3-8B; its release describes staged 64K/512K continuation. | Long documents and context extension are not a new intervention. Its data/packing and mature starting model are not equivalent to our current source stocks or young parents. |
| [HALO/HypeNet, v1](https://arxiv.org/html/2601.22156v1), §§4–6 | Transformer-to-hybrid distillation and long-context architecture changes include small-model experiments and inference comparisons. | Adapting an existing model could be assessed as a different product strategy, but conversion or long-context hybrid efficiency alone would not provide a fresh paper contribution. |
| [SmolLM2](https://arxiv.org/abs/2502.02737) | The report describes a 1.7B model trained on approximately 11T tokens, with data-centric development and released datasets. | A strong broad model plus transparent data work is a useful release pattern. Our 320B target is a materially smaller training horizon; equal-footprint leaderboard leadership cannot be assumed from size or architecture. Token count alone does not predict our final quality. |

No reviewed comparison supplies Speck's new experimental outcomes. The review did not identify the
exact proposed paired 2×2 intervention in these sources; that limited observation is not proof that
it is absent from the literature. Closely related papers and their citations still require independent
challenge before confirmation.

## Options assessed

| Direction | Value if successful | Main unresolved risk | Current judgment |
| --- | --- | --- | --- |
| General model plus bounded document/history study | Useful supplied-information capability and a controlled interaction/trade-off finding, while retaining broad ability | Small-parent task floor; matching; three-seed uncertainty; natural-task transfer; real hardware advantage | Retain as a conditional candidate through pre-results review and pilots |
| General model plus broad data-recipe paper | Reusable training evidence and an open model | Considerable existing data-recipe literature; source supply is unfinished; a large selection funnel consumes budget without ensuring novelty | Do not revive the retired wave automatically; requires a narrower unanswered question first |
| Code/math specialization | Clear task utility and verifiable outputs | Qualified code supply is scarce locally; stronger-task training may compete with general ability and the separate post-training effort | No evidence yet that this is the better first direction; do not substitute domain data silently |
| Adapt a strong public base | Access to already learned capabilities and less from-scratch pretraining work | Different scientific/product attribution, inherited training cost and rights, crowded adaptation literature, compatibility with the submitted program | A legitimate contingency to assess explicitly, not an interchangeable version of our own pretraining plan |

These judgments are inference from the reviewed work and our preparation state, not measured rankings
of research directions. No new architecture axes, model-size search, teacher generation or training
runs are added here. Final response behavior and deeper post-training remain deferred.

## What makes the current candidate worth testing

The useful question is whether changing dependency supervision produces a reproducible benefit under
each memory architecture, with matched source/task exposure, preserved general ability and measured
implementation cost. Positive interaction is not required for a useful model or an informative paper.
An unresolved interval is not evidence of independence. A surprising regression, reversal or practical
trade-off can be valuable if the controls and scope are sound; it cannot be marketed as a win.

Before R2, the existing review/R1 gates must resolve:

1. A closest-work comparison identifying the remaining empirical question and explaining why the
   matched intervention adds evidence beyond existing training, distillation and diagnostic work.
2. Disjoint task pilots showing that both small parents can solve accessible short/oracle-evidence
   versions, and that full-context versions are informative rather than all-floor/all-ceiling tests.
3. Verified support and counterfactual checks; source-family separation; matched processed and
   supervised exposure; and baseline optimization with the same bounded tuning opportunity.
4. Numerical useful-quality floors and general/per-family margins, plus an honest detectable-effect
   assessment. Three paired training seeds are not thousands of independent document replicates.
5. Actual supply/repetition, full parent/branch/teacher/evaluation costs and hardware measurements that
   fit the selected budget while preserving base training, capability work, validation and reserve.

If these gates fail, stop the affected confirmatory work and write a costed scope successor before
outputs. Do not buy a positive result by loosening matching, tuning the final test or adding unbudgeted
seeds. Keep general-model preparation useful while deciding. The future paper should report what was
learned and what a reader can reproduce, rather than promise popularity or frontier parity.
