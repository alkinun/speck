# STEC: candidate-specific evidence compression for multi-hop answer selection

- **Paper:** [arXiv:2607.10795v1](https://arxiv.org/abs/2607.10795v1)
- **Version reviewed:** v1, 12 July 2026
- **Code:** no dedicated implementation declared on the arXiv record
- **Primary topic:** organize multi-trajectory search evidence by candidate answer before constrained
  final verification

## Mechanism

STEC begins after a deep-search system has produced multiple retrieval/reasoning trajectories and raw
candidate answers. It conservatively normalizes answer strings, groups trajectories by candidate, and
compresses each group into the same structured fields: raw answer variants, supporting evidence,
contradicting/weakening evidence, a concise reasoning path, and trajectory-count statistics.

A verifier compares these candidate-specific representations and must select an answer already present
in the candidate set. It considers whether evidence covers the entities, relations, and constraints in
the question and whether the reasoning path coherently connects that evidence to the candidate. This is
evidence organization for answer selection, not context/KV compression before reasoning.

## Evidence boundary

Main experiments use eight Search-R1 trajectories, Qwen2.5-7B, and EM on HotpotQA dev, 2WikiMultiHopQA
dev, MuSiQue dev, and the 125-example Bamboogle test set. STEC averages 0.347 EM versus 0.301 for the
underlying Search-R1 and 0.339 for a verifier-only ablation. The compression ablation gains 0.008 macro
EM, regresses slightly on MuSiQue, and has no multiplicity or paired uncertainty report.

Trajectory counts 2/4/8 and base/instruct 3B/7B configurations are varied. The reported benefit is
larger for 3B configurations and small for 7B, with some per-dataset regressions. The paper reports no
token/byte compression ratio, latency, FLOPs, monetary cost, or end-to-end search/compression/verifier
budget. No dedicated code/data artifact or immutable model/retriever/corpus identity is declared.

## What matters for Speck

Structured supporting/conflicting evidence, explicit reasoning paths, coverage of question constraints,
and candidate-level verification are prior art for external multi-hop systems. A residual N2 empirical
predictor must compare against STEC-style structured-evidence verification when evaluated on agentic or
multi-trajectory tasks.

STEC does not expose token identities inside an internal cache, define survival of independently
necessary KV sources, predict failure before output, or perform equal-state removal/restoration. It is
therefore an adjacent external baseline, not direct evidence that such an internal statistic works.

## Bottom line

STEC does not restore N2 concept novelty and does not itself eliminate the narrow empirical-law question.
It makes the required external baseline stronger: internal diagnostics must add value beyond a verifier
that already compares normalized candidate evidence, contradictions, and reasoning paths.
