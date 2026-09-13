# 87 — N1 released-artifact availability audit

## Hybrid-role repository

The official `thunlp/rethinking-hybrid-attention` repository is pinned at
`feaad0899bea98e0bf32de9693f37a176713f3ad` (17 June 2026). Its root MIT license and analysis-source
identity are qualified. The tree has only 20 files, 12 of them code/data-like.

Available pieces cover scaling fitting with one SWA-128 LongPPL TSV, NIAH probe preparation/hidden-state
extraction/logistic regression, gradient influence, retrieval-head identification, attention entropy,
and Q/K distance. Model collections are linked separately.

## Missing reproduction chain

The repository does not contain from-scratch training code/configs for the main architecture matrix,
data-mixture/tokenizer/optimizer/schedule manifests, the receptive-field `modeling.py` named by its
README, complete scaling observations, immutable checkpoint revisions, benchmark source manifests,
tests, or target-hardware systems evidence. Core dependencies are entirely unpinned; optional kernels
include lower-bounded packages and a moving FlashAttention feature branch.

Thus the role paper's analysis algorithms are readable, but its checkpoints and generated observations
cannot be reconstructed from the Git tree. Analysis-code rights do not qualify full experiment behavior.

The systematic 72-model study exposes a Hugging Face collection, but immutable repository revisions,
file manifests, configs, training code, and data identity remain unaudited. Distill-then-Replace declares
no official code/model artifact on arXiv.

## Decision

The three N1 sources are mandatory conceptual baselines. Upstream execution, direct reuse, full
reproduction, and placement-protocol authority remain blocked. Across fifteen audited sources, only two
repositories have root rights and none of the three additions creates a newly qualified full
reproduction path. Novelty and architecture decisions do not change.

## Artifacts

- [Hybrid-role code audit](../research/paper-1/rethinking_hybrid_code_audit_v1.json)
- [Novelty code availability v4](../research/paper-1/novelty_code_availability_v4.json)
