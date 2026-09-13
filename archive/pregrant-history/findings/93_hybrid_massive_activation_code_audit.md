# 93 — Hybrid massive-activation artifact audit

## Qualified identities

The official analysis tree is pinned at `190e1b795262b91159a01f846ebbca799c45be8f`: 57 files, 23
package Python files, five scripts, five tests, and an MIT root license. The legacy aggregate checkpoint
tree is separately pinned at `5a98aec28a52efe0303545d929d0964a5973b127`: 62 files across ten
named experiment directories, with Apache-2.0 model-card rights. Both were inspected through metadata-
only clones; no working tree, model weights, imports, or execution occurred.

The repository is materially stronger than most prior N1 artifacts. It supplies analysis code, metric
tests, exact A800/CUDA package pins, model registries, and a release manifest. Baseline and no-GDN-
output-gate checkpoints may have a bounded future analysis-reproduction path.

## Boundaries

It explicitly omits from-scratch training code. The gated-full-attention integration is undistributed
and its checkpoints are weights-only. Exact sampled JSONL inputs are absent, M-A-P and modern-model
revisions are moving, and checkpoint bytes/table parity have not been validated. Native installers and
third-party code were not executed.

## Decision

Code/root-rights and aggregate checkpoint identities are qualified, but full reproduction, upstream
execution, and placement-protocol authority remain blocked. Because the N1 residual already has a low
prior, independent review must justify even a bounded checkpoint-analysis reproduction first.

## Artifact

- [Code audit v1](../research/paper-1/massive_hla_code_audit_v1.json)
