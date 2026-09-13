# 162 — AttnRes blocks must align to logical Transformer boundaries

The official 21-page Attention Residuals report and K3 Block AttnRes implementation are now pinned and
network-validated. The audit preserves Speck's 40 ordered residual modules and Full AttnRes's 40-source
maximum, but rejects readiness v1's eight equal five-module blocks.

The report counts attention and MLP residual sublayers while starting blocks only at Transformer-layer
boundaries. The K3 implementation also evaluates boundaries by decoder-layer index and maintains
separate pre-attention and pre-MLP depth projections. An odd five-module block can split one logical
block between its sequence and FFN stages and is not source faithful.

Readiness v2 partitions 20 logical blocks first, then doubles the boundaries. Four summary blocks use
`[10,10,10,10]` residual modules; eight use `[4,6,4,6,4,6,4,6]`; twelve use
`[2,4,4,2,4,4,2,4,4,2,4,4]`. The eight-block maximum remains nine sources.

The source also requires every pseudo-query to initialize at exact zero. K3's released checkpoint path
reproduces Block AttnRes inference semantics, but its generic linear initializer is random normal and
has no special AttnRes zeroing. The release contains no Full mode, two-phase online-softmax path,
pipeline training cache, or full-model training support. It is not a from-scratch training authority.

V1 is preserved with partition authority removed. V2 changes no active program, selects no residual or
block count, and authorizes no reference implementation, model integration, training, or promotion.

## Artifacts

- [Official-source note](../papers/45_attnres_official_source.md)
- [Primary-source audit](../results/Speck-Paper1/attnres-primary-source-audit-v1.json)
- [Corrected readiness v2](../research/paper-1/attnres_readiness_v2.json)
- [Qualification](../results/Speck-Paper1/attnres-readiness-v2-qualified.json)
