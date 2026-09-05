# 91 — N1 claim-table correction after direct placement overlap

## Claim disposition

The v3 table adds two explicit do-not-claim rows: from-scratch non-uniform Transformer/Mamba placement,
and the avoid-early-attention/middle-later recipe with its role explanation. The systematic placement
study directly occupies both.

The residual row is renamed to prospective prediction of arbitrary unseen layouts. It is technically
different because the source does not freeze a predictor before held-out layout training. It is not yet
a scientific contribution: feasibility, calibration, cross-mixer transfer, causal validity, and an
improved selected architecture are all absent.

## Priority

N1 remains first only among two unestablished questions, not among architecture candidates. Its next
gate is independent claim review, which may retire it. No protocol, placement run, architecture freeze,
or new training is authorized. N2 remains deferred. The frozen baseline sequence is unaffected.

## Artifact

- [Novelty claim-overlap v3](../research/paper-1/novelty_claim_overlap_v3.json)
