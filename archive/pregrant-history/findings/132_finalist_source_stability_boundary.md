# 132 — Finalist finiteness and complete-source boundary

## Training stability is fail-fast

Every accumulated training loss and clipped gradient norm is checked for finiteness before the
optimizer update. A failure raises and prevents a complete final-summary event. The collected
`non_finite_steps: 0` field is therefore a logical completion invariant—zero non-finite steps were
permitted—not a directly incremented event counter. Aggregate validation loss and cumulative timing
are independently checked finite during collection.

## Exact source coverage

The validation loader deterministically round-robins all 11 manifest sources. Each intermediate
4,997,120-token evaluation contains 305 batches and 27–28 batches per source; the final
19,988,480-token evaluation contains 1,220 batches and 110–111 per source. Every expected source thus
has a positive denominator by construction.

The frozen analyzer nevertheless checks only that all run results contain the same nonempty source
set. A synthetic counterexample confirms that omitting the same source everywhere can still pass the
remaining guards. Source values are not explicitly checked finite before JSON serialization, although
non-finite comparisons normally fail the guard.

## Append-only sidecar

A new verifier requires the exact 11-source set and finite values at every one of the five validation
points, plus the exact token budgets and final-history equality. It is not wired into the hash-frozen
runner. It must be run on each collected result before interpretation; any failure rejects the result
and final language screen without imputation, deletion, skipping, or replacement.

## Artifact

- [Source-stability audit](../results/Speck-Paper1/finalist-source-stability-audit-v1.json)

Run the sidecar with:

```bash
python -m scripts.paper_finalist_source_coverage_validate \
  results/Speck-Paper1/finalist-runs/<run>.json
```
