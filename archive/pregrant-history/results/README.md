# Result directory status

Result JSON is append-only evidence, not a work queue.

- `SpeckLC-*`, `Speck1-*`, and `hardware/` contain completed measurements from the current codebase's
  development history.
- `Speck-Paper1/` and `Speck-Architecture-Promotion-v1/` preserve the retired gate program and the
  reusable evaluation-policy evidence it produced.
- `data/` contains checked, hash-bound flagship source qualification summaries; runtime corpora stay
  on the data volume and remain non-authoritative until every listed gate passes.
- [`Speck2-Instruct-Data-Pilot`](Speck2-Instruct-Data-Pilot/summary.json) consolidates the completed
  pre-compute 100K/500K SFT comparison against the release: improved diagnostics, mixed benchmark
  results, and no release promotion.
- A file's presence does not activate its experiment or authorize its historical claim.

Current decisions must be derived through [`research/flagship/`](../research/flagship/) and cite the
exact result artifacts they reuse. New flagship results should receive a dedicated family and must not
overwrite historical JSON.

Large corpora, checkpoints, complete logs, predictions, and traces stay in the runtime artifact store.
Checked manifests identify those bytes; Git retains compact results and bounded raw evidence needed to
audit a conclusion. Retention and backup classes are defined in
[`research/DATA_MANAGEMENT.md`](../research/DATA_MANAGEMENT.md). Paper-level use is tracked separately
in [`paper/claims.json`](../paper/claims.json).
