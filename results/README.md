# Result directory status

Result JSON is append-only evidence, not a work queue.

- `SpeckLC-*`, `Speck1-*`, and `hardware/` contain completed measurements from the current codebase's
  development history.
- `Speck-Paper1/` and `Speck-Architecture-Promotion-v1/` preserve the retired gate program and the
  reusable evaluation-policy evidence it produced.
- `data/` contains checked, hash-bound flagship source qualification summaries; runtime corpora stay
  on the data volume and remain non-authoritative until every listed gate passes.
- A file's presence does not activate its experiment or authorize its historical claim.

Current decisions must be derived through [`research/flagship/`](../research/flagship/) and cite the
exact result artifacts they reuse. New flagship results should receive a dedicated family and must not
overwrite historical JSON.
