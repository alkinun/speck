# Flagship planning targets

Targets describe candidate geometry for accounting and preflight. They are not runnable experiments:
they intentionally contain no training or data configuration.

- [`scale-targets-v2.json`](scale-targets-v2.json) binds the 60M, 150M, 220M, 350M, 750M, retained
  600M, and fixed 1.2B flagship geometries to the tied embedding/head decision. [`ACCOUNTING.md`](ACCOUNTING.md)
  summarizes the exact machine-validated totals; [`accounting-v2.json`](accounting-v2.json) is the
  generated record. The v1 spec/accounting remain hash-bound predecessor evidence.
- [`shape-a`](shape-a/) preserves the historical 1.2B/400B planning candidate. Its hashes are pinned
  by the v1 scale contract rather than rewriting it for the three reserved tokenizer rows.

The active architecture-plan successor fixes the 1.2B/400B target with a 1.2B/320B throughput
fallback. The 600M geometry remains for provenance and future model-by-token research; the grant-1
scale ladder does not select between them. The fixed geometry receives a complete, hash-bound
experiment under `experiments/` during the day-21 configuration freeze. None of these targets grants
data or training authority.
