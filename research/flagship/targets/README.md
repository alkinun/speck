# Flagship planning targets

Targets describe candidate geometry for accounting and preflight. They are not runnable experiments:
they intentionally contain no training or data configuration.

- [`scale-targets-v1.json`](scale-targets-v1.json) compactly defines the 60M, 150M, 220M, 350M,
  750M, 600M flagship, and 1.2B flagship geometries. [`ACCOUNTING.md`](ACCOUNTING.md) summarizes the
  exact machine-validated totals; [`accounting-v1.json`](accounting-v1.json) is the generated record.
- [`shape-a`](shape-a/) preserves the historical 1.2B/400B planning candidate. Its hashes are pinned
  by the v1 scale contract rather than rewriting it for the three reserved tokenizer rows.

The scale ladder selects one shape by day 18. The selected geometry then receives a complete,
hash-bound experiment under `experiments/` during the day-21 configuration freeze. None of these
targets grants data or training authority.
