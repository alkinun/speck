# Flagship planning targets

Targets describe candidate geometry for accounting and preflight. They are not runnable experiments:
they intentionally contain no training or data configuration.

- [`shape-a`](shape-a/) is the default 1.2B/400B candidate.
- Shape B is the 600M/800B alternative and remains unmaterialized until the scale-geometry task.

The scale ladder selects one shape by day 18. The selected geometry then receives a complete,
hash-bound experiment under `experiments/` during the day-21 configuration freeze.
